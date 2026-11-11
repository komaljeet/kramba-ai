"""Does a local loss per BCD block keep quality while skipping the layers after it?

Under block-coordinate descent only one block trains per step, but the loss sits
at the end, so every step still runs the frozen layers after the active block
forward and backpropagates through them. bcd_balanced_local instead puts the
model's own final_norm + decoder (shared and frozen while a middle block trains)
right after the active block, scores the loss there, and never runs the layers
after it. The embedding block and the last block keep the normal end-to-end loss.

The risk is quality: a feature that is only useful to a later layer gets no
signal. Deep supervision already collapsed recall to 0% in this project, so this
measures in-context copying as well as held-out loss.

Arms (identical seed, batches, LR schedule; all use the same model):
  full                no BCD (quality reference, more memory)
  bcd_balanced        today's production BCD
  bcd_balanced_local  bcd_balanced + local loss for middle blocks

Guards against a vacuous result:
  * known-answer: the manual forward used by the local arm must match
    model(x) when run to the last layer, or nothing it measures means anything.
  * bcd_balanced must trail full by a real margin, or there is no gap to judge.
  * every parameter must move in every BCD arm.
  * the copy test only counts if the full arm learned to copy; otherwise its
    verdict is VOID rather than "no difference".
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ab_prodigy import (                                   # noqa: E402
    as_pair, enable_scan_flags, evaluate, require_cuda,
)
from hvn.core.block_coordinate import BlockCoordinateScheduler     # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model                # noqa: E402
from hvn.optimizers import build_optimizer                         # noqa: E402

# (partition, lookahead): lookahead None = normal end loss; k = run k frozen layers
# past the active block before the shared head (0 = plain local loss, #188).
ARMS = {"full": (None, None), "bcd_balanced": ("balanced", None),
        "bcd_balanced_local": ("balanced", 0),
        "bcd_balanced_look2": ("balanced", 2), "bcd_balanced_look4": ("balanced", 4),
        "bcd_balanced_emb50": ("balanced", None), "bcd_balanced_emb25": ("balanced", None),
        "bcd_balanced_emb150": ("balanced", None), "bcd_balanced_emb200": ("balanced", None),
        "bcd_k1": ("balanced", None), "bcd_k2": ("balanced", None), "bcd_k3": ("balanced", None),
        "bcd_k6": ("balanced", None), "bcd_k8": ("balanced", None)}
# Block-size cap in layers (#192): partition_budget = k x the largest layer. The default
# arm's cap is the embedding table (4 layers at d1024; 2 layers at 1B).
BUDGET_LAYERS = {"bcd_k1": 1, "bcd_k2": 2, "bcd_k3": 3, "bcd_k6": 6, "bcd_k8": 8}
# Embedding-block turn length per arm (#190); every other block keeps --steps-per-block.
EMB_STEPS = {"bcd_balanced_emb50": 50, "bcd_balanced_emb25": 25,
             "bcd_balanced_emb150": 150, "bcd_balanced_emb200": 200}
DEFAULT_ARMS = ["full", "bcd_balanced", "bcd_balanced_local"]
LOCAL_ARMS = [a for a, (_, k) in ARMS.items() if k is not None]
VARIANT_ARMS = LOCAL_ARMS + list(EMB_STEPS) + list(BUDGET_LAYERS)


class UnequalTurnsBCD(BlockCoordinateScheduler):
    """BCD with a different turn length for the token_embedding block."""

    def __init__(self, *a, emb_steps: int, **k):
        super().__init__(*a, **k)
        self.base_steps = self.steps_per_block
        self.emb_steps = emb_steps

    def step(self) -> bool:
        self.steps_per_block = (self.emb_steps if self.active_block_name == "token_embedding"
                                else self.base_steps)
        return super().step()
COPY_LEN = 128      # random tokens, then the same tokens again
COPY_SEQS = 32


def build(args, device, grad_checkpoint=False):
    from hvn.edge.model import EdgeHVN
    plan = ["deltanet"] * args.num_layers
    for i in args.attn_layers:
        plan[i] = "transformer"
    model = EdgeHVN(
        vocab_size=args.vocab_size, d_model=args.d_model, num_layers=len(plan),
        n_heads=args.n_heads, block_plan=plan, attn_window=args.seq_len,
        tie_embeddings=True, dropout=0.0, grad_checkpoint=grad_checkpoint,
    ).to(device)
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(
        model, target_names=[n for n in linears if not any(p in n for p in exclude)])
    enable_scan_flags(model)
    return model


def forward_to(model, input_ids, last: int):
    """Embedding, blocks[0..last], then the shared final_norm + decoder."""
    S = input_ids.shape[1]
    pos = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(input_ids.shape[0], -1)
    x = model.token_embedding(input_ids)
    for block in model.blocks[:last + 1]:
        x = block(x, position_ids=pos)
    return model.decoder(model.final_norm(x))


def local_target(sched, lookahead: int, num_layers: int) -> Optional[int]:
    """Last layer to run before the head, or None if it needs the full loss."""
    members = sched.active_block_members
    if any(not m.startswith("blocks.") for m in members):
        return None                     # token_embedding or final_norm+decoder
    last = max(int(m.split(".")[1]) for m in members) + lookahead
    return None if last >= num_layers - 1 else last


@torch.no_grad()
def copy_accuracy(model, seqs, device) -> Dict[str, float]:
    """Top-1 accuracy predicting the repeated half of [random | same random]."""
    model.eval()
    hit_first = hit_repeat = n = 0
    for s in seqs:
        x = s.unsqueeze(0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model(x).argmax(-1)[0]
        nxt = s.to(device)[1:]
        p = pred[:-1]
        half = COPY_LEN - 1
        hit_first += int((p[:half] == nxt[:half]).sum())
        # Position COPY_LEN-1 predicts the first repeated token, which is not
        # predictable yet; start one later.
        hit_repeat += int((p[COPY_LEN:] == nxt[COPY_LEN:]).sum())
        n += 1
    rep_n = n * (2 * COPY_LEN - 1 - COPY_LEN)
    return {"first_half": hit_first / (n * half), "repeat": hit_repeat / rep_n}


def run_arm(arm, args, batches, probes, copy_seqs, device) -> dict:
    partition, lookahead = ARMS[arm]
    local = lookahead is not None
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # The full arm keeps every layer's grads and optimizer state, which spilled past
    # this 6 GiB card in the smoke run. Recompute is exact, and this arm is the
    # quality reference only; train-time comparisons are between the BCD arms.
    model = build(args, device, grad_checkpoint=partition is None)
    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt, desc = build_optimizer(model, "muon", lr=args.aux_lr,
                                weight_decay=args.weight_decay, muon_lr=args.muon_lr)
    sched = None
    if partition is not None and arm in EMB_STEPS:
        sched = UnequalTurnsBCD(model, opt, steps_per_block=args.steps_per_block,
                                partition=partition, emb_steps=EMB_STEPS[arm])
    elif partition is not None:
        budget = None
        if arm in BUDGET_LAYERS:
            layer_sizes = [sum(p.numel() for p in b.parameters()) for b in model.blocks]
            budget = BUDGET_LAYERS[arm] * max(layer_sizes)
            # Keep the embedding table in its own block in every budget arm, so only
            # layers-per-block changes (a big budget would otherwise pack it with layers).
            import hvn.core.block_coordinate as _bc
            _orig_bg = _bc._balance_groups
            def _keep_emb(blocks, b=None):
                groups = _orig_bg(blocks[1:], b)
                # A tail-only final_norm+decoder block trains just a LayerNorm (the decoder
                # is the tied embedding), so its turn is wasted; merge it into the last
                # layer group as the production partition does (#193).
                if len(groups) > 1 and groups[-1][0] == ["final_norm+decoder"]:
                    groups[-2] = (groups[-2][0] + groups[-1][0],
                                  list(groups[-2][1]) + list(groups[-1][1]))
                    groups.pop()
                return [([blocks[0][0]], blocks[0][1])] + groups
            _bc._balance_groups = _keep_emb
        sched = BlockCoordinateScheduler(model, opt, steps_per_block=args.steps_per_block,
                                         partition=partition, partition_budget=budget)
        if arm in BUDGET_LAYERS:
            _bc._balance_groups = _orig_bg
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    print(f"\n[ab] === arm '{arm}' ===\n[ab]     {desc}", flush=True)
    blocks_desc = []
    if sched is not None:
        for name, params in sched.blocks:
            blocks_desc.append((name, sum(p.numel() for p in params)))
            print(f"[ab]     block {name:<70} {blocks_desc[-1][1]/1e6:7.2f}M", flush=True)

    traj = [{"step": 0, "train_sec": 0.0, "heldout_loss": evaluate(model, probes, device)}]
    model.train()
    train_sec = 0.0
    warmup_done = args.warmup_steps <= 0
    block_time: Dict[str, List[float]] = {}
    block_steps: Dict[str, int] = {}
    torch.cuda.reset_peak_memory_stats()

    for i, (bx, by) in enumerate(batches, start=1):
        torch.cuda.synchronize()
        t = time.time()
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)
        last = (local_target(sched, lookahead, args.num_layers)
                if (local and sched is not None) else None)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x) if last is None else forward_to(model, x, last)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        if not warmup_done:
            s = (i - 1) / max(1, args.warmup_steps)
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = b * s
            if i >= args.warmup_steps:
                warmup_done = True
        opt.step()
        opt.zero_grad(set_to_none=True)
        if warmup_done:
            cos.step()
        name = sched.active_block_name if sched is not None else "all"
        if sched is not None:
            sched.step()
        torch.cuda.synchronize()
        dt = time.time() - t
        train_sec += dt
        block_time.setdefault(name, []).append(dt)
        block_steps[name] = block_steps.get(name, 0) + 1

        if i % args.eval_every == 0:
            hl = evaluate(model, probes, device)
            traj.append({"step": i, "train_sec": train_sec, "heldout_loss": hl})
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  train {train_sec/60:.1f} min",
                  flush=True)
            model.train()

    copy = copy_accuracy(model, copy_seqs, device)
    unmoved = [n for n, p in model.named_parameters() if torch.equal(p.detach(), init[n])]
    # Median, so the first step of each block (switch overhead) doesn't dominate.
    per_block = {n: sorted(v)[len(v) // 2] for n, v in block_time.items()}
    out = {"arm": arm, "partition": partition, "local": local, "lookahead": lookahead,
           "blocks": blocks_desc,
           "final_heldout_loss": traj[-1]["heldout_loss"],
           "final_ppl": math.exp(traj[-1]["heldout_loss"]),
           "train_sec": train_sec, "median_step_sec_by_block": per_block,
           "steps_by_block": block_steps,
           "copy": copy, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
           "unmoved_params": unmoved, "trajectory": traj}
    print(f"[ab]   copy acc first-half {copy['first_half']:.3f} repeat {copy['repeat']:.3f}; "
          f"train {train_sec/60:.1f} min; unmoved {len(unmoved)}", flush=True)
    for n, v in per_block.items():
        print(f"[ab]   median step {v*1000:7.1f} ms  {block_steps[n]:>5} steps  {n}", flush=True)
    del model, opt, sched, init
    torch.cuda.empty_cache()
    return out


def heldout_at(traj, sec) -> Optional[float]:
    """Held-out loss at a given training time, linearly interpolated."""
    for a, b in zip(traj, traj[1:]):
        if a["train_sec"] <= sec <= b["train_sec"]:
            f = (sec - a["train_sec"]) / max(b["train_sec"] - a["train_sec"], 1e-9)
            return a["heldout_loss"] + f * (b["heldout_loss"] - a["heldout_loss"])
    return None


def check_forward_to(args, batches, device) -> float:
    model = build(args, device).eval()
    x = batches[0][0][:1].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        diff = float((model(x).float() - forward_to(model, x, args.num_layers - 1).float())
                     .abs().max())
    del model
    torch.cuda.empty_cache()
    return diff


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    p.add_argument("--baseline", default=None,
                   help="JSON from an earlier run with the same config; its arms are "
                        "reused instead of retrained")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--steps-per-block", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--attn-layers", type=int, nargs="*", default=[4, 12])
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--aux-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--out", default="results/local_loss/ab.json")
    args = p.parse_args(argv)

    # Import BEFORE touching CUDA (pyarrow after CUDA init aborts natively).
    import train_edge_conversational as tec
    device = require_cuda()
    tok = tec.load_tokenizer(args.tokenizer)
    args.vocab_size = len(tok)
    torch.manual_seed(args.seed)
    loader, _v, _s = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    pairs = []
    for b in loader:
        x, y = as_pair(b)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= args.probe_batches + args.steps:
            break
    probes, batches = pairs[:args.probe_batches], pairs[args.probe_batches:]
    assert len(batches) == args.steps, f"only {len(batches)} batches"
    g = torch.Generator().manual_seed(args.seed + 1)
    copy_seqs = []
    for _ in range(COPY_SEQS):
        r = torch.randint(0, args.vocab_size, (COPY_LEN,), generator=g)
        copy_seqs.append(torch.cat([r, r]))

    diff = check_forward_to(args, batches, device)
    print(f"[ab] known-answer: forward_to(last layer) vs model(x) max |diff| {diff:.3e}",
          flush=True)
    if diff > 1e-2:
        raise SystemExit("forward_to does not reproduce model(x); local arm would be wrong")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    results: Dict[str, dict] = {}
    if args.baseline:
        base = json.load(open(args.baseline, encoding="utf-8"))
        same = ("steps", "steps_per_block", "warmup_steps", "eval_every", "batch_size",
                "seq_len", "d_model", "num_layers", "attn_layers", "n_heads", "muon_lr",
                "aux_lr", "weight_decay", "probe_batches", "seed", "tokenizer")
        diff_keys = [k for k in same if base["config"].get(k) != getattr(args, k)]
        if diff_keys:
            raise SystemExit(f"--baseline config differs on {diff_keys}; not comparable")
        results.update({n: r for n, r in base["arms"].items() if n not in args.arms})
    for arm in args.arms:
        results[arm] = run_arm(arm, args, batches, probes, copy_seqs, device)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "forward_to_diff": diff, "arms": results},
                      fh, indent=2)

    print("\n[ab] ================ local-loss BCD ================")
    print(f"[ab] {'arm':<20}{'held-out':>10}{'ppl':>9}{'copy':>8}{'train min':>11}{'GiB':>7}")
    for n, r in results.items():
        print(f"[ab] {n:<20}{r['final_heldout_loss']:>10.4f}{r['final_ppl']:>9.1f}"
              f"{r['copy']['repeat']:>8.3f}{r['train_sec']/60:>11.1f}"
              f"{r['peak_allocated_gib']:>7.2f}")
    verdict = []
    locals_ = [a for a in VARIANT_ARMS if a in results]
    if {"full", "bcd_balanced"} <= set(results) and locals_:
        f, b = results["full"], results["bcd_balanced"]
        gap = b["final_heldout_loss"] - f["final_heldout_loss"]
        if gap < 0.1:
            verdict.append(f"VOID: bcd_balanced only {gap:+.3f} nats behind full")
        for a in locals_:
            l = results[a]
            if gap < 0.1:
                break
            if b["unmoved_params"] or l["unmoved_params"]:
                verdict.append(f"{a} VOID: some parameters never trained")
                continue
            d_tok = l["final_heldout_loss"] - b["final_heldout_loss"]
            line = (f"{a}: equal tokens {d_tok:+.4f} nats vs bcd_balanced, "
                    f"{l['train_sec']/b['train_sec']:.2f}x the train time")
            at = heldout_at(b["trajectory"], l["train_sec"])
            if at is not None:
                line += f"; equal train time {l['final_heldout_loss'] - at:+.4f} nats"
            verdict.append(line)
        if f["copy"]["repeat"] - f["copy"]["first_half"] < 0.10:
            verdict.append(f"copy test VOID: full arm only {f['copy']['repeat']:.3f} on the "
                           f"repeat vs {f['copy']['first_half']:.3f} first half")
        else:
            verdict.append("copy (repeat acc): " + ", ".join(
                f"{a} {results[a]['copy']['repeat']:.3f}"
                for a in ["full", "bcd_balanced"] + locals_))
    for v in verdict:
        print(f"[ab] VERDICT: {v}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "forward_to_diff": diff, "arms": results,
                   "verdict": verdict}, fh, indent=2)
    print(f"[ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
