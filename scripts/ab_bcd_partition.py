"""Does packing layers into balanced BCD groups close BCD's quality gap?

Author: Rambarun Komaljeet

#112 measured block-coordinate descent at ~3.9x worse perplexity and #151 showed
the cost is under-training (each block trains 1/N of the steps), not optimizer
amnesia. This A/B tests the one scheduling lever that attacks the 1/N without
touching memory: BCD's peak grad + optimizer-state residency is set by its
LARGEST block -- the 50,257 x d vocab table -- while one layer is ~5x smaller at
d=1024. partition="balanced" packs consecutive layers up to that same size,
so the peak is unchanged by construction and each layer trains ~3-4x as often.

Arms (identical seed, batches, LR schedule):
  full          no BCD (reference ceiling)
  bcd_layer     today's BCD: embedding | one block per layer | final_norm+decoder
  bcd_balanced  same scheduler, layers packed up to the largest block's size

Optional arm (#164), not in the default set:
  bcd_balanced_offload  bcd_balanced + CPU weight offload exactly as
                        train_edge_conversational.py --offload runs it (bf16
                        parking, prefetch, grad checkpointing). Offload is
                        meant to be a placement change only; this measures
                        whether the quality survives it, including bf16
                        parking's rounding of each block's weights on every
                        switch. --baseline loads full/bcd_layer/bcd_balanced
                        from an earlier run so only this arm has to train.

Guards against a vacuous result:
  * known-answer: bcd_layer must reproduce a real gap vs full, or the harness is
    not measuring BCD's cost and no verdict is printed.
  * every parameter must have moved from its init by the end of each BCD arm
    (a group that never activated would leave its weights untouched).
  * peak grad + optimizer-state bytes are MEASURED per arm, not assumed, so the
    "memory unchanged" claim is checked, not asserted.

Shape defaults mirror the 270M POC's per-layer/vocab ratio (d=1024, pure
DeltaNet, tied GPT-2 vocab, ternary): a layer is 10.5M params and the table
51.5M, so balanced groups are 4 layers, as they would be at 270M.
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

ARMS = {"full": None, "bcd_layer": "layer", "bcd_balanced": "balanced",
        "bcd_balanced_offload": "balanced"}
OFFLOAD_ARMS = {"bcd_balanced_offload"}
DEFAULT_ARMS = ["full", "bcd_layer", "bcd_balanced"]


def build(args, device, grad_checkpoint=False):
    from hvn.edge.model import EdgeHVN
    plan = ["deltanet"] * args.num_layers
    model = EdgeHVN(
        vocab_size=args.vocab_size, d_model=args.d_model,
        num_layers=len(plan), n_heads=args.n_heads, block_plan=plan,
        tie_embeddings=args.tie, grad_checkpoint=grad_checkpoint,
    ).to(device)
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(
        model, target_names=[n for n in linears
                             if not any(p in n for p in exclude)])
    enable_scan_flags(model)
    return model


def _state_bytes(model, opt) -> int:
    """Bytes of .grad + optimizer state currently resident (what BCD shrinks)."""
    n = 0
    for p in model.parameters():
        if p.grad is not None:
            n += p.grad.numel() * p.grad.element_size()
    for st in opt.state.values():
        for v in st.values():
            if torch.is_tensor(v):
                n += v.numel() * v.element_size()
    return n


def evaluate_offloaded(model, probes, device, sched) -> float:
    """evaluate() for an offloaded model, on the weights it is actually training.

    EdgeHVN.forward() only runs the offload load/evict path when training with
    grad enabled, so a plain eval forward hits the parked CPU (bf16) weights and
    dies on a device mismatch. Bring every CPU-parked parameter to the GPU at its
    native dtype for the eval, then put the SAME CPU tensor back: the pool's
    registered CPU home is that tensor, so training resumes untouched. The
    parked values are already bf16-rounded, which is the cost being measured.
    """
    if sched is None or not sched.offload_inactive_weights:
        return evaluate(model, probes, device)
    torch.cuda.synchronize()            # no prefetch copy may still be in flight
    pool = sched.buffer_pool
    homes = {}
    for p in model.parameters():
        if p.device.type == "cpu":
            homes[p] = p.detach()       # same storage as the pool's CPU home
            p.data = p.data.to(device=device, dtype=pool.native_dtype_for(p))
    assert homes, "offload arm has no CPU-parked parameters -- not offloading"
    try:
        loss = evaluate(model, probes, device)
    finally:
        for p, home in homes.items():
            p.data = home
    # .data returns a fresh wrapper each access, so compare storage, not identity.
    assert all(p.device.type == "cpu" and p.data_ptr() == h.data_ptr()
               for p, h in homes.items())
    return loss


def run_arm(arm: str, args, batches, probes, device) -> dict:
    partition = ARMS[arm]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    offload = arm in OFFLOAD_ARMS
    model = build(args, device, grad_checkpoint=offload)
    init = {n: p.detach().clone() for n, p in model.named_parameters()}
    opt, desc = build_optimizer(model, "muon", lr=args.aux_lr,
                                weight_decay=args.weight_decay,
                                muon_lr=args.muon_lr)
    sched = None
    if partition is not None:
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=args.steps_per_block,
            partition=partition,
            # Same settings as train_edge_conversational.py --offload.
            offload_inactive_weights=offload,
            device=device if offload else None,
            offload_park_dtype=torch.bfloat16 if offload else None,
            offload_prefetch=offload)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    print(f"\n[ab] === arm '{arm}' ===\n[ab]     {desc}", flush=True)
    blocks_desc = []
    if sched is not None:
        for name, params in sched.blocks:
            blocks_desc.append((name, sum(p.numel() for p in params)))
            print(f"[ab]     block {name:<60} {blocks_desc[-1][1]/1e6:8.2f}M")
        rot = len(sched.blocks) * args.steps_per_block
        print(f"[ab]     {len(sched.blocks)} blocks x {args.steps_per_block} steps "
              f"= rotation {rot} steps; {args.steps / rot:.2f} rotations", flush=True)

    traj = [{"step": 0, "heldout_loss": evaluate_offloaded(model, probes, device, sched)}]
    curve: List[dict] = []
    model.train()
    t0 = time.time()
    warmup_done = args.warmup_steps <= 0
    peak_state = 0
    torch.cuda.reset_peak_memory_stats()

    for i, (bx, by) in enumerate(batches, start=1):
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        if not warmup_done:
            s = (i - 1) / max(1, args.warmup_steps)
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = b * s
            if i >= args.warmup_steps:
                warmup_done = True
        opt.step()
        # Measured after step(), before zero_grad: grads AND state both live.
        peak_state = max(peak_state, _state_bytes(model, opt))
        opt.zero_grad(set_to_none=True)
        if warmup_done:
            cos.step()
        if sched is not None:
            sched.step()
        curve.append({"step": i, "loss": float(loss.detach())})

        if i % args.eval_every == 0:
            hl = evaluate_offloaded(model, probes, device, sched)
            traj.append({"step": i, "heldout_loss": hl,
                         "wall_sec": time.time() - t0})
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            model.train()

    # Offload parks inactive params on the CPU in bf16: compare against the init
    # rounded the same way, so an untrained parked param still counts as unmoved.
    unmoved = [n for n, p in model.named_parameters()
               if torch.equal(p.detach().cpu(), init[n].to(p.dtype).cpu())]
    out = {"arm": arm, "partition": partition, "blocks": blocks_desc,
           "final_heldout_loss": traj[-1]["heldout_loss"],
           "final_ppl": math.exp(traj[-1]["heldout_loss"]),
           "wall_clock_sec": time.time() - t0,
           "peak_grad_plus_state_gib": peak_state / 2**30,
           "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
           "unmoved_params": unmoved,
           "trajectory": traj, "curve": curve}
    print(f"[ab]   peak grad+optimizer state {out['peak_grad_plus_state_gib']:.3f} GiB, "
          f"peak allocated {out['peak_allocated_gib']:.3f} GiB, "
          f"unmoved params: {len(unmoved)} {unmoved[:5]}", flush=True)
    del model, opt, sched, init
    torch.cuda.empty_cache()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS, choices=list(ARMS))
    p.add_argument("--baseline", default=None,
                   help="JSON from an earlier run of this script with the same "
                        "config; its arms are reused for the verdict")
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--steps-per-block", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=1024)
    p.add_argument("--num-layers", type=int, default=8)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--tie", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--muon-lr", type=float, default=0.02)
    # 3e-4 = train_edge_conversational's default --lr, which is what the AdamW
    # aux group (embedding/head/norms) actually runs at in real training.
    p.add_argument("--aux-lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--out", default="results/bcd_partition/ab.json")
    args = p.parse_args(argv)

    # Import BEFORE touching CUDA: it pulls in pandas/pyarrow, and loading
    # pyarrow after CUDA is initialised aborts the process natively with no
    # traceback (train_edge_conversational.py:40-61).
    import train_edge_conversational as tec
    device = require_cuda()
    tok = tec.load_tokenizer(args.tokenizer)
    args.vocab_size = len(tok)          # CLAUDE.md: never tokenizer.vocab_size
    torch.manual_seed(args.seed)
    loader, _v, _stats = tec.build_dataset_pipeline(
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
    print(f"[ab] {len(probes)} held-out probe batches + {len(batches)} identical "
          f"training batches per arm", flush=True)

    results: Dict[str, dict] = {}
    if args.baseline:
        base = json.load(open(args.baseline, encoding="utf-8"))
        same = ("steps", "steps_per_block", "batch_size", "seq_len", "d_model",
                "num_layers", "n_heads", "tie", "muon_lr", "aux_lr", "seed",
                "warmup_steps", "eval_every", "probe_batches", "tokenizer")
        diff = [k for k in same if base["config"].get(k) != getattr(args, k)]
        if diff:
            raise SystemExit(f"--baseline config differs on {diff}; not comparable")
        for n, r in base["arms"].items():
            if n not in args.arms:
                results[n] = r
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for arm in args.arms:
        results[arm] = run_arm(arm, args, batches, probes, device)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"config": vars(args), "arms": results}, fh, indent=2)

    print("\n[ab] ================ BCD partition ================")
    print(f"[ab] {'arm':<14}{'held-out':>10}{'ppl':>10}{'state GiB':>11}{'min':>7}")
    for n, r in results.items():
        print(f"[ab] {n:<14}{r['final_heldout_loss']:>10.4f}{r['final_ppl']:>10.1f}"
              f"{r['peak_grad_plus_state_gib']:>11.3f}{r['wall_clock_sec']/60:>7.1f}")
    verdict = None
    if {"full", "bcd_layer", "bcd_balanced"} <= set(results):
        f = results["full"]["final_heldout_loss"]
        lay = results["bcd_layer"]["final_heldout_loss"]
        bal = results["bcd_balanced"]["final_heldout_loss"]
        if lay - f < 0.3:
            verdict = f"VOID: bcd_layer only {lay - f:+.3f} nats behind full; no gap to close"
        elif any(results[a]["unmoved_params"] for a in ("bcd_layer", "bcd_balanced")):
            verdict = "VOID: some parameters never trained in a BCD arm"
        else:
            closed = (lay - bal) / (lay - f) * 100
            verdict = (f"balanced closes {closed:.0f}% of BCD's gap "
                       f"(layer {lay - f:+.3f} nats vs full, balanced {bal - f:+.3f})")
    off_note = None
    if {"bcd_balanced", "bcd_balanced_offload"} <= set(results):
        a = results["bcd_balanced"]["trajectory"]
        b = results["bcd_balanced_offload"]["trajectory"]
        worst = max(abs(x["heldout_loss"] - y["heldout_loss"]) for x, y in zip(a, b))
        d_fin = (results["bcd_balanced_offload"]["final_heldout_loss"]
                 - results["bcd_balanced"]["final_heldout_loss"])
        off_note = (f"offload vs no offload (balanced): final {d_fin:+.4f} nats, "
                    f"worst eval point {worst:.4f} nats over {len(a)} evals")
        if "full" in results and "bcd_layer" in results:
            f = results["full"]["final_heldout_loss"]
            lay = results["bcd_layer"]["final_heldout_loss"]
            bo = results["bcd_balanced_offload"]["final_heldout_loss"]
            off_note += (f"; balanced+offload closes "
                         f"{(lay - bo) / (lay - f) * 100:.0f}% of BCD's gap")
        print(f"[ab] {off_note}")
    print(f"\n[ab] VERDICT: {verdict}")
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "arms": results, "verdict": verdict,
                   "offload_note": off_note}, fh, indent=2)
    print(f"[ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
