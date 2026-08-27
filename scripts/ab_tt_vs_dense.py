"""A/B: what does Tensor-Train factorisation cost, and does it buy anything? (Issue #161)

Author: Rambarun Komaljeet

#161 landed TTLinear and measured its SIZE and SPEED: 3.96x fewer block
parameters at d=896 for 1.05x step time. It measured nothing about QUALITY.
This is that measurement.

The two questions, which need different arms
--------------------------------------------
1. **At equal width**, what does compressing the projections cost?
   ``tt`` vs ``dense`` at the same d_model. TT has ~3x fewer params and does
   the SAME FLOPs. Biased AGAINST TT.
2. **At equal parameters**, is TT-wider better than dense-narrower?
   ``tt_wide`` vs ``dense`` with d_model chosen so body params match. This is
   the question that decides whether TT is a capability lever or only a storage
   one.

The bias in each direction, stated BEFORE the run rather than after
------------------------------------------------------------------
TT reduces PARAMETERS, not FLOPs. A matched-parameter TT arm is therefore
WIDER, and does substantially MORE compute per step -- so:

* ``tt_wide`` beating ``dense`` at matched STEPS is **weak** evidence: it was
  handed more compute to get there.
* ``tt_wide`` beating ``dense`` at matched WALL CLOCK is **strong** evidence,
  because that is the budget a real run actually spends.
* ``tt`` (equal width) losing to ``dense`` is **expected** -- it has 3x fewer
  parameters at identical compute. The number that matters there is HOW MUCH it
  loses, against the 3x it saves.

All three are reported. A single headline would hide which one it is.

Discipline inherited from ab_looped_small.py (#122) / ab_n_heads.py (#134)
--------------------------------------------------------------------------
Identical seed, identical pre-materialised batches for every arm, graded on
**held-out loss, never train loss** (#132 was retracted for exactly that),
``--deterministic`` ON, and a **known-answer twin cell**: ``dense_twin`` is
``dense`` again under another name. #122 measured that the twin must run
FULL LENGTH -- drift accumulates, and a smoke-length twin understated the real
floor by ~50x -- so the twin gets the same step count as every other arm and
its drift, not a guess, sets the significance band.

Falsifiability, asked before the run
------------------------------------
If TT silently did nothing, would this catch it? Yes: the script asserts the TT
arms have strictly fewer body parameters than dense before training starts, and
aborts if they do not. Without that, a mis-wired flag would produce three arms
of the same model and a beautiful, meaningless null result.

If TT were catastrophic for quality, would this catch it? Yes -- held-out loss
at matched tokens separates that cleanly. What it canNOT resolve is a small
win: arms landing inside ``--inconclusive-band`` are reported INCONCLUSIVE and
the script refuses to call it.

Peak VRAM is recorded per arm because "does TT save memory" is the question
that prompted this, and it is cheap to answer properly rather than by
arithmetic.

One number to read carefully
----------------------------
A DeltaNet *block* compresses 3.07x at d=384 rank 32, but the *model body* only
1.81x -- because 4 of the 12 layers are attention and are never TT'd, and the
gates/norms inside each DeltaNet block stay dense too. Quote the model-level
number when talking about a checkpoint; the block-level one describes a layer,
not a model. In a pure-DeltaNet stack the two would be much closer.

Usage::

    python scripts/ab_tt_vs_dense.py --steps 5000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ab_prodigy import as_pair, enable_scan_flags, require_cuda  # noqa: E402
from hvn.optimizers import build_optimizer                              # noqa: E402

GPT2_VOCAB_SIZE = 50257

#: One repeating unit, tiled to num_layers: attention then two recurrent layers.
UNIT = ["transformer", "deltanet", "deltanet"]

TT_TARGETS = ("qkv", "out", "ffn")


def block_plan_for(num_layers: int) -> List[str]:
    return [UNIT[i % len(UNIT)] for i in range(num_layers)]


def build_model(d_model: int, n_heads: int, tt: bool, args, device):
    from hvn.edge.model import EdgeHVN

    kw = {}
    if tt:
        kw = dict(tt_targets=TT_TARGETS, tt_rank=args.tt_rank,
                  tt_contract="fused")
    model = EdgeHVN(
        vocab_size=args.vocab_size,
        d_model=d_model,
        num_layers=args.num_layers,
        n_heads=n_heads,
        block_plan=block_plan_for(args.num_layers),
        attn_window=args.attn_window,
        tie_embeddings=True,
        **kw,
    ).to(device)
    return model


def body_params(model) -> int:
    """Params excluding the token embedding.

    At d_model 384 a 50257-row table is ~19M and would swamp the body ratio this
    A/B exists to measure. It is also the ONLY term that changes with d_model
    for reasons unrelated to TT, so folding it in would make the matched-params
    arm look matched when its bodies were not.
    """
    total = sum(p.numel() for p in model.parameters())
    return total - model.token_embedding.weight.numel()


@torch.no_grad()
def evaluate(model, pairs, device) -> float:
    model.eval()
    total = 0.0
    for x, y in pairs:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            total += float(nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1)))
    return total / max(len(pairs), 1)


def run_arm(arm: str, spec: dict, args, batches, probes, device) -> dict:
    # Re-seed identically per arm: arms must differ in the factorisation and
    # width alone, not in their init draw or data order.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.cuda.reset_peak_memory_stats()

    model = build_model(spec["d_model"], spec["n_heads"], spec["tt"], args, device)
    enable_scan_flags(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_body = body_params(model)
    opt, desc = build_optimizer(model, "muon", lr=args.aux_lr,
                                weight_decay=args.weight_decay,
                                muon_lr=args.muon_lr)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    print(f"\n[ab] === arm '{arm}' (d_model {spec['d_model']}, "
          f"{'TT r%d' % args.tt_rank if spec['tt'] else 'dense'}, "
          f"{n_params:,} params / {n_body:,} body) ===\n[ab]     {desc}",
          flush=True)

    traj = [{"step": 0, "heldout_loss": evaluate(model, probes, device),
             "wall_sec": 0.0}]
    print(f"[ab]   step {0:>5}  held-out {traj[0]['heldout_loss']:.5f}", flush=True)

    curve: List[dict] = []
    model.train()
    t0 = time.time()
    warmup_done = args.warmup_steps <= 0

    for i, (bx, by) in enumerate(batches, start=1):
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            logits = logits[0] if isinstance(logits, tuple) else logits
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

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

        curve.append({"step": i, "loss": float(loss.detach())})

        if i % args.eval_every == 0:
            hl = evaluate(model, probes, device)
            wall = time.time() - t0
            traj.append({"step": i, "heldout_loss": hl, "wall_sec": wall})
            recent = [c["loss"] for c in curve[-args.eval_every:]]
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  "
                  f"train {sum(recent)/len(recent):.5f}  ({wall/60:.1f} min)",
                  flush=True)
            model.train()

    wall_total = time.time() - t0
    peak_alloc = torch.cuda.max_memory_allocated() / 2**20
    peak_reserved = torch.cuda.max_memory_reserved() / 2**20

    out = {
        "arm": arm, "d_model": spec["d_model"], "n_heads": spec["n_heads"],
        "tt": spec["tt"], "tt_rank": args.tt_rank if spec["tt"] else None,
        "params": n_params, "body_params": n_body,
        "peak_alloc_mib": peak_alloc, "peak_reserved_mib": peak_reserved,
        "wall_clock_sec": wall_total,
        "steps_per_sec": args.steps / wall_total,
        "final_heldout_loss": traj[-1]["heldout_loss"],
        "start_heldout_loss": traj[0]["heldout_loss"],
        "trajectory": traj, "curve": curve,
    }
    print(f"[ab]   peak VRAM: {peak_alloc:.0f} MiB alloc / {peak_reserved:.0f} MiB reserved")
    del model, opt
    torch.cuda.empty_cache()
    return out


def heldout_at_wall(traj: List[dict], wall_sec: float) -> Optional[float]:
    ok = [t for t in traj if t.get("wall_sec", 0.0) <= wall_sec]
    return ok[-1]["heldout_loss"] if ok else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+",
                   default=["dense", "dense_twin", "tt", "tt_wide"])
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--d-model", type=int, default=384, help="baseline width")
    p.add_argument("--n-heads", type=int, default=6, help="d_head 64 at d_model 384")
    # Solved, not guessed: TT body at d=512 is 16,920,721 against the dense
    # d=384 body of 17,808,769 -- 0.950x. The nearest alternative (d=576) is
    # 1.165x, i.e. 16% OVER. 512 is both closer and biased the safe way: the TT
    # arm gets slightly FEWER parameters, so a TT win cannot be explained by
    # extra capacity. d_head stays 64 (power of 2, or the Triton scan crashes).
    p.add_argument("--wide-d-model", type=int, default=512,
                   help="TT width chosen so body params ~match the dense arm")
    p.add_argument("--wide-n-heads", type=int, default=8, help="d_head 64 at 512")
    p.add_argument("--tt-rank", type=int, default=32)
    p.add_argument("--attn-window", type=int, default=4096)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--aux-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vocab-size", type=int, default=GPT2_VOCAB_SIZE)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--inconclusive-band", type=float, default=0.02)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", default="results/tt/ab_tt_vs_dense.json")
    args = p.parse_args(argv)

    specs = {
        "dense":      dict(d_model=args.d_model, n_heads=args.n_heads, tt=False),
        "dense_twin": dict(d_model=args.d_model, n_heads=args.n_heads, tt=False),
        "tt":         dict(d_model=args.d_model, n_heads=args.n_heads, tt=True),
        "tt_wide":    dict(d_model=args.wide_d_model, n_heads=args.wide_n_heads, tt=True),
    }

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[ab] deterministic ON -- dense and dense_twin set the noise floor")

    device = require_cuda()
    import train_edge_conversational as tec

    tok = tec.load_tokenizer(args.tokenizer)
    # #155: len(tokenizer), NOT tokenizer.vocab_size.
    if args.vocab_size != len(tok):
        args.vocab_size = len(tok)

    # ---- guard: TT must actually be smaller, BEFORE spending GPU hours -----
    # A mis-wired flag would otherwise give several arms of the same model and
    # a beautiful, meaningless null result.
    probe_counts = {}
    for name in ("dense", "tt", "tt_wide"):
        if name not in args.arms:
            continue
        s = specs[name]
        m = build_model(s["d_model"], s["n_heads"], s["tt"], args, "cpu")
        probe_counts[name] = body_params(m)
        del m
    print("[ab] body params:", {k: f"{v:,}" for k, v in probe_counts.items()})
    if "tt" in probe_counts and "dense" in probe_counts:
        if probe_counts["tt"] >= probe_counts["dense"]:
            print("[ab] ABORT: the 'tt' arm is not smaller than 'dense'. TT is "
                  "not taking effect; fix the wiring before spending GPU time.")
            return 2
        print(f"[ab] equal-width TT is {probe_counts['dense']/probe_counts['tt']:.2f}x "
              f"smaller -- the compression under test")
    if "tt_wide" in probe_counts and "dense" in probe_counts:
        ratio = probe_counts["tt_wide"] / probe_counts["dense"]
        print(f"[ab] matched-params arm is {ratio:.3f}x the dense body "
              f"(1.000 would be exact)")
        if not 0.75 <= ratio <= 1.35:
            print("[ab] WARNING: 'tt_wide' is not close to matched. Adjust "
                  "--wide-d-model; the matched-params comparison is only "
                  "meaningful if the bodies really match.")

    torch.manual_seed(args.seed)
    loader, _v, stats = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    print(f"[ab] data: {stats}")

    pairs = []
    for batch in loader:
        x, y = as_pair(batch)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= args.probe_batches + args.steps:
            break
    probes, batches = pairs[:args.probe_batches], pairs[args.probe_batches:]
    print(f"[ab] {len(probes)} held-out probes + {len(batches)} identical "
          f"training batches for every arm")

    results: Dict[str, dict] = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for arm in args.arms:
        results[arm] = run_arm(arm, specs[arm], args, batches, probes, device)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"issue": 161, "config": vars(args), "arms": results},
                      fh, indent=2)

    # ---- verdict -----------------------------------------------------------
    print("\n[ab] ================= TT vs dense =================")
    print(f"[ab] {'arm':<12}{'d':>6}{'body':>12}{'held-out':>11}"
          f"{'VRAM MiB':>10}{'min':>8}{'steps/s':>9}")
    for name, r in results.items():
        print(f"[ab] {name:<12}{r['d_model']:>6}{r['body_params']:>12,}"
              f"{r['final_heldout_loss']:>11.5f}{r['peak_reserved_mib']:>10.0f}"
              f"{r['wall_clock_sec']/60:>8.1f}{r['steps_per_sec']:>9.2f}")

    floor = 0.0
    verdict: Dict[str, str] = {}
    if "dense" in results and "dense_twin" in results:
        floor = abs(results["dense"]["final_heldout_loss"]
                    - results["dense_twin"]["final_heldout_loss"])
        print(f"\n[ab] KNOWN-ANSWER CELL -- dense vs dense_twin: {floor:.3e}")
        if floor == 0.0:
            print("[ab] twin arms bit-identical -> noise floor exactly 0.")
        else:
            print("[ab] *** noise floor is NOT zero. Any gap at or below this "
                  "drift cannot be distinguished from it.")
    band = max(args.inconclusive_band, floor)

    def call(label: str, a: dict, b: dict, note: str) -> None:
        """b - a, where positive means b is WORSE."""
        gap = b["final_heldout_loss"] - a["final_heldout_loss"]
        print(f"\n[ab] {label}: {gap:+.5f} nats  (band {band:.5f})")
        print(f"[ab]   {note}")
        if abs(gap) <= band:
            verdict[label] = f"INCONCLUSIVE ({gap:+.5f} within {band:.5f})"
        else:
            verdict[label] = (f"{'WORSE' if gap > 0 else 'BETTER'} by "
                              f"{abs(gap):.5f} nats ({abs(gap)/max(band,1e-9):.1f}x band)")
        print(f"[ab]   VERDICT: {verdict[label]}")

    if "tt" in results and "dense" in results:
        ratio = results["dense"]["body_params"] / results["tt"]["body_params"]
        call("EQUAL WIDTH (tt vs dense)", results["dense"], results["tt"],
             f"TT has {ratio:.2f}x fewer body params at the SAME FLOPs. A loss "
             f"here is expected; the question is whether it is small enough to "
             f"be worth {ratio:.2f}x the storage.")

    if "tt_wide" in results and "dense" in results:
        a, b = results["dense"], results["tt_wide"]
        call("MATCHED PARAMS, matched steps (tt_wide vs dense)", a, b,
             "WEAK evidence either way: tt_wide is wider, so it did MORE "
             "compute per step to get here. Read the wall-clock line below.")
        budget = min(a["wall_clock_sec"], b["wall_clock_sec"])
        ha, hb = heldout_at_wall(a["trajectory"], budget), heldout_at_wall(b["trajectory"], budget)
        if ha is not None and hb is not None:
            gap = hb - ha
            print(f"\n[ab] MATCHED WALL CLOCK ({budget/60:.1f} min): "
                  f"tt_wide - dense = {gap:+.5f} nats")
            print("[ab]   This is the STRONG comparison -- equal budget spent.")
            verdict["MATCHED WALL CLOCK"] = (
                "INCONCLUSIVE" if abs(gap) <= band
                else f"{'WORSE' if gap > 0 else 'BETTER'} by {abs(gap):.5f} nats")
            print(f"[ab]   VERDICT: {verdict['MATCHED WALL CLOCK']}")

    print("\n[ab] VRAM, measured rather than argued:")
    for name, r in results.items():
        print(f"[ab]   {name:<12} {r['peak_reserved_mib']:>7.0f} MiB reserved")
    print("[ab]   NOTE: TT shrinks weights+grads+optimiser state, NOT "
          "activations and NOT the CE logits. At this size the logits dominate "
          "(#158), so the total saving is far smaller than the param ratio.")

    print("\n[ab] LIMITS: one seed, one corpus, short from-scratch runs at "
          f"d_model {args.d_model}/{args.wide_d_model}. Bounds EARLY-training "
          "quality only -- not convergence, and not the sizes actually "
          "deployed. Ternary is NOT applied in any arm, so this measures TT "
          "against dense fp32, not against the ternary baseline TT would have "
          "to beat to be adopted.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 161, "config": vars(args), "arms": results,
                   "noise_floor": floor, "verdict": verdict}, fh, indent=2)
    print(f"[ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
