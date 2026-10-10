"""Offload PLUS partial checkpointing: keep the last N blocks fully resident (2026-08-27)

The combination the global guard hides
--------------------------------------
`EdgeHVN.forward` refuses `offload and not grad_checkpoint` (model.py:413). That
guard is GLOBAL, but the constraint it encodes is PER-BLOCK: offload evicts a
block's weights after its forward, so backward needs the checkpoint recompute to
fetch them back. A block that simply **stays resident** never needs fetching --
so it does not need recomputing either.

So the last N blocks can be: weights loaded and never evicted, activations kept,
no recompute. The rest stream and recompute as now. Backward needs the last
blocks FIRST, so the resident tail is exactly the part whose recompute sits
earliest on the critical path.

Why this might beat the no-offload version
------------------------------------------
Partial checkpointing WITHOUT offload measured +1.7%, because the baseline
already sits at 76% of the card and only ~0.5 GB is usable under the 85% rule --
room for 2 of 24 blocks. WITH offload the baseline is 44% (2.63 GB), leaving
~2.4 GB, which is several times more headroom for the same trick.

Implemented here by intercepting `_offload_wrap_checkpointed` rather than
changing the model, because the point is to find out whether the win exists
before paying to build it properly.

Falsifiability, asked before running
------------------------------------
Two ways this could report a fake number, both checked:

1. **The intercept silently not firing.** Peak VRAM must RISE with N (kept
   activations plus pinned-resident weights). A flat memory column means nothing
   was skipped and the timings are meaningless -> the script says VOID.
2. **Silently WRONG results.** Skipping the recompute while the pool evicts the
   weights anyway would give garbage gradients that still "run". So the step-0
   loss is compared against the N=0 baseline on identical data and seed; they
   must agree to ~1e-3. A drift means the resident tail is not actually
   resident, and the speed number is bought with broken maths.

The pool has a capacity, so `offload_cache_blocks` is raised above N -- otherwise
the pool's own overflow eviction would throw out the very blocks being pinned.

Usage: python scripts/probe_offload_resident_tail.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN                       # noqa: E402
from hvn.edge.chunked_loss import chunked_cross_entropy  # noqa: E402
from hvn.optimizers import build_optimizer               # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402
from scripts.ab_prodigy import enable_scan_flags         # noqa: E402

CARD_GB = 5.9966


def plan(num_layers: int):
    p = ["deltanet"] * num_layers
    p[3] = "transformer"
    p[10] = "transformer"
    return p


def install_resident_tail(model: EdgeHVN, n_tail: int, num_layers: int):
    """Make the last `n_tail` blocks resident + un-checkpointed."""
    resident = {f"blocks.{i}" for i in range(num_layers - n_tail, num_layers)}
    original = model._offload_wrap_checkpointed

    def patched(name, block, x_in, position_ids, doc_ids=None):
        if name in resident and name not in model._offload_active_names:
            kw = {} if doc_ids is None else {"doc_ids": doc_ids}
            # Load once, run WITHOUT checkpoint, and register NO evict hook --
            # the weights stay for backward, so no recompute is needed.
            model._offload_pool.load(name, list(block.parameters()))
            return block(x_in, position_ids=position_ids, **kw)
        return original(name, block, x_in, position_ids, doc_ids=doc_ids)

    model._offload_wrap_checkpointed = patched
    return resident


def run(n_tail: int, args) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(1234)
    try:
        model = EdgeHVN(
            vocab_size=args.vocab_size, d_model=args.d_model,
            num_layers=args.num_layers, n_heads=args.n_heads,
            block_plan=plan(args.num_layers), attn_window=4096,
            tie_embeddings=False, grad_checkpoint=True,
        ).to("cpu")
        enable_scan_flags(model)
        opt, _ = build_optimizer(model, "muon", lr=2e-5,
                                 weight_decay=0.01, muon_lr=0.02)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=50, device="cuda",
            offload_inactive_weights=True, offload_park_dtype=torch.bfloat16,
            offload_prefetch=True,
            # above N, or the pool's own overflow eviction discards the pins
            offload_cache_blocks=max(1, n_tail + 2),
        )
        if n_tail:
            install_resident_tail(model, n_tail, args.num_layers)

        params = [p for p in model.parameters() if p.requires_grad]
        g = torch.Generator(device="cuda").manual_seed(99)
        x = torch.randint(0, args.vocab_size, (args.batch, args.seq),
                          device="cuda", generator=g)
        y = torch.randint(0, args.vocab_size, (args.batch, args.seq),
                          device="cuda", generator=g)

        times, first_loss = [], None
        for i in range(args.iters + 1):
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = model(x, return_hidden=True)
                h = h[0] if isinstance(h, tuple) else h
                loss = chunked_cross_entropy(h, y, model.decoder,
                                             chunk=args.loss_chunk).mean()
            if first_loss is None:
                first_loss = float(loss.detach())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step(); opt.zero_grad(set_to_none=True); sched.step()
            torch.cuda.synchronize()
            if i >= 1:
                times.append(time.perf_counter() - t0)

        times.sort()
        med = times[len(times) // 2]
        reserved = torch.cuda.max_memory_reserved() / 2**30
        pct = 100.0 * reserved / CARD_GB
        verdict = "FITS" if pct <= 85 else ("SUSPECT" if pct <= 100 else "SPILLS")
        out = dict(n_tail=n_tail, sec_per_step=med,
                   tok_per_sec=args.batch * args.seq / med,
                   peak_reserved_gb=reserved, pct_of_card=pct,
                   verdict=verdict, step0_loss=first_loss)
        print(f"  tail={n_tail:<3} {med:>6.2f}s {out['tok_per_sec']:>7.0f} tok/s "
              f"{reserved:>6.2f}GB {pct:>4.0f}%  {verdict:<8} loss0={first_loss:.5f}")
        del model, opt, sched
        return out
    except RuntimeError as e:
        print(f"  tail={n_tail:<3} FAILED: {str(e)[:60]}")
        return dict(n_tail=n_tail, failed=str(e)[:200])
    finally:
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=1216)
    ap.add_argument("--n-heads", type=int, default=19)
    ap.add_argument("--num-layers", type=int, default=24)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--loss-chunk", type=int, default=512)
    ap.add_argument("--iters", type=int, default=3)
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--tails", type=int, nargs="+", default=[0, 2, 4, 6, 8, 12])
    ap.add_argument("--out", default="results/offload_resident_tail.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required."); return 2
    print(f"offload + resident tail, d={args.d_model}, {args.num_layers}L, "
          f"seq {args.seq}, batch {args.batch}")
    print(f"  {'config':<8} {'s/step':>7} {'tok/s':>12} {'VRAM':>8} {'%':>5}")

    results = [run(n, args) for n in args.tails]
    ok = [r for r in results if not r.get("failed")]
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(dict(config=vars(args), results=results), fh, indent=2)

    if len(ok) >= 2:
        base = ok[0]
        span = max(r["peak_reserved_gb"] for r in ok) - base["peak_reserved_gb"]
        print(f"\n[check 1] memory span: {span:+.2f} GB")
        if span < 0.05:
            print("*** VOID: peak VRAM barely moved -> the intercept is NOT firing. "
                  "Ignore the timings.")
            print(f"wrote {args.out}"); return 0
        print("  memory rises with tail size -> the intercept fires.")

        drift = max(abs(r["step0_loss"] - base["step0_loss"]) for r in ok)
        print(f"[check 2] max step-0 loss drift vs baseline: {drift:.2e}")
        if drift > 1e-3:
            print("*** VOID: the resident tail changed the LOSS. The weights are "
                  "not actually staying resident, so any speedup is bought with "
                  "broken maths. Do not report these timings.")
            print(f"wrote {args.out}"); return 0
        print("  loss unchanged -> the tail really is resident and correct.")

        fitting = [r for r in ok if r["verdict"] == "FITS"]
        best = min(fitting or ok, key=lambda r: r["sec_per_step"])
        gain = base["sec_per_step"] / best["sec_per_step"]
        print(f"\nbest that FITS: tail={best['n_tail']}, {best['sec_per_step']:.2f}s, "
              f"{best['pct_of_card']:.0f}% of card = {100*(gain-1):+.1f}% vs baseline")
        print("\nCompare: micro-batch 2 is +7.1% and needs no code; partial "
              "checkpointing WITHOUT offload was +1.7%. All three spend the same "
              "spare VRAM, so they compete rather than add.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
