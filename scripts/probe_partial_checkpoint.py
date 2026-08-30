"""Is PARTIAL gradient checkpointing worth building? (2026-08-27)

The idea, and why it is the right shape
---------------------------------------
Gradient checkpointing recomputes each block's forward during the backward pass.
At 1B that recompute is ~1.0 s of a 4.02 s step -- and it is ARITHMETIC, not
waiting. Since the step is 98.7% GPU-bound, removing real FLOPs is the only
class of fix that can help, which makes this the one lever pointed at the actual
bottleneck. (Caching WEIGHTS in spare VRAM -- `offload_cache_blocks` -- was
measured at +0.3%, because it saves transfers and transfers are not the cost.)

Keeping a block's activations resident instead of recomputing them trades VRAM
for FLOPs, continuously. Checkpointing is bit-identical either way (#49/#100),
so any win here is free of quality risk.

What is measured
----------------
The last N of 24 blocks are left UN-checkpointed (their activations kept); the
rest recompute as now. The last blocks are chosen deliberately -- backward needs
them first, so they are the ones whose recompute is on the critical path
earliest.

Measured WITHOUT offload. `EdgeHVN.forward` refuses `offload and not
grad_checkpoint` (model.py:413) because offload evicts a block's weights after
its forward and relies on the recompute to bring them back. Partial
checkpointing under offload therefore needs the eviction logic changed too --
out of scope for a probe that only asks whether the win exists at all.

Falsifiability, asked before running
------------------------------------
If the intercept silently failed to skip anything, would this catch it? Yes:
peak VRAM must RISE with N, because kept activations occupy memory. A flat
memory column means the patch is not taking effect and the flat TIME column
would be meaningless -- so the script checks and says so rather than reporting
a null result it cannot distinguish from a broken probe.

Usage: python scripts/probe_partial_checkpoint.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hvn.edge.model as edge_model                      # noqa: E402
from hvn.edge.model import EdgeHVN                       # noqa: E402
from hvn.edge.chunked_loss import chunked_cross_entropy  # noqa: E402
from hvn.optimizers import build_optimizer               # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402
from scripts.ab_prodigy import enable_scan_flags         # noqa: E402

CARD_GB = 5.9966
_real_checkpoint = edge_model.checkpoint


class SkipLast:
    """Make the last `n` checkpoint calls of each forward pass pass-through."""

    def __init__(self, n: int, total: int):
        self.n, self.total, self.i = n, total, 0

    def reset(self):
        self.i = 0

    def __call__(self, func, *args, **kwargs):
        idx = self.i
        self.i += 1
        if self.n > 0 and idx >= self.total - self.n:
            kwargs.pop("use_reentrant", None)
            return func(*args)          # no checkpoint: keep the activations
        return _real_checkpoint(func, *args, **kwargs)


def plan(num_layers: int):
    p = ["deltanet"] * num_layers
    p[3] = "transformer"
    p[10] = "transformer"
    return p


def run(n_skip: int, args) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    patch = SkipLast(n_skip, args.num_layers)
    edge_model.checkpoint = patch
    try:
        model = EdgeHVN(
            vocab_size=args.vocab_size, d_model=args.d_model,
            num_layers=args.num_layers, n_heads=args.n_heads,
            block_plan=plan(args.num_layers), attn_window=4096,
            tie_embeddings=True, grad_checkpoint=True,
        ).cuda()
        enable_scan_flags(model)
        opt, _ = build_optimizer(model, "muon", lr=2e-5,
                                 weight_decay=0.01, muon_lr=0.02)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=50, device="cuda",
            offload_inactive_weights=False,
        )
        params = [p for p in model.parameters() if p.requires_grad]
        x = torch.randint(0, args.vocab_size, (args.batch, args.seq), device="cuda")
        y = torch.randint(0, args.vocab_size, (args.batch, args.seq), device="cuda")

        times = []
        for i in range(args.iters + 1):
            patch.reset()
            torch.cuda.synchronize(); t0 = time.perf_counter()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                h = model(x, return_hidden=True)
                h = h[0] if isinstance(h, tuple) else h
                loss = chunked_cross_entropy(h, y, model.decoder,
                                             chunk=args.loss_chunk).mean()
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
        out = dict(n_skip=n_skip, sec_per_step=med,
                   tok_per_sec=args.batch * args.seq / med,
                   peak_reserved_gb=reserved, pct_of_card=pct, verdict=verdict,
                   ckpt_calls=patch.i)
        print(f"  skip_last={n_skip:<3} {med:>6.2f}s {out['tok_per_sec']:>7.0f} tok/s "
              f"{reserved:>6.2f}GB {pct:>4.0f}%  {verdict}")
        del model, opt, sched
        return out
    except RuntimeError as e:
        print(f"  skip_last={n_skip:<3} FAILED: {str(e)[:60]}")
        return dict(n_skip=n_skip, failed=str(e)[:200])
    finally:
        edge_model.checkpoint = _real_checkpoint
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
    ap.add_argument("--skips", type=int, nargs="+", default=[0, 2, 4, 6, 8, 12])
    ap.add_argument("--out", default="results/partial_checkpoint.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required."); return 2
    print(f"partial gradient checkpointing, d={args.d_model}, {args.num_layers}L, "
          f"seq {args.seq}, batch {args.batch}, NO offload")
    print(f"  {'config':<14} {'s/step':>7} {'tok/s':>12} {'VRAM':>8} {'%':>5}")

    results = [run(n, args) for n in args.skips]
    ok = [r for r in results if not r.get("failed")]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(dict(config=vars(args), results=results), fh, indent=2)

    if len(ok) >= 2:
        base = ok[0]
        # The probe's own sanity check: kept activations MUST cost memory. If
        # they do not, the intercept is not taking effect and any timing
        # conclusion below would be measuring nothing.
        mem_span = max(r["peak_reserved_gb"] for r in ok) - base["peak_reserved_gb"]
        print(f"\nmemory span across the sweep: {mem_span:+.2f} GB")
        if mem_span < 0.05:
            print("*** VOID: peak VRAM barely moved, so the intercept is probably "
                  "NOT skipping any checkpoints. Do not read the timings below.")
        else:
            print("  (memory rises with n_skip -> the intercept is taking effect)")
            best = min(ok, key=lambda r: r["sec_per_step"])
            gain = base["sec_per_step"] / best["sec_per_step"]
            print(f"\nbest: skip_last={best['n_skip']} at {best['sec_per_step']:.2f}s "
                  f"vs {base['sec_per_step']:.2f}s baseline = {gain:.3f}x "
                  f"({100*(gain-1):+.1f}%)")
            fitting = [r for r in ok if r["verdict"] == "FITS"]
            if fitting:
                bf = min(fitting, key=lambda r: r["sec_per_step"])
                print(f"best that FITS (<=85% of card): skip_last={bf['n_skip']}, "
                      f"{bf['sec_per_step']:.2f}s, {bf['pct_of_card']:.0f}% "
                      f"= {100*(base['sec_per_step']/bf['sec_per_step']-1):+.1f}%")
            print("\nCompare against micro-batch 2 (+7.1% at 478M), which needs NO "
                  "code and spends the SAME spare VRAM -- the two compete, they do "
                  "not add.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
