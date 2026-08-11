"""Is chunk=32 wrong for d_head=64? Sweep the scan's chunk size (Issue #154)

Author: Rambarun Komaljeet

`DeltaNetBlock.train_chunk` defaults to **32**, and the docstring says why: *"Kept
modest (32) for numerical stability."* It was chosen for correctness and never
swept for speed. #95 and #96 compared *chunked vs sequential* at that default;
neither asked whether 32 is the right number.

**Why the optimum should move with head width.** The chunk-parallel form replaces
`C` cheap sequential steps with one `C x C` interaction matrix plus a triangular
solve. Cost of the table grows with `C^2`; benefit grows with `C`. Per-step
sequential cost grows with `d_head^2`. So at small `d_head` the steps being
skipped are nearly free and the optimum `C` should be **smaller**, not 32.

A 4x4 table is trivial and still skips 3 of every 4 sequential steps. That
configuration has never been measured.

**Scope, stated honestly:** `train_chunk` is unreachable while the Triton scan is
on (`deltanet.py:689` takes the fused path first), so this sweeps the **eager**
scan. It measures the SHAPE of the cost tradeoff, not production speed. If the
curve says small chunks win decisively, that is an argument for building the
chunk into the kernel -- not a speedup anyone can use today. Recorded so the
result is not over-read.

**Exactness check, not just timing.** The chunked path claims to compute the
identical result. Every chunk size is compared against the sequential scan on the
same input; a chunk that is fast because it is *wrong* must not be reported as
fast. This repo has seen that exact failure -- the eager chunked scan is
documented as "silently wrong on decayed channels at large chunks / hard decay".

Usage::

    python scripts/sweep_train_chunk.py --chunks 0 4 8 16 32 64 128
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ab_prodigy import require_cuda                        # noqa: E402
from hvn.core.deltanet import DeltaNetBlock                        # noqa: E402


def make_block(d_model, n_heads, chunk, device, seed=0):
    torch.manual_seed(seed)
    blk = DeltaNetBlock(d_model=d_model, n_heads=n_heads, dropout=0.0,
                        train_chunk=chunk).to(device)
    # The whole point is the eager path: the Triton scan bypasses train_chunk.
    blk.use_triton_scan = False
    blk.use_megakernel_scan = False
    return blk


def time_chunk(d_model, n_heads, chunk, args, device):
    blk = make_block(d_model, n_heads, chunk, device)
    g = torch.Generator(device="cpu").manual_seed(1)
    x = torch.randn(args.batch_size, args.seq_len, d_model, generator=g).to(device)

    def one():
        blk.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = blk(x)
        out.float().pow(2).mean().backward()

    for _ in range(args.warmup):
        one()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.iters):
        one()
    torch.cuda.synchronize()
    el = time.perf_counter() - t0
    toks = args.iters * args.batch_size * args.seq_len
    del blk
    torch.cuda.empty_cache()
    return {"seconds": el, "tokens_per_sec": toks / el}


@torch.no_grad()
def exactness(d_model, n_heads, chunk, args, device):
    """Max |chunked - sequential| on identical weights and input.

    A chunk that is fast because it is WRONG must not be reported as fast -- the
    eager chunked scan is documented as silently wrong at large chunks / hard
    decay, so this is checked rather than trusted.
    """
    ref = make_block(d_model, n_heads, 0, device, seed=7)
    tst = make_block(d_model, n_heads, chunk, device, seed=7)
    tst.load_state_dict(ref.state_dict())
    ref.eval(); tst.eval()
    g = torch.Generator(device="cpu").manual_seed(3)
    x = torch.randn(2, args.seq_len, d_model, generator=g).to(device)
    a, b = ref(x).float(), tst(x).float()
    d = (a - b).abs().max().item()
    scale = a.abs().max().item() or 1.0
    del ref, tst
    torch.cuda.empty_cache()
    return d, d / scale


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=int, nargs="+",
                   default=[0, 4, 8, 16, 32, 64, 128])
    p.add_argument("--d-heads", type=int, nargs="+", default=[64, 128],
                   help="64 = today's 83M; 128 = what a wider model would have")
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--iters", type=int, default=10)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--out", default="results/train_chunk/sweep.json")
    args = p.parse_args()

    device = require_cuda()
    torch.backends.cuda.matmul.allow_tf32 = True
    print(f"[chunk] d_model={args.d_model}, batch {args.batch_size} x seq {args.seq_len}, "
          f"EAGER scan (triton off -- train_chunk is unreachable otherwise)")

    results = {}
    for dh in args.d_heads:
        if args.d_model % dh:
            continue
        n_heads = args.d_model // dh
        print(f"\n[chunk] === d_head={dh} ({n_heads} heads) ===")
        samples = {c: [] for c in args.chunks}
        # Interleaved rounds so thermal drift hits every chunk size equally.
        for r in range(args.rounds):
            for c in args.chunks:
                s = time_chunk(args.d_model, n_heads, c, args, device)
                samples[c].append(s["tokens_per_sec"])
                print(f"[chunk]   round {r+1}  chunk {c:>4}  "
                      f"{s['tokens_per_sec']:9.1f} tok/s")
        med = {c: statistics.median(v) for c, v in samples.items()}
        base = med[0] if 0 in med else max(med.values())
        rows = {}
        for c in args.chunks:
            if c == 0:
                err = (0.0, 0.0)
            else:
                err = exactness(args.d_model, n_heads, c, args, device)
            rows[c] = {"tok_s": med[c], "vs_sequential": (med[c] - base) / base * 100,
                       "max_abs_err": err[0], "rel_err": err[1]}
        results[f"d_head_{dh}"] = rows

        print(f"\n[chunk] {'chunk':>7}{'tok/s':>11}{'vs seq':>10}{'rel err':>12}")
        for c in args.chunks:
            r = rows[c]
            tag = "  <-- default" if c == 32 else ""
            print(f"[chunk] {('seq' if c == 0 else c):>7}{r['tok_s']:>11.1f}"
                  f"{r['vs_sequential']:>9.1f}%{r['rel_err']:>12.2e}{tag}")
        best = max((c for c in args.chunks), key=lambda c: rows[c]["tok_s"])
        print(f"[chunk] fastest at d_head={dh}: "
              f"{'sequential' if best == 0 else f'chunk {best}'} "
              f"({rows[best]['vs_sequential']:+.1f}% vs sequential)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 154, "config": vars(args), "results": results},
                  fh, indent=2)
    print(f"\n[chunk] wrote {args.out}")
    print("[chunk] NOTE: eager path only. A win here argues for building the "
          "chunk into the Triton kernel; it is not a speedup available today.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
