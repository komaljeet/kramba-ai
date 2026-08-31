"""What actually limits the DeltaNet scan, and which levers move it?

Author: Rambarun Komaljeet

Why this exists
---------------
`scripts/roofline_1b_step.py` measured the scan at **1.3% of this card's compute
ceiling and 7.6% of its bandwidth ceiling** -- near NEITHER. It is not short of
math and not short of memory, so the remaining explanation is parallelism. This
splits that into its two possible causes, which need opposite fixes:

* **too few programs** -- `N = batch x n_heads` against 20 SMs. Fixable by
  raising N (bigger batch, more heads) or by splitting the sequence.
* **the dependency chain** -- 4,096 steps that must run in order, no matter how
  many SMs are free. Not fixable by occupancy at all; needs the recurrence
  restructured.

They look identical from a wall-clock number and are told apart by how the cost
responds to N and to S. That is what tests A and B do.

Test C exists because of a mistake in the earlier pass
--------------------------------------------------------
`gate_scan_vs_dhead.py` concluded "chunk-parallel loses, do not build" -- from a
**single chunk size (128)**. Chunk size is the parameter that trades sequential
carries against per-chunk parallel work, so testing one value and declaring the
approach dead is exactly the premature discard this repo keeps writing up. The
verdict stands only if it survives a sweep.

Falsifiability check, asked before running
--------------------------------------------
Each test makes a directional prediction that the data can refute:

* **A**: if occupancy binds, head-tokens/s RISES with N and plateaus once the SMs
  fill. A FLAT curve from N=8 upward refutes occupancy outright -- the card is
  already as full as it will get, and the chain is the whole story.
* **B**: if the chain binds, time is LINEAR in S -- constant cost per sequential
  step. Sublinear scaling means some parallelism is already being extracted and
  the chain is not fully serial.
* **C**: if chunking has any regime that pays, some chunk size beats the
  sequential kernel. If every size loses, the earlier verdict is confirmed on
  evidence rather than on one sample.

**Known-answer cell:** test C at chunk 128 must reproduce
`gate_scan_vs_dhead.py`'s measured ~2x loss at d_head 64. If it does not, the two
harnesses disagree and neither is evidence.

Usage::

    python scripts/probe_scan_levers.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.core.deltanet_triton import (                          # noqa: E402
    gated_delta_scan, megakernel_scan, chunked_scan_forward,
)

SMS = 20  # RTX 4050 Laptop


def mk_inputs(N, S, d, device):
    g = torch.Generator(device=device).manual_seed(0)
    f = lambda *s: torch.randn(*s, device=device, dtype=torch.bfloat16, generator=g)
    q, k, v = f(N, S, d), f(N, S, d), f(N, S, d)
    a = torch.rand(N, S, d, device=device, dtype=torch.bfloat16, generator=g) * .1 + .9
    b = torch.rand(N, S, 1, device=device, dtype=torch.bfloat16, generator=g)
    return q, k, v, a, b


def timeit(fn, warmup=3, iters=8):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def run(fn, *a):
    with torch.no_grad():
        return timeit(lambda: fn(*a))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-head", type=int, default=32)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--n-base", type=int, default=64, help="N at batch 1, 64 heads")
    p.add_argument("--out", default="results/roofline/scan_levers.json")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md)")
    dev = torch.device("cuda")
    d = args.d_head
    print(f"[probe] {torch.cuda.get_device_name(0)}  d_head={d}  {SMS} SMs\n")
    out = {"d_head": d, "sms": SMS}

    # ---- A. occupancy: does more parallel work help? ----------------------
    print("[probe] === A. occupancy sweep (S fixed, N varies) ===")
    print("[probe] prediction: if occupancy binds, head-tokens/s RISES then plateaus")
    print(f"[probe] {'N':>6}{'progs/SM':>10}{'ms':>9}{'Mhead-tok/s':>14}{'vs N=8':>9}")
    rowsA, base = [], None
    for N in (8, 16, 20, 32, 64, 128, 256):
        try:
            q, k, v, a, b = mk_inputs(N, args.seq, d, dev)
            t = run(gated_delta_scan, q, k, v, a, b)
            thru = N * args.seq / t / 1e6
            base = base or thru
            print(f"[probe] {N:>6}{N/SMS:>10.1f}{t*1e3:>9.2f}{thru:>14.1f}"
                  f"{thru/base:>9.2f}x")
            rowsA.append({"N": N, "ms": t * 1e3, "mhead_tok_s": thru})
            del q, k, v, a, b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"[probe] {N:>6}       OOM"); torch.cuda.empty_cache(); break
    out["A_occupancy"] = rowsA
    if len(rowsA) > 1:
        gain = rowsA[-1]["mhead_tok_s"] / rowsA[0]["mhead_tok_s"]
        print(f"[probe] -> throughput scaled {gain:.2f}x from N=8 to "
              f"N={rowsA[-1]['N']}")
        out["A_gain"] = gain

    # ---- B. is the chain fully serial? ------------------------------------
    print("\n[probe] === B. sequence scaling (N fixed, S varies) ===")
    print("[probe] prediction: a fully serial chain gives CONSTANT ns per step")
    print(f"[probe] {'S':>7}{'ms':>9}{'ns/step':>10}{'vs S=512':>10}")
    rowsB, base_ns = [], None
    for S in (512, 1024, 2048, 4096, 8192):
        try:
            q, k, v, a, b = mk_inputs(args.n_base, S, d, dev)
            t = run(gated_delta_scan, q, k, v, a, b)
            ns = t / S * 1e9
            base_ns = base_ns or ns
            print(f"[probe] {S:>7}{t*1e3:>9.2f}{ns:>10.1f}{ns/base_ns:>10.2f}x")
            rowsB.append({"S": S, "ms": t * 1e3, "ns_per_step": ns})
            del q, k, v, a, b
            torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            print(f"[probe] {S:>7}       OOM"); torch.cuda.empty_cache(); break
    out["B_seq_scaling"] = rowsB

    # ---- C. chunk sweep: the lever discarded on one sample -----------------
    print("\n[probe] === C. chunk-size sweep (the earlier verdict used ONE size) ===")
    q, k, v, a, b = mk_inputs(args.n_base, args.seq, d, dev)
    seq_ms = run(gated_delta_scan, q, k, v, a, b) * 1e3
    mega_ms = run(megakernel_scan, q, k, v, a, b) * 1e3
    print(f"[probe] sequential baseline {seq_ms:.2f} ms   megakernel {mega_ms:.2f} ms")
    print(f"[probe] {'chunk':>7}{'ms':>9}{'vs sequential':>16}  faster?")
    rowsC, best = [], None
    for chunk in (16, 32, 64, 128, 256, 512):
        try:
            t = run(chunked_scan_forward, q, k, v, a, b, chunk) * 1e3
            r = t / seq_ms
            print(f"[probe] {chunk:>7}{t:>9.2f}{r:>15.2f}x  "
                  f"{'YES' if r < 1 else 'no'}")
            rowsC.append({"chunk": chunk, "ms": t, "ratio_vs_sequential": r})
            if best is None or r < best[1]:
                best = (chunk, r)
        except Exception as e:
            print(f"[probe] {chunk:>7}    FAILED  {type(e).__name__}: {str(e)[:70]}")
            torch.cuda.empty_cache()
    out["C_chunk_sweep"] = rowsC
    out["C_best"] = {"chunk": best[0], "ratio": best[1]} if best else None

    # ---- verdict ----------------------------------------------------------
    print("\n[probe] ================== READING ==================")
    if len(rowsA) > 1:
        g = rowsA[-1]["mhead_tok_s"] / rowsA[0]["mhead_tok_s"]
        if g > 1.5:
            print(f"[probe] A: OCCUPANCY BINDS -- throughput scaled {g:.2f}x with "
                  f"more programs. Bigger batch / more heads is a real lever, and "
                  f"the plateau point says where it stops paying.")
        else:
            print(f"[probe] A: occupancy does NOT bind -- {g:.2f}x from 8 to "
                  f"{rowsA[-1]['N']} programs. The card is already as full as it "
                  f"gets; adding parallel work will not help.")
    if len(rowsB) > 1:
        spread = max(r["ns_per_step"] for r in rowsB) / min(r["ns_per_step"] for r in rowsB)
        if spread < 1.3:
            print(f"[probe] B: the chain IS fully serial -- ns/step constant to "
                  f"within {spread:.2f}x across a 16x sequence range. Cost is set "
                  f"by the number of sequential steps, nothing else.")
        else:
            print(f"[probe] B: ns/step varies {spread:.2f}x -- not purely serial; "
                  f"some parallelism is already being extracted at longer S.")
    if best:
        if best[1] < 1.0:
            print(f"[probe] C: *** chunk={best[0]} BEATS the sequential kernel "
                  f"({best[1]:.2f}x). The earlier one-sample verdict was WRONG.")
        else:
            print(f"[probe] C: every chunk size loses; best is {best[0]} at "
                  f"{best[1]:.2f}x. The verdict now rests on a sweep, not one "
                  f"sample -- chunk-parallel is genuinely dead at this d_head.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
