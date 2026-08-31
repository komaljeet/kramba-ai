"""Roofline: WHY is the 1B step at ~44% of this card's demonstrated FLOP rate?

Author: Rambarun Komaljeet

The gap this exists to explain
------------------------------
The 1B step is **98.7% GPU-busy** (2026-08-27) and yet achieves only ~10 TFLOP/s
of real work, against **23 TFLOP/s measured on a plain bf16 GEMM** on this same
card (`scripts/bench_logmul_ceiling.py`). Busy is not the same as productive: the
GPU is occupied ~100% of the time while delivering under half the arithmetic it
demonstrably can. That ~2.3x is the largest unexplained quantity left in the
training loop, and nobody has attributed it.

Three ways to be slow, and they need DIFFERENT fixes
-----------------------------------------------------
A kernel can be near the machine's FLOP ceiling, near its BANDWIDTH ceiling, or
near **neither**:

* **compute-bound**  -> at peak FLOP/s. Nothing to reclaim; only fewer FLOPs help.
* **memory-bound**   -> at peak GB/s. Fusion helps: keep intermediates in
  registers/SRAM so they never round-trip to DRAM.
* **latency-bound**  -> at neither ceiling. A serialized dependency chain, or too
  few programs to fill the SMs, leaves the machine idle *within* a kernel while
  the kernel itself is "running". Fusion does nothing here; more parallelism or a
  restructured recurrence is the only lever.

**This distinction is the whole point of the script.** The DeltaNet scan is a
recurrence over 4,096 tokens, so latency-bound is a live hypothesis, not a
formality -- and it is the one under which the fusion work would be wasted
effort. A test that assumed "memory-bound" and only measured bandwidth would
confirm that assumption no matter which of the three was true.

Falsifiability check, asked before running
--------------------------------------------
*"If there were NO headroom, would this catch it?"* Yes, and it is the expected
outcome for at least one category: if the dominant kernels come back at or near
the measured GEMM ceiling, they are compute-bound, the ~2.3x is an artifact of
averaging cheap kernels in with expensive ones, and there is nothing to win.

**Both ceilings are MEASURED on this card, not taken from a spec sheet.** A
roofline drawn against a vendor's theoretical peak grades every kernel against a
number no code on the machine ever reaches, which makes everything look
memory-bound. `peak_flops` comes from the same cuBLAS GEMM the training step
actually issues; `peak_bandwidth` from a large streaming copy.

Method note, stated because it bounds the conclusion
------------------------------------------------------
Bytes moved are computed ANALYTICALLY from tensor shapes (what the kernel must
read and write at minimum), not measured from hardware counters -- Nsight Compute
is not installed on this box. So a kernel that re-reads a tile many times shows a
LOWER apparent bandwidth than it truly achieves, which biases the classification
toward "latency-bound". Where that matters the verdict says so rather than
hiding it. The FLOP counts are exact.

Usage::

    python scripts/roofline_1b_step.py
    python scripts/roofline_1b_step.py --seq 4096 --n-heads 64
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.core.deltanet_triton import gated_delta_scan          # noqa: E402

BF16 = 2  # bytes


def timeit(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


# --------------------------------------------------------------------------
# the two ceilings -- both measured on this card
# --------------------------------------------------------------------------

def measure_peak_flops(device) -> float:
    """Best sustained bf16 GEMM rate. The same cuBLAS path training issues."""
    best = 0.0
    for m, k, n in ((4096, 4096, 4096), (4096, 2048, 2048), (8192, 2048, 2048)):
        a = torch.randn(m, k, device=device, dtype=torch.bfloat16)
        b = torch.randn(k, n, device=device, dtype=torch.bfloat16)
        t = timeit(lambda: a @ b)
        best = max(best, 2 * m * k * n / t)
    return best


def measure_peak_bandwidth(device) -> float:
    """Best sustained DRAM rate, from a large streaming copy (1 read + 1 write)."""
    best = 0.0
    for numel in (64 << 20, 128 << 20):
        src = torch.empty(numel, device=device, dtype=torch.bfloat16)
        dst = torch.empty_like(src)
        t = timeit(lambda: dst.copy_(src), warmup=3, iters=10)
        best = max(best, 2 * numel * BF16 / t)
    return best


# --------------------------------------------------------------------------

def classify(achieved_flops, achieved_bw, peak_flops, peak_bw,
             flop_frac=0.50, bw_frac=0.50):
    """Name the binding resource, or say that neither binds."""
    f = achieved_flops / peak_flops
    b = achieved_bw / peak_bw
    if f >= flop_frac:
        return "COMPUTE-bound", f, b
    if b >= bw_frac:
        return "MEMORY-bound", f, b
    return "LATENCY-bound", f, b


def bench_scan(N, S, d, device):
    """The DeltaNet recurrence, in isolation, at real 1B shapes."""
    g = torch.Generator(device=device).manual_seed(0)
    mk = lambda *s: torch.randn(*s, device=device, dtype=torch.bfloat16, generator=g)
    q, k, v = mk(N, S, d), mk(N, S, d), mk(N, S, d)
    a = torch.rand(N, S, d, device=device, dtype=torch.bfloat16, generator=g) * .1 + .9
    b = torch.rand(N, S, 1, device=device, dtype=torch.bfloat16, generator=g)
    with torch.no_grad():
        t = timeit(lambda: gated_delta_scan(q, k, v, a, b), warmup=3, iters=10)
    # FLOPs: per token per head the state is [d,d]; the gated delta update
    # (S*a, outer-product correction, and the read o = S q) is ~4 d^2 MACs.
    flops = N * S * 4 * d * d * 2
    # Bytes: q,k,v,alpha in + o out at [N,S,d], beta at [N,S,1]. The state stays
    # in registers/SRAM across the sequence, so it is not DRAM traffic.
    byts = (4 * N * S * d + N * S + N * S * d) * BF16
    return t, flops, byts


def bench_gemm(M, K, N_, device):
    a = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    b = torch.randn(K, N_, device=device, dtype=torch.bfloat16)
    t = timeit(lambda: a @ b)
    return t, 2 * M * K * N_, (M * K + K * N_ + M * N_) * BF16


def bench_elementwise(numel, device):
    """SiLU-gate shaped: two reads, one write, a handful of FLOPs."""
    a = torch.randn(numel, device=device, dtype=torch.bfloat16)
    b = torch.randn(numel, device=device, dtype=torch.bfloat16)
    t = timeit(lambda: torch.nn.functional.silu(a) * b, warmup=3, iters=10)
    return t, 5 * numel, 3 * numel * BF16


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=2048)
    p.add_argument("--n-heads", type=int, default=64)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--out", default="results/roofline/roofline_1b.json")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md)")
    dev = torch.device("cuda")
    d_head = args.d_model // args.n_heads
    N = args.batch * args.n_heads

    print(f"[roofline] {torch.cuda.get_device_name(0)}")
    print(f"[roofline] d_model {args.d_model}  n_heads {args.n_heads} "
          f"(d_head {d_head})  seq {args.seq}  batch {args.batch}\n")

    peak_f = measure_peak_flops(dev)
    peak_b = measure_peak_bandwidth(dev)
    balance = peak_f / peak_b
    print(f"[roofline] MEASURED ceilings on this card (not spec-sheet):")
    print(f"[roofline]   peak compute   {peak_f/1e12:8.2f} TFLOP/s  (bf16 GEMM)")
    print(f"[roofline]   peak bandwidth {peak_b/1e9:8.1f} GB/s     (streaming copy)")
    print(f"[roofline]   machine balance {balance:7.1f} FLOP/byte -- a kernel below")
    print(f"[roofline]     this arithmetic intensity CANNOT be compute-bound\n")

    ops = []
    t, fl, by = bench_scan(N, args.seq, d_head, dev)
    ops.append(("DeltaNet scan (fwd)", t, fl, by))
    t, fl, by = bench_gemm(args.batch * args.seq, args.d_model, args.d_model, dev)
    ops.append(("projection GEMM", t, fl, by))
    t, fl, by = bench_elementwise(args.batch * args.seq * args.d_model, dev)
    ops.append(("elementwise (SiLU gate)", t, fl, by))

    print(f"[roofline] {'op':<26}{'ms':>8}{'TFLOP/s':>10}{'GB/s':>9}"
          f"{'FLOP/byte':>11}{'% peak F':>10}{'% peak BW':>11}  verdict")
    rows = []
    for name, t, fl, by in ops:
        af, ab = fl / t, by / t
        verdict, ff, bf_ = classify(af, ab, peak_f, peak_b)
        ai = fl / by
        print(f"[roofline] {name:<26}{t*1e3:>8.2f}{af/1e12:>10.2f}{ab/1e9:>9.1f}"
              f"{ai:>11.1f}{ff*100:>9.1f}%{bf_*100:>10.1f}%  {verdict}")
        rows.append({"op": name, "ms": t * 1e3, "tflops": af / 1e12,
                     "gb_s": ab / 1e9, "flop_per_byte": ai,
                     "pct_peak_flops": ff * 100, "pct_peak_bw": bf_ * 100,
                     "verdict": verdict})

    print(f"\n[roofline] ================== READING ==================")
    scan = rows[0]
    if scan["verdict"] == "LATENCY-bound":
        print("[roofline] The scan is at NEITHER ceiling. It is not short of "
              "bandwidth and not short of FLOPs -- it is short of PARALLELISM: a "
              "4,096-step dependency chain, and only "
              f"{N} programs against this card's 20 SMs.")
        print("[roofline] => FUSION WOULD NOT HELP. The lever is restructuring "
              "the recurrence or raising occupancy, not avoiding DRAM traffic.")
        print("[roofline]    NOTE: bytes are analytic minimums, so a kernel that "
              "re-reads tiles shows lower apparent bandwidth than it achieves. "
              "This verdict is safe only because the FLOP fraction is low TOO -- "
              "both ceilings are far away, and no re-read accounting changes that.")
    elif scan["verdict"] == "MEMORY-bound":
        print("[roofline] The scan is at the bandwidth ceiling -> fusion is the "
              "right lever: keep intermediates off DRAM.")
    else:
        print("[roofline] The scan is at the compute ceiling -> there is no "
              "headroom here; only fewer FLOPs help.")

    print(f"\n[roofline] The ~2.3x whole-step gap is NOT one number to reclaim: "
          f"different kernels are limited by different resources, and any single "
          f"fix addresses only its own slice.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"question": "what binds the 1B step",
                   "config": vars(args),
                   "peak_tflops_measured": peak_f / 1e12,
                   "peak_bandwidth_gb_s_measured": peak_b / 1e9,
                   "machine_balance_flop_per_byte": balance,
                   "ops": rows}, fh, indent=2)
    print(f"[roofline] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
