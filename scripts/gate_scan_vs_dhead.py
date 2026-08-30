"""Gate: does the chunk-parallel scan win at larger d_head, at 1B shapes?

Author: Rambarun Komaljeet

The question
------------
The DeltaNet scan walks tokens one at a time -- the only genuinely serial part
of the model. Two levers pull in opposite directions and neither has paid off
alone:

* **more heads** -> better occupancy (the Triton scan launches ``N = B*H``
  programs against this card's 20 SMs) but SMALLER d_head
* **fewer heads** -> larger d_head, which is the regime where the chunk-parallel
  form is predicted to win, but FEWER programs

docs/state.md's "RETEST AT SCALE" table says the chunk-parallel scan lost at
d_head 64 because "the parallel form replaces C cheap sequential steps with one
C x C interaction solve. Per-step cost grows with d_head^2, the solve does not
-- so the trade flips as d_head grows."

That prediction has never been tested at 1B shapes. This measures it directly
rather than modelling it: ``chunked_scan_forward`` already exists
(``hvn/core/deltanet_triton.py:461``), so no kernel needs writing to find out.

Method
------
Total work is held CONSTANT. d_model 2048 is split into (n_heads x d_head) so
``N*d`` is identical in every row -- only the SHAPE of the split changes. A row
that is faster is faster for a structural reason, not because it did less work.

Forward only, all three paths, so the comparison is like-for-like
(``chunked_scan_forward`` is a plain function; the other two are autograd
Functions whose backward differs in kind). Backward is 73% of a real step, so a
forward-only win is necessary but NOT sufficient -- flagged in the verdict.

Falsifiability check, asked before running
-------------------------------------------
*"If d_head did NOT matter, would this catch it?"* Yes: total work is fixed, so
a flat row-to-row time refutes the whole premise outright.

**Known-answer cell.** #95/#96 measured the chunked path 1.5-4x SLOWER than
sequential at d_head 64. That row is in this sweep. If it comes out FASTER here,
this harness disagrees with two committed measurements and its other rows cannot
be trusted -- the run is VOID rather than a discovery. Stated before the numbers
land, not after.

Usage::

    python scripts/gate_scan_vs_dhead.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.core.deltanet_triton import (                      # noqa: E402
    gated_delta_scan, megakernel_scan, chunked_scan_forward,
)

#: (n_heads, d_head) at d_model = 2048. N*d is constant across every row.
#: d_head stays a power of two >= 32 (Triton requires it; #159 calls 32 the floor).
SPLITS = [(64, 32), (32, 64), (16, 128), (8, 256)]
CURRENT = (32, 64)          # what the 1B config runs today


def timeit(fn, warmup=3, iters=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def run_split(n_heads, d_head, B, S, device, dtype, chunk):
    N = B * n_heads
    g = torch.Generator(device=device).manual_seed(0)

    def mk(*shape):
        return torch.randn(*shape, device=device, dtype=dtype, generator=g)

    q, k, v = mk(N, S, d_head), mk(N, S, d_head), mk(N, S, d_head)
    # alpha is a decay gate in (0,1]; beta a per-token scalar in (0,1).
    alpha = torch.rand(N, S, d_head, device=device, dtype=dtype,
                       generator=g) * 0.1 + 0.9
    beta = torch.rand(N, S, 1, device=device, dtype=dtype, generator=g)

    row = {"n_heads": n_heads, "d_head": d_head, "N": N,
           "elements": N * S * d_head}

    for name, fn in (
        ("sequential", lambda: gated_delta_scan(q, k, v, alpha, beta)),
        ("megakernel", lambda: megakernel_scan(q, k, v, alpha, beta)),
        ("chunked",    lambda: chunked_scan_forward(q, k, v, alpha, beta,
                                                    chunk=chunk)),
    ):
        try:
            with torch.no_grad():
                row[name] = timeit(fn) * 1e3          # ms
        except Exception as e:
            row[name] = None
            row[name + "_error"] = f"{type(e).__name__}: {e}"[:200]
            torch.cuda.empty_cache()
    return row


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq", type=int, default=4096, help="1B config runs 4096")
    p.add_argument("--d-model", type=int, default=2048)
    p.add_argument("--chunk", type=int, default=128)
    p.add_argument("--dtype", default="bf16", choices=("bf16", "fp32"))
    p.add_argument("--out", default="results/scan_dhead/gate.json")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md)")
    device = torch.device("cuda")
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float32
    print(f"[gate] {torch.cuda.get_device_name(0)}  "
          f"batch={args.batch} seq={args.seq} d_model={args.d_model} "
          f"dtype={args.dtype} chunk={args.chunk}")
    print(f"[gate] total scan elements held CONSTANT at "
          f"{args.batch * args.d_model * args.seq:,} per tensor\n")

    rows = []
    for n_heads, d_head in SPLITS:
        assert n_heads * d_head == args.d_model, "split must preserve d_model"
        r = run_split(n_heads, d_head, args.batch, args.seq, device, dtype,
                      args.chunk)
        rows.append(r)
        mark = "  <- current 1B config" if (n_heads, d_head) == CURRENT else ""
        def f(x):
            return f"{x:8.2f}" if isinstance(x, float) else "     n/a"
        print(f"[gate] heads {n_heads:>3}  d_head {d_head:>4}  N {r['N']:>3}   "
              f"seq {f(r['sequential'])} ms   mega {f(r['megakernel'])} ms   "
              f"chunked {f(r['chunked'])} ms{mark}")
        for key in ("sequential", "megakernel", "chunked"):
            if r.get(key + "_error"):
                print(f"[gate]        {key} failed: {r[key + '_error']}")

    # ---- verdict -----------------------------------------------------------
    print("\n[gate] ================== VERDICT ==================")
    print(f"[gate] {'heads':>6}{'d_head':>8}{'chunked/seq':>14}  faster?")
    ratios = {}
    for r in rows:
        if isinstance(r.get("chunked"), float) and isinstance(r.get("sequential"), float):
            ratio = r["chunked"] / r["sequential"]
            ratios[r["d_head"]] = ratio
            print(f"[gate] {r['n_heads']:>6}{r['d_head']:>8}{ratio:>14.2f}x  "
                  f"{'YES' if ratio < 1.0 else 'no'}")

    void = []
    # Known-answer cell: d_head 64 must reproduce the published loss.
    if 64 in ratios and ratios[64] < 1.0:
        void.append(f"d_head 64 came out {ratios[64]:.2f}x (chunked FASTER), "
                    f"contradicting #95/#96 which measured it 1.5-4x slower")
        print(f"\n[gate] *** KNOWN-ANSWER CELL FAILED: d_head 64 should be "
              f"SLOWER under the chunked path (#95/#96 measured 1.5-4x). "
              f"Got {ratios[64]:.2f}x. This harness disagrees with two "
              f"committed measurements, so its other rows are not evidence.")
    elif 64 in ratios:
        print(f"\n[gate] known-answer cell OK: d_head 64 chunked is "
              f"{ratios[64]:.2f}x sequential (slower, as #95/#96 measured)")

    # Is the premise even alive? Fixed work, so flat => d_head does not matter.
    seqs = [r["sequential"] for r in rows if isinstance(r.get("sequential"), float)]
    if seqs and max(seqs) / min(seqs) < 1.10:
        print(f"[gate] NOTE: sequential time is flat across d_head "
              f"({min(seqs):.2f}-{max(seqs):.2f} ms, <10% spread). The head "
              f"split does not move the scan at these shapes at all.")

    if void:
        verdict = "VOID: " + "; ".join(void)
    else:
        wins = {d: r for d, r in ratios.items() if r < 1.0}
        if wins:
            best = min(wins, key=lambda d: wins[d])
            verdict = (f"chunk-parallel wins at d_head >= {best} "
                       f"({wins[best]:.2f}x sequential, forward only)")
        else:
            verdict = ("chunk-parallel loses at EVERY d_head tested up to "
                       f"{max(ratios) if ratios else 'n/a'} -- the predicted "
                       "crossover is not in this range; do not build the kernel")
    print(f"\n[gate] {verdict}")
    print("[gate] NOTE: forward only. Backward is ~73% of a real step, so a "
          "forward win here is necessary but not sufficient -- it would still "
          "need a backward before any step-time claim.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"question": "scan time vs d_head at 1B shapes",
                   "config": vars(args), "rows": rows,
                   "chunked_over_sequential": ratios,
                   "verdict": verdict, "void_reasons": void}, fh, indent=2)
    print(f"[gate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
