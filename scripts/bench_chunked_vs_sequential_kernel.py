"""Is the EXISTING chunked Triton forward faster than the sequential one? (#161)

Author: Rambarun Komaljeet

`deltanet_triton.py` contains **two** forwards:

``gated_delta_forward``    what production runs -- `for t in range(S)`, i.e.
                           4,096 sequential steps per sequence at seq 4096.
``chunked_scan_forward``   already written (#95), forward-only: 32 chunk-steps
                           of 128 tokens each, with a fused Triton log-Gram and
                           a triangular solve per chunk.

The second is a ~128x reduction in *serial depth* and it is sitting unused. The
reason appears to be that it is forward-only -- training needs a backward, and
`_GatedDeltaScan` only wraps the sequential kernel.

**But nobody has measured whether it is actually faster at this shape.** #95/#96
compared chunked forms at d_model 512 / seq 512 and they lost; production is now
seq **4096**, where there are 8x more sequential steps to eliminate and the
trade may have flipped. Writing a backward is days of work in the file this repo
has twice recorded going silently wrong, so it is worth ten minutes to check the
premise first.

Correctness is checked against the sequential kernel at every chunk size,
because a chunked scan that is fast and wrong is worthless -- and `_scan_chunked`
is documented as silently wrong on decayed channels at large chunks.

Usage::

    python scripts/bench_chunked_vs_sequential_kernel.py --chunks 64 128 256 512
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.core import deltanet_triton as dt          # noqa: E402


def make_inputs(N, S, D, device, seed=0):
    g = torch.Generator(device="cpu").manual_seed(seed)
    mk = lambda *sh: torch.randn(*sh, generator=g).to(device).float()
    q, k, v = mk(N, S, D), mk(N, S, D), mk(N, S, D)
    # alpha ~ sigmoid(+3) = 0.95: the trained decay regime (alpha_proj bias is 3.0)
    alpha = torch.sigmoid(mk(N, S, D) + 3.0)
    beta = torch.sigmoid(mk(N, S, 1))
    return q, k, v, alpha, beta


def timeit(fn, iters, warmup=2):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out = fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters, out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--chunks", type=int, nargs="+", default=[64, 128, 256, 512])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=14)
    p.add_argument("--d-head", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--out", default="results/chunked_kernel/bench.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    N, D, S = args.batch * args.heads, args.d_head, args.seq_len
    print(f"N = {args.batch} x {args.heads} heads = {N} | d_head {D} | seq {S}")
    print(f"sequential kernel does {S} serial steps; chunked does S/chunk\n")

    q, k, v, alpha, beta = make_inputs(N, S, D, device)

    seq_ms, seq_out = timeit(
        lambda: dt.gated_delta_forward(q, k, v, alpha, beta), args.iters)
    print(f"{'variant':<22}{'serial steps':>14}{'ms':>10}{'speedup':>10}{'rel err':>11}")
    print(f"{'sequential (prod)':<22}{S:>14}{seq_ms*1e3:>10.2f}{1.0:>9.2f}x{'-':>11}")

    scale = seq_out.abs().max().item() or 1.0
    results = {"sequential": {"ms": seq_ms * 1e3, "serial_steps": S}}
    for c in args.chunks:
        if S % c:
            print(f"chunk {c}: skipped (seq {S} not divisible)")
            continue
        try:
            ms, out = timeit(
                lambda c=c: dt.chunked_scan_forward(q, k, v, alpha, beta, chunk=c),
                args.iters)
        except Exception as e:
            print(f"{'chunked ' + str(c):<22}{'':>14}{'FAILED':>10}  {type(e).__name__}: {e}"[:110])
            results[f"chunk_{c}"] = {"error": f"{type(e).__name__}: {e}"[:200]}
            continue
        err = (out - seq_out).abs().max().item() / scale
        results[f"chunk_{c}"] = {"ms": ms * 1e3, "serial_steps": S // c,
                                 "rel_err": err, "speedup": seq_ms / ms}
        flag = "" if err < 2e-3 else "  <-- WRONG"
        print(f"{'chunked ' + str(c):<22}{S//c:>14}{ms*1e3:>10.2f}"
              f"{seq_ms/ms:>9.2f}x{err:>11.1e}{flag}")

    ok = {k_: v_ for k_, v_ in results.items()
          if k_ != "sequential" and "ms" in v_ and v_.get("rel_err", 1) < 2e-3}
    if ok:
        best = min(ok, key=lambda k_: ok[k_]["ms"])
        sp = ok[best]["speedup"]
        verdict = (f"{best} is {sp:.2f}x the production sequential forward "
                   f"({(sp-1)*100:+.0f}%) at rel err {ok[best]['rel_err']:.1e}. "
                   f"A matching BACKWARD is what stands between this and using it."
                   if sp > 1.1 else
                   f"no worthwhile gain (best {best} at {sp:.2f}x) -- the sequential "
                   f"kernel is already competitive at this shape, and writing a "
                   f"backward for the chunked path would not pay.")
    else:
        verdict = ("every chunked variant was wrong or failed -- nothing to build on")
    print(f"\nVERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 161, "config": vars(args), "results": results,
                   "verdict": verdict}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
