"""Sweep the Triton scan's BV block size (#160).

Author: Rambarun Komaljeet

`gated_delta_forward(..., BV: int = 16)` splits the value dimension into blocks
and launches `grid = (N, D // BV)`, where `N = batch x heads`. So BV directly
sets how many programs run in parallel:

    production today: 1 x 14 heads x (64 // 16) = 56 programs on 20 SMs

BV=16 is a **hardcoded default nobody has tuned**. Smaller BV means more
programs each doing less work; larger means fewer doing more. There is an
optimum and it has never been looked for.

**Why this lever is unusually clean:** it is a kernel launch parameter, not a
model change. No shapes move, no weights change, the maths is identical -- so
unlike head count it needs no quality A/B and it can be applied to an existing
checkpoint mid-run.

Also worth measuring here: `deltanet_triton.py:203` falls back to `BV = 16 if
D % 16 == 0 else 1`. At d_head 32 (the 28-head config) that still gives 2
blocks, so part of the 20% attributed to head count today may in fact be this
launch-grid effect rather than the heads.

Correctness is checked against the eager sequential scan for every BV, because a
block size that is fast but wrong is worthless -- and this file has a documented
history of being silently wrong.

Usage::

    python scripts/sweep_bv.py --bvs 4 8 16 32
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
    alpha = torch.sigmoid(mk(N, S, D) + 3.0)         # ~0.95, the trained regime
    beta = torch.sigmoid(mk(N, S, 1))
    return q, k, v, alpha, beta


def reference(q, k, v, alpha, beta):
    """Eager sequential scan -- the ground truth every BV must reproduce."""
    N, S, D = q.shape
    state = torch.zeros(N, D, D, device=q.device, dtype=torch.float32)
    out = torch.zeros_like(v)
    for t in range(S):
        state = state * alpha[:, t].unsqueeze(1)
        Sk = torch.einsum('nvk,nk->nv', state, k[:, t])
        u = beta[:, t] * (v[:, t] - Sk)
        state = state + u.unsqueeze(2) * k[:, t].unsqueeze(1)
        out[:, t] = torch.einsum('nvk,nk->nv', state, q[:, t])
    return out


def bench_bv(bv, N, S, D, device, iters=5, warmup=2):
    q, k, v, alpha, beta = make_inputs(N, S, D, device)
    for _ in range(warmup):
        dt.gated_delta_forward(q, k, v, alpha, beta, BV=bv)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        o = dt.gated_delta_forward(q, k, v, alpha, beta, BV=bv)
    torch.cuda.synchronize()
    el = (time.perf_counter() - t0) / iters
    return el, o


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--bvs", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64])
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--heads", type=int, default=14)
    p.add_argument("--d-head", type=int, default=64)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--check-seq", type=int, default=256,
                   help="shorter sequence for the correctness check -- the eager "
                        "reference is O(S) python and 4096 would take minutes")
    p.add_argument("--out", default="results/bv_sweep/sweep.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    N = args.batch * args.heads
    D = args.d_head
    sms = torch.cuda.get_device_properties(0).multi_processor_count
    print(f"N = {args.batch} batch x {args.heads} heads = {N} | d_head {D} | "
          f"seq {args.seq_len} | {sms} SMs")
    print(f"production default is BV=16 -> {N} x {D//16} = {N*(D//16)} programs\n")

    # --- correctness first: a fast-but-wrong BV is worthless ---
    print("correctness vs the eager sequential scan "
          f"(seq {args.check_seq}):")
    qc, kc, vc, ac, bc = make_inputs(N, args.check_seq, D, device, seed=7)
    ref = reference(qc, kc, vc, ac, bc)
    ok = {}
    for bv in args.bvs:
        if D % bv:
            ok[bv] = None
            print(f"  BV={bv:<3} skipped (d_head {D} not divisible)")
            continue
        o = dt.gated_delta_forward(qc, kc, vc, ac, bc, BV=bv)
        err = (o - ref).abs().max().item() / (ref.abs().max().item() or 1)
        ok[bv] = err
        print(f"  BV={bv:<3} rel err {err:.2e}  {'OK' if err < 2e-3 else 'WRONG'}")

    print(f"\n{'BV':>4}{'programs':>10}{'ms':>10}{'vs BV=16':>10}")
    res = {}
    for bv in args.bvs:
        if ok.get(bv) is None:
            continue
        el, _ = bench_bv(bv, N, args.seq_len, D, device, iters=args.iters)
        res[bv] = {"ms": el * 1e3, "programs": N * (D // bv), "rel_err": ok[bv]}
        print(f"{bv:>4}{N*(D//bv):>10}{el*1e3:>10.2f}", end="")
        print(f"{'':>10}" if bv == 16 else "")
    if 16 in res:
        base = res[16]["ms"]
        print(f"\n{'BV':>4}{'programs':>10}{'ms':>10}{'speedup':>10}")
        for bv, r in sorted(res.items()):
            print(f"{bv:>4}{r['programs']:>10}{r['ms']:>10.2f}{base/r['ms']:>9.2f}x")
        best = min(res, key=lambda b: res[b]["ms"])
        gain = base / res[best]["ms"]
        verdict = (f"BV={best} is {gain:.2f}x the BV=16 default "
                   f"({(gain-1)*100:+.1f}%) with rel err {res[best]['rel_err']:.1e}"
                   if gain > 1.02 else
                   f"BV=16 is already optimal (best alternative {gain:.2f}x)")
        print(f"\nVERDICT: {verdict}")
        res["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 160, "config": vars(args), "sms": sms,
                   "results": {str(k): v for k, v in res.items()}}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
