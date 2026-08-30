"""Gate: can log-addition be reformulated as a tensor-core GEMM? (3 questions)

Author: Rambarun Komaljeet

The proposal
------------
    standard GEMM :  D = A @ B
    log-addition  :  log(D) = log(A) + log(B)
    reformulated  :  log(output) = log(W) @ ones + log(x)

with log(W) precomputed once per block, so a tensor-core GEMM does the work.

Three questions were asked -- speed, accuracy, overhead -- with "do not build
further if the benchmark fails". This answers all three, and adds the one check
that decides it: the CORRECT log-domain formulation, so the proposal is not
rejected merely for being stated loosely.

What the arms are
-----------------
A. ``gemm``        -- ``x @ W`` in bf16. The reference, and the thing to beat.
B. ``proposed``    -- ``ones @ log(W) + log(x)``, exactly as specified.
C. ``logsumexp``   -- the mathematically correct log-domain matmul,
                      ``log(D)[i,j] = logsumexp_k( log(x)[i,k] + log(W)[k,j] )``.
D. ``logw_precompute`` -- question 3: what ``log(W)`` itself costs.

Why C has to be in the table
----------------------------
A matmul is multiply **and sum**. Logs convert the multiply to an add, but
``log(a + b) != log(a) + log(b)``, so the sum has to happen in linear space. The
exact log-domain identity is therefore a log-sum-exp reduction, whose inner
operation is ``exp(la + lb)`` followed by an add -- the same count of operations
as ``a*b`` followed by an add, except ``exp`` is a special-function-unit op that
is more expensive than a multiply and that tensor cores cannot issue at all.
Arm C measures that rather than asserting it.

Three structural problems the numbers below should confirm
----------------------------------------------------------
1. **Rank.** ``(ones @ log(W))[i,j] = sum_k log(W[k,j])`` is independent of *i*,
   so the term is rank-1 -- constant down every column. A general matmul output
   is full rank. Arm B's output therefore cannot represent arm A's for any W, x,
   which is a stronger statement than "it is inaccurate".
2. **Shape.** ``log(W) @ ones + log(x)`` only type-checks when K == N: the first
   term is [M,N] and ``log(x)`` is [M,K]. A valid reformulation of a matmul has
   to work for any shape, so this gate can only be run at all on square weights.
   The 1B and 48M projection shapes happen to be square, so it is testable.
3. **Sign.** ``log`` is undefined for negatives and ``-inf`` at zero. Weights and
   activations are signed, and ternary weights are exactly ``{-s, 0, +s}`` --
   two of the three values have no real logarithm. Recovering the sign needs
   elementwise bookkeeping outside the GEMM, which is where the cost went.

Usage::

    python scripts/gate_log_gemm.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SHAPES = [
    ("1B proj",  4096, 2048, 2048),
    ("48M proj", 2048, 512, 512),
]


def timeit(fn, warmup=5, iters=30):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def run(tag, M, K, N, device, args):
    assert K == N, "the proposed formula only type-checks for square W"
    flops = 2 * M * K * N

    torch.manual_seed(0)
    x = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    W = torch.randn(K, N, device=device, dtype=torch.bfloat16)
    ones = torch.ones(M, K, device=device, dtype=torch.bfloat16)

    ref = (x.float() @ W.float())

    # ---- Q3 first: what does log(W) cost? ------------------------------------
    t_logw = timeit(lambda: torch.log(W.abs().float()))
    logW = torch.log(W.abs().float()).to(torch.bfloat16)
    logx = torch.log(x.abs().float()).to(torch.bfloat16)

    # ---- A. the reference ----------------------------------------------------
    t_gemm = timeit(lambda: x @ W)

    # ---- B. the proposal, exactly as written ---------------------------------
    def proposed():
        return (ones @ logW) + logx
    t_prop = timeit(proposed)
    out_b = proposed().float()
    # charitable read: the caller means log(|output|), so exponentiate back
    recon_b = torch.exp(out_b.clamp(max=60.0))

    # ---- C. the correct log-domain identity ----------------------------------
    def logsumexp_matmul():
        # log(D)[i,j] = logsumexp_k ( logx[i,k] + logW[k,j] )
        return torch.logsumexp(logx.float().unsqueeze(2)
                               + logW.float().unsqueeze(0), dim=1)
    try:
        t_lse = timeit(logsumexp_matmul, warmup=1, iters=3)
        lse_ok = True
    except torch.cuda.OutOfMemoryError:
        t_lse, lse_ok = float("nan"), False
        torch.cuda.empty_cache()

    # ---- accuracy ------------------------------------------------------------
    def err(a, b):
        d = (a - b).abs()
        den = b.abs().clamp(min=1e-6)
        return {"mean_abs": float(d.mean()), "max_abs": float(d.max()),
                "mean_rel": float((d / den).mean()),
                "corr": float(torch.corrcoef(
                    torch.stack([a.flatten().float(),
                                 b.flatten().float()]))[0, 1])
                # nan here is a RESULT, not a missing measurement: it means the
                # reconstruction has zero variance (it underflowed to a constant
                # 0), so it carries no signal to correlate with at all.
                }

    e_b = err(recon_b, ref)

    # Rank of the proposed term -- the structural claim, proved directly.
    #
    # torch.linalg.matrix_rank is NOT used for the numerator: it returned 15-16
    # here purely as a tolerance artifact (the trailing singular values are fp
    # noise ~1e-13 relative, and an absolute tol misreads them as signal). The
    # rigorous statement is stronger and needs no tolerance at all: every ROW of
    # (ones @ log W) is bit-identical, because the entry does not depend on i.
    # A matrix whose rows are all equal has rank exactly 1.
    term = (ones.float() @ torch.log(W.abs().float()))
    row_spread = float((term - term[0:1]).abs().max())
    rank_term = 1 if row_spread == 0.0 else int(
        torch.linalg.matrix_rank(term[:min(256, M), :min(256, N)]))
    rank_ref = int(torch.linalg.matrix_rank(ref[:min(256, M), :min(256, N)]))

    # Dynamic range: summing K logs of |N(0,1)| lands near -0.6*K, and exp() of
    # that underflows to exactly zero. A third, independent failure.
    logsum_min, logsum_max = float(term.min()), float(term.max())
    underflowed = float((recon_b == 0).float().mean())

    # sign: how many entries does a log-domain path get the sign wrong on?
    sign_lost = float((torch.sign(ref) != torch.sign(recon_b)).float().mean())

    res = {
        "tag": tag, "M": M, "K": K, "N": N,
        "gemm_ms": t_gemm * 1e3, "gemm_tflops": flops / t_gemm / 1e12,
        "proposed_ms": t_prop * 1e3, "proposed_tflops": flops / t_prop / 1e12,
        "proposed_speedup_vs_gemm": t_gemm / t_prop,
        "logsumexp_ms": (t_lse * 1e3) if lse_ok else None,
        "logsumexp_slowdown_vs_gemm": (t_lse / t_gemm) if lse_ok else None,
        "logw_precompute_ms": t_logw * 1e3,
        "logw_share_of_gemm": t_logw / t_gemm,
        "accuracy_vs_gemm": e_b,
        "rank_proposed_term": rank_term, "rank_true_gemm": rank_ref,
        "row_spread_of_proposed_term": row_spread,
        "logsum_range": [logsum_min, logsum_max],
        "fraction_underflowed_to_zero": underflowed,
        "fraction_sign_wrong": sign_lost,
    }

    print(f"\n=== {tag}  M={M} K={K} N={N} ===")
    print(f"  A gemm (bf16, tensor cores)   {t_gemm*1e3:9.3f} ms   "
          f"{flops/t_gemm/1e12:6.2f} TFLOP/s")
    print(f"  B proposed  ones@logW + logx  {t_prop*1e3:9.3f} ms   "
          f"{flops/t_prop/1e12:6.2f} TFLOP/s   "
          f"({t_gemm/t_prop:.2f}x vs gemm)")
    if lse_ok:
        print(f"  C logsumexp (CORRECT form)    {t_lse*1e3:9.3f} ms   "
              f"({t_lse/t_gemm:.0f}x SLOWER than gemm)")
    else:
        print(f"  C logsumexp (CORRECT form)    OOM — the [M,K,N] intermediate "
              f"does not fit; that is the point")
    print(f"  D log(W) precompute           {t_logw*1e3:9.3f} ms   "
          f"({t_logw/t_gemm*100:.1f}% of one GEMM)")
    print(f"  accuracy of B vs true GEMM:")
    print(f"    mean abs err {e_b['mean_abs']:.4e}   max abs err {e_b['max_abs']:.4e}")
    print(f"    mean rel err {e_b['mean_rel']:.4e}   correlation {e_b['corr']:+.4f}")
    print(f"    rank(ones@logW) = {rank_term} (all rows identical, spread "
          f"{row_spread:.1e})   rank(true GEMM) = {rank_ref}")
    print(f"    log-sum term spans [{logsum_min:.0f}, {logsum_max:.0f}] -> exp() "
          f"underflows {underflowed*100:.0f}% of entries to exactly 0")
    print(f"    sign wrong on {sign_lost*100:.1f}% of entries")
    return res


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="results/log_gemm/gate.json")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md)")
    device = torch.device("cuda")
    print(f"[gate] {torch.cuda.get_device_name(0)}")

    results = [run(*s, device=device, args=args) for s in SHAPES]

    # ---- verdict -------------------------------------------------------------
    print("\n[gate] ================== VERDICT ==================")
    fast = all(r["proposed_speedup_vs_gemm"] >= 1.0 for r in results)
    def _corr_ok(c):
        return c == c and c > 0.99      # nan (constant output) is a FAIL
    accurate = all(_corr_ok(r["accuracy_vs_gemm"]["corr"]) for r in results)
    rank_ok = all(r["rank_proposed_term"] >= r["rank_true_gemm"] for r in results)

    print(f"[gate] Q1 speed    : proposed {'MATCHES/BEATS' if fast else 'does NOT beat'} "
          f"the bf16 GEMM")
    for r in results:
        print(f"[gate]              {r['tag']:<10} {r['proposed_speedup_vs_gemm']:.2f}x")
    print(f"[gate] Q2 accuracy : {'PASS' if accurate else 'FAIL'}")
    for r in results:
        a = r["accuracy_vs_gemm"]
        print(f"[gate]              {r['tag']:<10} corr {a['corr']:+.4f}  "
              f"mean rel err {a['mean_rel']:.3e}  "
              f"rank {r['rank_proposed_term']} vs {r['rank_true_gemm']}")
    print("[gate] Q3 overhead : log(W) as a share of one GEMM")
    for r in results:
        print(f"[gate]              {r['tag']:<10} "
              f"{r['logw_share_of_gemm']*100:.1f}%  "
              f"({r['logw_precompute_ms']:.3f} ms)")

    go = fast and accurate and rank_ok
    print(f"\n[gate] {'GO' if go else 'NO-GO'}")
    if not go:
        print("[gate] The blocker is Q2, and it is structural rather than "
              "numerical: (ones @ log W)[i,j] does not depend on i, so the term "
              "is RANK-1 while a real GEMM output is full rank. No choice of W "
              "or x makes it match. Separately, log(a+b) != log(a)+log(b), so "
              "the sum in a matmul cannot be done in log space at all -- the "
              "exact form is the logsumexp in arm C, whose inner op is exp+add "
              "(more expensive than mul+add, and not a tensor-core op).")
        print("[gate] Per the stated instruction, STOP HERE. The 1000-step 48M "
              "loss-curve run was NOT launched: an approximation that is rank-1 "
              "against a full-rank target has no threshold to find.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"question": "log-addition as a tensor-core GEMM",
                   "results": results, "go": go}, fh, indent=2)
    print(f"[gate] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
