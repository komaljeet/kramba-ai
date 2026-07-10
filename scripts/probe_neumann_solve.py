"""
De-risk the fused chunk-parallel megakernel's hard part BEFORE writing Triton.

The crux of #96 is the triangular solve  U = (I + A)^{-1} R  inside one fused
kernel. Classical forward substitution (U[i] = R[i] - A[i,:i]·U[:i]) is NOT
expressible in Triton — it requires indexing a tile by a loop variable, which
Triton cannot do. The escape hatch is the Neumann series: A is strictly lower
triangular => nilpotent (A^L = 0), so

    (I + A)^{-1} = sum_{k=0}^{L-1} (-A)^k          (EXACT in real arithmetic)

i.e.  U = R - A R + A^2 R - A^3 R + ... + (-A)^{L-1} R   — a chain of plain
matmuls, fully vectorizable, no dynamic indexing. That is what makes a fused
Triton kernel expressible at all.

The RISK is fp32: this is L-1 sequential matmuls of accumulated roundoff, and
our parity bar is strict (fwd < 1e-5 vs the exact `_scan_chunked_stable`, which
uses cuBLAS solve_triangular = exact forward sub). At hard decay (alpha ~ 0.1)
A's entries are O(1) and A^k decays slowly, so the series is long and roundoff
is worst. If Neumann-in-fp32 cannot hit 1e-5 at hard decay, the fused-kernel
design is dead before any Triton is written — pivot to forward-sub-by-other-
means or to Design B.

This probe compares, in pure eager fp32, U_neumann vs U_exact (solve_triangular)
for the SAME A, R that the stable chunked formulation produces, across decay
regimes incl hard alpha=0.1 and across chunk sizes. No Triton, runs in seconds.

    python scripts/probe_neumann_solve.py
"""
from __future__ import annotations

import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hvn.core.deltanet import DeltaNetBlock

N, D = 8, 64
DEV = torch.device("cuda")
DECAYS = [(0.10, 0.15), (0.25, 0.35), (0.50, 0.60), (0.85, 0.95), (0.95, 0.99)]
CHUNKS = (32, 64, 128, 256)
EXACT_BAR = 1e-5      # the test_issue95 "exact stable" bar
GATE_BAR = 2e-3      # the #79 gate bar


def _chunk_AR(qc, kc, vc, ac, bc, S_in):
    """Build the exact A, R for one chunk using the STABLE formulation (the one
    the megakernel must replicate). Returns A [N,L,L] (strict lower), R [N,L,d]."""
    N, L, d = kc.shape
    logP = torch.cumsum(torch.log(ac.clamp_min(1e-30)), dim=1)   # [N,L,d]
    P = torch.exp(logP)
    kbar = kc * P
    Sk = torch.einsum('nvk,nlk->nlv', S_in, kbar)               # [N,L,d]
    R = bc * (vc - Sk)
    # stable causal ratio Gram (strict lower) — same as _scan_chunked_stable
    idx = torch.arange(L, device=kc.device)
    mask = idx[None, :] < idx[:, None]                          # j < i
    logR = logP[:, :, None] - logP[:, None, :]                  # [N,L,L,d]
    ratio = torch.exp(logR.masked_fill(~mask[:, :, None], float('-inf')))
    A = bc * torch.einsum('nid,njd,nijd->nij', kc, kc, ratio)    # [N,L,L]
    return A, R, logP, P


def _neumann_U(A, R):
    """U = sum_{k=0}^{L-1} (-A)^k R  via iterative matmul. fp32."""
    N, L, d = R.shape
    U = R.clone()                  # k=0 term: R
    term = R.clone()               # (-A)^k R, starts at R (k=0)
    for _ in range(L - 1):
        term = -torch.bmm(A, term)  # term_{k+1} = -A @ term_k  -> (-A)^{k+1} R
        U = U + term
    return U


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} d={D}")
    print(f"  bars: exact<{EXACT_BAR:.0e}  #79-gate<{GATE_BAR:.0e}\n")
    worst_exact_fail = 0
    worst_gate_fail = 0
    for lo, hi in DECAYS:
        g = torch.Generator(device=DEV).manual_seed(int(100 * lo) + 7)
        q = torch.nn.functional.normalize(torch.randn(N, 256, D, device=DEV, generator=g), dim=-1)
        k = torch.nn.functional.normalize(torch.randn(N, 256, D, device=DEV, generator=g), dim=-1)
        v = torch.randn(N, 256, D, device=DEV, generator=g)
        a = torch.empty(N, 256, D, device=DEV).uniform_(lo, hi, generator=g)
        b = torch.sigmoid(torch.randn(N, 256, 1, device=DEV, generator=g))
        S_in = torch.zeros(N, D, D, device=DEV, dtype=torch.float32)
        for L in CHUNKS:
            qc, kc, vc, ac, bc = q[:, :L], k[:, :L], v[:, :L], a[:, :L], b[:, :L]
            A, R, logP, P = _chunk_AR(qc, kc, vc, ac, bc, S_in)
            M = torch.eye(L, device=DEV, dtype=torch.float32) + A
            U_exact = torch.linalg.solve_triangular(M, R, upper=False, unitriangular=True)
            U_neum = _neumann_U(A, R)
            rel = ((U_neum - U_exact).norm() / U_exact.norm().clamp_min(1e-12)).item()
            tag_e = "OK" if rel < EXACT_BAR else "FAIL(exact)"
            tag_g = "OK" if rel < GATE_BAR else "FAIL(gate)"
            worst_exact_fail += not (rel < EXACT_BAR)
            worst_gate_fail += not (rel < GATE_BAR)
            print(f"  decay[{lo:.2f},{hi:.2f}] L={L:<3}  Neumann-vs-exact rel {rel:.2e}  {tag_e:10s} {tag_g}")
        print()
    print(f"=== exact-bar(<{EXACT_BAR:.0e}) fails: {worst_exact_fail}   "
          f"#79-gate(<{GATE_BAR:.0e}) fails: {worst_gate_fail} ===")
    if worst_gate_fail == 0 and worst_exact_fail == 0:
        print("VERDICT: Neumann-in-fp32 is EXACT-grade at all decay — fused-kernel design is VIABLE.")
    elif worst_gate_fail == 0:
        print("VERDICT: Neumann passes the #79 gate everywhere but misses the strict 1e-5 exact bar somewhere.")
        print("         Fused kernel viable at the #79 tolerance; tighten (fp64 accum / Kahan) if 1e-5 needed.")
    else:
        print("VERDICT: Neumann-in-fp32 FAILS the #79 gate at some decay — fused-kernel design NOT viable as-is.")
        print("         Pivot: blocked forward-sub, or fp64 Neumann accumulation, or Design B.")


if __name__ == "__main__":
    main()