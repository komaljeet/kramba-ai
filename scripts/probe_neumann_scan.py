"""
Lock the FULL chunked-scan algorithm (not just the solve) with the Neumann
series, in eager, BEFORE writing any Triton.

scripts/probe_neumann_solve.py validated the solve alone: U = sum_{k<L} (-A)^k R
matches solve_triangular to ~1e-8 at all decay. But the megakernel fuses the WHOLE
pipeline:

    logP -> A (log-Gram, strict) -> Neumann-solve U -> QK (log-Gram, lower)
         -> O = S_in@qbar + QK@U -> cross-chunk carry S_out

So the question this answers: does the *entire* chunked scan, with the
triangular solve replaced by the Neumann series, still match the ground-truth
sequential scan (`_scan_sequential`) to <1e-5 at all decay incl hard alpha=0.1,
across chunk sizes? If yes, the algorithm is fully validated in eager — the
Triton megakernel is then a pure performance rewrite of known-correct math, and
parity debugging reduces to "does my kernel match this eager oracle".

This `_neumann_scan` is also the parity ORACLE the Triton kernel will be checked
against (it is the exact math the kernel implements, including the Neumann
solve — so kernel-vs-this should be ~1e-7, tighter than kernel-vs-sequential).

    python scripts/probe_neumann_scan.py
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

N, S, D = 4, 256, 64
DEV = torch.device("cuda")
DECAYS = [(0.10, 0.15), (0.25, 0.35), (0.50, 0.60), (0.85, 0.95), (0.95, 0.99)]
CHUNKS = (32, 64, 128, 256)
FWD_BAR = 1e-5
GATE_BAR = 2e-3


def _neumann_U(A, R):
    """U = (I + A)^{-1} R = sum_{k=0}^{L-1} (-A)^k R  (A strictly lower => nilpotent)."""
    N, L, d = R.shape
    U = R.clone()
    term = R.clone()
    for _ in range(L - 1):
        term = -torch.bmm(A, term)
        U = U + term
    return U


def _neumann_scan(q, k, v, alpha, beta, chunk):
    """Eager chunk-parallel scan with the Neumann solve — the EXACT math the
    Triton megakernel will implement. fp32, stable (bounded log-space ratios)."""
    N, S, d = q.shape
    dev = q.device
    q, k, v, alpha = q.float(), k.float(), v.float(), alpha.float()
    beta = beta.reshape(N, S).float()
    S_in = torch.zeros(N, d, d, device=dev, dtype=torch.float32)
    outs = []
    for s in range(0, S, chunk):
        e = min(s + chunk, S)
        L = e - s
        qc, kc, vc, ac, bc = q[:, s:e], k[:, s:e], v[:, s:e], alpha[:, s:e], beta[:, s:e][:, :, None]
        logP = torch.cumsum(torch.log(ac.clamp_min(1e-30)), dim=1)
        P = torch.exp(logP)
        P_last = P[:, -1, :]
        logP_last = logP[:, -1, :]
        kbar = kc * P
        qbar = qc * P
        Sk = torch.einsum('nvk,nlk->nlv', S_in, kbar)
        R = bc * (vc - Sk)
        # A = bc * strict log-Gram(k,k)
        idx = torch.arange(L, device=dev)
        mstrict = idx[None, :] < idx[:, None]
        logR = logP[:, :, None] - logP[:, None, :]
        ratio_s = torch.exp(logR.masked_fill(~mstrict[:, :, None], float('-inf')))
        A = bc * torch.einsum('nid,njd,nijd->nij', kc, kc, ratio_s)
        U = _neumann_U(A, R)
        # QK = lower-incl-diag log-Gram(q,k)
        mlower = idx[None, :] <= idx[:, None]
        ratio_l = torch.exp(logR.masked_fill(~mlower[:, :, None], float('-inf')))
        QK = torch.einsum('nid,njd,nijd->nij', qc, kc, ratio_l)
        O = (torch.einsum('nvk,nlk->nlv', S_in, qbar)
             + torch.einsum('nij,njv->niv', QK, U))
        outs.append(O)
        k_scaled = kc * torch.exp(logP_last[:, None, :] - logP)
        S_in = (S_in * P_last[:, None, :]
                + torch.einsum('niv,nik->nvk', U, k_scaled))
    return torch.cat(outs, dim=1)


def _seq_ref(q, k, v, alpha, beta):
    """Ground truth via the block's own sequential scan."""
    with torch.no_grad():
        return DeltaNetBlock._scan_sequential(q, k, v, alpha, beta, return_state=False)


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} S={S} d={D}")
    print(f"  bars: fwd<{FWD_BAR:.0e}  #79-gate<{GATE_BAR:.0e}\n")
    worst_fwd = 0.0
    n_fail_exact = n_fail_gate = 0
    for lo, hi in DECAYS:
        g = torch.Generator(device=DEV).manual_seed(int(100 * lo) + 11)
        q = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
        k = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
        v = torch.randn(N, S, D, device=DEV, generator=g)
        a = torch.empty(N, S, D, device=DEV).uniform_(lo, hi, generator=g)
        b = torch.sigmoid(torch.randn(N, S, 1, device=DEV, generator=g))
        ref = _seq_ref(q, k, v, a, b)
        for L in CHUNKS:
            out = _neumann_scan(q, k, v, a, b, L)
            rel = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            worst_fwd = max(worst_fwd, rel)
            n_fail_exact += not (rel < FWD_BAR)
            n_fail_gate += not (rel < GATE_BAR)
            tag = "OK" if rel < FWD_BAR else ("gate" if rel < GATE_BAR else "FAIL")
            print(f"  decay[{lo:.2f},{hi:.2f}] L={L:<3}  Neumann-scan vs sequential rel {rel:.2e}  {tag}")
        print()
    print(f"=== worst fwd rel {worst_fwd:.2e}   exact(<{FWD_BAR:.0e}) fails {n_fail_exact}   "
          f"#79-gate(<{GATE_BAR:.0e}) fails {n_fail_gate} ===")
    if n_fail_gate == 0 and n_fail_exact == 0:
        print("VERDICT: full Neumann-scan is EXACT-grade vs sequential at all decay & chunk sizes.")
        print("         Algorithm fully validated in eager. Triton megakernel = pure perf rewrite of this.")
    elif n_fail_gate == 0:
        print("VERDICT: passes #79 gate everywhere; misses strict 1e-5 somewhere (still usable; tighten later).")
    else:
        print("VERDICT: FAILS the #79 gate — algorithm has a bug, fix before any Triton.")


if __name__ == "__main__":
    main()