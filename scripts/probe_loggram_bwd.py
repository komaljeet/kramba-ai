"""
Validate the fused log-Gram BACKWARD vs PyTorch autograd of the eager reference
(the ground truth for the math). If the Triton adjoint matches eager autograd,
the scan's autograd-composed backward is correct.

    python scripts/probe_loggram_bwd.py
"""
from __future__ import annotations

import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hvn.core.deltanet_triton import log_gram, _log_gram_ref

N, L, D = 4, 64, 64
DEV = torch.device("cuda")
DECAYS = [(0.10, 0.15), (0.50, 0.60), (0.95, 0.99)]


def _tensors(lo, hi, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    a = torch.nn.functional.normalize(torch.randn(N, L, D, device=DEV, generator=g), dim=-1)
    b = torch.nn.functional.normalize(torch.randn(N, L, D, device=DEV, generator=g), dim=-1)
    al = torch.empty(N, L, D, device=DEV).uniform_(lo, hi, generator=g)
    logP = torch.cumsum(torch.log(al.clamp_min(1e-30)), dim=1)
    return a, b, logP


def _grads(fn, a, b, logP, strict, dG):
    a = a.clone().requires_grad_(True)
    b = b.clone().requires_grad_(True)
    logP = logP.clone().requires_grad_(True)
    G = fn(a, b, logP, strict)
    G.backward(dG)
    return a.grad, b.grad, logP.grad


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} L={L} d={D}\n")
    worst = 0.0
    for lo, hi in DECAYS:
        for strict in (True, False):
            a, b, logP = _tensors(lo, hi, seed=int(100 * lo) + int(strict))
            dG = torch.randn(N, L, L, device=DEV, generator=torch.Generator(DEV).manual_seed(7))
            da_e, db_e, dlp_e = _grads(_log_gram_ref, a, b, logP, strict, dG)
            da_t, db_t, dlp_t = _grads(log_gram, a, b, logP, strict, dG)
            for name, e, t in (("da", da_e, da_t), ("db", db_e, db_t), ("dlogP", dlp_e, dlp_t)):
                rel = ((t - e).norm() / e.norm().clamp_min(1e-12)).item()
                worst = max(worst, rel)
                ok = rel < 1e-4
                tag = "strict" if strict else "lower"
                print(f"  decay[{lo:.2f},{hi:.2f}] {tag:6s} {name:5s} rel {rel:.2e} {'OK' if ok else 'FAIL'}")
    print(f"\nworst grad rel = {worst:.2e} ({'PASS' if worst < 1e-4 else 'FAIL'})")
    return 0 if worst < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())