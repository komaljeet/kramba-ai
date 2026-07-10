"""
Validate the Triton log-space Gram kernel (fwd) vs the eager fp32 reference,
across decay regimes and chunk sizes. This is the novel fused op of the #95
chunk-parallel scan; it must match the oracle's A/QK core to ~1e-6.

    python scripts/probe_loggram.py
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

DEV = torch.device("cuda")
N, D = 8, 64
LS = [32, 64, 128, 256]
DECAYS = [(0.10, 0.15), (0.50, 0.60), (0.95, 0.99)]


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}\n")
    worst = 0.0
    for lo, hi in DECAYS:
        for L in LS:
            g = torch.Generator(device=DEV).manual_seed(L + int(100 * lo))
            a = torch.nn.functional.normalize(torch.randn(N, L, D, device=DEV, generator=g), dim=-1)
            b = torch.nn.functional.normalize(torch.randn(N, L, D, device=DEV, generator=g), dim=-1)
            al = torch.empty(N, L, D, device=DEV).uniform_(lo, hi, generator=g)
            logP = torch.cumsum(torch.log(al.clamp_min(1e-30)), dim=1)
            for strict in (True, False):
                ref = _log_gram_ref(a, b, logP, strict)
                out = log_gram(a, b, logP, strict)
                rel = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
                worst = max(worst, rel)
                ok = rel < 1e-5
                tag = "strict" if strict else "lower"
                print(f"  decay[{lo:.2f},{hi:.2f}] L {L:>3} {tag:6s} rel {rel:.2e} {'OK' if ok else 'FAIL'}")
    print(f"\nworst rel = {worst:.2e} ({'PASS' if worst < 1e-5 else 'FAIL'})")
    return 0 if worst < 1e-5 else 1


if __name__ == "__main__":
    sys.exit(main())