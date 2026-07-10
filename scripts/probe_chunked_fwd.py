"""
Validate the Triton chunk-parallel scan FORWARD vs the eager stable oracle and
the sequential ground truth, across decay regimes and chunk sizes.

    python scripts/probe_chunked_fwd.py
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
from hvn.core.deltanet_triton import chunked_scan_forward

N, S, D = 8, 512, 64
DEV = torch.device("cuda")
DECAYS = [(0.10, 0.15), (0.50, 0.60), (0.85, 0.95), (0.95, 0.99)]
CHUNKS = (64, 128, 256)


def _inputs(lo, hi, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    v = torch.randn(N, S, D, device=DEV, generator=g)
    a = torch.empty(N, S, D, device=DEV).uniform_(lo, hi, generator=g)
    b = torch.empty(N, S, 1, device=DEV).uniform_(0.0, 1.0, generator=g)
    return q, k, v, a, b


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} S={S} d={D}\n")
    worst_seq = 0.0
    worst_oracle = 0.0
    for lo, hi in DECAYS:
        q, k, v, a, b = _inputs(lo, hi, seed=1)
        seq = DeltaNetBlock._scan_sequential(q, k, v, a, b)
        oracle = DeltaNetBlock._scan_chunked_stable(q, k, v, a, b, 256)
        for c in CHUNKS:
            out = chunked_scan_forward(q, k, v, a, b, c)
            r_seq = ((out - seq).norm() / seq.norm().clamp_min(1e-12)).item()
            r_or = ((out - oracle).norm() / oracle.norm().clamp_min(1e-12)).item()
            worst_seq = max(worst_seq, r_seq)
            worst_oracle = max(worst_oracle, r_or)
            ok = r_seq < 1e-4
            print(f"  decay[{lo:.2f},{hi:.2f}] chunk {c:>3}  vs seq {r_seq:.2e}  vs oracle {r_or:.2e}  {'OK' if ok else 'FAIL'}")
    print(f"\nworst vs seq    = {worst_seq:.2e}")
    print(f"worst vs oracle = {worst_oracle:.2e}  (kernel==oracle up to fp32 roundoff)")
    return 0 if worst_seq < 1e-4 else 1


if __name__ == "__main__":
    sys.exit(main())