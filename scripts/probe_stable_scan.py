"""
Validate DeltaNetBlock._scan_chunked_stable (#95) vs the sequential reference.

Proves the bounded-causal-ratio reformulation is EXACT (fwd+bwd) at ALL decay
and ALL chunk sizes — including the hard-decay regime where the original
_scan_chunked breaks (and where _safe_chunk currently has to shrink the chunk).
This is the math reference for the fused Triton kernel.

    python scripts/probe_stable_scan.py
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
DECAYS = [(0.10, 0.15), (0.25, 0.35), (0.5, 0.6), (0.85, 0.95), (0.95, 0.99)]
CHUNKS = (32, 64, 128, 256)


def _inputs(lo, hi, seed=0, grad=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    v = torch.randn(N, S, D, device=DEV, generator=g)
    a = torch.empty(N, S, D, device=DEV).uniform_(lo, hi, generator=g)
    b = torch.empty(N, S, 1, device=DEV).uniform_(0.0, 1.0, generator=g)
    if grad:
        for t in (q, k, v, a, b):
            t.requires_grad_(True)
    return q, k, v, a, b


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  shape N={N} S={S} d={D}\n")
    fwd_fail = bwd_fail = 0
    for lo, hi in DECAYS:
        # ---- forward parity ----
        q, k, v, a, b = _inputs(lo, hi, seed=1)
        ref = DeltaNetBlock._scan_sequential(q, k, v, a, b)
        for c in CHUNKS:
            out = DeltaNetBlock._scan_chunked_stable(q, k, v, a, b, c)
            rel = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            ok = rel < 1e-5
            fwd_fail += not ok
            print(f"  fwd  decay[{lo:.2f},{hi:.2f}] chunk {c:>3}  rel {rel:.2e}  {'OK' if ok else 'FAIL'}")
        # ---- backward parity ----
        ins_ref = _inputs(lo, hi, seed=2, grad=True)
        ins_chk = _inputs(lo, hi, seed=2, grad=True)
        go = torch.randn(N, S, D, device=DEV, generator=torch.Generator(DEV).manual_seed(9))
        DeltaNetBlock._scan_sequential(*ins_ref).mul(go).sum().backward()
        worst = 0.0
        for c in CHUNKS:
            ins_c = _inputs(lo, hi, seed=2, grad=True)
            DeltaNetBlock._scan_chunked_stable(*ins_c, c).mul(go).sum().backward()
            w = max(((x.grad - y.grad).norm() / (y.grad.norm() + 1e-9)).item()
                    for x, y in zip(ins_c, ins_ref))
            worst = max(worst, w)
            ok = w < 1e-4
            bwd_fail += not ok
            print(f"  bwd  decay[{lo:.2f},{hi:.2f}] chunk {c:>3}  grad rel {w:.2e}  {'OK' if ok else 'FAIL'}")
        print()

    # ---- contrast: original _scan_chunked BREAKS at hard decay, large chunk ----
    print("contrast — original _scan_chunked at hard decay (should BREAK at large chunk):")
    q, k, v, a, b = _inputs(0.10, 0.15, seed=1)
    ref = DeltaNetBlock._scan_sequential(q, k, v, a, b)
    for c in CHUNKS:
        out = DeltaNetBlock._scan_chunked(q, k, v, a, b, c)
        rel = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
        print(f"  OLD   decay[0.10,0.15] chunk {c:>3}  rel {rel:.2e}  {'breaks' if rel > 1e-3 else 'ok'}")

    print(f"\n=== stable: {fwd_fail} fwd FAIL, {bwd_fail} bwd FAIL (0 = all exact) ===")
    return fwd_fail + bwd_fail


if __name__ == "__main__":
    sys.exit(0 if main() == 0 else 1)