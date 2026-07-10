"""
Chunked-parallel scan parity gate (#79/#95): forward AND backward must match the
sequential ground truth at ALL decay incl hard alpha~0.3, at the #79 tolerances
(fwd 2e-3 / bwd 5e-3). This is the gate the chunked kernel must pass before it is
allowed anywhere near the training path.

    python scripts/probe_chunked_bwd.py
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

N, S, D = 8, 256, 64
DEV = torch.device("cuda")
FWD_RTOL, BWD_RTOL = 2e-3, 5e-3
DECAYS = [(0.10, 0.20), (0.30, 0.40), (0.50, 0.60), (0.85, 0.95), (0.95, 0.99)]
CHUNKS = (64, 128, 256)


def _inputs(lo, hi, seed, grad=False):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    v = torch.randn(N, S, D, device=DEV, generator=g)
    a = torch.empty(N, S, D, device=DEV).uniform_(lo, hi, generator=g)
    b = torch.empty(N, S, 1, device=DEV).uniform_(0.0, 1.0, generator=g)
    if grad:
        for t in (q, k, v, a, b):
            t.requires_grad_(True)
    return [q, k, v, a, b]


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} S={S} d={D}")
    print(f"  tolerances: fwd {FWD_RTOL:.0e} / bwd {BWD_RTOL:.0e}\n")
    fwd_fail = bwd_fail = 0
    for lo, hi in DECAYS:
        # forward
        q, k, v, a, b = _inputs(lo, hi, 1)
        ref = DeltaNetBlock._scan_sequential(q, k, v, a, b)
        for c in CHUNKS:
            out = chunked_scan_forward(q, k, v, a, b, c)
            r = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            ok = r < FWD_RTOL
            fwd_fail += not ok
            print(f"  fwd  decay[{lo:.2f},{hi:.2f}] C {c:>3} rel {r:.2e} {'OK' if ok else 'FAIL'}")
        # backward
        ins_ref = _inputs(lo, hi, 2, grad=True)
        go = torch.randn(N, S, D, device=DEV, generator=torch.Generator(DEV).manual_seed(9))
        DeltaNetBlock._scan_sequential(*ins_ref).mul(go).sum().backward()
        for c in CHUNKS:
            ins_c = _inputs(lo, hi, 2, grad=True)
            chunked_scan_forward(*ins_c, c).mul(go).sum().backward()
            worst = max(((x.grad - y.grad).norm() / (y.grad.norm() + 1e-9)).item()
                        for x, y in zip(ins_c, ins_ref))
            ok = worst < BWD_RTOL
            bwd_fail += not ok
            print(f"  bwd  decay[{lo:.2f},{hi:.2f}] C {c:>3} grad rel {worst:.2e} {'OK' if ok else 'FAIL'}")
        print()
    print(f"=== {fwd_fail} fwd FAIL, {bwd_fail} bwd FAIL ({'GATE PASSED' if fwd_fail+bwd_fail == 0 else 'GATE FAILED'}) ===")
    return 0 if fwd_fail + bwd_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(main())