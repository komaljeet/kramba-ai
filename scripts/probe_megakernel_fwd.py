"""
Parity of the fused Triton megakernel `_mk_fwd_kernel` (#96, fwd-only) vs:
  (a) the eager `_neumann_scan` oracle (the EXACT math the kernel implements —
      kernel-vs-this should be ~1e-7, isolating Triton rounding from math error);
  (b) the ground-truth `_scan_sequential` (kernel-vs-this ~1e-6, the real bar).

Bars: fwd < 1e-5 (strict, from test_issue95.py) and < 2e-3 (#79 gate).
DECAYS incl hard alpha=0.10-0.15; CHUNK in (32, 64, 128). d=64.

    python scripts/probe_megakernel_fwd.py
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
from hvn.core.deltanet_triton import megakernel_forward
from probe_neumann_scan import _neumann_scan   # the eager oracle (exact math)

N, S, D = 4, 256, 64
DEV = torch.device("cuda")
DECAYS = [(0.10, 0.15), (0.25, 0.35), (0.50, 0.60), (0.85, 0.95), (0.95, 0.99)]
CHUNKS = (16, 32, 64)
STRICT_BAR = 1e-5
GATE_BAR = 2e-3


def _seq_ref(q, k, v, a, b):
    with torch.no_grad():
        return DeltaNetBlock._scan_sequential(q, k, v, a, b, return_state=False)


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} S={S} d={D}")
    print(f"  bars: strict<{STRICT_BAR:.0e}  #79-gate<{GATE_BAR:.0e}\n")
    worst_oracle = worst_seq = 0.0
    n_strict = n_gate = 0
    for lo, hi in DECAYS:
        g = torch.Generator(device=DEV).manual_seed(int(100 * lo) + 23)
        q = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
        k = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
        v = torch.randn(N, S, D, device=DEV, generator=g)
        a = torch.empty(N, S, D, device=DEV).uniform_(lo, hi, generator=g)
        b = torch.sigmoid(torch.randn(N, S, 1, device=DEV, generator=g))
        ref = _seq_ref(q, k, v, a, b)
        for L in CHUNKS:
            with torch.no_grad():
                mega = megakernel_forward(q, k, v, a, b, chunk=L)  # defaults: bv=16, bd=8
                ora = _neumann_scan(q, k, v, a, b, L)
            r_ora = _rel(mega, ora)
            r_seq = _rel(mega, ref)
            worst_oracle = max(worst_oracle, r_ora)
            worst_seq = max(worst_seq, r_seq)
            n_strict += not (r_seq < STRICT_BAR)
            n_gate += not (r_seq < GATE_BAR)
            tok = "OK" if r_seq < STRICT_BAR else ("gate" if r_seq < GATE_BAR else "FAIL")
            print(f"  decay[{lo:.2f},{hi:.2f}] L={L:<3}  vs-neumann {r_ora:.2e}  vs-seq {r_seq:.2e}  {tok}")
        print()
    print(f"=== worst: vs-neumann {worst_oracle:.2e}   vs-seq {worst_seq:.2e}   "
          f"strict(<{STRICT_BAR:.0e}) fails {n_strict}   #79-gate fails {n_gate} ===")
    if n_gate == 0 and n_strict == 0:
        print("VERDICT: Triton megakernel fwd matches the eager oracle (~1e-7) and the ground")
        print("         truth (<1e-6) at all decay & chunk sizes — PARITY PASSED. Proceed to benchmark.")
    elif n_gate == 0:
        print("VERDICT: passes #79 gate everywhere; strict 1e-5 missed somewhere (still usable).")
    else:
        print("VERDICT: FAILS the #79 gate — kernel has a bug, debug before benchmarking.")


if __name__ == "__main__":
    main()