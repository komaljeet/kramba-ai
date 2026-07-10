"""
Parity of Design B (`megakernel_scan` = optimal sequential fwd + fused
blockwise-recompute bwd) vs the validated current `_GatedDeltaScan` (gated_delta_scan),
whose backward is verified vs autograd at 1e-16 in fp64.

The Design B adjoint body is verbatim from `_gd_bwd_kernel`; only the Sprev source
differs (per-block SRAM recompute from block-boundary checkpoints vs the full HBM
states array). So this isolates whether the blockwise recompute reproduces the
exact entering states and the exact adjoint — it should be ~1e-6 (fp32 recompute
reordering), well under the bars.

Checks both FORWARD (must be bit-identical — same fwd kernel) and BACKWARD grads
(dq, dk, dv, dalpha, dbeta) per-tensor rel.

Bars: bwd < 1e-4 (strict, from test_issue95.py) and < 5e-3 (#79 bwd gate).
DECAYS incl hard alpha=0.10-0.15. d=64.

    python scripts/probe_megakernel_bwd.py
"""
from __future__ import annotations

import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hvn.core.deltanet_triton import gated_delta_scan, megakernel_scan

N, S, D = 4, 256, 64
DEV = torch.device("cuda")
DECAYS = [(0.10, 0.15), (0.25, 0.35), (0.50, 0.60), (0.85, 0.95), (0.95, 0.99)]
STRICT_BAR = 1e-4
GATE_BAR = 5e-3


def _inputs(lo, hi, seed):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    v = torch.randn(N, S, D, device=DEV, generator=g)
    a = torch.empty(N, S, D, device=DEV).uniform_(lo, hi, generator=g)
    b = torch.sigmoid(torch.randn(N, S, 1, device=DEV, generator=g))
    for t in (q, k, v, a, b):
        t.requires_grad_(True)
    return q, k, v, a, b


def _clone_inputs(args):
    return tuple(t.detach().clone().requires_grad_(True) for t in args)


def _rel(a, b):
    return ((a - b).norm() / b.norm().clamp_min(1e-12)).item()


def _grads(args, out):
    q, k, v, a, b = args
    out.sum().backward()
    return [q.grad, k.grad, v.grad, a.grad, b.grad]


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} S={S} d={D}")
    print(f"  bars: bwd strict<{STRICT_BAR:.0e}  #79-gate<{GATE_BAR:.0e}\n")
    worst_fwd = 0.0
    worst_bwd = 0.0
    n_strict = n_gate = 0
    names = ["dq", "dk", "dv", "dalpha", "dbeta"]
    for lo, hi in DECAYS:
        seed = int(100 * lo) + 9
        ref_args = _inputs(lo, hi, seed)
        ref_out = gated_delta_scan(*ref_args)
        ref_g = _grads(ref_args, ref_out)

        mega_args = _clone_inputs(ref_args)
        mega_out = megakernel_scan(*mega_args)
        mega_g = _grads(mega_args, mega_out)

        fwd_rel = _rel(mega_out, ref_out)
        worst_fwd = max(worst_fwd, fwd_rel)
        print(f"  decay[{lo:.2f},{hi:.2f}]  fwd rel {fwd_rel:.2e}")
        worst_here = 0.0
        for nm, rg, mg in zip(names, ref_g, mega_g):
            r = _rel(mg, rg)
            worst_here = max(worst_here, r)
            worst_bwd = max(worst_bwd, r)
            n_strict += not (r < STRICT_BAR)
            n_gate += not (r < GATE_BAR)
            tag = "OK" if r < STRICT_BAR else ("gate" if r < GATE_BAR else "FAIL")
            print(f"      {nm:7s} rel {r:.2e}  {tag}")
        print(f"      worst bwd this decay: {worst_here:.2e}")
        print()
    print(f"=== worst fwd {worst_fwd:.2e}   worst bwd {worst_bwd:.2e}   "
          f"strict(<{STRICT_BAR:.0e}) fails {n_strict}   #79-gate(<{GATE_BAR:.0e}) fails {n_gate} ===")
    if n_gate == 0 and n_strict == 0:
        print("VERDICT: Design B backward matches the validated sequential backward (strict 1e-4) at")
        print("         all decay — PARITY PASSED. Proceed to the speed gate.")
    elif n_gate == 0:
        print("VERDICT: passes #79 bwd gate everywhere; strict 1e-4 missed somewhere (still usable).")
    else:
        print("VERDICT: FAILS the #79 bwd gate — blockwise recompute has a bug, debug before benchmarking.")


if __name__ == "__main__":
    main()