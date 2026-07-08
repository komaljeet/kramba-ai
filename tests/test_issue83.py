"""
Test suite for Issue #83 — custom Triton gated-delta-rule kernel.

Validates hvn.core.deltanet_triton against the ground-truth reference
(DeltaNetBlock._scan_sequential), which the #79 harness established. Unlike the
eager _scan_chunked (numerically broken at hard decay), the direct-recurrence
Triton kernel must match at ALL decay regimes — including harsh α where the
1/P reparameterization blows up.

Run on GPU (RTX 4050):  python tests/test_issue83.py
"""

import time
import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from hvn.core.deltanet import DeltaNetBlock, _l2norm
from hvn.core.deltanet_triton import gated_delta_forward, _HAS_TRITON


def _device():
    assert torch.cuda.is_available(), "CUDA required — this project runs on GPU only"
    return torch.device("cuda")


def _rel(a, b):
    return (a - b).abs().max().item() / b.abs().max().clamp_min(1e-12).item()


def _mk(N, S, d, abias, seed, dev):
    g = torch.Generator(dev).manual_seed(seed)
    q = _l2norm(torch.randn(N, S, d, generator=g, device=dev))
    k = _l2norm(torch.randn(N, S, d, generator=g, device=dev))
    v = torch.randn(N, S, d, generator=g, device=dev)
    alpha = torch.sigmoid(abias + torch.randn(N, S, d, generator=g, device=dev))
    beta = torch.sigmoid(torch.randn(N, S, 1, generator=g, device=dev))
    return q, k, v, alpha, beta


def test_triton_forward_matches_sequential_all_decay():
    """Kernel == sequential reference across gentle→very-harsh decay (stability)."""
    assert _HAS_TRITON, "triton not available"
    dev = _device()
    N, S, d = 8, 512, 64
    worst = 0.0
    for abias, label in [(3.0, "gentle~.95"), (0.0, "mixed~.5"),
                         (-1.0, "harsh~.27"), (-3.0, "vharsh~.05")]:
        q, k, v, alpha, beta = _mk(N, S, d, abias, 7, dev)
        ref = DeltaNetBlock._scan_sequential(q, k, v, alpha, beta)
        out = gated_delta_forward(q, k, v, alpha, beta)
        r = _rel(out, ref)
        worst = max(worst, r)
        assert r < 1e-3, f"[{label}] Triton fwd rel-diff {r:.2e} ≥ 1e-3"
        print(f"[fwd] {label:>11}  rel {r:.2e}  OK")
    print(f"[fwd] worst rel {worst:.2e} — stable at ALL decay (eager chunk is not)")


def bench_forward(N=32, S=512, d=64):
    """Time Triton forward vs eager _scan_chunked (forward only)."""
    dev = _device()
    q, k, v, alpha, beta = _mk(N, S, d, 3.0, 1, dev)

    def timeit(fn, warm=5, it=20):
        for _ in range(warm):
            fn()
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(it):
            fn()
        torch.cuda.synchronize(); return (time.perf_counter() - t0) / it * 1000

    te = timeit(lambda: DeltaNetBlock._scan_chunked(q, k, v, alpha, beta, 32))
    tt = timeit(lambda: gated_delta_forward(q, k, v, alpha, beta))
    print(f"\n  scan fwd, N={N} S={S} d={d}:")
    print(f"    eager _scan_chunked : {te:7.2f} ms")
    print(f"    triton fwd          : {tt:7.2f} ms   ({te/tt:.2f}x vs eager)")


if __name__ == "__main__":
    test_triton_forward_matches_sequential_all_decay()
    bench_forward()
    print("\nMilestone 1 (forward) complete.")
