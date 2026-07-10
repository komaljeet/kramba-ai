"""
Test suite for Issue #95 — adaptive-exact chunk sizing.

``DeltaNetBlock._safe_chunk`` grows the chunk only to a size whose worst-case
intra-chunk cumprod-α stays above ``_P_FLOOR`` — so ``_scan_chunked`` runs
WITHOUT clamping and is numerically exact vs the sequential reference at ANY
decay, including the hard-decay regime where a fixed large chunk breaks (and
where the current fixed chunk-32 is itself silently wrong at α<0.31).

This locks the invariant (fwd + bwd parity + the safe-bound property).

Run on GPU (RTX 4050):  python tests/test_issue95.py
"""
import math

import torch

import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
from hvn.core.deltanet import DeltaNetBlock, _P_FLOOR

N, S, D = 8, 256, 64
# Includes the hard-decay regime that breaks a fixed large chunk.
DECAYS = [(0.10, 0.15), (0.25, 0.35), (0.5, 0.6), (0.85, 0.95), (0.95, 0.99)]


def _device():
    assert torch.cuda.is_available(), "CUDA required — this project runs on GPU only"
    return torch.device("cuda")


def _inputs(dev, lo, hi, seed=0, grad=False):
    torch.manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(N, S, D, device=dev), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(N, S, D, device=dev), dim=-1)
    v = torch.randn(N, S, D, device=dev)
    a = torch.empty(N, S, D, device=dev).uniform_(lo, hi)
    b = torch.empty(N, S, 1, device=dev).uniform_(0.0, 1.0)
    if grad:
        for t in (q, k, v, a, b):
            t.requires_grad_(True)
    return q, k, v, a, b


def test_adaptive_forward_exact_all_decay():
    dev = _device()
    for lo, hi in DECAYS:
        q, k, v, a, b = _inputs(dev, lo, hi)
        ref = DeltaNetBlock._scan_sequential(q, k, v, a, b)
        chunk = DeltaNetBlock._safe_chunk(a, 32, S)
        out = DeltaNetBlock._scan_chunked(q, k, v, a, b, chunk)
        rel = ((out - ref).norm() / ref.norm()).item()
        assert rel < 1e-5, f"decay [{lo},{hi}] chunk {chunk}: fwd rel_err {rel:.2e}"
        print(f"  fwd  decay [{lo},{hi}]  chunk {chunk:>3}  rel {rel:.2e}  OK")


def test_adaptive_backward_exact_all_decay():
    dev = _device()
    for lo, hi in DECAYS:
        q1, k1, v1, a1, b1 = _inputs(dev, lo, hi, grad=True)
        q2, k2, v2, a2, b2 = _inputs(dev, lo, hi, grad=True)
        DeltaNetBlock._scan_sequential(q1, k1, v1, a1, b1).sum().backward()
        chunk = DeltaNetBlock._safe_chunk(a2, 32, S)
        DeltaNetBlock._scan_chunked(q2, k2, v2, a2, b2, chunk).sum().backward()
        worst = 0.0
        for x, y in [(q2, q1), (k2, k1), (v2, v1), (a2, a1), (b2, b1)]:
            worst = max(worst, ((x.grad - y.grad).norm() / (y.grad.norm() + 1e-9)).item())
        assert worst < 1e-4, f"decay [{lo},{hi}] chunk {chunk}: grad rel {worst:.2e}"
        print(f"  bwd  decay [{lo},{hi}]  chunk {chunk:>3}  grad rel {worst:.2e}  OK")


def test_safe_chunk_within_exact_bound():
    dev = _device()
    for lo, hi in DECAYS:
        _, _, _, a, _ = _inputs(dev, lo, hi)
        chunk = DeltaNetBlock._safe_chunk(a, 32, S)
        a_min = float(a.min())
        bound = S if a_min >= 0.999 else int(math.log(_P_FLOOR) / math.log(a_min))
        assert chunk <= max(2, bound), \
            f"chunk {chunk} > exact bound {bound} (a_min={a_min:.3f})"
        print(f"  bound decay [{lo},{hi}]  chunk {chunk:>3} <= {bound}  OK")


# Hard-decay stress for the stable bounded-ratio scan (#95). The original
# _scan_chunked uses the 1/P reparam and is silently wrong once cumprod-α
# underflows (rel ~1..7 at α=0.1, even at chunk 32 — see scan-chunk finding).
# _scan_chunked_stable forms the causal ratios in log space (exp(logP_i-logP_j),
# masked before exp) so LARGE chunks stay exact at HARD decay. This locks that
# invariant and is the math reference for the fused Triton kernel.
STABLE_CHUNKS = (32, 64, 128, 256)


def test_stable_chunk_forward_exact_all_decay():
    dev = _device()
    for lo, hi in DECAYS:
        q, k, v, a, b = _inputs(dev, lo, hi)
        ref = DeltaNetBlock._scan_sequential(q, k, v, a, b)
        for chunk in STABLE_CHUNKS:
            out = DeltaNetBlock._scan_chunked_stable(q, k, v, a, b, chunk)
            rel = ((out - ref).norm() / ref.norm().clamp_min(1e-12)).item()
            assert rel < 1e-5, f"stable decay [{lo},{hi}] chunk {chunk}: fwd rel {rel:.2e}"
            print(f"  stable fwd  decay [{lo},{hi}]  chunk {chunk:>3}  rel {rel:.2e}  OK")


def test_stable_chunk_backward_exact_all_decay():
    dev = _device()
    for lo, hi in DECAYS:
        ins_ref0 = _inputs(dev, lo, hi, grad=True)
        go = torch.randn(N, S, D, device=dev, generator=torch.Generator(dev).manual_seed(9))
        DeltaNetBlock._scan_sequential(*ins_ref0).mul(go).sum().backward()
        for chunk in STABLE_CHUNKS:
            ins_c = _inputs(dev, lo, hi, grad=True)
            DeltaNetBlock._scan_chunked_stable(*ins_c, chunk).mul(go).sum().backward()
            worst = max(((x.grad - y.grad).norm() / (y.grad.norm() + 1e-9)).item()
                       for x, y in zip(ins_c, ins_ref0))
            assert worst < 1e-4, f"stable decay [{lo},{hi}] chunk {chunk}: grad rel {worst:.2e}"
            print(f"  stable bwd  decay [{lo},{hi}]  chunk {chunk:>3}  grad rel {worst:.2e}  OK")


if __name__ == "__main__":
    test_adaptive_forward_exact_all_decay()
    test_adaptive_backward_exact_all_decay()
    test_safe_chunk_within_exact_bound()
    test_stable_chunk_forward_exact_all_decay()
    test_stable_chunk_backward_exact_all_decay()
    print("\nAll #95 adaptive-chunk + stable-scan parity tests passed.")
