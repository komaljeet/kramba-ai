"""Fused Triton L2 norm (hvn/core/fused_l2norm.py) vs the PyTorch reference.

What would make these vacuous, and the guard:
  * A backward that drops the projection term (dx = g/c only) is the classic
    wrong L2-norm gradient. `test_backward_projection_term_is_exercised`
    checks that this wrong gradient is FAR from the reference on the same
    inputs, so the gradient tests can tell the two apart.
  * Comparing only to the bf16 reference cannot tell "fused is wrong by 1 ulp"
    from "reference is off by 1 ulp". The fp64 test requires the fused result
    to be at least as accurate as the reference, element-wise in aggregate.
  * The integration test counts kernel calls, so it fails if the block quietly
    keeps using the reference path.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.core.deltanet import _l2norm  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SHAPES = [(1, 4096, 64, 32), (2, 129, 4, 48), (3, 7, 5, 64), (1, 33, 28, 32)]


def _fused():
    from hvn.core.fused_l2norm import fused_l2norm
    return fused_l2norm


def _inputs(shape, dtype, seed=0, clamp_rows=False):
    """clamp_rows adds a zero row and a below-eps row. Kept OUT of the general
    inputs: their gradient is g/eps (~1e6x everything else), so any max-relative
    comparison would see only those rows (found by the anti-vacuity test)."""
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, device="cuda", generator=g).to(dtype)
    if clamp_rows:
        x.view(-1, shape[-1])[0] = 0            # zero row: clamp active, norm 0
        x.view(-1, shape[-1])[1] *= 1e-8        # tiny row: below eps
    gy = torch.randn(shape, device="cuda", generator=g).to(dtype)
    return x, gy


def _run(fn, x, gy):
    xr = x.clone().requires_grad_(True)
    y = fn(xr)
    y.backward(gy)
    return y.detach(), xr.grad.detach()


@pytest.mark.parametrize("shape", SHAPES)
def test_fp32_matches_reference(shape):
    x, gy = _inputs(shape, torch.float32)
    y0, g0 = _run(_l2norm, x, gy)
    y1, g1 = _run(_fused(), x, gy)
    assert (y1 - y0).abs().max().item() < 1e-6
    assert ((g1 - g0).abs().max() / g0.abs().max()).item() < 1e-6


@pytest.mark.parametrize("shape", SHAPES)
def test_bf16_matches_reference_within_one_ulp(shape):
    x, gy = _inputs(shape, torch.bfloat16)
    y0, g0 = _run(_l2norm, x, gy)
    y1, g1 = _run(_fused(), x, gy)
    # y has no cancellation: every element within one bf16 ulp of the reference.
    a, b = y0.float(), y1.float()
    diff = (a - b).abs()
    assert (diff <= a.abs() * 2.0 ** -7).all(), "y: a difference exceeds one bf16 ulp"
    assert (diff > 0).float().mean().item() < 0.02, "y: too many elements differ"
    # dx = g/c - x*coef cancels, so near-zero elements carry large RELATIVE
    # error in any bf16 implementation; bound each row by its own scale instead.
    d = g0.shape[-1]
    a, b = g0.float().reshape(-1, d), g1.float().reshape(-1, d)
    row_scale = a.abs().amax(dim=1, keepdim=True)
    assert ((a - b).abs() <= row_scale * 2.0 ** -6).all(), "dx: beyond 2 ulp of its row scale"


@pytest.mark.parametrize("shape", SHAPES[:2])
def test_at_least_as_accurate_as_reference_vs_fp64(shape):
    x, gy = _inputs(shape, torch.bfloat16, seed=3)
    # Explicit fp64 math: _l2norm itself would downcast doubles via .float().
    def ref64(t):
        return t / t.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    y64, g64 = _run(ref64, x.double(), gy.double())
    assert y64.dtype == torch.float64
    y0, g0 = _run(_l2norm, x, gy)
    y1, g1 = _run(_fused(), x, gy)
    e_ref = ((g0.double() - g64).abs().mean().item(), (y0.double() - y64).abs().mean().item())
    e_fus = ((g1.double() - g64).abs().mean().item(), (y1.double() - y64).abs().mean().item())
    assert e_fus[0] <= e_ref[0] * 1.05 + 1e-12, f"dx less accurate than reference: {e_fus[0]} vs {e_ref[0]}"
    assert e_fus[1] <= e_ref[1] * 1.05 + 1e-12, f"y less accurate than reference: {e_fus[1]} vs {e_ref[1]}"


def test_backward_projection_term_is_exercised():
    """Anti-vacuity: the wrong gradient g/c must fail the same bound the fused
    gradient passes, on the same inputs."""
    x, gy = _inputs(SHAPES[1], torch.float32)
    _, g0 = _run(_l2norm, x, gy)
    c = x.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    wrong = gy / c
    rel = ((wrong - g0).abs().max() / g0.abs().max()).item()
    assert rel > 1e-2, f"inputs cannot distinguish the projection term (rel {rel:.2e})"


def test_zero_and_tiny_rows_follow_the_clamp():
    x, gy = _inputs(SHAPES[2], torch.float32, clamp_rows=True)
    y0, g0 = _run(_l2norm, x, gy)
    y1, g1 = _run(_fused(), x, gy)
    for r in (0, 1):
        a, b = y0.view(-1, x.shape[-1])[r], y1.view(-1, x.shape[-1])[r]
        assert torch.allclose(a, b, rtol=1e-6, atol=0), f"row {r} forward"
        a, b = g0.view(-1, x.shape[-1])[r], g1.view(-1, x.shape[-1])[r]
        assert torch.allclose(a, b, rtol=1e-6, atol=0), f"row {r} backward"
    assert torch.isfinite(g1).all()


def test_noncontiguous_input():
    x, gy = _inputs((2, 64, 8, 32), torch.bfloat16)
    xt = x.transpose(1, 2)                       # non-contiguous view
    y0, g0 = _run(_l2norm, xt, gy.transpose(1, 2))
    y1, g1 = _run(_fused(), xt, gy.transpose(1, 2))
    assert ((y0.float() - y1.float()).abs() <= y0.float().abs() * 2 ** -7).all()
    assert ((g0.float() - g1.float()).abs() <= g0.float().abs() * 2 ** -7 + 1e-30).all()


def test_deltanet_block_uses_fused_path_and_matches():
    """The training path (use_triton_scan) must route q/k through the fused
    kernel, and the block's output and gradients must match the reference
    path to bf16 tolerance."""
    import hvn.core.fused_l2norm as fl
    from hvn.core.deltanet import DeltaNetBlock

    torch.manual_seed(0)
    blk = DeltaNetBlock(256, n_heads=8, dropout=0.0, use_triton_scan=True,
                        use_megakernel_scan=True).cuda()
    x = torch.randn(1, 256, 256, device="cuda")

    def run(fused, autocast=True):
        blk.use_fused_l2norm = fused
        for p in blk.parameters():
            p.grad = None
        xr = x.clone().requires_grad_(True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            y = blk(xr)
        y.float().pow(2).mean().backward()
        return y.float().detach(), xr.grad.clone(), blk.q_proj.weight.grad.float().clone()

    calls = {"n": 0}
    orig = fl._FusedL2Norm.apply

    def counting(*a):
        calls["n"] += 1
        return orig(*a)

    truth = run(False, autocast=False)          # fp32 everywhere: the accuracy yardstick
    fl._FusedL2Norm.apply = counting
    try:
        ref = run(False)
        assert calls["n"] == 0
        got = run(True)
        assert calls["n"] == 2, f"expected q and k through the fused kernel, saw {calls['n']}"
    finally:
        fl._FusedL2Norm.apply = orig
    # The fused path must be no less accurate than the reference bf16 path.
    for t, a, b, what in zip(truth, ref, got, ("out", "dx", "dWq")):
        e_ref = (a - t).abs().mean().item()
        e_fus = (b - t).abs().mean().item()
        assert e_fus <= 1.1 * e_ref + 1e-12, f"{what}: fused err {e_fus:.3e} vs reference {e_ref:.3e}"
        assert e_ref > 0, f"{what}: reference equals fp32 truth -- yardstick is vacuous"
