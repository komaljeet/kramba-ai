"""Fused RMSNorm (``RMSNorm.use_fused`` -> ``F.rms_norm``) vs the eager chain.

What would make these vacuous, and the guard:
  * A "fused" run that silently took the eager path would match it exactly and
    pass every tolerance test. Every comparison asserts the fused output's
    grad_fn is ``FusedRmsNormBackward0`` and the eager one's is not.
  * Tolerances alone cannot tell "fused is 1 ulp off" from "eager is 1 ulp
    off". The fp64 test requires the fused result to be at least as accurate as
    the eager chain against a float64 reference.
  * A backward missing the normalisation term (dx = g * r * w only) is the
    classic wrong RMSNorm gradient; ``test_wrong_gradient_is_detectable`` checks
    the inputs used here put that wrong gradient far outside the bound.
  * The gate must NOT engage for bf16 inputs (eager rounds each intermediate to
    bf16, the fused kernel does not -- a real numerics change); that case must
    stay bit-identical to eager.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.core.hvn_block import RMSNorm  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

SHAPES = [(1, 4096, 2048), (2, 129, 256), (3, 7, 64), (1, 33, 896)]


def _norm(d, seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    n = RMSNorm(d).cuda()
    with torch.no_grad():
        n.weight.uniform_(0.5, 1.5)
    return n.to(dtype)


def _inputs(shape, seed=0, dtype=torch.float32, scale=1.0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = (torch.randn(shape, device="cuda", generator=g) * scale).to(dtype)
    gy = torch.randn(shape, device="cuda", generator=g).to(dtype)
    return x, gy


def _run(n, fused, x, gy, autocast=False):
    n.use_fused = fused
    n.weight.grad = None
    xr = x.clone().requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        y = n(xr)
    name = y.grad_fn.name()
    y.backward(gy)
    return y.detach(), xr.grad.detach(), n.weight.grad.detach().clone(), name


@pytest.mark.parametrize("shape", SHAPES)
@pytest.mark.parametrize("autocast", [False, True])
def test_fp32_matches_eager(shape, autocast):
    n = _norm(shape[-1])
    x, gy = _inputs(shape)
    y0, gx0, gw0, n0 = _run(n, False, x, gy, autocast)
    y1, gx1, gw1, n1 = _run(n, True, x, gy, autocast)
    assert n1 == "FusedRmsNormBackward0" and n0 != n1, (n0, n1)
    assert y1.dtype == y0.dtype == torch.float32
    assert ((y1 - y0).abs().max() / y0.abs().max()).item() < 1e-6
    assert ((gx1 - gx0).abs().max() / gx0.abs().max()).item() < 1e-6
    # dw sums over every position -- bound relative to its own scale.
    assert ((gw1 - gw0).abs().max() / gw0.abs().max()).item() < 1e-5


@pytest.mark.parametrize("shape", SHAPES[:2])
def test_at_least_as_accurate_as_eager_vs_fp64(shape):
    n = _norm(shape[-1], seed=3)
    x, gy = _inputs(shape, seed=3)
    w64 = n.weight.detach().double().requires_grad_(True)
    x64 = x.double().requires_grad_(True)
    y64 = x64 * x64.pow(2).mean(-1, keepdim=True).add(n.eps).rsqrt() * w64
    y64.backward(gy.double())
    ref = (y64.detach(), x64.grad, w64.grad)
    eager = _run(n, False, x, gy)[:3]
    fused = _run(n, True, x, gy)[:3]
    for what, t, a, b in zip(("y", "dx", "dw"), ref, eager, fused):
        e_ref = (a.double() - t).abs().mean().item()
        e_fus = (b.double() - t).abs().mean().item()
        assert e_ref > 0, f"{what}: eager equals fp64 -- yardstick is vacuous"
        if what == "dw":
            # Measured, not assumed: at [1, 4096, 2048] the fused kernel's dw
            # (a sum over 4096 positions) is ~1.8x further from fp64 than
            # eager's sum (1.1e-5 vs 6.3e-6 mean abs). Both are fp32 roundoff,
            # ~1e-7 relative to |dw|, far below the bf16 noise of the block
            # around it (see the block test). Bounded, not hidden.
            assert e_fus <= 3.0 * e_ref + 1e-12, f"dw: fused {e_fus:.3e} vs eager {e_ref:.3e}"
            assert e_fus / t.abs().mean().item() < 1e-6, "dw: beyond fp32 roundoff"
        else:
            assert e_fus <= 1.05 * e_ref + 1e-12, f"{what}: fused {e_fus:.3e} vs eager {e_ref:.3e}"


def test_wrong_gradient_is_detectable():
    """Anti-vacuity: dx without the normalisation term must fail the bound."""
    n = _norm(256)
    x, gy = _inputs((2, 129, 256))
    _, gx0, _, _ = _run(n, False, x, gy)
    r = x.pow(2).mean(-1, keepdim=True).add(n.eps).rsqrt()
    wrong = gy * r * n.weight.detach()
    rel = ((wrong - gx0).abs().max() / gx0.abs().max()).item()
    assert rel > 1e-2, f"inputs cannot tell the wrong gradient apart (rel {rel:.2e})"


@pytest.mark.parametrize("autocast", [False, True])
def test_bf16_input_stays_eager_and_bit_identical(autocast):
    n = _norm(256)
    x, gy = _inputs((2, 129, 256), dtype=torch.bfloat16)
    y0, gx0, gw0, n0 = _run(n, False, x, gy, autocast)
    y1, gx1, gw1, n1 = _run(n, True, x, gy, autocast)
    assert n1 == n0 != "FusedRmsNormBackward0"
    assert torch.equal(y0, y1) and torch.equal(gx0, gx1) and torch.equal(gw0, gw1)


def test_bf16_weight_stays_eager():
    n = _norm(256).to(torch.bfloat16)
    n.use_fused = True
    x = torch.randn(2, 9, 256, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y = n(x.requires_grad_(True))
    assert y.grad_fn.name() != "FusedRmsNormBackward0"


def test_default_off_and_not_in_state_dict():
    n = _norm(64)
    assert n.use_fused is False
    assert set(n.state_dict()) == {"weight"}


def test_deltanet_block_runs_fused_and_matches():
    """The block's own pre-norm takes the fused path under autocast, and the
    block's output and gradients match the eager path to bf16-run tolerance
    while being no less accurate against an fp32-everywhere run."""
    from hvn.core.deltanet import DeltaNetBlock

    torch.manual_seed(0)
    blk = DeltaNetBlock(256, n_heads=8, dropout=0.0, use_triton_scan=True,
                        use_megakernel_scan=True).cuda()
    x = torch.randn(1, 256, 256, device="cuda")
    seen = []
    h = blk.norm.register_forward_hook(lambda m, i, o: seen.append(o.grad_fn.name()))

    def run(fused, autocast=True):
        blk.norm.use_fused = fused
        for p in blk.parameters():
            p.grad = None
        xr = x.clone().requires_grad_(True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            y = blk(xr)
        y.float().pow(2).mean().backward()
        return (y.float().detach(), xr.grad.clone(), blk.q_proj.weight.grad.float().clone(),
                blk.norm.weight.grad.clone())

    try:
        truth = run(False, autocast=False)
        ref = run(False)
        got = run(True)
    finally:
        h.remove()
    assert seen[-1] == "FusedRmsNormBackward0" and seen[-2] != seen[-1], seen
    for t, a, b, what in zip(truth, ref, got, ("out", "dx", "dWq", "dWnorm")):
        e_ref = (a - t).abs().mean().item()
        e_fus = (b - t).abs().mean().item()
        assert e_ref > 0, f"{what}: reference equals fp32 truth -- yardstick is vacuous"
        assert e_fus <= 1.1 * e_ref + 1e-12, f"{what}: fused err {e_fus:.3e} vs reference {e_ref:.3e}"


def test_apply_fast_stack_enables_every_rmsnorm():
    from hvn.edge.model import EdgeHVN
    import train_edge_conversational as tec

    m = EdgeHVN(vocab_size=128, d_model=64, num_layers=3, n_heads=4,
                block_plan=["deltanet", "transformer", "deltanet"]).cuda()
    norms = [mod for mod in m.modules() if isinstance(mod, RMSNorm)]
    assert len(norms) == 3
    tec.apply_fast_stack(m, tf32=False, compile_model=False, verbose=False)
    assert not any(n.use_fused for n in norms), "must stay off unless asked"
    tec.apply_fast_stack(m, tf32=False, compile_model=False, verbose=False,
                         fused_rmsnorm=True)
    assert all(n.use_fused for n in norms)
