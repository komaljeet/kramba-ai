"""Known-answer tests for group-wise ternary scales (Issue #146).

Author: Rambarun Komaljeet

The load-bearing test is the first one: ``group_size=0`` must be **bit-identical**
to the pre-#146 implementation, in forward *and* backward. Every existing
checkpoint was trained under per-tensor scales, so a default that drifted even
slightly would silently change every model in the repo.

Also deliberately present: ``test_grouped_modes_are_not_secretly_per_tensor``.
If ``group_size`` stopped being threaded from ``apply_ternary_to_model`` down to
``ternary_quantize_ste``, the identity test above would still pass and every arm
of the A/B would silently measure per-tensor. That test exists to fail there.

Run: ``python -m pytest tests/test_ternary_grouping.py -q``
"""

import math

import pytest
import torch
import torch.nn as nn

from hvn.edge.ternary import (
    GROUP_PER_ROW, GROUP_PER_TENSOR, TernaryLinear, apply_ternary_to_model,
    dequantize_codes, scale_count, ternary_codes, ternary_quantize_ste,
)


def _legacy_quantize(w, eps=1e-6):
    """The pre-#146 implementation, inlined verbatim as the reference."""
    scale = w.detach().abs().mean().clamp(min=eps)
    w_ternary = torch.clamp(torch.round(w / scale), -1, 1) * scale
    return w + (w_ternary - w).detach()


# ----------------------------------------------------------------------
# 1. The known-answer cell.
# ----------------------------------------------------------------------
def test_per_tensor_is_bit_identical_to_the_legacy_implementation():
    torch.manual_seed(0)
    w = torch.randn(64, 128, dtype=torch.float64)
    assert torch.equal(ternary_quantize_ste(w), _legacy_quantize(w))


def test_per_tensor_backward_is_bit_identical():
    """The STE gradient must be untouched too -- forward parity alone would let a
    changed backward through, and that is what actually trains the model."""
    torch.manual_seed(0)
    w0 = torch.randn(32, 64, dtype=torch.float64, requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)
    up = torch.randn(32, 64, dtype=torch.float64)

    (ternary_quantize_ste(w0) * up).sum().backward()
    (_legacy_quantize(w1) * up).sum().backward()
    assert torch.equal(w0.grad, w1.grad)


def test_default_group_size_is_per_tensor():
    """The default must be the incumbent -- not merely equivalent today."""
    assert GROUP_PER_TENSOR == 0
    assert TernaryLinear(8, 8).group_size == GROUP_PER_TENSOR
    m = nn.Sequential(nn.Linear(8, 8))
    assert apply_ternary_to_model(m)["0"].group_size == GROUP_PER_TENSOR


# ----------------------------------------------------------------------
# 2. Anti-vacuity: the grouped modes must actually reach the quantizer.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("group_size", [GROUP_PER_ROW, 32, 16])
def test_grouped_modes_are_not_secretly_per_tensor(group_size):
    """If group_size stopped being threaded through, everything else still passes.

    This fails instead.
    """
    torch.manual_seed(1)
    w = torch.randn(16, 64, dtype=torch.float64)
    # Rows deliberately given very different magnitudes -- the exact case a
    # single shared scale handles badly and a grouped scale is meant to fix.
    w[0] *= 100.0
    w[-1] *= 0.01
    d = (ternary_quantize_ste(w, group_size=group_size)
         - ternary_quantize_ste(w)).abs().max()
    assert d > 0, f"group_size={group_size} produced per-tensor output"


def test_apply_ternary_threads_group_size_to_the_layers():
    m = nn.Sequential(nn.Linear(64, 32), nn.Linear(32, 32))
    layers = apply_ternary_to_model(m, group_size=16)
    assert [l.group_size for l in layers.values()] == [16, 16]


# ----------------------------------------------------------------------
# 3. The grouped maths itself.
# ----------------------------------------------------------------------
def test_per_row_uses_one_scale_per_output_row():
    torch.manual_seed(2)
    w = torch.randn(8, 32, dtype=torch.float64)
    q = ternary_quantize_ste(w, group_size=GROUP_PER_ROW)
    for r in range(8):
        expected = w[r].abs().mean().clamp(min=1e-6)
        nz = q[r][q[r] != 0].abs()
        assert torch.allclose(nz, expected.expand_as(nz), atol=1e-12)


def test_group_size_matches_a_manual_per_group_computation():
    torch.manual_seed(3)
    w = torch.randn(4, 32, dtype=torch.float64)
    g = 8
    q = ternary_quantize_ste(w, group_size=g)
    for r in range(4):
        for gi in range(32 // g):
            sl = slice(gi * g, (gi + 1) * g)
            s = w[r, sl].abs().mean().clamp(min=1e-6)
            expect = torch.clamp(torch.round(w[r, sl] / s), -1, 1) * s
            assert torch.allclose(q[r, sl], expect, atol=1e-12)


def test_finer_scales_reduce_reconstruction_error():
    """Strictly better approximation -- true by construction, so a violation
    means the implementation is wrong. (#145 measured that this does NOT imply
    a better model, which is exactly why it is only asserted here.)"""
    torch.manual_seed(4)
    w = torch.randn(32, 128, dtype=torch.float64)
    w[0] *= 50.0
    err = lambda gs: (ternary_quantize_ste(w, group_size=gs) - w).pow(2).sum().item()
    assert err(16) < err(32) < err(GROUP_PER_ROW) < err(GROUP_PER_TENSOR)


def test_codes_roundtrip_through_dequantize_at_every_granularity():
    torch.manual_seed(5)
    w = torch.randn(16, 64, dtype=torch.float64)
    for gs in (GROUP_PER_TENSOR, GROUP_PER_ROW, 8, 32):
        codes, scale = ternary_codes(w, group_size=gs)
        assert set(codes.unique().tolist()) <= {-1, 0, 1}
        rebuilt = dequantize_codes(codes, scale, gs, torch.float64)
        assert torch.allclose(rebuilt, ternary_quantize_ste(w, group_size=gs),
                              atol=1e-12)


# ----------------------------------------------------------------------
# 4. Storage — a finer scheme must not be able to look free.
# ----------------------------------------------------------------------
def test_scale_count_matches_the_granularity():
    assert scale_count(512, 512, GROUP_PER_TENSOR) == 1
    assert scale_count(512, 512, GROUP_PER_ROW) == 512
    assert scale_count(512, 512, 128) == 512 * 4
    assert scale_count(512, 512, 32) == 512 * 16


def test_memory_bytes_charges_for_the_extra_scales():
    """A quality win at 32 groups costs ~63% more storage at d=512. If
    memory_bytes() kept counting one scalar, every grouped scheme would look
    free and the A/B's verdict would be wrong in the model's favour."""
    per_tensor = TernaryLinear(512, 512).memory_bytes()
    per_row = TernaryLinear(512, 512, group_size=GROUP_PER_ROW).memory_bytes()
    g32 = TernaryLinear(512, 512, group_size=32).memory_bytes()
    assert per_tensor < per_row < g32
    codes_only = 512 * 512 * math.log2(3) / 8.0
    assert g32 - per_tensor == pytest.approx(512 * 16 * 4 - 4, rel=1e-6)
    assert g32 / codes_only > 1.5  # the +63% that decides the tradeoff


# ----------------------------------------------------------------------
# 5. Guard rails.
# ----------------------------------------------------------------------
def test_non_dividing_group_size_raises_rather_than_padding():
    with pytest.raises(ValueError, match="does not divide"):
        ternary_quantize_ste(torch.randn(4, 30), group_size=16)
    with pytest.raises(ValueError, match="does not divide"):
        TernaryLinear(30, 4, group_size=16)


def test_state_dict_is_unchanged_by_group_size():
    """Scales are derived, never stored, so a grouped model must still load a
    per-tensor checkpoint strict=True -- otherwise the A/B arms would not be
    swappable and no existing checkpoint could be re-quantized at all."""
    a = TernaryLinear(64, 32)
    b = TernaryLinear(64, 32, group_size=16)
    missing, unexpected = b.load_state_dict(a.state_dict(), strict=True), None
    assert set(a.state_dict()) == set(b.state_dict())


def test_forward_runs_and_grads_flow_at_every_granularity():
    for gs in (GROUP_PER_TENSOR, GROUP_PER_ROW, 16):
        layer = TernaryLinear(64, 32, group_size=gs)
        out = layer(torch.randn(4, 64))
        out.sum().backward()
        assert layer.weight.grad is not None
        assert torch.isfinite(layer.weight.grad).all()


def test_freeze_then_infer_matches_the_live_path():
    """#131's frozen/cached inference path must agree with the training path at
    every granularity -- a mismatch there is silent staleness, the #117 bug."""
    torch.manual_seed(6)
    for gs in (GROUP_PER_TENSOR, GROUP_PER_ROW, 16):
        layer = TernaryLinear(64, 32, group_size=gs).eval()
        x = torch.randn(3, 64)
        with torch.no_grad():
            before = layer(x).clone()
        layer.freeze()
        layer.invalidate_cache()
        with torch.no_grad():
            after = layer(x)
        assert torch.allclose(before, after, atol=1e-6), f"group_size={gs}"
