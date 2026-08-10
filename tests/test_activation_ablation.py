"""Known-answer tests for the pluggable FFN activation (Issue #140).

Author: Rambarun Komaljeet

The point of these tests is the *first* one: ``ffn_activation="gelu"`` must be
bit-identical to the model built the old way.  Everything downstream -- the
throughput comparison, any later quality run -- is only meaningful if the harness
can reproduce the incumbent exactly.  This is the same discipline as the
``num_experts=1`` MoE known-answer (docs/architecture.md).

Deliberately included: ``test_arms_are_not_secretly_gelu``.  A registry that fell
back to GELU on an unknown name, or plumbing that dropped ``ffn_activation``
somewhere between EdgeHVN and the block, would make the identity test above pass
*and* make every arm silently measure GELU -- a green suite measuring nothing.
That test exists to fail in exactly that case.

CPU-only, no checkpoint needed.  Run: ``python -m pytest tests/test_activation_ablation.py -q``
"""

import torch
import torch.nn as nn
import pytest

from hvn.core.activations import (
    make_activation, ReLU2, PowLU, ACTIVATION_NAMES,
)
from hvn.core.deltanet import DeltaNetBlock
from hvn.edge.model import EdgeHVN


def _tiny_model(ffn_activation="gelu", seed=0):
    torch.manual_seed(seed)
    return EdgeHVN(
        vocab_size=64, d_model=32, num_layers=2, n_heads=4,
        use_deltanet=True, dropout=0.0,
        block_plan=["deltanet", "transformer"], attn_window=16,
        ffn_activation=ffn_activation,
    ).eval()


# ----------------------------------------------------------------------
# 1. The known-answer cell: gelu == the pre-#140 path, exactly.
# ----------------------------------------------------------------------
def test_gelu_is_bit_identical_to_the_incumbent():
    """max|dlogits| must be exactly 0.0 -- not 'small', zero."""
    ref = _tiny_model(seed=0)
    # Rebuild the incumbent FFN the way the block did before #140 and splice it
    # in, so this compares against the literal old construction, not against a
    # second call to the same new code path (which would be circular).
    for blk in ref.blocks:
        if isinstance(blk, DeltaNetBlock):
            blk.ffn[1] = nn.GELU()

    new = _tiny_model(ffn_activation="gelu", seed=0)
    new.load_state_dict(ref.state_dict(), strict=True)

    x = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        d = (ref(x) - new(x)).abs().max().item()
    assert d == 0.0, f"gelu path is not the incumbent: max|dlogits|={d:.3e}"


def test_gelu_state_dict_loads_strict_both_ways():
    """A gelu checkpoint and a relu2 model share every shape and key.

    That is what makes the arms warm-startable -- and also why EdgeHVN records
    ffn_activation: nothing in the state dict would otherwise reveal the swap.
    """
    gelu_sd = _tiny_model("gelu", seed=1).state_dict()
    relu2 = _tiny_model("relu2", seed=2)
    missing, unexpected = relu2.load_state_dict(gelu_sd, strict=False)
    assert not missing and not unexpected, (missing, unexpected)


def test_powlu_adds_exactly_one_scalar_per_ffn():
    """PowLU's only new parameter is p -- one scalar per DeltaNet FFN."""
    gelu_keys = set(_tiny_model("gelu", seed=1).state_dict())
    powlu = _tiny_model("powlu", seed=1)
    extra = set(powlu.state_dict()) - gelu_keys
    n_deltanet = sum(isinstance(b, DeltaNetBlock) for b in powlu.blocks)
    assert len(extra) == n_deltanet, extra
    for k in extra:
        assert k.endswith("ffn.1.p")
        assert powlu.state_dict()[k].numel() == 1


# ----------------------------------------------------------------------
# 2. Anti-vacuity: prove the arms actually differ from the control.
# ----------------------------------------------------------------------
@pytest.mark.parametrize("name", ["relu2", "powlu"])
def test_arms_are_not_secretly_gelu(name):
    """If plumbing dropped ffn_activation, every test above still passes.

    This one fails instead. Same weights, same input, different activation ->
    the logits MUST differ by a visible margin.
    """
    ref = _tiny_model("gelu", seed=3)
    arm = _tiny_model(name, seed=3)
    arm.load_state_dict(ref.state_dict(), strict=False)

    x = torch.randint(0, 64, (2, 24))
    with torch.no_grad():
        d = (ref(x) - arm(x)).abs().max().item()
    assert d > 1e-4, (
        f"{name} produced logits within {d:.3e} of gelu with identical weights "
        f"-- the activation is probably not reaching the FFN"
    )


def test_unknown_activation_raises_rather_than_defaulting():
    """A silent fallback to gelu would make a typo'd arm measure the control."""
    with pytest.raises(ValueError, match="unknown ffn_activation"):
        make_activation("gelu2")  # plausible typo
    with pytest.raises(ValueError):
        DeltaNetBlock(d_model=32, n_heads=4, ffn_activation="swish")


def test_activation_is_rejected_where_it_would_be_ignored():
    """SwiGLU/MoE own their nonlinearity; asking for another must not be a no-op."""
    with pytest.raises(ValueError, match="only supported by the plain FFN"):
        DeltaNetBlock(d_model=32, n_heads=4, use_swiglu=True, ffn_activation="relu2")
    with pytest.raises(ValueError, match="only supported by the plain FFN"):
        DeltaNetBlock(d_model=32, n_heads=4, moe_experts=4, ffn_activation="relu2")


# ----------------------------------------------------------------------
# 3. The activations themselves.
# ----------------------------------------------------------------------
def test_relu2_matches_its_definition():
    x = torch.randn(500, dtype=torch.float64)
    expected = torch.clamp(x, min=0.0) ** 2
    assert torch.equal(ReLU2()(x), expected)


def test_powlu_matches_its_definition_away_from_zero():
    x = torch.randn(500, dtype=torch.float64)
    x = x[x.abs() > 1e-3]
    act = PowLU(p_init=1.5).double()
    expected = torch.sign(x) * x.abs() ** 1.5
    assert torch.allclose(act(x), expected, atol=1e-12)


def test_powlu_is_odd():
    x = torch.randn(200, dtype=torch.float64)
    act = PowLU(p_init=1.5).double()
    assert torch.allclose(act(-x), -act(x), atol=1e-12)


def test_powlu_gradient_is_finite_at_exactly_zero():
    """The eps clamp is load-bearing: d/dp of |x|^p contains log|x|.

    Without the clamp this test NaNs -- which is precisely the failure that would
    otherwise show up 40 minutes into a training run, not here.
    """
    act = PowLU(p_init=1.5)
    x = torch.zeros(16, requires_grad=True)
    act(x).sum().backward()
    assert torch.isfinite(x.grad).all(), "input grad not finite at x=0"
    assert torch.isfinite(act.p.grad).all(), "exponent grad not finite at x=0"


def test_powlu_exponent_receives_gradient():
    """p must actually be learnable -- a detached p would make this arm 'PowLU
    with a frozen 1.5', a different experiment than the one reported."""
    act = PowLU(p_init=1.5)
    x = torch.randn(64) * 2.0
    act(x).pow(2).sum().backward()
    assert act.p.grad is not None and act.p.grad.abs().item() > 0.0


def test_registry_covers_exactly_the_three_arms():
    assert set(ACTIVATION_NAMES) == {"gelu", "relu2", "powlu"}
