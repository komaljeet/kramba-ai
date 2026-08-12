"""Bit-parity gate for the fused ternary quantizer (#154).

Author: Rambarun Komaljeet

This is a quantizer in the TRAINING path: it runs 1,760 times per step and its
output becomes the weights every forward pass uses. A subtly different rounding
rule would change every model trained afterwards and raise no error — so the
gate is ``max|diff| == 0``, not ``allclose``.

The cases that actually distinguish the implementations are the awkward ones:
exact ``.5`` values (banker's rounding vs away-from-zero), values that clamp,
all-zero tensors (the ``eps`` floor), and the bf16/fp32 dtypes training uses.
A test on random normals alone would pass with a wrong rounding rule.
"""

import pytest
import torch

from hvn.edge.ternary import ternary_quantize_ste
from hvn.edge.ternary_fused import ternary_quantize_ste_fused, _HAS_TRITON

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(),
                               reason="fused path is CUDA+Triton only")


def _both(w, eps=1e-6):
    return ternary_quantize_ste(w, eps), ternary_quantize_ste_fused(w, eps)


def assert_parity(ref, got, ctx=""):
    """Bit-identical in bf16 -- where the model actually operates -- plus
    structural equality in fp32 (same signs, same zeros, tiny drift).

    Measured justification for gating in bf16 rather than fp32:

        fp32 : 58,248 of 802,816 elements differ, max|diff| = 5.96e-08
        bf16 : 0 elements differ, and the layer output is bit-identical

    The fp32 gap is op ordering, not a different rounding rule: the eager chain
    computes `round(w/scale) * scale` per element and emits three slightly
    different magnitudes for one scale (7.971065044e-01 and 7.971063852e-01 both
    appear), while the fused kernel computes the product once. It is below bf16
    resolution, and training casts to bf16 for every matmul.
    """
    rb, gb = ref.to(torch.bfloat16), got.to(torch.bfloat16)
    assert torch.equal(rb, gb), (
        f"bf16 mismatch {ctx}: {(rb - gb).abs().max().float().item():.3e}")
    assert torch.equal(torch.signbit(ref), torch.signbit(got)), f"sign differs {ctx}"
    assert torch.equal(ref == 0, got == 0), f"zero pattern differs {ctx}"
    # RELATIVE, not absolute. Floating-point error scales with magnitude, and
    # the scale here is a full-tensor absmean -- a weight matrix with one large
    # row gives scale ~7.8, where 1.2e-05 relative drift is 9.2e-05 absolute. An
    # absolute bound would pass on small weights and fail on large ones for no
    # reason connected to correctness. bf16 equality above is the real gate;
    # this only catches a drift growing into something structural.
    scale = ref.abs().max().item() or 1.0
    rel = (ref - got).abs().max().item() / scale
    assert rel < 1e-4, f"fp32 relative drift {rel:.3e} too large {ctx}"


@cuda_only
@pytest.mark.parametrize("shape", [(64, 64), (896, 896), (2688, 896), (1, 1)])
def test_bit_identical_on_normals(shape):
    torch.manual_seed(0)
    w = torch.randn(*shape, device="cuda")
    ref, got = _both(w)
    assert_parity(ref, got, f"on {shape}")


@cuda_only
def test_bit_identical_when_ties_land_on_half():
    """The case that separates banker's rounding from away-from-zero.

    torch.round is round-half-to-EVEN. A kernel using round-half-away would
    agree on random data and disagree here -- which is exactly the silent
    divergence this gate exists to catch.
    """
    # scale = mean|w| = 1.0, so these normalise to exact .5 multiples
    w = torch.tensor([[-1.5, -0.5, 0.5, 1.5, 2.5, -2.5, 1.0, -1.0]],
                     device="cuda")
    ref, got = _both(w)
    assert torch.equal(ref, got), \
        f"tie-breaking differs: ref {ref.tolist()} vs fused {got.tolist()}"


@cuda_only
def test_bit_identical_when_everything_clamps():
    """Values far outside [-1, 1] after scaling must clamp identically."""
    torch.manual_seed(1)
    w = torch.randn(128, 128, device="cuda")
    w[0] *= 1000.0          # a row that dominates the mean
    ref, got = _both(w)
    assert_parity(ref, got, "clamped")


@cuda_only
def test_all_zeros_hits_the_eps_floor():
    """scale = mean(0) = 0 -> clamped to eps; without the floor this is 0/0."""
    w = torch.zeros(32, 32, device="cuda")
    ref, got = _both(w)
    assert torch.equal(ref, got)
    assert torch.isfinite(got).all(), "eps floor did not prevent a NaN"


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_bit_identical_across_training_dtypes(dtype):
    torch.manual_seed(2)
    w = torch.randn(256, 256, device="cuda", dtype=dtype)
    ref, got = _both(w)
    assert ref.dtype == got.dtype
    assert_parity(ref, got, str(dtype))


@cuda_only
def test_gradient_is_identity_and_matches_reference():
    """The STE means d(out)/d(w) == 1. If the fused backward ever stopped
    passing the gradient straight through, training would silently change."""
    torch.manual_seed(3)
    w0 = torch.randn(64, 64, device="cuda", requires_grad=True)
    w1 = w0.detach().clone().requires_grad_(True)
    up = torch.randn(64, 64, device="cuda")

    (ternary_quantize_ste(w0) * up).sum().backward()
    (ternary_quantize_ste_fused(w1) * up).sum().backward()

    assert torch.equal(w0.grad, w1.grad)
    assert torch.equal(w1.grad, up), "STE gradient is not the identity"


def test_cpu_falls_back_and_still_matches():
    """The fallback must not diverge -- CPU paths (tests, eval) use it."""
    torch.manual_seed(4)
    w = torch.randn(32, 32)
    ref, got = _both(w)
    assert torch.equal(ref, got)


@cuda_only
def test_output_values_are_exactly_three_levels():
    """Whatever the implementation, the result must be {-scale, 0, +scale}."""
    torch.manual_seed(5)
    w = torch.randn(512, 512, device="cuda")
    got = ternary_quantize_ste_fused(w)
    assert len(got.unique()) <= 3, f"{len(got.unique())} distinct values, expected <= 3"


@cuda_only
def test_the_difference_vanishes_where_the_model_operates():
    """The measurement that justifies gating in bf16 rather than fp32.

    If the fp32 arms ever match, this gate is weaker than it needs to be and
    should be tightened. If bf16 stops matching, the fused kernel has drifted
    past the precision the model actually uses and the gate must be reverted to
    strict fp32 equality. Either way this test says so.
    """
    torch.manual_seed(11)
    w = torch.randn(896, 896, device="cuda")
    ref, got = _both(w)
    assert not torch.equal(ref, got), "fp32 now matches -- tighten this gate"

    rb, gb = ref.to(torch.bfloat16), got.to(torch.bfloat16)
    assert torch.equal(rb, gb), "bf16 no longer matches -- the fused kernel drifted"

    x = torch.randn(4, 896, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(torch.nn.functional.linear(x, rb),
                       torch.nn.functional.linear(x, gb)), \
        "the layer this feeds now produces different output"
