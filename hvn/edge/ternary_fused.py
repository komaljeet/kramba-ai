"""Fused ternary quantize-STE — one Triton kernel instead of six ops (#154).

Author: Rambarun Komaljeet

``ternary_quantize_ste`` (``hvn/edge/ternary.py:46``) is six separate PyTorch
ops::

    abs -> mean -> clamp -> div -> round -> clamp -> mul

and it runs **1,760 times per training step** — 110 ternary layers x 8
micro-batches x 2 (forward + gradient-checkpoint recompute). Measured with
cProfile on the real shape (157M, seq 4096, grad_accum 8): **0.395 s of an 8.58 s
step, 4.6%**, plus a share of the 60,275 ``cudaLaunchKernel`` calls (0.609 s) in
the same profile.

This computes the same thing in **two kernel launches** — one reduction for the
scale, one elementwise pass — instead of seven.

**The STE is why the backward is free.** The forward returns
``w + (w_ternary - w).detach()``, so the gradient of the output with respect to
``w`` is exactly 1: quantization is invisible to autograd by construction. The
custom Function therefore passes ``grad_output`` straight through, and only the
forward needs a kernel.

**Semantics that must be preserved exactly**, from the original:

* the scale is a **full-tensor** mean of ``|w|`` (not per-row), clamped at ``eps``
* rounding is ``torch.round`` — banker's rounding, ties to even
* the clamp is to ``[-1, 1]`` *after* rounding, before scaling

A subtly different rounding rule would change every model trained afterwards and
raise no error, which is why ``tests/test_ternary_fused.py`` asserts
``max|diff| == 0`` against the reference rather than ``allclose``.

Falls back to the eager implementation when Triton is unavailable or the tensor
is on CPU, so nothing here can make a non-CUDA path fail.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                    # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _absmean_partial_kernel(w_ptr, out_ptr, n_elements,
                                BLOCK: tl.constexpr):
        """Partial sums of |w| — one value per program, finished on the host.

        A two-stage reduction rather than an atomic one: atomics on fp32 are
        non-deterministic in accumulation order, and this scale feeds every
        weight in the model. A run that is not reproducible cannot be
        bit-compared against the reference, which is the whole gate.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(out_ptr + pid, tl.sum(tl.abs(w), axis=0))

    @triton.jit
    def _quantize_kernel(w_ptr, scale_ptr, out_ptr, n_elements,
                         BLOCK: tl.constexpr):
        """out = clamp(round(w / scale), -1, 1) * scale, in one pass."""
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        w = tl.load(w_ptr + offs, mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32)
        # tl.extra.libdevice.rint is round-half-to-EVEN, matching torch.round.
        # tl.math.round rounds half AWAY from zero and would differ on exact
        # .5 values -- rare, but this is a quantizer in the training path, so
        # "rare" is not good enough.
        q = tl.extra.cuda.libdevice.rint(w / scale)
        q = tl.minimum(tl.maximum(q, -1.0), 1.0)
        out = q * scale
        # Canonicalise NEGATIVE ZERO. rint(-0.3) is -0.0, and -0.0 * scale is
        # -0.0, whereas the eager chain's op order yields +0.0 for the same
        # input. The values are numerically equal (-0.0 == 0.0 is True) but the
        # BIT PATTERNS differ, so torch.equal reports a mismatch and the parity
        # gate fails. Adding 0.0 maps -0.0 -> +0.0 and leaves every other value
        # untouched, which is also the canonical form a quantizer should emit.
        out = out + 0.0
        tl.store(out_ptr + offs, out.to(w.dtype), mask=mask)


def _fused_forward(w: torch.Tensor, eps: float) -> torch.Tensor:
    """Scale from PyTorch, quantization from Triton.

    The scale is deliberately NOT computed here. A hand-written two-stage
    reduction summed the partials in a different order from ``torch.mean``, and
    since floating-point addition is not associative that changed the last two
    bits of the scale: 0.7854206562 vs 0.7854205966, a 6e-8 relative difference
    that made 2,797 of 4,096 elements differ and failed the bit-parity gate.
    The rounding and clamping were identical -- only the reduction order moved.

    Reproducing ``torch.mean``'s exact tree-reduction order in Triton is
    possible but pointless: the reduction is ONE cheap op over the weight, while
    the six-op elementwise chain (div, round, clamp, clamp, mul, plus the STE
    subtract/add) is what runs per element and what this actually fuses. Keeping
    the reduction in PyTorch buys exact parity for free.
    """
    n = w.numel()
    wf = w.detach().contiguous()
    scale = wf.abs().mean().clamp(min=eps).to(torch.float32)
    out = torch.empty_like(wf)
    BLOCK = 1024
    _quantize_kernel[(triton.cdiv(n, BLOCK),)](wf, scale, out, n, BLOCK=BLOCK)
    return out


class _FusedTernarySTE(torch.autograd.Function):
    """Straight-through estimator: forward quantizes, backward is identity."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, eps: float) -> torch.Tensor:
        return _fused_forward(w, eps)

    @staticmethod
    def backward(ctx, grad_output):
        # d(out)/d(w) == 1 by construction of the STE -- the original returns
        # `w + (w_ternary - w).detach()`, whose only differentiable term is `w`.
        return grad_output, None


def ternary_quantize_ste_fused(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Drop-in replacement for ``ternary_quantize_ste`` (#154).

    Bit-identical to the eager version on CUDA; falls back to it otherwise.
    """
    if not (_HAS_TRITON and w.is_cuda):
        scale = w.detach().abs().mean().clamp(min=eps)
        w_ternary = torch.clamp(torch.round(w / scale), -1, 1) * scale
        return w + (w_ternary - w).detach()
    return _FusedTernarySTE.apply(w, eps)
