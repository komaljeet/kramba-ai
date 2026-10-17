"""Two small bit-identical fusions in DeltaNetBlock's forward (2026-10-17).

Author: Rambarun Komaljeet

1. ``fused_sigmoid_f32(h)`` == ``torch.sigmoid(h.float())`` for a 16-bit ``h``
   (the alpha / beta gates, Issue #44 keeps them fp32). Eager is two passes:
   the cast writes an fp32 copy, sigmoid reads it and writes another; backward
   writes an fp32 ``sigmoid_backward`` result and then casts it to bf16. Here:
   one pass each way.

   Same arithmetic as ATen, which is what makes it bit-identical:
   forward ``1 / (1 + exp(-x))`` with the full-precision libdevice ``exp``
   (NOT ``tl.exp``, which lowers to the approximate ``ex2.approx``) and an
   IEEE round-to-nearest divide (``div_rn``, not Triton's approximate ``/`` --
   the #154 lesson); backward ``(g * (1 - y)) * y``, then the cast.
   Forward is checked over ALL 65,536 bf16 inputs (tests/test_fused_gates.py).

2. ``fused_residual_add3(x, d, f)`` == ``x + d + f`` for an fp32 residual
   ``x`` and 16-bit branch outputs ``d``, ``f``: the parallel residual. Eager
   is two fp32 passes (each re-reading the fp32 intermediate) and, backward,
   two separate fp32 -> bf16 casts of the same gradient. Same order of fp32
   adds here, ``(x + d) + f``; backward writes both casts from one read.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

_HALF = (torch.bfloat16, torch.float16)
#: Forward calls taken by each fast path (tests assert these move: a fast
#: path that silently fell back would otherwise compare eager to eager).
STATS = {"sigmoid": 0, "add3": 0}
_BLOCK = 1024

if _HAS_TRITON:

    @triton.jit
    def _sig_fwd_kernel(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        e = tl.extra.cuda.libdevice.exp(-x)
        y = tl.div_rn(1.0, 1.0 + e)
        tl.store(y_ptr + offs, y, mask=m)

    @triton.jit
    def _sig_bwd_kernel(g_ptr, y_ptr, dx_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(g_ptr + offs, mask=m, other=0.0)
        y = tl.load(y_ptr + offs, mask=m, other=0.0)
        dx = g * (1.0 - y) * y
        tl.store(dx_ptr + offs, dx.to(dx_ptr.dtype.element_ty), mask=m)

    @triton.jit
    def _add3_fwd_kernel(x_ptr, d_ptr, f_ptr, o_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0)
        d = tl.load(d_ptr + offs, mask=m, other=0.0).to(tl.float32)
        f = tl.load(f_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(o_ptr + offs, (x + d) + f, mask=m)

    @triton.jit
    def _cast2_kernel(g_ptr, a_ptr, b_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(g_ptr + offs, mask=m, other=0.0)
        tl.store(a_ptr + offs, g.to(a_ptr.dtype.element_ty), mask=m)
        tl.store(b_ptr + offs, g.to(b_ptr.dtype.element_ty), mask=m)


class _FusedSigmoidF32(torch.autograd.Function):
    @staticmethod
    def forward(ctx, h):
        STATS["sigmoid"] += 1
        hc = h.contiguous()
        y = torch.empty(hc.shape, device=hc.device, dtype=torch.float32)
        n = hc.numel()
        _sig_fwd_kernel[(triton.cdiv(n, _BLOCK),)](hc, y, n, BLOCK=_BLOCK)
        ctx.save_for_backward(y)
        ctx.h_dtype = h.dtype
        return y

    @staticmethod
    def backward(ctx, g):
        (y,) = ctx.saved_tensors
        gc = g.contiguous()
        dx = torch.empty(y.shape, device=y.device, dtype=ctx.h_dtype)
        n = y.numel()
        _sig_bwd_kernel[(triton.cdiv(n, _BLOCK),)](gc, y, dx, n, BLOCK=_BLOCK)
        return dx


def fused_sigmoid_f32(h: torch.Tensor) -> torch.Tensor:
    """``torch.sigmoid(h.float())`` for 16-bit CUDA ``h``; fp32 result."""
    return _FusedSigmoidF32.apply(h)


def fused_sigmoid_f32_available(h: torch.Tensor) -> bool:
    return _HAS_TRITON and h.is_cuda and h.dtype in _HALF


class _FusedAdd3(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, d, f):
        STATS["add3"] += 1
        out = torch.empty_like(x)
        n = x.numel()
        _add3_fwd_kernel[(triton.cdiv(n, _BLOCK),)](x, d, f, out, n, BLOCK=_BLOCK)
        ctx.dtypes = (d.dtype, f.dtype)
        return out

    @staticmethod
    def backward(ctx, g):
        _, need_d, need_f = ctx.needs_input_grad
        gx = g if ctx.needs_input_grad[0] else None
        if not (need_d or need_f):
            return gx, None, None
        gc = g.contiguous()
        gd = torch.empty(gc.shape, device=gc.device, dtype=ctx.dtypes[0])
        gf = torch.empty(gc.shape, device=gc.device, dtype=ctx.dtypes[1])
        n = gc.numel()
        _cast2_kernel[(triton.cdiv(n, _BLOCK),)](gc, gd, gf, n, BLOCK=_BLOCK)
        return gx, (gd if need_d else None), (gf if need_f else None)


def fused_residual_add3(x: torch.Tensor, d: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
    """``x + d + f`` (fp32 ``x``, 16-bit ``d``/``f``), one pass each way."""
    return _FusedAdd3.apply(x, d, f)


def fused_residual_add3_available(x: torch.Tensor, d: torch.Tensor,
                                  f: torch.Tensor) -> bool:
    return (_HAS_TRITON and x.is_cuda and d.is_cuda and f.is_cuda
            and x.dtype == torch.float32 and d.dtype in _HALF and f.dtype in _HALF
            and x.shape == d.shape == f.shape
            and x.is_contiguous() and d.is_contiguous() and f.is_contiguous())
