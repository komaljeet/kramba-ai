"""Single-pass Triton L2 normalisation over the last dim, with its backward.

Drop-in for ``hvn.core.deltanet._l2norm`` on CUDA (2026-10-12, speed lever 3).

Why: ``_l2norm`` is four bandwidth-bound PyTorch ops (cast to fp32, norm,
clamp + broadcast divide, cast back), and its autograd backward is a longer
chain still. At the 1B shape (B=1, S=4096, 64 heads x d_head 32) one call
measured 1.94 ms forward and 7.70 ms forward+backward, and every DeltaNet layer
calls it twice (q and k) on each forward, recompute and backward. This does the
same arithmetic in ONE read and ONE write per direction.

Same math as the reference, in fp32, cast back to the input dtype:
    forward:  n = sqrt(sum x^2);  c = max(n, eps);  y = x / c
    backward: dx = g / c - x * sum(g * x) / (c^2 * n)   if n > eps
              dx = g / c                                  otherwise
(the clamp passes no gradient to the norm once it is active, as in autograd).
Not bit-identical to the reference: reduction order differs, so the fp32
result can differ by an ulp, which very rarely flips the bf16 rounding. The
tests bound that against an fp64 reference.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _l2n_fwd_kernel(x_ptr, y_ptr, n_ptr, R, eps,
                        D: tl.constexpr, BD: tl.constexpr, ROWS: tl.constexpr):
        pid = tl.program_id(0)
        r = pid * ROWS + tl.arange(0, ROWS)
        c = tl.arange(0, BD)
        rm = r < R
        m = rm[:, None] & (c[None, :] < D)
        off = r[:, None] * D + c[None, :]
        x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        n = tl.sqrt(tl.sum(x * x, axis=1))
        cl = tl.maximum(n, eps)
        y = x / cl[:, None]
        tl.store(y_ptr + off, y.to(y_ptr.dtype.element_ty), mask=m)
        tl.store(n_ptr + r, n, mask=rm)

    @triton.jit
    def _l2n_bwd_kernel(x_ptr, g_ptr, n_ptr, dx_ptr, R, eps,
                        D: tl.constexpr, BD: tl.constexpr, ROWS: tl.constexpr):
        pid = tl.program_id(0)
        r = pid * ROWS + tl.arange(0, ROWS)
        c = tl.arange(0, BD)
        rm = r < R
        m = rm[:, None] & (c[None, :] < D)
        off = r[:, None] * D + c[None, :]
        x = tl.load(x_ptr + off, mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + off, mask=m, other=0.0).to(tl.float32)
        n = tl.load(n_ptr + r, mask=rm, other=1.0)
        cl = tl.maximum(n, eps)
        gx = tl.sum(g * x, axis=1)
        # n > eps: the clamp is inactive and the norm carries gradient.
        coef = tl.where(n > eps, gx / (cl * cl * n), 0.0)
        dx = g / cl[:, None] - x * coef[:, None]
        tl.store(dx_ptr + off, dx.to(dx_ptr.dtype.element_ty), mask=m)


def _launch_cfg(D: int):
    BD = triton.next_power_of_2(D)
    ROWS = max(1, min(64, 2048 // BD))
    return BD, ROWS


class _FusedL2Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, eps):
        D = x.shape[-1]
        xc = x.contiguous()
        R = xc.numel() // D
        y = torch.empty_like(xc)
        n = torch.empty(R, device=x.device, dtype=torch.float32)
        BD, ROWS = _launch_cfg(D)
        _l2n_fwd_kernel[(triton.cdiv(R, ROWS),)](xc, y, n, R, eps, D=D, BD=BD, ROWS=ROWS)
        ctx.save_for_backward(xc, n)
        ctx.eps = eps
        return y

    @staticmethod
    def backward(ctx, g):
        xc, n = ctx.saved_tensors
        D = xc.shape[-1]
        R = xc.numel() // D
        gc = g.contiguous()
        dx = torch.empty_like(xc)
        BD, ROWS = _launch_cfg(D)
        _l2n_bwd_kernel[(triton.cdiv(R, ROWS),)](xc, gc, n, dx, R, ctx.eps,
                                                 D=D, BD=BD, ROWS=ROWS)
        return dx, None


def fused_l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise ``x`` over its last dim (fp32 math, result in x.dtype)."""
    return _FusedL2Norm.apply(x, float(eps))


def fused_l2norm_available(x: torch.Tensor) -> bool:
    return (_HAS_TRITON and x.is_cuda
            and x.dtype in (torch.float32, torch.bfloat16, torch.float16))
