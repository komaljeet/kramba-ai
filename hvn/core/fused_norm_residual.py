"""#208: the residual gradient folded into the RMSNorm backward (opt-in, not bit-identical).

Author: Rambarun Komaljeet

In a DeltaNet block (parallel residual) the fp32 input ``x`` has two consumers:
``x_norm = rms_norm(x)`` and the residual sum ``out = x + deflected + ffn``. Autograd
therefore produces ``dx`` as two tensors and adds them: the residual branch hands back
``g`` (the output gradient, unchanged) and the norm backward writes its own fp32
``dx_norm``; the engine then runs ``dx_norm += g`` -- one more read-read-write pass over a
[4096, 2048] fp32 tensor per block backward (the ``add_ [1, 4096, 2048]`` fp32 item in
#207's / #208's profile, ~0.22 s per 1B step).

``norm_residual(x, w, eps)`` returns ``(x_norm, x_link)``. ``x_link`` is ``x`` (a view;
the residual sum must use it instead of ``x``), so both gradients arrive at THIS
Function's backward, which writes ``dx = rms_norm_backward(g_norm) + g_link`` in one
Triton pass. Forward is ``F.rms_norm`` itself, so forward values are unchanged bit for
bit. The backward recomputes ``rstd`` from the saved ``x`` and reduces in its own order,
so ``dx`` / ``dw`` differ from ATen's in the last bits (checked against fp64 in
tests/test_fused_norm_residual.py: same error size as ATen's).

Switch: ``ENABLED[0]`` (trainer ``--fused-norm-residual``, default off).
"""
import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

ENABLED = [False]
STATS = {"fwd": 0, "bwd": 0, "bwd_no_res": 0}

if _HAS_TRITON:

    @triton.jit
    def _norm_res_bwd_kernel(x_ptr, w_ptr, g_ptr, r_ptr, dx_ptr, dwp_ptr, M, eps,
                             N: tl.constexpr, ROWS: tl.constexpr, HAS_RES: tl.constexpr,
                             NEED_DW: tl.constexpr):
        pid = tl.program_id(0)
        cols = tl.arange(0, N)
        w = tl.load(w_ptr + cols).to(tl.float32)
        dw = tl.zeros((N,), dtype=tl.float32)
        for i in range(ROWS):
            row = pid * ROWS + i
            if row < M:
                off = row * N + cols
                x = tl.load(x_ptr + off).to(tl.float32)
                g = tl.load(g_ptr + off).to(tl.float32)
                rstd = 1.0 / tl.sqrt(tl.sum(x * x, axis=0) / N + eps)
                xhat = x * rstd
                wg = w * g
                c = tl.sum(xhat * wg, axis=0) / N
                dx = (wg - xhat * c) * rstd
                if HAS_RES:
                    dx += tl.load(r_ptr + off).to(tl.float32)
                tl.store(dx_ptr + off, dx.to(dx_ptr.dtype.element_ty))
                if NEED_DW:
                    dw += g * xhat
        if NEED_DW:
            tl.store(dwp_ptr + pid * N + cols, dw)


_ROWS = 32


class _NormResidual(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps):
        y = F.rms_norm(x, (x.shape[-1],), w, eps)
        ctx.save_for_backward(x, w)
        ctx.eps = eps
        ctx.set_materialize_grads(False)    # an unused output's grad arrives as None
        STATS["fwd"] += 1
        return y, x.view_as(x)

    @staticmethod
    def backward(ctx, g, g_res):
        x, w = ctx.saved_tensors
        need_x, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        N = x.shape[-1]
        M = x.numel() // N
        if g is None:
            return (g_res if need_x else None), None, None
        g = g.contiguous()
        has_res = g_res is not None
        r = g_res.contiguous() if has_res else g
        STATS["bwd"] += 1
        STATS["bwd_no_res"] += int(not has_res)
        dx = torch.empty_like(x)
        P = triton.cdiv(M, _ROWS)
        dwp = torch.empty((P, N), device=x.device, dtype=torch.float32) if need_w else dx
        _norm_res_bwd_kernel[(P,)](x, w, g, r, dx, dwp, M, ctx.eps, N=N, ROWS=_ROWS,
                                   HAS_RES=has_res, NEED_DW=need_w, num_warps=8)
        dw = dwp.sum(0).to(w.dtype) if need_w else None
        return (dx if need_x else None), dw, None


def available(x: torch.Tensor, w: torch.Tensor) -> bool:
    N = x.shape[-1]
    return (_HAS_TRITON and x.is_cuda and x.dtype == torch.float32 and x.is_contiguous()
            and w.dtype == torch.float32 and w.shape == (N,) and N & (N - 1) == 0
            and N <= 8192)


def norm_residual(x: torch.Tensor, w: torch.Tensor, eps: float):
    """``(rms_norm(x) * w, x_link)``; use ``x_link`` for the residual sum."""
    return _NormResidual.apply(x, w, eps)
