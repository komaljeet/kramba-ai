"""Fused per-head read RMSNorm for DeltaNet (speed lever 3, 2026-10-12).

Drop-in for ``DeltaNetBlock._read_normalize`` + the following
``reshape(B, S, D)`` on CUDA. The reference is six bandwidth-bound PyTorch ops
(float, pow, mean, add+rsqrt, two multiplies) on a head-major permuted view,
then a reshape that has to COPY because the result keeps the permuted strides.
Measured at the 1B shape: 2.47 ms forward, 8.25 ms forward+backward per layer.

Input ``oN`` is the scan output in its native ``[B*H, S, d]`` layout (no
permute materialised). Output is written directly as contiguous
``[B, S, H*d]`` -- the layout the next op wants -- in the input's dtype.

Same math as the reference, all fp32:
    r = rsqrt(mean_j(o^2) + eps);  y = o * r * w[h]
    do = r * w*g - o * r^3 * mean_j(w*g*o)
    dw[h] = sum over (b, s) of g * o * r      (two-stage, deterministic)
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
    def _rn_fwd_kernel(o_ptr, w_ptr, y_ptr, r_ptr, S, H, eps,
                       D: tl.constexpr, BD: tl.constexpr, BS: tl.constexpr):
        n = tl.program_id(0)          # n = b*H + h  (row of oN)
        sb = tl.program_id(1)
        b = n // H
        h = n % H
        s = sb * BS + tl.arange(0, BS)
        j = tl.arange(0, BD)
        sm = s < S
        jm = j < D
        m = sm[:, None] & jm[None, :]
        o = tl.load(o_ptr + (n * S + s)[:, None] * D + j[None, :], mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + h * D + j, mask=jm, other=0.0).to(tl.float32)
        r = 1.0 / tl.sqrt(tl.sum(o * o, axis=1) / D + eps)
        y = o * r[:, None] * w[None, :]
        # y is [B, S, H, D] contiguous
        tl.store(y_ptr + ((b * S + s) * H + h)[:, None] * D + j[None, :],
                 y.to(y_ptr.dtype.element_ty), mask=m)
        tl.store(r_ptr + n * S + s, r, mask=sm)

    @triton.jit
    def _rn_bwd_kernel(o_ptr, w_ptr, r_ptr, g_ptr, do_ptr, dwp_ptr, S, H,
                       D: tl.constexpr, BD: tl.constexpr, BS: tl.constexpr):
        n = tl.program_id(0)
        sb = tl.program_id(1)
        b = n // H
        h = n % H
        s = sb * BS + tl.arange(0, BS)
        j = tl.arange(0, BD)
        sm = s < S
        jm = j < D
        m = sm[:, None] & jm[None, :]
        o = tl.load(o_ptr + (n * S + s)[:, None] * D + j[None, :], mask=m, other=0.0).to(tl.float32)
        g = tl.load(g_ptr + ((b * S + s) * H + h)[:, None] * D + j[None, :],
                    mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + h * D + j, mask=jm, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + n * S + s, mask=sm, other=0.0)
        wg = w[None, :] * g
        c = tl.sum(wg * o, axis=1) / D
        do = r[:, None] * wg - o * (r * r * r * c)[:, None]
        tl.store(do_ptr + (n * S + s)[:, None] * D + j[None, :],
                 do.to(do_ptr.dtype.element_ty), mask=m)
        # partial dw for this (n, s-block): [D], summed on the host side
        dwp = tl.sum(g * o * r[:, None], axis=0)
        nsb = tl.num_programs(1)
        tl.store(dwp_ptr + (n * nsb + sb) * D + j, dwp, mask=jm)


_BS = 64


class _FusedReadNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, oN, w, B, H, eps):
        N, S, D = oN.shape
        oc = oN.contiguous()
        y = torch.empty(B, S, H * D, device=oN.device, dtype=oN.dtype)
        r = torch.empty(N, S, device=oN.device, dtype=torch.float32)
        BD = triton.next_power_of_2(D)
        grid = (N, triton.cdiv(S, _BS))
        _rn_fwd_kernel[grid](oc, w, y, r, S, H, eps, D=D, BD=BD, BS=_BS)
        ctx.save_for_backward(oc, w, r)
        ctx.dims = (B, H)
        return y

    @staticmethod
    def backward(ctx, gy):
        oc, w, r = ctx.saved_tensors
        B, H = ctx.dims
        N, S, D = oc.shape
        gc = gy.contiguous()
        do = torch.empty_like(oc)
        nsb = triton.cdiv(S, _BS)
        dwp = torch.empty(N, nsb, D, device=oc.device, dtype=torch.float32)
        BD = triton.next_power_of_2(D)
        _rn_bwd_kernel[(N, nsb)](oc, w, r, gc, do, dwp, S, H, D=D, BD=BD, BS=_BS)
        # dwp rows are ordered n = b*H + h; reduce over b and s-blocks.
        dw = dwp.view(B, H, nsb, D).sum(dim=(0, 2)).to(w.dtype)
        return do, dw, None, None, None


def fused_read_norm(oN: torch.Tensor, weight: torch.Tensor, B: int, H: int,
                    eps: float) -> torch.Tensor:
    """oN [B*H, S, d] -> per-head RMSNorm * weight[H, d], returned [B, S, H*d]."""
    return _FusedReadNorm.apply(oN, weight, int(B), int(H), float(eps))


def fused_read_norm_available(oN: torch.Tensor, weight: torch.Tensor) -> bool:
    return (_HAS_TRITON and oN.is_cuda and oN.dim() == 3
            and oN.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and weight.dim() == 2)
