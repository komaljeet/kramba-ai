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

    # ---- read norm + output gate + cast, one pass (2026-10-17) ----------
    # The eager tail after the read norm was three full passes over [B,S,D]:
    # write y fp32, ``y * gate`` (fp32 out, bf16 gate promoted), then
    # autocast's cast to bf16 for out_proj. Same arithmetic here, in order:
    # y = o*r*w (fp32), t = y * float(gate) (fp32), out = bf16(t).
    #
    # BIT-identity also needs the SAME reduction order for r (and c, dw) as
    # _rn_fwd/_rn_bwd. Triton picks the register layout -- and so the order
    # tl.sum adds a row in -- from the widest memory access: fp32 rows give
    # sizePerThread [1,4] x threadsPerWarp [4,8], but a 16-bit gate/out row
    # vectorises to [1,8] x [8,4], a different add tree, and r moved by an ulp
    # on 42,697 of 262,144 rows (caught by tests/test_fused_gates.py). The
    # max_contiguous(j, 4) hint caps the 16-bit accesses at 4 elements, which
    # restores the fp32 kernels' layout.

    @triton.jit
    def _rng_fwd_kernel(o_ptr, w_ptr, g8_ptr, y_ptr, r_ptr, S, H, eps,
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
        w = tl.load(w_ptr + h * D + j, mask=jm, other=0.0).to(tl.float32)
        r = 1.0 / tl.sqrt(tl.sum(o * o, axis=1) / D + eps)
        y = o * r[:, None] * w[None, :]
        off = ((b * S + s) * H + h)[:, None] * D + tl.max_contiguous(j, 4)[None, :]
        gt = tl.load(g8_ptr + off, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + off, (y * gt).to(y_ptr.dtype.element_ty), mask=m)
        tl.store(r_ptr + n * S + s, r, mask=sm)

    @triton.jit
    def _rng_bwd_kernel(o_ptr, w_ptr, r_ptr, g8_ptr, gy_ptr, do_ptr, dg8_ptr, dwp_ptr,
                        S, H, D: tl.constexpr, BD: tl.constexpr, BS: tl.constexpr):
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
        off = ((b * S + s) * H + h)[:, None] * D + tl.max_contiguous(j, 4)[None, :]
        # cast backward: bf16 grad -> fp32; mul backward: grad_y = gy * gate
        gyf = tl.load(gy_ptr + off, mask=m, other=0.0).to(tl.float32)
        gt = tl.load(g8_ptr + off, mask=m, other=0.0).to(tl.float32)
        w = tl.load(w_ptr + h * D + j, mask=jm, other=0.0).to(tl.float32)
        r = tl.load(r_ptr + n * S + s, mask=sm, other=0.0)
        # grad_gate = bf16(gy * y), y recomputed exactly as the forward did
        y = o * r[:, None] * w[None, :]
        tl.store(dg8_ptr + off, (gyf * y).to(dg8_ptr.dtype.element_ty), mask=m)
        g = gyf * gt
        wg = w[None, :] * g
        c = tl.sum(wg * o, axis=1) / D
        do = r[:, None] * wg - o * (r * r * r * c)[:, None]
        tl.store(do_ptr + (n * S + s)[:, None] * D + j[None, :],
                 do.to(do_ptr.dtype.element_ty), mask=m)
        dwp = tl.sum(g * o * r[:, None], axis=0)
        nsb = tl.num_programs(1)
        tl.store(dwp_ptr + (n * nsb + sb) * D + j, dwp, mask=jm)


_BS = 64
#: Forward calls of the gate-folded path (tests assert it actually ran).
STATS = {"read_gate": 0}


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


class _FusedReadNormGate(torch.autograd.Function):
    @staticmethod
    def forward(ctx, oN, w, gate, B, H, eps):
        STATS["read_gate"] += 1
        N, S, D = oN.shape
        oc = oN.contiguous()
        gc = gate.contiguous()
        y = torch.empty(B, S, H * D, device=oN.device, dtype=gate.dtype)
        r = torch.empty(N, S, device=oN.device, dtype=torch.float32)
        BD = triton.next_power_of_2(D)
        grid = (N, triton.cdiv(S, _BS))
        _rng_fwd_kernel[grid](oc, w, gc, y, r, S, H, eps, D=D, BD=BD, BS=_BS)
        ctx.save_for_backward(oc, w, r, gc)
        ctx.dims = (B, H)
        return y

    @staticmethod
    def backward(ctx, gy):
        oc, w, r, gc = ctx.saved_tensors
        B, H = ctx.dims
        N, S, D = oc.shape
        gyc = gy.contiguous()
        do = torch.empty_like(oc)
        dg = torch.empty_like(gc)
        nsb = triton.cdiv(S, _BS)
        dwp = torch.empty(N, nsb, D, device=oc.device, dtype=torch.float32)
        BD = triton.next_power_of_2(D)
        _rng_bwd_kernel[(N, nsb)](oc, w, r, gc, gyc, do, dg, dwp, S, H,
                                  D=D, BD=BD, BS=_BS)
        dw = dwp.view(B, H, nsb, D).sum(dim=(0, 2)).to(w.dtype)
        return do, dw, dg, None, None, None


def fused_read_norm_gate(oN: torch.Tensor, weight: torch.Tensor, gate: torch.Tensor,
                         B: int, H: int, eps: float) -> torch.Tensor:
    """``(fused_read_norm(oN, weight) * gate).to(gate.dtype)`` in one pass.

    Bit-identical to that eager tail (forward and backward) for an fp32 ``oN``
    and a 16-bit ``gate`` -- the case autocast produces, where ``out_proj``
    would cast the fp32 product to the gate's dtype anyway. Returned in the
    gate's dtype, [B, S, H*d].
    """
    return _FusedReadNormGate.apply(oN, weight, gate, int(B), int(H), float(eps))


def fused_read_norm_gate_available(oN: torch.Tensor, weight: torch.Tensor,
                                   gate: torch.Tensor | None) -> bool:
    """The gate fold applies only where it reproduces the eager tail exactly:
    fp32 scan output, a bf16/fp16 gate, and CUDA autocast to that same dtype
    (so ``out_proj`` would have cast ``o * gate`` to it regardless)."""
    if gate is None or not fused_read_norm_available(oN, weight):
        return False
    if oN.dtype != torch.float32 or gate.dtype not in (torch.bfloat16, torch.float16):
        return False
    if not gate.is_cuda or gate.dim() != 3:
        return False
    return (torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == gate.dtype)


def fused_read_norm_available(oN: torch.Tensor, weight: torch.Tensor) -> bool:
    return (_HAS_TRITON and oN.is_cuda and oN.dim() == 3
            and oN.dtype in (torch.float32, torch.bfloat16, torch.float16)
            and weight.dim() == 2)
