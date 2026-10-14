"""Causal depthwise conv read straight from the [B, S, D] layout (Triton).

Drop-in for the unpacked branch of ``DeltaNetBlock._causal_conv`` on CUDA
under bf16/fp16 autocast (2026-10-14).

Why: the eager path is ``x.transpose(1, 2)`` -> ``F.pad`` -> autocast cast ->
``conv1d``. ``F.pad`` on the transposed fp32 view is a full strided copy
(reads [B, S, D] column-wise), autocast then makes a second bf16 copy, and the
backward pays the mirror images (a pad-backward clone and a bf16->fp32 cast)
on top of the conv kernels. Measured at the 1B shape (B=1, S=4096, D=2048):
~3.4 ms per DeltaNet layer per forward+backward, of which the conv arithmetic
itself is a small part. Here the forward is ONE pass (read x, write y) and the
input-gradient is ONE pass (read g, write dx).

**Bit-identical to the eager path**, not ulp-close, because it copies the
arithmetic of PyTorch's ``conv_depthwise2d`` kernels exactly:

* forward: ``acc = fp32(cdt(bias)); for k in 0..K-1: acc += fp32(cdt(w_k)) *
  fp32(cdt(x[t-(K-1)+k]))``, zero-padded taps included, then ``cdt(acc)``.
  ``cdt`` is the autocast dtype; a bf16 x bf16 (or fp16 x fp16) product is
  exact in fp32, so FMA contraction cannot change a bit.
* input grad: ``acc = 0; for k: if in range: acc += w_k * g[t+(K-1)-k]``,
  out-of-range taps SKIPPED (not added as zero, which could flip -0.0), then
  ``cdt(acc)`` cast to x's dtype -- the eager path's conv backward followed by
  autocast's ToCopyBackward.
* weight / bias grads are rare (only the BCD-active block's convs need them)
  and are delegated to the very same ``aten.convolution_backward`` call the
  eager path makes, on a re-materialised padded input, so they are identical
  by construction.

The output is returned as the same transposed view of a [B, D, S] buffer the
eager path returns, so every downstream GEMM sees identical strides and picks
the identical kernel. ``tests/test_fused_causal_conv.py`` asserts torch.equal on
the output and all three gradients, and fails if the fast path is not taken.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

#: Call counters, so a test can prove the fast path actually ran.
STATS = {"fwd": 0, "bwd": 0, "bwd_param": 0}

_BS, _BD = 64, 64


if _HAS_TRITON:

    @triton.jit
    def _cc_fwd_kernel(x_ptr, w_ptr, b_ptr, y_ptr, S, D, sxb, sxs, sxd,
                       HAS_BIAS: tl.constexpr, K: tl.constexpr,
                       BS: tl.constexpr, BD: tl.constexpr):
        ps = tl.program_id(0)
        pd = tl.program_id(1)
        b = tl.program_id(2)
        s = ps * BS + tl.arange(0, BS)
        d = pd * BD + tl.arange(0, BD)
        dm = d < D
        CDT = y_ptr.dtype.element_ty
        if HAS_BIAS:
            bias = tl.load(b_ptr + d, mask=dm, other=0.0).to(CDT).to(tl.float32)
        else:
            bias = tl.zeros([BD], dtype=tl.float32)
        acc = tl.zeros([BS, BD], dtype=tl.float32) + bias[None, :]
        xb = x_ptr + b.to(tl.int64) * sxb
        for kk in tl.static_range(K):
            src = s - (K - 1) + kk
            m = ((src >= 0) & (src < S))[:, None] & dm[None, :]
            xv = tl.load(xb + src[:, None].to(tl.int64) * sxs + d[None, :] * sxd,
                         mask=m, other=0.0).to(CDT).to(tl.float32)
            wv = tl.load(w_ptr + d * K + kk, mask=dm, other=0.0).to(CDT).to(tl.float32)
            acc = acc + wv[None, :] * xv
        # y is a contiguous [B, D, S] buffer (the eager conv's output layout).
        yo = (b.to(tl.int64) * D * S + d[None, :].to(tl.int64) * S + s[:, None])
        tl.store(y_ptr + yo, acc.to(CDT), mask=(s < S)[:, None] & dm[None, :])

    @triton.jit
    def _cc_bwd_kernel(g_ptr, w_ptr, dx_ptr, S, D, sgb, sgs, sgd,
                       CDT: tl.constexpr, K: tl.constexpr,
                       BS: tl.constexpr, BD: tl.constexpr):
        ps = tl.program_id(0)
        pd = tl.program_id(1)
        b = tl.program_id(2)
        s = ps * BS + tl.arange(0, BS)
        d = pd * BD + tl.arange(0, BD)
        dm = d < D
        sm = s < S
        acc = tl.zeros([BS, BD], dtype=tl.float32)
        gb = g_ptr + b.to(tl.int64) * sgb
        for kk in tl.static_range(K):
            o = s + (K - 1) - kk
            m = (sm & (o < S))[:, None] & dm[None, :]
            gv = tl.load(gb + o[:, None].to(tl.int64) * sgs + d[None, :] * sgd,
                         mask=m, other=0.0).to(tl.float32)
            wv = tl.load(w_ptr + d * K + kk, mask=dm, other=0.0).to(CDT).to(tl.float32)
            acc = tl.where(m, acc + wv[None, :] * gv, acc)
        dxo = b.to(tl.int64) * S * D + s[:, None].to(tl.int64) * D + d[None, :]
        tl.store(dx_ptr + dxo, acc.to(CDT).to(dx_ptr.dtype.element_ty),
                 mask=sm[:, None] & dm[None, :])


class _FusedCausalConv(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, cdt):
        B, S, D = x.shape
        K = weight.shape[-1]
        w = weight.detach().reshape(D, K).contiguous()
        y = torch.empty(B, D, S, device=x.device, dtype=cdt)
        grid = (triton.cdiv(S, _BS), triton.cdiv(D, _BD), B)
        _cc_fwd_kernel[grid](x, w, bias if bias is not None else w, y, S, D,
                             x.stride(0), x.stride(1), x.stride(2),
                             HAS_BIAS=bias is not None, K=K, BS=_BS, BD=_BD)
        ctx.save_for_backward(x, weight, bias)
        ctx.cdt = cdt
        STATS["fwd"] += 1
        return y.transpose(1, 2)

    @staticmethod
    def backward(ctx, g):
        x, weight, bias = ctx.saved_tensors
        cdt = ctx.cdt
        B, S, D = x.shape
        K = weight.shape[-1]
        need_x, need_w, need_b = ctx.needs_input_grad[:3]
        dx = dw = db = None
        STATS["bwd"] += 1
        if need_x:
            w = weight.detach().reshape(D, K).contiguous()
            dx = torch.empty(B, S, D, device=x.device, dtype=x.dtype)
            grid = (triton.cdiv(S, _BS), triton.cdiv(D, _BD), B)
            _cc_bwd_kernel[grid](g, w, dx, S, D, g.stride(0), g.stride(1), g.stride(2),
                                 CDT=_TL_DTYPE[cdt], K=K, BS=_BS, BD=_BD)
        if need_w or (bias is not None and need_b):
            # The eager path's own call, on the inputs it would have saved.
            STATS["bwd_param"] += 1
            x_pad = F.pad(x.detach().transpose(1, 2), (K - 1, 0)).to(cdt)
            _, gw, gb = torch.ops.aten.convolution_backward(
                g.transpose(1, 2), x_pad, weight.detach().to(cdt),
                [D] if bias is not None else None, [1], [0], [1], False, [0], D,
                [False, bool(need_w), bool(bias is not None and need_b)])
            if need_w:
                dw = gw.to(weight.dtype)
            if bias is not None and need_b:
                db = gb.to(bias.dtype)
        return dx, dw, db, None


_TL_DTYPE = {}
if _HAS_TRITON:
    _TL_DTYPE = {torch.bfloat16: tl.bfloat16, torch.float16: tl.float16}


def fused_causal_conv_dtype(x: torch.Tensor, weight: torch.Tensor,
                            bias: torch.Tensor | None = None):
    """The conv's compute dtype if the fused path applies, else ``None``.

    Covers exactly the case whose arithmetic it reproduces: CUDA, a bf16/fp16
    compute dtype (autocast's, or the operands' own when autocast is off), a
    unit-stride channel dim. bf16 compute only (see below). Anything else takes the eager path unchanged.
    """
    if not (_HAS_TRITON and x.is_cuda and weight.is_cuda and x.dim() == 3):
        return None
    if x.dtype not in (torch.float32, torch.bfloat16, torch.float16):
        return None
    if torch.is_autocast_enabled("cuda"):
        cdt = torch.get_autocast_dtype("cuda")
    else:
        if weight.dtype != x.dtype or (bias is not None and bias.dtype != x.dtype):
            return None
        cdt = x.dtype
    # bf16 only: for fp16 PyTorch routes depthwise convs to cuDNN, whose
    # arithmetic differs (measured: ~1 ulp off), so parity would not hold.
    if cdt != torch.bfloat16 or x.stride(2) != 1:
        return None
    return cdt


def fused_causal_conv(x: torch.Tensor, weight: torch.Tensor,
                      bias: torch.Tensor | None, cdt: torch.dtype) -> torch.Tensor:
    """Causal depthwise conv. x: [B, S, D], weight: [D, 1, K] -> [B, S, D] (cdt)."""
    return _FusedCausalConv.apply(x, weight, bias, cdt)
