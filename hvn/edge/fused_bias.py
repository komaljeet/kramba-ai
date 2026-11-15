"""Bias add after an fp8 GEMM, fused with what reads it (#207). Bit-identical.

Author: Rambarun Komaljeet

The fp8 Linear (``fp8_fused.py``) returns ``y = x @ W.T`` in bf16, then adds the
bias as a separate op: ``y + bias.to(bf16)`` reads y and writes a new bf16
tensor ``t``. In DeltaNetBlock every biased fp8 Linear's output goes straight
into one more elementwise op that reads ``t`` again:

* ``gate_proj``  -> ``F.silu``                       (gate)
* ``ffn[0]``     -> ``nn.GELU`` (exact erf)          (FFN hidden)
* ``ffn[3]``     -> the parallel residual ``(x + deflected) + ffn_out``

Here each pair is one kernel: it reads ``y`` and the bias, rounds ``y + b`` to
bf16 IN REGISTERS exactly as the separate add would, and applies the consumer
to that rounded value. ``t`` is never written or read: 4 bytes per element
less traffic, per forward and per checkpoint recompute. The backward
recomputes ``t`` from the saved ``y`` (same bytes as saving ``t``) and applies
the consumer's backward; the bias gradient is ``grad_t.sum_to_size(b.shape)``,
the reduction the autograd engine does for a broadcast add.

Bit-identity is by matching ATen's arithmetic, operation for operation:

* add: ``bf16(f32(y) + f32(b))``, round to nearest even (ATen ``AddFunctor``
  with alpha 1, opmath float).
* silu: ``x / (1 + exp(-x))`` with the full-precision libdevice ``expf`` and an
  IEEE divide (``div_rn``; Triton's ``/`` is approximate, the #154 lesson).
  Backward ``(dy * s) * (1 + x * (1 - s))``, ``s = 1 / (1 + exp(-x))``.
* gelu (approximate='none'): computing ``(x * 0.5) * (1 + erf(x / sqrt 2))``
  in Triton does NOT match: Triton's libdevice ``erff`` differs from the one
  PyTorch is built with in the last bit for 11 of the 65,536 bf16 inputs (all
  below -3, where ``1 + erf`` cancels). The input is a bf16 value, so the
  output is a function of 16 bits: a 65,536-entry table made by ATen's own
  ``F.gelu`` gives its bits by construction. Backward is ATen's
  ``dy * (cdf + x * pdf)``; the factor depends on x alone, so it is a second
  table (fp32, ATen's ``gelu_backward`` with dy = 1.0) and one multiply,
  checked against ATen over every bf16 x and 20 scales of dy.
* residual: ``(x + d) + f`` in fp32, as ``fused_gates._add3_fwd_kernel``.

``tests/test_fused_bias.py`` checks every one of the 65,536 bf16 inputs (and
random upstream gradients) against ATen with ``torch.equal``, forward and
backward, plus real 1B shapes, and must-fail partners show the check sees a
sub-ulp change.
"""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False

#: Calls taken by each fused path (tests assert these move).
STATS = {"bias": 0, "silu": 0, "gelu": 0, "add3": 0, "amax_hits": 0}

#: ``--fp8-bias`` (#207): "fused" (default) = bias folded into its consumer in
#: DeltaNetBlock, the 2-D add kernel elsewhere; "add" = the 2-D add kernel
#: everywhere, consumers eager; "off" = ``y + bias.to(bf16)`` and eager
#: consumers (the pre-#207 path). All three give the same bits.
MODE = ["fused"]
_MODES = ("off", "add", "fused")

ACT_NONE, ACT_SILU, ACT_GELU = 0, 1, 2
_K_ALPHA = float(torch.tensor(math.sqrt(0.5), dtype=torch.float32))      # (float) M_SQRT1_2
# ATen: constexpr opmath_t kBeta = M_2_SQRTPI * M_SQRT1_2 * opmath_t(0.5);
# a double product rounded once to float.
_K_BETA = float(torch.tensor((2.0 / math.sqrt(math.pi)) * math.sqrt(0.5) * 0.5,
                             dtype=torch.float64).float())
_BM, _BN = 32, 128
#: Test hook only (the must-fail partner): False rounds ``y + b`` to bf16
#: toward zero instead of to nearest, a sub-ulp change the tests must see.
_ROUND_T = [True]

if _HAS_TRITON:

    @triton.jit
    def _bits(t):
        # uint16 bit pattern of bf16(t), as an index into a 65,536-entry table.
        return t.to(tl.bfloat16).to(tl.int16, bitcast=True).to(tl.int32) & 0xFFFF

    @triton.jit
    def _tile_amax(r, mask, amax_ptr):
        # fp8_fused._amax_kernel for this tile: max |bf16 value|, NaN published
        # explicitly (Triton's max drops NaN), atomic max of tile maxima. Max is
        # exact and order-free, so the result equals _amax_kernel's bit for bit.
        v = tl.where(mask, r.to(tl.bfloat16).to(tl.float32), 0.0)
        a = tl.max(tl.max(tl.abs(v), axis=1), axis=0)
        has_nan = tl.max(tl.max((v != v).to(tl.int32), axis=1), axis=0) > 0
        a = tl.where(has_nan, float("nan"), a)
        tl.atomic_max(amax_ptr, a)

    @triton.jit
    def _bias_act_fwd_kernel(y_ptr, b_ptr, o_ptr, lut_ptr, amax_ptr, M, N, ACT: tl.constexpr,
                             K_ALPHA: tl.constexpr, ROUND_T: tl.constexpr, AMAX: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        off = rm[:, None] * N + rn[None, :]
        y = tl.load(y_ptr + off, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        t = y + b[None, :]
        if ROUND_T:
            t = t.to(tl.bfloat16).to(tl.float32)
        else:
            t = t.to(tl.bfloat16, fp_downcast_rounding="rtz").to(tl.float32)
        if ACT == 1:
            r = tl.div_rn(t, 1.0 + tl.extra.cuda.libdevice.exp(-t)).to(tl.bfloat16)
        elif ACT == 2:
            # GELU of a bf16 input, looked up: the table IS ATen's bf16 GELU
            # of every bf16 value (Triton's libdevice erff differs from the
            # one PyTorch is built with in the last bit for 11 inputs, x < -3).
            r = tl.load(lut_ptr + _bits(t), mask=mask, other=0.0)
        else:
            r = t.to(tl.bfloat16)
        tl.store(o_ptr + off, r, mask=mask)
        if AMAX:
            _tile_amax(r, mask, amax_ptr)

    @triton.jit
    def _bias_act_bwd_kernel(y_ptr, b_ptr, g_ptr, dt_ptr, lut_ptr, amax_ptr, M, N,
                             ACT: tl.constexpr, K_ALPHA: tl.constexpr, K_BETA: tl.constexpr,
                             ROUND_T: tl.constexpr, AMAX: tl.constexpr,
                             BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        off = rm[:, None] * N + rn[None, :]
        y = tl.load(y_ptr + off, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        t = y + b[None, :]
        if ROUND_T:
            t = t.to(tl.bfloat16).to(tl.float32)
        else:
            t = t.to(tl.bfloat16, fp_downcast_rounding="rtz").to(tl.float32)
        dy = tl.load(g_ptr + off, mask=mask, other=0.0).to(tl.float32)
        if ACT == 1:
            s = tl.div_rn(1.0, 1.0 + tl.extra.cuda.libdevice.exp(-t))
            dt = (dy * s) * (1.0 + t * (1.0 - s))
        else:
            # ATen: dy * (cdf + x * pdf). The factor (cdf + x * pdf) depends
            # on x alone; the table holds it in fp32 exactly as ATen computes
            # it (gelu_backward with dy = 1.0 in fp32), so one multiply
            # remains, as in ATen.
            h = tl.load(lut_ptr + _bits(t), mask=mask, other=0.0)
            dt = dy * h
        dt = dt.to(tl.bfloat16)
        tl.store(dt_ptr + off, dt, mask=mask)
        if AMAX:
            _tile_amax(dt, mask, amax_ptr)

    @triton.jit
    def _add3_bias_fwd_kernel(x_ptr, d_ptr, y_ptr, b_ptr, o_ptr, M, N, ROUND_T: tl.constexpr,
                              BM: tl.constexpr, BN: tl.constexpr):
        rm = tl.program_id(0) * BM + tl.arange(0, BM)
        rn = tl.program_id(1) * BN + tl.arange(0, BN)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        off = rm[:, None] * N + rn[None, :]
        x = tl.load(x_ptr + off, mask=mask, other=0.0)
        d = tl.load(d_ptr + off, mask=mask, other=0.0).to(tl.float32)
        y = tl.load(y_ptr + off, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(b_ptr + rn, mask=rn < N, other=0.0).to(tl.float32)
        f = y + b[None, :]
        if ROUND_T:
            f = f.to(tl.bfloat16).to(tl.float32)
        else:
            f = f.to(tl.bfloat16, fp_downcast_rounding="rtz").to(tl.float32)
        tl.store(o_ptr + off, (x + d) + f, mask=mask)

    @triton.jit
    def _cast2_kernel(g_ptr, a_ptr, c_ptr, amax_ptr, n, AMAX: tl.constexpr,
                      BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        g = tl.load(g_ptr + offs, mask=m, other=0.0)
        tl.store(a_ptr + offs, g.to(a_ptr.dtype.element_ty), mask=m)
        tl.store(c_ptr + offs, g.to(c_ptr.dtype.element_ty), mask=m)
        if AMAX:
            # Both outputs are bf16(g): one amax serves both.
            v = tl.where(m, g.to(tl.bfloat16).to(tl.float32), 0.0)
            a = tl.max(tl.abs(v), axis=0)
            has_nan = tl.max((v != v).to(tl.int32), axis=0) > 0
            a = tl.where(has_nan, float("nan"), a)
            tl.atomic_max(amax_ptr, a)


def _grid(M, N):
    return (triton.cdiv(M, _BM), triton.cdiv(N, _BN))


def _bias_grad(gt: torch.Tensor, bshape):
    # What the autograd engine does for the broadcast operand of ``y + b``
    # (validate_outputs -> at::sum_to), on the same 2-D [M, O] gradient.
    return gt.sum_to_size(bshape)


#: device -> (bf16 GELU of every bf16 value, fp32 GELU-backward factor of every
#: bf16 value), both indexed by the uint16 bit pattern, both computed by ATen
#: itself, so a lookup returns ATen's bits by construction.
_LUT = {}


@torch.no_grad()
def gelu_tables(device):
    key = str(device)
    if key not in _LUT:
        with torch.autocast("cuda", enabled=False):
            v = torch.arange(65536, dtype=torch.int32, device=device).to(torch.int16).view(
                torch.bfloat16)
            fwd = torch.nn.functional.gelu(v)
            h = torch.ops.aten.gelu_backward(torch.ones(v.shape, device=device), v.float())
        _LUT[key] = (fwd.contiguous(), h.contiguous())
    return _LUT[key]


# ---------------------------------------------------------------------------
# amax hints (#207). Each tensor these kernels write that goes straight into an
# fp8 GEMM -- the GELU output (ffn[3]'s input) and, in backward, the gradients
# reaching gate_proj, ffn[0], ffn[3] and out_proj -- is quantized next by
# ``fp8_fused.fp8_quantize``, whose first pass re-reads the whole tensor for
# its amax. The producing kernel already holds every value in registers, so it
# computes the same amax (max is exact and order-free: same bits) and leaves
# it here; ``fp8_quantize`` takes it instead of running ``_amax_kernel``.
#
# A hint is valid only for the same bytes: the producer's tensor must still be
# alive (weak reference: no memory is kept alive by a hint), and the consumer's
# tensor must have its data pointer, element count, dtype and version counter
# (any in-place write bumps it). Anything else misses and the normal amax runs.
# ---------------------------------------------------------------------------
#: ``--fused-bias-amax`` (default on); False measures/uses the fusions without hints.
AMAX_HINTS = [True]
_HINTS = []                      # [(weakref, ptr, numel, dtype, version, amax)]
_MAX_HINTS = 4


def _new_amax(t: torch.Tensor):
    return torch.zeros((), device=t.device, dtype=torch.float32)


def _put_hint(t: torch.Tensor, amax: torch.Tensor) -> None:
    import weakref
    _HINTS.append((weakref.ref(t), t.data_ptr(), t.numel(), t.dtype, t._version, amax))
    del _HINTS[:-_MAX_HINTS]


def take_amax_hint(t: torch.Tensor):
    """The amax a producer computed for exactly ``t``'s bytes, or None."""
    if not _HINTS or t.is_inference():
        return None
    # Drop hints whose tensor is gone: its address may now hold another tensor.
    _HINTS[:] = [h for h in _HINTS if h[0]() is not None]
    ptr, n, dt, ver = t.data_ptr(), t.numel(), t.dtype, t._version
    for i, (ref, p, nn_, d, v, amax) in enumerate(_HINTS):
        o = ref()
        if (p == ptr and nn_ == n and d == dt and v == ver and o is not None
                and o.data_ptr() == ptr and o._version == ver):
            del _HINTS[i]
            STATS["amax_hits"] += 1
            return amax
    return None


def clear_amax_hints() -> None:
    _HINTS.clear()


class _BiasAct(torch.autograd.Function):
    """``act(y + b)`` for bf16 ``y`` [M, O] (row-major) and bf16 ``b`` [O]."""

    @staticmethod
    def forward(ctx, y, b, act):
        M, N = y.shape
        out = torch.empty_like(y)
        lut = gelu_tables(y.device)[0] if act == ACT_GELU else y
        hint = AMAX_HINTS[0] and act == ACT_GELU      # GELU output = ffn[3]'s fp8 input
        amax = _new_amax(y) if hint else y
        _bias_act_fwd_kernel[_grid(M, N)](y, b, out, lut, amax, M, N, ACT=act,
                                          K_ALPHA=_K_ALPHA, ROUND_T=_ROUND_T[0], AMAX=hint,
                                          BM=_BM, BN=_BN)
        if hint:
            _put_hint(out, amax)
        ctx.round_t = _ROUND_T[0]
        ctx.act = act
        ctx.bshape = b.shape
        if act == ACT_NONE:
            ctx.save_for_backward()
        else:
            ctx.save_for_backward(y, b)
        return out

    @staticmethod
    def backward(ctx, g):
        need_y, need_b = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        if ctx.act == ACT_NONE:
            gt = g
        else:
            y, b = ctx.saved_tensors
            M, N = y.shape
            gc = g.contiguous()
            gt = torch.empty_like(y)
            lut = gelu_tables(y.device)[1] if ctx.act == ACT_GELU else y
            hint = AMAX_HINTS[0] and need_y          # grad into gate_proj / ffn[0]
            amax = _new_amax(y) if hint else y
            _bias_act_bwd_kernel[_grid(M, N)](y, b, gc, gt, lut, amax, M, N, ACT=ctx.act,
                                              K_ALPHA=_K_ALPHA, K_BETA=_K_BETA,
                                              ROUND_T=ctx.round_t, AMAX=hint,
                                              BM=_BM, BN=_BN)
            if hint:
                _put_hint(gt, amax)
        gb = _bias_grad(gt, ctx.bshape) if need_b else None
        return (gt if need_y else None), gb, None


def _bias_act(y, b, act):
    return _BiasAct.apply(y, b, act)


class _Add3Bias(torch.autograd.Function):
    """``(x + d) + (y + b)`` with fp32 ``x``, bf16 ``d``/``y``, bf16 ``b`` [O]."""

    @staticmethod
    def forward(ctx, x, d, y, b):
        M, N = y.shape
        out = torch.empty_like(x)
        _add3_bias_fwd_kernel[_grid(M, N)](x, d, y, b, out, M, N, ROUND_T=_ROUND_T[0],
                                           BM=_BM, BN=_BN)
        ctx.dtypes = (d.dtype, y.dtype)
        ctx.bshape, ctx.yshape = b.shape, y.shape
        return out

    @staticmethod
    def backward(ctx, g):
        need_x, need_d, need_y, need_b = ctx.needs_input_grad
        gx = g if need_x else None
        if not (need_d or need_y or need_b):
            return gx, None, None, None
        gc = g.contiguous()
        gd = torch.empty(gc.shape, device=gc.device, dtype=ctx.dtypes[0])
        gy = torch.empty(ctx.yshape, device=gc.device, dtype=ctx.dtypes[1])
        n = gc.numel()
        # Grads into out_proj (gd) and ffn[3] (gy): both bf16(g), one amax.
        hint = AMAX_HINTS[0] and ctx.dtypes == (torch.bfloat16, torch.bfloat16)
        amax = _new_amax(gc) if hint else gc
        _cast2_kernel[(triton.cdiv(n, 1024),)](gc, gd, gy, amax, n, AMAX=hint, BLOCK=1024)
        if hint:
            if need_d:
                _put_hint(gd, amax)
            if need_y:
                _put_hint(gy, amax)
        gb = _bias_grad(gy, ctx.bshape) if need_b else None
        return gx, (gd if need_d else None), (gy if (need_y or need_b) else None), gb


def set_mode(mode: str) -> None:
    if mode not in _MODES:
        raise ValueError(f"fp8 bias mode {mode!r} not in {_MODES}")
    MODE[0] = mode


def available(y: torch.Tensor, b) -> bool:
    """The 2-D kernels apply to (y, b): bf16 [M, O] row-major and bf16 [O]."""
    return (_HAS_TRITON and b is not None and y.is_cuda and y.dim() == 2
            and y.dtype == torch.bfloat16 and b.dtype == torch.bfloat16
            and y.is_contiguous() and b.is_contiguous() and b.shape == (y.shape[1],))


def bias_add(y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``y + b`` (bf16 [M, O] + bf16 [O]) in one 2-D tiled pass."""
    STATS["bias"] += 1
    return _bias_act(y, b, ACT_NONE)


def bias_silu(y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``F.silu(y + b)``; ``y + b`` is never materialised."""
    STATS["silu"] += 1
    return _bias_act(y, b, ACT_SILU)


def bias_gelu(y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``F.gelu(y + b)`` (exact erf GELU); ``y + b`` is never materialised."""
    STATS["gelu"] += 1
    return _bias_act(y, b, ACT_GELU)


def add3_bias(x: torch.Tensor, d: torch.Tensor, y: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(x + d) + (y + b)``: the parallel residual with the FFN's last bias folded in.

    ``x`` fp32 and ``d`` bf16, same shape and contiguous; ``y`` the bf16 [M, O]
    GEMM output with the same number of elements; ``b`` bf16 [O].
    """
    assert (x.is_contiguous() and d.is_contiguous() and x.shape == d.shape
            and x.dtype == torch.float32 and x.numel() == y.numel()), (x.shape, d.shape, y.shape)
    STATS["add3"] += 1
    return _Add3Bias.apply(x, d, y, b)
