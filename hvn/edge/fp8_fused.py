"""FP8 Linear with a fused Triton quantizer -- ``--fp8-linear`` (default OFF).

Author: Rambarun Komaljeet

**What it is.** ``y = x @ W.T`` with all three GEMMs of a Linear (forward, dX,
dW) on the Ada FP8 tensor cores via ``torch._scaled_mm``. The recipe is the
"tensorwise dynamic" one validated on ``exp/fp8-pretrain-feasibility``
(f1b7fd3): per-tensor amax scaling recomputed per call, forward ``x``/``W`` in
e4m3, backward grad in e5m2, fp32 accumulate, bf16 out. 3000-step real run:
final val bf16 5.2313 / fp8 (this exemption set) 5.2292, zero NaN/Inf.

**Why this file exists.** That branch's quantizer (``_q``, reproduced here as
:func:`fp8_quantize_reference`) is ten eager kernels per call -- abs, amax,
float, mul, clamp, cast plus scalar ops, ~33 bytes of traffic per element. It
made one Linear fwd+bwd 0.55-0.69x the speed of bf16 although the raw FP8 GEMM
is ~2x. Here it is two kernels (+ a memset): an amax reduction and one
scale/clamp/cast pass, ~5 bytes per element.

**Semantics preserved exactly** (``tests/test_fp8_fused.py`` asserts the fp8
BYTES and the scale bits are equal to the reference, not ``allclose``):

* the input is first rounded to bf16 (what autocast feeds a Linear), then
* ``amax = max|t|`` clamped at 1e-12, ``s = MAX / amax``, ``q = clamp(t*s)``
  cast to fp8 with round-to-nearest-even, and the dequant scale is ``1 / s``.
  Both divisions are evaluated the way PyTorch evaluates ``float / tensor``:
  an IEEE reciprocal, then a multiply (see ``_cast_kernel``).

**The reciprocal must be ``tl.div_rn``**, not Triton's ``/``. Triton's fp32 ``/``
is the approximate ``div.full``; it is off by one ulp for some amax values, and
the scale multiplies every element, so a single ulp moves elements that sit on a
rounding boundary. The measurement-only prototype that justified this file used
``/`` and differed from eager in 1 element of 3.67M at 1792x896. Same failure
class, same fix, as the ternary quantizer (``ternary_fused.py`` #154).

**Unlike #154, the reduction is in Triton.** #154 kept its absmean in PyTorch
because a SUM depends on reduction order. A MAX does not: it is exact and
order-free, so ``atomic_max`` over block maxima gives the same bits as
``torch.amax`` on every run.

**Transposed copies are written in the same pass.** On Ada ``_scaled_mm`` only
takes row-major A and column-major B, so the backward needs ``x`` and the
grad in BOTH layouts (dW = g^T @ x). The prototype made those with separate
``.t().contiguous()`` passes, which the debugger measured at roughly one GEMM's
time each at 4096x896. The cast kernel writes the transposed copy from the same
registers instead.

**The weight is quantized once per optimizer step**, not per micro-batch and
per checkpoint recompute (16x per step at grad_accum 8). Same cache contract as
``ternary_step_cache`` (``hvn/edge/ternary.py``): keyed on the step epoch +
storage pointer + version counter, and every optimizer step (pre and post
hooks) and BCD switch drops every entry. Only TRAINABLE weights are cached; a
BCD-frozen weight is streamed in per use and keeping an fp8 copy of every frozen
block on the GPU would undo the offload.

Scope: training forward with grad enabled under bf16 autocast, on CUDA with
FP8 tensor cores (sm_89+). Eval / no-grad forwards keep the bf16 path, so
validation loss measures the weights, not the training arithmetic.
"""

from __future__ import annotations

import contextlib

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except Exception:                                    # pragma: no cover
    _HAS_TRITON = False

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
_FP8_MAX = {E4M3: 448.0, E5M2: 57344.0}

#: Default exemptions, matching the validated ``fp8_mixed`` arm: the decay /
#: write-strength gates (fp32-sensitive) and the vocab head stay in bf16.
DEFAULT_EXEMPT = ("alpha_proj", "beta_proj", "decoder")

#: Call counters. Python ints, cheap; tests use them to prove the path ran.
FP8_STATS = {"fwd": 0, "bwd": 0, "w_hits": 0, "w_misses": 0, "fallback_rows": 0,
             "x_hits": 0, "skipped": 0, "frozen_fused": 0}

#: #202 switch for A/B measurement: False sends BCD-frozen ternary weights back
#: through the old ternary-then-fp8 route (same bits, slower).
FROZEN_FUSED = [True]

#: Test hook: route every fp8 module through the eager reference (feasibility
#: branch semantics, no fused kernels, no weight cache) instead.
_REFERENCE_MODE = [False]


# ---------------------------------------------------------------------------
# Reference (eager) -- verbatim from exp/fp8-pretrain-feasibility f1b7fd3.
# ---------------------------------------------------------------------------
def fp8_quantize_reference(t: torch.Tensor, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Eager per-tensor fp8 quantize of ``bf16(t)``. Returns (codes, dequant scale)."""
    t = t.to(torch.bfloat16)
    maxv = _FP8_MAX[dtype]
    amax = t.detach().abs().amax().float().clamp(min=1e-12)
    s = maxv / amax
    return (t.float() * s).clamp(-maxv, maxv).to(dtype), (1.0 / s)


def _colmajor(t: torch.Tensor) -> torch.Tensor:
    return t.t().contiguous().t()


class _FP8MMReference(torch.autograd.Function):
    """The feasibility branch's ``_FP8MM``: y = x @ w.T, all GEMMs fp8, eager quant."""

    @staticmethod
    def forward(ctx, x, w):
        xq, xs = fp8_quantize_reference(x, E4M3)
        wq, ws = fp8_quantize_reference(w, E4M3)
        y = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws,
                             out_dtype=torch.bfloat16)
        ctx.save_for_backward(xq, xs, wq, ws)
        FP8_STATS["fwd"] += 1
        return y

    @staticmethod
    def backward(ctx, g):
        xq, xs, wq, ws = ctx.saved_tensors
        gq, gs = fp8_quantize_reference(g.contiguous(), E5M2)
        dx = torch._scaled_mm(gq, _colmajor(wq), scale_a=gs, scale_b=ws,
                              out_dtype=torch.bfloat16)
        dw = torch._scaled_mm(gq.t().contiguous(), _colmajor(xq), scale_a=gs,
                              scale_b=xs, out_dtype=torch.bfloat16)
        FP8_STATS["bwd"] += 1
        return dx, dw


def fp8_linear_reference(x: torch.Tensor, w_bf16: torch.Tensor, bias) -> torch.Tensor:
    """Feasibility-branch ``fp8_linear`` (no padding: callers guarantee %16 dims)."""
    lead, K, O = x.shape[:-1], x.shape[-1], w_bf16.shape[0]
    x2 = x.reshape(-1, K).to(torch.bfloat16).contiguous()
    with torch.autocast("cuda", enabled=False):
        y = _FP8MMReference.apply(x2, w_bf16.to(torch.bfloat16))
    if bias is not None:
        y = y + bias.to(torch.bfloat16)
    return y.reshape(*lead, O)


# ---------------------------------------------------------------------------
# Fused quantizer.
# ---------------------------------------------------------------------------
if _HAS_TRITON:

    @triton.jit
    def _amax_kernel(x_ptr, amax_ptr, n_elements, BLOCK: tl.constexpr):
        """amax_ptr[0] = max(|bf16(x)|) via atomic_max of per-block maxima.

        Max is exact and order-independent, so the atomic is deterministic
        (unlike an atomic SUM). NaN: Triton's max drops NaN where torch.amax
        propagates it, so a block holding a NaN publishes NaN explicitly. As a
        positive float NaN's bit pattern beats +inf in the int-ordered atomic,
        the result is NaN exactly when torch.amax's would be.
        """
        pid = tl.program_id(0)
        offs = pid * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        v = tl.load(x_ptr + offs, mask=mask, other=0.0)
        v = v.to(tl.bfloat16).to(tl.float32)
        a = tl.max(tl.abs(v), axis=0)
        has_nan = tl.max((v != v).to(tl.int32), axis=0) > 0
        a = tl.where(has_nan, float("nan"), a)
        tl.atomic_max(amax_ptr, a)

    @triton.jit
    def _cast_kernel(x_ptr, amax_ptr, out_ptr, out_t_ptr, inv_ptr, M, N,
                     MAXV: tl.constexpr, WRITE_T: tl.constexpr,
                     BM: tl.constexpr, BN: tl.constexpr):
        """out = fp8(clamp(bf16(x) * s)), s = MAXV / max(amax, 1e-12).

        ``x`` is row-major [M, N]. With WRITE_T it also writes ``out_t`` =
        out.T as a row-major [N, M] -- the layout ``_scaled_mm`` needs for the
        backward -- from the same registers, instead of a separate transpose.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        v = tl.load(x_ptr + rm[:, None] * N + rn[None, :], mask=mask, other=0.0)
        v = v.to(tl.bfloat16).to(tl.float32)
        amax = tl.load(amax_ptr)
        amax = tl.maximum(amax, 1e-12, propagate_nan=tl.PropagateNan.ALL)
        one = tl.full([], 1.0, tl.float32)
        # The reference writes `maxv / amax` with a Python float on the left,
        # which PyTorch evaluates as `amax.reciprocal() * maxv` -- TWO
        # roundings, not one IEEE division. Computing 448/amax directly (even
        # correctly rounded) gives a different last bit of the scale for some
        # amax: measured on 3 of 8 test tensors, moving 18 codes at 896x50272.
        # The reciprocal itself must be IEEE round-to-nearest (div_rn): `/` in
        # Triton is approximate, which is the 1-in-3.67M mismatch the
        # prototype showed at 1792x896.
        s = tl.div_rn(one, amax) * MAXV
        q = tl.clamp(v * s, -MAXV, MAXV, propagate_nan=tl.PropagateNan.ALL)
        q8 = q.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + rm[:, None] * N + rn[None, :], q8, mask=mask)
        if WRITE_T:
            tl.store(out_t_ptr + rn[None, :] * M + rm[:, None], q8, mask=mask)
        if (pid_m == 0) & (pid_n == 0):
            # `1.0 / s` is likewise s.reciprocal() * 1.0 == div_rn(1, s).
            tl.store(inv_ptr, tl.div_rn(one, s))

    @triton.jit
    def _ternary_fp8_kernel(w_ptr, scale_ptr, out_ptr, out_t_ptr, flag_ptr, inv_ptr, M, N,
                            MAXV: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr):
        """fp8 codes of a per-tensor TERNARY weight, straight from the latent weight.

        Same arithmetic, element by element, as ``_quantize_kernel``
        (ternary_fused.py) followed by ``_amax_kernel`` + ``_cast_kernel``, in one
        pass. A ternary weight holds only 0 and +-bf16(scale), so its amax is
        bf16(scale) whenever any code is non-zero, and the scale needs no
        reduction pass. ``flag`` records whether any code is non-zero; ``inv``
        gets the dequant scale for both cases (amax = bf16(scale), amax = 0), and
        the caller picks one -- the all-zero case only changes the scale, every
        code is 0 either way.
        """
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        rm = pid_m * BM + tl.arange(0, BM)
        rn = pid_n * BN + tl.arange(0, BN)
        mask = (rm[:, None] < M) & (rn[None, :] < N)
        w = tl.load(w_ptr + rm[:, None] * N + rn[None, :], mask=mask, other=0.0).to(tl.float32)
        scale = tl.load(scale_ptr).to(tl.float32)
        # _quantize_kernel, verbatim: code, clamp, rescale, canonical +0.
        q = tl.extra.cuda.libdevice.rint(tl.div_rn(w, scale))
        q = tl.minimum(tl.maximum(q, -1.0), 1.0)
        t = q * scale
        t = t + 0.0
        # ... stored as bf16 (the weight the fp8 quantizer reads) ...
        v = t.to(tl.bfloat16).to(tl.float32)
        # ... and _cast_kernel with amax = bf16(scale).
        one = tl.full([], 1.0, tl.float32)
        amax = scale.to(tl.bfloat16).to(tl.float32)
        amax = tl.maximum(amax, 1e-12, propagate_nan=tl.PropagateNan.ALL)
        s = tl.div_rn(one, amax) * MAXV
        q8 = tl.clamp(v * s, -MAXV, MAXV, propagate_nan=tl.PropagateNan.ALL)
        q8 = q8.to(out_ptr.dtype.element_ty)
        tl.store(out_ptr + rm[:, None] * N + rn[None, :], q8, mask=mask)
        tl.store(out_t_ptr + rn[None, :] * M + rm[:, None], q8, mask=mask)
        nz = tl.max(tl.max(((q != 0.0) & mask).to(tl.int32), axis=1), axis=0)
        tl.atomic_max(flag_ptr, nz)
        if (pid_m == 0) & (pid_n == 0):
            tl.store(inv_ptr, tl.div_rn(one, s))
            s0 = tl.div_rn(one, tl.full([], 1e-12, tl.float32)) * MAXV
            tl.store(inv_ptr + 1, tl.div_rn(one, s0))


def fp8_quantize(t: torch.Tensor, dtype, transpose: bool = False):
    """Fused per-tensor fp8 quantize of ``bf16(t)`` for a 2-D ``t``.

    Returns ``(q, inv_scale, q_t)``; ``q_t`` is ``q.T`` materialised row-major
    when ``transpose`` (else None). Bit-identical to
    :func:`fp8_quantize_reference` -- codes, scale and NaN behaviour. Falls back
    to the reference on CPU / without Triton.
    """
    if not (_HAS_TRITON and t.is_cuda):
        q, inv = fp8_quantize_reference(t, dtype)
        return q, inv, (q.t().contiguous() if transpose else None)
    assert t.dim() == 2, t.shape
    if not t.is_contiguous() and t.t().is_contiguous():
        # A transposed view: the fused causal conv returns ``y.transpose(1, 2)``,
        # so q/k/v/gate_proj all get a column-major x (#195). Quantize the
        # row-major ``t.t()`` instead, writing its transposed copy, and swap the
        # two outputs: same elements, so the same amax and scale, and the
        # "transposed copy" of t.t() is q(t) row-major -- the same bytes as the
        # path below, minus a bf16 ``.contiguous()`` copy of the whole tensor.
        q_t, inv, q = fp8_quantize(t.t(), dtype, transpose=True)
        return q, inv, (q_t if transpose else None)
    t = t.contiguous()
    M, N = t.shape
    amax = torch.zeros((), device=t.device, dtype=torch.float32)
    inv = torch.empty((), device=t.device, dtype=torch.float32)
    out = torch.empty((M, N), device=t.device, dtype=dtype)
    out_t = torch.empty((N, M), device=t.device, dtype=dtype) if transpose else out
    n = t.numel()
    BLOCK = 4096
    _amax_kernel[(triton.cdiv(n, BLOCK),)](t, amax, n, BLOCK=BLOCK)
    BM, BN = 64, 64
    _cast_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        t, amax, out, out_t, inv, M, N, MAXV=_FP8_MAX[dtype],
        WRITE_T=transpose, BM=BM, BN=BN)
    return out, inv, (out_t if transpose else None)


def ternary_fp8_frozen(w: torch.Tensor, scale: torch.Tensor):
    """``fp8_quantize(ternary(w, scale).to(bf16), E4M3, transpose=True)`` in one pass (#202).

    For a BCD-frozen per-tensor TernaryLinear weight, which is quantized on
    every use (forward and checkpoint recompute, every micro-batch): the old
    route writes the bf16 ternary weight, reads it for the amax, reads it again
    for the cast. Here the latent weight is read once and the fp8 codes and
    their transposed copy are written directly; no bf16 temporary. Bit-identical
    codes and scale (tests/test_fp8_fused.py). CUDA + Triton only.
    """
    assert w.dim() == 2 and w.is_contiguous() and w.is_cuda, (w.shape, w.stride())
    M, N = w.shape
    out = torch.empty((M, N), device=w.device, dtype=E4M3)
    out_t = torch.empty((N, M), device=w.device, dtype=E4M3)
    flag = torch.zeros((), device=w.device, dtype=torch.int32)
    inv = torch.empty(2, device=w.device, dtype=torch.float32)
    BM, BN = 64, 64
    _ternary_fp8_kernel[(triton.cdiv(M, BM), triton.cdiv(N, BN))](
        w, scale, out, out_t, flag, inv, M, N, MAXV=_FP8_MAX[E4M3], BM=BM, BN=BN)
    # All codes zero (all-zero weight): the reference's amax is 0, clamped to
    # 1e-12. Only the scale differs; every code is 0 either way.
    return out, torch.where(flag > 0, inv[0], inv[1]), out_t


#: One-entry cache of the last activation quantized by ``_FP8Linear`` (#195):
#: q/k/v/gate_proj read the same x, and each used to quantize it again.
#: ``(key, x_ref, (xq, xs, xq_t))``. The stale-cache hazard is a hit on a
#: tensor whose bytes are not the cached ones, so the key holds everything that
#: can change them -- storage address, version counter (any in-place write to
#: x or a view of it bumps it), shape, stride, dtype, device -- and ``x_ref``
#: is a strong reference (``detach()``: same storage and version counter, no
#: autograd graph kept alive), so the address cannot be freed and reused by a
#: different tensor while it is cached. Dropped at every optimizer step and BCD
#: switch with the weight caches (``ternary._invalidate_step_cache``).
_X_CACHE = [None]


def _quantize_input(x: torch.Tensor, transpose: bool):
    if x.is_inference():             # no version counter: freshness unprovable
        return fp8_quantize(x, E4M3, transpose=transpose)
    key = (x.data_ptr(), x._version, tuple(x.shape), x.stride(), x.dtype, x.device)
    c = _X_CACHE[0]
    if c is not None and c[0] == key and (c[2][2] is not None or not transpose):
        FP8_STATS["x_hits"] += 1
        return c[2]
    val = fp8_quantize(x, E4M3, transpose=transpose)
    _X_CACHE[0] = (key, x.detach(), val)
    return val


class _FP8Linear(torch.autograd.Function):
    """y = x @ W.T with x [N,K] (bf16 or fp32), fp8 weight supplied pre-quantized.

    ``w`` is only the autograd handle for the weight Parameter; its data is not
    read. Its gradient is dW cast to ``w.dtype`` -- exactly what the STE /
    autocast-cast chain of the bf16 path does (identity, then ToCopyBackward).
    """

    @staticmethod
    def forward(ctx, x, w, wq, ws, wq_t):
        xq, xs, xq_t = _quantize_input(x, transpose=w.requires_grad)
        # B = wq.t(): [K, O] column-major, as _scaled_mm requires on Ada.
        y = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws,
                             out_dtype=torch.bfloat16)
        # Save x^T, not x: x's row-major copy is only needed by the GEMM above,
        # the dW GEMM wants x column-major (= x^T row-major). Same bytes saved.
        ctx.save_for_backward(xq_t, xs, wq_t, ws)
        ctx.x_dtype, ctx.w_dtype = x.dtype, w.dtype
        FP8_STATS["fwd"] += 1
        return y

    @staticmethod
    def backward(ctx, g):
        xq_t, xs, wq_t, ws = ctx.saved_tensors
        need_x, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
        gq, gs, gq_t = fp8_quantize(g, E5M2, transpose=need_w)
        dx = dw = None
        if need_x:
            dx = torch._scaled_mm(gq, wq_t.t(), scale_a=gs, scale_b=ws,
                                  out_dtype=torch.bfloat16).to(ctx.x_dtype)
        if need_w:
            dw = torch._scaled_mm(gq_t, xq_t.t(), scale_a=gs, scale_b=xs,
                                  out_dtype=torch.bfloat16).to(ctx.w_dtype)
        FP8_STATS["bwd"] += 1
        return dx, dw, None, None, None


class _FP8LinearOutputUnread(_FP8Linear):
    """``_FP8Linear`` for a FROZEN weight whose output nothing reads (#195).

    In a frozen offloaded block's checkpoint recompute, out_proj and the last
    FFN Linear only feed dropout + the residual sum, whose backwards read no
    values, and the recomputed block output is discarded -- so their GEMM is
    dead work. Only the saved tensors matter, and non-reentrant checkpoint
    checks their count, shape, dtype and device against the first call, so this
    saves the same layout ``_FP8Linear.forward`` does with a frozen weight:
    ``(None, xs, wq_t, ws)``. ``ws`` stands in for ``xs`` (both 0-d fp32 on the
    same device); backward never reads ``xs`` because a frozen weight has no dW.
    """

    @staticmethod
    def forward(ctx, x, w, wq, ws, wq_t):
        assert not w.requires_grad, "dW needs x; only a frozen weight may skip"
        ctx.save_for_backward(None, ws, wq_t, ws)
        ctx.x_dtype, ctx.w_dtype = x.dtype, w.dtype
        FP8_STATS["fwd"] += 1
        FP8_STATS["skipped"] += 1
        return x.new_empty((x.shape[0], wq.shape[0]), dtype=torch.bfloat16)


#: ids of the Linears whose GEMM may be skipped right now: set only while
#: EdgeHVN re-runs a frozen offloaded block inside backward (see
#: :func:`skip_unread_outputs`); empty everywhere else.
_SKIP_IDS = [frozenset()]


@contextlib.contextmanager
def skip_unread_outputs(block: nn.Module):
    """Within: skip the GEMM of ``block``'s output-unread fp8 Linears.

    Blocks declare those Linears via ``fp8_unread_output_linears()``. Only for
    a checkpoint RECOMPUTE of a frozen block: there the block's output is
    discarded. Never wrap a forward whose output is used.
    """
    ids = set()
    for m in block.modules():
        f = getattr(m, "fp8_unread_output_linears", None)
        if f is not None:
            ids.update(id(lin) for lin in f())
    prev = _SKIP_IDS[0]
    _SKIP_IDS[0] = frozenset(ids)
    try:
        yield
    finally:
        _SKIP_IDS[0] = prev


# ---------------------------------------------------------------------------
# Module integration.
# ---------------------------------------------------------------------------
def _bf16_weight(mod: nn.Module) -> torch.Tensor:
    """The bf16 weight the bf16 path would multiply by (ternary: dequantized)."""
    from .ternary import TernaryLinear
    if isinstance(mod, TernaryLinear):
        return mod._dequantized(torch.bfloat16)
    return mod.weight.to(torch.bfloat16)


def _frozen_ternary_scale(mod: nn.Module):
    """The per-step ternary scale if ``_bf16_weight(mod)`` would be the fused
    frozen-weight quantize (``TernaryLinear._quantize_raw``'s fused branch with
    the #199 frozen scale), else None. Mirrors those conditions exactly, so
    every other case keeps the old route."""
    from . import ternary as T
    if not (_HAS_TRITON and isinstance(mod, T.TernaryLinear)):
        return None
    w = mod.weight
    if (mod._frozen_codes is not None or w.requires_grad or not w.is_cuda
            or not (T._USE_FUSED_TERNARY or mod.use_fused)
            or mod.group_size != T.GROUP_PER_TENSOR or not mod.use_step_cache
            or w.dim() != 2 or not w.is_contiguous()):
        return None
    return mod._frozen_step_scale(w)


def _fp8_weight(mod: nn.Module):
    """(wq, ws, wq_t) for ``mod``, from the per-step cache when trainable."""
    from . import ternary as T
    w = mod.weight
    cacheable = mod.fp8_weight_cache and w.requires_grad
    if cacheable:
        key = (T._STEP_EPOCH[0], w.data_ptr(), w._version, w.dtype, w.device)
        c = mod._f8cache
        if c is not None and c[0] == key:
            FP8_STATS["w_hits"] += 1
            return c[1]
    with torch.no_grad():
        scale = None if cacheable or not FROZEN_FUSED[0] else _frozen_ternary_scale(mod)
        if scale is not None:
            wq, ws, wq_t = ternary_fp8_frozen(w, scale)
            FP8_STATS["frozen_fused"] += 1
        else:
            wq, ws, wq_t = fp8_quantize(_bf16_weight(mod), E4M3, transpose=True)
    if cacheable:
        mod._f8cache = (key, (wq, ws, wq_t))
        T._STEP_LIVE.add(mod)
        FP8_STATS["w_misses"] += 1
    return wq, ws, wq_t


def fp8_active(mod: nn.Module, x: torch.Tensor) -> bool:
    """Take the fp8 path? Training forward, CUDA, bf16 autocast only.

    ``training or grad`` is TernaryLinear's own training-branch test: a
    reentrant checkpoint runs its first pass under no_grad, and that pass must
    compute the same thing as the recompute. Eval under no_grad stays bf16.
    """
    if not (mod.use_fp8 and (mod.training or torch.is_grad_enabled()) and x.is_cuda
            and torch.is_autocast_enabled("cuda")
            and torch.get_autocast_dtype("cuda") == torch.bfloat16):
        return False
    # The dW GEMM contracts over the token count, and _scaled_mm needs that
    # %16. batch*seq always is in pretraining; a ragged call falls back to
    # bf16 for that call and is counted, never silently mis-shaped.
    if (x.numel() // x.shape[-1]) % 16:
        FP8_STATS["fallback_rows"] += 1
        return False
    return True


def fp8_module_forward(mod: nn.Module, x: torch.Tensor) -> torch.Tensor:
    if _REFERENCE_MODE[0]:
        return fp8_linear_reference(x, _bf16_weight(mod), mod.bias)
    lead, K = x.shape[:-1], x.shape[-1]
    wq, ws, wq_t = _fp8_weight(mod)
    x2 = x.reshape(-1, K)
    fn = _FP8Linear
    if id(mod) in _SKIP_IDS[0] and not mod.weight.requires_grad:
        fn = _FP8LinearOutputUnread
    with torch.autocast("cuda", enabled=False):
        y = fn.apply(x2, mod.weight, wq, ws, wq_t)
    if mod.bias is not None:
        y = y + mod.bias.to(torch.bfloat16)
    return y.reshape(*lead, wq.shape[0])


class FP8Linear(nn.Linear):
    """``nn.Linear`` whose training forward can run in fp8.

    Installed by swapping ``__class__`` on an existing ``nn.Linear`` (no new
    parameters, identical state_dict keys), so a checkpoint neither knows nor
    cares. TernaryLinear has the same switch built in.
    """

    use_fp8 = False
    fp8_weight_cache = False
    _f8cache = None

    def forward(self, x):
        if fp8_active(self, x):
            return fp8_module_forward(self, x)
        return super().forward(x)

    def train(self, mode: bool = True):
        self._f8cache = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._f8cache = None
        return super()._load_from_state_dict(*args, **kwargs)


def fp8_supported() -> bool:
    return (_HAS_TRITON and torch.cuda.is_available()
            and torch.cuda.get_device_capability() >= (8, 9))


def enable_fp8_linear(model: nn.Module, exempt: tuple[str, ...] = DEFAULT_EXEMPT,
                      weight_cache: bool = True) -> tuple[list[str], list[str]]:
    """Route eligible Linear / TernaryLinear layers through the fp8 path.

    Skips names matching ``exempt`` and any layer whose dims are not multiples
    of 16 (a ``_scaled_mm`` requirement; padding is deliberately not done --
    every target projection in this model is already aligned). Returns
    ``(enabled, skipped_for_shape)``.
    """
    from .ternary import TernaryLinear, _ensure_step_hooks
    if not fp8_supported():
        raise RuntimeError("--fp8-linear needs CUDA FP8 tensor cores (sm_89+) and Triton")
    if weight_cache:
        _ensure_step_hooks()
    enabled, skipped = [], []
    for name, mod in model.named_modules():
        if not (type(mod) is nn.Linear or isinstance(mod, (FP8Linear, TernaryLinear))):
            continue
        if any(e in name for e in exempt):
            continue
        if mod.in_features % 16 or mod.out_features % 16:
            skipped.append(name)
            continue
        if type(mod) is nn.Linear:
            mod.__class__ = FP8Linear
        mod.use_fp8 = True
        mod.fp8_weight_cache = bool(weight_cache)
        mod._f8cache = None
        enabled.append(name)
    return enabled, skipped
