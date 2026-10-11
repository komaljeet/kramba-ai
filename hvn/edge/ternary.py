"""
Ternary (1.58-bit) Weights — BitNet b1.58 (Issue #32, Phase 2.1)
===================================================================

Trains weights constrained to ``{-1, 0, +1}`` (Wang et al. 2023 / Ma et al.
2024, "The Era of 1-bit LLMs").  Each ternary weight costs ``log2(3) ≈
1.58`` bits, giving a 10B-parameter model a ~1.98 GB footprint.  Because
values are drawn from ``{-1, 0, +1}``, the core matmul ``H · ω_t`` in the
Biot-Savart velocity computation becomes pure addition/subtraction — no
multiplications.

Weights are quantized with a **straight-through estimator (STE)**: the
forward pass uses the ternary-quantized weight, but gradients flow to the
underlying full-precision "shadow" weight as if quantization were the
identity function.  This is what makes ternary weights trainable at all —
the true gradient of ``round()`` is zero almost everywhere.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

#: Issue #154: route TernaryLinear through the fused Triton quantizer.
#: Read at import so a run can enable it without a code change. DEFAULT OFF:
#: measured at only +1.9% in the real training loop (7.118 -> 6.985 s/step),
#: which is not worth changing the training-path quantizer for. See the note
#: in `_dequantized`.
_USE_FUSED_TERNARY = os.environ.get("HVN_FUSED_TERNARY", "0") == "1"


#: ``group_size`` sentinel: one scale for the whole tensor (BitNet b1.58 absmean).
#: The default everywhere, and byte-identical to the pre-#146 implementation.
GROUP_PER_TENSOR = 0
#: ``group_size`` sentinel: one scale per output row (per output channel).
GROUP_PER_ROW = -1


def _absmean_scale(w: torch.Tensor, group_size: int, eps: float) -> torch.Tensor:
    """Absmean scale at the requested granularity, shaped to broadcast over *w*.

    ``0`` (:data:`GROUP_PER_TENSOR`) one scalar for the whole tensor -- BitNet
    b1.58, and what this repo has always done (at d=512, 262,144 weights sharing
    one normaliser). ``-1`` (:data:`GROUP_PER_ROW`) one scale per output row.
    ``k > 0`` one scale per contiguous run of ``k`` along the input dim.
    """
    if group_size == GROUP_PER_TENSOR:
        return w.abs().mean().clamp(min=eps)
    if w.dim() != 2:
        raise ValueError(f"group_size={group_size} needs a 2-D [out, in] weight, "
                         f"got shape {tuple(w.shape)}")
    if group_size == GROUP_PER_ROW:
        return w.abs().mean(dim=1, keepdim=True).clamp(min=eps)
    if group_size < 0:
        raise ValueError(f"invalid group_size {group_size}")
    out, inn = w.shape
    if inn % group_size:
        raise ValueError(
            f"group_size={group_size} does not divide in_features={inn}. "
            f"Padding is deliberately refused: a short final group would carry a "
            f"scale computed over fewer weights, quietly making the last columns "
            f"of every row a different quantizer.")
    g = w.view(out, inn // group_size, group_size)
    return g.abs().mean(dim=2, keepdim=True).clamp(min=eps)


def _apply_scale(w: torch.Tensor, scale: torch.Tensor, group_size: int) -> torch.Tensor:
    """Round ``w`` to ``{-1,0,1} * scale`` at the granularity ``scale`` was built for."""
    if group_size in (GROUP_PER_TENSOR, GROUP_PER_ROW):
        return torch.clamp(torch.round(w / scale), -1, 1) * scale
    out, inn = w.shape
    g = w.view(out, inn // group_size, group_size)
    return (torch.clamp(torch.round(g / scale), -1, 1) * scale).view(out, inn)


def ternary_quantize_ste(w: torch.Tensor, eps: float = 1e-6,
                         group_size: int = GROUP_PER_TENSOR) -> torch.Tensor:
    """Quantize a tensor to ``{-1, 0, +1} · scale`` with a straight-through gradient.

    Uses the BitNet b1.58 "absmean" scheme: the scale is the mean absolute
    value of *w*, and each element is rounded to the nearest of
    ``{-1, 0, +1}`` after normalizing by that scale.

    Parameters
    ----------
    w : torch.Tensor
        Full-precision "shadow" weight.
    eps : float
        Numerical floor for the scale to avoid division by zero.
    group_size : int
        Scale granularity (Issue #146). ``0`` = one scale per tensor (default,
        byte-identical to the original), ``-1`` = per output row, ``k`` = per
        ``k`` inputs. Finer is a strictly better approximation of *w* but costs
        storage -- and #145 showed that being closer to the shadow weight is
        **not** the same as being a better model, so this is a lever to measure,
        never one to assume.

    Returns
    -------
    torch.Tensor
        Same shape as *w*.  Values are exactly ``{-scale, 0, +scale}``, but
        the tensor carries a gradient identical to ``w``'s (straight-through).
    """
    scale = _absmean_scale(w.detach(), group_size, eps)
    w_ternary = _apply_scale(w, scale, group_size)
    return w + (w_ternary - w).detach()


def ternary_codes(w: torch.Tensor, eps: float = 1e-6,
                  group_size: int = GROUP_PER_TENSOR
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the raw ``{-1, 0, +1}`` codes and scale for a weight tensor (no STE, inference-only).

    With a positive ``group_size`` the scale broadcasts against the codes'
    *grouped* view, so consumers must rebuild through :func:`dequantize_codes`
    rather than assuming a plain elementwise multiply.
    """
    scale = _absmean_scale(w.detach(), group_size, eps)
    if group_size in (GROUP_PER_TENSOR, GROUP_PER_ROW):
        codes = torch.clamp(torch.round(w.detach() / scale), -1, 1).to(torch.int8)
    else:
        out, inn = w.shape
        g = w.detach().view(out, inn // group_size, group_size)
        codes = torch.clamp(torch.round(g / scale), -1, 1).to(torch.int8).view(out, inn)
    return codes, scale


def dequantize_codes(codes: torch.Tensor, scale: torch.Tensor, group_size: int,
                     dtype: torch.dtype) -> torch.Tensor:
    """Rebuild the dense weight from codes + scale at any granularity."""
    if group_size in (GROUP_PER_TENSOR, GROUP_PER_ROW):
        return codes.to(dtype) * scale.to(dtype)
    out, inn = codes.shape
    g = codes.view(out, inn // group_size, group_size).to(dtype)
    return (g * scale.to(dtype)).view(out, inn)


def scale_count(out_features: int, in_features: int, group_size: int) -> int:
    """How many scale values a weight of this shape needs -- the storage price.

    Ternary's whole point is bytes, so a quality win from finer scales has to be
    judged against this rather than reported as free.
    """
    if group_size == GROUP_PER_TENSOR:
        return 1
    if group_size == GROUP_PER_ROW:
        return out_features
    return out_features * (in_features // group_size)


class TernaryLinear(nn.Module):
    """``nn.Linear`` drop-in with BitNet b1.58 ternary weights.

    Maintains a full-precision shadow weight for training (needed for the
    STE gradient), but the forward pass always uses the ternary-quantized
    value.  After training, call :meth:`freeze` to discard the shadow
    weight and store only the ``{-1,0,1}`` codes + scale (the true
    1.58-bit-per-weight deployment form).

    Parameters
    ----------
    in_features, out_features, bias : as in ``nn.Linear``.
    group_size : int
        Scale granularity (Issue #146). ``0`` (default) is one scale for the
        whole weight -- byte-identical to the pre-#146 behaviour, and the
        configuration every existing checkpoint was trained under.  ``-1`` is
        per output row, ``k`` is per ``k`` inputs.  Note this is a *training*
        knob: #145 measured that re-grouping an already-trained checkpoint makes
        it worse, so changing it on an existing model is not a free upgrade.
    """

    #: Per-module switch for the fused #154 quantizer, set by ``apply_fast_stack``
    #: under ``--fast`` (bf16 only). A class attribute, not an ``__init__`` field,
    #: so it never enters a state_dict and modules built or unpickled before it
    #: existed read False -- the same pattern as ``use_megakernel_scan``.
    use_fused = False

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 group_size: int = GROUP_PER_TENSOR):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.group_size = int(group_size)
        # Fail at construction, not at the first forward halfway through a run.
        _absmean_scale(torch.zeros(out_features, in_features), self.group_size, 1e-6)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._frozen_codes: torch.Tensor | None = None
        self._frozen_scale: torch.Tensor | None = None
        # Issue #131: dequantized weight reused across inference-only forwards.
        # A plain attribute, NOT a buffer -- it must never enter a state_dict.
        self._wcache: torch.Tensor | None = None

    def _dequantized(self, dtype: torch.dtype) -> torch.Tensor:
        if self._frozen_codes is not None:
            return dequantize_codes(self._frozen_codes, self._frozen_scale,
                                    self.group_size, dtype)
        # Issue #154: an optional fused Triton path for the default per-tensor
        # case. It replaces a six-op chain (abs/mean/clamp/div/round/clamp/mul)
        # that cProfile measured at 0.395 s of an 8.58 s step -- it runs 1,760
        # times per step (110 ternary layers x 8 micro-batches x 2 for the
        # gradient-checkpoint recompute) -- and is 4.14x faster in isolation.
        #
        # In the REAL loop it is worth only +1.9%, because the step is GPU-bound:
        # the host spends 68% of it blocked in cudaStreamSynchronize, so removing
        # host work mostly removes waiting. Default OFF at the module level;
        # ``--fast`` with bf16 turns it on per module (``use_fused``), because the
        # fused output is bit-identical to the eager one only after the bf16
        # cast (fp32 differs by ~6e-8, see tests/test_ternary_fused.py).
        #
        # Only the per-tensor path is fused; grouped scales (#146) keep the eager
        # implementation rather than silently taking a different code path.
        if ((_USE_FUSED_TERNARY or self.use_fused)
                and self.group_size == GROUP_PER_TENSOR):
            from .ternary_fused import ternary_quantize_ste_fused
            return ternary_quantize_ste_fused(self.weight).to(dtype)
        return ternary_quantize_ste(self.weight, group_size=self.group_size).to(dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training or torch.is_grad_enabled():
            # Training / grad-enabled path is unchanged and uncached: the STE
            # needs the live shadow weight inside the autograd graph, and the
            # weight moves every optimizer step.
            return F.linear(x, self._dequantized(x.dtype), self.bias)

        # Inference: the weight is frozen for the duration, so re-deriving it
        # per call is pure waste. Issue #131 measured this at ~2x of EdgeHVN's
        # entire decode wall-clock -- ``ternary_quantize_ste`` runs
        # abs/mean/div/round/clamp/mul/sub/add over the WHOLE weight matrix on
        # every token, which is the same O(d^2) order as the matvec it feeds.
        # Caching is a mathematical no-op (verified bit-identical logits, both
        # arm orders, results/matmul_free/ternary_cache_ab*.json).
        w = self._wcache
        if w is None or w.dtype != x.dtype or w.device != x.device:
            w = self._dequantized(x.dtype).detach()
            self._wcache = w
        return F.linear(x, w, self.bias)

    def invalidate_cache(self) -> None:
        """Drop the cached dequantized weight (see :meth:`train`)."""
        self._wcache = None

    def train(self, mode: bool = True):
        """Invalidate the #131 cache on ANY mode transition.

        ``nn.Module.train()`` is the single entry point for both ``.train()``
        and ``.eval()``, so clearing here covers both directions. That matters:
        the failure mode of a weight cache is silent staleness -- a model that
        trains, returns to eval, and then multiplies by pre-training weights
        while the loss curve still looks plausible. Issue #117 was exactly that
        bug (``register_cpu_home``'s register-if-absent dedup reset every block
        to its untrained init, and zero cross-cycle learning happened while
        aggregate loss still trended down), so this cache is invalidated
        aggressively rather than cleverly.
        """
        self._wcache = None
        return super().train(mode)

    def _load_from_state_dict(self, *args, **kwargs):
        self._wcache = None
        return super()._load_from_state_dict(*args, **kwargs)

    def freeze(self) -> None:
        """Discard the full-precision shadow weight, keeping only ternary codes + scale.

        This is the actual 1.58-bit deployment form.  The module remains
        usable for inference (:meth:`forward`) but is no longer trainable.
        """
        codes, scale = ternary_codes(self.weight, group_size=self.group_size)
        self._frozen_codes = codes
        self._frozen_scale = scale
        self._wcache = None  # #131: freezing changes the weight source
        self.weight.requires_grad_(False)

    def memory_bytes(self) -> int:
        """Storage footprint at 1.58 bits/weight (packed) + the fp32 scales.

        Issue #146: the scale term is no longer a constant 4 bytes. At
        ``group_size=32`` a 512x512 weight carries 8,192 scales -- ~32 KiB
        against ~51 KiB of codes, i.e. **+63% storage**, which is large enough
        to decide whether a finer scheme is worth its quality gain. Counting it
        as one scalar would have made every grouped scheme look free.
        """
        import math
        bits_per_weight = math.log2(3)
        weight_bytes = self.weight.numel() * bits_per_weight / 8.0
        scale_bytes = scale_count(self.out_features, self.in_features,
                                  self.group_size) * 4
        bias_bytes = self.bias.numel() * 4 if self.bias is not None else 0
        return int(weight_bytes + scale_bytes + bias_bytes)

    @classmethod
    def from_linear(cls, linear: nn.Linear,
                    group_size: int = GROUP_PER_TENSOR) -> "TernaryLinear":
        """Build a (still-trainable) :class:`TernaryLinear` seeded from a dense layer.

        Useful for a short fine-tuning "recovery" pass after converting an
        already-trained model (as opposed to training ternary from scratch).
        """
        module = cls(linear.in_features, linear.out_features,
                     bias=linear.bias is not None, group_size=group_size)
        module.weight.data = linear.weight.data.clone()
        if module.bias is not None:
            module.bias.data = linear.bias.data.clone()
        module.to(linear.weight.device)
        return module


class TernaryHamiltonian:
    """Ternary-quantized Hamiltonian state, for inference-time memory savings.

    Mirrors the API of ``QuantizedHamiltonian`` in ``hvn/quantize.py`` but
    at 1.58 bits/element instead of 4 or 8.  Intended for the runtime
    Hamiltonian *state* tensors (not the learnable projection weights,
    which should use :class:`TernaryLinear` if trained ternary from
    scratch).

    Parameters
    ----------
    H : torch.Tensor
        Hamiltonian state tensor to quantize.
    """

    def __init__(self, H: torch.Tensor):
        self.codes, self.scale = ternary_codes(H)
        self._shape = tuple(H.shape)

    def dequantize(self, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        return (self.codes.to(dtype) * self.scale.to(dtype))

    def memory_bytes(self) -> int:
        import math
        return int(self.codes.numel() * math.log2(3) / 8.0) + 4

    @property
    def shape(self) -> tuple[int, ...]:
        return self._shape


def apply_ternary_to_model(model: nn.Module, target_names: list[str] | None = None,
                           group_size: int = GROUP_PER_TENSOR) -> dict[str, "TernaryLinear"]:
    """Replace named ``nn.Linear`` submodules in *model* with (trainable) :class:`TernaryLinear`.

    The new layers are seeded from the existing dense weights so the model
    can immediately continue training/fine-tuning with the STE gradient —
    this is the practical route to "ternary from scratch" quality without
    a multi-day full retrain: a short recovery fine-tune (see
    ``hvn/edge/recovery.py``) closes most of the gap.

    Parameters
    ----------
    model : nn.Module
        Model to modify in-place.
    target_names : list[str], optional
        Dotted module names of the ``nn.Linear`` layers to convert.  If
        ``None``, every ``nn.Linear`` in the model is converted.
    group_size : int
        Scale granularity for every converted layer (Issue #146).  ``0``
        (default) reproduces the pre-#146 behaviour exactly, which is what every
        existing checkpoint was trained under.

    Returns
    -------
    dict[str, TernaryLinear]
        Layer name -> the new ternary module.
    """
    linear_modules = {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
    names = target_names or list(linear_modules.keys())
    results: dict[str, TernaryLinear] = {}

    for name in names:
        if name not in linear_modules:
            continue
        *parent_path, attr = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        ternary_layer = TernaryLinear.from_linear(linear_modules[name],
                                                  group_size=group_size)
        setattr(parent, attr, ternary_layer)
        results[name] = ternary_layer

    return results
