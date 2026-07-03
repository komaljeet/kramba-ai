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

import torch
import torch.nn as nn
import torch.nn.functional as F


def ternary_quantize_ste(w: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
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

    Returns
    -------
    torch.Tensor
        Same shape as *w*.  Values are exactly ``{-scale, 0, +scale}``, but
        the tensor carries a gradient identical to ``w``'s (straight-through).
    """
    scale = w.detach().abs().mean().clamp(min=eps)
    w_ternary = torch.clamp(torch.round(w / scale), -1, 1) * scale
    return w + (w_ternary - w).detach()


def ternary_codes(w: torch.Tensor, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the raw ``{-1, 0, +1}`` codes and scale for a weight tensor (no STE, inference-only)."""
    scale = w.detach().abs().mean().clamp(min=eps)
    codes = torch.clamp(torch.round(w.detach() / scale), -1, 1).to(torch.int8)
    return codes, scale


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
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self._frozen_codes: torch.Tensor | None = None
        self._frozen_scale: torch.Tensor | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self._frozen_codes is not None:
            w = self._frozen_codes.to(x.dtype) * self._frozen_scale
        else:
            w = ternary_quantize_ste(self.weight)
        return F.linear(x, w, self.bias)

    def freeze(self) -> None:
        """Discard the full-precision shadow weight, keeping only ternary codes + scale.

        This is the actual 1.58-bit deployment form.  The module remains
        usable for inference (:meth:`forward`) but is no longer trainable.
        """
        codes, scale = ternary_codes(self.weight)
        self._frozen_codes = codes
        self._frozen_scale = scale
        self.weight.requires_grad_(False)

    def memory_bytes(self) -> int:
        """Storage footprint at 1.58 bits/weight (packed) + a single fp32 scale."""
        import math
        bits_per_weight = math.log2(3)
        weight_bytes = self.weight.numel() * bits_per_weight / 8.0
        bias_bytes = self.bias.numel() * 4 if self.bias is not None else 0
        return int(weight_bytes + 4 + bias_bytes)

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "TernaryLinear":
        """Build a (still-trainable) :class:`TernaryLinear` seeded from a dense layer.

        Useful for a short fine-tuning "recovery" pass after converting an
        already-trained model (as opposed to training ternary from scratch).
        """
        module = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
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


def apply_ternary_to_model(model: nn.Module, target_names: list[str] | None = None) -> dict[str, "TernaryLinear"]:
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
        ternary_layer = TernaryLinear.from_linear(linear_modules[name])
        setattr(parent, attr, ternary_layer)
        results[name] = ternary_layer

    return results
