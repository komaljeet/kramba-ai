"""Pluggable FFN activations for the DeltaNet block (Issue #140).

Author: Rambarun Komaljeet

The block's feed-forward sub-layer has always been ``Linear -> GELU -> Linear``
(``deltanet.py``).  This module makes that middle op selectable so the choice can
be *measured* rather than assumed, without touching the recurrence, the attention
layers, or any weight shape.

Three activations, all pointwise, none introducing a new matrix:

``gelu``
    ``nn.GELU()`` -- the incumbent.  Returned as the *same module type* the block
    used before this issue, so ``ffn_activation="gelu"`` is byte-identical to the
    pre-#140 path in both forward and backward.  This is the known-answer
    configuration (cf. the ``num_experts=1`` MoE precedent in
    docs/architecture.md): a harness that cannot reproduce the incumbent exactly
    cannot be trusted to measure a replacement for it.

``relu2``
    ``max(0, x)^2`` -- squared ReLU (Primer, So et al. 2021).  Same parameter
    count and same tensor shapes as GELU, so a GELU checkpoint loads ``strict=True``.

``powlu``
    ``sign(x) * |x|^p`` with ``p`` learnable, initialised to 1.5.  Adds exactly one
    scalar parameter per FFN.  Unlike the other two this is odd (no flat region)
    and grows superlinearly, so it is the arm most likely to destabilise -- see the
    gradient note on :class:`PowLU`.

No new dependencies; plain PyTorch throughout.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ReLU2(nn.Module):
    """Squared ReLU: ``max(0, x) ** 2``.

    Written as ``relu(x) * relu(x)`` rather than ``relu(x) ** 2``: the pow
    overload dispatches to a slower generic kernel in eager mode, and the two are
    mathematically identical here since ``relu(x) >= 0``.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r = torch.nn.functional.relu(x)
        return r * r

    def extra_repr(self) -> str:
        return "max(0,x)^2"


class PowLU(nn.Module):
    """``sign(x) * |x|^p``, with ``p`` a learnable scalar (init 1.5).

    Parameters
    ----------
    p_init : float
        Initial exponent.  1.5 per the ablation spec.
    learnable : bool
        Whether ``p`` receives gradient.  ``False`` pins it (useful for isolating
        "is the exponent helping" from "is *learning* the exponent helping").
    eps : float
        Floor on ``|x|`` inside the power.  **Load-bearing, not cosmetic** --
        without it this module produces NaNs during training:

        * d/dx of ``|x|^p`` is ``p|x|^(p-1)``, which at ``p < 1`` blows up at 0;
        * d/dp is ``|x|^p * log|x|``, and ``log(0) = -inf`` -- so the *learnable*
          exponent alone is enough to poison the backward pass the first time any
          pre-activation lands exactly on zero.

        Clamping ``|x|`` to ``eps`` before the power bounds both.  The forward
        error introduced is at most ``eps^p`` (~3e-9 at eps=1e-6, p=1.5), far
        below bf16 resolution.

    Notes
    -----
    ``p`` is stored as a plain scalar ``nn.Parameter``.  Nothing constrains it to
    stay above 1 -- if training drives it below 1 the function develops an
    infinite slope at the origin.  That is a real failure mode this ablation is
    meant to *detect*, so it is deliberately not clamped away; the runner records
    the final value of ``p``.
    """

    def __init__(self, p_init: float = 1.5, learnable: bool = True,
                 eps: float = 1e-6):
        super().__init__()
        self.eps = float(eps)
        p = torch.tensor(float(p_init))
        self.p = nn.Parameter(p, requires_grad=bool(learnable))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mag = x.abs().clamp_min(self.eps)
        return torch.sign(x) * mag.pow(self.p.to(x.dtype))

    def extra_repr(self) -> str:
        return (f"p={float(self.p):.4g}, learnable={self.p.requires_grad}, "
                f"eps={self.eps:g}")


#: Registry of the activations this ablation covers.  Kept explicit (rather than
#: a getattr into ``torch.nn``) so an unknown name fails loudly at construction
#: instead of silently falling back to the incumbent -- a silent fallback would
#: make every arm secretly measure GELU.
_ACTIVATIONS = {
    "gelu": lambda: nn.GELU(),
    "relu2": lambda: ReLU2(),
    "powlu": lambda: PowLU(p_init=1.5, learnable=True),
}

ACTIVATION_NAMES = tuple(_ACTIVATIONS)


def make_activation(name: str) -> nn.Module:
    """Build the activation module named ``name``.

    Raises ``ValueError`` on an unknown name -- see the note on ``_ACTIVATIONS``
    for why this must not degrade gracefully.
    """
    key = (name or "gelu").lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"unknown ffn_activation {name!r}; expected one of "
            f"{', '.join(ACTIVATION_NAMES)}"
        )
    return _ACTIVATIONS[key]()
