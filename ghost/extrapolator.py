"""GHOST Phase 2 — weight-delta extrapolator (issue #126, epic #124).

Written by: Rambarun Komaljeet

Takes three consecutive checkpoints from a stable (inter-transition) stretch of
training, measures how each weight tensor is moving, and predicts where it will
be N steps later — so those N steps never have to be computed.

    D1 = W2 - W1        D2 = W3 - W2        f = N / gap

    linear      W_pred = W3 + f * D2
    polynomial  W_pred = W3 + f * D2 + (f * (f + 1) / 2) * (D2 - D1)

Per-tensor gating is the whole game. A tensor whose two observed deltas point
in a consistent direction with a stable magnitude is on a trajectory worth
extrapolating; one whose deltas are uncorrelated is being driven by gradient
noise and extrapolating it just injects that noise, amplified by ``f``. The
latter are HELD at their last known value (W3), per the #126 spec.

Consistency is measured per tensor as

    cos   = <D1, D2> / (|D1| |D2|)     directional agreement
    ratio = |D2| / |D1|                magnitude stability

and a tensor is extrapolated only when ``cos >= cos_threshold`` and ``ratio``
lies inside ``[1/ratio_tol, ratio_tol]``.

This module is pure tensor arithmetic — no model, no data, no GPU. The loss
evaluation that scores its predictions lives in ``ghost/harness.py``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch

__all__ = [
    "ExtrapolationConfig",
    "TensorPlan",
    "ExtrapolationPlan",
    "build_plan",
    "extrapolate",
    "extrapolate_state_dict",
]

Tensors = Mapping[str, torch.Tensor]


# --------------------------------------------------------------------------


@dataclass
class ExtrapolationConfig:
    """Gating thresholds for which tensors may be extrapolated."""

    cos_threshold: float = 0.30    # min directional agreement between D1 and D2
    ratio_tol: float = 2.0         # |D2|/|D1| must lie in [1/tol, tol]
    poly_cos_threshold: float = 0.90  # above this, a quadratic term is allowed
    allow_poly: bool = True
    eps: float = 1e-12             # dead-tensor guard (|D1| or |D2| ~ 0)


@dataclass
class TensorPlan:
    """The decision made for one weight tensor."""

    name: str
    numel: int
    cos: float
    ratio: float
    d1_norm: float
    d2_norm: float
    mode: str          # "linear" | "poly" | "hold"
    reason: str

    @property
    def extrapolated(self) -> bool:
        return self.mode in ("linear", "poly")


@dataclass
class ExtrapolationPlan:
    """Per-tensor decisions plus the aggregate coverage the #126 spec asks for."""

    config: dict
    tensors: List[TensorPlan]

    @property
    def n_extrapolated(self) -> int:
        return sum(1 for t in self.tensors if t.extrapolated)

    @property
    def n_held(self) -> int:
        return sum(1 for t in self.tensors if not t.extrapolated)

    @property
    def params_extrapolated(self) -> int:
        return sum(t.numel for t in self.tensors if t.extrapolated)

    @property
    def params_total(self) -> int:
        return sum(t.numel for t in self.tensors)

    @property
    def param_fraction_extrapolated(self) -> float:
        total = self.params_total
        return (self.params_extrapolated / total) if total else 0.0

    def summary(self) -> dict:
        return {
            "config": self.config,
            "n_tensors": len(self.tensors),
            "n_extrapolated": self.n_extrapolated,
            "n_held": self.n_held,
            "params_total": self.params_total,
            "params_extrapolated": self.params_extrapolated,
            "param_fraction_extrapolated": self.param_fraction_extrapolated,
            "tensors": [asdict(t) for t in self.tensors],
        }


# --------------------------------------------------------------------------


def _stats(d1: torch.Tensor, d2: torch.Tensor, eps: float) -> Tuple[float, float, float, float]:
    """Directional agreement and magnitude ratio of two deltas."""
    a = d1.detach().to(torch.float64).flatten()
    b = d2.detach().to(torch.float64).flatten()
    na = float(torch.linalg.vector_norm(a))
    nb = float(torch.linalg.vector_norm(b))
    if na <= eps or nb <= eps:
        return 0.0, 0.0, na, nb
    cos = float(torch.dot(a, b)) / (na * nb)
    return cos, nb / na, na, nb


def build_plan(
    w1: Tensors,
    w2: Tensors,
    w3: Tensors,
    cfg: Optional[ExtrapolationConfig] = None,
) -> ExtrapolationPlan:
    """Decide, per tensor, whether to extrapolate and with which fit."""
    cfg = cfg or ExtrapolationConfig()
    plans: List[TensorPlan] = []

    for name in w3:
        t3 = w3[name]
        if name not in w1 or name not in w2:
            plans.append(TensorPlan(name, t3.numel(), 0.0, 0.0, 0.0, 0.0,
                                    "hold", "absent from an earlier checkpoint"))
            continue
        if not t3.is_floating_point():
            plans.append(TensorPlan(name, t3.numel(), 0.0, 0.0, 0.0, 0.0,
                                    "hold", f"non-float dtype {t3.dtype}"))
            continue

        d1 = (w2[name] - w1[name]).to(torch.float64)
        d2 = (t3 - w2[name]).to(torch.float64)
        cos, ratio, n1, n2 = _stats(d1, d2, cfg.eps)

        if n1 <= cfg.eps or n2 <= cfg.eps:
            mode, reason = "hold", "tensor is not moving (zero delta)"
        elif not math.isfinite(cos) or not math.isfinite(ratio):
            mode, reason = "hold", "non-finite delta statistics"
        elif cos < cfg.cos_threshold:
            mode, reason = "hold", f"cos {cos:.3f} < {cfg.cos_threshold} (noise-driven)"
        elif not (1.0 / cfg.ratio_tol <= ratio <= cfg.ratio_tol):
            mode, reason = "hold", f"|D2|/|D1| {ratio:.3f} outside magnitude tolerance"
        elif cfg.allow_poly and cos >= cfg.poly_cos_threshold:
            mode, reason = "poly", f"cos {cos:.3f} >= {cfg.poly_cos_threshold}"
        else:
            mode, reason = "linear", f"cos {cos:.3f}, ratio {ratio:.3f}"

        plans.append(TensorPlan(name, t3.numel(), cos, ratio, n1, n2, mode, reason))

    return ExtrapolationPlan(config=asdict(cfg), tensors=plans)


def extrapolate(
    w1: Tensors,
    w2: Tensors,
    w3: Tensors,
    n_steps: int,
    gap: int,
    cfg: Optional[ExtrapolationConfig] = None,
    plan: Optional[ExtrapolationPlan] = None,
) -> Tuple[Dict[str, torch.Tensor], ExtrapolationPlan]:
    """Predict the weights ``n_steps`` optimizer steps past ``w3``.

    ``gap`` is the step spacing between consecutive input checkpoints, so
    ``f = n_steps / gap`` is the prediction horizon in units of one observed
    delta. ``f`` may be fractional (N=10 against a 50-step checkpoint cadence
    is f=0.2) — that is the regime the #126 gate scores.
    """
    cfg = cfg or ExtrapolationConfig()
    plan = plan or build_plan(w1, w2, w3, cfg)
    if gap <= 0:
        raise ValueError(f"gap must be positive, got {gap}")
    f = n_steps / float(gap)

    by_name = {t.name: t for t in plan.tensors}
    out: Dict[str, torch.Tensor] = {}
    for name, t3 in w3.items():
        tp = by_name.get(name)
        if tp is None or tp.mode == "hold":
            out[name] = t3.clone()
            continue
        d1 = (w2[name] - w1[name]).to(torch.float64)
        d2 = (t3 - w2[name]).to(torch.float64)
        step = f * d2
        if tp.mode == "poly":
            step = step + (f * (f + 1.0) / 2.0) * (d2 - d1)
        out[name] = (t3.to(torch.float64) + step).to(t3.dtype)
    return out, plan


def extrapolate_state_dict(
    ckpts: Sequence[Mapping],
    n_steps: int,
    cfg: Optional[ExtrapolationConfig] = None,
    key: str = "model_state_dict",
) -> Tuple[dict, ExtrapolationPlan]:
    """Extrapolate three loaded checkpoint dicts into a new checkpoint dict.

    ``gap`` is read from the checkpoints' own ``global_step`` fields and the
    two gaps must match — unevenly spaced checkpoints would silently corrupt
    the velocity estimate, so this raises rather than guessing.

    The returned dict is a shallow copy of the newest checkpoint with its
    weights replaced and ``global_step`` advanced, so it stays loadable by
    ``train_edge_conversational.load_checkpoint_for_resume`` unchanged.
    """
    if len(ckpts) != 3:
        raise ValueError(f"need exactly 3 checkpoints, got {len(ckpts)}")
    c1, c2, c3 = ckpts
    s1, s2, s3 = (int(c.get("global_step", 0)) for c in ckpts)
    g1, g2 = s2 - s1, s3 - s2
    if g1 <= 0 or g2 <= 0:
        raise ValueError(f"checkpoints must be increasing in global_step: {s1}, {s2}, {s3}")
    if g1 != g2:
        raise ValueError(
            f"checkpoint spacing must be uniform: {s1}->{s2} is {g1} steps but "
            f"{s2}->{s3} is {g2}. Unequal gaps corrupt the velocity estimate."
        )

    weights, plan = extrapolate(c1[key], c2[key], c3[key], n_steps, g2, cfg)
    out = dict(c3)
    out[key] = weights
    out["global_step"] = s3 + n_steps
    out["ghost_extrapolated_from"] = [s1, s2, s3]
    out["ghost_n_steps"] = n_steps
    return out, plan
