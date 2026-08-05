"""GHOST Phase 2 follow-up — window fit (issue #126, epic #124).

Written by: Rambarun Komaljeet

#126 predicted the future weights from the LAST observed delta, rescaled:

    W_pred = W3 + (N / gap) * D2

That is one noisy sample of the velocity, multiplied by N. It loses to doing
nothing at every horizon, and a cheating oracle allowed to pick the best scalar
multiple of D2 also loses. But the oracle only searched *how much of D2* to
apply — it never searched a different direction, so it does not bound this:

    observe a WINDOW of K consecutive steps and least-squares fit each weight
    tensor's motion against step number, then evaluate the fit N steps ahead.

The distinction that makes this worth measuring: a fit over K steps AVERAGES
the per-step deltas, and averaging is the arithmetic that cancels their noise
rather than amplifying it. That is the same mechanism behind the one positive
result in this epic (#128 weight averaging, median delta cos +0.890 at 1-step
spacing). A single delta at that spacing is roughly half signal; the mean of K
of them is much better conditioned, so the fitted DIRECTION may be usable even
though a single delta's is not.

Model
-----
Per tensor, with ``u = t - t_anchor`` (so the anchor is u = 0 and the window is
u in [-(K-1), 0]), fit each coordinate independently:

    W(u) ~= c_0 + c_1 * u + ... + c_D * u^D

Two predictions are worth scoring separately, and they are NOT the same thing:

    fit_eval       W_pred = sum_d c_d * N^d
                   the fit's own value. Note c_0 is a SMOOTHED anchor, not the
                   real one, so at N=0 this is a weighted average of the window
                   — i.e. it inherits the #128 averaging gain for free, and any
                   result at N>0 has to be read against that.

    anchor_locked  W_pred = W_anchor + sum_{d>=1} c_d * N^d
                   keeps the real anchor and steps along the denoised velocity.
                   This isolates the extrapolation from the averaging.

Memory
------
The window is never stored. The normal equations need only

    S_p = sum_u u^p            p = 0..2D   (scalars)
    T_p = sum_u u^p * W(u)     p = 0..D    (tensors)

so cost is (D+1) tensors regardless of K, and the fit is a (D+1)x(D+1) solve
per tensor done once at the end.

Numerics
--------
Accumulation runs on ``W(u) - W_ref`` for a fixed reference, never on the raw
weights. Per-step motion is ~1e-5 of a weight's magnitude, which is close
enough to fp32 epsilon (~1e-7 relative) that summing raw weights would grind
away a meaningful part of the very signal being fitted. Subtracting a reference
makes the accumulated quantity the motion itself, at full relative precision.
The small normal-equation solve is done in float64.

Pure tensor arithmetic — no model, no data, no GPU. The loss evaluation that
scores these predictions lives in ``ghost/harness.py --window-fit``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Sequence

import torch

__all__ = [
    "WindowFit",
    "FitResult",
    "predict",
    "direction_cosine",
]

Tensors = Mapping[str, torch.Tensor]


# --------------------------------------------------------------------------


@dataclass
class FitResult:
    """Least-squares coefficients for every fitted tensor.

    ``coeffs[name][d]`` is the degree-``d`` coefficient tensor, expressed in
    the reference-subtracted frame (so ``coeffs[name][0]`` is the smoothed
    anchor MINUS the reference). Tensors that were not fitted — integer
    buffers, anything absent from a snapshot — are listed in ``held``.
    """

    coeffs: Dict[str, List[torch.Tensor]]
    held: List[str]
    degree: int
    window: int
    n_observations: int
    condition_number: float

    def summary(self) -> dict:
        return {
            "degree": self.degree,
            "window": self.window,
            "n_observations": self.n_observations,
            "n_fitted": len(self.coeffs),
            "n_held": len(self.held),
            "condition_number": self.condition_number,
        }


class WindowFit:
    """Streaming least-squares polynomial fit of weight motion vs step.

    Feed it one state dict per training step with ``observe``; call ``solve``
    once at the anchor. Memory is O(degree) tensors, not O(window).
    """

    def __init__(self, reference: Tensors, degree: int = 1, window: Optional[int] = None):
        """``degree`` is the MAXIMUM degree accumulated; ``solve(d)`` may then
        return any fit of degree ``d <= degree`` from the same accumulators,
        because the degree-``d`` normal equations are the leading (d+1)x(d+1)
        block of the degree-``degree`` ones. One accumulator set therefore
        serves both the linear and the quadratic arm — worth the small extra
        complexity, since each additional set costs (degree+1) x 332 MB.
        """
        if degree < 1:
            raise ValueError("degree must be >= 1 (degree 0 is just averaging)")
        self.degree = int(degree)
        self.window = int(window) if window is not None else -1
        # Reference kept in fp32 on CPU: it is subtracted from every
        # observation, so it must not itself be a source of rounding.
        self.reference: Dict[str, torch.Tensor] = {
            k: v.detach().to("cpu", torch.float32, copy=True)
            for k, v in reference.items() if v.is_floating_point()
        }
        self._held: List[str] = [k for k, v in reference.items()
                                 if not v.is_floating_point()]
        self._acc: Dict[str, List[torch.Tensor]] = {
            k: [torch.zeros_like(v) for _ in range(self.degree + 1)]
            for k, v in self.reference.items()
        }
        self._moments = [0.0] * (2 * self.degree + 1)   # S_p = sum u^p
        self._n = 0

    # ------------------------------------------------------------------

    @torch.no_grad()
    def observe(self, state: Tensors, u: float) -> None:
        """Accumulate one snapshot taken at offset ``u`` from the anchor."""
        for p in range(2 * self.degree + 1):
            self._moments[p] += float(u) ** p
        for name, acc in self._acc.items():
            t = state.get(name)
            if t is None:
                continue
            d = t.detach().to("cpu", torch.float32) - self.reference[name]
            for p in range(self.degree + 1):
                if p == 0:
                    acc[0] += d
                else:
                    acc[p].add_(d, alpha=float(u) ** p)
        self._n += 1

    # ------------------------------------------------------------------

    def release(self) -> None:
        """Drop the accumulators once every ``solve`` has been taken.

        They are (degree+1) full-model tensors per window and are dead weight
        during the scoring phase, which has several sets of coefficients and
        several checkpoints resident at once.
        """
        self._acc.clear()

    @torch.no_grad()
    def solve(self, degree: Optional[int] = None) -> FitResult:
        """Solve the normal equations once, for every tensor.

        ``degree`` defaults to the maximum accumulated; pass a smaller value
        for a lower-order fit over the same window at no extra observation
        cost.
        """
        D = self.degree if degree is None else int(degree)
        if not 1 <= D <= self.degree:
            raise ValueError(f"degree must be in [1, {self.degree}], got {D}")
        if self._n < D + 1:
            raise RuntimeError(
                f"need at least {D + 1} observations for a degree-{D} fit, "
                f"have {self._n}"
            )
        # Gram matrix G[i][j] = S_{i+j}; right-hand side is the T_p stack.
        G = torch.tensor(
            [[self._moments[i + j] for j in range(D + 1)] for i in range(D + 1)],
            dtype=torch.float64,
        )
        cond = float(torch.linalg.cond(G))
        Ginv = torch.linalg.inv(G)

        coeffs: Dict[str, List[torch.Tensor]] = {}
        for name, acc in self._acc.items():
            # c_i = sum_j Ginv[i][j] * T_j, done tensor-wise.
            out = []
            for i in range(D + 1):
                c = torch.zeros_like(acc[0], dtype=torch.float32)
                for j in range(D + 1):
                    w = float(Ginv[i, j])
                    if w != 0.0:
                        c.add_(acc[j], alpha=w)
                out.append(c)
            coeffs[name] = out
        return FitResult(
            coeffs=coeffs,
            held=list(self._held),
            degree=D,
            window=self.window if self.window > 0 else self._n,
            n_observations=self._n,
            condition_number=cond,
        )


# --------------------------------------------------------------------------


@torch.no_grad()
def predict(
    fit: FitResult,
    reference: Tensors,
    anchor: Tensors,
    n_steps: float,
    mode: str = "anchor_locked",
) -> Dict[str, torch.Tensor]:
    """Evaluate the fit ``n_steps`` past the anchor.

    ``mode="fit_eval"``      the fit's own value (smoothed anchor included).
    ``mode="anchor_locked"`` the real anchor plus the fitted DISPLACEMENT only.

    Non-float tensors, and any tensor the fit skipped, are taken from the
    anchor unchanged — the #126 rule that an unpredictable tensor is held at
    its last known value rather than guessed at.
    """
    if mode not in ("fit_eval", "anchor_locked"):
        raise ValueError(f"unknown mode {mode!r}")
    out: Dict[str, torch.Tensor] = {}
    for name, t in anchor.items():
        c = fit.coeffs.get(name)
        if c is None or not t.is_floating_point():
            out[name] = t.clone()
            continue
        # Displacement from the fitted anchor: sum_{d>=1} c_d * N^d
        disp = torch.zeros_like(c[0])
        for d in range(1, fit.degree + 1):
            disp.add_(c[d], alpha=float(n_steps) ** d)
        if mode == "anchor_locked":
            out[name] = (t.to(torch.float32) + disp).to(t.dtype)
        else:
            base = reference[name].to(torch.float32) if name in reference \
                else t.to(torch.float32)
            out[name] = (base + c[0] + disp).to(t.dtype)
    return out


@torch.no_grad()
def direction_cosine(
    predicted: Tensors, anchor: Tensors, truth: Tensors
) -> dict:
    """How well does the predicted DIRECTION match the true one?

    The loss numbers say whether a prediction is useful; this says whether it
    is pointed the right way, which is the part #126's oracle could not test
    (that oracle was confined to rescaling a fixed direction). Reported as the
    median over tensors of

        cos( W_pred - W_anchor , W_true - W_anchor )

    together with the median magnitude ratio, so an over-long step in a good
    direction is distinguishable from a short step in a bad one.
    """
    cosines: List[float] = []
    ratios: List[float] = []
    for name, a in anchor.items():
        if not a.is_floating_point() or name not in predicted or name not in truth:
            continue
        p = (predicted[name].to(torch.float64) - a.to(torch.float64)).flatten()
        q = (truth[name].to(torch.float64) - a.to(torch.float64)).flatten()
        pn, qn = float(p.norm()), float(q.norm())
        if pn <= 1e-30 or qn <= 1e-30:
            continue
        cosines.append(float(torch.dot(p, q)) / (pn * qn))
        ratios.append(pn / qn)
    if not cosines:
        return {"n_tensors": 0}
    cosines.sort()
    ratios.sort()
    return {
        "n_tensors": len(cosines),
        "median_cos": cosines[len(cosines) // 2],
        "min_cos": cosines[0],
        "max_cos": cosines[-1],
        "median_norm_ratio": ratios[len(ratios) // 2],
    }


@torch.no_grad()
def oracle_rescale(
    predicted: Tensors, anchor: Tensors, truth: Tensors
) -> tuple:
    """Best per-tensor rescaling of the fitted step, chosen using the truth.

    Same construction as #126's oracle but applied to the WINDOW-FITTED
    direction instead of the last delta: alpha* = <dW_true, s> / |s|^2 with
    s = W_pred - W_anchor. It is the upper bound on any method that keeps this
    direction and only tunes how far to go along it — so if it still loses to
    the frozen anchor, tuning the step size cannot rescue the window fit
    either, and only a different direction could.
    """
    out: Dict[str, torch.Tensor] = {}
    alphas: List[float] = []
    for name, a in anchor.items():
        if not a.is_floating_point() or name not in predicted or name not in truth:
            out[name] = a.clone()
            continue
        s = (predicted[name].to(torch.float64) - a.to(torch.float64))
        target = (truth[name].to(torch.float64) - a.to(torch.float64))
        denom = float(torch.dot(s.flatten(), s.flatten()))
        if denom <= 1e-30:
            out[name] = a.clone()
            continue
        alpha = float(torch.dot(target.flatten(), s.flatten())) / denom
        alphas.append(alpha)
        out[name] = (a.to(torch.float64) + alpha * s).to(a.dtype)
    alphas.sort()
    stats = {
        "n_tensors_fitted": len(alphas),
        "alpha_median": alphas[len(alphas) // 2] if alphas else 0.0,
        "alpha_min": alphas[0] if alphas else 0.0,
        "alpha_max": alphas[-1] if alphas else 0.0,
    }
    return out, stats
