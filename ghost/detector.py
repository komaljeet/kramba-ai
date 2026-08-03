"""GHOST Phase 1 — phase-transition detector (issue #125, epic #124).

Written by: Rambarun Komaljeet

Reads a training loss log, computes the loss-curve gradient over a rolling
window, and flags *phase transitions*: stretches where the loss drops
faster than ``k_start`` sigma below the trailing mean gradient, ending when the
curve returns to smooth descent.

The point of the detector is to tell GHOST when NOT to teleport: extrapolating
weights through a phase transition is exactly where a linear fit through past
deltas should be expected to fail.

Unit note — "a rolling window of 50 steps"
------------------------------------------
The spec states a 50-step window. This project's real logs
(``checkpoints/<run>/training_log.csv``) record one row every 50 optimizer
steps, so a literal 50-*step* window is a single sample and no gradient
statistic exists. ``window`` is therefore expressed in **log samples**
(default 50 samples = 2,500 steps at the standard 50-step logging cadence).
``--window-steps`` converts from steps using the log's median sample spacing.

CLI
---
    python -m ghost.detector --log checkpoints/edge_chinchilla83m/training_log.csv \
        --truth 6000:7500 \
        --out results/ghost/detected_transitions.json \
        --plot results/ghost/transition_plot.png
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass, asdict, field
from typing import List, Optional, Sequence, Tuple

import numpy as np

__all__ = [
    "DetectorConfig",
    "Transition",
    "load_loss_curve",
    "smooth_curve",
    "loss_gradient",
    "detect_transitions",
    "synthetic_smooth_curve",
    "run_detector",
]


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class DetectorConfig:
    """Free parameters of the detector.

    Defaults follow the #125 spec: a 50-sample rolling window and a 2-sigma
    start threshold. ``k_end``/``end_patience`` implement the "returns to
    smooth descent" end condition with hysteresis, so a single noisy sample
    inside a transition does not close it prematurely.
    """

    window: int = 50           # trailing baseline length, in log samples
    smooth: int = 15           # centred moving average applied before differencing
    k_start: float = 2.0       # sigma below the trailing mean gradient to open
    open_run: int = 5          # consecutive below-threshold samples required to open
    k_end: float = 1.0         # sigma band the gradient must return into to close
    end_patience: int = 3      # consecutive in-band samples required to close
    merge_gap: int = 5         # merge detections separated by fewer samples than this
    min_samples: int = 1       # discard detections shorter than this
    sigma_floor: float = 1e-12  # guards a degenerate (zero-variance) baseline


@dataclass
class Transition:
    """One detected phase transition."""

    start_step: int
    end_step: int
    start_index: int
    end_index: int
    peak_sigma: float          # how far below the baseline the steepest sample fell
    mean_gradient: float       # loss change per step inside the transition
    baseline_gradient: float   # trailing mean gradient at the moment it opened
    closed: bool = True        # False if the log ended before it returned to band

    def covers(self, lo: int, hi: int, tol: int = 0) -> bool:
        """True if this transition overlaps the window ``[lo, hi]`` within ``tol`` steps."""
        return (self.start_step - tol) <= hi and (self.end_step + tol) >= lo


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_loss_curve(
    path: str, step_col: str = "step", loss_col: str = "loss"
) -> Tuple[np.ndarray, np.ndarray]:
    """Load ``(steps, loss)`` from a training-log CSV.

    Resumed runs append to the same CSV, so a real log can contain duplicate
    and non-monotonic step numbers (``checkpoints/edge_chinchilla83m/
    training_log.csv`` re-logs steps 1550-1800 and 2000-4000 after its resume).
    Duplicates keep the LAST row for a step — the most recent training — and the
    result is sorted by step, so downstream gradients are always taken along a
    monotonically increasing axis.
    """
    by_step: dict[int, float] = {}
    with open(path, "r", newline="") as fh:
        for row in csv.DictReader(fh):
            if not row.get(step_col) or not row.get(loss_col):
                continue
            try:
                step = int(float(row[step_col]))
                loss = float(row[loss_col])
            except (TypeError, ValueError):
                continue
            if math.isfinite(loss):
                by_step[step] = loss
    if not by_step:
        raise ValueError(f"no usable ({step_col}, {loss_col}) rows in {path}")
    steps = np.array(sorted(by_step), dtype=np.int64)
    loss = np.array([by_step[int(s)] for s in steps], dtype=np.float64)
    return steps, loss


# --------------------------------------------------------------------------
# Gradient
# --------------------------------------------------------------------------


def smooth_curve(loss: np.ndarray, span: int) -> np.ndarray:
    """Centred moving average of width ``span`` (odd), edges handled by shrinking."""
    if span <= 1:
        return loss.astype(np.float64, copy=True)
    if span % 2 == 0:
        span += 1
    half = span // 2
    out = np.empty_like(loss, dtype=np.float64)
    for i in range(loss.size):
        lo = max(0, i - half)
        hi = min(loss.size, i + half + 1)
        out[i] = float(loss[lo:hi].mean())
    return out


def loss_gradient(steps: np.ndarray, loss: np.ndarray, smooth: int) -> np.ndarray:
    """Loss change per optimizer step, from the smoothed curve.

    Smoothing before differencing is what makes the statistic usable at all: raw
    per-50-step training loss on this project's runs has a sample-to-sample
    standard deviation (~0.15 nats) far larger than the descent it rides on
    (~0.1 nats per 1,000 steps), so an unsmoothed difference is essentially pure
    noise. Units are nats/step (negative while learning).
    """
    sm = smooth_curve(loss, smooth)
    return np.gradient(sm, steps.astype(np.float64))


# --------------------------------------------------------------------------
# Detection
# --------------------------------------------------------------------------


def detect_transitions(
    steps: np.ndarray, loss: np.ndarray, cfg: Optional[DetectorConfig] = None
) -> List[Transition]:
    """Flag phase transitions in a loss curve.

    A transition OPENS when ``open_run`` CONSECUTIVE samples have a gradient at
    least ``k_start`` sigma below the mean of the preceding ``window`` gradients
    (i.e. sustained anomalously fast descent), and CLOSES once the gradient has
    sat back inside ``baseline - k_end * sigma`` for ``end_patience``
    consecutive samples ("returned to smooth descent").

    ``open_run`` is not cosmetic. A bare k-sigma test fires on pure noise at a
    rate fixed by the threshold alone: on a smooth 1,200-sample curve, k=2 gives
    ~1200 * Phi(-2) ~= 27 detections *by construction*. Requiring persistence
    suppresses that geometrically while leaving a genuine multi-sample
    transition untouched — measured on the synthetic control in
    ``tests/test_ghost_detector.py``: 28 false positives at ``open_run=1`` vs 0
    at the default. The defaults were chosen against that synthetic control
    ONLY, never against the real 6K-7.5K ground-truth window the gate scores.

    The baseline is FROZEN at the moment of opening. Letting it keep updating
    inside the transition would fold the anomaly into its own reference and the
    detector would close itself immediately.
    """
    cfg = cfg or DetectorConfig()
    n = steps.size
    if n < cfg.window + 2:
        return []

    grad = loss_gradient(steps, loss, cfg.smooth)

    transitions: List[Transition] = []
    i = cfg.window
    while i < n:
        hist = grad[i - cfg.window : i]
        mu = float(hist.mean())
        sd = max(float(hist.std(ddof=1)) if hist.size > 1 else 0.0, cfg.sigma_floor)
        thresh = mu - cfg.k_start * sd
        run = grad[i : i + cfg.open_run]
        if run.size < cfg.open_run or not bool(np.all(run <= thresh)):
            i += 1
            continue

        # --- open a transition; baseline frozen at (mu, sd) ---
        start = i
        in_band = 0
        j = i
        closed = False
        while j < n:
            if grad[j] > mu - cfg.k_end * sd:
                in_band += 1
                if in_band >= cfg.end_patience:
                    closed = True
                    break
            else:
                in_band = 0
            j += 1
        end = min(j - in_band if closed else n - 1, n - 1)
        end = max(end, start)

        seg = grad[start : end + 1]
        transitions.append(
            Transition(
                start_step=int(steps[start]),
                end_step=int(steps[end]),
                start_index=start,
                end_index=end,
                peak_sigma=float((mu - seg.min()) / sd),
                mean_gradient=float(seg.mean()),
                baseline_gradient=mu,
                closed=closed,
            )
        )
        i = end + 1

    return _postprocess(transitions, cfg)


def _postprocess(transitions: List[Transition], cfg: DetectorConfig) -> List[Transition]:
    """Merge near-adjacent detections, then drop ones shorter than ``min_samples``."""
    if not transitions:
        return []
    merged = [transitions[0]]
    for t in transitions[1:]:
        prev = merged[-1]
        if t.start_index - prev.end_index <= cfg.merge_gap:
            merged[-1] = Transition(
                start_step=prev.start_step,
                end_step=t.end_step,
                start_index=prev.start_index,
                end_index=t.end_index,
                peak_sigma=max(prev.peak_sigma, t.peak_sigma),
                mean_gradient=(prev.mean_gradient + t.mean_gradient) / 2.0,
                baseline_gradient=prev.baseline_gradient,
                closed=t.closed,
            )
        else:
            merged.append(t)
    return [t for t in merged if (t.end_index - t.start_index + 1) >= cfg.min_samples]


# --------------------------------------------------------------------------
# Negative control
# --------------------------------------------------------------------------


def synthetic_smooth_curve(
    n: int = 2000,
    step: int = 50,
    start: float = 8.4,
    floor: float = 3.9,
    tau_steps: float = 20000.0,
    noise: float = 0.15,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray]:
    """A smooth exponential-decay loss curve with matched noise — no transitions.

    Defaults mirror the real 83M pretrain (loss 8.4 -> 3.9 over ~102K steps,
    per-sample sigma ~0.15). The detector MUST report zero transitions here;
    this is the configuration-with-a-known-answer that #123's ``E=1 == dense``
    property established as a requirement for any new measurement harness in
    this repo.
    """
    rng = np.random.default_rng(seed)
    steps = np.arange(1, n + 1, dtype=np.int64) * step
    clean = floor + (start - floor) * np.exp(-steps / tau_steps)
    return steps, clean + rng.normal(0.0, noise, size=n)


# --------------------------------------------------------------------------
# Report / CLI
# --------------------------------------------------------------------------


@dataclass
class DetectorReport:
    log: str
    n_samples: int
    step_range: List[int]
    config: dict
    n_detections: int
    transitions: List[dict]
    ground_truth: Optional[dict] = None
    negative_control: Optional[dict] = None
    sweep: Optional[List[dict]] = field(default=None)


def run_detector(
    steps: np.ndarray,
    loss: np.ndarray,
    cfg: DetectorConfig,
    truth: Optional[Tuple[int, int]] = None,
    truth_tol: int = 500,
) -> Tuple[List[Transition], Optional[dict]]:
    """Detect, and (if a ground-truth window is given) score the gate."""
    transitions = detect_transitions(steps, loss, cfg)
    gt = None
    if truth is not None:
        lo, hi = truth
        hits = [t for t in transitions if t.covers(lo, hi, truth_tol)]
        gt = {
            "window": [lo, hi],
            "tolerance_steps": truth_tol,
            "detected": bool(hits),
            "matching_transitions": [asdict(t) for t in hits],
            "n_other_detections": len(transitions) - len(hits),
        }
    return transitions, gt


def _sweep(steps: np.ndarray, loss: np.ndarray, base: DetectorConfig,
           truth: Optional[Tuple[int, int]], truth_tol: int) -> List[dict]:
    """Sensitivity of the verdict to the two free parameters."""
    out: List[dict] = []
    for window in (20, 35, 50, 75, 100):
        for k in (1.5, 2.0, 2.5, 3.0):
            for open_run in (1, 3, 5, 10):
                cfg = DetectorConfig(**{**asdict(base), "window": window,
                                        "k_start": k, "open_run": open_run})
                ts, gt = run_detector(steps, loss, cfg, truth, truth_tol)
                ctrl_steps, ctrl_loss = synthetic_smooth_curve(n=steps.size)
                ctrl = detect_transitions(ctrl_steps, ctrl_loss, cfg)
                out.append({
                    "window": window,
                    "k_start": k,
                    "open_run": open_run,
                    "n_detections": len(ts),
                    "hits_ground_truth": None if gt is None else gt["detected"],
                    "control_false_positives": len(ctrl),
                })
    return out


def _plot(steps, loss, cfg, transitions, truth, path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(13, 5.5))
    ax.plot(steps, loss, lw=0.6, alpha=0.35, color="#9aa0a6", label="train loss (raw)")
    ax.plot(steps, smooth_curve(loss, cfg.smooth * 5), lw=1.6, color="#1a73e8",
            label=f"smoothed ({cfg.smooth * 5} samples)")
    for i, t in enumerate(transitions):
        ax.axvspan(t.start_step, t.end_step, color="#ea4335", alpha=0.25,
                   label="detected transition" if i == 0 else None)
    if truth is not None:
        ax.axvspan(truth[0], truth[1], facecolor="none", edgecolor="#137333",
                   lw=2.0, hatch="//", label=f"documented transition {truth[0]}-{truth[1]}")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("training loss (nats)")
    ax.set_title(f"GHOST phase-transition detector — {len(transitions)} detection(s), "
                 f"window={cfg.window}, k_start={cfg.k_start}")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="GHOST Phase 1 phase-transition detector")
    p.add_argument("--log", required=True, help="training-log CSV (step,loss)")
    p.add_argument("--step-col", default="step")
    p.add_argument("--loss-col", default="loss")
    p.add_argument("--window", type=int, default=DetectorConfig.window,
                   help="trailing baseline length in LOG SAMPLES (default 50)")
    p.add_argument("--window-steps", type=int, default=None,
                   help="same, expressed in optimizer steps; converted via median sample spacing")
    p.add_argument("--smooth", type=int, default=DetectorConfig.smooth)
    p.add_argument("--k-start", type=float, default=DetectorConfig.k_start)
    p.add_argument("--k-end", type=float, default=DetectorConfig.k_end)
    p.add_argument("--end-patience", type=int, default=DetectorConfig.end_patience)
    p.add_argument("--truth", default=None, metavar="LO:HI",
                   help="documented transition window to gate on, e.g. 6000:7500")
    p.add_argument("--truth-tol", type=int, default=500)
    p.add_argument("--out", default="results/ghost/detected_transitions.json")
    p.add_argument("--plot", default="results/ghost/transition_plot.png")
    p.add_argument("--sweep", action="store_true", help="also run the window/k sensitivity sweep")
    args = p.parse_args(argv)

    steps, loss = load_loss_curve(args.log, args.step_col, args.loss_col)

    window = args.window
    if args.window_steps is not None:
        spacing = float(np.median(np.diff(steps))) if steps.size > 1 else 1.0
        window = max(2, int(round(args.window_steps / max(spacing, 1.0))))
        print(f"[ghost] --window-steps {args.window_steps} -> {window} samples "
              f"(median spacing {spacing:.0f} steps)")

    cfg = DetectorConfig(window=window, smooth=args.smooth, k_start=args.k_start,
                         k_end=args.k_end, end_patience=args.end_patience)
    truth = None
    if args.truth:
        lo, hi = args.truth.split(":")
        truth = (int(lo), int(hi))

    transitions, gt = run_detector(steps, loss, cfg, truth, args.truth_tol)

    # Negative control: same config, synthetic smooth decay with matched noise.
    nsteps, nloss = synthetic_smooth_curve(n=steps.size, step=int(np.median(np.diff(steps)))
                                           if steps.size > 1 else 50)
    control = detect_transitions(nsteps, nloss, cfg)

    report = DetectorReport(
        log=args.log.replace("\\", "/"),
        n_samples=int(steps.size),
        step_range=[int(steps[0]), int(steps[-1])],
        config=asdict(cfg),
        n_detections=len(transitions),
        transitions=[asdict(t) for t in transitions],
        ground_truth=gt,
        negative_control={
            "description": "smooth exponential decay, matched noise — must detect 0",
            "n_detections": len(control),
            "passed": len(control) == 0,
            "transitions": [asdict(t) for t in control],
        },
        sweep=_sweep(steps, loss, cfg, truth, args.truth_tol) if args.sweep else None,
    )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(asdict(report), fh, indent=2)
    _plot(steps, loss, cfg, transitions, truth, args.plot)

    print(f"[ghost] {steps.size} samples, steps {steps[0]}-{steps[-1]}")
    print(f"[ghost] detections: {len(transitions)}")
    for t in transitions:
        print(f"         steps {t.start_step}-{t.end_step}  "
              f"peak {t.peak_sigma:.2f} sigma  grad {t.mean_gradient:+.3e}/step"
              f"{'' if t.closed else '  (open at end of log)'}")
    print(f"[ghost] negative control: {len(control)} detections "
          f"({'PASS' if not control else 'FAIL'})")
    if gt is not None:
        verdict = "PASS" if gt["detected"] else "FAIL"
        print(f"[ghost] ground-truth gate {truth[0]}-{truth[1]} "
              f"(+/-{args.truth_tol}): {verdict}")
    print(f"[ghost] wrote {args.out} and {args.plot}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
