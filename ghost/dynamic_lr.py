"""Dynamic phase-aware global learning-rate scheduler (Issue #133).

Written by: Rambarun Komaljeet

Muon already supplies *per-parameter* adaptivity. This supplies the macroscopic
half: one global multiplier, applied on top of whatever base schedule (warmup +
cosine) the training script already produces, chosen from the shape of the live
loss curve.

    effective_lr(group) = base_schedule_lr(group) * multiplier

Nothing inside Muon changes. The multiplier is applied to every param group by
the same factor, so a multi-group optimizer (Muon runs high-LR matrices and a
low-LR AdamW auxiliary group) keeps its LR *ratio* exactly.

Phases and their responses
--------------------------
======================  ==========================================  =========
phase                   trigger                                     response
======================  ==========================================  =========
``spike``               loss up > 5 pct across 10 steps              x0.50
``transition``          smoothed slope > 2 sigma below window mean   x1.25
``plateau``             |100-step smoothed delta| < 1e-3            x1.50
``dropping_fast``       slope steeper than the window mean           x1.05
``post_transition``     within 300 steps of a transition ending      x0.90
``normal``              none of the above                           x1.00
======================  ==========================================  =========

Priority is top-to-bottom. The multiplier is clamped to [0.1, 3.0] overall and
each single adjustment is clamped to [0.5x, 1.5x], both per the #133 spec.

Three places where a literal reading of the spec would not work, and what was
done instead. All three are recorded in the log's ``config`` block so a reader
of ``lr_log.json`` sees them without reading this file.

1. **"Plateau = loss delta < 0.001 for 100 consecutive steps" cannot fire on the
   raw curve.** Step-to-step training loss on this model moves by ~0.1-0.5
   nats; the raw delta is never below 1e-3 twice in a row, let alone 100 times.
   Read as the *smoothed* 100-step delta instead: the mean of the window's
   second half minus the mean of its first half. That is the quantity the
   condition was plainly describing, and it is measurable.

2. **"Loss increases > 5 pct in 10 steps" on single points is a noise detector.**
   Compared as means of 5 samples at each end of the 10-step gap, so a single
   unlucky batch cannot trigger an LR cut. `ghost/detector.py` reached the same
   conclusion for the same reason ("a bare k-sigma test fires on pure noise").

3. **"Drops > 2 sigma faster than rolling mean"** is tested against the standard
   error of the recent *mean* slope (sigma / sqrt(n_recent)), not against a
   single sample's sigma. Testing one noisy sample at 2 sigma fires ~2.5 pct of
   the time on a pure random walk; testing a 10-sample mean does not.

One mechanism is here that the spec does not name: a **cooldown** after a spike
cut. Without it the same window that triggered the cut is still in memory at the
next check, reads as a plateau (a spike flattens the smoothed delta), and the
scheduler would immediately raise the LR it had just lowered. During cooldown
only ``spike`` and ``normal`` can fire.

Usage
-----
    sched = DynamicLRScheduler(log_path="results/dynamic_lr/lr_log.json")
    ...
    mult = sched.observe(global_step, float(step_loss))
    with sched.applied(optimizer):     # scales every group's lr by `mult`
        optimizer.step()               # restored on exit, so a recursive
    cosine_scheduler.step()            # CosineAnnealingLR is not compounded
"""

from __future__ import annotations

import contextlib
import json
import os
import statistics
from collections import deque
from dataclasses import asdict, dataclass, field
from typing import Deque, Dict, List, Optional

__all__ = ["DynamicLRConfig", "Adjustment", "DynamicLRScheduler"]


@dataclass
class DynamicLRConfig:
    """Every threshold in the #133 spec, in one place."""

    window: int = 100              # rolling loss window (steps)
    check_every: int = 50          # phase detection cadence (steps)

    plateau_delta: float = 1e-3    # smoothed 100-step |delta| below this = plateau
    spike_frac: float = 0.05       # >5 pct rise ...
    spike_lookback: int = 10       # ... across this many steps
    spike_smooth: int = 5          # samples averaged at each end of the lookback
    transition_sigma: float = 2.0  # sigma below the PRECEDING half-window's slope
    post_transition_steps: int = 300  # how long "post-transition" lasts

    # Responses (all inside the [0.5, 1.5] per-adjustment clamp by construction).
    factor_spike: float = 0.50
    factor_transition: float = 1.25
    factor_plateau: float = 1.50
    factor_dropping_fast: float = 1.05
    factor_post_transition: float = 0.90

    min_mult: float = 0.1
    max_mult: float = 3.0
    max_increase: float = 1.5      # never more than 1.5x in a single adjustment
    max_decrease: float = 0.5      # never less than 0.5x in a single adjustment

    # Steps after a cut during which neither another cut nor any raise may fire.
    # Must exceed spike_lookback + spike_smooth so the offending rise has rolled
    # out of the spike comparison before another cut can be considered.
    spike_cooldown_steps: int = 50


@dataclass
class Adjustment:
    """One logged decision. Emitted on every check, adjusted or not."""

    step: int
    phase: str
    multiplier_before: float
    multiplier_after: float
    factor: float
    loss: float
    window_mean: float
    window_std: float
    smoothed_delta: float          # mean(2nd half) - mean(1st half) of the window
    recent_slope: float            # loss change per step over the last N steps
    baseline_slope: float          # mean per-step slope across the whole window
    slope_sigma: float             # sigma of the per-step slopes in the window
    slope_z: float                 # (recent - baseline) / stderr(recent)
    spike_ratio: float             # smoothed loss now / smoothed loss 10 steps ago
    clamped: bool


class DynamicLRScheduler:
    """Watches the loss curve; emits a global LR multiplier.

    Owns no optimizer state and mutates no optimizer permanently -- ``applied()``
    scales the live LRs for the duration of one ``optimizer.step()`` and puts
    them back. That matters here specifically: this repo's base schedule is
    ``CosineAnnealingLR``, whose ``get_lr()`` is recursive (it derives the next
    LR from the current one, see ``build_cosine_scheduler``), so leaving a
    multiplier written into ``param_group["lr"]`` across a ``scheduler.step()``
    would compound it into the cosine itself.
    """

    def __init__(
        self,
        config: Optional[DynamicLRConfig] = None,
        log_path: str = "results/dynamic_lr/lr_log.json",
        run_name: str = "dynamic",
        write_log: bool = True,
    ) -> None:
        self.cfg = config or DynamicLRConfig()
        self.log_path = log_path
        self.run_name = run_name
        self.write_log = write_log

        self.multiplier: float = 1.0
        self.losses: Deque[float] = deque(maxlen=self.cfg.window)
        self.adjustments: List[Adjustment] = []
        self.history: List[Dict[str, float]] = []   # (step, loss, multiplier) per observe

        self._last_check: int = -1
        self._transition_end_step: Optional[int] = None
        self._in_transition: bool = False
        self._cooldown_until: int = -1
        self._phase_counts: Dict[str, int] = {}

    # -- public API --------------------------------------------------------

    def observe(self, step: int, loss: float) -> float:
        """Feed one step's training loss; get the multiplier to use now.

        Safe to call every step. Phase detection runs on the ``check_every``
        cadence, except the spike test, which runs on every call -- the spec
        says a spike is reduced "immediately", and waiting up to 50 steps to
        react to a diverging loss is not immediately.
        """
        if loss != loss:  # NaN: the one case where doing nothing is wrong
            return self.multiplier
        self.losses.append(float(loss))
        self.history.append({"step": step, "loss": float(loss),
                             "multiplier": self.multiplier})

        if len(self.losses) < self.cfg.window:
            return self.multiplier

        m = self._measure(step, loss)

        # The spike test runs every step, not on the cadence: the spec says a
        # spike is reduced "immediately", and waiting up to 50 steps to react to
        # a diverging loss is not immediately. It is rate-limited, though -- a
        # sustained climb satisfies the >5 pct condition on every step of the
        # climb, and without the cooldown that is three cuts in ten steps and
        # the multiplier on the floor before the first cut has had a chance to
        # work. One cut, then look again after a full check period.
        if step >= self._cooldown_until and m["spike_ratio"] > 1.0 + self.cfg.spike_frac:
            self._apply(step, "spike", self.cfg.factor_spike, m)
            self._cooldown_until = step + self.cfg.spike_cooldown_steps
            self._last_check = step
            return self.multiplier

        if step - self._last_check < self.cfg.check_every:
            return self.multiplier
        self._last_check = step

        phase, factor = self._classify(step, m)
        self._apply(step, phase, factor, m)
        return self.multiplier

    @contextlib.contextmanager
    def applied(self, optimizer):
        """Scale every param group's LR by the multiplier for one step."""
        saved = [pg["lr"] for pg in optimizer.param_groups]
        for pg, lr in zip(optimizer.param_groups, saved):
            pg["lr"] = lr * self.multiplier
        try:
            yield
        finally:
            for pg, lr in zip(optimizer.param_groups, saved):
                pg["lr"] = lr

    def close(self) -> None:
        self._flush()

    def state_dict(self) -> dict:
        return {
            "multiplier": self.multiplier,
            "losses": list(self.losses),
            "last_check": self._last_check,
            "transition_end_step": self._transition_end_step,
            "in_transition": self._in_transition,
            "cooldown_until": self._cooldown_until,
        }

    def load_state_dict(self, sd: dict) -> None:
        self.multiplier = float(sd["multiplier"])
        self.losses = deque(sd["losses"], maxlen=self.cfg.window)
        self._last_check = int(sd["last_check"])
        self._transition_end_step = sd["transition_end_step"]
        self._in_transition = bool(sd["in_transition"])
        self._cooldown_until = int(sd["cooldown_until"])

    # -- detection ---------------------------------------------------------

    def _measure(self, step: int, loss: float) -> Dict[str, float]:
        """Every statistic the classifier needs, computed once per call."""
        w = list(self.losses)
        n = len(w)
        half = n // 2
        first, second = w[:half], w[half:]

        # The window is split into a PRECEDING half (the baseline) and a RECENT
        # half (what is being judged). Two reasons the baseline must exclude the
        # recent half rather than span the whole window:
        #
        #   * a transition inflates the very sigma it would be tested against,
        #     so a whole-window sigma hides the event that created it;
        #   * the recent half must cover the entire gap between checks, or a
        #     transition that lands mid-interval is never in view at any check.
        #
        # This is the same "compare against the preceding window" convention
        # ghost/detector.py uses offline.
        deltas = [w[i + 1] - w[i] for i in range(n - 1)]
        hist, recent = deltas[:half - 1], deltas[half - 1:]
        baseline_slope = statistics.fmean(hist) if hist else 0.0
        slope_sigma = statistics.stdev(hist) if len(hist) > 1 else 0.0
        recent_slope = statistics.fmean(recent) if recent else 0.0
        # Standard error of the recent MEAN, not a single sample's sigma: a
        # 2-sigma test on one noisy sample fires on a random walk alone.
        stderr = (slope_sigma / (len(recent) ** 0.5)) if slope_sigma > 0 else 0.0
        slope_z = ((recent_slope - baseline_slope) / stderr) if stderr > 0 else 0.0

        s = self.cfg.spike_smooth
        lb = self.cfg.spike_lookback
        now = statistics.fmean(w[-s:])
        then = statistics.fmean(w[-(lb + s):-lb]) if n >= lb + s else now
        spike_ratio = (now / then) if then > 1e-12 else 1.0

        return {
            "window_mean": statistics.fmean(w),
            "window_std": statistics.stdev(w) if n > 1 else 0.0,
            "smoothed_delta": statistics.fmean(second) - statistics.fmean(first),
            "recent_slope": recent_slope,
            "baseline_slope": baseline_slope,
            "slope_sigma": slope_sigma,
            "slope_z": slope_z,
            "spike_ratio": spike_ratio,
        }

    def _classify(self, step: int, m: Dict[str, float]) -> tuple:
        cfg = self.cfg

        if step < self._cooldown_until:
            return "cooldown", 1.0

        # Phase transition: the recent mean slope is >2 sigma steeper (more
        # negative) than the window's own mean slope.
        if m["slope_z"] <= -cfg.transition_sigma:
            self._in_transition = True
            return "transition", cfg.factor_transition

        if self._in_transition:
            # Back inside the normal band -> the transition has ended.
            self._in_transition = False
            self._transition_end_step = step

        if abs(m["smoothed_delta"]) < cfg.plateau_delta:
            return "plateau", cfg.factor_plateau

        if (self._transition_end_step is not None
                and step - self._transition_end_step <= cfg.post_transition_steps):
            return "post_transition", cfg.factor_post_transition

        # Descending faster than the window's own average, but not 2 sigma so.
        if m["recent_slope"] < m["baseline_slope"] and m["recent_slope"] < 0:
            return "dropping_fast", cfg.factor_dropping_fast

        return "normal", 1.0

    # -- application + logging --------------------------------------------

    def _apply(self, step: int, phase: str, factor: float, m: Dict[str, float]) -> None:
        cfg = self.cfg
        before = self.multiplier
        f = min(max(factor, cfg.max_decrease), cfg.max_increase)
        raw = before * f
        after = min(max(raw, cfg.min_mult), cfg.max_mult)
        self.multiplier = after
        self._phase_counts[phase] = self._phase_counts.get(phase, 0) + 1

        self.adjustments.append(Adjustment(
            step=step, phase=phase,
            multiplier_before=before, multiplier_after=after,
            factor=f, loss=self.losses[-1],
            window_mean=m["window_mean"], window_std=m["window_std"],
            smoothed_delta=m["smoothed_delta"], recent_slope=m["recent_slope"],
            baseline_slope=m["baseline_slope"], slope_sigma=m["slope_sigma"],
            slope_z=m["slope_z"], spike_ratio=m["spike_ratio"],
            clamped=(abs(raw - after) > 1e-12 or abs(f - factor) > 1e-12),
        ))
        self._flush()

    def _flush(self) -> None:
        if not self.write_log or not self.log_path:
            return
        os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
        payload = {
            "run_name": self.run_name,
            "config": asdict(self.cfg),
            "spec_deviations": {
                "plateau": "smoothed 100-step delta (mean of 2nd half - mean of "
                           "1st half); the raw per-step delta never falls below "
                           "1e-3 on this model",
                "spike": f"means of {self.cfg.spike_smooth} samples at each end of "
                         f"the {self.cfg.spike_lookback}-step gap, not single points",
                "transition": "z-test of the recent half-window's mean slope "
                              "against the PRECEDING half-window's mean and "
                              "sigma. A whole-window sigma is inflated by the "
                              "very event being detected, and a short recent "
                              "segment misses any transition landing between "
                              "checks",
                "cooldown": f"{self.cfg.spike_cooldown_steps} steps after a spike cut "
                            "during which neither another cut nor any raise may "
                            "fire (not in the spec; without it a sustained climb "
                            "cuts on every step and hits the 0.1 floor in ~10 "
                            "steps, and a flat stretch right after a cut reads as "
                            "a plateau and instantly undoes it)",
            },
            "final_multiplier": self.multiplier,
            "phase_counts": self._phase_counts,
            "n_adjustments": len(self.adjustments),
            "adjustments": [asdict(a) for a in self.adjustments],
            "curve": self.history,
        }
        tmp = self.log_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        os.replace(tmp, self.log_path)
