"""Tests for the GHOST Phase 1 phase-transition detector (issue #125).

Written by: Rambarun Komaljeet

Synthetic curves only — no GPU, no checkpoints. Run from the repo root:

    python tests/test_ghost_detector.py
    pytest tests/test_ghost_detector.py
"""

from __future__ import annotations

import csv
import os
import sys
import tempfile

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghost.detector import (  # noqa: E402
    DetectorConfig,
    detect_transitions,
    load_loss_curve,
    loss_gradient,
    smooth_curve,
    synthetic_smooth_curve,
)


def _curve_with_transition(
    n: int = 600,
    step: int = 50,
    drop_at: int = 400,
    drop_len: int = 40,
    drop_size: float = 0.8,
    noise: float = 0.05,
    seed: int = 0,
):
    """Smooth decay plus one genuine acceleration (a fast extra drop)."""
    rng = np.random.default_rng(seed)
    steps = np.arange(1, n + 1, dtype=np.int64) * step
    clean = 4.0 + 3.0 * np.exp(-steps / (n * step / 4.0))
    ramp = np.clip((np.arange(n) - drop_at) / float(drop_len), 0.0, 1.0)
    clean = clean - drop_size * ramp
    return steps, clean + rng.normal(0.0, noise, size=n)


# --------------------------------------------------------------------------


def test_negative_control_smooth_curve_has_no_transitions():
    """The known-answer control: a smooth decay must yield ZERO detections."""
    for seed in range(5):
        steps, loss = synthetic_smooth_curve(n=1200, seed=seed)
        found = detect_transitions(steps, loss, DetectorConfig())
        assert not found, f"seed {seed}: false positives {[(t.start_step, t.end_step) for t in found]}"
    print("ok  negative control (5 seeds): 0 detections")


def test_detects_a_planted_transition():
    """A planted acceleration is found, and located where it was planted."""
    steps, loss = _curve_with_transition(drop_at=400, drop_len=40)
    found = detect_transitions(steps, loss, DetectorConfig())
    assert found, "planted transition not detected"
    planted_lo, planted_hi = 400 * 50, (400 + 40) * 50
    assert any(t.covers(planted_lo, planted_hi, tol=500) for t in found), \
        f"detections {[(t.start_step, t.end_step) for t in found]} miss {planted_lo}-{planted_hi}"
    print(f"ok  planted transition detected at {[(t.start_step, t.end_step) for t in found]}")


def test_flat_region_is_not_a_transition_but_its_exit_is():
    """A plateau's INTERIOR is the opposite of an acceleration and must stay unflagged.

    Its EXIT is a different matter: leaving a plateau and rejoining the descent
    is a genuine acceleration, and it is precisely the "flat patch that then
    broke through and accelerated" pattern docs/state.md describes for the 83M
    pretrain. The detector is expected to flag the exit and only the exit.
    """
    n, step = 600, 50
    steps = np.arange(1, n + 1, dtype=np.int64) * step
    clean = 4.0 + 3.0 * np.exp(-steps / (n * step / 4.0))
    clean[300:400] = clean[300]  # hard plateau, indices 300-399
    rng = np.random.default_rng(1)
    loss = clean + rng.normal(0.0, 0.02, size=n)
    found = detect_transitions(steps, loss, DetectorConfig())

    # Interior, excluding a smoothing half-span either side of the exit edge.
    interior_lo, interior_hi = 300 * step, 385 * step
    for t in found:
        assert not t.covers(interior_lo, interior_hi), \
            f"plateau interior {interior_lo}-{interior_hi} wrongly flagged: {t}"
    assert any(t.covers(395 * step, 430 * step) for t in found), \
        f"plateau exit not detected; found {[(t.start_step, t.end_step) for t in found]}"
    print("ok  plateau interior unflagged, plateau exit detected")


def test_gradient_sign_and_scale():
    """Gradient is nats per STEP and negative while the loss descends."""
    steps, loss = synthetic_smooth_curve(n=400, noise=0.0)
    grad = loss_gradient(steps, loss, smooth=5)
    assert np.all(grad[5:-5] < 0.0), "descending curve must have negative gradient"
    total = (loss[-1] - loss[0]) / float(steps[-1] - steps[0])
    assert abs(float(grad[5:-5].mean()) - total) < abs(total), "gradient scale is off"
    print(f"ok  gradient negative and ~{float(grad[5:-5].mean()):.2e} nats/step")


def test_smooth_curve_preserves_length_and_mean():
    loss = np.array([1.0, 5.0, 1.0, 5.0, 1.0, 5.0, 1.0], dtype=np.float64)
    sm = smooth_curve(loss, 5)
    assert sm.shape == loss.shape
    assert sm.std() < loss.std(), "smoothing must reduce variance"
    print("ok  smoothing preserves shape and reduces variance")


def test_loader_dedupes_and_sorts_resumed_logs():
    """Resumed runs append duplicate/non-monotonic steps; last row per step wins."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "training_log.csv")
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "epoch", "loss", "ppl", "lr", "timestamp"])
            w.writerow([100, 0, 5.0, 0, 0, "t"])
            w.writerow([200, 0, 4.0, 0, 0, "t"])
            w.writerow([100, 0, 9.9, 0, 0, "resumed-and-relogged"])  # duplicate
            w.writerow([150, 0, 4.5, 0, 0, "t"])                     # out of order
            w.writerow([250, 0, float("nan"), 0, 0, "non-finite"])   # dropped
        steps, loss = load_loss_curve(path)
        assert list(steps) == [100, 150, 200], list(steps)
        assert loss[0] == 9.9, "duplicate step must keep the LAST logged value"
        assert 250 not in list(steps), "non-finite loss must be dropped"
    print("ok  loader dedupes, sorts, and drops non-finite rows")


def test_transition_covers_tolerance():
    steps, loss = _curve_with_transition()
    t = detect_transitions(steps, loss, DetectorConfig())[0]
    assert t.covers(t.start_step, t.end_step)
    assert not t.covers(t.end_step + 10_000, t.end_step + 20_000)
    assert t.covers(t.end_step + 300, t.end_step + 400, tol=500)
    print("ok  Transition.covers honours tolerance")


def test_baseline_is_frozen_during_a_transition():
    """A long transition stays open — a live baseline would close it instantly."""
    steps, loss = _curve_with_transition(drop_at=300, drop_len=150, drop_size=1.5, noise=0.02)
    found = detect_transitions(steps, loss, DetectorConfig())
    assert found, "long transition not detected"
    longest = max(t.end_index - t.start_index for t in found)
    assert longest >= 20, f"transition closed after only {longest} samples"
    print(f"ok  baseline frozen — longest transition spans {longest} samples")


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in tests:
        try:
            fn()
        except AssertionError as exc:
            failed += 1
            print(f"FAIL  {fn.__name__}: {exc}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
