"""Tests for the dynamic phase-aware LR scheduler (Issue #133).

Written by: Rambarun Komaljeet

Every phase is driven by a synthetic loss curve with the property the phase is
supposed to detect, and -- just as importantly -- the phases that should NOT
fire on that curve are asserted absent. A detector that fires on everything
would pass a one-sided test.
"""

from __future__ import annotations

import json
import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghost.dynamic_lr import DynamicLRConfig, DynamicLRScheduler  # noqa: E402


def make(tmp_path=None, **kw):
    cfg = DynamicLRConfig(**kw)
    log = str(tmp_path / "lr_log.json") if tmp_path is not None else ""
    return DynamicLRScheduler(cfg, log_path=log, write_log=bool(log))


def feed(sched, losses, start=0):
    out = []
    for i, l in enumerate(losses, start=start):
        out.append(sched.observe(i, l))
    return out


def phases(sched):
    return [a.phase for a in sched.adjustments]


# --------------------------------------------------------------------------
# window / cadence
# --------------------------------------------------------------------------


def test_no_adjustment_until_window_is_full():
    s = make()
    feed(s, [3.0] * 99)
    assert s.adjustments == []
    assert s.multiplier == 1.0
    feed(s, [3.0], start=99)
    assert len(s.adjustments) == 1


def test_checks_run_on_the_50_step_cadence():
    s = make()
    feed(s, [3.0] * 300)
    steps = [a.step for a in s.adjustments]
    # First at the step the window fills, then every 50 thereafter.
    assert steps[0] == 99
    assert all(b - a == 50 for a, b in zip(steps, steps[1:]))


# --------------------------------------------------------------------------
# phases
# --------------------------------------------------------------------------


def test_plateau_raises_the_multiplier():
    s = make()
    feed(s, [3.0] * 200)
    assert "plateau" in phases(s)
    assert "spike" not in phases(s)
    assert "transition" not in phases(s)
    assert s.multiplier > 1.0


def test_plateau_does_not_fire_on_a_descending_curve():
    s = make()
    feed(s, [3.0 - 0.01 * i for i in range(200)])
    assert "plateau" not in phases(s)


def test_spike_cuts_immediately_not_on_the_cadence():
    s = make()
    feed(s, [3.0] * 100)          # fills the window, fires a plateau at step 99
    before = s.multiplier
    n = len(s.adjustments)
    # A 10 pct rise over the next 10 steps -- well inside the 50-step cadence.
    feed(s, [3.0 * (1 + 0.01 * i) for i in range(1, 11)], start=100)
    assert len(s.adjustments) > n
    last = s.adjustments[-1]
    assert last.phase == "spike"
    assert last.step < 149                      # did not wait for the next check
    assert s.multiplier == pytest.approx(before * 0.5)


def test_small_wobble_does_not_trigger_a_spike():
    rng = random.Random(0)
    s = make()
    # 1 pct noise, no trend: below the 5 pct spike bar.
    feed(s, [3.0 * (1 + 0.01 * rng.uniform(-1, 1)) for _ in range(400)])
    assert "spike" not in phases(s)


def _transition_curve(seed, tail=0):
    """Settled slow descent, then a sudden 20x faster one, then optional tail.

    The steep segment is 50 steps -- the check cadence -- so it fully occupies
    the recent half-window at one check. A shorter burst is genuinely invisible
    to a detector that only looks every 50 steps, which is a property of the
    spec's cadence, not of this implementation.
    """
    rng = random.Random(seed)
    losses, l = [], 5.0
    for _ in range(150):
        l += -0.001 + 0.002 * rng.uniform(-1, 1)
        losses.append(l)
    for _ in range(50):
        l += -0.02
        losses.append(l)
    for _ in range(tail):
        l += -0.001 + 0.002 * rng.uniform(-1, 1)
        losses.append(l)
    return losses


def test_transition_fires_on_sudden_acceleration():
    s = make()
    feed(s, _transition_curve(1))
    assert "transition" in phases(s)


def test_no_transition_on_a_steady_descent():
    s = make()
    feed(s, [5.0 - 0.002 * i for i in range(400)])
    assert "transition" not in phases(s)


def test_post_transition_reduces_the_multiplier():
    s = make()
    feed(s, _transition_curve(2, tail=300))     # settles back down afterwards
    ph = phases(s)
    assert "transition" in ph
    assert "post_transition" in ph
    assert ph.index("transition") < ph.index("post_transition")


def test_spike_cooldown_blocks_an_immediate_re_raise():
    s = make()
    feed(s, [3.0] * 100)
    feed(s, [3.0 * (1 + 0.01 * i) for i in range(1, 11)], start=100)
    assert s.adjustments[-1].phase == "spike"
    after_cut = s.multiplier
    cut_step = s.adjustments[-1].step
    # A flat stretch right after the cut reads as a plateau. Nothing may raise
    # the multiplier until the cooldown has expired -- otherwise the cut is
    # undone by the very stretch it was meant to produce.
    mults = feed(s, [3.3] * 60, start=110)
    for i, m in enumerate(mults, start=110):
        if i < cut_step + s.cfg.spike_cooldown_steps:
            assert m == pytest.approx(after_cut), f"multiplier moved at step {i}"


def test_a_cooldown_longer_than_the_cadence_suppresses_a_whole_check():
    """The guard is only reachable when the cooldown outlasts the cadence.

    At the default (both 50) the next check lands exactly as the cooldown
    expires, so no check is ever suppressed -- which is the intent, not an
    accident. Stretch the cooldown and the suppression becomes observable.
    """
    s = make(spike_cooldown_steps=120)
    feed(s, [3.0] * 100)
    feed(s, [3.0 * (1 + 0.01 * i) for i in range(1, 11)], start=100)
    assert s.adjustments[-1].phase == "spike"
    after_cut = s.multiplier
    feed(s, [3.3] * 100, start=110)
    assert "cooldown" in phases(s)
    assert s.multiplier == pytest.approx(after_cut)


def test_a_sustained_climb_cuts_once_per_cooldown_not_once_per_step():
    """The failure this cooldown exists for, pinned directly.

    A monotonic climb satisfies the >5 pct-in-10-steps condition on *every*
    step. Uncontrolled that is a cut per step: 1.5 -> 0.1 inside ten steps,
    before the first cut could possibly have taken effect.
    """
    s = make()
    feed(s, [3.0] * 100)
    feed(s, [3.0 * (1.01 ** i) for i in range(1, 101)], start=100)
    spikes = [a.step for a in s.adjustments if a.phase == "spike"]
    assert len(spikes) <= 3                       # ~100 steps / 50-step cooldown
    assert all(b - a >= 50 for a, b in zip(spikes, spikes[1:]))


# --------------------------------------------------------------------------
# clamps
# --------------------------------------------------------------------------


def test_multiplier_never_leaves_the_allowed_range():
    s = make()
    feed(s, [3.0] * 3000)                       # relentless plateau -> pushes up
    assert s.multiplier <= 3.0
    assert max(a.multiplier_after for a in s.adjustments) <= 3.0
    s2 = make()
    feed(s2, [3.0] * 100)
    # A long climb -> repeated spike cuts, pushing the multiplier down.
    feed(s2, [3.0 * (1.01 ** i) for i in range(1, 500)], start=100)
    assert s2.multiplier >= 0.1
    assert min(a.multiplier_after for a in s2.adjustments) >= 0.1


def test_single_adjustment_never_exceeds_the_per_step_clamps():
    s = make(factor_plateau=9.0, factor_spike=0.01)   # deliberately illegal
    feed(s, [3.0] * 400)
    for a in s.adjustments:
        assert 0.5 <= a.factor <= 1.5
        if a.multiplier_before > 0:
            ratio = a.multiplier_after / a.multiplier_before
            assert ratio <= 1.5 + 1e-9 and ratio >= 0.5 - 1e-9


# --------------------------------------------------------------------------
# application to an optimizer
# --------------------------------------------------------------------------


def test_applied_scales_every_group_and_restores_exactly():
    torch = pytest.importorskip("torch")
    p1 = torch.nn.Parameter(torch.zeros(2))
    p2 = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([{"params": [p1], "lr": 0.02},
                           {"params": [p2], "lr": 0.0004}])
    s = make()
    s.multiplier = 2.5
    with s.applied(opt):
        inside = [pg["lr"] for pg in opt.param_groups]
    assert inside == pytest.approx([0.05, 0.001])
    # The Muon LR ratio between groups is preserved exactly.
    assert inside[0] / inside[1] == pytest.approx(0.02 / 0.0004)
    assert [pg["lr"] for pg in opt.param_groups] == pytest.approx([0.02, 0.0004])


def test_applied_restores_lrs_even_if_the_step_raises():
    torch = pytest.importorskip("torch")
    p = torch.nn.Parameter(torch.zeros(2))
    opt = torch.optim.SGD([p], lr=0.01)
    s = make()
    s.multiplier = 0.3
    with pytest.raises(RuntimeError):
        with s.applied(opt):
            raise RuntimeError("boom")
    assert opt.param_groups[0]["lr"] == pytest.approx(0.01)


def test_cosine_schedule_is_not_compounded_by_the_multiplier():
    """The reason ``applied`` is a context manager and not a setter.

    ``CosineAnnealingLR.get_lr()`` derives the next LR from the current one, so
    a multiplier left written into ``param_group['lr']`` across a
    ``scheduler.step()`` would be folded into the cosine permanently.
    """
    torch = pytest.importorskip("torch")

    def run(with_dynamic):
        torch.manual_seed(0)
        p = torch.nn.Parameter(torch.zeros(2))
        opt = torch.optim.SGD([p], lr=0.01)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=50)
        s = make()
        s.multiplier = 2.0
        out = []
        for _ in range(50):
            if with_dynamic:
                with s.applied(opt):
                    out.append(opt.param_groups[0]["lr"])
            else:
                out.append(opt.param_groups[0]["lr"])
            cos.step()
        return out

    plain, dynamic = run(False), run(True)
    assert dynamic == pytest.approx([2.0 * x for x in plain])


# --------------------------------------------------------------------------
# logging
# --------------------------------------------------------------------------


def test_log_is_written_and_round_trips(tmp_path):
    s = make(tmp_path)
    feed(s, [3.0] * 300)
    s.close()
    with open(tmp_path / "lr_log.json", encoding="utf-8") as fh:
        payload = json.load(fh)
    assert payload["n_adjustments"] == len(s.adjustments)
    assert len(payload["curve"]) == 300
    assert payload["final_multiplier"] == pytest.approx(s.multiplier)
    assert set(payload["phase_counts"]) <= {
        "spike", "transition", "plateau", "dropping_fast",
        "post_transition", "normal", "cooldown"}
    assert "plateau" in payload["spec_deviations"]


def test_state_dict_round_trips():
    s = make()
    feed(s, [3.0] * 300)
    s2 = make()
    s2.load_state_dict(s.state_dict())
    assert s2.multiplier == s.multiplier
    assert list(s2.losses) == list(s.losses)


def test_nan_loss_is_ignored_not_propagated():
    s = make()
    feed(s, [3.0] * 200)
    m = s.multiplier
    assert s.observe(200, float("nan")) == m
    assert len(s.losses) == 100
