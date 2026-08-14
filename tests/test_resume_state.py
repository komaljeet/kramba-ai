"""Resume must CONTINUE the optimization, not restart it (Issue #112).

The ~355M pretrain needs `--block-coordinate` to fit at all, and it will restart
many times over a multi-week run. Before this fix, every restart:

  * threw away Muon/AdamW momentum (no optimizer state was ever saved),
  * rewound the BCD round-robin to block 0 (the scheduler was rebuilt fresh), and
  * restarted the cosine LR curve at peak.

The second is the one that changes the trained model: with a fresh cycle every
restart, block 0 is retrained after every crash while the last block only advances
during uninterrupted stretches, so late blocks are systematically undertrained.

These tests assert the *behaviour* (state survives a save/load round trip and the
cycle lands where it left off), not merely that the new keys exist.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_edge_conversational import (  # noqa: E402
    RESUME_STATE_FILENAME,
    _load_resume_state,
    _save_resume_state,
    build_cosine_scheduler,
)
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402


def _tiny_model() -> nn.Module:
    """Small stack of named blocks — enough for the BCD scheduler to partition."""
    m = nn.Sequential(
        nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8), nn.Linear(8, 8),
    )
    return m


def test_optimizer_state_round_trips(tmp_path):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # Take real steps so the optimizer accumulates non-trivial moments.
    for _ in range(5):
        loss = model(torch.randn(4, 8)).square().mean()
        opt.zero_grad(); loss.backward(); opt.step()

    _save_resume_state(str(tmp_path), global_step=5, optimizer=opt)
    assert os.path.exists(tmp_path / RESUME_STATE_FILENAME)

    # A fresh optimizer has empty state; loading must repopulate it exactly.
    fresh = torch.optim.AdamW(model.parameters(), lr=1e-3)
    assert len(fresh.state_dict()["state"]) == 0

    state = _load_resume_state(str(tmp_path), expected_step=5)
    assert state is not None
    fresh.load_state_dict(state["optimizer_state_dict"])

    got = fresh.state_dict()["state"]
    want = opt.state_dict()["state"]
    assert set(got) == set(want) and len(got) > 0
    for k in want:
        assert torch.allclose(got[k]["exp_avg"], want[k]["exp_avg"])
        assert torch.allclose(got[k]["exp_avg_sq"], want[k]["exp_avg_sq"])
        assert got[k]["step"] == want[k]["step"]


def test_bcd_cycle_resumes_mid_block_instead_of_rewinding(tmp_path):
    """The regression that actually corrupts a long run."""
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = BlockCoordinateScheduler(model, opt, steps_per_block=3)

    # Advance well past the first block so a rewind is detectable.
    for _ in range(7):
        sched.step()
    assert sched._cycle_pos > 0, "test needs to have left block 0 to be meaningful"
    expected_pos, expected_in_block = sched._cycle_pos, sched._step_in_block
    expected_active = sched.active_block_name

    _save_resume_state(str(tmp_path), global_step=7, optimizer=opt, bcd_scheduler=sched)

    # A fresh scheduler is what the old code always used — it starts at block 0.
    model2 = _tiny_model()
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = BlockCoordinateScheduler(model2, opt2, steps_per_block=3)
    assert sched2._cycle_pos == 0, "fresh scheduler should start at block 0 (the old bug)"

    state = _load_resume_state(str(tmp_path), expected_step=7)
    sched2.load_state_dict(state["bcd_state_dict"])

    assert sched2._cycle_pos == expected_pos
    assert sched2._step_in_block == expected_in_block
    assert sched2.active_block_name == expected_active

    # And exactly one parameter block is trainable after restore — restoring the
    # bookkeeping without re-applying the freeze would silently train everything.
    trainable = {n.split(".")[0] for n, p in model2.named_parameters() if p.requires_grad}
    assert len(trainable) == 1


def test_stale_resume_state_is_refused_not_silently_used(tmp_path):
    """Mismatched weights + momentum is worse than a fresh optimizer."""
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _save_resume_state(str(tmp_path), global_step=1000, optimizer=opt)

    assert _load_resume_state(str(tmp_path), expected_step=500) is None
    assert _load_resume_state(str(tmp_path), expected_step=1000) is not None


def test_missing_resume_state_is_not_an_error(tmp_path):
    """First run of a fresh checkpoint dir, and every pre-#112 checkpoint."""
    assert _load_resume_state(str(tmp_path), expected_step=0) is None


def test_corrupt_resume_state_degrades_instead_of_crashing(tmp_path):
    """A crash mid-write must not make the run unresumable."""
    (tmp_path / RESUME_STATE_FILENAME).write_bytes(b"not a torch file")
    assert _load_resume_state(str(tmp_path), expected_step=10) is None


def test_save_is_atomic_leaving_no_temp_file(tmp_path):
    model = _tiny_model()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    _save_resume_state(str(tmp_path), global_step=1, optimizer=opt)
    leftovers = [p.name for p in tmp_path.iterdir() if p.name.endswith(".tmp")]
    assert leftovers == []


def test_cosine_schedule_continues_rather_than_restarting():
    """Resuming must not jump the LR back to peak (the sawtooth bug).

    Calls the real `build_cosine_scheduler`, not a reimplementation of it — an
    earlier version of this test reimplemented the intended fix and therefore
    passed while the shipped code was wrong.
    """
    model = _tiny_model()
    total, resume_at = 1000, 400

    # Reference: one uninterrupted run, LR after `resume_at` steps.
    opt_ref = torch.optim.SGD(model.parameters(), lr=0.1)
    sch_ref = build_cosine_scheduler(opt_ref, total, resume_step=0)
    for _ in range(resume_at):
        sch_ref.step()
    lr_uninterrupted = opt_ref.param_groups[0]["lr"]

    # Resumed: a fresh scheduler fast-forwarded to the same point.
    opt_new = torch.optim.SGD(model.parameters(), lr=0.1)
    build_cosine_scheduler(opt_new, total, resume_step=resume_at)
    lr_resumed = opt_new.param_groups[0]["lr"]

    assert lr_resumed == pytest.approx(lr_uninterrupted, rel=1e-6)
    # And it is genuinely mid-decay, not back at the peak the old code restarted from.
    assert lr_resumed < 0.1 * 0.95


def test_cosine_resume_beyond_total_steps_does_not_hang_or_crash():
    """Guard the degenerate case: a checkpoint past --max-steps."""
    model = _tiny_model()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    build_cosine_scheduler(opt, total_steps=100, resume_step=5000)
    assert opt.param_groups[0]["lr"] >= 0.0


# ---------------------------------------------------------------------------
# LR continuity across CHAINED resumes (#165).
#
# The 48M code run lost 26% of its intended learning rate this way. On resume,
# optimizer.load_state_dict() restores the already-DECAYED lr into param_groups,
# and the cosine scheduler was then built on top of that value -- so the whole
# curve was applied again, multiplicatively, once per resume:
#
#     step 5050  after resume 1:  0.016614 (intended 0.018372)   -9.7%
#     step 12050 after resume 2:  0.005167 (intended 0.009781)  -47%
#
# test_cosine_schedule_continues_rather_than_restarting above CANNOT catch this:
# it builds the resumed scheduler on a FRESH optimizer sitting at peak LR, so it
# never creates the condition. It also resumes only once, and the error only
# becomes obvious when it compounds.
#
# So these tests go through the real save/load path, chain TWO resumes, and
# compare the whole LR SEQUENCE against an uninterrupted reference.
# ---------------------------------------------------------------------------

TOTAL, PEAK, WARMUP = 25_000, 0.02, 500


def _fresh_opt():
    return torch.optim.SGD(_tiny_model().parameters(), lr=PEAK)


def _reference_lrs(total=TOTAL, warmup=WARMUP):
    """LR at every step of a run that is never interrupted.

    Warmup is modelled because the production loop overrides the LR for the
    first ``warmup`` steps and does NOT advance the cosine during them. Leaving
    it out shifts the whole curve by ~500 steps and the reproduction no longer
    matches the values the incident actually logged.
    """
    opt = _fresh_opt()
    sched = build_cosine_scheduler(opt, total, 0)
    out = []
    for step in range(total):
        if step < warmup:
            lr = PEAK * step / warmup
            for pg in opt.param_groups:
                pg["lr"] = lr
            out.append(lr)
        else:
            out.append(opt.param_groups[0]["lr"])
            sched.step()
    return out


def _run_with_resumes(tmp_path, breaks, total=TOTAL, restore_sched=True):
    """Run to ``total`` steps, saving and reloading at each step in ``breaks``.

    Mirrors the production path: save via _save_resume_state, then on resume
    build a fresh optimizer, load the optimizer state (which restores the
    decayed LR -- the trigger), then restore the schedule.
    """
    opt = _fresh_opt()
    sched = build_cosine_scheduler(opt, total, 0)
    lrs, step = [], 0
    for stop in list(breaks) + [total]:
        while step < stop:
            if step < WARMUP:
                lr = PEAK * step / WARMUP
                for pg in opt.param_groups:
                    pg["lr"] = lr
                lrs.append(lr)
            else:
                lrs.append(opt.param_groups[0]["lr"])
                sched.step()
            step += 1
        if stop == total:
            break
        _save_resume_state(str(tmp_path), step, opt, None, lr_scheduler=sched)
        state = _load_resume_state(str(tmp_path), expected_step=step)
        opt = _fresh_opt()
        opt.load_state_dict(state["optimizer_state_dict"])   # restores decayed lr
        sched = build_cosine_scheduler(opt, total, 0)
        if restore_sched and state.get("sched_state_dict") is not None:
            sched.load_state_dict(state["sched_state_dict"])
        else:
            for _ in range(step):        # the old path: replays warmup steps too
                sched.step()
    return lrs


def test_two_chained_resumes_reproduce_the_uninterrupted_lr_exactly(tmp_path):
    """The real gate: a SEQUENCE, not one step.

    Compares every step from the first break to the end. Comparing only the
    step after a resume would pass even if the curve's shape were wrong.
    """
    ref = _reference_lrs()
    got = _run_with_resumes(tmp_path, breaks=[5000, 12000])
    assert len(got) == len(ref)
    worst = max(abs(a - b) for a, b in zip(got[5000:], ref[5000:]))
    assert worst == 0.0, f"LR diverged after resume, max |delta| = {worst:.3e}"


def test_the_error_compounds_so_one_resume_is_not_enough_to_test(tmp_path):
    """Anti-vacuity: the OLD path must fail this suite, and fail worse with each
    resume. If replaying ever stops diverging, these tests are measuring nothing
    and should be deleted rather than trusted."""
    ref = _reference_lrs()
    one = _run_with_resumes(tmp_path, breaks=[5000], restore_sched=False)
    two = _run_with_resumes(tmp_path, breaks=[5000, 12000], restore_sched=False)

    err_one = abs(one[12050] - ref[12050]) / ref[12050]
    err_two = abs(two[12050] - ref[12050]) / ref[12050]
    assert err_one > 0.01, "the old replay path no longer diverges at all"
    assert err_two > err_one * 2, (
        f"error did not compound across resumes ({err_one:.1%} -> {err_two:.1%}); "
        f"a single-resume test would be enough and this one is redundant")


def test_reproduces_the_lr_actually_logged_by_the_broken_48m_run(tmp_path):
    """Pins the fix to the observed failure, not to a theory of it.

    These are the values from checkpoints/code_48m/training_log.csv. If a future
    change reintroduces the bug, this says so in the same numbers we debugged.
    """
    broken = _run_with_resumes(tmp_path, breaks=[5000, 12000], restore_sched=False)
    assert broken[12050] == pytest.approx(0.00516675, rel=1e-4), (
        f"expected the logged broken value 0.00516675, got {broken[12050]:.8f} "
        f"-- the reproduction no longer matches the incident")
    fixed = _run_with_resumes(tmp_path, breaks=[5000, 12000])
    assert fixed[12050] == pytest.approx(_reference_lrs()[12050], rel=1e-12)


def test_scheduler_state_is_actually_persisted(tmp_path):
    """The plumbing. Without this key the code silently takes the legacy path,
    which is the bug -- and every other test here would still pass by falling
    back to it."""
    opt = _fresh_opt()
    sched = build_cosine_scheduler(opt, TOTAL, 0)
    for _ in range(10):
        sched.step()
    _save_resume_state(str(tmp_path), 10, opt, None, lr_scheduler=sched)
    state = _load_resume_state(str(tmp_path), expected_step=10)
    assert state.get("sched_state_dict") is not None, \
        "_save_resume_state did not persist the LR schedule"
    assert state["sched_state_dict"]["last_epoch"] == 10
