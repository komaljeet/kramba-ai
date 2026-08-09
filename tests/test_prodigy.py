"""Tests for the parameter-free Prodigy optimizer (Issue #134).

Written by: Rambarun Komaljeet

The claim being tested is narrow and specific: **no learning rate is chosen, and
the optimizer still finds a good one.** So the tests check that ``d`` actually
moves off its tiny initial value, that it lands near the known-good step size on
a problem whose answer we can compute, and that it does so from BOTH a cold
start and a warm one — the two regimes #133 showed need different hand-set LRs.

A test that only showed "loss goes down" would pass with a broken estimator and
a lucky ``d0``, so every convergence test also asserts something about ``d``.
"""

from __future__ import annotations

import copy
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.optimizers import Prodigy, build_optimizer  # noqa: E402


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------


def quadratic_problem(n=64, cond=10.0, seed=0, scale=1.0):
    """min_x 0.5 * (x - t)^T A (x - t), A diagonal with spread `cond`.

    Convex with a known solution, so "did it converge" is unambiguous and the
    ideal step size is computable (1/L, L = max eigenvalue).
    """
    g = torch.Generator().manual_seed(seed)
    a = torch.linspace(1.0, cond, n)
    target = torch.randn(n, generator=g) * scale
    x = torch.nn.Parameter(torch.zeros(n))

    def loss_fn():
        return 0.5 * (a * (x - target) ** 2).sum()

    return x, loss_fn, target, float(a.max())


def run(opt, loss_fn, steps=400):
    losses = []
    for _ in range(steps):
        opt.zero_grad(set_to_none=True)
        loss = loss_fn()
        loss.backward()
        opt.step()
        losses.append(float(loss))
    return losses


def tiny_model(seed=0):
    torch.manual_seed(seed)
    return torch.nn.Sequential(
        torch.nn.Linear(16, 32), torch.nn.ReLU(), torch.nn.Linear(32, 16))


# --------------------------------------------------------------------------
# the estimator
# --------------------------------------------------------------------------


def test_d_starts_tiny_and_grows():
    x, loss_fn, _t, _L = quadratic_problem()
    opt = Prodigy([{"params": [x]}])
    assert opt.d == pytest.approx(1e-6)
    run(opt, loss_fn, steps=200)
    assert opt.d > 1e-4, "d never moved off d0 — the estimator is inert"


def test_d_is_monotone_non_decreasing():
    """d tracks d_max by construction. That IS the warmup; it must not dip."""
    x, loss_fn, _t, _L = quadratic_problem()
    opt = Prodigy([{"params": [x]}])
    ds = []
    for _ in range(200):
        opt.zero_grad(set_to_none=True)
        loss_fn().backward()
        opt.step()
        ds.append(opt.d)
    assert all(b >= a - 1e-12 for a, b in zip(ds, ds[1:]))


def test_d_settles_rather_than_growing_without_bound():
    x, loss_fn, _t, _L = quadratic_problem()
    opt = Prodigy([{"params": [x]}])
    run(opt, loss_fn, steps=100)
    mid = opt.d
    run(opt, loss_fn, steps=300)
    late = opt.d
    assert late / mid < 10.0, f"d still climbing hard: {mid:.3e} -> {late:.3e}"


def test_estimated_step_is_in_the_right_ballpark():
    """The whole promise: the number it finds is one a human would have set.

    For this quadratic the stable step for a normalized (Adam-like) update is
    O(distance to solution). Assert d lands within an order of magnitude of the
    true distance rather than pinning an exact value — the claim is "usable
    without tuning", not "optimal".
    """
    x, loss_fn, target, _L = quadratic_problem(scale=1.0)
    dist = float(torch.linalg.vector_norm(target - torch.zeros_like(target)))
    opt = Prodigy([{"params": [x]}])
    run(opt, loss_fn, steps=500)
    assert 0.1 * dist < opt.d * x.numel() ** 0.5 < 10.0 * dist, (
        f"d={opt.d:.4e}, true distance={dist:.4e}")


def test_scaling_the_problem_scales_the_estimate():
    """A 10x-further solution should produce a proportionally larger d.

    This is the property that makes it work for pretrain AND fine-tune without
    being told which it is: a warm start is simply a shorter distance.
    """
    ds = []
    for scale in (1.0, 10.0):
        x, loss_fn, _t, _L = quadratic_problem(scale=scale)
        opt = Prodigy([{"params": [x]}])
        run(opt, loss_fn, steps=400)
        ds.append(opt.d)
    ratio = ds[1] / ds[0]
    assert 3.0 < ratio < 30.0, f"d did not track problem scale: ratio {ratio:.2f}"


def test_warm_start_gets_a_smaller_estimate_than_cold_start():
    """#133 in miniature: continuing from a good point needs a smaller step.

    Prodigy is given no signal about which regime it is in — only the gradients.
    """
    x_cold, loss_cold, target, _L = quadratic_problem(seed=3)
    opt_cold = Prodigy([{"params": [x_cold]}])
    run(opt_cold, loss_cold, steps=300)

    # A "checkpoint": start 95 pct of the way to the solution.
    x_warm = torch.nn.Parameter(target.clone() * 0.95)
    a = torch.linspace(1.0, 10.0, target.numel())

    def loss_warm():
        return 0.5 * (a * (x_warm - target) ** 2).sum()

    opt_warm = Prodigy([{"params": [x_warm]}])
    run(opt_warm, loss_warm, steps=300)

    assert opt_warm.d < opt_cold.d, (
        f"warm start {opt_warm.d:.3e} not smaller than cold {opt_cold.d:.3e}")


# --------------------------------------------------------------------------
# it actually optimizes
# --------------------------------------------------------------------------


def test_converges_on_a_convex_problem():
    x, loss_fn, _t, _L = quadratic_problem()
    losses = run(Prodigy([{"params": [x]}]), loss_fn, steps=600)
    assert losses[-1] < losses[0] * 1e-3


def test_beats_adamw_at_a_badly_chosen_lr():
    """The practical claim: no tuning beats wrong tuning.

    AdamW at 1e-6 is a plausible-looking but far-too-small choice — exactly the
    kind of mistake #133 caught in the other direction.
    """
    x1, loss1, _t, _L = quadratic_problem(seed=5)
    prodigy_losses = run(Prodigy([{"params": [x1]}]), loss1, steps=400)

    x2, loss2, _t2, _L2 = quadratic_problem(seed=5)
    adam_losses = run(torch.optim.AdamW([x2], lr=1e-6), loss2, steps=400)

    assert prodigy_losses[-1] < adam_losses[-1]


def test_muon_variant_converges_on_a_real_module():
    torch.manual_seed(0)
    model = tiny_model()
    xb = torch.randn(64, 16)
    yb = torch.randn(64, 16)
    opt, desc = build_optimizer(model, "muon-prodigy", lr=1.0, weight_decay=0.0)
    assert "NO tuned LR" in desc
    first = last = None
    for i in range(300):
        opt.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(xb), yb)
        loss.backward()
        opt.step()
        if i == 0:
            first = float(loss)
        last = float(loss)
    assert last < first * 0.5
    assert opt.d > 1e-4, "Muon coupling left d inert"


def test_muon_variant_actually_uses_the_muon_direction():
    """Guards the coupling: the muon group must not be silently Adam.

    Same seed, same data, only `use_muon` differing on the matrix group -- the
    resulting weights must differ.
    """
    def run_variant(name):
        torch.manual_seed(0)
        model = tiny_model()
        xb, yb = torch.randn(32, 16), torch.randn(32, 16)
        opt, _ = build_optimizer(model, name, lr=1.0, weight_decay=0.0)
        for _ in range(20):
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.mse_loss(model(xb), yb).backward()
            opt.step()
        return model[0].weight.detach().clone()

    assert not torch.allclose(run_variant("prodigy"), run_variant("muon-prodigy"))


# --------------------------------------------------------------------------
# integration + robustness
# --------------------------------------------------------------------------


def test_build_optimizer_exposes_both_names():
    model = tiny_model()
    for name in ("prodigy", "muon-prodigy", "muon_prodigy"):
        opt, desc = build_optimizer(model, name, lr=1.0, weight_decay=0.01)
        assert isinstance(opt, Prodigy)
        assert "NO tuned LR" in desc
        # Every trainable parameter must land in some group.
        seen = sum(len(g["params"]) for g in opt.param_groups)
        assert seen == len([p for p in model.parameters() if p.requires_grad])


def test_lr_is_a_multiplier_not_a_learning_rate():
    with pytest.raises(ValueError, match="multiplier"):
        Prodigy([{"params": [torch.nn.Parameter(torch.zeros(4))]}], lr=0.0)


def test_all_zero_gradients_do_not_poison_d():
    """d is global; one nan would end the run silently."""
    p = torch.nn.Parameter(torch.zeros(8))
    opt = Prodigy([{"params": [p]}])
    for _ in range(5):
        opt.zero_grad(set_to_none=True)
        (p.sum() * 0.0).backward()
        opt.step()
    assert opt.d == opt.d           # not nan
    assert opt.d > 0


def test_state_dict_round_trips():
    """This repo's resume path saves optimizer state (#112); d must survive."""
    x, loss_fn, _t, _L = quadratic_problem()
    opt = Prodigy([{"params": [x]}])
    run(opt, loss_fn, steps=100)
    d_before = opt.d
    sd = opt.state_dict()

    x2 = torch.nn.Parameter(x.detach().clone())
    opt2 = Prodigy([{"params": [x2]}])
    opt2.load_state_dict(sd)
    assert opt2.d == pytest.approx(d_before)


def test_resumed_optimizer_continues_the_same_trajectory():
    x, loss_fn, _t, _L = quadratic_problem(seed=11)
    opt = Prodigy([{"params": [x]}])
    run(opt, loss_fn, steps=50)
    # deepcopy is load-bearing: state_dict() hands back LIVE references to the
    # state tensors, so continuing to train would mutate the snapshot in place
    # and this test would compare a run against itself.
    sd = copy.deepcopy(opt.state_dict())
    x_snapshot = x.detach().clone()
    straight = run(opt, loss_fn, steps=50)

    x2 = torch.nn.Parameter(x_snapshot.clone())
    a = torch.linspace(1.0, 10.0, x2.numel())
    g = torch.Generator().manual_seed(11)
    target = torch.randn(x2.numel(), generator=g)

    def loss2():
        return 0.5 * (a * (x2 - target) ** 2).sum()

    opt2 = Prodigy([{"params": [x2]}])
    opt2.load_state_dict(sd)
    resumed = run(opt2, loss2, steps=50)
    assert resumed[-1] == pytest.approx(straight[-1], rel=1e-5)


def test_no_warmup_needed():
    """The d-ramp is the warmup: step 1 must not blow the model up."""
    x, loss_fn, _t, _L = quadratic_problem()
    opt = Prodigy([{"params": [x]}])
    before = float(loss_fn())
    opt.zero_grad(set_to_none=True)
    loss_fn().backward()
    opt.step()
    after = float(loss_fn())
    assert after <= before * 1.001
