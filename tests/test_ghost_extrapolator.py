"""Tests for the GHOST Phase 2 weight-delta extrapolator (issue #126).

Written by: Rambarun Komaljeet

Synthetic tensors only — no GPU, no real checkpoints. Run from the repo root:

    python tests/test_ghost_extrapolator.py
    pytest tests/test_ghost_extrapolator.py
"""

from __future__ import annotations

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghost.extrapolator import (  # noqa: E402
    ExtrapolationConfig,
    build_plan,
    extrapolate,
    extrapolate_state_dict,
)


def _sd(**kw):
    return dict(kw)


def test_linear_trajectory_is_predicted_exactly():
    """A tensor on a constant-velocity path must be predicted to machine precision."""
    torch.manual_seed(0)
    w1 = torch.randn(64, 32, dtype=torch.float64)
    v = torch.randn(64, 32, dtype=torch.float64) * 0.01
    w2, w3 = w1 + v, w1 + 2 * v
    for n, gap in [(10, 50), (50, 50), (100, 50), (500, 50)]:
        pred, plan = extrapolate(_sd(w=w1), _sd(w=w2), _sd(w=w3), n, gap)
        want = w3 + (n / gap) * v
        err = float((pred["w"] - want).abs().max())
        assert err < 1e-9, f"N={n}: max error {err}"
        assert plan.tensors[0].extrapolated, f"N={n}: linear tensor was held"
    print("ok  constant-velocity tensor predicted exactly at N=10/50/100/500")


def test_pure_noise_tensor_is_held():
    """Uncorrelated deltas must be classified high-variance and held at W3."""
    torch.manual_seed(1)
    w1 = torch.randn(128, 64, dtype=torch.float64)
    w2 = w1 + torch.randn(128, 64, dtype=torch.float64) * 0.01
    w3 = w2 + torch.randn(128, 64, dtype=torch.float64) * 0.01
    pred, plan = extrapolate(_sd(w=w1), _sd(w=w2), _sd(w=w3), 100, 50)
    tp = plan.tensors[0]
    assert not tp.extrapolated, f"noise tensor was extrapolated: {tp}"
    assert torch.equal(pred["w"], w3), "held tensor must equal W3 exactly"
    print(f"ok  noise tensor held (cos={tp.cos:.3f}) and left at W3")


def test_quadratic_trajectory_uses_the_poly_fit():
    """Constant acceleration is caught by the poly branch and beats the linear fit."""
    torch.manual_seed(2)
    base = torch.randn(32, 16, dtype=torch.float64)
    v = torch.full((32, 16), 0.01, dtype=torch.float64)
    a = torch.full((32, 16), 0.002, dtype=torch.float64)
    # W(k) = base + k*v + k(k-1)/2 * a  -> D1 = v, D2 = v + a
    w1 = base
    w2 = base + v
    w3 = base + 2 * v + a

    cfg_poly = ExtrapolationConfig(allow_poly=True)
    cfg_lin = ExtrapolationConfig(allow_poly=False)
    pred_p, plan_p = extrapolate(_sd(w=w1), _sd(w=w2), _sd(w=w3), 50, 50, cfg_poly)
    pred_l, _ = extrapolate(_sd(w=w1), _sd(w=w2), _sd(w=w3), 50, 50, cfg_lin)
    truth = base + 3 * v + 3 * a  # W(3)

    assert plan_p.tensors[0].mode == "poly", plan_p.tensors[0]
    err_p = float((pred_p["w"] - truth).abs().max())
    err_l = float((pred_l["w"] - truth).abs().max())
    assert err_p < err_l, f"poly ({err_p:.2e}) must beat linear ({err_l:.2e})"
    print(f"ok  poly fit chosen and beats linear ({err_p:.2e} vs {err_l:.2e})")


def test_non_float_and_missing_tensors_are_held():
    w1 = _sd(buf=torch.tensor([1, 2, 3]), w=torch.zeros(4, dtype=torch.float64))
    w2 = _sd(buf=torch.tensor([2, 3, 4]), w=torch.ones(4, dtype=torch.float64))
    w3 = _sd(buf=torch.tensor([3, 4, 5]), w=torch.ones(4, dtype=torch.float64) * 2,
             fresh=torch.ones(2, dtype=torch.float64))
    pred, plan = extrapolate(w1, w2, w3, 100, 50)
    modes = {t.name: t.mode for t in plan.tensors}
    assert modes["buf"] == "hold", "integer buffer must be held"
    assert modes["fresh"] == "hold", "tensor absent from earlier ckpts must be held"
    assert torch.equal(pred["buf"], w3["buf"])
    assert pred["w"].dtype == torch.float64, "dtype must round-trip"
    print("ok  non-float and newly-appearing tensors held")


def test_dead_tensor_is_held():
    """A frozen tensor (zero delta) must not divide by zero or be 'extrapolated'."""
    w = torch.ones(8, dtype=torch.float64)
    pred, plan = extrapolate(_sd(w=w), _sd(w=w), _sd(w=w), 100, 50)
    assert plan.tensors[0].mode == "hold"
    assert torch.equal(pred["w"], w)
    print("ok  zero-delta tensor held without NaN")


def test_plan_reports_parameter_coverage():
    """#126 requires the extrapolated-vs-held fraction, by parameter count."""
    torch.manual_seed(3)
    v = torch.randn(100, 100, dtype=torch.float64) * 0.01
    big1 = torch.randn(100, 100, dtype=torch.float64)
    small1 = torch.randn(10, dtype=torch.float64)
    w1 = _sd(big=big1, small=small1)
    w2 = _sd(big=big1 + v, small=small1 + torch.randn(10, dtype=torch.float64))
    w3 = _sd(big=big1 + 2 * v, small=small1 + torch.randn(10, dtype=torch.float64))
    plan = build_plan(w1, w2, w3)
    assert plan.params_total == 10_000 + 10
    assert plan.n_extrapolated == 1 and plan.n_held == 1
    assert abs(plan.param_fraction_extrapolated - 10_000 / 10_010) < 1e-9
    print(f"ok  coverage reported: {plan.param_fraction_extrapolated:.4%} of params")


def test_uneven_checkpoint_spacing_raises():
    """Unequal gaps corrupt the velocity estimate — must raise, not guess."""
    w = torch.zeros(4, dtype=torch.float64)
    ck = [{"model_state_dict": _sd(w=w + i), "global_step": s}
          for i, s in enumerate((1000, 1500, 2400))]
    try:
        extrapolate_state_dict(ck, 10)
    except ValueError as exc:
        assert "uniform" in str(exc), exc
        print("ok  uneven checkpoint spacing raises")
        return
    raise AssertionError("uneven spacing was silently accepted")


def test_state_dict_roundtrip_preserves_metadata():
    """The predicted checkpoint must stay loadable by the existing resume path."""
    torch.manual_seed(4)
    v = torch.randn(8, 8, dtype=torch.float64) * 0.01
    base = torch.randn(8, 8, dtype=torch.float64)
    ck = []
    for i, s in enumerate((1000, 1050, 1100)):
        ck.append({
            "model_state_dict": _sd(w=base + i * v),
            "global_step": s, "vocab_size": 50257, "d_model": 512,
            "num_layers": 12, "ternary": True, "block_plan": ["deltanet"] * 12,
        })
    out, plan = extrapolate_state_dict(ck, 10)
    assert out["global_step"] == 1110
    assert out["vocab_size"] == 50257 and out["ternary"] is True
    assert out["block_plan"] == ["deltanet"] * 12
    assert out["ghost_extrapolated_from"] == [1000, 1050, 1100]
    want = base + 2 * v + (10 / 50) * v
    assert float((out["model_state_dict"]["w"] - want).abs().max()) < 1e-9
    print("ok  predicted checkpoint keeps architecture metadata")


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
