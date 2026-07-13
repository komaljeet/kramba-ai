"""Tests for the benchmark + identity-guardrail harness (Issue #79).

The harness's own comparison logic (compare()) is pure Python -- these tests
exercise it directly, CPU-only, fast, no GPU required. What actually needs
verifying: it must PASS when two configs produce identical losses (the
common case: an optimization that should be numerically transparent, e.g.
grad_checkpoint, BCD, offload) and FAIL when they genuinely diverge (a real
regression, or an intentionally lossy optimization exceeding tolerance) --
a harness that can't tell these apart is worse than no harness.

Also a tiny CPU end-to-end smoke test of run_probe() itself, confirming the
full train-loop-and-report path runs without CUDA (`require_cuda=False`).

Run:  python -m pytest tests/test_bench_harness.py -q
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.bench_harness import compare, run_probe

VOCAB = 8000


def _tiny_cfg(label, **overrides):
    cfg = {
        "label": label,
        "require_cuda": False,
        "model_kwargs": {
            "d_model": 32, "num_layers": 2, "n_heads": 4,
            "use_deltanet": True, "dropout": 0.0, "chunk_size": 8,
        },
        "optimizer": "AdamW",
        "lr": 1e-3,
        "batch_size": 2,
        "seq_len": 16,
        "steps": 3,
        "data_seed": 0,
        "model_seed": 0,
    }
    cfg.update(overrides)
    return cfg


def test_run_probe_cpu_smoke():
    """The full harness train loop runs end to end without CUDA."""
    result = run_probe(_tiny_cfg("cpu_smoke"))
    assert result["verdict"] == "ok"
    assert len(result["losses"]) == 3
    assert all(isinstance(l, float) for l in result["losses"])


def test_compare_identical_configs_pass():
    """Same seed, same everything -> losses must match exactly, parity PASS.
    This is the harness's core job: confirm a numerically-transparent change
    (in this synthetic case, no change at all) reports PASS, not a false
    alarm."""
    a = run_probe(_tiny_cfg("a"))
    b = run_probe(_tiny_cfg("b"))
    verdict = compare(a, b, loss_atol=1e-9, loss_rtol=0.0)
    assert verdict["loss_parity"].startswith("PASS"), verdict


def test_compare_diverging_configs_fails():
    """Different learning rates -> loss trajectories must diverge -> parity
    FAIL. Confirms the harness actually catches a real difference instead of
    always passing."""
    a = run_probe(_tiny_cfg("a", lr=1e-3, steps=5))
    b = run_probe(_tiny_cfg("b", lr=1.0, steps=5))  # deliberately blown-up lr
    verdict = compare(a, b, loss_atol=1e-4, loss_rtol=1e-4)
    assert verdict["loss_parity"].startswith("FAIL"), verdict


def test_compare_skips_on_non_ok_verdict():
    a = run_probe(_tiny_cfg("a"))
    b = {"label": "b", "verdict": "oom"}
    verdict = compare(a, b)
    assert verdict["loss_parity"] == "skipped (non-ok verdict)"


def test_compare_length_mismatch_fails():
    a = run_probe(_tiny_cfg("a", steps=3))
    b = run_probe(_tiny_cfg("b", steps=5))
    verdict = compare(a, b)
    assert verdict["loss_parity"].startswith("FAIL (length mismatch")


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
