"""scripts/size_for_pretrain.py's parser must fail loudly, not publish a number,
when the calibration run could not have measured what it claims.

CPU only; fabricates a training log + CSV in tmp_path.
"""
from __future__ import annotations

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "scripts"))
sys.path.insert(0, _REPO)

import size_for_pretrain as S  # noqa: E402


def _write(tmp_path, secs, log_extra=""):
    log = tmp_path / "run.log"
    log.write_text("  Architecture: hvn-hybrid  |  total params: 1,000\n"
                   "  [vram] peak 50% of card\n" + log_extra, encoding="utf-8")
    csvp = tmp_path / "training_log.csv"
    rows = ["step,epoch,loss,ppl,lr,peak_vram_gb,reserved_vram_gb,tokens_sec,timestamp"]
    for i, sec in enumerate(secs, 1):
        rows.append(f"{i},0,9,9,0.001,2.0,2.5,{S.TOKENS_PER_STEP / sec:.4f},t")
    csvp.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return str(log), str(csvp)


def test_bcd_rate_is_rotation_mean_and_drops_warmup(tmp_path):
    # 3 blocks with very different costs; step 1 is a 100 s warm-up outlier.
    secs = [100.0] + [1.0, 2.0, 6.0] * 3
    r = S.parse_run(*_write(tmp_path, secs,
                    "Block-coordinate-descent: 3 blocks (balanced partition)\n"), 3)
    assert "fail" not in r, r
    assert abs(r["s_per_step"] - 3.0) < 1e-6          # mean over whole rotations
    assert r["measured_steps"] == 9
    assert r["half_drift"] < 1e-9                       # rotation vs rotation


def test_partial_rotation_is_trimmed(tmp_path):
    # 4 full rotations + 2 stray steps of the cheap blocks: must be excluded,
    # or the cheap blocks get over-weighted and BCD looks faster than it is.
    secs = [100.0] + [1.0, 2.0, 6.0] * 4 + [1.0, 2.0]
    r = S.parse_run(*_write(tmp_path, secs), 3)
    assert r["measured_steps"] == 12 and abs(r["s_per_step"] - 3.0) < 1e-6


def test_too_short_run_fails_instead_of_reporting(tmp_path):
    r = S.parse_run(*_write(tmp_path, [100.0, 1.0, 1.0]), None)
    assert "fail" in r and "tok_s" not in r


def test_under_two_rotations_fails(tmp_path):
    r = S.parse_run(*_write(tmp_path, [100.0] + [1.0] * 30), 20)
    assert "fail" in r


def test_spill_guard_text_is_a_failure(tmp_path):
    r = S.parse_run(*_write(tmp_path, [100.0] + [1.0] * 30,
                    "VRAM SPILL -- refusing to continue\n"), None)
    assert r["fail"].startswith("spill")


def test_planned_param_counts_match_todays_sweep():
    # 2026-09-29 measured 1024x20 hybrid at 262,052,497 total params in the real
    # script. If plan_shape drifts from what the script builds, days are wrong.
    s = S.plan_shape(1024, 20)
    assert s["total"] == 262_052_497
    assert s["blocks"] == {"layer": 22, "balanced": 6}
