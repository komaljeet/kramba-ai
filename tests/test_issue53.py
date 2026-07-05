"""
Test suite for Issue #53 — curriculum dataset loaders.

Validates the coding-curriculum loaders in ``eval/curriculum.py`` (re-exported
from ``eval/datasets.py``): stage order, offline synthetic fallback, license
filtering (permissive only), near-dedup, budget caps, train/valid disjointness,
and a local-dir override. Runs fully offline (``allow_download=False``) so it is
CI/6 GB-box safe.

Run from the repo root:
    python tests/test_issue53.py
"""

import os
import sys
import tempfile

# Force UTF-8 stdout so the ✓ characters don't crash a legacy cp1252 console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.datasets import (
    CURRICULUM_STAGES,
    CURRICULUM_BUDGETS,
    load_curriculum_stage,
    is_permissive_source,
    dedup_documents,
    smoke_load_curriculum,
    DATASET_REGISTRY,
)
from eval.curriculum import CURRICULUM_EXTENSIONS


def test_stage_order():
    """Curriculum is the documented 6-stage order and every stage is configured."""
    assert CURRICULUM_STAGES == ["html", "css", "js", "php", "python", "devops"]
    for stage in CURRICULUM_STAGES:
        assert stage in CURRICULUM_BUDGETS, f"no budget for {stage}"
        assert stage in CURRICULUM_EXTENSIONS, f"no extensions for {stage}"
        assert f"curriculum:{stage}" in DATASET_REGISTRY, f"{stage} not registered"
    print("[1/6] curriculum stage order + config .......................... OK")


def test_license_filter():
    """Copyleft markers are rejected; permissive / unlicensed code passes."""
    gpl = "/* This program is free software under the GNU General Public License */\nx=1"
    lgpl = "// Licensed under the LGPL\nfunction f(){}"
    ccsa = "<!-- CC-BY-SA 4.0 -->\n<p>hi</p>"
    mit = "// SPDX-License-Identifier: MIT\nconst a = 1;"
    plain = "def add(a, b):\n    return a + b\n"
    assert not is_permissive_source(gpl)
    assert not is_permissive_source(lgpl)
    assert not is_permissive_source(ccsa)
    assert is_permissive_source(mit)
    assert is_permissive_source(plain)
    print("[2/6] license filter (permissive only) ......................... OK")


def test_dedup():
    """Exact and near-duplicate documents are removed; distinct ones kept."""
    a = "\n".join(f"def step_{i}(value):\n    return value + {i}" for i in range(40))
    a_dup = a  # exact duplicate
    a_near = a.replace("return value + 39", "return value + 40")  # one-token change
    b = "class Widget:\n    def render(self):\n        return '<div/>'\n"
    kept = dedup_documents([a, a_dup, a_near, b])
    assert a in kept and b in kept, "distinct docs must survive"
    assert len(kept) == 2, f"expected exact+near dups removed, got {len(kept)}"
    print("[3/6] near-dedup (exact + Jaccard) ............................. OK")


def test_offline_load_all_stages():
    """Every stage loads offline (synthetic fallback), respecting budget caps."""
    for stage in CURRICULUM_STAGES:
        train, valid = load_curriculum_stage(
            stage, allow_download=False,
            max_train_chars=200_000, max_eval_chars=20_000,
        )
        assert len(train) > 1000, f"{stage}: train too small ({len(train)})"
        assert len(valid) > 0, f"{stage}: empty valid"
        assert len(train) <= 200_000 and len(valid) <= 20_000, f"{stage}: budget exceeded"
        # No copyleft leaked through into the assembled streams.
        assert is_permissive_source(train), f"{stage}: copyleft leaked into train"
    print("[4/6] offline load all 6 stages (budget-capped) ................ OK")


def test_deterministic_and_disjoint():
    """Loads are deterministic for a fixed seed; train/valid are disjoint docs."""
    t1, v1 = load_curriculum_stage("python", allow_download=False, seed=7)
    t2, v2 = load_curriculum_stage("python", allow_download=False, seed=7)
    assert t1 == t2 and v1 == v2, "not deterministic for fixed seed"
    # Different seed → different shuffle/split (very likely different text).
    t3, _ = load_curriculum_stage("python", allow_download=False, seed=99)
    assert t1 != t3, "seed had no effect"
    # Train and valid should not be the identical stream.
    assert t1 != v1
    print("[5/6] deterministic + disjoint train/valid ..................... OK")


def test_local_dir_override():
    """A local directory of files is preferred over the synthetic fallback."""
    with tempfile.TemporaryDirectory() as tmp:
        stage_dir = os.path.join(tmp, "python")
        os.makedirs(stage_dir)
        marker = "def unmistakable_marker_function():\n    return 1234567\n"
        # Write several distinct files so there is a train + valid split.
        for i in range(12):
            with open(os.path.join(stage_dir, f"mod_{i}.py"), "w", encoding="utf-8") as f:
                f.write(f"# file {i}\n" + marker + f"\nVALUE_{i} = {i}\n")
        train, valid = load_curriculum_stage(
            "python", local_dir=tmp, allow_download=False,
        )
        assert "unmistakable_marker_function" in train, "local files not used"
        assert len(valid) > 0
    print("[6/6] local-dir override ....................................... OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Issue #53 — curriculum dataset loaders")
    print("=" * 60)
    test_stage_order()
    test_license_filter()
    test_dedup()
    test_offline_load_all_stages()
    test_deterministic_and_disjoint()
    test_local_dir_override()
    # Print a small size report for the record (offline).
    print("-" * 60)
    report = smoke_load_curriculum(allow_download=False)
    for stage, info in report.items():
        print(f"  {stage:8s} train={info['train_chars']:>9,} eval={info['eval_chars']:>8,} "
              f"ext={info['extensions']}")
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
