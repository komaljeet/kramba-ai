"""--max-train-windows: the capped pool the trainer uses is the one the gap
evaluator rebuilds, it avoids the held-out tail, and the unseen set never
overlaps or touches it."""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_edge_conversational as tec  # noqa: E402
from scripts import eval_repeat_gap as erg  # noqa: E402

SEQ = 16
N_TOK = SEQ * 5000 + 7


def _trainer_pool(tmp_path, n):
    path = tmp_path / "code.bin"
    np.arange(N_TOK, dtype=np.uint16).tofile(path)
    ds = object.__new__(tec.MixedLocalDataset)
    ds.sources = [np.memmap(path, dtype=np.uint16, mode="r")]
    ds.total_tokens = N_TOK
    ds._build_code_index(SEQ, 0.0, n)
    return path, ds


def test_trainer_pool_matches_evaluator(tmp_path):
    path, ds = _trainer_pool(tmp_path, 300)
    assert len(ds) == 300
    pool, unseen = erg.pool_and_unseen(str(path), SEQ, 300, 50)
    # Same windows in the same order: the evaluator's "seen" set is really seen.
    assert np.array_equal(pool, ds.starts)


def test_uncapped_unchanged(tmp_path):
    _, ds = _trainer_pool(tmp_path, None)
    assert len(ds) == (N_TOK - SEQ - 1) // SEQ + 1


def test_pool_avoids_tail_and_unseen_is_disjoint(tmp_path):
    path, ds = _trainer_pool(tmp_path, 1000)
    cut = N_TOK - int(N_TOK * tec.CAP_TAIL_FRACTION)
    assert ds.starts.max() < cut
    _, unseen = erg.pool_and_unseen(str(path), SEQ, 1000, 200)
    assert len(unseen) == 200
    pool = set(ds.starts.tolist())
    for s in unseen.tolist():
        assert s not in pool and s - SEQ not in pool and s + SEQ not in pool
        assert s < cut


def test_cap_too_large_fails_loudly(tmp_path):
    with pytest.raises(ValueError):
        _trainer_pool(tmp_path, 5000)
