"""--dropout: the flag reaches every module, survives a checkpoint, and resume
refuses a mismatch (2026-10-15 speed sweep follow-up).

Author: Rambarun Komaljeet

Dropout has no parameters, so a run that silently resumed at the wrong p would
load cleanly and train a different recipe with nothing raising. These tests
pin the value end to end and check that p=0 really is off in train mode.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN  # noqa: E402

PLAN = ["deltanet", "transformer", "deltanet", "deltanet"]
VOCAB = 512


def _build(p):
    torch.manual_seed(0)
    return EdgeHVN(vocab_size=VOCAB, d_model=64, num_layers=len(PLAN), n_heads=4,
                   block_plan=PLAN, attn_window=64, tie_embeddings=True, dropout=p)


def _ps(m):
    return [mod.p for mod in m.modules() if isinstance(mod, torch.nn.Dropout)]


@pytest.mark.parametrize("p", [0.0, 0.1, 0.25])
def test_every_dropout_module_gets_the_requested_p(p):
    ps = _ps(_build(p))
    assert ps, "no Dropout modules found -- this test would pass vacuously"
    assert set(ps) == {p}, f"requested p={p}, modules have {sorted(set(ps))}"


def test_p0_is_off_in_train_mode_and_p01_is_not():
    """The second half keeps the first honest: if train-mode forwards were
    deterministic anyway (e.g. dropout never reached), p=0 would prove nothing."""
    x = torch.randint(0, VOCAB, (2, 32))
    m0, m1 = _build(0.0).train(), _build(0.1).train()
    with torch.no_grad():
        assert torch.equal(m0(x), m0(x)), "p=0 forward is stochastic -- dropout still on"
        assert not torch.equal(m1(x), m1(x)), "p=0.1 forward is deterministic -- dropout never ran"


def test_checkpoint_round_trip_keeps_p(tmp_path):
    from train_edge_conversational import _save_edge_checkpoint, load_checkpoint_for_resume

    path = str(tmp_path / "p0.pt")
    _save_edge_checkpoint(_build(0.0), path, global_step=3)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["dropout"] == 0.0
    model, step, _ = load_checkpoint_for_resume(path, torch.device("cpu"))
    assert step == 3
    assert set(_ps(model)) == {0.0}, "resume rebuilt the model at a different dropout"


def test_pre_flag_checkpoint_resumes_at_the_old_default(tmp_path):
    from train_edge_conversational import _save_edge_checkpoint, load_checkpoint_for_resume

    path = str(tmp_path / "old.pt")
    _save_edge_checkpoint(_build(0.1), path, global_step=1)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    del ckpt["dropout"]
    torch.save(ckpt, path)
    model, _, _ = load_checkpoint_for_resume(path, torch.device("cpu"))
    assert set(_ps(model)) == {0.1}


def test_resume_guard_refuses_a_dropout_mismatch():
    from train_edge_conversational import _check_resume_matches_request

    args = SimpleNamespace(dropout=0.1, arch="hvn-hybrid")
    requested = SimpleNamespace(vocab_size=None)
    ckpt = {"dropout": 0.0, "block_plan": PLAN}
    with pytest.raises(SystemExit, match="dropout"):
        _check_resume_matches_request(requested, ckpt, "x.pt", args)
    # Matching value, and a checkpoint that predates the field, both pass.
    _check_resume_matches_request(requested, {"dropout": 0.1, "block_plan": PLAN}, "x.pt", args)
    _check_resume_matches_request(requested, {"block_plan": PLAN}, "x.pt", args)
