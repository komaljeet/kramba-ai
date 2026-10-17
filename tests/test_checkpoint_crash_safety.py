"""Checkpoint saves survive being killed, and validation does not leak RAM.

Author: Rambarun Komaljeet

The 270M pretrain (1024x20, hvn-hybrid) was killed at its first save, step
2000, leaving edge_conversational_step2000.pt complete and
resume_state.pt.tmp 99.8% written. Two separate defects:

1. Ordering. The model checkpoint was written first, the resume state second,
   so a kill in between left the step-N weights with no step-N optimizer, data
   position or LR state; a restart silently began fresh. The model file itself
   was also written non-atomically, so a kill mid-write left a truncated
   *_stepN.pt that resume would pick as newest and fail to load.
2. RAM. evaluate() ignored --chunked-loss and built full logits; on the
   6 GB card that grew the CUDA reserve past physical VRAM, which WDDM backs
   with host RAM for the rest of the run (+1.28 GB working set, measured). The
   first validation runs at step 2000 -- right before the first save.

The crash tests interrupt the second of two saves at EVERY torch.save and
os.replace call it makes and assert the invariant resume depends on: the
newest complete model checkpoint has a resume state at exactly its step.
test_crash_matrix_is_not_vacuous fails if the matrix ever stops reaching
the interesting points.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_edge_conversational as tec  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402

PLAN = ["deltanet", "transformer", "deltanet", "deltanet"]
VOCAB = 512


def _model():
    torch.manual_seed(0)
    return EdgeHVN(vocab_size=VOCAB, d_model=64, num_layers=len(PLAN), n_heads=4,
                   block_plan=PLAN, attn_window=64, tie_embeddings=True, dropout=0.0)


def _opt_after_steps(model, n=2, opt=None):
    opt = opt or torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(n):
        x = torch.randint(0, VOCAB, (1, 16))
        model(x).float().square().mean().backward()
        opt.step()
        opt.zero_grad()
    return opt


class _Killed(Exception):
    pass


def _save_pair(model, opt, d, step):
    tec._save_checkpoint_pair(model, os.path.join(d, f"edge_conversational_step{step}.pt"),
                              d, step, opt, data_state={"epoch": 0, "consumed": step})


def _count_calls(model, opt, d, step):
    """How many torch.save / os.replace calls one pair-save makes."""
    counts = {"save": 0, "replace": 0}
    real_save, real_replace = torch.save, os.replace

    def save(*a, **k):
        counts["save"] += 1
        return real_save(*a, **k)

    def replace(*a, **k):
        counts["replace"] += 1
        return real_replace(*a, **k)

    torch.save, os.replace = save, replace
    try:
        _save_pair(model, opt, d, step)
    finally:
        torch.save, os.replace = real_save, real_replace
    return counts


def _interrupted_pair(model, opt, d, step, kind, at):
    """Run a pair-save that is killed at call number `at` of `kind`.

    A killed torch.save leaves a PARTIAL file at its target (what a real kill
    does), so a non-atomic writer would expose a truncated checkpoint.
    """
    real_save, real_replace = torch.save, os.replace
    n = {"save": 0, "replace": 0}

    def save(obj, f, *a, **k):
        n["save"] += 1
        if kind == "save" and n["save"] == at:
            with open(f, "wb") as fh:
                fh.write(b"PK\x03\x04truncated")
            raise _Killed()
        return real_save(obj, f, *a, **k)

    def replace(src, dst):
        n["replace"] += 1
        if kind == "replace" and n["replace"] == at:
            raise _Killed()
        return real_replace(src, dst)

    torch.save, os.replace = save, replace
    try:
        with pytest.raises(_Killed):
            _save_pair(model, opt, d, step)
    finally:
        torch.save, os.replace = real_save, real_replace


def _crash_points(tmp_path):
    m = _model()
    o = _opt_after_steps(m)
    probe = tmp_path / "probe"
    probe.mkdir()
    _save_pair(m, o, str(probe), 1)
    c = _count_calls(m, o, str(probe), 2)   # second save: a previous state exists
    return ([("save", i) for i in range(1, c["save"] + 1)]
            + [("replace", i) for i in range(1, c["replace"] + 1)])


def test_crash_matrix_is_not_vacuous(tmp_path):
    pts = _crash_points(tmp_path)
    # two writes (resume state, model); renames: tmp->final for each + .prev demotion
    assert sum(k == "save" for k, _ in pts) >= 2, pts
    assert sum(k == "replace" for k, _ in pts) >= 3, pts


@pytest.mark.parametrize("idx", range(5))
def test_any_interrupted_save_leaves_a_matching_resume_point(tmp_path, idx):
    pts = _crash_points(tmp_path)
    assert idx < len(pts), "crash matrix shrank -- see test_crash_matrix_is_not_vacuous"
    kind, at = pts[idx]
    d = tmp_path / "run"
    d.mkdir()
    model = _model()
    opt = _opt_after_steps(model)
    _save_pair(model, opt, str(d), 2000)
    good = {k: v.clone() for k, v in opt.state_dict()["state"][0].items() if torch.is_tensor(v)}
    _opt_after_steps(model, 1, opt)           # state moves on before the next save
    _interrupted_pair(model, opt, str(d), 4000, kind, at)

    latest = tec.find_latest_checkpoint(str(d))
    assert latest is not None
    ck = torch.load(latest, map_location="cpu", weights_only=False)  # must be loadable
    step = ck["global_step"]
    state = tec._load_resume_state(str(d), step)
    assert state is not None, (
        f"killed at {kind} #{at}: newest checkpoint is step {step} but no resume "
        f"state matches it -- a restart would silently drop optimizer/data/LR state")
    assert state["global_step"] == step
    if step == 2000:
        saved = state["optimizer_state_dict"]["state"][0]
        for k, v in good.items():
            assert torch.equal(saved[k], v), k


def test_uninterrupted_saves_round_trip_and_keep_prev(tmp_path):
    d = str(tmp_path)
    model = _model()
    opt = _opt_after_steps(model)
    _save_pair(model, opt, d, 10)
    _opt_after_steps(model, 1, opt)
    _save_pair(model, opt, d, 20)
    assert not [f for f in os.listdir(d) if f.endswith(".tmp")]
    assert os.path.exists(os.path.join(d, tec.RESUME_STATE_PREV_FILENAME))
    st = tec._load_resume_state(d, 20)
    assert st["global_step"] == 20 and st["data_state"]["consumed"] == 20
    n = 0
    for pid, s in opt.state_dict()["state"].items():
        for k, v in s.items():
            if torch.is_tensor(v):
                assert torch.equal(st["optimizer_state_dict"]["state"][pid][k], v)
                n += 1
    assert n > 0
    ck = torch.load(os.path.join(d, "edge_conversational_step20.pt"),
                    map_location="cpu", weights_only=False)
    for k, v in model.state_dict().items():
        assert torch.equal(ck["model_state_dict"][k], v), k
    # The older checkpoint pairs with .prev.
    assert tec._load_resume_state(d, 10)["global_step"] == 10


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU path under test")
def test_evaluate_chunked_matches_full_logits():
    dev = torch.device("cuda")
    model = _model().to(dev)
    torch.manual_seed(1)
    loader = [(torch.randint(0, VOCAB, (2, 128)), torch.randint(0, VOCAB, (2, 128)),
               torch.tensor([0, 2])) for _ in range(3)]
    full, full_src = tec.evaluate(model, loader, dev, precision="fp32")
    chk, chk_src = tec.evaluate(model, loader, dev, precision="fp32", chunked_loss=32)
    assert full > 1.0                     # untrained: about ln(512)=6.2, not floored
    assert abs(full - chk) < 1e-5, (full, chk)
    assert abs(full_src[0] - chk_src[0]) < 1e-5 and abs(full_src[2] - chk_src[2]) < 1e-5


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU path under test")
def test_evaluate_chunked_never_builds_full_logits():
    """The point of the fix: no [B, S, vocab] tensor. Peak CUDA memory of the
    chunked eval must be far below one full-logit tensor."""
    dev = torch.device("cuda")
    vocab = 32768
    torch.manual_seed(0)
    model = EdgeHVN(vocab_size=vocab, d_model=64, num_layers=2, n_heads=4,
                    block_plan=["deltanet", "transformer"], attn_window=64,
                    tie_embeddings=True, dropout=0.0).to(dev)
    S = 2048
    loader = [(torch.randint(0, vocab, (1, S)), torch.randint(0, vocab, (1, S)),
               torch.tensor([0]))]
    logits_bytes = S * vocab * 4
    extra = {}
    for chunk in (0, 128):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        base = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        tec.evaluate(model, loader, dev, precision="fp32", chunked_loss=chunk)
        extra[chunk] = torch.cuda.max_memory_allocated() - base
    # Full path holds at least the logits (the honesty check: if it did not,
    # this comparison could not detect anything) ...
    assert extra[0] >= logits_bytes, (extra, logits_bytes)
    # ... and the chunked path saves at least one whole logit tensor of it.
    assert extra[0] - extra[128] >= logits_bytes, (extra, logits_bytes)
