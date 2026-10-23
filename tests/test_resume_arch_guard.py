"""Resume arch guard compares block plans, not "has a plan" (#175).

Author: Rambarun Komaljeet

Since b0cef7b EdgeHVN fills in a block_plan for every arch, so the old test
"checkpoint has a plan => it is hybrid" refused every plain hvn / transformer /
hvn-original-hybrid resume, and ACCEPTED a plain hvn checkpoint resumed as
hvn-hybrid (every other field matches; only the plan differs).

These go through the real save -> find -> load -> check path. Models are built
with the same constructor args build_model_and_config passes, with literal
plans for the hybrids, so the expected plans are not derived from the helper
under test.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN  # noqa: E402
from train_edge_conversational import (  # noqa: E402
    _check_resume_matches_request,
    _save_edge_checkpoint,
    find_latest_checkpoint,
    load_checkpoint_for_resume,
)

L, D, H, VOCAB = 2, 64, 2, 512
ATTN = "1"     # --attn-layers for the hybrid cases

# Constructor args main() -> build_model_and_config passes for each --arch.
CTOR = {
    "hvn": dict(use_deltanet=True, block_type=None, block_plan=None),
    "transformer": dict(use_deltanet=False, block_type="transformer", block_plan=None),
    "hvn-hybrid": dict(use_deltanet=False, block_type=None,
                       block_plan=["deltanet", "transformer"], attn_window=64),
    "hvn-original-hybrid": dict(use_deltanet=False, block_type=None,
                                block_plan=["hvn", "transformer"], attn_window=64),
}


def _build(arch):
    torch.manual_seed(0)
    return EdgeHVN(vocab_size=VOCAB, d_model=D, num_layers=L, n_heads=H,
                   tie_embeddings=False, dropout=0.1, deltanet_conv_kernel=4,
                   **CTOR[arch])


def _args(arch):
    return SimpleNamespace(arch=arch, d_model=D, num_layers=L, n_heads=H,
                           tie_embeddings=False, dropout=0.1,
                           attn_layers=ATTN if "hybrid" in arch else None)


def _saved_ckpt(tmp_path, arch, mutate=None):
    _save_edge_checkpoint(_build(arch), str(tmp_path / "edge_conversational_step10.pt"),
                          global_step=10)
    path = find_latest_checkpoint(str(tmp_path))
    if mutate is not None:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        mutate(ck)
        torch.save(ck, path)
    _, step, ck = load_checkpoint_for_resume(path, torch.device("cpu"))
    assert step == 10
    return path, ck


def _check(path, ck, arch, via_model):
    """via_model=True is main()'s call: the CLI-built model's plan is passed.
    False exercises the args-only fallback (plan derived from --arch flags)."""
    if via_model:
        m = _build(arch)
        req = SimpleNamespace(vocab_size=VOCAB, block_plan=list(m.block_plan))
    else:
        req = SimpleNamespace(vocab_size=VOCAB)
    _check_resume_matches_request(req, ck, path, _args(arch))


ROUND_TRIP = ["hvn", "transformer", "hvn-hybrid", "hvn-original-hybrid"]


@pytest.mark.parametrize("via_model", [True, False])
@pytest.mark.parametrize("arch", ROUND_TRIP)
def test_same_arch_round_trip_resumes(tmp_path, arch, via_model):
    path, ck = _saved_ckpt(tmp_path, arch)
    _check(path, ck, arch, via_model)     # must not raise


@pytest.mark.parametrize("via_model", [True, False])
@pytest.mark.parametrize("saved,requested", [
    ("hvn", "hvn-hybrid"),                  # the silent-acceptance hole
    ("hvn", "transformer"),
    ("transformer", "hvn"),
    ("hvn-hybrid", "hvn"),
    ("hvn-original-hybrid", "hvn-hybrid"),  # same attn slots, other recurrent block
    ("hvn-hybrid", "hvn-original-hybrid"),
])
def test_cross_arch_resume_is_refused(tmp_path, saved, requested, via_model):
    path, ck = _saved_ckpt(tmp_path, saved)
    with pytest.raises(SystemExit, match="block_plan"):
        _check(path, ck, requested, via_model)


def test_hybrid_with_different_attn_positions_is_refused(tmp_path):
    path, ck = _saved_ckpt(tmp_path, "hvn-hybrid")        # attention at layer 1
    args = _args("hvn-hybrid")
    args.attn_layers = "0"
    with pytest.raises(SystemExit, match="block_plan"):
        _check_resume_matches_request(SimpleNamespace(vocab_size=VOCAB), ck, path, args)


def test_plain_checkpoint_really_saves_a_plan(tmp_path):
    """Guard: the bug only exists because plain checkpoints carry a non-None
    plan. If that ever reverts to None, the round-trip tests above would pass
    through the legacy-reconstruction branch instead -- fail loudly here."""
    _, ck = _saved_ckpt(tmp_path, "hvn")
    assert ck["block_plan"] == ["deltanet"] * L
    _, ck = _saved_ckpt(tmp_path, "transformer")
    assert ck["block_plan"] == ["transformer"] * L


def _make_legacy(ck):
    ck["block_plan"] = None


@pytest.mark.parametrize("via_model", [True, False])
def test_legacy_plain_checkpoint_resumes_as_hvn_only(tmp_path, via_model):
    path, ck = _saved_ckpt(tmp_path, "hvn", mutate=_make_legacy)
    assert ck.get("block_plan") is None and ck["use_deltanet"] is True
    _check(path, ck, "hvn", via_model)
    with pytest.raises(SystemExit, match="block_plan"):
        _check(path, ck, "hvn-hybrid", via_model)
