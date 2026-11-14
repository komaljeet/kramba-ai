"""#201: 4 layers per BCD block is the trainer default under the balanced partition.

CPU only. What each test would catch:
  - the default silently staying at the budget packing (resolve + 1B layout tests);
  - the old 13-block layout becoming unreachable (explicit 0);
  - the per-layer partition starting to refuse the DEFAULT (it may only refuse an
    explicit value);
  - a pre-#201 checkpoint (13 blocks) resuming into the 7-block layout without a
    loud error naming the flag. The same-layout round trip is the control: a check
    that always raised would pass every mismatch test.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.core.block_coordinate import (  # noqa: E402
    BlockCoordinateScheduler, _balance_groups, _layer_count_groups,
    partition_named_blocks)
from hvn.edge.model import EdgeHVN  # noqa: E402
from train_edge_conversational import (  # noqa: E402
    BCD_DEFAULT_LAYERS_PER_BLOCK, _restore_optimizer_and_bcd,
    resolve_bcd_layers_per_block)


# --- the default -----------------------------------------------------------------

def test_default_is_4_under_balanced():
    assert BCD_DEFAULT_LAYERS_PER_BLOCK == 4
    assert resolve_bcd_layers_per_block(None, "balanced") == 4


def test_default_does_not_apply_to_layer_partition():
    # No flag + "layer" must stay None, or the scheduler's "needs balanced" guard
    # would refuse every --bcd-partition layer run.
    assert resolve_bcd_layers_per_block(None, "layer") is None


def test_explicit_values_kept_and_idempotent():
    for k in (0, 1, 2, 4, 5):
        r = resolve_bcd_layers_per_block(k, "balanced")
        assert r == k
        # train() re-resolves the config main() already resolved: 0 must stay 0
        # (budget), not turn back into the default 4.
        assert resolve_bcd_layers_per_block(r, "balanced") == k
    assert resolve_bcd_layers_per_block(2, "layer") == 2   # kept, so it can be refused


def test_negative_rejected():
    with pytest.raises(ValueError):
        resolve_bcd_layers_per_block(-1, "balanced")


def _tiny(num_layers=7):
    return EdgeHVN(vocab_size=512, d_model=32, num_layers=num_layers, n_heads=4,
                   block_plan=["deltanet"] * num_layers, tie_embeddings=True)


def test_layer_partition_runs_with_default_and_refuses_explicit():
    m = _tiny()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    s = BlockCoordinateScheduler(
        m, opt, partition="layer",
        layers_per_block=resolve_bcd_layers_per_block(None, "layer") or None)
    assert s.num_blocks == 1 + 7 + 1
    with pytest.raises(ValueError):
        BlockCoordinateScheduler(
            m, opt, partition="layer",
            layers_per_block=resolve_bcd_layers_per_block(4, "layer") or None)


# --- the 1B production layout (meta device: no memory, real parameter counts) ------

def _meta_1b(vocab=50260):
    plan = ["deltanet"] * 24
    plan[6] = plan[18] = "transformer"      # --attn-layers 6,18
    with torch.device("meta"):
        m = EdgeHVN(vocab_size=vocab, d_model=2048, num_layers=24, n_heads=64,
                    block_plan=plan, tie_embeddings=True, attn_window=4096, dropout=0.0)
    return partition_named_blocks(m)


def test_1b_default_is_7_blocks_embedding_alone_tail_merged():
    blocks = _meta_1b()
    k = resolve_bcd_layers_per_block(None, "balanced")
    groups = [names for names, _ in _layer_count_groups(blocks, k)]
    assert len(groups) == 7, groups
    assert groups[0] == ["token_embedding"]
    assert [sum(n.startswith("blocks.") for n in g) for g in groups[1:]] == [4] * 6
    assert groups[-1][-1] == "final_norm+decoder"
    assert all("final_norm+decoder" not in g for g in groups[:-1])


def test_1b_old_layout_still_reachable_with_0():
    # --bcd-layers-per-block 0 -> scheduler gets None -> budget packing: the
    # pre-#201 13 blocks of about 2 layers each.
    blocks = _meta_1b()
    assert (resolve_bcd_layers_per_block(0, "balanced") or None) is None
    groups = [names for names, _ in _balance_groups(blocks)]
    assert len(groups) == 13, [len(g) for g in groups]
    assert groups[0] == ["token_embedding"]
    assert groups[1] == ["blocks.0", "blocks.1"]


# --- resume across layouts -----------------------------------------------------------

def _sched(m, k):
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    return BlockCoordinateScheduler(m, opt, steps_per_block=3, partition="balanced",
                                    layers_per_block=k)


def _old_layout_state(m):
    """A pre-#201 checkpoint's BCD state: budget packing, no names recorded."""
    old = _sched(m, None)
    sd = old.state_dict()
    sd.pop("block_names"), sd.pop("layers_per_block")
    return old, sd


def test_precondition_tiny_layouts_differ_in_count():
    # Otherwise the count-mismatch tests below would test nothing.
    m = _tiny()
    assert _sched(m, None).num_blocks != _sched(m, 4).num_blocks


def test_same_layout_resumes():
    m = _tiny()
    a = _sched(m, 4)
    for _ in range(4):
        a.step()
    sd = a.state_dict()
    assert sd["block_names"] == [n for n, _ in a.blocks] and sd["layers_per_block"] == 4
    b = _sched(m, 4)
    b.load_state_dict(sd)
    assert (b._cycle_pos, b._step_in_block) == (a._cycle_pos, a._step_in_block)


def test_old_checkpoint_into_new_default_fails_loudly_and_names_the_flag():
    m = _tiny()
    _, sd = _old_layout_state(m)
    new = _sched(m, resolve_bcd_layers_per_block(None, "balanced"))
    with pytest.raises(ValueError, match="--bcd-layers-per-block 0"):
        new.load_state_dict(sd)


def test_old_checkpoint_through_the_trainer_restore_path():
    # The trainer must not swallow it (its optimizer restore does catch errors).
    m = _tiny()
    _, sd = _old_layout_state(m)
    new = _sched(m, 4)
    state = {"bcd_state_dict": sd, "optimizer_state_dict": new.optimizer.state_dict()}
    with pytest.raises(ValueError, match="different block layout"):
        _restore_optimizer_and_bcd(new.optimizer, new, state)


def test_old_checkpoint_resumes_with_0():
    m = _tiny()
    old, sd = _old_layout_state(m)
    again = _sched(m, resolve_bcd_layers_per_block(0, "balanced") or None)
    again.load_state_dict(sd)
    assert [n for n, _ in again.blocks] == [n for n, _ in old.blocks]


def test_saved_active_block_out_of_range_names_the_flag():
    # A 13-block checkpoint whose active block is >= 7 fails in the constructor
    # (#169 activates the saved block first), before load_state_dict runs.
    m = _tiny()
    old, _ = _old_layout_state(m)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    with pytest.raises(ValueError, match="--bcd-layers-per-block 0"):
        BlockCoordinateScheduler(m, opt, partition="balanced", layers_per_block=4,
                                 initial_active_idx=old.num_blocks - 1)


def test_same_count_different_layout_fails_when_names_saved():
    # 6 layers: k=4 -> emb | 0-3 | 4,5+tail and k=5 -> emb | 0-4 | 5+tail.
    # Same count (3), different split: only the saved names can tell them apart.
    m = _tiny(6)
    a, b = _sched(m, 4), _sched(m, 5)
    assert a.num_blocks == b.num_blocks
    assert [n for n, _ in a.blocks] != [n for n, _ in b.blocks]
    with pytest.raises(ValueError, match="different block layout"):
        b.load_state_dict(a.state_dict())


def test_legacy_same_count_warns(capsys):
    # Pre-#201 state (no names) where the budget packing and this run's layout
    # have the same count but a different split: it cannot be checked, so it must
    # say so instead of passing quietly. The tiny model's budget packing is one
    # layer per block, so stand in a same-count budget layout (k=5's) directly.
    m = _tiny(6)
    new = _sched(m, 4)
    new._budget_block_names = [n for n, _ in _sched(m, 5).blocks]
    sd = new.state_dict()
    sd.pop("block_names"), sd.pop("layers_per_block")
    new.load_state_dict(sd)
    assert "cannot be checked" in capsys.readouterr().out
    # Control: with names saved, the same resume is silent.
    new.load_state_dict(new.state_dict())
    assert "cannot be checked" not in capsys.readouterr().out
