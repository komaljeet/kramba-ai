"""--bcd-layers-per-block (#194): exact layer count per BCD block, tail merged.

CPU only, tiny dims. If the option silently did nothing, the block count would equal the
default and the first test would fail; if the tail were left alone, the last block would
be a LayerNorm-only turn and the second would fail.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402
from train_edge_conversational import resolve_bcd_partition  # noqa: E402,F401


def _model(num_layers=7):
    return EdgeHVN(vocab_size=512, d_model=32, num_layers=num_layers, n_heads=4,
                   block_plan=["deltanet"] * num_layers, tie_embeddings=True)


def _sched(m, k):
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    return BlockCoordinateScheduler(m, opt, steps_per_block=1, partition="balanced",
                                    layers_per_block=k)


@pytest.mark.parametrize("k", [1, 2, 3, 4])
def test_exact_layer_count_embedding_alone_tail_merged(k):
    m = _model(7)
    s = _sched(m, k)
    members = s._block_members
    assert members[0] == ("token_embedding",)
    layer_groups = members[1:]
    # Layers per group: k each, the last one takes the remainder (7 layers).
    counts = [sum(n.startswith("blocks.") for n in g) for g in layer_groups]
    assert counts == [k] * (7 // k) + ([7 % k] if 7 % k else []), counts
    # The tail rides with the last group, never alone.
    assert members[-1][-1] == "final_norm+decoder" and counts[-1] >= 1
    assert all("final_norm+decoder" not in g for g in members[:-1])
    # Every parameter covered once.
    ids = [id(p) for _, ps in s.blocks for p in ps]
    assert sorted(ids) == sorted(id(p) for p in m.parameters())


def test_default_is_unchanged_without_the_option():
    m = _model(7)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    a = BlockCoordinateScheduler(m, opt, steps_per_block=1, partition="balanced")
    b = BlockCoordinateScheduler(m, opt, steps_per_block=1, partition="balanced",
                                 layers_per_block=None)
    assert [n for n, _ in a.blocks] == [n for n, _ in b.blocks]


def test_every_layer_of_a_group_trains_over_a_rotation():
    torch.manual_seed(0)
    m = _model(7)
    s = _sched(m, 3)
    x = torch.randint(0, 512, (2, 16))
    trained = set()
    for _ in range(s.num_blocks):
        before = {id(p): p.detach().clone() for p in m.parameters()}
        logits = m(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x.reshape(-1))
        s.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        s.optimizer.step()
        for i, blk in enumerate(m.blocks):
            if any(not torch.equal(p.detach(), before[id(p)]) for p in blk.parameters()):
                trained.add(i)
        s.step()
    assert trained == set(range(7)), trained


def test_rejects_option_under_layer_partition():
    # "layer" used to ignore it silently: an arm asking for 4 would train 1.
    m = _model(7)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    with pytest.raises(ValueError):
        BlockCoordinateScheduler(m, opt, steps_per_block=1, partition="layer",
                                 layers_per_block=2)


def test_rejects_unordered_blocks():
    from hvn.core.block_coordinate import _layer_count_groups
    with pytest.raises(ValueError):
        _layer_count_groups([("a", []), ("b", [])], 2)
