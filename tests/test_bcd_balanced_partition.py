"""BCD balanced partition: layers packed up to the largest block's size.

No GPU needed -- tiny dims on CPU, same convention as test_block_coordinate.py.

The claim the feature makes is "each layer trains more often, peak residency
unchanged". These tests pin both halves and the ways it could silently break:
  1. every parameter is covered exactly once, groups are contiguous in depth,
     and no group exceeds the largest single block (the peak is unchanged);
  2. the scheduler really trains the WHOLE active group (not just its first
     layer) and nothing outside it;
  3. the per-layer partition is still the default and is byte-identical;
  4. combining it with CPU offload is refused, not silently wrong;
  5. resuming a run saved under the other partition is refused.
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
    BlockCoordinateScheduler, balance_blocks, partition_named_blocks,
)
from hvn.edge.model import EdgeHVN  # noqa: E402


def _model(num_layers=6, vocab=512, d=32, tie=True):
    # vocab 512 x d 32 = 16,384 table vs a ~13K-param layer: a table that is
    # larger than one layer, like the real GPT-2-vocab shapes.
    return EdgeHVN(vocab_size=vocab, d_model=d, num_layers=num_layers, n_heads=4,
                   block_plan=["deltanet"] * num_layers, tie_embeddings=tie)


def _size(params):
    return sum(p.numel() for p in params)


def test_balanced_covers_all_params_and_respects_budget():
    m = _model()
    layer = partition_named_blocks(m)
    bal = balance_blocks(layer)
    budget = max(_size(p) for _, p in layer)

    ids = [id(p) for _, ps in bal for p in ps]
    assert sorted(ids) == sorted(id(p) for p in m.parameters())
    assert len(ids) == len(set(ids))
    assert all(_size(ps) <= budget for _, ps in bal), "a group exceeds the peak block"
    # Fewer blocks than per-layer, otherwise the test shape proves nothing.
    assert len(bal) < len(layer), (len(bal), len(layer))
    # Contiguous and order-preserving: concatenated names == per-layer order.
    # ("final_norm+decoder" already contains '+', so compare the joined form.)
    assert "+".join(n for n, _ in bal) == "+".join(n for n, _ in layer)


def test_scheduler_trains_whole_active_group_only():
    torch.manual_seed(0)
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    sched = BlockCoordinateScheduler(m, opt, steps_per_block=1, partition="balanced")
    x = torch.randint(0, 512, (2, 16))
    for _ in range(sched.num_blocks):
        name, active = sched.blocks[sched._active_idx]
        before = {id(p): p.detach().clone() for p in m.parameters()}
        logits = m(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), x.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        active_ids = {id(p) for p in active}
        for p in m.parameters():
            moved = not torch.equal(p.detach(), before[id(p)])
            if id(p) not in active_ids:
                assert not moved, f"param outside active group {name!r} moved"
        # Every layer in a merged group must actually have been updated.
        for sub in name.split("+"):
            if sub.startswith("blocks."):
                blk = m.blocks[int(sub.split(".")[1])]
                assert any(not torch.equal(p.detach(), before[id(p)])
                           for p in blk.parameters()), f"{sub} in {name!r} never trained"
        sched.step()


def test_layer_partition_is_default_and_unchanged():
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sched = BlockCoordinateScheduler(m, opt, steps_per_block=5)
    assert [n for n, _ in sched.blocks] == [n for n, _ in partition_named_blocks(m)]


def test_balanced_with_offload_is_refused():
    m = _model()
    m.grad_checkpoint = True
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    with pytest.raises(ValueError, match="offload"):
        BlockCoordinateScheduler(m, opt, partition="balanced",
                                 offload_inactive_weights=True, device="cpu")


def test_resume_across_partitions_is_refused():
    m = _model()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3)
    sd = BlockCoordinateScheduler(m, opt, partition="layer").state_dict()
    other = BlockCoordinateScheduler(m, opt, partition="balanced")
    with pytest.raises(ValueError, match="partition"):
        other.load_state_dict(sd)
