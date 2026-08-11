"""Block-coordinate descent must work with tied embeddings (Issue #151).

Author: Rambarun Komaljeet

`--tie-embeddings` has defaulted ON since #139, and #112 records
`--block-coordinate` as **mandatory at 355M**. Those two documented defaults were
mutually incompatible: the tied table is one `nn.Parameter` under two names, so
the BCD partition placed it in both `token_embedding` and `final_norm+decoder`
and `_validate_partition` raised. A 355M BCD run — the exact scale-up config —
died at scheduler construction.

The bug predates #151's default flip; the flip only surfaced it. These tests
pin the fix and, more importantly, pin the *semantics*: a shared parameter
belongs to exactly one block, so it trains only while that block is active. The
alternative (updating it under both) would give the tied table twice the update
frequency of every other parameter — a silent, hard-to-see training bug, which
is why it is asserted rather than left to the partition's discretion.
"""

import pytest
import torch

from hvn.core.block_coordinate import (
    BlockCoordinateScheduler, partition_named_blocks as partition_blocks,
)
from hvn.edge.model import EdgeHVN


def _model(tie: bool):
    torch.manual_seed(0)
    return EdgeHVN(vocab_size=256, d_model=64, num_layers=4, n_heads=4,
                   use_deltanet=True, dropout=0.0, tie_embeddings=tie)


@pytest.mark.parametrize("tie", [False, True])
def test_partition_accepts_both_tyings(tie):
    """The regression: tie=True raised ValueError before #151."""
    partition_blocks(_model(tie))


@pytest.mark.parametrize("tie", [False, True])
def test_scheduler_constructs(tie):
    m = _model(tie)
    BlockCoordinateScheduler(m, torch.optim.AdamW(m.parameters(), lr=1e-4))


def test_tied_table_appears_in_exactly_one_block():
    """The semantic guarantee, not just 'it does not crash'.

    A shared weight in two blocks would be stepped twice per cycle — more
    updates than any other parameter, and invisible in a loss curve.
    """
    m = _model(True)
    blocks = partition_blocks(m)
    shared = m.token_embedding.weight
    owners = [name for name, ps in blocks if any(p is shared for p in ps)]
    assert owners == ["token_embedding"], owners


def test_partition_still_covers_every_parameter_exactly_once():
    """Dedup must not silently drop a parameter from training entirely."""
    for tie in (False, True):
        m = _model(tie)
        blocks = partition_blocks(m)
        ids = [id(p) for _n, ps in blocks for p in ps]
        assert len(ids) == len(set(ids)), "a parameter is in two blocks"
        assert set(ids) == {id(p) for p in m.parameters()}, \
            "partition does not cover model.parameters()"


def test_untied_partition_is_unchanged_by_the_fix():
    """Known-answer: the dedup must be a no-op when nothing is shared."""
    m = _model(False)
    blocks = partition_blocks(m)
    names = [n for n, _ in blocks]
    assert names == ["token_embedding", "blocks.0", "blocks.1", "blocks.2",
                     "blocks.3", "final_norm+decoder"]
    assert all(len(ps) > 0 for _n, ps in blocks)


def test_offload_with_tied_embeddings_is_rejected_loudly():
    """Unsupported combinations must fail at construction, not mid-run.

    Before #151 this surfaced as a bare `KeyError: <id>` inside the prefetch
    CUDA stream -- far from the cause, and in the subsystem #117 caught silently
    discarding all training. A clear refusal beats an improvised fix here.
    """
    m = _model(True)
    m.grad_checkpoint = True
    with pytest.raises(ValueError, match="not supported with tie_embeddings"):
        BlockCoordinateScheduler(
            m, torch.optim.AdamW(m.parameters(), lr=1e-4),
            offload_inactive_weights=True, device=torch.device("cpu"))


def test_offload_still_allowed_untied():
    """The guard must not block the configuration offload was built for."""
    m = _model(False)
    m.grad_checkpoint = True
    BlockCoordinateScheduler(
        m, torch.optim.AdamW(m.parameters(), lr=1e-4),
        offload_inactive_weights=True, device=torch.device("cpu"))
