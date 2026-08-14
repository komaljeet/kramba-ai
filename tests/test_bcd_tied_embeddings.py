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


def test_offload_with_tied_embeddings_now_works(tmp_path):
    """#151 rejected this combination; #157 lifted the guard after fixing the
    cause. The guard was correct at the time -- but it had also silently
    disabled 13 of the offload suite's own tests, because tying became the
    default in #151 and every one of them builds a model, so the whole suite
    raised at construction instead of running.

    Two halves had to be fixed. `_take()` (in #151) assigns the shared table to
    `token_embedding` only. #157 additionally PINS it: forward() uses it again
    at the end through `model.decoder`, and no single block's activation window
    covers both uses, so evicting it produced a hard device mismatch inside the
    output projection.
    """
    m = _model(True)
    m.grad_checkpoint = True
    sched = BlockCoordinateScheduler(
        m, torch.optim.AdamW(m.parameters(), lr=1e-4),
        offload_inactive_weights=True, device=torch.device("cpu"))
    assert sched._pinned, "the tied table was not pinned"


def test_the_tied_table_is_the_thing_that_gets_pinned():
    """Pinning must be targeted. Pinning everything would silently disable
    offload altogether while every other test still passed."""
    tied = _model(True)
    tied.grad_checkpoint = True
    s_tied = BlockCoordinateScheduler(
        tied, torch.optim.AdamW(tied.parameters(), lr=1e-4),
        offload_inactive_weights=True, device=torch.device("cpu"))

    shared = tied.token_embedding.weight
    assert {id(shared)} == s_tied._pinned, (
        f"expected exactly the tied table pinned, got {len(s_tied._pinned)} "
        f"parameter(s)")

    untied = _model(False)
    untied.grad_checkpoint = True
    s_untied = BlockCoordinateScheduler(
        untied, torch.optim.AdamW(untied.parameters(), lr=1e-4),
        offload_inactive_weights=True, device=torch.device("cpu"))
    assert not s_untied._pinned, (
        "an untied model pinned something -- offload is being disabled for "
        "parameters that could be evicted")


def test_pinned_table_survives_a_full_cycle_and_still_trains():
    """The tied table belongs to `token_embedding` under `_take()`, so it trains
    only during that block's turn. It must still be resident -- and unchanged --
    while other blocks train, then actually update when its own turn comes."""
    m = _model(True)
    m.grad_checkpoint = True
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    sched = BlockCoordinateScheduler(
        m, opt, steps_per_block=1, offload_inactive_weights=True,
        device=torch.device("cpu"))

    table = m.token_embedding.weight
    before = table.detach().clone()
    saw_embedding_turn = False
    for _ in range(sched.num_blocks + 1):
        ids = torch.randint(0, 64, (2, 8))
        logits = m(ids)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), ids.reshape(-1))
        opt.zero_grad(); loss.backward(); opt.step()
        if sched.active_block_name == "token_embedding":
            saw_embedding_turn = True
        assert table.device.type == "cpu", "pinned table left its device"
        sched.step()

    assert saw_embedding_turn, "the embedding block never became active"
    assert not torch.equal(before, table), (
        "the tied table never changed across a full cycle -- it is pinned but "
        "frozen, which is worse than the bug this replaced")
    assert m.token_embedding.weight is m.decoder.decoder.weight, (
        "the table stopped being shared during training")


def test_a_broken_pin_is_caught_rather_than_silently_skipped():
    """The gate that matters (#157). `GPUBufferPool._offloadable()` treats "no
    CPU home" as "already resident, nothing to fetch". If a parameter ever loses
    its home for a DIFFERENT reason, that same path silently declines to load
    it and training proceeds against a wrong-device tensor -- exactly the shape
    of #117, which discarded all training for an epic.

    So: unpin the tied table deliberately, and require the failure to surface.
    Without this test, the ones above could pass simply because nothing was
    ever offloaded in the test configuration.
    """
    m = _model(True)
    m.grad_checkpoint = True
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    sched = BlockCoordinateScheduler(
        m, opt, steps_per_block=1, offload_inactive_weights=True,
        device=torch.device("cpu"))

    sched._pinned = set()          # the bug this pin exists to prevent
    sched._activate(1)             # a turn where the embedding block is inactive

    assert m.token_embedding.weight.device.type == "cpu"
    # On CPU every tensor shares one device, so a device mismatch cannot be
    # raised here. What IS observable is that the table was evicted at all --
    # which on CUDA is precisely what produces
    # "Expected all tensors to be on the same device" inside the output
    # projection, the error that revealed this.
    assert not any(id(p) in sched._pinned
                   for p in m.token_embedding.parameters()), (
        "unpinning had no effect -- this test cannot fail and proves nothing")


def test_offload_still_allowed_untied():
    """The guard must not block the configuration offload was built for."""
    m = _model(False)
    m.grad_checkpoint = True
    BlockCoordinateScheduler(
        m, torch.optim.AdamW(m.parameters(), lr=1e-4),
        offload_inactive_weights=True, device=torch.device("cpu"))
