"""Unit tests for BAdam-style block-coordinate-descent training (Issue #99).
No GPU needed — runs on CPU with tiny dims.

Covers:
  1. partition_named_blocks covers every model parameter exactly once.
  2. Only the active block has requires_grad=True at any time.
  3. Optimizer state is genuinely cleared on switch (the actual VRAM-saving
     mechanism — not just requires_grad=False, which alone would leave stale
     state from a block's previous activation resident).
  4. The scheduler advances through all blocks over a full cycle, in order.
  5. Loss decreases over multiple full cycles on a tiny model (correctness
     smoke — the whole model DOES get trained, just not simultaneously).
  6. state_dict / load_state_dict round-trips (resume support).

Run:  python -m pytest tests/test_block_coordinate.py -q
  or:  python tests/test_block_coordinate.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.core.block_coordinate import BlockCoordinateScheduler, partition_named_blocks
from hvn.edge.model import EdgeHVN


def _tiny_model(num_layers=4):
    return EdgeHVN(
        vocab_size=64, d_model=32, num_layers=num_layers, n_heads=4,
        use_deltanet=True, chunk_size=8,
    )


def _tiny_hybrid_model():
    plan = ["deltanet", "transformer", "deltanet", "transformer"]
    return EdgeHVN(
        vocab_size=64, d_model=32, num_layers=len(plan), n_heads=4,
        block_plan=plan, attn_window=16,
    )


# ---------------------------------------------------------------------------
# 1. partition covers every parameter exactly once
# ---------------------------------------------------------------------------

def test_partition_covers_all_params_exactly_once():
    model = _tiny_model(num_layers=4)
    blocks = partition_named_blocks(model)
    # embedding + 4 layers + tail = 6 blocks
    assert len(blocks) == 6, f"expected 6 blocks, got {len(blocks)}"

    all_ids = [id(p) for p in model.parameters()]
    block_ids = [id(p) for _, params in blocks for p in params]
    assert sorted(block_ids) == sorted(all_ids)
    assert len(block_ids) == len(set(block_ids)), "a parameter appeared twice"


def test_partition_works_on_hybrid_block_plan():
    """The DeltaNet/attention hybrid partitions the same way — one block per
    layer regardless of block type."""
    model = _tiny_hybrid_model()
    blocks = partition_named_blocks(model)
    assert len(blocks) == 6  # embedding + 4 layers + tail
    names = [n for n, _ in blocks]
    assert names == [
        "token_embedding", "blocks.0", "blocks.1", "blocks.2", "blocks.3",
        "final_norm+decoder",
    ]


# ---------------------------------------------------------------------------
# 2/3. only the active block has requires_grad + optimizer state
# ---------------------------------------------------------------------------

def test_only_active_block_trainable():
    model = _tiny_model(num_layers=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = BlockCoordinateScheduler(model, opt, steps_per_block=2)

    for i, (name, params) in enumerate(sched.blocks):
        expected = (i == sched._active_idx)
        for p in params:
            assert p.requires_grad == expected, (
                f"block {name!r} requires_grad={p.requires_grad}, expected {expected}"
            )


def test_optimizer_state_cleared_on_switch():
    model = _tiny_model(num_layers=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = BlockCoordinateScheduler(model, opt, steps_per_block=1)

    first_active = sched._active_idx
    _, first_params = sched.blocks[first_active]

    # Do a real step so AdamW actually allocates state for the active block.
    ids = torch.randint(0, 64, (2, 8))
    logits = model(ids)
    loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
    opt.zero_grad()
    loss.backward()
    opt.step()

    for p in first_params:
        assert p in opt.state and len(opt.state[p]) > 0, (
            "AdamW should have allocated state for the active block after step()"
        )

    switched = sched.step()  # steps_per_block=1 -> switches immediately
    assert switched

    # The block that was active is now frozen AND its optimizer state is gone.
    for p in first_params:
        assert not p.requires_grad
        assert p not in opt.state, "stale optimizer state was not cleared on switch"
        assert p.grad is None, "stale .grad was not cleared on switch"

    # The new active block has requires_grad=True and (not yet stepped) no state.
    new_active = sched._active_idx
    assert new_active != first_active
    _, new_params = sched.blocks[new_active]
    for p in new_params:
        assert p.requires_grad
        assert p not in opt.state  # fresh -- hasn't been stepped yet


# ---------------------------------------------------------------------------
# 4. full cycle visits every block in order
# ---------------------------------------------------------------------------

def test_full_cycle_visits_every_block_in_order():
    model = _tiny_model(num_layers=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = BlockCoordinateScheduler(model, opt, steps_per_block=2, order="sequential")

    n = sched.num_blocks
    visited = [sched.active_block_name]
    for _ in range(n * 2):  # 2 steps per block
        sched.step()
        visited.append(sched.active_block_name)

    expected_names = [name for name, _ in sched.blocks]
    # after exactly n*2 step() calls we should have completed one full cycle
    # and returned to the first block
    assert visited[0] == expected_names[0]
    assert visited[-1] == expected_names[0], "did not wrap back to the first block"
    # the sequence of *switches* (deduped consecutive) should equal one full
    # cycle through every block, in order, then back to the start
    deduped = [visited[0]]
    for v in visited[1:]:
        if v != deduped[-1]:
            deduped.append(v)
    assert deduped == expected_names + [expected_names[0]]


# ---------------------------------------------------------------------------
# 5. correctness smoke — loss decreases over multiple full cycles
# ---------------------------------------------------------------------------

def test_loss_decreases_over_multiple_cycles():
    torch.manual_seed(0)
    model = _tiny_model(num_layers=3)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
    sched = BlockCoordinateScheduler(model, opt, steps_per_block=5)

    torch.manual_seed(1)
    ids = torch.randint(0, 64, (4, 16))

    losses = []
    n_steps = sched.num_blocks * 5 * 3  # 3 full cycles
    for _ in range(n_steps):
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        losses.append(loss.item())

    early_mean = sum(losses[:10]) / 10
    late_mean = sum(losses[-10:]) / 10
    assert late_mean < early_mean, (
        f"loss did not decrease over 3 full cycles: early={early_mean:.4f} "
        f"late={late_mean:.4f}"
    )


# ---------------------------------------------------------------------------
# 6. state_dict round-trip
# ---------------------------------------------------------------------------

def test_state_dict_roundtrip():
    model = _tiny_model(num_layers=3)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    sched = BlockCoordinateScheduler(model, opt, steps_per_block=2)

    sched.step()
    sched.step()
    sched.step()  # now 1 step into the 2nd block of a fresh cycle
    sd = sched.state_dict()
    active_before = sched.active_block_name

    model2 = _tiny_model(num_layers=3)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    sched2 = BlockCoordinateScheduler(model2, opt2, steps_per_block=2)
    sched2.load_state_dict(sd)

    assert sched2.active_block_name == active_before
    assert sched2._step_in_block == sched._step_in_block
    assert sched2._cycle_pos == sched._cycle_pos


# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
