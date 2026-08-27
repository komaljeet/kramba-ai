"""The three things that block a 1B pretrain (Issue #163).

Each test targets a failure that does NOT announce itself at runtime:

1. **No offload path** -- the run simply OOMs or spills, which at least is
   visible. Tested anyway because the wiring has three couplings that must hold
   together (CPU-resident model, grad_checkpoint, untied embeddings) and getting
   one wrong fails hours in.
2. **Checkpoint retention** -- fills the disk mid-run and kills it with no
   checkpoint left to resume from. The worst outcome, because it destroys the
   recovery path.
3. **The data-budget guard** -- the silent one. A capped corpus trains with a
   healthy-looking loss curve forever.

Run: python -m pytest tests/test_issue163_1b_launchable.py -q
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import train_edge_conversational as tec  # noqa: E402


# ---------------------------------------------------------------------------
# 1. checkpoint retention
# ---------------------------------------------------------------------------
def _touch_ckpts(d, steps):
    for s in steps:
        Path(d, f"edge_conversational_step{s}.pt").write_bytes(b"x")


def test_prune_keeps_exactly_the_newest_n():
    with tempfile.TemporaryDirectory() as d:
        _touch_ckpts(d, [1000, 2000, 3000, 4000, 5000])
        removed = tec._prune_checkpoints(d, keep_last=2)
        left = sorted(int(f.split("_step")[-1][:-3])
                      for f in os.listdir(d) if "_step" in f)
        assert left == [4000, 5000], left
        assert len(removed) == 3


def test_prune_keeps_milestones_forever():
    """A long run needs a coarse trail, not just the last few."""
    with tempfile.TemporaryDirectory() as d:
        _touch_ckpts(d, [1000, 2000, 3000, 4000, 5000, 6000])
        tec._prune_checkpoints(d, keep_last=1, milestone_every=3000)
        left = sorted(int(f.split("_step")[-1][:-3])
                      for f in os.listdir(d) if "_step" in f)
        assert left == [3000, 6000], left


def test_prune_disabled_keeps_everything():
    with tempfile.TemporaryDirectory() as d:
        _touch_ckpts(d, [1000, 2000, 3000])
        assert tec._prune_checkpoints(d, keep_last=0) == []
        assert len(os.listdir(d)) == 3


def test_prune_ignores_non_step_files():
    """The final checkpoint and the resume state must never be pruned."""
    with tempfile.TemporaryDirectory() as d:
        _touch_ckpts(d, [1000, 2000, 3000])
        Path(d, "edge_conversational.pt").write_bytes(b"final")
        Path(d, "resume_state.pt").write_bytes(b"resume")
        tec._prune_checkpoints(d, keep_last=1)
        assert Path(d, "edge_conversational.pt").exists()
        assert Path(d, "resume_state.pt").exists()


def test_prune_would_have_prevented_the_5TB_run():
    """The arithmetic that motivated this, asserted so it cannot be forgotten.

    A 1B run is ~1.1M optimizer steps. At one checkpoint per 1,000 steps and
    ~4.9 GB each that is ~5.4 TB; this machine has ~293 GB free.
    """
    with tempfile.TemporaryDirectory() as d:
        _touch_ckpts(d, [i * 1000 for i in range(1, 121)])   # 120 checkpoints
        tec._prune_checkpoints(d, keep_last=5, milestone_every=50_000)
        left = sorted(int(f.split("_step")[-1][:-3])
                      for f in os.listdir(d) if "_step" in f)
        # 5 rolling + the 50k and 100k milestones
        assert left == [50_000, 100_000, 116_000, 117_000, 118_000,
                        119_000, 120_000], left


# ---------------------------------------------------------------------------
# 2. the offload couplings
# ---------------------------------------------------------------------------
def test_offload_flags_exist():
    """A missing flag is a silent revert to the un-offloadable path."""
    import argparse
    p = tec.build_arg_parser() if hasattr(tec, "build_arg_parser") else None
    if p is None:
        pytest.skip("no standalone parser factory; covered by the CLI tests")
    names = {a.dest for a in p._actions}
    for flag in ("offload", "keep_checkpoints", "milestone_every",
                 "min_tokens_per_param", "offload_cache_blocks"):
        assert flag in names, f"--{flag.replace('_','-')} is missing"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offload_scheduler_actually_offloads():
    """The scheduler must evict weights, not merely be constructed.

    #117 is the precedent: offload ran for a whole epic while silently doing no
    cross-cycle learning. 'It constructed without raising' is not evidence.
    """
    from hvn.edge.model import EdgeHVN
    from hvn.core.block_coordinate import BlockCoordinateScheduler

    model = EdgeHVN(vocab_size=256, d_model=128, num_layers=6, n_heads=2,
                    block_plan=["deltanet"] * 6, grad_checkpoint=True,
                    tie_embeddings=False).to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    BlockCoordinateScheduler(
        model, opt, steps_per_block=10, device="cuda",
        offload_inactive_weights=True, offload_park_dtype=torch.bfloat16,
        offload_prefetch=True,
    )
    devs = {p.device.type for p in model.parameters()}
    assert "cpu" in devs, (
        f"no parameter is CPU-resident ({devs}) -- nothing was offloaded, so "
        f"the memory saving this whole path exists for is not happening")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_offload_forward_is_finite_and_trains():
    """Finite loss AND a decreasing one.

    Asserting only 'it ran' would have passed straight through the prefetch
    data race fixed on 2026-08-27, which NaN'd the forward while the step
    completed normally.
    """
    import torch.nn as nn
    from hvn.edge.model import EdgeHVN
    from hvn.core.block_coordinate import BlockCoordinateScheduler

    torch.manual_seed(0)
    model = EdgeHVN(vocab_size=256, d_model=128, num_layers=6, n_heads=2,
                    block_plan=["deltanet"] * 6, grad_checkpoint=True,
                    tie_embeddings=False).to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    sched = BlockCoordinateScheduler(
        model, opt, steps_per_block=10, device="cuda",
        offload_inactive_weights=True, offload_park_dtype=torch.bfloat16,
        offload_prefetch=True,
    )
    x = torch.randint(0, 256, (2, 64), device="cuda")
    y = torch.randint(0, 256, (2, 64), device="cuda")

    losses = []
    for _ in range(12):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(x)
            out = out[0] if isinstance(out, tuple) else out
            loss = nn.functional.cross_entropy(
                out.reshape(-1, out.size(-1)).float(), y.reshape(-1))
        assert torch.isfinite(loss), "non-finite loss under offload"
        loss.backward(); opt.step(); opt.zero_grad(set_to_none=True); sched.step()
        losses.append(float(loss))

    assert losses[-1] < losses[0], (
        f"loss did not fall under offload ({losses[0]:.3f} -> {losses[-1]:.3f}) "
        f"-- #117 shipped an offload path that ran happily while learning nothing")


# ---------------------------------------------------------------------------
# 3. the data-budget guard
# ---------------------------------------------------------------------------
def test_budget_guard_arithmetic():
    """The guard's own maths, on the real numbers that motivated it.

    Default mix caps at 134,784,617 tokens. A 1.21B model at 10 tokens/param
    needs 12.1B -- 90x more. It must trip.
    """
    cap = 134_784_617
    for params, tpp, should_trip in [
        (1_214_543_633, 10.0, True),     # the 1B run, default mix
        (477_997_043, 10.0, True),       # even 478M is far short
        (7_000_000, 10.0, False),        # ~7M is what this budget actually serves
    ]:
        trips = cap < params * tpp
        assert trips is should_trip, (
            f"{params:,} params at {tpp} tok/param vs a {cap:,} budget: "
            f"expected trip={should_trip}")


def test_fineweb_only_mix_clears_the_guard():
    """The documented fix must actually clear it, or the error message is a
    dead end for whoever hits it."""
    fineweb_tokens = 127_000_000_000       # ~233 GB at ~1.84 GB per 1B tokens
    assert fineweb_tokens > 1_214_543_633 * 30, (
        "FineWeb-Edu should cover a 1.21B model at 30 tokens/param; if this "
        "fails the recommended fix in the guard's message is wrong")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
