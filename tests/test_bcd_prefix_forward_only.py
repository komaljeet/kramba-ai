"""Frozen blocks BEFORE the active BCD block must run forward only (#199).

With block j active and grad checkpointing on, the chain rule needs a backward
(and so a checkpoint recompute) through every block AFTER j, but nothing before
j: their params are frozen and their input does not require grad, so autograd
never builds a graph through them. If something ever made the prefix input
require grad (an embedding output marked requires_grad, a frozen param left
trainable, ...), every prefix block would silently pay a recompute + backward
per micro-batch and the loss would not change at all -- only the step time.

The counter is a forward pre-hook that records whether autograd is currently
running a backward (`_current_graph_task_id() != -1`). Positive control: the
blocks after the active one MUST show a recompute, or the counter is vacuous.
`test_counter_catches_forced_prefix_grad` is the falsifier: forcing the bug
must make the prefix show recomputes.

Run:  python -m pytest tests/test_bcd_prefix_forward_only.py -q
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.core.block_coordinate import BlockCoordinateScheduler
from hvn.edge.model import EdgeHVN

N_LAYERS = 5


def _setup(device, active, offload=False, tie=False):
    torch.manual_seed(0)
    model = EdgeHVN(vocab_size=64, d_model=32, num_layers=N_LAYERS, n_heads=4,
                    use_deltanet=True, chunk_size=8, dropout=0.0,
                    grad_checkpoint=True, tie_embeddings=tie).to(device)
    model.train()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    # block index: 0 = token_embedding, 1..N = blocks.0..N-1, N+1 = tail
    sched = BlockCoordinateScheduler(
        model, opt, steps_per_block=10, initial_active_idx=active,
        offload_inactive_weights=offload, device=device if offload else None)
    return model, sched


def _count(model, device, force_prefix_grad=False):
    calls = {}
    hs = []
    for i, b in enumerate(model.blocks):
        def pre(m, args, i=i):
            ph = "bwd" if torch._C._current_graph_task_id() != -1 else "fwd"
            calls.setdefault(i, {"fwd": 0, "bwd": 0})[ph] += 1
        hs.append(b.register_forward_pre_hook(pre))
    if force_prefix_grad:
        hs.append(model.token_embedding.register_forward_hook(
            lambda m, a, o: o.detach().requires_grad_(True)))
    x = torch.randint(0, 64, (2, 16), device=device)
    y = torch.randint(0, 64, (2, 16), device=device)
    if device == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            loss = F.cross_entropy(model(x).float().reshape(-1, 64), y.reshape(-1))
    else:
        loss = F.cross_entropy(model(x).reshape(-1, 64), y.reshape(-1))
    loss.backward()
    for h in hs:
        h.remove()
    return calls


def _check(calls, active_layer):
    """active_layer: index into model.blocks of the active block (-1 = embedding,
    N_LAYERS = tail)."""
    for i in range(N_LAYERS):
        c = calls[i]
        assert c["fwd"] == 1, (i, c)
        want = 0 if i < active_layer else 1
        assert c["bwd"] == want, (
            f"blocks.{i} with active blocks.{active_layer}: {c['bwd']} backward "
            f"recomputes, expected {want} ({'prefix' if want == 0 else 'suffix'})")


def _devices_offload():
    out = [("cpu", False)]
    if torch.cuda.is_available():
        out += [("cuda", False), ("cuda", True)]
    return out


@pytest.mark.parametrize("device,offload", _devices_offload())
@pytest.mark.parametrize("tie", [False, True])
@pytest.mark.parametrize("active", [0, 1, 3, N_LAYERS, N_LAYERS + 1])
def test_prefix_blocks_run_forward_only(device, offload, tie, active):
    model, _ = _setup(device, active, offload=offload, tie=tie)
    calls = _count(model, device)
    _check(calls, active - 1)
    # Positive control is built into _check whenever a suffix exists; make sure
    # this parametrization is not all-prefix or all-suffix by accident.
    if 0 < active - 1 < N_LAYERS - 1:
        assert any(calls[i]["bwd"] for i in range(N_LAYERS))
        assert not all(calls[i]["bwd"] for i in range(N_LAYERS))


@pytest.mark.parametrize("device,offload", _devices_offload())
def test_counter_catches_forced_prefix_grad(device, offload):
    """Falsifier: the exact bug (prefix input requires grad) must be visible."""
    active = 4  # blocks.3
    model, _ = _setup(device, active, offload=offload)
    calls = _count(model, device, force_prefix_grad=True)
    assert all(calls[i]["bwd"] == 1 for i in range(N_LAYERS)), calls
    with pytest.raises(AssertionError):
        _check(calls, active - 1)
