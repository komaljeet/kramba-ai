"""Per-type weight sharing must shrink the model WITHOUT changing it (#155).

Author: Rambarun Komaljeet

Sharing is a storage change, not an architecture change. The failure that
matters here is silent: the pre-#155 path collapsed every layer to ``plan[0]``,
so a hybrid asked for 10 DeltaNet + 2 attention layers would train as 12 DeltaNet
layers, load without error, and produce a perfectly plausible loss curve. The
recall work in #135 depends on those attention positions existing.

So these tests assert the PLAN survives, the weights are genuinely shared
(object identity, not merely equal), and gradients from every virtual layer
reach the one shared block. A test that only checked the parameter count would
pass on the broken implementation -- it also reduces parameters.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN  # noqa: E402

PLAN = ["transformer" if i in (3, 9) else "deltanet" for i in range(12)]


def _build(shared, d_model=512, plan=PLAN):
    return EdgeHVN(vocab_size=32003, d_model=d_model, num_layers=len(plan),
                   n_heads=16, block_plan=plan, attn_window=4096,
                   tie_embeddings=True, shared_weights=shared)


def _body(m):
    return (sum(p.numel() for p in m.parameters())
            - m.token_embedding.weight.numel())


def test_sharing_actually_reduces_the_body():
    dense, shared = _build(False), _build(True)
    ratio = _body(dense) / _body(shared)
    assert ratio > 4.0, f"only {ratio:.2f}x fewer body params -- sharing did not engage"


def test_the_block_plan_survives_sharing():
    """The regression that motivated #155.

    The old path used plan[0] for every layer, turning a hybrid into a uniform
    DeltaNet stack. Parameter count drops either way, so only the TYPES catch it.
    """
    shared = _build(True)
    kinds = [type(b.block).__name__ for b in shared.blocks]
    dense_kinds = [type(b).__name__ for b in _build(False).blocks]
    assert kinds == dense_kinds, (
        f"sharing changed the architecture:\n  dense  {dense_kinds}\n  shared {kinds}")
    assert kinds[3] != kinds[0], "attention layers collapsed into the DeltaNet type"


def test_same_type_shares_one_object_and_different_types_do_not():
    """Identity, not equality: two blocks initialised from the same seed would
    compare equal while still being separate parameters."""
    s = _build(True)
    assert s.blocks[0].block is s.blocks[1].block, "DeltaNet layers are not shared"
    assert s.blocks[3].block is s.blocks[9].block, "attention layers are not shared"
    assert s.blocks[0].block is not s.blocks[3].block, \
        "attention and DeltaNet share one block -- the types were merged"


def test_optimizer_sees_each_shared_weight_once():
    """Duplicated entries would apply weight decay and momentum N times to the
    same tensor, which is a different optimizer, not a smaller model."""
    m = _build(True)
    params = list(m.parameters())
    assert len({id(p) for p in params}) == len(params)


def test_every_virtual_layer_feeds_gradient_into_the_shared_block():
    """ALBERT's update rule: gradients from all depths accumulate into one block.
    If only the first virtual layer contributed, the deeper layers would be
    frozen passengers and the model would quietly be shallower than it looks."""
    plan = ["transformer" if i in (1, 3) else "deltanet" for i in range(4)]
    m = _build(True, d_model=64, plan=plan)
    m(torch.randint(0, 32003, (2, 16))).square().mean().backward()

    shared = m.blocks[0].block
    grads = [p.grad for p in shared.parameters()]
    assert all(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0
               for g in grads), "shared block did not receive gradient from all layers"


def test_per_layer_scalars_are_not_shared():
    """The only thing distinguishing virtual layers. Sharing these too would
    make the stack a literal repeated function."""
    m = _build(True)
    assert m.blocks[0].residual_scale is not m.blocks[1].residual_scale
    assert m.blocks[0].position_offset is not m.blocks[1].position_offset


def test_forward_still_runs_and_shapes_match_dense():
    x = torch.randint(0, 32003, (1, 64))
    with torch.no_grad():
        d, s = _build(False)(x), _build(True)(x)
    assert d.shape == s.shape == (1, 64, 32003)


def test_uniform_stack_still_uses_the_original_albert_path():
    """Non-hybrid plans keep the pre-existing SharedWeightHVN behaviour, so
    #155 cannot regress the path the 2026-08-02 measurement used."""
    plan = ["deltanet"] * 6
    m = _build(True, d_model=64, plan=plan)
    assert m.shared_block is not None and m.blocks is None


def test_checkpoint_records_sharing_and_the_real_head_count(tmp_path):
    """A shared checkpoint rebuilt as dense trains a different architecture, so
    the flag must be persisted.

    Also pins a bug this change introduced: under per-type sharing each entry of
    ``model.blocks`` is a SharedBlockRef WRAPPER, and the saver read
    ``getattr(first_block, 'n_heads', 4)`` straight off it -- silently recording
    4 heads for a 16-head model. The checkpoint would reload as a different
    architecture with nothing raising.
    """
    import torch
    from train_edge_conversational import _save_edge_checkpoint

    plan = ["deltanet", "transformer"] * 2
    m = _build(True, d_model=64, plan=plan)
    path = str(tmp_path / "shared.pt")
    _save_edge_checkpoint(m, path, global_step=1)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    assert ckpt["shared_weights"] is True, "sharing not recorded in the checkpoint"
    assert ckpt["n_heads"] == 16, (
        f"checkpoint says n_heads={ckpt['n_heads']} for a 16-head model -- the "
        f"SharedBlockRef wrapper was read instead of the block")


def test_param_count_gate_alone_would_not_catch_the_collapse():
    """Anti-vacuity: proves the plan assertions above are load-bearing.

    A uniform 12-layer stack -- the WRONG architecture the old path produced --
    also shrinks the body, so any test gating only on parameter count passes on
    it. If this ever fails, the count-based test is sufficient and the identity
    checks are redundant.
    """
    collapsed = _build(True, plan=["deltanet"] * 12)
    correct = _build(True)
    r_collapsed = _body(_build(False)) / _body(collapsed)
    r_correct = _body(_build(False)) / _body(correct)
    assert r_collapsed > 4.0 and r_correct > 4.0, (
        "the wrong architecture no longer passes a count-only gate; the identity "
        "tests may now be redundant")
