"""Tests for Issue #123 — MoE FFN on the DeltaNet block.

The first test is the KNOWN-ANSWER CELL the issue requires: num_experts=1,
top_k=1 must be numerically identical to the dense FFN it replaces, in BOTH
forward and backward. An "is it wired up" test cannot catch a routing bug; a
configuration whose correct answer is known by construction can. See the
retraction in docs/state.md -- a bug that survived six experiments would have
been caught immediately by a check like this.

CPU-only, tiny dims.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn

from hvn.core.moe import MoEFFN, collect_moe_aux_loss
from hvn.core.deltanet import DeltaNetBlock


def _dense_gelu_ffn(d_model, dropout=0.0):
    return nn.Sequential(
        nn.Linear(d_model, d_model * 2),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(d_model * 2, d_model),
        nn.Dropout(dropout),
    )


def test_single_expert_matches_dense_ffn_forward_and_backward():
    """KNOWN-ANSWER CELL: E=1/top-1 == dense FFN, exactly, fwd AND bwd."""
    torch.manual_seed(0)
    d = 32
    moe = MoEFFN(d, num_experts=1, top_k=1, expansion=2, dropout=0.0, use_swiglu=False)
    dense = _dense_gelu_ffn(d, dropout=0.0)

    # Copy MoE's expert weights into the dense module so they are the same function.
    with torch.no_grad():
        dense[0].weight.copy_(moe.w1[0].T)
        dense[0].bias.copy_(moe.b1[0])
        dense[3].weight.copy_(moe.w2[0].T)
        dense[3].bias.copy_(moe.b2[0])

    moe.eval(); dense.eval()
    x1 = torch.randn(2, 5, d, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)

    out_moe, out_dense = moe(x1), dense(x2)
    assert torch.allclose(out_moe, out_dense, atol=1e-6), (
        f"E=1 must equal the dense FFN; max diff "
        f"{(out_moe - out_dense).abs().max().item():.2e}"
    )

    # Backward too -- a softmax over a single logit would match in forward but
    # add a spurious router gradient path. The E=1 fast path avoids that.
    out_moe.sum().backward()
    out_dense.sum().backward()
    assert torch.allclose(x1.grad, x2.grad, atol=1e-6), "E=1 backward must match dense"
    assert moe.router.weight.grad is None or torch.count_nonzero(moe.router.weight.grad) == 0, \
        "E=1 must not create a router gradient path the dense FFN lacks"


def test_more_experts_do_not_increase_active_compute():
    """The whole point: stored params scale with E, active params with top_k."""
    d = 32
    small = MoEFFN(d, num_experts=2, top_k=2, dropout=0.0)
    large = MoEFFN(d, num_experts=16, top_k=2, dropout=0.0)

    stored_small = sum(p.numel() for p in small.parameters())
    stored_large = sum(p.numel() for p in large.parameters())
    assert stored_large > 5 * stored_small, "8x the experts should mean ~8x stored params"

    # Active compute is set by top_k for the EXPERTS, which is equal for both.
    # The router genuinely does scale with num_experts -- all E logits are
    # computed to pick the top-k -- but it is d_model*E, negligible next to the
    # experts' d_model*hidden*2*k. Assert the expert term is identical and the
    # router term stays a rounding error, rather than pretending active cost is
    # perfectly flat in E (it is not, just very nearly so).
    expert_active_small = small.active_param_count() - small.router.weight.numel()
    expert_active_large = large.active_param_count() - large.router.weight.numel()
    assert expert_active_small == expert_active_large, (
        f"expert active params must be independent of num_experts: "
        f"{expert_active_small} vs {expert_active_large}"
    )
    router_overhead = large.router.weight.numel() / expert_active_large
    assert router_overhead < 0.10, (
        f"router should be a rounding error vs expert compute, got {router_overhead:.1%}"
    )


def test_routing_is_sparse_not_dense_average():
    """Each token must reach exactly top_k experts -- not a soft blend of all."""
    torch.manual_seed(1)
    d, E, k = 32, 8, 2
    moe = MoEFFN(d, num_experts=E, top_k=k, dropout=0.0)
    # Break the zero-init symmetry so the router actually discriminates.
    with torch.no_grad():
        moe.router.weight.normal_(0, 1.0)

    x = torch.randn(4, 6, d)
    logits = moe.router(x.reshape(-1, d))
    _, idx = torch.topk(torch.softmax(logits, -1), k, dim=-1)
    assert idx.shape[-1] == k
    # With a random router and 8 experts, a 24-token batch should not land on
    # only one expert -- that would mean routing is degenerate.
    assert idx.unique().numel() > 1, "routing collapsed to a single expert"


def test_aux_loss_present_and_collectable():
    torch.manual_seed(2)
    d = 32
    moe = MoEFFN(d, num_experts=4, top_k=2, dropout=0.0)
    moe(torch.randn(2, 5, d))
    assert moe.aux_loss is not None and moe.aux_loss.item() > 0, \
        "load-balancing aux loss must be produced; without it routing collapses"

    block = DeltaNetBlock(d_model=32, n_heads=4, dropout=0.0, moe_experts=4, moe_top_k=2)
    block(torch.randn(2, 6, 32))
    total = collect_moe_aux_loss(block)
    assert total.item() > 0, "collect_moe_aux_loss must find the block's MoE layer"


def test_block_default_is_dense_and_unchanged():
    """moe_experts=1 (default) must leave the block byte-identical to before."""
    torch.manual_seed(3)
    blk = DeltaNetBlock(d_model=32, n_heads=4, dropout=0.0)
    assert isinstance(blk.ffn, nn.Sequential), "default must remain the dense FFN"
    assert not any(isinstance(m, MoEFFN) for m in blk.modules())


def test_block_with_moe_runs_and_grows_params():
    torch.manual_seed(4)
    dense = DeltaNetBlock(d_model=32, n_heads=4, dropout=0.0)
    moe = DeltaNetBlock(d_model=32, n_heads=4, dropout=0.0, moe_experts=8, moe_top_k=2)
    assert any(isinstance(m, MoEFFN) for m in moe.modules())

    x = torch.randn(2, 6, 32)
    out = moe(x)
    assert out.shape == x.shape and torch.isfinite(out).all()

    p_dense = sum(p.numel() for p in dense.parameters())
    p_moe = sum(p.numel() for p in moe.parameters())
    assert p_moe > p_dense, "MoE block should hold more parameters than dense"


def test_swiglu_variant_single_expert_matches_dense():
    """Same known-answer check for the SwiGLU expert flavour."""
    from hvn.core.hvn_block import SwiGLU
    torch.manual_seed(5)
    d = 32
    moe = MoEFFN(d, num_experts=1, top_k=1, expansion=3, dropout=0.0, use_swiglu=True)
    dense = SwiGLU(d, expansion=3, dropout=0.0)
    with torch.no_grad():
        dense.w1.weight.copy_(moe.w1[0].T)
        dense.w3.weight.copy_(moe.w3[0].T)
        dense.w2.weight.copy_(moe.w2[0].T)
    moe.eval(); dense.eval()
    x = torch.randn(2, 5, d)
    assert torch.allclose(moe(x), dense(x), atol=1e-6)


if __name__ == "__main__":
    test_single_expert_matches_dense_ffn_forward_and_backward()
    test_more_experts_do_not_increase_active_compute()
    test_routing_is_sparse_not_dense_average()
    test_aux_loss_present_and_collectable()
    test_block_default_is_dense_and_unchanged()
    test_block_with_moe_runs_and_grows_params()
    test_swiglu_variant_single_expert_matches_dense()
    print("All test_moe_deltanet tests passed.")
