"""In-place DeltaNet decode state must be numerically identical (#138).

Written by: Rambarun Komaljeet

The recurrent state and conv buffer are now mutated in place rather than
reallocated per token, so a CUDA graph has stable addresses to replay against.
This is the serving path, and #117 is the precedent for what an unverified cache
change costs: a register-if-absent dedup silently served untrained weights for an
entire epic, undetected, because the only test looked at aggregate loss.

So the gate here is a **reference implementation** of the old rebinding
arithmetic, run token by token against the live one — not "generation still looks
reasonable".
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.core.deltanet import DeltaNetBlock  # noqa: E402


def make_block(seed=0, d_model=64, n_heads=4, conv_kernel=4):
    torch.manual_seed(seed)
    blk = DeltaNetBlock(d_model=d_model, n_heads=n_heads,
                             conv_kernel=conv_kernel)
    blk.eval()
    return blk


@torch.no_grad()
def reference_step(blk, x_t, state):
    """The pre-#138 arithmetic: rebind instead of mutate.

    Deliberately a copy of the old code rather than a call into the new one — a
    reference that shares an implementation proves nothing.
    """
    B, _, D = x_t.shape
    H, d = blk.n_heads, blk.d_head
    N = B * H

    x_norm = blk.norm(x_t)
    if blk.conv_kernel:
        xt = x_norm.transpose(1, 2)
        window = torch.cat([state["conv"], xt], dim=2)
        y = blk.conv(window)
        new_conv = window[:, :, 1:].clone()
        xc = y.transpose(1, 2)
    else:
        xc, new_conv = x_norm, None
    q, k, v, alpha, beta, gate = blk._project(xc)

    def to_N(t):
        return t.permute(0, 2, 1, 3).reshape(N, -1)

    q_t, k_t, v_t, beta_t = to_N(q), to_N(k), to_N(v), to_N(beta)
    S = state["S"]
    if alpha is not None:
        S = S * to_N(alpha).unsqueeze(1)
    Sk = (S * k_t.unsqueeze(1)).sum(-1)
    u = v_t - Sk
    S = S + beta_t.unsqueeze(-1) * (u.unsqueeze(2) * k_t.unsqueeze(1))
    o_t = (S * q_t.unsqueeze(1)).sum(-1)

    o = o_t.view(B, H, 1, d).permute(0, 2, 1, 3)
    o = blk._read_normalize(o).reshape(B, 1, D)
    if gate is not None:
        o = o * gate
    x_out = x_t + blk.dropout(blk.out_proj(o)) + blk.ffn(x_norm)
    ns = {"S": S}
    if blk.conv_kernel:
        ns["conv"] = new_conv
    return x_out, ns


# --------------------------------------------------------------------------


@pytest.mark.parametrize("conv_kernel", [0, 4])
@pytest.mark.parametrize("n_tokens", [1, 3, 12])
def test_inplace_matches_the_rebinding_reference(conv_kernel, n_tokens):
    blk = make_block(conv_kernel=conv_kernel)
    torch.manual_seed(1)
    tokens = [torch.randn(2, 1, 64) for _ in range(n_tokens)]

    live = blk.init_state(2)
    ref = {k: v.clone() for k, v in blk.init_state(2).items()}

    with torch.no_grad():
        for i, x in enumerate(tokens):
            out_live, live = blk.forward_step(x, live)
            out_ref, ref = reference_step(blk, x, ref)
            assert torch.allclose(out_live, out_ref, atol=1e-6), (
                f"token {i}: max|diff| = {(out_live - out_ref).abs().max():.3e}")
            assert torch.allclose(live["S"], ref["S"], atol=1e-6), (
                f"token {i}: state diverged")


def test_state_tensors_are_never_reallocated():
    """The property CUDA graph capture actually needs: stable addresses."""
    blk = make_block()
    state = blk.init_state(1)
    s_ptr = state["S"].data_ptr()
    c_ptr = state["conv"].data_ptr()
    with torch.no_grad():
        for i in range(20):
            _out, state = blk.forward_step(torch.randn(1, 1, 64), state)
            assert state["S"].data_ptr() == s_ptr, f"S reallocated at token {i}"
            assert state["conv"].data_ptr() == c_ptr, f"conv reallocated at {i}"


def test_conv_window_shifts_by_exactly_one():
    """A shift applied before the convolution reads it would drop the oldest tap.

    Checked by content, not by shape: the buffer after the step must equal the
    previous buffer shifted left with the new token appended.
    """
    blk = make_block(conv_kernel=4)
    state = blk.init_state(1)
    with torch.no_grad():
        for _ in range(6):                       # fill with distinguishable values
            blk.forward_step(torch.randn(1, 1, 64), state)
        before = state["conv"].clone()
        x = torch.randn(1, 1, 64)
        blk.forward_step(x, state)
    expected_tail = blk.norm(x).transpose(1, 2)[:, :, 0]
    assert torch.allclose(state["conv"][:, :, :-1], before[:, :, 1:], atol=1e-6)
    assert torch.allclose(state["conv"][:, :, -1], expected_tail, atol=1e-5)


def test_caller_holding_the_old_state_sees_the_mutation():
    """Documents a real behaviour change rather than hiding it.

    In-place update means the state dict handed IN is modified. Any caller that
    kept a reference expecting an immutable snapshot must clone it. hvn/chat.py's
    StateCheckpoint serialises to JSON (a copy), so it is unaffected — but this
    pins the contract so a future caller cannot be surprised silently.
    """
    blk = make_block()
    state = blk.init_state(1)
    kept = state["S"]
    with torch.no_grad():
        blk.forward_step(torch.randn(1, 1, 64), state)
    assert kept is state["S"]
    assert kept.abs().sum() > 0        # the caller's own reference moved
