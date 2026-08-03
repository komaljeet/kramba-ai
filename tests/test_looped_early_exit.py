"""Tests for Issue #122 — Looped Early Exit (hvn/edge/looped.py).

CPU-only, tiny dims — these are correctness/contract tests, not the GPU
hyperparameter sweep (see scripts/sweep_looped_early_exit.py for that).
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hvn.edge.model import EdgeHVN
from hvn.edge.looped import (
    enable_looped_early_exit,
    disable_looped_early_exit,
    pop_loop_log,
    normalized_entropy,
    _clone_state,
)


def _tiny_hybrid_model(seed=0):
    torch.manual_seed(seed)
    model = EdgeHVN(
        vocab_size=64,
        d_model=32,
        num_layers=3,
        n_heads=4,
        use_deltanet=True,
        block_plan=["deltanet", "transformer", "deltanet"],
        attn_window=16,
    )
    model.eval()
    return model


def test_max_loops_1_is_bit_identical_to_base_forward_step():
    model = _tiny_hybrid_model()
    states_a = model.init_state(1)
    states_b = [_clone_state(s) for s in states_a]

    tok = torch.tensor([[5]])
    with torch.no_grad():
        base_logits, base_states = model.forward_step(tok, states_a, position_idx=0)

    enable_looped_early_exit(model, max_loops=1, entropy_threshold=1.1)  # always "confident" after 1
    with torch.no_grad():
        looped_logits, looped_states = model.forward_step(tok, states_b, position_idx=0)

    assert torch.equal(base_logits, looped_logits), "max_loops=1 must reproduce the base forward_step exactly"
    for sa, sb in zip(base_states, looped_states):
        for k in sa:
            assert torch.equal(sa[k], sb[k]), f"state mismatch in key {k!r}"


def test_commits_loop1_state_and_final_loop_logits():
    """max_loops=2, threshold that can never be satisfied -> always runs both loops.

    Checks the two halves of the contract separately:
      * LOGITS come from the LAST loop run (loop 2 here), computed by feeding
        loop 1's hidden state back through the stack with the per-loop state
        reset to the pre-token snapshot.
      * STATE committed is LOOP 1's, NOT loop 2's -- only loop 1 runs the blocks
        on the real token embedding, so only its state records "I saw token X".
        Committing loop 2's state instead measured 40% -> 0% near-window recall
        on the frozen checkpoint (scripts/test_state_commit_hypothesis.py); an
        earlier version of this test asserted the loop-2 state and so locked the
        bug in place.
    """
    model = _tiny_hybrid_model(seed=1)
    base_states = model.init_state(1)
    tok = torch.tensor([[3]])

    # Manual 2-loop replay (the reference the wrapper must match).
    with torch.no_grad():
        x0 = model.token_embedding(tok)

        loop1_states = [_clone_state(s) for s in base_states]
        h = x0
        for j, block in enumerate(model.blocks):
            h, st = block.forward_step(h, loop1_states[j], position_idx=0)
            loop1_states[j] = st
        loop1_hidden = h

        # Loop 2 MUST start from a fresh clone of base_states, not loop1_states.
        loop2_states = [_clone_state(s) for s in base_states]
        h = loop1_hidden
        for j, block in enumerate(model.blocks):
            h, st = block.forward_step(h, loop2_states[j], position_idx=0)
            loop2_states[j] = st
        expected_logits = model.decoder(model.final_norm(h))

    enable_looped_early_exit(model, max_loops=2, entropy_threshold=-1.0)  # never satisfied -> always 2 loops
    with torch.no_grad():
        looped_logits, looped_states = model.forward_step(tok, [_clone_state(s) for s in base_states], position_idx=0)

    # Logits: from the LAST loop (loop 2).
    assert torch.equal(expected_logits, looped_logits)

    # State: from LOOP 1 -- and provably NOT loop 2's.
    for sa, sb in zip(loop1_states, looped_states):
        for k in sa:
            assert torch.equal(sa[k], sb[k]), f"committed state should be loop 1's; mismatch in {k!r}"

    differs = any(
        not torch.equal(s1[k], s2[k])
        for s1, s2 in zip(loop1_states, loop2_states) for k in s1
    )
    assert differs, "loop-1 and loop-2 states are identical, so this test cannot detect a regression"

    log = pop_loop_log(model)
    assert log == [2], f"expected exactly 2 loops used, got {log}"


def test_entropy_threshold_gates_exit():
    model = _tiny_hybrid_model(seed=2)
    states = model.init_state(1)
    tok = torch.tensor([[7]])

    # Threshold impossible to satisfy (entropy is always in [0,1], so a
    # negative threshold never fires) -> must run every one of max_loops.
    enable_looped_early_exit(model, max_loops=5, entropy_threshold=-1.0)
    with torch.no_grad():
        model.forward_step(tok, [_clone_state(s) for s in states], position_idx=0)
    assert pop_loop_log(model) == [5]

    # Threshold trivially satisfied by any distribution (normalized entropy
    # <= 1 always) -> must exit after exactly 1 loop.
    enable_looped_early_exit(model, max_loops=5, entropy_threshold=1.1)
    with torch.no_grad():
        model.forward_step(tok, [_clone_state(s) for s in states], position_idx=0)
    assert pop_loop_log(model) == [1]


def test_disable_restores_original_forward_step():
    model = _tiny_hybrid_model(seed=3)
    original = model.forward_step
    enable_looped_early_exit(model, max_loops=3, entropy_threshold=0.3)
    assert model.forward_step is not original
    disable_looped_early_exit(model)
    assert model.forward_step == original
    assert not hasattr(model, "_looped_max_loops")


def test_chat_session_needs_zero_changes():
    """hvn/chat.py::ChatSession only ever calls model.forward_step,
    model.init_conversation_state, and model.chat_step -- none of which
    change signature here. Exercise the real EdgeHVN.chat_step (unmodified,
    inherited) end-to-end and confirm it transparently loops via the
    monkeypatched forward_step.
    """
    model = _tiny_hybrid_model(seed=4)
    enable_looped_early_exit(model, max_loops=3, entropy_threshold=0.2)

    conv_state = model.init_conversation_state()
    response_ids, conv_state = model.chat_step(
        input_ids=[1, 2, 3], conv_state=conv_state, max_new_tokens=4, top_k=5,
    )
    assert isinstance(response_ids, list)
    log = pop_loop_log(model)
    assert len(log) > 0
    assert all(1 <= n <= 3 for n in log)


def test_normalized_entropy_bounds():
    vocab = 100
    # Uniform distribution -> max entropy -> normalized entropy ~= 1.0
    uniform_logits = torch.zeros(1, 1, vocab)
    assert abs(normalized_entropy(uniform_logits, vocab) - 1.0) < 1e-5

    # One-hot-ish (huge logit on one class) -> near-zero entropy
    peaked_logits = torch.full((1, 1, vocab), -1e4)
    peaked_logits[0, 0, 0] = 1e4
    assert normalized_entropy(peaked_logits, vocab) < 1e-3


if __name__ == "__main__":
    test_max_loops_1_is_bit_identical_to_base_forward_step()
    test_commits_loop1_state_and_final_loop_logits()
    test_entropy_threshold_gates_exit()
    test_disable_restores_original_forward_step()
    test_chat_session_needs_zero_changes()
    test_normalized_entropy_bounds()
    print("All test_looped_early_exit tests passed.")
