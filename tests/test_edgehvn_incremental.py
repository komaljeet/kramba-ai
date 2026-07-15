"""CPU tests for EdgeHVN's incremental decode path (issue #97 Phase B).

Covers:
  1. TransformerBlock.forward_step parity vs a plain forward() (both within
     and beyond the sliding window, exercising cache truncation).
  2. EdgeHVN.forward_step parity vs forward() on a real mixed hybrid
     block_plan (deltanet + transformer layers).
  3. init_conversation_state / chat_step: correct dict shape, multi-turn
     state carries across calls.
  4. The mixed state round-trips through hvn/chat.py's generic
     _encode_state_tree/_decode_state_tree (the mechanism StateCheckpoint
     uses to persist conversations to disk).
  5. ChatSession + EdgeConversationFormatter integration: the formatter
     duck-types correctly (system-prompt no-op, EOS-terminated turns, no
     closing tags) and a full multi-turn session runs without error.

Tiny dims keep this fast on CPU. Run: ``python -m pytest tests/test_edgehvn_incremental.py``
"""

from __future__ import annotations

import torch

from hvn.core.transformer_block import TransformerBlock
from hvn.edge.model import EdgeHVN
from hvn.edge.chat import EdgeConversationFormatter, USER_PREFIX, ASSISTANT_PREFIX, EOS_ID
from hvn.chat import ChatSession, _encode_state_tree, _decode_state_tree


# ---------------------------------------------------------------------------
# 1. TransformerBlock parity
# ---------------------------------------------------------------------------

def test_transformer_block_forward_step_parity_within_window():
    torch.manual_seed(0)
    blk = TransformerBlock(d_model=32, n_heads=4, dropout=0.0, window=8)
    blk.eval()
    B, S, D = 1, 6, 32
    x = torch.randn(B, S, D)

    with torch.no_grad():
        full_out = blk(x)
        state = blk.init_state(B)
        outs = []
        for t in range(S):
            o, state = blk.forward_step(x[:, t:t + 1, :], state, position_idx=t)
            outs.append(o)
        step_out = torch.cat(outs, dim=1)

    assert (full_out - step_out).abs().max().item() < 1e-4


def test_transformer_block_forward_step_parity_beyond_window():
    """S > window exercises both the training-time sliding mask AND cache truncation."""
    torch.manual_seed(1)
    blk = TransformerBlock(d_model=32, n_heads=4, dropout=0.0, window=4)
    blk.eval()
    B, S, D = 1, 10, 32
    x = torch.randn(B, S, D)

    with torch.no_grad():
        full_out = blk(x)
        state = blk.init_state(B)
        outs = []
        for t in range(S):
            o, state = blk.forward_step(x[:, t:t + 1, :], state, position_idx=t)
            outs.append(o)
        step_out = torch.cat(outs, dim=1)

    assert (full_out - step_out).abs().max().item() < 1e-4
    assert state["k_cache"].shape[2] <= 4, "cache must stay bounded to the window"


# ---------------------------------------------------------------------------
# 2. EdgeHVN parity on a real hybrid block_plan
# ---------------------------------------------------------------------------

def _tiny_hybrid_model(seed=0):
    torch.manual_seed(seed)
    block_plan = ["deltanet", "deltanet", "transformer", "deltanet", "transformer", "deltanet"]
    model = EdgeHVN(vocab_size=100, d_model=32, num_layers=6, n_heads=4, dropout=0.0,
                    block_plan=block_plan, attn_window=8)
    model.eval()
    return model


def test_edgehvn_forward_step_parity():
    model = _tiny_hybrid_model()
    B, S = 1, 10
    torch.manual_seed(2)
    ids = torch.randint(0, 100, (B, S))

    with torch.no_grad():
        full_logits = model(ids)
        states = model.init_state(B)
        step_logits = []
        for t in range(S):
            logit, states = model.forward_step(ids[:, t:t + 1], states, position_idx=t)
            step_logits.append(logit)
        step_logits = torch.cat(step_logits, dim=1)

    assert (full_logits - step_logits).abs().max().item() < 1e-3


def test_deltanet_forward_step_matches_scan_sequential_not_scan_chunked():
    """Regression test locking in a real finding from manual GPU/CPU
    investigation (2026-07-15): comparing forward_step against a trained
    checkpoint's plain forward() can show LARGE divergence (order 1-3 in
    logit scale over ~20 tokens) that looks like a serving bug but isn't.

    Root cause: forward() dispatches to DeltaNetBlock._scan_chunked (a fast,
    reassociated approximation used for training speed) whenever
    `train_chunk` is set, NOT the exact `_scan_sequential` formula that
    `forward_step` implements token-by-token. `_scan_chunked` itself
    diverges non-trivially from `_scan_sequential` on real trained weights
    (verified directly, bypassing forward_step entirely) -- this is a
    pre-existing numerical property of the training-speed optimization
    (#83/#96), not a bug introduced by the incremental decode path (#97
    Phase B). forward_step's actual formula matches `_scan_sequential`
    (the documented reference recurrence) to ~1e-6, proving it's correct.

    This test uses train_chunk=4 (< S) specifically to force the chunked
    path in forward(), the same regime that surfaced the finding, and pins
    the two comparisons that matter: forward_step ~= _scan_sequential
    (tight), independent of whatever forward()/_scan_chunked does.
    """
    torch.manual_seed(7)
    from hvn.core.deltanet import DeltaNetBlock
    blk = DeltaNetBlock(d_model=32, n_heads=4, dropout=0.0, train_chunk=4)
    blk.eval()
    B, S, D = 1, 10, 32
    x = torch.randn(B, S, D)

    with torch.no_grad():
        x_norm = blk.norm(x)
        xc = blk._causal_conv(x_norm)
        q, k, v, alpha, beta, gate = blk._project(xc)
        H, d = blk.n_heads, blk.d_head
        N = B * H

        def to_N(t):
            return t.permute(0, 2, 1, 3).reshape(N, S, -1)

        qN, kN, vN, betaN = to_N(q), to_N(k), to_N(v), to_N(beta)
        alphaN = to_N(alpha) if alpha is not None else None
        o_seq = blk._scan_sequential(qN, kN, vN, alphaN, betaN)

        state = blk.init_state(B)
        outs = []
        for t in range(S):
            o, state = blk.forward_step(x[:, t:t + 1, :], state, position_idx=t)
            outs.append(o)
        step_out = torch.cat(outs, dim=1)

        # Recompute o_seq through the same post-processing as forward_step's
        # output for an apples-to-apples comparison.
        o = o_seq.view(B, H, S, d).permute(0, 2, 1, 3)
        o = blk._read_normalize(o).reshape(B, S, D)
        if gate is not None:
            o = o * gate
        full_from_seq = x + blk.dropout(blk.out_proj(o)) + blk.ffn(x_norm)

    diff_vs_sequential = (full_from_seq - step_out).abs().max().item()
    assert diff_vs_sequential < 1e-4, (
        "forward_step must match the exact _scan_sequential reference tightly "
        f"(got {diff_vs_sequential}) -- this is the correctness bar, not "
        "parity against forward()'s fast chunked approximation."
    )


# ---------------------------------------------------------------------------
# 3. init_conversation_state / chat_step
# ---------------------------------------------------------------------------

def test_chat_step_multiturn_state():
    model = _tiny_hybrid_model(seed=3)
    conv_state = model.init_conversation_state()
    assert set(conv_state.keys()) == {"hamiltonian_states", "position", "history_ids"}
    assert conv_state["position"] == 0
    assert conv_state["history_ids"] == []

    resp1, conv_state = model.chat_step([1, 2, 3], conv_state, max_new_tokens=10,
                                        temperature=1.0, top_k=5)
    assert conv_state["position"] == 3 + len(resp1)
    assert conv_state["history_ids"] == [1, 2, 3] + resp1

    pos_after_turn1 = conv_state["position"]
    resp2, conv_state = model.chat_step([4, 5], conv_state, max_new_tokens=10,
                                        temperature=1.0, top_k=5)
    # State carries: position keeps accumulating, doesn't reset.
    assert conv_state["position"] == pos_after_turn1 + 2 + len(resp2)


def test_chat_step_stops_at_eos():
    """chat_step must stop before max_new_tokens if EOS is sampled -- force
    it by using a model whose decoder is biased to always emit EOS_ID.
    Needs vocab_size > EOS_ID (50256, GPT-2's real id, hardcoded in
    chat_step) -- unlike the other tests, which don't care about the
    specific token ids and use a tiny vocab for speed."""
    torch.manual_seed(4)
    block_plan = ["deltanet", "transformer"]
    model = EdgeHVN(vocab_size=EOS_ID + 1, d_model=16, num_layers=2, n_heads=2,
                    dropout=0.0, block_plan=block_plan, attn_window=8)
    model.eval()
    with torch.no_grad():
        model.decoder.decoder.bias = torch.nn.Parameter(
            torch.full((model.decoder.decoder.out_features,), -10.0)
        )
        model.decoder.decoder.bias.data[EOS_ID] = 10.0
    conv_state = model.init_conversation_state()
    resp, conv_state = model.chat_step([1], conv_state, max_new_tokens=50,
                                       temperature=1.0, top_k=1)
    assert len(resp) < 50, "should stop early at EOS, not run the full budget"
    assert EOS_ID not in resp, "EOS itself should not appear in the returned response"


# ---------------------------------------------------------------------------
# 4. State round-trips through hvn/chat.py's generic encode/decode
# ---------------------------------------------------------------------------

def test_state_roundtrip_through_state_tree_codec():
    model = _tiny_hybrid_model(seed=5)
    conv_state = model.init_conversation_state()
    _, conv_state = model.chat_step([1, 2, 3], conv_state, max_new_tokens=5,
                                    temperature=1.0, top_k=5)

    encoded = _encode_state_tree(conv_state["hamiltonian_states"])
    import json
    json.dumps(encoded)  # must be JSON-serializable (StateCheckpoint writes JSON)
    decoded = _decode_state_tree(encoded, device=torch.device("cpu"), dtype=torch.float32)

    cur = torch.tensor([[7]], dtype=torch.long)
    with torch.no_grad():
        logit_orig, _ = model.forward_step(cur, conv_state["hamiltonian_states"],
                                           position_idx=conv_state["position"])
        logit_decoded, _ = model.forward_step(cur, decoded,
                                              position_idx=conv_state["position"])
    assert (logit_orig - logit_decoded).abs().max().item() < 1e-5


# ---------------------------------------------------------------------------
# 5. ChatSession + EdgeConversationFormatter integration
# ---------------------------------------------------------------------------

class _FakeTokenizer:
    """Minimal encode/decode stand-in -- avoids a network fetch for GPT2Tokenizer
    in this CPU unit test; ChatSession only needs .encode()/.decode()."""

    def encode(self, text: str) -> list[int]:
        return [min(ord(c), 99) for c in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(i) for i in ids)


def test_edge_conversation_formatter_no_system_prompt():
    fmt = EdgeConversationFormatter(system_prompt="You are helpful.")
    assert fmt.format_system("anything") == "", (
        "EdgeHVN was never trained with a system-prompt convention -- must be a no-op"
    )
    assert fmt.format_user("hi").startswith(USER_PREFIX)
    assert fmt.format_assistant_open() == ASSISTANT_PREFIX
    assert fmt.parse_response("  hello  ") == "hello"


def test_chat_session_with_edgehvn_end_to_end():
    model = _tiny_hybrid_model(seed=6)
    session = ChatSession(
        model, _FakeTokenizer(), device=torch.device("cpu"),
        system_prompt="You are a helpful assistant.",
        max_new_tokens=8, temperature=1.0, top_k=5,
        formatter=EdgeConversationFormatter(),
    )
    assert isinstance(session.formatter, EdgeConversationFormatter)
    # System prompt must NOT have been fed (format_system is a no-op) --
    # position stays 0 after _init_state, nothing was pre-fed.
    assert session.conv_state["position"] == 0

    reply1 = session.user_turn("hi")
    assert isinstance(reply1, str)
    pos_after_1 = session.conv_state["position"]
    assert pos_after_1 > 0

    reply2 = session.user_turn("how are you?")
    assert isinstance(reply2, str)
    assert session.conv_state["position"] > pos_after_1
