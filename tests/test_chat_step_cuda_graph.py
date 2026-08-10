"""chat_step with CUDA-graph decode must generate the SAME tokens (#138).

Written by: Rambarun Komaljeet

Written to be capable of failing on the bug that actually shipped, which the
previous check was not. That bug: `written` was a Python int, so a captured graph
baked in one write slot and one attention extent, and every replay decoded as
though the context had stopped growing. A single-step comparison passed it at
max|dlogits| = 0.000e+00, because one replay and one eager step from the same
state do the same thing.

**The falsifiability question, asked before running:** if the graph replayed a
stale slot, would this test fail? Yes — greedy decode compares the whole emitted
SEQUENCE, and a stale slot changes the second token onward. That is why the gate
here is token equality over many steps rather than logit equality on one.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN  # noqa: E402

requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(),
                                   reason="CUDA graphs need a GPU")


def tiny_hybrid(vocab=256, d_model=64, layers=4, window=16):
    torch.manual_seed(0)
    plan = ["deltanet"] * layers
    plan[1] = "transformer"       # at least one attention block, so the ring
    plan[3] = "transformer"       # counter and RoPE table are both exercised
    m = EdgeHVN(vocab_size=vocab, d_model=d_model, num_layers=layers,
                n_heads=4, block_plan=plan, attn_window=window)
    return m.cuda().eval()


def greedy(model, prompt, n, use_graph, seed=0):
    """Deterministic decode: top_k=1 makes multinomial pick the argmax."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    state = model.init_conversation_state()
    out, _ = model.chat_step(prompt, state, max_new_tokens=n, temperature=1.0,
                             top_k=1, use_cuda_graph=use_graph)
    return out


@requires_cuda
def test_graph_and_eager_emit_identical_token_sequences():
    """The gate. A baked-in slot or position diverges from token 2 onward."""
    model = tiny_hybrid()
    model.enable_graph_decode(max_positions=64)
    prompt = [3, 9, 14, 2, 7]

    eager = greedy(model, prompt, 24, use_graph=False)
    graphed = greedy(model, prompt, 24, use_graph=True)

    assert len(eager) > 3, "decode stopped too early to be a meaningful test"
    assert eager == graphed, (
        f"diverged at token "
        f"{next((i for i, (a, b) in enumerate(zip(eager, graphed)) if a != b), '?')}"
        f"\n eager   {eager}\n graphed {graphed}")


@requires_cuda
def test_generation_is_long_enough_to_expose_a_stale_slot():
    """Guards the guard: a 1-token generation could not detect the shipped bug.

    If a future change shortens the decode above, this fails loudly rather than
    letting the real test silently become vacuous.
    """
    model = tiny_hybrid()
    model.enable_graph_decode(max_positions=64)
    out = greedy(model, [3, 9, 14], 24, use_graph=True)
    assert len(out) >= 8, (
        f"only {len(out)} tokens generated; the stale-slot bug shows from token 2, "
        f"but a handful of tokens makes the comparison weak")


@requires_cuda
def test_ring_counter_advances_under_graph_decode():
    """Directly the quantity that was frozen: the ring counter.

    The shipped bug left it pinned while logits kept changing, so the run looked
    alive. Asserted on its own rather than inferred from output equality.
    """
    model = tiny_hybrid()
    model.enable_graph_decode(max_positions=64)
    state = model.init_conversation_state()
    model.chat_step([3, 9, 14], state, max_new_tokens=12, temperature=1.0,
                    top_k=1, use_cuda_graph=True)
    counters = [int(s["written"]) for s in state["hamiltonian_states"]
                if isinstance(s, dict) and torch.is_tensor(s.get("written"))]
    assert counters, "no device-resident ring counter found — graph prep did not run"
    for c in counters:
        assert c >= 12, f"ring counter stuck at {c}; the context stopped growing"


@requires_cuda
def test_capture_refuses_without_graph_prep_and_falls_back():
    """Without enable_graph_decode the state is int-based, which cannot be graphed.

    Capturing anyway is the silent-wrong path, so _capture_decode_graph must
    refuse — and chat_step must still produce the eager answer rather than raise.
    A serving path that dies on an optimisation is worse than a slow one.
    """
    model = tiny_hybrid()                    # deliberately NOT prepped
    prompt = [3, 9, 14]
    eager = greedy(model, prompt, 12, use_graph=False)
    attempted = greedy(model, prompt, 12, use_graph=True)
    assert attempted == eager


@requires_cuda
def test_use_cuda_graph_defaults_off():
    """Existing callers must be byte-identical to before this change."""
    import inspect
    sig = inspect.signature(EdgeHVN.chat_step)
    assert sig.parameters["use_cuda_graph"].default is False
