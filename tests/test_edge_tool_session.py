"""Tests for hvn/edge/tool_session.py's EdgeToolCallingSession (issue #110 /
#97 Phase C).

The core mechanic under test: does the session correctly (a) detect a newly
COMPLETED ``[TOOL_CALL]...[/TOOL_CALL]`` block as soon as it appears in the
generated-so-far text, (b) execute it for real via ToolExecutor, (c) feed
the real result back through ``forward_step`` as CONTEXT rather than
sampling it, and (d) resume sampling for the final answer, stopping at EOS?

To test this deterministically (no real trained checkpoint needed --
that's exercised separately by the GPU agentic gate, C6), a scripted stub
model replaces EdgeHVN. Rather than hand-counting forward_step call
indices (fragile: prompt-feeding and tool-result-feeding each trigger a
variable number of calls, only the LAST of which actually matters --
everything else is provably discarded before ever being sampled from), the
stub's output is a pure function of its INPUT token id: a small
transition table {input_token_id: forced_next_token_id}. This sidesteps
needing to know exactly how many forward_step calls happen at each phase.
"""

from __future__ import annotations

import os
import tempfile

import torch

from hvn.edge.tool_session import EdgeToolCallingSession, EOS_ID
from hvn.tools import ToolExecutor, create_coding_registry, TOOL_TOKENS


class _SeqTokenizer:
    """Whole-string-to-one-token-id tokenizer: encode(text) always returns
    exactly one id (registering `text` in the vocab on first use, reusing
    the same id thereafter). This makes exact scripted control simple --
    each meaningful string (prompt, tool-call block, tool-result block,
    answer) is exactly one token, no subword bookkeeping."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {EOS_ID: ""}  # EOS never appears in decoded text

    def encode(self, text: str) -> list[int]:
        if text not in self.vocab:
            i = len(self.vocab)
            self.vocab[text] = i
            self.rev[i] = text
        return [self.vocab[text]]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.rev[int(i)] for i in ids)


class _ScriptedModel:
    """Stub replacing EdgeHVN: forward_step's output logits are a pure
    function of the input token id, via `transition` ({in_id: forced_out_id}).
    Implements exactly the surface EdgeToolCallingSession.run() touches."""

    def __init__(self, transition: dict[int, int], vocab_size: int,
                device: torch.device, dtype: torch.dtype = torch.float32):
        self.transition = transition
        self.vocab_size = vocab_size
        self.device = device
        self.dtype = dtype
        self._param = torch.zeros(1, dtype=dtype, device=device)

    def parameters(self):
        yield self._param

    def init_conversation_state(self, device=None, dtype=None) -> dict:
        return {"hamiltonian_states": None, "position": 0, "history_ids": []}

    def forward_step(self, input_ids, states, position_idx=None):
        cur_id = int(input_ids.item())
        target = self.transition.get(cur_id, cur_id)
        logits = torch.full((1, 1, self.vocab_size), -100.0,
                            device=self.device, dtype=self.dtype)
        logits[0, 0, target] = 100.0
        return logits, states

    def _apply_repetition_penalty(self, logits, seen_ids, penalty):
        return logits  # no-op: determinism, not sampling realism, is the point

    def _filter_top_k(self, logits, top_k):
        return logits


def test_session_detects_executes_and_continues_past_tool_call():
    with tempfile.TemporaryDirectory() as sandbox:
        with open(os.path.join(sandbox, "x.py"), "w", encoding="utf-8") as f:
            f.write("VALUE = 1\n")
        executor = ToolExecutor(create_coding_registry(sandbox))

        tok = _SeqTokenizer()

        # Fix PID (prompt) and TID (the tool-call block) ids by pre-encoding
        # them exactly as the session will (same formatter, same text).
        from hvn.edge.chat import EdgeConversationFormatter
        formatter = EdgeConversationFormatter()
        prompt_text = formatter.format_user("read x.py") + formatter.format_assistant_open()
        pid = tok.encode(prompt_text)[0]

        tool_call_json = '{"name":"read_file","arguments":{"path":"x.py"}}'
        tool_call_block = (f"{TOOL_TOKENS['tool_call_start']}\n{tool_call_json}\n"
                          f"{TOOL_TOKENS['tool_call_end']}")
        tid = tok.encode(tool_call_block)[0]

        # Predict the EXACT real tool-result block the session will produce
        # (same executor, same args) so its token id (RID) is fixed too.
        result_block = executor.execute({"name": "read_file", "arguments": {"path": "x.py"}})
        result_text = "\n" + result_block + "\n"
        rid = tok.encode(result_text)[0]

        answer_text = "Done reading."
        aid = tok.encode(answer_text)[0]

        transition = {
            pid: tid,   # after the prompt -> sample the tool-call block
            tid: tid,   # discarded call (overwritten by the result-feed's own output)
            rid: aid,   # after the injected result -> sample the answer
            aid: EOS_ID,  # after the answer -> stop
        }
        model = _ScriptedModel(transition, vocab_size=EOS_ID + 1, device=torch.device("cpu"))

        session = EdgeToolCallingSession(model, tok, executor, device=torch.device("cpu"),
                                         max_new_tokens=20, max_tool_calls=3)
        result = session.run("read x.py")

        assert result["num_tool_calls"] == 1
        assert result["tool_calls"][0]["name"] == "read_file"
        assert result["tool_calls"][0]["arguments"] == {"path": "x.py"}
        assert "VALUE = 1" in result["text"], (
            "the REAL tool result must be present in the generated text (injected as context)"
        )
        assert answer_text in result["text"]
        assert result["answer"].strip() == answer_text, (
            "the answer field must strip tool-call/tool-result blocks, leaving just the "
            "final answer text"
        )


def test_session_stops_at_max_tool_calls_safety_cap():
    """A model that would call a tool forever must be stopped by the safety
    cap, not hang or loop indefinitely.

    Uses read_file (fully deterministic output) rather than run_tests: an
    earlier version of this test used run_tests, whose real pytest output
    includes a wall-clock timing string ("1 passed in 0.23s") that differs
    on every call -- since this test predicts the tool-result's token id by
    pre-encoding the SAME text the session will independently produce, a
    non-deterministic result string means the session's actual call gets a
    DIFFERENT (unregistered) token id than predicted, silently falling back
    to an identity transition and breaking the scripted loop after a
    random number of iterations. Caught via genuine test flakiness (passed
    ~1 in 4 runs) -- not a bug in EdgeToolCallingSession itself, but in this
    test's tool choice.
    """
    with tempfile.TemporaryDirectory() as sandbox:
        with open(os.path.join(sandbox, "x.py"), "w", encoding="utf-8") as f:
            f.write("VALUE = 1\n")
        executor = ToolExecutor(create_coding_registry(sandbox))
        tok = _SeqTokenizer()

        from hvn.edge.chat import EdgeConversationFormatter
        formatter = EdgeConversationFormatter()
        prompt_text = formatter.format_user("loop please") + formatter.format_assistant_open()
        pid = tok.encode(prompt_text)[0]

        tool_call_block = (f"{TOOL_TOKENS['tool_call_start']}\n"
                          f'{{"name":"read_file","arguments":{{"path":"x.py"}}}}\n'
                          f"{TOOL_TOKENS['tool_call_end']}")
        tid = tok.encode(tool_call_block)[0]

        result_block = executor.execute({"name": "read_file", "arguments": {"path": "x.py"}})
        result_text = "\n" + result_block + "\n"
        rid = tok.encode(result_text)[0]

        # The model ALWAYS re-emits the same tool call after any result --
        # an infinite-tool-calling loop, if not for the safety cap.
        transition = {pid: tid, tid: tid, rid: tid}
        model = _ScriptedModel(transition, vocab_size=EOS_ID + 1, device=torch.device("cpu"))

        session = EdgeToolCallingSession(model, tok, executor, device=torch.device("cpu"),
                                         max_new_tokens=1000, max_tool_calls=3)
        result = session.run("loop please")

        assert result["num_tool_calls"] == 3, "must stop exactly at max_tool_calls"


def test_session_with_no_tool_call_still_returns_cleanly():
    """A model that never emits a tool call (just answers directly) must
    work too -- the tool-calling machinery is opt-in per response, not
    mandatory."""
    with tempfile.TemporaryDirectory() as sandbox:
        executor = ToolExecutor(create_coding_registry(sandbox))
        tok = _SeqTokenizer()

        from hvn.edge.chat import EdgeConversationFormatter
        formatter = EdgeConversationFormatter()
        prompt_text = formatter.format_user("hi") + formatter.format_assistant_open()
        pid = tok.encode(prompt_text)[0]
        answer_text = "Hello there."
        aid = tok.encode(answer_text)[0]

        transition = {pid: aid, aid: EOS_ID}
        model = _ScriptedModel(transition, vocab_size=EOS_ID + 1, device=torch.device("cpu"))

        session = EdgeToolCallingSession(model, tok, executor, device=torch.device("cpu"),
                                         max_new_tokens=20)
        result = session.run("hi")

        assert result["num_tool_calls"] == 0
        assert result["answer"].strip() == answer_text
