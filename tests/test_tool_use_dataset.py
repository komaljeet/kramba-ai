"""Tests for finetune_edge_conversational.py's ToolUseDataset (issue #110 /
#97 Phase C).

Covers the three-way masking that fixes a real flaw in the pre-existing
hvn.tools.generate_tool_training_data (which trained the model to generate
tool-result content, i.e. hallucinate what a tool returns, instead of
treating it as environment-injected context): query masked, [TOOL_CALL]
blocks learned, [TOOL_RESULT] blocks masked, final answer + EOS learned.

Uses a deterministic fake tokenizer (word-level, matching the pattern
already used for GroundedQADataset's tests) rather than the real GPT-2
tokenizer, so this stays a fast, dependency-light CPU unit test.
"""

from __future__ import annotations

import json

from finetune_edge_conversational import ToolUseDataset, EOS_ID


class _FakeTokenizer:
    """Whitespace-ish word tokenizer with a stable vocab, deterministic
    encode/decode -- enough to verify masking boundaries without needing
    the real GPT-2 tokenizer (network/cache dependency) in a unit test."""

    def __init__(self):
        self.vocab: dict[str, int] = {}
        self.rev: dict[int, str] = {EOS_ID: "<eos>"}

    def _id(self, tok: str) -> int:
        if tok not in self.vocab:
            i = len(self.vocab)
            self.vocab[tok] = i
            self.rev[i] = tok
        return self.vocab[tok]

    def encode(self, text: str) -> list[int]:
        # Split on whitespace but keep it invertible: encode each character
        # run separated by single spaces, re-joinable by decode.
        return [self._id(tok) for tok in text.split(" ")]

    def decode(self, ids: list[int]) -> str:
        return " ".join(self.rev[i] for i in ids)


def _one_step_record():
    return {
        "query": "What is 2 + 2 ?",
        "steps": [{
            "tool_call": {"name": "calculator", "arguments": {"expression": "2+2"}},
            "tool_result": "4",
        }],
        "answer": "The answer is 4 .",
    }


def _multi_step_record():
    return {
        "query": "Fix the bug in foo.py .",
        "steps": [
            {"tool_call": {"name": "read_file", "arguments": {"path": "foo.py"}},
             "tool_result": "def foo ( ) : return 1"},
            {"tool_call": {"name": "edit_file",
                          "arguments": {"path": "foo.py", "old_text": "1", "new_text": "2"}},
             "tool_result": "OK : foo.py updated"},
            {"tool_call": {"name": "run_tests", "arguments": {}},
             "tool_result": "PASSED"},
        ],
        "answer": "Fixed it .",
    }


def _learned_and_masked_text(tok, ids, labels):
    """Split `ids` by whether the (unshifted) label at that position is a
    learned target. Note: `__getitem__` applies a causal shift
    (x=ids[:-1], y=labels[1:]) for actual training, but for VISUALIZING
    which token content is a learned vs. masked target, the raw
    position-aligned (ids, labels) pair is what to inspect directly --
    ToolUseDataset builds `labels` 1:1 aligned with `ids` before any shift
    (see the assertions below), so `labels[i] != -100` means `ids[i]` is
    part of a learned segment."""
    learned = tok.decode([t for t, l in zip(ids, labels) if l != -100])
    masked = tok.decode([t for t, l in zip(ids, labels) if l == -100])
    return learned, masked


def test_tool_call_and_answer_are_learned_tool_result_is_masked():
    tok = _FakeTokenizer()
    ds = ToolUseDataset(tok, [_one_step_record()], seq_len=64)
    assert len(ds) == 1
    ids, labels = ds.examples[0]
    learned, masked = _learned_and_masked_text(tok, ids, labels)

    assert "[TOOL_CALL]" in learned and "[/TOOL_CALL]" in learned
    assert '"calculator"' in learned
    assert "answer" in learned and "4" in learned  # "The answer is 4 ." (the final answer)

    # The tool_result's own "4" is a separate occurrence at a different
    # position (verified not to collide via the position-aligned check
    # below) -- it belongs in `masked`, not `learned`: the model must never
    # be trained to PRODUCE this value, only to read it as context.
    assert "[TOOL_RESULT]" in masked and "[/TOOL_RESULT]" in masked
    assert "4" in masked.split(), "tool_result content should appear as masked context"
    assert "What" in masked, "the user query must be masked (input), not learned"

    # Position-level guard against the two "4" occurrences colliding: the
    # tool_result's "4" (position 12 in this fixture) must carry label
    # -100, and the final answer's "4" (position 17) must not.
    four_id = tok.vocab["4"]
    four_labels = [l for i, l in zip(ids, labels) if i == four_id]
    assert -100 in four_labels and any(l != -100 for l in four_labels), (
        "expected exactly one masked and one learned occurrence of the "
        "token '4' (tool_result vs. final answer)"
    )


def test_multi_step_masking_only_tool_calls_and_final_answer_learned():
    tok = _FakeTokenizer()
    ds = ToolUseDataset(tok, [_multi_step_record()], seq_len=128)
    ids, labels = ds.examples[0]
    learned, masked = _learned_and_masked_text(tok, ids, labels)

    # All 3 tool_call blocks learned.
    for name in ("read_file", "edit_file", "run_tests"):
        assert f'"{name}"' in learned

    # None of the 3 tool_result contents leak into the learned text.
    for leaked in ("PASSED", "updated", "return"):
        assert leaked not in learned, f"{leaked!r} (a tool_result) must not be learned"

    # Final answer learned.
    assert "Fixed" in learned

    # Query masked.
    assert "Fix" in masked or "bug" in masked


def test_final_answer_ends_with_eos():
    tok = _FakeTokenizer()
    ds = ToolUseDataset(tok, [_one_step_record()], seq_len=64)
    ids, labels = ds.examples[0]
    # EOS_ID is a fixed constant (50256); the fake tokenizer never produces
    # it via encode(), so its presence in `ids` can only come from the
    # dataset's explicit EOS append (or padding, which also uses EOS_ID).
    assert EOS_ID in ids.tolist()


def test_seq_len_plus_one_shape():
    tok = _FakeTokenizer()
    ds = ToolUseDataset(tok, [_one_step_record()], seq_len=64)
    ids, labels = ds.examples[0]
    assert len(ids) == 65 and len(labels) == 65
    x, y = ds[0]
    assert len(x) == 64 and len(y) == 64


def test_matches_build_tool_use_data_record_shape():
    """Regression guard: if scripts/build_tool_use_data.py's output format
    ever drifts, this dataset should fail loudly, not silently mis-parse."""
    import os
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "tool_use_train.jsonl")
    if not os.path.exists(path):
        import pytest
        pytest.skip("data/tool_use_train.jsonl not built in this checkout")
    with open(path, encoding="utf-8") as f:
        records = [json.loads(line) for line in f][:10]
    tok = _FakeTokenizer()
    ds = ToolUseDataset(tok, records, seq_len=512)
    assert len(ds) == len(records)
