"""
Test suite for Issue #42 — real reasoning/thinking capability.

Makes the ``[THINK]`` / ``[/THINK]`` tokens carry real chain-of-thought:
synthetic multi-step arithmetic problems paired with step-by-step reasoning
traces (``hvn/reasoning.py``), wired into ``train.py`` (``--reasoning_training``)
and graded by an A/B harness (``eval/reasoning.py``).  This is a data +
training + decoding-convention feature — the HVN architecture is unchanged and
O(1) / no-KV inference is preserved.

Checks (fast, mostly CPU-only):
  1. Generated arithmetic answers are correct (safe-eval vs. stated answer).
  2. format_reasoning_example produces a well-formed [THINK] block that parses
     and strips back to the answer.
  3. generate_reasoning_training_data returns well-formed {prompt,response} dicts
     whose [THINK] blocks round-trip through the parser.
  4. The --hide_thinking decoding path strips reasoning but keeps the answer.
  5. Tiny end-to-end smoke: the reasoning tokens actually train (loss drops).

Run:  python test_issue42.py
"""

import random

import torch

# --- repo-root path bootstrap (added during reorg, issue #48) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---
from hvn.reasoning import (
    ANSWER_PREFIX,
    safe_eval_int,
    generate_arithmetic_problem,
    format_reasoning_example,
    generate_reasoning_training_data,
    THINK_START,
    THINK_END,
)
from hvn.tools import ToolCallParser


def test_arithmetic_ground_truth():
    """[1/5] Every generated problem's stated answer matches its expression."""
    rng = random.Random(0)
    for _ in range(500):
        p = generate_arithmetic_problem(rng)
        assert safe_eval_int(p["expr"]) == int(p["answer"]), (
            f"answer mismatch: {p['expr']} -> {p['answer']}"
        )
        assert p["steps"], "problem has no reasoning steps"
        # The last reasoning step must reveal the final answer.
        assert p["answer"] in p["steps"][-1], (
            f"final step {p['steps'][-1]!r} does not contain answer {p['answer']}"
        )
    print("[1/5] arithmetic ground truth correct over 500 problems ... OK")


def test_think_block_roundtrip():
    """[2/5] format_reasoning_example emits a parseable [THINK] block."""
    rng = random.Random(1)
    p = generate_arithmetic_problem(rng)

    with_think = format_reasoning_example(p, with_think=True)
    assert THINK_START in with_think and THINK_END in with_think
    assert f"{ANSWER_PREFIX}{p['answer']}." in with_think

    # parse_thinking recovers the reasoning steps.
    blocks = ToolCallParser.parse_thinking(with_think)
    assert len(blocks) == 1, f"expected 1 think block, got {len(blocks)}"
    for step in p["steps"]:
        assert step in blocks[0], f"step {step!r} missing from parsed think block"

    # Stripping the think block leaves the user-facing answer.
    stripped = ToolCallParser.strip_tool_blocks(with_think)
    assert THINK_START not in stripped and THINK_END not in stripped
    assert f"{ANSWER_PREFIX}{p['answer']}." in stripped

    # The no-think rendering has no block at all.
    no_think = format_reasoning_example(p, with_think=False)
    assert THINK_START not in no_think
    assert f"{ANSWER_PREFIX}{p['answer']}." in no_think
    print("[2/5] [THINK] block round-trips (parse + strip) ... OK")


def test_training_data_shape():
    """[3/5] generate_reasoning_training_data yields well-formed pairs."""
    data = generate_reasoning_training_data(num_examples=64, seed=42)
    assert len(data) == 64
    for ex in data:
        assert set(ex.keys()) == {"prompt", "response"}
        assert ex["prompt"].startswith("Question: ")
        assert ex["prompt"].rstrip().endswith("Answer:")
        # Response carries a reasoning trace that parses to >=1 step.
        assert THINK_START in ex["response"] and THINK_END in ex["response"]
        blocks = ToolCallParser.parse_thinking(ex["response"])
        assert blocks and blocks[0].strip(), "empty reasoning block in training data"
        assert ANSWER_PREFIX in ex["response"]

    # Reproducible for a fixed seed.
    again = generate_reasoning_training_data(num_examples=64, seed=42)
    assert [e["prompt"] for e in data] == [e["prompt"] for e in again]
    print("[3/5] training-data shape + determinism ... OK")


def test_hide_thinking_decoding():
    """[4/5] The --hide_thinking convention strips reasoning, keeps the answer."""
    # Simulate a model generation that reasons then answers.
    generated = (
        f"{THINK_START}\n12 + 7 = 19\n19 - 5 = 14\n{THINK_END} "
        f"{ANSWER_PREFIX}14."
    )
    shown = ToolCallParser.strip_tool_blocks(generated)
    assert "12 + 7 = 19" not in shown, "reasoning leaked into shown answer"
    assert THINK_START not in shown and THINK_END not in shown
    assert f"{ANSWER_PREFIX}14." in shown
    print("[4/5] --hide_thinking strips reasoning, keeps answer ... OK")


def test_tiny_training_smoke():
    """[5/5] Reasoning tokens actually train — tiny overfit drives loss down."""
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from hvn.core import HamiltonianVortexNetwork
    from train import CharTokenizer, TextDataset

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build a small corpus from reasoning examples so the [THINK] tokens appear.
    examples = generate_reasoning_training_data(num_examples=32, seed=7)
    corpus = "\n".join(e["prompt"] + e["response"] for e in examples)
    tokenizer = CharTokenizer(corpus, add_tool_tokens=True)
    assert THINK_START in tokenizer.special_tokens
    assert THINK_END in tokenizer.special_tokens

    token_ids = tokenizer.encode(corpus)
    # Sanity: the special think tokens survive encoding as single ids.
    assert tokenizer.special_tokens[THINK_START] in token_ids

    dataset = TextDataset(token_ids, seq_len=48)
    loader = DataLoader(dataset, batch_size=8, shuffle=True, drop_last=False)

    torch.manual_seed(0)
    model = HamiltonianVortexNetwork(
        vocab_size=tokenizer.vocab_size, d_model=64, num_layers=2,
        n_heads=2, bond_dimension=16, dropout=0.0,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    criterion = nn.CrossEntropyLoss()

    model.train()
    losses = []
    step = 0
    for _epoch in range(6):
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            logits = model(bx)
            loss = criterion(logits.reshape(-1, model.vocab_size), by.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            losses.append(loss.item())
            step += 1
            if step >= 40:
                break
        if step >= 40:
            break

    first = sum(losses[:3]) / 3
    last = sum(losses[-3:]) / 3
    assert last < first, f"loss did not decrease ({first:.3f} -> {last:.3f})"
    print(f"[5/5] tiny reasoning training smoke: loss {first:.3f} -> {last:.3f} ... OK")


if __name__ == "__main__":
    print("=" * 72)
    print("Issue #42 — real reasoning/thinking capability acceptance suite")
    print("=" * 72)
    test_arithmetic_ground_truth()
    test_think_block_roundtrip()
    test_training_data_shape()
    test_hide_thinking_decoding()
    test_tiny_training_smoke()
    print("=" * 72)
    print("ALL TESTS PASSED")
