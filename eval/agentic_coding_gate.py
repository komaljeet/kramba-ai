"""Agentic coding gate (issue #110 / #97 Phase C) — the real "can it do
agentic coding" measurement, beyond the recall gate (#97) and the plain
chat-quality checks (#108).

Builds a small temp-dir mock repo with a seeded one-line bug + a pytest
test that currently fails against it (same shape as
``scripts/build_tool_use_data.py``'s coding-example bug templates, but
with a fresh function/file name and larger id so it isn't a literal
training example — still the same learned *family* of bug, which is the
honest thing to test: whether the model generalizes the skill, not
whether it memorized one example). Drives the checkpoint through the real
``EdgeToolCallingSession`` (not the training-time ``ToolUseDataset``) with
the real sandboxed coding tool registry.

**Pass bar (required):** the model emits a well-formed
read_file -> edit_file -> run_tests tool-call sequence that actually
executes (real file I/O, real subprocess) without malformed-JSON/parse
failures.

**Stretch signal (reported, not required):** the bug is actually fixed
(the test suite passes afterward). Per the project's calibrated
expectations for an 83M model (docs/state.md's 270M-milestone framing):
correct tool-call format-following is a learnable format skill; picking
the *right* fix is a reasoning skill this scale may not reliably have.

Usage::

    python -m eval.agentic_coding_gate --checkpoint checkpoints/hybrid_tooluse_v1.pt
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch  # noqa: E402

from hvn.edge.tool_session import EdgeToolCallingSession  # noqa: E402
from hvn.tools import ToolExecutor, create_coding_registry  # noqa: E402


def _build_mock_repo(sandbox: str) -> dict:
    """A fresh wrong-arithmetic-operator bug -- same family as
    scripts/build_tool_use_data.py's _family_wrong_operator, different
    name/numbers so this isn't a literal training example."""
    filename = "gate_calc.py"
    test_filename = "test_gate_calc.py"
    buggy_src = "def total(x, y):\n    return x * y\n"
    test_src = ("from gate_calc import total\n\n"
               "def test_total():\n    assert total(4, 5) == 9\n")
    with open(os.path.join(sandbox, filename), "w", encoding="utf-8") as f:
        f.write(buggy_src)
    with open(os.path.join(sandbox, test_filename), "w", encoding="utf-8") as f:
        f.write(test_src)
    query = ("There's a bug in gate_calc.py. total(4, 5) should return 9 but it "
             "doesn't. Please find and fix the bug, then run the tests to confirm.")
    return {"filename": filename, "query": query}


def run_gate(checkpoint_path: str, device: torch.device | None = None,
            max_new_tokens: int = 300, temperature: float = 0.7,
            repetition_penalty: float = 1.2) -> dict:
    from train_edge_conversational import load_checkpoint_for_resume
    from transformers import GPT2Tokenizer

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step, _meta = load_checkpoint_for_resume(checkpoint_path, device)
    model.eval()
    print(f"Loaded {checkpoint_path} (step {step}), block_plan={model.block_plan}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    with tempfile.TemporaryDirectory() as sandbox:
        repo = _build_mock_repo(sandbox)
        executor = ToolExecutor(create_coding_registry(sandbox))

        session = EdgeToolCallingSession(
            model, tokenizer, executor, device=device,
            max_new_tokens=max_new_tokens, max_tool_calls=5,
            temperature=temperature, top_k=10, repetition_penalty=repetition_penalty,
        )
        result = session.run(repo["query"])

        call_names = [c["name"] for c in result["tool_calls"]]
        expected_order = ["read_file", "edit_file", "run_tests"]
        well_formed_sequence = call_names == expected_order

        # Independently verify the pass/fail bug-fix bar by reading the
        # sandbox's ACTUAL current state after the session ran -- not by
        # trusting the model's run_tests transcript (the model could
        # hallucinate a PASSED result never actually produced, though
        # EdgeToolCallingSession always injects the REAL executor output as
        # context, so this should already agree; verifying independently
        # anyway is the "measure, don't guess" discipline this project
        # otherwise follows throughout).
        verify_executor = ToolExecutor(create_coding_registry(sandbox))
        final_test_result = verify_executor.execute({"name": "run_tests", "arguments": {}})
        bug_actually_fixed = "PASSED" in final_test_result

    return {
        "pass": well_formed_sequence,
        "stretch_pass": bug_actually_fixed,
        "tool_call_sequence": call_names,
        "num_tool_calls": result["num_tool_calls"],
        "generated_text": result["text"],
        "answer": result["answer"],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--max-new-tokens", type=int, default=300)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--repetition-penalty", type=float, default=1.2)
    args = ap.parse_args()

    result = run_gate(args.checkpoint, max_new_tokens=args.max_new_tokens,
                      temperature=args.temperature,
                      repetition_penalty=args.repetition_penalty)

    print("\n" + "=" * 70)
    print("AGENTIC CODING GATE (issue #110)")
    print("=" * 70)
    print(f"Tool-call sequence: {result['tool_call_sequence']}")
    print(f"Number of tool calls: {result['num_tool_calls']}")
    print(f"\nGenerated text:\n{result['generated_text']}")
    print(f"\nFinal answer: {result['answer']}")
    print("\n" + "-" * 70)
    print(f"PASS (well-formed read_file->edit_file->run_tests sequence): "
          f"{'YES' if result['pass'] else 'NO'}")
    print(f"STRETCH (bug actually fixed): "
          f"{'YES' if result['stretch_pass'] else 'NO'}")
    print("=" * 70)

    return 0 if result["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
