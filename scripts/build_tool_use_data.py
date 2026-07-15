"""Build the tool-use training corpus for issue #110 (agentic coding FT, #97
Phase C).

Two categories:
  * **calculator** — single tool-call examples (query -> [TOOL_CALL] calculator
    [/TOOL_CALL] -> [TOOL_RESULT] -> answer). Real arithmetic, computed via
    ``hvn.tools.calculator`` so the recorded result is always correct.
  * **coding** — three-step examples (read_file -> edit_file -> run_tests)
    built from parameterized bug templates (wrong arithmetic operator, wrong
    comparison operator/boundary, off-by-one range). Every example is
    **actually executed** through a real sandboxed ``ToolExecutor`` (the same
    ``create_coding_registry`` issue #110 tools) during corpus generation --
    not fabricated -- so the recorded [TOOL_RESULT] blocks are authentic
    (real file contents, real "OK: updated", real pytest output), and
    generation self-validates: any template whose "fix" doesn't actually
    make the test pass raises instead of silently emitting a broken example.

Output: ``data/tool_use_train.jsonl`` — one JSON object per line::

    {"query": "...", "steps": [{"tool_call": {...}, "tool_result": "..."}, ...],
     "answer": "..."}

``steps`` is 1 entry for calculator, 3 for coding (read_file, edit_file,
run_tests, in order). ``finetune_edge_conversational.py``'s ``ToolUseDataset``
renders each record in EdgeHVN's trained chat format with the tool-call
segments loss-masked as learned targets and the tool-result segments masked
as environment-injected context (see that module for the exact masking).

Usage::

    python scripts/build_tool_use_data.py                 # write data/tool_use_train.jsonl
    python scripts/build_tool_use_data.py --n-calc 300 --n-coding-per-family 60
    python scripts/build_tool_use_data.py --validate       # re-load + sanity check the output
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import tempfile

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.tools import calculator, create_coding_registry, ToolExecutor  # noqa: E402


def _tool_call_json(name: str, arguments: dict) -> dict:
    return {"name": name, "arguments": arguments}


# ---------------------------------------------------------------------------
# Calculator examples (single tool call)
# ---------------------------------------------------------------------------

_CALC_OPS = ["+", "-", "*"]


def build_calculator_examples(n: int, rng: random.Random) -> list[dict]:
    examples = []
    for _ in range(n):
        a, b = rng.randint(1, 100), rng.randint(1, 100)
        op = rng.choice(_CALC_OPS)
        expression = f"{a}{op}{b}"
        result = calculator(expression)  # real computation, guaranteed correct
        query = f"What is {a} {op} {b}?"
        answer = f"{a} {op} {b} = {result}."
        examples.append({
            "query": query,
            "steps": [{
                "tool_call": _tool_call_json("calculator", {"expression": expression}),
                "tool_result": result,
            }],
            "answer": answer,
        })
    return examples


# ---------------------------------------------------------------------------
# Coding examples (read_file -> edit_file -> run_tests, 3 steps)
# ---------------------------------------------------------------------------
# Each family returns (filename, buggy_src, test_filename, test_src, old_text,
# new_text, query, answer). old_text/new_text are the edit_file arguments;
# applying them to buggy_src must make the recorded test pass (asserted below
# by actually running it, not by construction).

def _family_wrong_operator(rng: random.Random, idx: int) -> dict:
    func = f"compute_{idx}"
    a, b = rng.randint(1, 20), rng.randint(1, 20)
    wrong_op = rng.choice(["-", "*"])
    correct = a + b
    filename = f"calc_{idx}.py"
    buggy_src = f"def {func}(x, y):\n    return x {wrong_op} y\n"
    old_text = f"return x {wrong_op} y"
    new_text = "return x + y"
    test_filename = f"test_calc_{idx}.py"
    test_src = (f"from calc_{idx} import {func}\n\n"
                f"def test_{func}():\n    assert {func}({a}, {b}) == {correct}\n")
    query = (f"There's a bug in {filename}. {func}({a}, {b}) should return {correct} "
             f"but it doesn't. Please find and fix the bug, then run the tests to confirm.")
    answer = (f"Found it: {func} used `{wrong_op}` instead of `+`. Fixed the operator "
             f"and the tests now pass.")
    return dict(filename=filename, buggy_src=buggy_src, test_filename=test_filename,
                test_src=test_src, old_text=old_text, new_text=new_text,
                query=query, answer=answer)


def _family_wrong_comparison(rng: random.Random, idx: int) -> dict:
    func = f"check_{idx}"
    threshold = rng.randint(10, 100)
    filename = f"bounds_{idx}.py"
    buggy_src = f"def {func}(x):\n    return x > {threshold}\n"
    old_text = f"return x > {threshold}"
    new_text = f"return x >= {threshold}"
    test_filename = f"test_bounds_{idx}.py"
    test_src = (f"from bounds_{idx} import {func}\n\n"
                f"def test_{func}():\n    assert {func}({threshold}) == True\n")
    query = (f"There's a bug in {filename}. {func}({threshold}) should return True "
             f"(the boundary value {threshold} should count) but it currently returns "
             f"False. Please fix it and verify with the tests.")
    answer = (f"The comparison in {func} used `>` instead of `>=`, excluding the "
             f"boundary value {threshold}. Fixed and verified with the tests.")
    return dict(filename=filename, buggy_src=buggy_src, test_filename=test_filename,
                test_src=test_src, old_text=old_text, new_text=new_text,
                query=query, answer=answer)


def _family_off_by_one_range(rng: random.Random, idx: int) -> dict:
    func = f"make_list_{idx}"
    n = rng.randint(3, 12)
    filename = f"seq_{idx}.py"
    buggy_src = f"def {func}(n):\n    return list(range(n))\n"
    old_text = "return list(range(n))"
    new_text = "return list(range(n + 1))"
    test_filename = f"test_seq_{idx}.py"
    test_src = (f"from seq_{idx} import {func}\n\n"
                f"def test_{func}():\n    assert len({func}({n})) == {n + 1}\n")
    query = (f"There's a bug in {filename}. {func}({n}) should return a list with "
             f"{n + 1} elements (inclusive of {n}) but it returns one short. Please "
             f"find and fix the off-by-one bug, then run the tests.")
    answer = (f"{func} used `range(n)` instead of `range(n + 1)`, missing the last "
             f"element. Fixed and the tests now pass.")
    return dict(filename=filename, buggy_src=buggy_src, test_filename=test_filename,
                test_src=test_src, old_text=old_text, new_text=new_text,
                query=query, answer=answer)


_CODING_FAMILIES = [_family_wrong_operator, _family_wrong_comparison, _family_off_by_one_range]


def build_coding_examples(n_per_family: int, rng: random.Random) -> list[dict]:
    """Build coding examples, one family at a time. Each example is executed
    for real through a sandboxed ToolExecutor (create_coding_registry) so the
    recorded tool_results are authentic and the fix is verified to actually
    pass -- a template bug (wrong old_text, a fix that doesn't work) raises
    here at build time instead of silently producing a broken training example.
    """
    examples = []
    idx = 0
    for family_fn in _CODING_FAMILIES:
        for _ in range(n_per_family):
            idx += 1
            spec = family_fn(rng, idx)
            with tempfile.TemporaryDirectory() as sandbox:
                with open(os.path.join(sandbox, spec["filename"]), "w", encoding="utf-8") as f:
                    f.write(spec["buggy_src"])
                with open(os.path.join(sandbox, spec["test_filename"]), "w", encoding="utf-8") as f:
                    f.write(spec["test_src"])

                executor = ToolExecutor(create_coding_registry(sandbox))

                read_call = _tool_call_json("read_file", {"path": spec["filename"]})
                read_result = executor.execute({"name": "read_file", "arguments": read_call["arguments"]})
                assert spec["buggy_src"].strip() in read_result, (
                    f"family {family_fn.__name__} idx {idx}: read_file result mismatch"
                )

                edit_call = _tool_call_json("edit_file", {
                    "path": spec["filename"], "old_text": spec["old_text"], "new_text": spec["new_text"],
                })
                edit_result = executor.execute({"name": "edit_file", "arguments": edit_call["arguments"]})
                assert "OK" in edit_result, (
                    f"family {family_fn.__name__} idx {idx}: edit_file failed: {edit_result}"
                )

                run_call = _tool_call_json("run_tests", {})
                run_result = executor.execute({"name": "run_tests", "arguments": {}})
                assert "PASSED" in run_result, (
                    f"family {family_fn.__name__} idx {idx}: fix did not make tests pass: {run_result}"
                )

            examples.append({
                "query": spec["query"],
                "steps": [
                    {"tool_call": read_call, "tool_result": _strip_result_tags(read_result)},
                    {"tool_call": edit_call, "tool_result": _strip_result_tags(edit_result)},
                    {"tool_call": run_call, "tool_result": _strip_result_tags(run_result)},
                ],
                "answer": spec["answer"],
            })
    return examples


def _strip_result_tags(result_block: str) -> str:
    """ToolExecutor.execute already wraps results in [TOOL_RESULT]...[/TOOL_RESULT]
    -- strip that here since ToolUseDataset re-wraps consistently for every
    step (including the single-call calculator examples, whose tool_result is
    the bare calculator() return value, not pre-wrapped)."""
    from hvn.tools import TOOL_TOKENS
    s = result_block.strip()
    if s.startswith(TOOL_TOKENS["tool_result_start"]):
        s = s[len(TOOL_TOKENS["tool_result_start"]):]
    if s.endswith(TOOL_TOKENS["tool_result_end"]):
        s = s[: -len(TOOL_TOKENS["tool_result_end"])]
    return s.strip()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=str, default="data/tool_use_train.jsonl")
    ap.add_argument("--n-calc", type=int, default=300)
    ap.add_argument("--n-coding-per-family", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--validate", action="store_true",
                    help="re-load the written file and sanity-check record shape")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    print(f"Building {args.n_calc} calculator examples...")
    calc = build_calculator_examples(args.n_calc, rng)

    print(f"Building {args.n_coding_per_family} x {len(_CODING_FAMILIES)} coding examples "
          f"(each executed + verified through a real sandboxed ToolExecutor)...")
    coding = build_coding_examples(args.n_coding_per_family, rng)

    records = calc + coding
    rng.shuffle(records)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print(f"Wrote {len(records)} examples ({len(calc)} calculator, {len(coding)} coding) "
          f"to {args.out}")

    if args.validate:
        with open(args.out, encoding="utf-8") as f:
            loaded = [json.loads(line) for line in f]
        assert len(loaded) == len(records)
        for r in loaded:
            assert "query" in r and "steps" in r and "answer" in r
            assert len(r["steps"]) >= 1
            for step in r["steps"]:
                assert "tool_call" in step and "tool_result" in step
                assert "name" in step["tool_call"] and "arguments" in step["tool_call"]
        print(f"--validate: OK ({len(loaded)} records, shape verified)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
