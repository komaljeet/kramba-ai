"""
Reasoning / Chain-of-Thought Training Data for HVN (Issue #42)
=============================================================

Turns the ``[THINK]`` / ``[/THINK]`` special tokens (defined in
:mod:`hvn.tools`) from a *format convention* into a real capability: this
module generates synthetic **multi-step arithmetic** problems paired with
explicit step-by-step reasoning traces, so the model can be trained to emit
useful chain-of-thought inside ``[THINK]`` blocks before committing to an
answer.

Like tool-calling (#24) and instruction-tuning (#29), this is a **data +
training + decoding-convention** feature — the HVN architecture is unchanged
and O(1) / no-KV-cache inference is preserved.

Reasoning domain — **multi-step arithmetic** — is deterministic, so an
answer can be auto-graded (exact match), which lets ``eval/reasoning.py`` run
a clean *with-think vs. no-think* accuracy A/B.

Training-string layout (mirrors the tool-calling data shape consumed by
``train.py``)::

    Question: 12 + 7 - 5
    Answer: [THINK]
    12 + 7 = 19
    19 - 5 = 14
    [/THINK] The answer is 14.

Public API
----------
- :func:`generate_arithmetic_problem` — one random problem dict.
- :func:`format_reasoning_example`    — render a problem as a training string.
- :func:`generate_reasoning_training_data` — a list of ``{"prompt","response"}``.
"""

from __future__ import annotations

import random
import re
from typing import Any, Dict, List

from hvn.tools import TOOL_TOKENS

THINK_START = TOOL_TOKENS["think_start"]
THINK_END = TOOL_TOKENS["think_end"]

# The fixed suffix that states the final answer.  Kept as a constant so the
# eval harness and tests can locate / parse the answer consistently.
ANSWER_PREFIX = "The answer is "


# ---------------------------------------------------------------------------
# Safe integer arithmetic evaluation (for ground-truth checking)
# ---------------------------------------------------------------------------

_SAFE_EXPR_RE = re.compile(r"^[0-9+\-*() ]+$")


def safe_eval_int(expr: str) -> int:
    """Evaluate a pure ``+ - *`` integer expression, safely.

    Only digits, ``+``, ``-``, ``*``, parentheses and spaces are allowed —
    this is used to compute / verify ground-truth answers, never to run model
    output.  Raises ``ValueError`` on anything unexpected.
    """
    if not _SAFE_EXPR_RE.match(expr):
        raise ValueError(f"unsafe or non-integer expression: {expr!r}")
    result = eval(expr, {"__builtins__": {}}, {})  # noqa: S307 (charset-restricted)
    if not isinstance(result, int):
        raise ValueError(f"expression did not evaluate to an int: {expr!r}")
    return result


# ---------------------------------------------------------------------------
# Problem generators — each returns (question, expr, steps)
# ---------------------------------------------------------------------------
#
# ``question`` is what the model is prompted with; ``expr`` is the canonical
# pure-arithmetic expression (safe-eval'd to get / check the answer); ``steps``
# are the natural chain-of-thought lines placed inside the [THINK] block.  Each
# family has genuine multi-step structure, so the reasoning trace carries real
# information (intermediate results) rather than restating the question.


def _p_chain_add_sub(rng: random.Random):
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    c = rng.randint(1, a + b)  # keep the answer non-negative
    s1 = a + b
    question = f"{a} + {b} - {c}"
    steps = [f"{a} + {b} = {s1}", f"{s1} - {c} = {s1 - c}"]
    return question, f"{a} + {b} - {c}", steps


def _p_mul_then_add(rng: random.Random):
    a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
    s1 = a * b
    question = f"{a} * {b} + {c}"
    steps = [f"{a} * {b} = {s1}", f"{s1} + {c} = {s1 + c}"]
    return question, f"{a} * {b} + {c}", steps


def _p_paren_add_then_mul(rng: random.Random):
    a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(2, 5)
    s1 = a + b
    question = f"({a} + {b}) * {c}"
    steps = [f"{a} + {b} = {s1}", f"{s1} * {c} = {s1 * c}"]
    return question, f"({a} + {b}) * {c}", steps


def _p_order_of_ops(rng: random.Random):
    # a + b * c  — tests that multiplication binds before addition.
    a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
    s1 = b * c
    question = f"{a} + {b} * {c}"
    steps = [
        f"multiply first: {b} * {c} = {s1}",
        f"then add: {a} + {s1} = {a + s1}",
    ]
    return question, f"{a} + {b} * {c}", steps


def _p_word_shopping(rng: random.Random):
    name = rng.choice(["Sam", "Ava", "Leo", "Mia", "Kai", "Zoe"])
    item = rng.choice(["apples", "coins", "marbles", "stickers", "cards"])
    a, b = rng.randint(1, 9), rng.randint(1, 9)
    c = rng.randint(1, a + b)  # keep the answer non-negative
    s1 = a + b
    question = (
        f"{name} had {a} {item}, then got {b} more, then gave away {c}. "
        f"How many {item} does {name} have now?"
    )
    steps = [
        f"start with {a}, got {b} more: {a} + {b} = {s1}",
        f"gave away {c}: {s1} - {c} = {s1 - c}",
    ]
    return question, f"{a} + {b} - {c}", steps


def _p_word_groups(rng: random.Random):
    thing = rng.choice(["boxes", "shelves", "crates", "bags", "baskets"])
    unit = rng.choice(["books", "cans", "toys", "eggs", "pens"])
    a, b, c = rng.randint(1, 9), rng.randint(1, 9), rng.randint(1, 9)
    s1 = a * b
    question = (
        f"There are {a} {thing} with {b} {unit} each, plus {c} extra {unit}. "
        f"How many {unit} in total?"
    )
    steps = [
        f"{a} {thing} of {b}: {a} * {b} = {s1}",
        f"add the {c} extra: {s1} + {c} = {s1 + c}",
    ]
    return question, f"{a} * {b} + {c}", steps


_PROBLEM_FAMILIES = [
    _p_chain_add_sub,
    _p_mul_then_add,
    _p_paren_add_then_mul,
    _p_order_of_ops,
    _p_word_shopping,
    _p_word_groups,
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_arithmetic_problem(rng: random.Random | None = None) -> Dict[str, Any]:
    """Generate one random multi-step arithmetic problem.

    Returns
    -------
    problem : dict
        Keys:
          - ``question`` : str — the prompt text (expression or word problem).
          - ``expr``     : str — canonical pure-arithmetic expression, safe to
            :func:`safe_eval_int` to obtain / verify the answer.
          - ``steps``    : list[str] — chain-of-thought lines for the [THINK] block.
          - ``answer``   : str — the ground-truth integer answer (as a string).
    """
    if rng is None:
        rng = random.Random()
    family = rng.choice(_PROBLEM_FAMILIES)
    question, expr, steps = family(rng)
    answer = safe_eval_int(expr)
    return {
        "question": question,
        "expr": expr,
        "steps": steps,
        "answer": str(answer),
    }


def _format_prompt(problem: Dict[str, Any]) -> str:
    """The prompt half: ``Question: ...\\nAnswer: `` (model continues from here)."""
    return f"Question: {problem['question']}\nAnswer: "


def _format_response(problem: Dict[str, Any], with_think: bool = True) -> str:
    """The continuation half the model is trained (or graded) to produce."""
    answer_line = f"{ANSWER_PREFIX}{problem['answer']}."
    if not with_think:
        return answer_line
    steps = "\n".join(problem["steps"])
    return f"{THINK_START}\n{steps}\n{THINK_END} {answer_line}"


def format_reasoning_example(problem: Dict[str, Any], with_think: bool = True) -> str:
    """Render a problem as a full training string (prompt + response).

    With ``with_think=True`` (default) the response contains a
    ``[THINK] ... [/THINK]`` reasoning trace before the answer.  With
    ``with_think=False`` it emits the answer directly — used only as the eval
    baseline, not for training.
    """
    return _format_prompt(problem) + _format_response(problem, with_think=with_think)


def generate_reasoning_training_data(
    num_examples: int = 200,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Generate synthetic chain-of-thought arithmetic training examples.

    Each example is a ``{"prompt": ..., "response": ...}`` dict — the same shape
    ``train.py`` already consumes for tool-calling data — where ``response``
    contains a ``[THINK] ... [/THINK]`` reasoning trace followed by the answer.

    Parameters
    ----------
    num_examples : int
        Number of examples to generate.
    seed : int
        Random seed for reproducibility.
    """
    rng = random.Random(seed)
    examples: List[Dict[str, str]] = []
    for _ in range(num_examples):
        problem = generate_arithmetic_problem(rng)
        examples.append({
            "prompt": _format_prompt(problem),
            "response": _format_response(problem, with_think=True),
        })
    return examples


__all__ = [
    "THINK_START",
    "THINK_END",
    "ANSWER_PREFIX",
    "safe_eval_int",
    "generate_arithmetic_problem",
    "format_reasoning_example",
    "generate_reasoning_training_data",
]
