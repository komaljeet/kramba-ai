"""
Reasoning Evaluation Harness (Issue #42)
========================================

Auto-graded **with-think vs. no-think** A/B for HVN on held-out multi-step
arithmetic.  Because the reasoning domain is deterministic, we can exact-match
the model's final integer against ground truth and report the *accuracy delta*
between:

  * **think**   — prompt ends at ``Answer: `` so the model may emit a
    ``[THINK] ... [/THINK]`` chain-of-thought before stating the answer.
  * **nothink** — prompt is primed with ``Answer: The answer is `` so the model
    must commit to the answer immediately, with no room to reason.

Same model, same held-out problems — the only difference is whether reasoning
is allowed.  A positive ``think - nothink`` delta is the issue's success
criterion: the ``[THINK]`` blocks carry reasoning that *measurably helps*.

Usage::

    python -m eval.reasoning --checkpoint hvn_checkpoint.pt
"""

from __future__ import annotations

import argparse
import random
import re
from typing import Any, Dict, List, Optional

import torch

from hvn.core import HamiltonianVortexNetwork
from hvn.reasoning import (
    ANSWER_PREFIX,
    generate_arithmetic_problem,
)
from train import CharTokenizer


# ---------------------------------------------------------------------------
# Answer extraction
# ---------------------------------------------------------------------------

_ANSWER_RE = re.compile(re.escape(ANSWER_PREFIX) + r"(-?\d+)")
_INT_RE = re.compile(r"-?\d+")


def extract_answer(text: str) -> Optional[str]:
    """Pull the final integer answer out of generated text.

    Prefers the number right after the canonical ``"The answer is "`` phrase;
    falls back to the last integer that appears outside a ``[THINK]`` block.
    Returns ``None`` if no integer is found.
    """
    # Strip any reasoning block so intermediate step numbers can't be mistaken
    # for the answer in the fallback path.
    from hvn.tools import ToolCallParser
    answer_region = ToolCallParser.strip_tool_blocks(text)

    m = _ANSWER_RE.search(answer_region) or _ANSWER_RE.search(text)
    if m:
        return m.group(1)
    ints = _INT_RE.findall(answer_region)
    if ints:
        return ints[-1]
    return None


# ---------------------------------------------------------------------------
# Held-out problem set
# ---------------------------------------------------------------------------

def make_eval_problems(num_problems: int, seed: int = 999) -> List[Dict[str, Any]]:
    """Generate a reproducible held-out set of arithmetic problems.

    Uses a different default seed from the training data (42) so evaluation
    problems are not memorised training instances.
    """
    rng = random.Random(seed)
    return [generate_arithmetic_problem(rng) for _ in range(num_problems)]


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_reasoning(
    model: HamiltonianVortexNetwork,
    tokenizer: CharTokenizer,
    device: torch.device,
    problems: List[Dict[str, Any]],
    mode: str = "think",
    max_new_tokens: int = 64,
    temperature: float = 0.0,
    top_k: int = 1,
) -> Dict[str, Any]:
    """Exact-match accuracy on held-out arithmetic, with or without reasoning.

    Parameters
    ----------
    mode : {"think", "nothink"}
        ``think`` lets the model emit a ``[THINK]`` block; ``nothink`` primes
        the answer prefix so it must answer immediately.
    temperature, top_k :
        Default to greedy (temp 0 / top_k 1) for a deterministic grade.

    Returns
    -------
    dict with ``accuracy``, ``n_correct``, ``n``, ``mode`` and per-example
    ``results``.
    """
    assert mode in ("think", "nothink"), mode
    model.eval()

    results = []
    n_correct = 0

    # Greedy decode: generate_incremental treats temperature<=0 by using top_k=1.
    gen_top_k = 1 if temperature <= 0.0 else top_k
    gen_temp = 1.0 if temperature <= 0.0 else temperature

    for problem in problems:
        prompt = f"Question: {problem['question']}\nAnswer: "
        if mode == "nothink":
            prompt += ANSWER_PREFIX  # force an immediate answer, no reasoning

        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [0]

        generated = model.generate_incremental(
            prompt_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=gen_temp,
            top_k=gen_top_k,
            device=device,
        )
        completion = tokenizer.decode(generated[len(ids):])
        # For nothink mode the answer prefix lives in the prompt, so re-attach
        # it before extraction.
        search_text = (ANSWER_PREFIX + completion) if mode == "nothink" else completion

        predicted = extract_answer(search_text)
        correct = predicted is not None and predicted == problem["answer"]
        n_correct += int(correct)

        results.append({
            "question": problem["question"],
            "expected": problem["answer"],
            "predicted": predicted,
            "correct": correct,
            "completion": completion,
        })

    n = len(problems)
    return {
        "mode": mode,
        "accuracy": n_correct / n if n else 0.0,
        "n_correct": n_correct,
        "n": n,
        "results": results,
    }


def run_reasoning_ab(
    model: HamiltonianVortexNetwork,
    tokenizer: CharTokenizer,
    device: torch.device,
    num_problems: int = 50,
    seed: int = 999,
    max_new_tokens: int = 64,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the with-think vs. no-think A/B and print the accuracy delta."""
    problems = make_eval_problems(num_problems, seed=seed)

    think = evaluate_reasoning(
        model, tokenizer, device, problems, mode="think",
        max_new_tokens=max_new_tokens,
    )
    nothink = evaluate_reasoning(
        model, tokenizer, device, problems, mode="nothink",
        max_new_tokens=max_new_tokens,
    )
    delta = think["accuracy"] - nothink["accuracy"]

    if verbose:
        print("\n" + "=" * 60)
        print("  Reasoning A/B — with-think vs. no-think (Issue #42)")
        print("=" * 60)
        print(f"  Held-out arithmetic problems: {think['n']}")
        print(f"  {'Mode':<12}{'Accuracy':>12}{'Correct':>12}")
        print("  " + "-" * 34)
        print(f"  {'think':<12}{think['accuracy']:>11.1%}{think['n_correct']:>12}")
        print(f"  {'nothink':<12}{nothink['accuracy']:>11.1%}{nothink['n_correct']:>12}")
        print("  " + "-" * 34)
        print(f"  {'delta':<12}{delta:>+11.1%}")
        verdict = "PASS — reasoning helps" if delta > 0 else (
            "TIE" if delta == 0 else "REGRESSION — reasoning hurts")
        print(f"  Verdict: {verdict}")

        # A couple of qualitative examples.
        print("\n  Sample think-mode completions:")
        for r in think["results"][:3]:
            comp = " ".join(r["completion"].split())[:120]
            print(f"    Q: {r['question']}")
            print(f"       -> pred={r['predicted']} expected={r['expected']} "
                  f"{'OK' if r['correct'] else 'X'}")
            print(f"       out: {comp}")

    return {"think": think, "nothink": nothink, "delta": delta}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_model(checkpoint: str, device: torch.device) -> HamiltonianVortexNetwork:
    ckpt = torch.load(checkpoint, weights_only=False, map_location=device)
    model = HamiltonianVortexNetwork(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=ckpt["num_layers"],
        n_heads=ckpt.get("n_heads", 4),
        bond_dimension=ckpt.get("bond_dimension", ckpt.get("bond_dim", 32)),
        chunk_size=ckpt.get("chunk_size", 64),
        n_threads=ckpt.get("n_threads", 1),
        use_deltanet=ckpt.get("use_deltanet", False),
        deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        weightless=ckpt.get("weightless", False),
        weightless_seed=ckpt.get("weightless_seed", 1234),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(description="Reasoning (chain-of-thought) A/B eval for HVN")
    parser.add_argument("--checkpoint", type=str, default="hvn_checkpoint.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer JSON (default: auto-detect from checkpoint)")
    parser.add_argument("--num_problems", type=int, default=50,
                        help="Number of held-out problems to evaluate")
    parser.add_argument("--seed", type=int, default=999,
                        help="Seed for the held-out problem set")
    parser.add_argument("--max_new_tokens", type=int, default=64,
                        help="Max tokens generated per problem")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    tok_path = args.tokenizer or args.checkpoint.replace(".pt", "_tokenizer.json")
    tokenizer = CharTokenizer.load(tok_path)
    print(f"Tokenizer loaded: {tokenizer.vocab_size} tokens")

    model = _load_model(args.checkpoint, device)
    print(f"Model loaded from {args.checkpoint}")

    run_reasoning_ab(
        model, tokenizer, device,
        num_problems=args.num_problems,
        seed=args.seed,
        max_new_tokens=args.max_new_tokens,
    )


if __name__ == "__main__":
    main()
