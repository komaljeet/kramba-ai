"""scripts/validate_looped_ft.py — Issue #122 stage 2 validation gate.

Does the loop-aware continued-FT (continue_ft_looped.py) actually fix what
the no-retrain sweep (scripts/sweep_looped_early_exit.py) proved was broken:
looping the frozen checkpoint always collapsed exact-token recall to 0%
whenever it fired more than once? Tests the FT'd checkpoint at FIXED loop
counts N=1,2,4 (entropy_threshold=-1.0 forces exactly N loops every time,
no early exit -- isolates the "does looping still corrupt recall" question
from any entropy-threshold tuning question) via the near-window synthetic
recall probe, and reports the same measurement on the frozen baseline
checkpoint for direct comparison.

Usage:
    python scripts/validate_looped_ft.py --n-trials 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402
from eval.grounded_qa import set_attention_window  # noqa: E402
from eval.synthetic_recall_probe import build_prompt, whole_word_hit  # noqa: E402
from scripts.sweep_looped_early_exit import generate_incremental  # noqa: E402
from hvn.edge.looped import enable_looped_early_exit  # noqa: E402


def measure_fixed_n_recall(model, tok, device, n_trials, seed, N):
    enable_looped_early_exit(model, max_loops=N, entropy_threshold=-1.0)  # never exits early -> always exactly N loops
    set_attention_window(model, None)
    rng = torch.Generator().manual_seed(seed)
    hits = 0
    for _ in range(n_trials):
        prompt, correct = build_prompt(rng, n_distractors=0)
        answer = generate_incremental(model, tok, device, prompt)
        hits += whole_word_hit(answer, correct)
    return hits / n_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ft-checkpoint", default="checkpoints/hybrid_looped_ft/hybrid_looped_step2000.pt")
    ap.add_argument("--baseline-checkpoint", default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/looped_early_exit/validate_looped_ft.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    results = {}
    for label, ckpt_path in [("ft", args.ft_checkpoint), ("baseline", args.baseline_checkpoint)]:
        print(f"\n=== {label}: {ckpt_path} ===")
        model, step, _ = load_checkpoint_for_resume(ckpt_path, device)
        model.eval()
        print(f"Loaded (step {step})")
        results[label] = {"checkpoint": ckpt_path, "step": step, "by_n": {}}
        for N in args.n_values:
            acc = measure_fixed_n_recall(model, tok, device, args.n_trials, args.seed, N)
            print(f"  N={N}: near-window recall = {acc:.1%} ({int(acc*args.n_trials)}/{args.n_trials})")
            results[label]["by_n"][str(N)] = acc
        del model
        torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nWrote {args.output}")

    print("\n=== Verdict ===")
    ft_fixed = all(results["ft"]["by_n"][str(N)] > 0 for N in args.n_values if N > 1)
    base_collapsed = all(results["baseline"]["by_n"][str(N)] == 0 for N in args.n_values if N > 1)
    print(f"Baseline collapses to 0% whenever N>1 (matches original sweep): {base_collapsed}")
    print(f"FT checkpoint stays nonzero at every N>1 tested: {ft_fixed}")
    if ft_fixed and base_collapsed:
        print("-> Loop-aware continue-FT FIXES the recall collapse.")
    elif not ft_fixed:
        print("-> Loop-aware continue-FT does NOT fix the recall collapse.")
    else:
        print("-> Baseline didn't reproduce its known collapse in this run -- inconclusive, check n-trials/noise.")


if __name__ == "__main__":
    main()
