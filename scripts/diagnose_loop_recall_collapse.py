"""scripts/diagnose_loop_recall_collapse.py — WHY does looping kill recall? (Issue #122)

Four training recipes all failed identically: recall -> 0% the instant N>1, while
general loss improved fine. That pattern says the problem may not be trainable at all --
it may be that looping DESTROYS the information exact recall depends on, in which case no
training objective can recover it.

Mechanism hypothesis. This project's own numbers (docs/state.md) say exact recall comes
from the 2 windowed-attention layers: pure DeltaNet scores 0/5 on grounded QA, the hybrid
scores 35.5%. Exact copying works by a sharp query<->key match against a token's
representation in context. On loop 2, those attention layers no longer receive token
embeddings -- they receive hidden states already transformed by all 12 layers, so the
"this token is literally <Blenthar>" signal has been smeared into an abstract
representation. Copying then has nothing sharp to match against.

Two candidate fixes follow, BOTH of which standard looped transformers do and this
implementation does NOT (hvn/edge/looped.py:104 is a plain `x = h`):

  A. INPUT INJECTION -- re-add the original token embedding every iteration
     (`h <- Blocks(h + x0)`), as in Universal Transformer / the Looped Transformer
     line. Keeps token identity anchored across loops instead of letting it drift.
  B. DON'T LOOP THE ATTENTION LAYERS -- run them once (loop 1) to establish the
     recall pathway, and iterate only the 10 DeltaNet layers on loops 2+.

The point of this script: both are INFERENCE-TIME changes, so they are testable on the
existing FROZEN checkpoint with ZERO training. If either keeps recall non-zero at N>1 on
weights that were never trained to loop, the collapse was a mechanism problem all along
and the four training runs were treating a symptom.

Usage:
    python scripts/diagnose_loop_recall_collapse.py --n-trials 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402
from eval.grounded_qa import set_attention_window  # noqa: E402
from eval.synthetic_recall_probe import build_prompt, whole_word_hit  # noqa: E402
from scripts.sweep_looped_early_exit import generate_incremental  # noqa: E402
from hvn.edge.looped import _clone_state  # noqa: E402


def make_variant_forward_step(input_injection: bool, skip_attention_on_later_loops: bool):
    """Build a forward_step that loops with the given mechanism variants.

    Always runs exactly `self._diag_max_loops` loops (no entropy early exit) so this
    measures the mechanism, not any threshold tuning.
    """
    def _forward_step(self, input_ids, states, position_idx=None):
        base_states = states
        x0 = self.token_embedding(input_ids)          # [B, 1, D] the anchor
        h_prev = None
        logits = None
        prev_candidates = None

        for i in range(self._diag_max_loops):
            loop_states = _clone_state(base_states)
            if i == 0:
                h = x0
            elif input_injection:
                h = h_prev + x0                        # re-anchor to token identity
            else:
                h = h_prev                             # current (failing) behavior

            candidates = []
            for j, block in enumerate(self.blocks):
                is_attn = self.block_plan[j] == "transformer"
                if i > 0 and skip_attention_on_later_loops and is_attn:
                    # Leave the recall pathway untouched after loop 1: pass the
                    # hidden state through unchanged and carry that block's
                    # already-computed state forward.
                    candidates.append(prev_candidates[j])
                    continue
                h, st = block.forward_step(h, loop_states[j], position_idx=position_idx)
                candidates.append(st)

            logits = self.decoder(self.final_norm(h))
            h_prev = h
            prev_candidates = candidates

        return logits, prev_candidates

    return _forward_step


def apply_variant(model, input_injection, skip_attention, max_loops):
    if not hasattr(model, "_diag_base_forward_step"):
        model._diag_base_forward_step = model.forward_step
    fn = make_variant_forward_step(input_injection, skip_attention)
    model.forward_step = types.MethodType(fn, model)
    model._diag_max_loops = max_loops
    return model


def restore(model):
    if hasattr(model, "_diag_base_forward_step"):
        model.forward_step = model._diag_base_forward_step
        del model._diag_base_forward_step
    return model


def measure_recall(model, tok, device, n_trials, seed):
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
    ap.add_argument("--checkpoint", default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt",
                    help="FROZEN checkpoint -- this diagnostic does no training at all")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--n-values", type=int, nargs="+", default=[1, 4])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/looped_early_exit/diagnose_loop_recall_collapse.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    model, step, _ = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    print(f"Loaded FROZEN {args.checkpoint} (step {step}) -- no training in this script\n")

    variants = [
        ("current (no injection, loops everything)", False, False),
        ("A: input injection", True, False),
        ("B: skip attention on loops 2+", False, True),
        ("A+B: injection + skip attention", True, True),
    ]

    results = {}
    for label, inject, skip_attn in variants:
        print(f"=== {label} ===")
        results[label] = {}
        for N in args.n_values:
            apply_variant(model, inject, skip_attn, N)
            acc = measure_recall(model, tok, device, args.n_trials, args.seed)
            results[label][str(N)] = acc
            print(f"  N={N}: recall = {acc:.1%} ({int(round(acc * args.n_trials))}/{args.n_trials})")
        restore(model)
        print()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "step": step,
                   "n_trials": args.n_trials, "results": results}, f, indent=2)
    print(f"Wrote {args.output}\n")

    print("=== Verdict ===")
    big_n = str(max(args.n_values))
    baseline_collapsed = results[variants[0][0]][big_n] == 0.0
    rescued = [lbl for lbl, _, _ in variants[1:] if results[lbl][big_n] > 0.0]
    print(f"Current implementation collapses at N={big_n}: {baseline_collapsed}")
    if rescued:
        print(f"MECHANISM CONFIRMED -- these variants keep recall alive with NO training: {rescued}")
        print("=> The collapse was architectural, not a training problem. The four training")
        print("   runs were treating a symptom.")
    else:
        print("No variant rescues recall without training. The mechanism hypothesis (looping")
        print("destroys the copy-anchor) is NOT supported by this test -- the cause lies")
        print("elsewhere, and looping this architecture may simply be unproductive.")


if __name__ == "__main__":
    main()
