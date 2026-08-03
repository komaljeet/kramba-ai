"""scripts/test_state_commit_hypothesis.py — which loop's state should be committed? (Issue #122)

scripts/observe_loop_failure.py's alpha sweep had a built-in sanity check: at alpha=0 the
blend `h <- h + alpha*(Blocks(h) - h)` leaves h at loop 1's value, so every loop decodes an
IDENTICAL hidden state and recall must reproduce the N=1 baseline (~45%). It did not --
it returned 0/20. That invalidates the alpha curve, but the failure itself is the finding:
at alpha=0 the logits are provably identical to N=1, so the ONLY remaining difference is
which recurrent state gets committed for the next token.

Both the harness and hvn/edge/looped.py:103 commit the LAST loop's state
(`new_states = candidate_states` from the final iteration). That state was produced by
running the blocks on an already-transformed hidden state rather than on the token
embedding -- and for DeltaNet the state IS the memory, so its k/v are derived from a
smeared h and pollute the association the next token needs to read back. Each token
position also writes to that memory N times instead of once.

This script completes the 2x2 that separates the two candidate failure modes:

                        | state from loop 1 | state from loop N
    logits from loop 1  | (a) = N=1  ~45%   | (b) = alpha=0  0%
    logits from loop N  | (c) = ???         | (d) = current  0%

(a), (b), (d) are already measured. (c) is decisive:
  * (c) works  -> committing loop 1's state is the fix; the extra loops may refine the
                  prediction as long as they do not touch the recurrence. Looping viable.
  * (c) fails  -> drifted logits and corrupted state are INDEPENDENTLY fatal, and looping
                  this architecture is dead regardless of state handling.

All four cells are inference-only on the FROZEN checkpoint. No training.

Usage:
    python scripts/test_state_commit_hypothesis.py --n-trials 20
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import types

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402
from eval.grounded_qa import set_attention_window, EOS  # noqa: E402
from eval.synthetic_recall_probe import build_prompt, whole_word_hit  # noqa: E402
from hvn.edge.looped import _clone_state  # noqa: E402


def make_forward_step(logits_from_loop: int | None, state_from_loop: int):
    """Run `self._n` loops; take logits from `logits_from_loop` (1-indexed, None = last)
    and commit the state captured after `state_from_loop` (1-indexed).
    """
    def _forward_step(self, input_ids, states, position_idx=None):
        base_states = states
        h = self.token_embedding(input_ids)
        chosen_logits = None
        chosen_states = None

        for i in range(self._n):
            loop_states = _clone_state(base_states)
            h_in = h
            candidates = []
            for j, block in enumerate(self.blocks):
                h_in, st = block.forward_step(h_in, loop_states[j], position_idx=position_idx)
                candidates.append(st)
            h = h_in
            logits = self.decoder(self.final_norm(h))

            loop_no = i + 1
            if logits_from_loop is None or loop_no == logits_from_loop:
                chosen_logits = logits
            if loop_no == state_from_loop:
                chosen_states = candidates

        if chosen_states is None:      # state_from_loop beyond range -> last
            chosen_states = candidates
        return chosen_logits, chosen_states

    return _forward_step


def apply(model, logits_from, state_from, n):
    model.forward_step = types.MethodType(make_forward_step(logits_from, state_from), model)
    model._n = n
    return model


@torch.no_grad()
def generate(model, tok, device, prompt, max_new=20, temp=0.4, top_p=0.9, rep_pen=1.3):
    ids = tok.encode(prompt)
    states = model.init_state(1, device=device)
    pos = 0
    logits_step = None
    for t in ids:
        cur = torch.tensor([[t]], device=device)
        logits_step, states = model.forward_step(cur, states, position_idx=pos)
        pos += 1
    out: list[int] = []
    next_logits = logits_step[0, -1].float()
    for _ in range(max_new):
        lg = next_logits.clone()
        for tkn in set(out):
            lg[tkn] /= rep_pen
        probs = F.softmax(lg / temp, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp[cum - sp > top_p] = 0
        sp = sp / sp.sum()
        nxt = int(si[torch.multinomial(sp, 1)])
        if nxt == EOS:
            break
        out.append(nxt)
        cur = torch.tensor([[nxt]], device=device)
        logits_step, states = model.forward_step(cur, states, position_idx=pos)
        pos += 1
        next_logits = logits_step[0, -1].float()
    return tok.decode(out).strip()


@torch.no_grad()
def measure(model, tok, device, n_trials, seed):
    rng = torch.Generator().manual_seed(seed)
    hits = 0
    for _ in range(n_trials):
        prompt, correct = build_prompt(rng, n_distractors=0)
        hits += whole_word_hit(generate(model, tok, device, prompt), correct)
    return hits / n_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt")
    ap.add_argument("--n-trials", type=int, default=20)
    ap.add_argument("--n", type=int, default=4, help="loop count for the N>1 cells")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--output", default="results/looped_early_exit/state_commit_2x2.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model, step, _ = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    set_attention_window(model, None)
    print(f"FROZEN {args.checkpoint} (step {step}) -- inference only, no training\n")

    N = args.n
    cells = [
        ("a: logits L1 / state L1  (== N=1 baseline, sanity check)", 1, 1, 1),
        ("b: logits L1 / state LN  (state corruption alone)",        1, N, N),
        ("c: logits LN / state L1  (DECISIVE -- clean state)",       None, 1, N),
        ("d: logits LN / state LN  (current behaviour)",             None, N, N),
    ]

    results = {}
    for label, lf, sf, n in cells:
        apply(model, lf, sf, n)
        acc = measure(model, tok, device, args.n_trials, args.seed)
        results[label] = acc
        print(f"  {label:<58} recall = {acc:.1%} ({int(round(acc*args.n_trials))}/{args.n_trials})")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump({"checkpoint": args.checkpoint, "step": step, "N": N,
                   "n_trials": args.n_trials, "cells": results}, f, indent=2)
    print(f"\nWrote {args.output}\n")

    a = results[cells[0][0]]
    c = results[cells[2][0]]
    print("=== Verdict ===")
    if a < 0.20:
        print(f"HARNESS INVALID: cell (a) is the N=1 baseline and returned {a:.1%}, not ~45%.")
        print("Nothing else here is interpretable. Fix the harness before reading the rest.")
    elif c > 0.20:
        print(f"STATE COMMIT IS THE FIX: cell (c) = {c:.1%} with clean state vs 0% with")
        print("corrupted state. Extra loops are survivable as long as they do not write")
        print("to the recurrence. Wire this into hvn/edge/looped.py and re-validate.")
    else:
        print(f"BOTH MODES FATAL: cell (c) = {c:.1%}. Clean state does not rescue drifted")
        print("logits, so looping this architecture fails for reasons beyond state handling.")


if __name__ == "__main__":
    main()
