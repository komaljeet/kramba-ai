"""scripts/observe_loop_failure.py — LOOK at the loop failure (Issue #122).

Five experiments (4 training recipes + the mechanism diagnostic) have measured this
collapse, and every one of them measured it the same binary way: did the generated answer
contain the target word. Not one of them looked at WHAT the model actually does when it
loops. This script closes that blind spot before proposing another fix.

Three observations, all inference-only on the FROZEN checkpoint (no training):

  1. WHAT IT SAYS. Generate side by side at N=1 and N=4 on identical prompts. Garbage vs
     fluent-but-wrong vs repetition vs empty are completely different failures and we do
     not currently know which one this is.

  2. WHETHER THE RESIDUAL STREAM BLOWS UP. Every block is residual
     (`x_out = x_t + attn + ffn`), so a single pass accumulates 12 additions into the
     stream; N=4 accumulates 48, because loop i+1 starts from loop i's raw un-normalized
     output (hvn/edge/looped.py:104 is a plain `x = h`). Each block normalizes its OWN
     input internally, so per-block contributions stay roughly constant in magnitude
     while the stream itself keeps growing -- which would progressively dilute the recall
     signal the attention layer wrote in loop 1 relative to everything else
     `final_norm` sees. Universal Transformer normalizes every iteration; this does not.
     Logs ||h|| per loop plus cosine similarity to loop 1's hidden state.

  3. WHERE IT BREAKS. Blend each loop's output instead of replacing outright:
     `h <- h + alpha * (Blocks(h) - h)`. alpha=0 is loop 1 repeated (must reproduce the
     45% baseline by construction -- a built-in sanity check); alpha=1 is current
     behavior (0%). Sweeping alpha locates the cliff. A collapse at tiny alpha means the
     readout is extraordinarily brittle and looping cannot work without retraining it;
     graceful degradation means there is a usable regime.

Usage:
    python scripts/observe_loop_failure.py --n-trials 15
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
from hvn.edge.looped import _clone_state, normalized_entropy  # noqa: E402


def make_blended_forward_step(alpha: float, trace: list | None = None):
    """forward_step that runs exactly `self._obs_n` loops, blending each loop's output
    into the previous state by `alpha` (alpha=1.0 == current replace-outright behavior).
    Optionally appends per-loop diagnostics to `trace`.
    """
    def _forward_step(self, input_ids, states, position_idx=None):
        base_states = states
        h = self.token_embedding(input_ids)
        h_first = None
        logits = None
        candidates = None

        for i in range(self._obs_n):
            loop_states = _clone_state(base_states)
            h_in = h
            candidates = []
            for j, block in enumerate(self.blocks):
                h_in, st = block.forward_step(h_in, loop_states[j], position_idx=position_idx)
                candidates.append(st)
            # blend: alpha=1 -> replace (current), alpha=0 -> keep previous entirely
            h = h + alpha * (h_in - h) if i > 0 else h_in
            if i == 0:
                h_first = h.detach().clone()
            logits = self.decoder(self.final_norm(h))

            if trace is not None:
                trace.append({
                    "loop": i + 1,
                    "h_norm": float(h.norm().item()),
                    "cos_to_loop1": float(F.cosine_similarity(
                        h.flatten().unsqueeze(0), h_first.flatten().unsqueeze(0)).item()),
                    "entropy": normalized_entropy(logits, self.vocab_size),
                    "argmax_token": int(torch.argmax(logits[0, -1]).item()),
                })
        return logits, candidates

    return _forward_step


def apply(model, alpha, n, trace=None):
    if not hasattr(model, "_obs_base_fs"):
        model._obs_base_fs = model.forward_step
    model.forward_step = types.MethodType(make_blended_forward_step(alpha, trace), model)
    model._obs_n = n
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
def measure_recall(model, tok, device, n_trials, seed):
    rng = torch.Generator().manual_seed(seed)
    hits = 0
    for _ in range(n_trials):
        prompt, correct = build_prompt(rng, n_distractors=0)
        hits += whole_word_hit(generate(model, tok, device, prompt), correct)
    return hits / n_trials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt")
    ap.add_argument("--n-trials", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alphas", type=float, nargs="+",
                    default=[0.0, 0.05, 0.1, 0.25, 0.5, 1.0])
    ap.add_argument("--output", default="results/looped_early_exit/observe_loop_failure.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model, step, _ = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    set_attention_window(model, None)
    print(f"FROZEN {args.checkpoint} (step {step}) -- inference only, no training\n")

    report = {"checkpoint": args.checkpoint, "step": step}

    # --- 1. What does it actually SAY? ---
    print("=" * 72)
    print("1. WHAT IT SAYS -- same prompts at N=1 vs N=4")
    print("=" * 72)
    samples = []
    rng = torch.Generator().manual_seed(args.seed)
    prompts = [build_prompt(rng, n_distractors=0) for _ in range(3)]
    for prompt, correct in prompts:
        apply(model, alpha=1.0, n=1)
        a1 = generate(model, tok, device, prompt)
        apply(model, alpha=1.0, n=4)
        a4 = generate(model, tok, device, prompt)
        print(f"  expect={correct!r}")
        print(f"    N=1: {a1!r}")
        print(f"    N=4: {a4!r}")
        samples.append({"expected": correct, "n1": a1, "n4": a4})
    report["samples"] = samples

    # --- 2. Does the residual stream blow up? ---
    print("\n" + "=" * 72)
    print("2. RESIDUAL STREAM across loops (one representative prompt, N=8)")
    print("=" * 72)
    trace: list = []
    apply(model, alpha=1.0, n=8, trace=trace)
    prompt, correct = prompts[0]
    generate(model, tok, device, prompt, max_new=1)
    # keep only the first token's trace (8 loops)
    first = trace[:8]
    for t in first:
        print(f"  loop {t['loop']}: ||h||={t['h_norm']:9.2f}  cos_to_loop1={t['cos_to_loop1']:+.4f}"
              f"  entropy={t['entropy']:.4f}  argmax={t['argmax_token']}")
    report["residual_trace"] = first
    if first:
        growth = first[-1]["h_norm"] / max(first[0]["h_norm"], 1e-9)
        print(f"\n  ||h|| growth loop1 -> loop8: {growth:.2f}x")
        print(f"  cos(loop8, loop1) = {first[-1]['cos_to_loop1']:+.4f}")
        report["h_norm_growth_1_to_8"] = growth

    # --- 3. Where does it break? ---
    print("\n" + "=" * 72)
    print("3. ALPHA SWEEP at N=4 -- where does recall die?")
    print("   (alpha=0 must reproduce the ~45% N=1 baseline: built-in sanity check)")
    print("=" * 72)
    alpha_results = {}
    for a in args.alphas:
        apply(model, alpha=a, n=4)
        acc = measure_recall(model, tok, device, args.n_trials, args.seed)
        alpha_results[str(a)] = acc
        print(f"  alpha={a:<5}: recall = {acc:.1%} ({int(round(acc * args.n_trials))}/{args.n_trials})")
    report["alpha_sweep_n4"] = alpha_results

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
