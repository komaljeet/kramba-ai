"""scripts/train_looped_sandbox_gated.py — gated-loop recipe check (Issue #122, stage 1b).

Stage 1 (train_looped_sandbox.py) and stage 2 (continue_ft_looped.py) both showed the
SAME failure: recall collapses to 0% the instant looping fires more than once, even
after real continued-FT on real data. Working hypothesis recorded in docs/state.md: the
plain deep-supervision recipe has a structural gap -- nothing rewards "leave the answer
alone once it's already right" for loop 2+, so a correct loop-1 answer gets overwritten
by loop 2's un-gated full re-transformation.

This script tests the fix BEFORE spending real GPU time on the 83M checkpoint again:
add a learned per-token GATE that blends each loop's raw block-stack output with the
PREVIOUS loop's hidden state, instead of replacing it outright --

    h_new_raw = Blocks(h)
    gate = sigmoid(loop_gate(h))            # [B, S, 1], starts near 0 (bias=-3 init)
    h = h + gate * (h_new_raw - h)           # loop 2+ only; loop 1 is always the full pass

so the model has an explicit, cheap mechanism to make loop 2+ a small perturbation
rather than a full overwrite -- it has to LEARN to change the answer, instead of being
structurally forced to every time. `loop_gate` is a genuinely new parameter (not
present in any existing checkpoint), so this can only be tested via training, not
inference-only wrapping.

Same cheap ~1M-param, from-scratch, char-level, train-in-minutes sandbox as stage 1 --
this is a mechanism check, not a quality bar. If the gate lets loss/entropy improve (or
at least NOT collapse) with more loops, the fix is worth porting to a real continued-FT.

Usage:
    python scripts/train_looped_sandbox_gated.py --steps 2000 --max-loops 4 \
        --output results/looped_early_exit/sandbox_1m_gated_results.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import CharTokenizer, SAMPLE_TEXT  # noqa: E402
from train_looped_sandbox import build_sandbox_model, split_corpus, make_batches  # noqa: E402


def attach_loop_gate(model, device, bias_init=-1.0):
    """A single per-token scalar gate, shared across all loop iterations (2+). Bias
    init at -3 -> sigmoid(-3) ~= 0.047, so early training starts near-identity for
    extra loops (barely changes the previous loop's hidden state) and the model has
    to learn to open the gate when a real refinement helps, rather than being forced
    to fully re-transform on every loop from step 0.
    """
    gate = nn.Linear(model.d_model, 1)
    nn.init.zeros_(gate.weight)
    nn.init.constant_(gate.bias, bias_init)
    model.loop_gate = gate.to(device)  # registers as a submodule (EdgeHVN is nn.Module)
    return model


def looped_forward_train_gated(model, input_ids, position_ids, k, deep_supervise=True):
    """Gated counterpart to hvn.edge.looped.looped_forward_train. Loop 1 is always the
    full, ungated transformation (there's no 'previous answer' to preserve yet); loop
    2+ blends the raw block-stack output with the previous hidden state via the gate.
    """
    h = model.token_embedding(input_ids)
    all_logits = []
    for i in range(k):
        h_raw = h
        for block in model.blocks:
            h_raw = block(h_raw, position_ids=position_ids)
        if i == 0:
            h = h_raw
        else:
            gate = torch.sigmoid(model.loop_gate(h))  # [B, S, 1]
            h = h + gate * (h_raw - h)
        logits = model.decoder(model.final_norm(h))
        if deep_supervise or i == k - 1:
            all_logits.append(logits)
    return all_logits


def normalized_entropy_batch(logits: torch.Tensor, vocab_size: int) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    ent = -(probs * logp).sum(dim=-1)
    return (ent / math.log(vocab_size)).mean().item()


def train(model, tok, device, steps, max_loops, seq_len, batch_size, lr, deep_supervise, train_text):
    corpus = train_text * 200
    ids = tok.encode(corpus)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()

    step = 0
    history = []
    while step < steps:
        for x, y in make_batches(ids, seq_len, batch_size, device):
            if step >= steps:
                break
            k = int(torch.randint(1, max_loops + 1, (1,)).item())
            position_ids = torch.arange(x.size(1), device=device).unsqueeze(0).expand(x.size(0), -1)

            per_loop_logits = looped_forward_train_gated(model, x, position_ids, k, deep_supervise)
            loss = sum(F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1)) for lg in per_loop_logits) / len(per_loop_logits)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % max(steps // 20, 1) == 0:
                gate_mean = torch.sigmoid(model.loop_gate.bias).item()
                print(f"step {step:5d}  k={k}  loss={loss.item():.4f}  gate_bias_sigmoid={gate_mean:.4f}")
                history.append({"step": step, "k": k, "loss": loss.item(), "gate_bias_sigmoid": gate_mean})
            step += 1

    return history


@torch.no_grad()
def evaluate_by_loop_depth(model, tok, device, max_loops, val_text, seq_len=64):
    model.eval()
    ids = tok.encode(val_text)
    ids = ids[:seq_len + 1] if len(ids) > seq_len + 1 else ids
    if len(ids) < 2:
        raise ValueError("held-out sample too short")
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    position_ids = torch.arange(x.size(1), device=device).unsqueeze(0)

    results = []
    for k in range(1, max_loops + 1):
        logits_list = looped_forward_train_gated(model, x, position_ids, k, deep_supervise=False)
        logits = logits_list[-1]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
        ent = normalized_entropy_batch(logits, model.vocab_size)
        results.append({"k": k, "held_out_loss": loss, "normalized_entropy": ent})
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--max-loops", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--no-deep-supervise", action="store_true")
    ap.add_argument("--gate-bias-init", type=float, default=-1.0,
                    help="sigmoid(-1)~=0.27 (more initial gradient signal than the "
                         "first attempt's -3~=0.047, which barely moved in 3000 steps)")
    ap.add_argument("--output", default="results/looped_early_exit/sandbox_1m_gated_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_text, val_text = split_corpus(SAMPLE_TEXT)
    tok = CharTokenizer(SAMPLE_TEXT, add_tool_tokens=False)
    model = build_sandbox_model(tok.vocab_size, device)
    attach_loop_gate(model, device, bias_init=args.gate_bias_init)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Sandbox model (gated): {n_params:,} params, vocab={tok.vocab_size}, device={device}")
    print(f"Corpus split: {len(train_text)} train chars / {len(val_text)} held-out chars")

    print("\n=== Training (loop-aware, GATED, deep-supervised) ===")
    history = train(model, tok, device, args.steps, args.max_loops, args.seq_len,
                    args.batch_size, args.lr, deep_supervise=not args.no_deep_supervise,
                    train_text=train_text)

    print("\n=== Eval: loss/entropy at fixed loop depths (held-out text, same distribution) ===")
    eval_results = evaluate_by_loop_depth(model, tok, device, args.max_loops, val_text)
    for r in eval_results:
        print(f"  k={r['k']}  held_out_loss={r['held_out_loss']:.4f}  "
             f"normalized_entropy={r['normalized_entropy']:.4f}")

    losses = [r["held_out_loss"] for r in eval_results]
    monotonic_improving = all(losses[i + 1] <= losses[i] + 1e-6 for i in range(len(losses) - 1))
    not_collapsing = losses[-1] < losses[0] * 1.5  # loose bar: doesn't blow up with more loops
    if monotonic_improving:
        verdict = "GATED RECIPE WORKS: loss improves monotonically with more loops"
    elif not_collapsing:
        verdict = "GATED RECIPE PARTIALLY HELPS: loss doesn't monotonically improve but doesn't collapse either"
    else:
        verdict = "GATED RECIPE STILL FAILS: loss gets meaningfully worse with more loops"
    print(f"\n{verdict}")
    print(f"final gate bias sigmoid (baseline openness): {torch.sigmoid(model.loop_gate.bias).item():.4f}")

    payload = {
        "n_params": n_params, "vocab_size": tok.vocab_size, "steps": args.steps,
        "max_loops": args.max_loops, "deep_supervise": not args.no_deep_supervise,
        "train_history": history, "eval_by_loop_depth": eval_results,
        "verdict": verdict,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
