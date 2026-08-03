"""scripts/train_looped_sandbox_fixed.py — "looped LLM, no early exit" check (Issue #122, stage 1c).

Every training-side attempt so far (train_looped_sandbox.py, train_looped_sandbox_gated.py)
used DEEP SUPERVISION: every sampled loop count k's prediction is trained to be independently
correct, because that's what entropy-gated early exit needs (loop 1 alone must already be
usable). That's a real constraint EARLY EXIT specifically requires -- but it's not required for
"does looping help at all" in general. Forcing loop 1 to already be a complete, correct answer
may have been fighting against the model ever using loop 2+ to build toward something BETTER
than loop 1 alone.

This script drops early exit entirely and tests the plain, standard way weight-tied/looped
depth is normally trained: run the block stack a FIXED number of times every step (no random k,
no early-exit machinery at all), and supervise ONLY the final loop's prediction. Intermediate
loops never need to independently decode into anything -- they're free to be incomplete
"work in progress" representations.

Comparison: train two models for the SAME number of optimizer steps on the SAME data --
--max-loops 1 (a plain single-pass baseline) vs --max-loops N (always exactly N passes,
final-loop-only supervision) -- and compare held-out loss. If loop N ends up LOWER loss than
the single-pass baseline, repeated computation is doing something useful (just not compatible
with early exit, which is a separate, harder property). If it's flat or worse, the block
stack's repeated application isn't productive at all, independent of the early-exit question.

Usage:
    python scripts/train_looped_sandbox_fixed.py --steps 3000 --max-loops 1 \
        --output results/looped_early_exit/sandbox_1m_fixed_N1.json
    python scripts/train_looped_sandbox_fixed.py --steps 3000 --max-loops 4 \
        --output results/looped_early_exit/sandbox_1m_fixed_N4.json
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import CharTokenizer, SAMPLE_TEXT  # noqa: E402
from train_looped_sandbox import build_sandbox_model, split_corpus, make_batches  # noqa: E402
from hvn.edge.looped import looped_forward_train  # noqa: E402


def normalized_entropy_batch(logits: torch.Tensor, vocab_size: int) -> float:
    probs = torch.softmax(logits.float(), dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    ent = -(probs * logp).sum(dim=-1)
    return (ent / math.log(vocab_size)).mean().item()


def train_fixed(model, tok, device, steps, N, seq_len, batch_size, lr, train_text):
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
            position_ids = torch.arange(x.size(1), device=device).unsqueeze(0).expand(x.size(0), -1)

            # deep_supervise=False -> only the FINAL loop's logits are supervised.
            logits = looped_forward_train(model, x, position_ids, N, deep_supervise=False)[0]
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % max(steps // 20, 1) == 0:
                print(f"step {step:5d}  N={N}  loss={loss.item():.4f}")
                history.append({"step": step, "loss": loss.item()})
            step += 1

    return history


@torch.no_grad()
def evaluate(model, tok, device, N, val_text, seq_len=64):
    model.eval()
    ids = tok.encode(val_text)
    ids = ids[:seq_len + 1] if len(ids) > seq_len + 1 else ids
    x = torch.tensor([ids[:-1]], device=device)
    y = torch.tensor([ids[1:]], device=device)
    position_ids = torch.arange(x.size(1), device=device).unsqueeze(0)
    logits = looped_forward_train(model, x, position_ids, N, deep_supervise=False)[0]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1)).item()
    ent = normalized_entropy_batch(logits, model.vocab_size)
    return {"N": N, "held_out_loss": loss, "normalized_entropy": ent}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--max-loops", type=int, default=4, help="fixed N -- always exactly this many loops")
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--output", default="results/looped_early_exit/sandbox_1m_fixed_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_text, val_text = split_corpus(SAMPLE_TEXT)
    tok = CharTokenizer(SAMPLE_TEXT, add_tool_tokens=False)
    model = build_sandbox_model(tok.vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Sandbox model (fixed N={args.max_loops}, final-loop-only supervision): "
         f"{n_params:,} params, vocab={tok.vocab_size}, device={device}")

    print(f"\n=== Training (fixed N={args.max_loops}, no deep supervision) ===")
    history = train_fixed(model, tok, device, args.steps, args.max_loops,
                          args.seq_len, args.batch_size, args.lr, train_text)

    print("\n=== Eval: held-out loss at the trained N ===")
    result = evaluate(model, tok, device, args.max_loops, val_text)
    print(f"  N={result['N']}  held_out_loss={result['held_out_loss']:.4f}  "
         f"normalized_entropy={result['normalized_entropy']:.4f}")

    payload = {
        "n_params": n_params, "vocab_size": tok.vocab_size, "steps": args.steps,
        "N": args.max_loops, "train_history": history, "eval": result,
    }
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
