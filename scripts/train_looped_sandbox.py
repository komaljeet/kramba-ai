"""scripts/train_looped_sandbox.py — cheap ~1M-param recipe validation (Issue #122, stage 1).

The GPU sweep in scripts/sweep_looped_early_exit.py proved untrained looping doesn't
work: re-running EdgeHVN's block stack on a checkpoint that was never trained to loop
corrupts its output almost immediately (docs/state.md). The open question is whether
a LOOP-AWARE TRAINING OBJECTIVE fixes that -- i.e. is this a training problem (fixable
cheaply) or an architectural dead end (not fixable by more training).

This script is a disposable, from-scratch sandbox (~1M params, char-level, trains in
minutes) built ONLY to answer that question fast and cheaply, before spending any real
time fine-tuning the actual 83M checkpoint. It is not meant to produce a usable model.

Training recipe (standard "looped transformer" / deep-supervision style):
  - Every step, sample a random loop count k ~ Uniform[1, max_loops].
  - Run the full block stack k times on the SAME sequence (loop i+1 re-processes loop
    i's output hidden state -- same "no fresh re-embedding, no external recurrent state
    carried across loops" semantics as hvn/edge/looped.py's inference-time wrapper, just
    over the whole sequence via block.forward() instead of per-token forward_step()).
  - DEEP SUPERVISION: cross-entropy loss is computed and averaged across ALL k loops'
    logits, not just the final one. This is what actually teaches "loop 1's output alone
    should already be a reasonable prediction" -- the property the inference-time
    entropy-gated exit depends on. Training only the final loop's loss would teach the
    model to rely on reaching loop k, defeating early exit entirely.

Evaluation: after training, compute held-out loss and normalized entropy at FIXED loop
counts 1..max_loops (deterministic, not sampled) on data the "easy"/"hard" split can
reuse from eval/loop_exit_probe.py's bundled sample. If loss/entropy improve
monotonically with more loops, the recipe works and is worth applying (via a continued
fine-tune, not from scratch) to the real checkpoint. If not, looping needs an
architectural fix, not just training.

Usage:
    python scripts/train_looped_sandbox.py --steps 2000 --max-loops 4 \
        --output results/looped_early_exit/sandbox_1m_results.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import CharTokenizer, SAMPLE_TEXT  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402
from hvn.edge.looped import looped_forward_train  # noqa: E402


def split_corpus(text: str, val_frac: float = 0.15) -> tuple[str, str]:
    """Train/held-out split WITHIN the same corpus (same style/vocab, unseen exact
    sequences) -- NOT eval.loop_exit_probe's BUNDLED_SAMPLE_TEXT, which is a
    different distribution entirely (children's-story prose vs this repo's own
    architecture-description sample text) and produced a misleading
    out-of-domain-generalization result rather than a real recipe verdict.
    """
    cut = int(len(text) * (1 - val_frac))
    return text[:cut], text[cut:]


def build_sandbox_model(vocab_size, device, d_model=160, num_layers=3, n_heads=4, attn_window=64):
    model = EdgeHVN(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        n_heads=n_heads,
        use_deltanet=True,
        block_plan=["deltanet", "transformer", "deltanet"][:num_layers],
        attn_window=attn_window,
        tie_embeddings=True,  # char vocab is tiny, but tying still halves the embed+decoder cost
    )
    return model.to(device)


def normalized_entropy_batch(logits: torch.Tensor, vocab_size: int) -> float:
    import math
    probs = torch.softmax(logits.float(), dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    ent = -(probs * logp).sum(dim=-1)
    return (ent / math.log(vocab_size)).mean().item()


def make_batches(ids, seq_len, batch_size, device):
    n = (len(ids) - 1) // seq_len
    n = max(n, 1)
    xs, ys = [], []
    for i in range(n):
        chunk = ids[i * seq_len: i * seq_len + seq_len + 1]
        if len(chunk) < seq_len + 1:
            continue
        xs.append(chunk[:-1])
        ys.append(chunk[1:])
    if not xs:
        raise ValueError("corpus too short for the requested seq_len")
    x = torch.tensor(xs, device=device)
    y = torch.tensor(ys, device=device)
    for i in range(0, len(xs), batch_size):
        yield x[i:i + batch_size], y[i:i + batch_size]


def train(model, tok, device, steps, max_loops, seq_len, batch_size, lr, deep_supervise, train_text):
    corpus = train_text * 200  # repeat -- this is a memorization-regime mechanism test
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

            per_loop_logits = looped_forward_train(model, x, position_ids, k, deep_supervise)
            loss = sum(F.cross_entropy(lg.reshape(-1, lg.size(-1)), y.reshape(-1)) for lg in per_loop_logits) / len(per_loop_logits)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if step % max(steps // 20, 1) == 0:
                print(f"step {step:5d}  k={k}  loss={loss.item():.4f}")
                history.append({"step": step, "k": k, "loss": loss.item()})
            step += 1

    return history


@torch.no_grad()
def evaluate_by_loop_depth(model, tok, device, max_loops, val_text, seq_len=64):
    """Deterministic eval: loss + normalized entropy at EACH fixed loop count 1..max_loops,
    on the held-out tail split from the SAME corpus (same style/vocab as training,
    unseen exact sequences) -- see split_corpus()."""
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
        logits_list = looped_forward_train(model, x, position_ids, k, deep_supervise=False)
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
    ap.add_argument("--no-deep-supervise", action="store_true",
                    help="ablation: supervise only the final loop's loss, not every loop")
    ap.add_argument("--output", default="results/looped_early_exit/sandbox_1m_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_text, val_text = split_corpus(SAMPLE_TEXT)
    tok = CharTokenizer(SAMPLE_TEXT, add_tool_tokens=False)  # vocab built over the FULL corpus so the held-out tail has no OOV chars
    model = build_sandbox_model(tok.vocab_size, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Sandbox model: {n_params:,} params, vocab={tok.vocab_size}, device={device}")
    print(f"Corpus split: {len(train_text)} train chars / {len(val_text)} held-out chars (same distribution)")

    print("\n=== Training (loop-aware, deep-supervised) ===")
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
    verdict = ("RECIPE WORKS: loss improves monotonically with more loops"
              if monotonic_improving else
              "RECIPE DOES NOT CLEANLY WORK: loss does not monotonically improve with more loops")
    print(f"\n{verdict}")

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
