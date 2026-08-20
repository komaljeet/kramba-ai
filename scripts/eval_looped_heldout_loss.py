"""scripts/eval_looped_heldout_loss.py -- the metric Issue #122 never measured at real scale.

Author: Rambarun Komaljeet

The fixed-N looping recipe (no early exit, final-loop-only supervision) won in the
~780K-param sandbox on HELD-OUT LOSS (N=4: 4.9621 vs N=1: 5.0050,
results/looped_early_exit/sandbox_1m_fixed_N{1,4}.json). The real-scale check
(continue_ft_looped.py --fixed-n --no-deep-supervise, 83M, 2000 steps) was then
graded ONLY on synthetic near-window recall -- see
results/looped_early_exit/validate_looped_ft_fixed_step2000.json, which contains
nothing but `by_n` recall percentages. So "the sandbox signal did not transfer"
(docs/state.md) rests on a different metric than the one that showed the signal.
Held-out loss under fixed-N looping at 83M is unmeasured. This measures it.

Why this cannot reuse eval/heldout_loss.py
------------------------------------------
Two independent reasons, both of which would silently produce a wrong number:

1. ``hvn.edge.looped.enable_looped_early_exit`` monkeypatches ONLY
   ``forward_step`` (incremental decode). ``eval/heldout_loss.py`` calls
   ``model(x)`` -- the full-sequence forward, which is NOT patched. Combining
   them returns byte-identical logits at every N: a test that cannot fail.
   The full-sequence looped path is ``looped_forward_train(..., k=N,
   deep_supervise=False)``, which is what the fixed-N FT actually trained with.
2. eval/heldout_loss.py defaults to the code corpus + the 32k FIM tokenizer.
   These checkpoints are the GPT2-tokenizer conversational hybrid.

Held-out set
------------
``data_cache/heldout_fineweb.txt`` -- the slice this repo's A/B scripts already
use as held-out. Neither checkpoint trained on it. (The looped FT trained ~12
epochs on data/grounded_qa_expanded.jsonl, so that file is memorisation-
contaminated for the FT arm and unusable here -- the #132 retraction trap.)

Two guards, because the interesting result is the one that would be wrong
-------------------------------------------------------------------------
* **Vacuity**: asserts loop N=1 and N=2 produce DIFFERENT logits. If the loop
  is not wired, every N returns the same NLL and the run would read as a clean
  "looping is neutral" -- the exact way #138 published a bug as a number.
* **Comparability**: asserts every arm scored the identical token count.

Usage::

    python scripts/eval_looped_heldout_loss.py --max-tokens 200000
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_edge_conversational as tec  # noqa: E402
from continue_ft_hybrid import enable_scan_flags  # noqa: E402
from hvn.edge.looped import looped_forward_train  # noqa: E402


def heldout_ids(path: str, tokenizer, max_tokens: int) -> list[int]:
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_tokens]


@torch.no_grad()
def evaluate_at_n(model, ids: list[int], seq_len: int, n_loops: int, device,
                  capture_first_logits: bool = False):
    """Mean next-token NLL over non-overlapping windows, with the block stack
    applied ``n_loops`` times per window (final loop's prediction only -- the
    recipe the fixed-N FT was trained with)."""
    model.eval()
    n_win = len(ids) // seq_len
    total, counted = 0.0, 0
    first_logits = None
    for w in range(n_win):
        chunk = ids[w * seq_len:(w + 1) * seq_len]
        x = torch.tensor([chunk], device=device)
        position_ids = torch.arange(x.size(1), device=device).unsqueeze(0)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            per_loop = looped_forward_train(model, x, position_ids, n_loops,
                                            deep_supervise=False)
        logits = per_loop[-1]
        if capture_first_logits and w == 0:
            first_logits = logits.detach().float().cpu().clone()
        lp = logits[0, :-1].float()
        tgt = torch.tensor(chunk[1:], device=device)
        total += F.cross_entropy(lp, tgt, reduction="sum").item()
        counted += tgt.numel()
    nll = total / max(1, counted)
    return ({"heldout_nll": nll,
             "heldout_ppl": float(torch.tensor(nll).exp()),
             "tokens_scored": counted,
             "windows": n_win},
            first_logits)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--ft-checkpoint",
                   default="checkpoints/hybrid_looped_ft_fixed/hybrid_looped_step2000.pt")
    p.add_argument("--baseline-checkpoint",
                   default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt")
    p.add_argument("--heldout-text", default="data_cache/heldout_fineweb.txt")
    p.add_argument("--n-values", type=int, nargs="+", default=[1, 2, 4])
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--max-tokens", type=int, default=200_000)
    p.add_argument("--out", default="results/looped_early_exit/heldout_loss_fixed_n.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    ids = heldout_ids(args.heldout_text, tok, args.max_tokens)
    n_win = len(ids) // args.seq_len
    print(f"held-out: {len(ids):,} tokens from {args.heldout_text}, "
          f"seq_len {args.seq_len} -> {n_win} windows")

    results = {}
    token_counts = set()

    for label, ckpt_path in [("ft_fixed_n", args.ft_checkpoint),
                             ("baseline", args.baseline_checkpoint)]:
        print(f"\n=== {label}: {ckpt_path} ===")
        model, step, _ = tec.load_checkpoint_for_resume(ckpt_path, device)
        enable_scan_flags(model, triton_scan=True, megakernel_scan=True)
        params = sum(q.numel() for q in model.parameters())
        print(f"  loaded (step {step}, {params:,} params)")

        results[label] = {"checkpoint": ckpt_path, "step": step,
                          "params": params, "by_n": {}}
        probe = {}
        for N in args.n_values:
            t0 = time.time()
            row, first = evaluate_at_n(model, ids, args.seq_len, N, device,
                                       capture_first_logits=(N in (1, 2)))
            row["seconds"] = round(time.time() - t0, 1)
            results[label]["by_n"][str(N)] = row
            token_counts.add(row["tokens_scored"])
            if first is not None:
                probe[N] = first
            print(f"  N={N}: held-out NLL {row['heldout_nll']:.4f}  "
                  f"ppl {row['heldout_ppl']:.3f}  ({row['seconds']}s)")

        # --- GUARD 1: the loop must actually change the computation ---
        if 1 in probe and 2 in probe:
            delta = (probe[1] - probe[2]).abs().max().item()
            results[label]["guard_max_logit_delta_n1_vs_n2"] = delta
            print(f"  guard: max|logits(N=1) - logits(N=2)| = {delta:.4e}")
            if delta == 0.0:
                sys.exit(
                    f"VACUOUS TEST -- {label}: N=1 and N=2 produced identical "
                    "logits, so looping is not running. Any NLL comparison "
                    "here would be meaningless. Aborting rather than "
                    "publishing a flat result as 'looping is neutral'."
                )
        del model
        torch.cuda.empty_cache()

    # --- GUARD 2: every arm must be scored on the same tokens ---
    if len(token_counts) != 1:
        sys.exit(f"arms scored different token counts {token_counts} -- not comparable")
    print(f"\nguard: all arms scored {token_counts.pop():,} tokens")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"heldout_text": args.heldout_text,
                   "heldout_tokens": len(ids),
                   "seq_len": args.seq_len,
                   "recipe": "fixed-N loops, final-loop-only (deep_supervise=False)",
                   "results": results}, fh, indent=2)
    print(f"wrote {args.out}")

    print("\n=== Verdict ===")
    for label in results:
        by_n = results[label]["by_n"]
        base = by_n[str(args.n_values[0])]["heldout_nll"]
        print(f"{label}:")
        for N in args.n_values:
            nll = by_n[str(N)]["heldout_nll"]
            d = nll - base
            tag = "" if N == args.n_values[0] else f"  ({d:+.4f} vs N={args.n_values[0]})"
            print(f"   N={N}: {nll:.4f}{tag}")
        best = min(args.n_values, key=lambda n: by_n[str(n)]["heldout_nll"])
        print(f"   -> best N = {best}"
              + ("  (looping HELPS held-out loss)" if best != args.n_values[0]
                 else "  (looping does NOT help held-out loss)"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
