"""How different is the gradient at b2/ga16 vs b4/ga8, really? (Issue #141 follow-up)

Author: Rambarun Komaljeet

#141 claimed `batch 2/grad_accum 16 -> batch 4/grad_accum 8` is "identical
effective batch, identical gradient, identical result".  **The middle claim is
wrong**, and this script measures by how much.

The loss is masked (``ignore_index=-100``, ``continue_ft_hybrid.py:189``) -- only
answer tokens contribute.  So micro-batch ``g``'s loss is a mean over its own
count of unmasked tokens ``N_g``, and the loop's ``loss / grad_accum`` then gives
every micro-batch equal weight regardless of ``N_g``::

    accumulated = (1/G) * sum_g [ (1/N_g) * sum_{i in g} l_i ]
    true batch  = (1/sum_g N_g) * sum_all l_i

Equal only when every ``N_g`` is identical.  Token ``i`` in micro-batch ``g``
carries effective weight ``1/(G*N_g)`` instead of ``1/sum(N)`` -- so tokens
sitting in token-sparse micro-batches are over-weighted, and the *grouping*
(2 rows per micro-batch vs 4) changes which tokens those are.

This is the well-known gradient-accumulation normalisation gap, and it means the
two configs train on subtly different objectives.  Whether that matters here is
an empirical question about the spread of ``N_g``, which is what this measures --
no model and no GPU needed, since the effect is entirely in the token counts.

Reports, over the real packed corpus:

* the spread of ``N_g`` at each grouping,
* the per-token weight distortion vs the true token-weighted batch loss,
* how much *closer* to the true objective b4/ga8 is than b2/ga16.

Usage::

    python scripts/check_grad_accum_weighting.py --data data/grounded_qa_expanded.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune_edge_conversational import GroundedQADataset, EOS_ID   # noqa: E402
from hvn.core.packing import PackedDataset                           # noqa: E402


def weights_for_grouping(counts: np.ndarray, batch_size: int) -> np.ndarray:
    """Effective per-ROW weight under `loss/grad_accum` accumulation.

    A row's tokens share its micro-batch's ``1/(G*N_g)``; the row contributes
    ``n_row`` of those tokens, so its share of the total gradient is
    ``n_row / (G * N_g)``.  Returned normalised to sum to 1 for comparison with
    the true token-weighted batch, whose row weights are ``n_row / sum(N)``.
    """
    n_rows = len(counts)
    assert n_rows % batch_size == 0
    groups = counts.reshape(-1, batch_size)
    n_g = groups.sum(axis=1)                      # unmasked tokens per micro-batch
    G = len(n_g)
    w = (groups / n_g[:, None]) / G               # [G, batch_size]
    return w.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--effective-batch", type=int, default=32)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--trials", type=int, default=200,
                    help="random effective-batch draws to average over")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str,
                    default="results/throughput_sweep/grad_accum_weighting.json")
    args = ap.parse_args()

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    records = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    ds = GroundedQADataset(tok, records, seq_len=args.seq_len, pad=False)
    packed = PackedDataset(ds.examples, seq_len=args.seq_len, pad_id=EOS_ID,
                           seed=0, strategy="shuffled_first_fit")

    # Unmasked (answer) token count per packed row -- the whole story lives here.
    counts = []
    for i in range(len(packed)):
        item = packed[i]
        y = item[1]
        counts.append(int((np.asarray(y) != -100).sum()))
    counts = np.asarray(counts, dtype=np.float64)

    print(f"\nPacked rows: {len(counts)}")
    print(f"Answer tokens per row: mean {counts.mean():.1f}, sd {counts.std():.1f}, "
          f"min {counts.min():.0f}, max {counts.max():.0f}, "
          f"CV {counts.std()/counts.mean():.3f}")

    rng = np.random.default_rng(args.seed)
    results = {}
    print(f"\nOver {args.trials} random draws of {args.effective_batch} rows:")
    print(f"{'micro-batch':>12}{'micro-batches':>15}{'max token over/under-weight':>30}"
          f"{'mean |dev| from true':>23}")
    print("-" * 80)
    for bs in args.batches:
        if args.effective_batch % bs:
            continue
        devs, maxes = [], []
        for _ in range(args.trials):
            idx = rng.choice(len(counts), size=args.effective_batch, replace=False)
            c = counts[idx]
            if c.sum() == 0:
                continue
            w = weights_for_grouping(c, bs)
            true_w = c / c.sum()          # token-weighted batch loss
            ratio = w / true_w            # per-row weight distortion
            devs.append(np.abs(ratio - 1.0).mean())
            maxes.append(ratio.max())
        results[f"b{bs}"] = {
            "grad_accum": args.effective_batch // bs,
            "mean_abs_deviation": float(np.mean(devs)),
            "mean_max_overweight": float(np.mean(maxes)),
        }
        print(f"{bs:>12}{args.effective_batch // bs:>15}"
              f"{np.mean(maxes):>29.3f}x{np.mean(devs)*100:>22.1f}%")

    b2, b4 = results.get("b2"), results.get("b4")
    if b2 and b4:
        print(f"\nb2/ga16 distorts per-token weighting by "
              f"{b2['mean_abs_deviation']*100:.1f}% on average; "
              f"b4/ga8 by {b4['mean_abs_deviation']*100:.1f}%.")
        better = (1 - b4["mean_abs_deviation"] / b2["mean_abs_deviation"]) * 100
        print(f"=> b4/ga8 is {better:.0f}% CLOSER to the true token-weighted "
              f"objective, not further from it.")
        print("\nSo the speedup does not come at a quality cost from this mechanism;"
              "\nbut the two configs are NOT the same gradient, and neither one is"
              "\nthe true batch loss. Both differ from it; the bigger micro-batch"
              "\ndiffers less. Actual quality effect is UNMEASURED.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"issue": 141, "config": vars(args),
                   "rows": len(counts),
                   "answer_tokens_per_row": {
                       "mean": float(counts.mean()), "sd": float(counts.std()),
                       "min": float(counts.min()), "max": float(counts.max()),
                       "cv": float(counts.std() / counts.mean())},
                   "results": results}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
