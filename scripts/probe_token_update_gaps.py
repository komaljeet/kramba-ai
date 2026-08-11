"""How long between updates for the same token? (Issue #151 follow-up gate)

Author: Rambarun Komaljeet

Per-token optimizer-state retention only has *quality* content if a token's
momentum is still worth something when that token next appears. Momentum with
beta=0.95 has a ~14-step half-life, so the question is entirely about the
distribution of **gaps between appearances**, which is a property of the DATA and
needs no GPU and no training run.

An embedding row receives a gradient only on steps where its token occurs. A
batch of 4x512 touches a couple of thousand distinct ids out of 50,257, so most
rows are idle most of the time -- their momentum decays toward nothing whether or
not any block scheduler is involved.

Two separable claims, and this gate speaks to the first only:

* **quality** -- retention helps if a large share of appearances have a SHORT gap
  (momentum still alive). If gaps are long, restoring momentum restores noise,
  exactly as measured for BCD at 100 steps/block (#151).
* **memory** -- keeping state only for rows touched this step is a real saving
  regardless of the above, and is NOT tested here. The embedding + decoder sit in
  the AdamW group (two fp32 moments), so at 50257x512 that is ~206 MB of
  optimizer state to hold for rows that are ~96% idle per step.

Usage::

    python scripts/probe_token_update_gaps.py --steps 2000
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--beta", type=float, default=0.95,
                   help="momentum coefficient whose half-life sets 'still alive'")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results/token_gaps/gaps.json")
    args = p.parse_args()

    import torch
    import train_edge_conversational as tec
    from scripts.ab_prodigy import as_pair

    tok = tec.load_tokenizer(args.tokenizer)
    torch.manual_seed(args.seed)
    loader, _v, _s = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)

    V = tok.vocab_size
    last_seen = np.full(V, -1, dtype=np.int64)
    gaps: list[int] = []
    touched_per_step: list[int] = []
    first_time = 0

    for step, batch in enumerate(loader, start=1):
        x, _y = as_pair(batch)
        ids = np.unique(x.numpy())
        touched_per_step.append(len(ids))
        prev = last_seen[ids]
        seen_before = prev >= 0
        gaps.extend((step - prev[seen_before]).tolist())
        first_time += int((~seen_before).sum())
        last_seen[ids] = step
        if step >= args.steps:
            break

    g = np.asarray(gaps, dtype=np.float64)
    half_life = np.log(0.5) / np.log(args.beta)
    alive = float((g <= half_life).mean() * 100)
    quarter = float((g <= 2 * half_life).mean() * 100)
    # Expected surviving momentum fraction at the moment a token returns.
    surviving = float(np.mean(args.beta ** g))

    print(f"\n[gaps] {args.steps} steps, batch {args.batch_size} x seq {args.seq_len}, "
          f"vocab {V:,}")
    print(f"[gaps] distinct ids touched per step: mean {np.mean(touched_per_step):.0f} "
          f"({np.mean(touched_per_step)/V*100:.2f}% of vocab)")
    print(f"[gaps] appearances measured: {len(g):,}  (first-time: {first_time:,})")
    print(f"\n[gaps] gap between appearances of the same token (steps):")
    for q in (50, 75, 90, 95, 99):
        print(f"[gaps]   p{q}: {np.percentile(g, q):>10.1f}")
    print(f"[gaps]   mean: {g.mean():>9.1f}   max: {g.max():>9.0f}")
    print(f"\n[gaps] momentum half-life at beta={args.beta}: {half_life:.1f} steps")
    print(f"[gaps] appearances with gap <= 1 half-life : {alive:.1f}%")
    print(f"[gaps] appearances with gap <= 2 half-lives: {quarter:.1f}%")
    print(f"[gaps] MEAN SURVIVING MOMENTUM on return (beta^gap): {surviving:.4f}")

    if surviving > 0.25:
        verdict = (f"WORTH TESTING: an average {surviving:.2f} of momentum survives to "
                   f"the next appearance, so per-token retention has real content.")
    elif alive > 50:
        verdict = (f"MIXED: mean surviving momentum is only {surviving:.3f}, but "
                   f"{alive:.0f}% of appearances are within one half-life -- the "
                   f"benefit would come from frequent tokens, which are also the ones "
                   f"needing it least.")
    else:
        verdict = (f"NO QUALITY CONTENT: only {surviving:.4f} of momentum survives to "
                   f"the average next appearance and just {alive:.1f}% of appearances "
                   f"fall inside one half-life. Restoring per-token momentum restores "
                   f"noise -- the same arithmetic that made BCD retention worthless at "
                   f"100 steps/block (#151). The MEMORY case is untouched by this and "
                   f"remains open.")
    print(f"\n[gaps] VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 151, "config": vars(args), "vocab": V,
                   "mean_touched_per_step": float(np.mean(touched_per_step)),
                   "appearances": len(g), "first_time": first_time,
                   "percentiles": {f"p{q}": float(np.percentile(g, q))
                                   for q in (50, 75, 90, 95, 99)},
                   "mean_gap": float(g.mean()), "max_gap": float(g.max()),
                   "half_life_steps": float(half_life),
                   "pct_within_one_half_life": alive,
                   "mean_surviving_momentum": surviving,
                   "verdict": verdict}, fh, indent=2)
    print(f"[gaps] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
