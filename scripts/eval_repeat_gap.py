"""Generalization gap across epochs for a --max-train-windows run (dropout A/B).

Author: Rambarun Komaljeet

For every ``edge_conversational_step*.pt`` in a run directory, scores three
fixed token sets, all in eval mode (dropout off), so the arms are compared on
the same footing:

- ``seen``:    windows from the capped TRAIN pool -- what the model repeats.
- ``unseen``:  windows from the same random draw over the corpus that the pool
               never contains or touches (see unseen_code_windows). Same
               language mix as the pool, so ``unseen - seen`` is the gap.
- ``tail``:    the project's pinned 400K-token held-out slice
               (eval/heldout_loss.py), for continuity with every other number.

Training loss from the CSV is NOT used for the gap: it is taken in train mode,
with dropout on in one arm and off in the other, so it is not comparable
across arms.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_edge_conversational as tec  # noqa: E402
from eval import heldout_loss as hl       # noqa: E402


def pool_and_unseen(bin_path: str, seq_len: int, n_pool: int, k: int):
    """Rebuild the trainer's exact capped pool (val_fraction 0) + unseen set."""
    total = os.path.getsize(bin_path) // 2
    n_win = (total - seq_len - 1) // seq_len + 1
    starts = np.arange(n_win, dtype=np.int64) * seq_len
    perm = np.random.RandomState(42).permutation(len(starts))
    sp = starts[perm]
    pool = tec.cap_code_windows(sp, n_pool, total)
    unseen = tec.unseen_code_windows(sp, n_pool, total, seq_len, k)
    return pool, unseen


@torch.no_grad()
def score_windows(model, arr, starts, seq_len, device) -> float:
    model.eval()
    tot, cnt = 0.0, 0
    for s in starts:
        chunk = np.asarray(arr[int(s):int(s) + seq_len + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1]).unsqueeze(0).to(device)
        y = torch.from_numpy(chunk[1:]).to(device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        tot += F.cross_entropy(logits[0].float(), y, reduction="sum").item()
        cnt += y.numel()
    return tot / cnt


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--sweep", required=True)
    p.add_argument("--bin", default="data_cache/tokenized/code.v32000.bin")
    p.add_argument("--tokenizer", default="data_cache/tokenizer_32k_fim")
    p.add_argument("--corpus", default="data_cache/code_corpus/stack_code.txt")
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--max-train-windows", type=int, required=True)
    p.add_argument("--k", type=int, default=98, help="windows per set (~400K tokens)")
    p.add_argument("--tokens-per-step", type=int, default=4096 * 8)
    p.add_argument("--out", required=True)
    a = p.parse_args()

    device = torch.device("cuda")
    arr = np.memmap(a.bin, dtype=np.uint16, mode="r")
    pool, unseen = pool_and_unseen(a.bin, a.seq_len, a.max_train_windows, a.k)
    seen = pool[:a.k]
    assert len(unseen) == a.k and not (set(seen.tolist()) & set(unseen.tolist()))

    from transformers import GPT2TokenizerFast
    tail_ids = hl.heldout_ids(a.corpus, GPT2TokenizerFast.from_pretrained(a.tokenizer))
    assert len(tail_ids) == 400_000, len(tail_ids)

    done = {}
    if os.path.exists(a.out):
        done = {r["step"]: r for r in json.load(open(a.out))["rows"]}
    paths = sorted(glob.glob(os.path.join(a.sweep, "edge_conversational_step*.pt")),
                   key=hl._step_of)
    for path in paths:
        step = hl._step_of(path)
        if step in done:
            continue
        model, step, _ = tec.load_checkpoint_for_resume(path, device)
        r = {"step": step,
             "tokens_seen": step * a.tokens_per_step,
             "passes": step * a.tokens_per_step / (a.max_train_windows * a.seq_len),
             "seen": score_windows(model, arr, seen, a.seq_len, device),
             "unseen": score_windows(model, arr, unseen, a.seq_len, device),
             "tail": hl.evaluate(model, tail_ids, 1024, device)["heldout_nll"]}
        r["gap"] = r["unseen"] - r["seen"]
        done[step] = r
        print(f"step {step:>6} passes {r['passes']:.2f}  seen {r['seen']:.4f}  "
              f"unseen {r['unseen']:.4f}  gap {r['gap']:+.4f}  tail {r['tail']:.4f}",
              flush=True)
        del model
        torch.cuda.empty_cache()
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump({"max_train_windows": a.max_train_windows, "k": a.k,
                       "rows": [done[s] for s in sorted(done)]}, fh, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
