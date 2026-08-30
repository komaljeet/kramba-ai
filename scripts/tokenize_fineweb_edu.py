"""Tokenize FineWeb-Edu into the .bin cache, standalone (Issue #163).

Why this exists as its own script
---------------------------------
Tokenization is normally triggered inside `build_dataset_pipeline`, which means
the only way to run it was to start a training run -- fine for a 5 GB corpus,
not for a multi-hour pass over 156 GB where the training would then begin
unattended on whatever config happened to be on the command line.

Measured on this machine 2026-08-27 (`--arch` irrelevant, this is data only):

  4.59 text bytes per token          (400 real FineWeb-Edu docs, GPT-2)
  ~667M tokens per 2.00 GB parquet file
  117 files on disk  ->  ~78B tokens available

  target                tokens    .txt     .bin    peak disk
  478M  @ 30 tok/param   14.3B    61 GB    27 GB      88 GB
  1.21B @ 30 tok/param   36.4B   156 GB    68 GB     224 GB   (293 GB free)

The .txt is an intermediate. Once the .bin exists the pipeline never reads it
again -- `_tokenize_text_to_bin`'s cache is keyed on the .bin -- so
`--drop-text` reclaims that space afterwards. Keep it only if you expect to
re-tokenize with a different tokenizer.

Falsifiability
--------------
The pass is worthless if it produces a .bin the trainer will not use, or one
that is silently short. So this verifies AFTER writing: the .bin exists, its
size implies the expected token count at 2 bytes/token (uint16), and it decodes
back to readable text. A byte count alone would pass on a file of zeros.

Usage::

    python scripts/tokenize_fineweb_edu.py                 # 1.21B target
    python scripts/tokenize_fineweb_edu.py --target-tokens 14.4e9 --drop-text
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_edge_conversational as tec  # noqa: E402

BYTES_PER_TOKEN = 4.59      # measured, see the module docstring
BIN_BYTES_PER_TOKEN = 2     # uint16


def free_gb(path: str = ".") -> float:
    return shutil.disk_usage(path).free / 2**30


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--fineweb-dir", default=None,
                    help="Defaults to <data-dir>/fineweb_edu")
    ap.add_argument("--target-tokens", type=float, default=36.44e9,
                    help="Tokens to make available. Default 36.44e9 = a 1.21B "
                         "model at the 30 tokens/param #159 measured. Use "
                         "14.4e9 for a 478M model.")
    ap.add_argument("--tokenizer", default="gpt2")
    ap.add_argument("--drop-text", action="store_true",
                    help="Delete the intermediate .txt once the .bin is "
                         "verified, reclaiming ~4.6 bytes per token.")
    ap.add_argument("--yes", action="store_true",
                    help="Skip the disk-space confirmation.")
    args = ap.parse_args()

    fineweb_dir = args.fineweb_dir or os.path.join(args.data_dir, "fineweb_edu")
    cache_dir = args.data_dir
    tok_dir = os.path.join(cache_dir, "tokenized")
    os.makedirs(tok_dir, exist_ok=True)
    bin_path = os.path.join(tok_dir, "fineweb_edu.bin")

    max_bytes = int(args.target_tokens * BYTES_PER_TOKEN)
    est_txt = max_bytes / 2**30
    est_bin = args.target_tokens * BIN_BYTES_PER_TOKEN / 2**30
    have = free_gb(cache_dir)

    print("=" * 72)
    print("FineWeb-Edu tokenization")
    print("=" * 72)
    print(f"  target        {args.target_tokens/1e9:.2f}B tokens")
    print(f"  source        {fineweb_dir}")
    print(f"  output        {bin_path}")
    print(f"  est .txt      {est_txt:.0f} GB   (intermediate"
          f"{', deleted after' if args.drop_text else ', KEPT'})")
    print(f"  est .bin      {est_bin:.0f} GB")
    print(f"  peak needed   {est_txt + est_bin:.0f} GB")
    print(f"  free now      {have:.0f} GB")

    if os.path.exists(bin_path):
        n = os.path.getsize(bin_path) // BIN_BYTES_PER_TOKEN
        print(f"\n  {bin_path} already exists: {n:,} tokens "
              f"({os.path.getsize(bin_path)/2**30:.1f} GB).")
        print("  The pipeline caches on the .bin existing, so this is a no-op.")
        print("  Delete it first if you want to re-tokenize.")
        return 0

    if est_txt + est_bin > have * 0.9 and not args.yes:
        print(f"\n  REFUSING: this needs ~{est_txt + est_bin:.0f} GB and only "
              f"{have:.0f} GB is free (90% guard).")
        print(f"  Lower --target-tokens, use --drop-text, or pass --yes.")
        print(f"  For reference, a 478M model needs --target-tokens 14.4e9 "
              f"(~88 GB).")
        return 2

    if not os.path.isdir(fineweb_dir):
        print(f"\n  No FineWeb-Edu at {fineweb_dir}. Run "
              f"scripts/download_fineweb_edu.py first.")
        return 2

    print("\n[1/3] Extracting parquet -> flat text ...", flush=True)
    t0 = time.time()
    txt_path = tec._load_fineweb_edu(cache_dir, fineweb_dir, max_bytes=max_bytes)
    if txt_path is None:
        print("  no parquet files found -- nothing to do.")
        return 2
    got = os.path.getsize(txt_path)
    print(f"  {txt_path}: {got/2**30:.1f} GB in {(time.time()-t0)/60:.1f} min")

    print("\n[2/3] Tokenizing -> uint16 .bin ...", flush=True)
    t1 = time.time()
    enc = tec.load_tokenizer(args.tokenizer)
    tec._tokenize_text_to_bin(txt_path, bin_path, enc, max_bytes=max_bytes)
    print(f"  done in {(time.time()-t1)/60:.1f} min")

    print("\n[3/3] Verifying ...", flush=True)
    if not os.path.exists(bin_path):
        print("  FAILED: no .bin was written."); return 1
    size = os.path.getsize(bin_path)
    n_tokens = size // BIN_BYTES_PER_TOKEN
    print(f"  {bin_path}: {size/2**30:.1f} GB = {n_tokens:,} tokens")

    # A size check alone would pass on a file of zeros. Decode a slice and
    # require it to look like text.
    import numpy as np
    arr = np.memmap(bin_path, dtype=np.uint16, mode="r")
    sample = arr[1000:1200].tolist()
    try:
        text = enc.decode(sample)
    except Exception as e:                       # pragma: no cover
        print(f"  FAILED: could not decode a sample: {e}"); return 1
    printable = sum(c.isprintable() or c.isspace() for c in text) / max(len(text), 1)
    print(f"  sample decode: {text[:90]!r}")
    if printable < 0.9:
        print(f"  FAILED: only {printable:.0%} of the decoded sample is "
              f"printable -- the .bin does not contain sane token ids.")
        return 1
    if n_tokens < args.target_tokens * 0.5:
        print(f"  WARNING: got {n_tokens/1e9:.1f}B tokens against a "
              f"{args.target_tokens/1e9:.1f}B target -- the source may be "
              f"smaller than assumed. The trainer's budget guard will tell you "
              f"whether this is enough for your model size.")
    print(f"  OK: {printable:.0%} printable, decodes cleanly.")

    if args.drop_text:
        freed = os.path.getsize(txt_path) / 2**30
        os.remove(txt_path)
        print(f"\n  removed {txt_path} (+{freed:.0f} GB free)")

    print(f"\n  free disk now: {free_gb(cache_dir):.0f} GB")
    print(f"\nDone. Use it with:  --pretrain-mix 0.0,0.0,0.0,1.0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
