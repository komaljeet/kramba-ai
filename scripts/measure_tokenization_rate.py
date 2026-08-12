"""How long does tokenizing the 11.2 GB code corpus actually take? (#157)

Author: Rambarun Komaljeet

This cost has never been measured, and a stale cache hid it: a run pointed at
`stack_code.txt` (11.20 GB) announced that path and then loaded a 121M-token
cache built from a different file, so tokenization appeared instantaneous. With
the cache correctly keyed on the source corpus (#157), the real cost has to be
paid -- and it gates whether a multi-day run can start today or tomorrow.

Measures on a SLICE and extrapolates, rather than running the full job to find
out. `_tokenize_text_to_bin` streams fixed-size character blocks and encodes
each with a worker pool, so throughput is flat in corpus size -- the extrapolation
is linear and safe, which is exactly why a slice suffices.

Reports MB/s and tokens/s, the projected wall clock for the full corpus, and the
resulting token count against the 3.14B corpus plan -- because "how long" and
"how many tokens do I get" are the two numbers that decide the run.

Usage::

    python scripts/measure_tokenization_rate.py --slice-mb 200
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data_cache/code_corpus/stack_code.txt")
    p.add_argument("--tokenizer", default="data_cache/tokenizer_32k_fim")
    p.add_argument("--slice-mb", type=int, default=200)
    p.add_argument("--fim-rate", type=float, default=0.5)
    p.add_argument("--fim-spm-rate", type=float, default=0.5)
    p.add_argument("--plan-tokens", type=float, default=3.14e9,
                   help="corpus_plan.json target, for the shortfall check")
    p.add_argument("--out", default="results/tokenization/rate.json")
    args = p.parse_args()

    import train_edge_conversational as tec

    size = os.path.getsize(args.corpus)
    print(f"corpus {args.corpus}: {size/1024**3:.2f} GB")
    tok = tec.load_tokenizer(args.tokenizer)
    print(f"tokenizer {args.tokenizer}: {len(tok):,} ids "
          f"({os.cpu_count()} cores available to the encode pool)")

    # A slice from the MIDDLE: the head of a corpus assembled by language is
    # not representative -- an all-Python opening would tokenize at a different
    # rate than the Java or minified-JS further in.
    n_bytes = args.slice_mb * 1024 * 1024
    start = max(0, size // 2 - n_bytes // 2)
    with open(args.corpus, "r", encoding="utf-8", errors="replace") as fh:
        fh.seek(start)
        text = fh.read(n_bytes)
    slice_bytes = len(text.encode("utf-8", errors="replace"))

    tmp_txt = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False,
                                          encoding="utf-8")
    tmp_txt.write(text)
    tmp_txt.close()
    tmp_bin = tmp_txt.name + ".bin"

    fim = {"rate": args.fim_rate, "spm_rate": args.fim_spm_rate, "seed": 0}
    print(f"\ntokenizing a {slice_bytes/1024**2:.0f} MB slice from the middle, "
          f"FIM rate={args.fim_rate} spm={args.fim_spm_rate} ...")
    t0 = time.perf_counter()
    n_tokens = tec._tokenize_text_to_bin(tmp_txt.name, tmp_bin, tok, fim=fim)
    elapsed = time.perf_counter() - t0

    mb_s = (slice_bytes / 1024**2) / elapsed
    tok_s = n_tokens / elapsed
    tokens_per_byte = n_tokens / slice_bytes
    full_tokens = tokens_per_byte * size
    full_hours = (size / 1024**2) / mb_s / 3600

    print(f"\n  {n_tokens:,} tokens in {elapsed:.1f}s")
    print(f"  {mb_s:.1f} MB/s | {tok_s:,.0f} tokens/s | "
          f"{tokens_per_byte:.4f} tokens/byte")
    print(f"\nprojected for the full {size/1024**3:.2f} GB corpus:")
    print(f"  tokens     : {full_tokens/1e9:.2f}B")
    print(f"  wall clock : {full_hours:.1f} hours")
    print(f"  vs plan    : {full_tokens/args.plan_tokens*100:.0f}% of the "
          f"{args.plan_tokens/1e9:.2f}B target")

    if full_hours > 6:
        verdict = (f"TOKENIZE FIRST, SEPARATELY: {full_hours:.1f} h is too long to "
                   f"sit in front of a training launch. Run it as its own job, "
                   f"confirm the token count, then start training against a cache "
                   f"that is already correct.")
    elif full_hours > 1:
        verdict = (f"{full_hours:.1f} h of tokenization before the first step. "
                   f"Acceptable inside a multi-day run, but expect silence for "
                   f"that long -- which is precisely the silence that has been "
                   f"misread as a hang repeatedly.")
    else:
        verdict = (f"{full_hours*60:.0f} min -- small enough to run inline.")
    print(f"\nVERDICT: {verdict}")

    for f in (tmp_txt.name, tmp_bin, tmp_bin + ".tokenizer"):
        try:
            os.remove(f)
        except OSError:
            pass

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 157, "corpus": args.corpus,
                   "corpus_bytes": size, "slice_bytes": slice_bytes,
                   "slice_tokens": n_tokens, "seconds": elapsed,
                   "mb_per_s": mb_s, "tokens_per_s": tok_s,
                   "tokens_per_byte": tokens_per_byte,
                   "projected_full_tokens": full_tokens,
                   "projected_hours": full_hours,
                   "verdict": verdict}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
