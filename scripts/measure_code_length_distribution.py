"""Are The Stack's files actually long? (Issue #135, step 2)

Author: Rambarun Komaljeet

#135's own instruction, before any training: *"Measure the length distribution
first, as #132 did before packing. If the corpus turns out to be mostly short
files, this whole issue is unfounded and should be closed on that measurement
rather than pursued."*

The premise of long-context training here is that code files contain real
long-range structure -- a function used thousands of tokens after it is defined.
That is exactly what the existing corpora cannot provide: `grounded_qa_expanded`
has a **median of 86 tokens** and OASST **359** (#132), so packing puts ~5
unrelated documents in a 512-token row behind a block-diagonal mask, and a longer
window would have nothing legitimate to attend to.

This measures, with the real tokenizer, what fraction of the corpus is long
enough to justify each candidate training length.

**Falsifiable, in #135's own words:** if the median file is short, the
long-context plan is unfounded and the issue should be closed on this number.

Usage::

    python scripts/measure_code_length_distribution.py \
        --corpus data_cache/code_corpus/stack_code.txt \
        --tokenizer data_cache/tokenizer_32k_fim
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

#: Documents in the corpus are separated by this marker (see
#: scripts/extract_the_stack.py). Checked at runtime rather than assumed --
#: a wrong separator would silently measure one giant "file".
DEFAULT_SEP = "<|endoftext|>"


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data_cache/code_corpus/stack_code.txt")
    p.add_argument("--tokenizer", default="data_cache/tokenizer_32k_fim")
    p.add_argument("--separator", default=DEFAULT_SEP)
    p.add_argument("--max-bytes", type=int, default=400_000_000,
                   help="sample cap; the full 12 GB is not needed for a "
                        "distribution and would take far longer than the "
                        "10 minutes this check is supposed to cost")
    p.add_argument("--out", default="results/code_length/distribution.json")
    args = p.parse_args()

    if not os.path.exists(args.corpus):
        sys.exit(f"corpus not found: {args.corpus}")

    size = os.path.getsize(args.corpus)
    print(f"[len] corpus {args.corpus} ({size / 1024**3:.2f} GB)")
    print(f"[len] sampling first {args.max_bytes / 1024**3:.2f} GB")

    with open(args.corpus, "r", encoding="utf-8", errors="replace") as fh:
        text = fh.read(args.max_bytes)

    docs = [d for d in text.split(args.separator) if d.strip()]
    # Drop the final fragment: a byte-capped read almost certainly truncates
    # mid-file, and a truncated tail would bias the distribution downward.
    if len(docs) > 1:
        docs = docs[:-1]
    print(f"[len] {len(docs):,} documents (separator {args.separator!r})")
    if len(docs) < 10:
        sys.exit(f"only {len(docs)} documents found -- the separator is probably "
                 f"wrong, and measuring one giant blob would be meaningless")

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.tokenizer)
    print(f"[len] tokenizer {args.tokenizer} (vocab {tok.vocab_size:,})")

    lens = []
    for i, d in enumerate(docs):
        lens.append(len(tok(d, add_special_tokens=False)["input_ids"]))
        if (i + 1) % 2000 == 0:
            print(f"[len]   tokenized {i+1:,}/{len(docs):,}", flush=True)
    L = np.asarray(lens)

    print(f"\n[len] token length per file, n={len(L):,}")
    for q in (10, 25, 50, 75, 90, 95, 99):
        print(f"[len]   p{q:<3}{np.percentile(L, q):>12,.0f}")
    print(f"[len]   mean{L.mean():>12,.0f}   max{L.max():>12,.0f}")

    print(f"\n[len] share of FILES at least as long as a candidate training length,")
    print(f"[len] and share of TOKENS living in files that long:")
    print(f"[len] {'length':>8}{'% files':>10}{'% tokens':>11}")
    total = L.sum()
    shares = {}
    for n in (512, 1024, 2048, 4096, 8192, 16384):
        f_share = float((L >= n).mean() * 100)
        t_share = float(L[L >= n].sum() / total * 100)
        shares[n] = {"pct_files": f_share, "pct_tokens": t_share}
        print(f"[len] {n:>8}{f_share:>9.1f}%{t_share:>10.1f}%")

    med = float(np.median(L))
    # The comparison #135 cares about: the corpora that made long context
    # impossible had medians of 86 and 359 tokens.
    if med < 512:
        verdict = (f"UNFOUNDED: median file is {med:,.0f} tokens, below a 512 "
                   f"training length. Per #135's own acceptance criterion the "
                   f"long-context plan should be closed on this measurement.")
    elif shares[2048]["pct_tokens"] < 25:
        verdict = (f"WEAK: median {med:,.0f} tokens, but only "
                   f"{shares[2048]['pct_tokens']:.0f}% of tokens live in files "
                   f">=2048. Training at 2-4K would spend most of its capacity on "
                   f"documents that do not need it.")
    else:
        verdict = (f"FOUNDED: median {med:,.0f} tokens, and "
                   f"{shares[2048]['pct_tokens']:.0f}% of tokens live in files "
                   f">=2048 ({shares[4096]['pct_tokens']:.0f}% in >=4096). There is "
                   f"real long-range structure to train on -- versus medians of 86 "
                   f"(grounded-QA) and 359 (OASST) for the existing corpora.")
    print(f"\n[len] VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 135, "corpus": args.corpus,
                   "corpus_bytes": size, "sampled_bytes": min(size, args.max_bytes),
                   "documents": len(L), "tokenizer": args.tokenizer,
                   "percentiles": {f"p{q}": float(np.percentile(L, q))
                                   for q in (10, 25, 50, 75, 90, 95, 99)},
                   "mean": float(L.mean()), "max": int(L.max()),
                   "shares": shares, "verdict": verdict}, fh, indent=2)
    print(f"[len] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
