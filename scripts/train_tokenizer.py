"""Train a mixed English+code BPE tokenizer and measure what it costs (#139).

Written by: Rambarun Komaljeet

A tokenizer is a **one-way door**: a model cannot be fine-tuned onto a different
one, so changing it later means retraining from scratch. It therefore has to be
chosen for the final target (#135 long-context, #136 code autocomplete), not for
today's corpus — hence English AND code, not English alone.

Uses `tokenizers`, which ships with `transformers`. No new dependency.

## The number this script exists to produce

Shrinking the vocab 50257 -> 32000 removes 14.7% of forward FLOPs (the output
head is a `d_model x vocab` GEMM per token). But a smaller vocab means **more
tokens per document**, and every extra token costs a full forward pass. If 32K
inflates sequences by more than ~15%, the FLOP saving is cancelled and only the
memory win survives.

So `--measure` reports tokens-per-document against GPT-2 BPE on the real
corpora. That is acceptance criterion 2 of #139, and it is measured BEFORE any
training run rather than assumed.

## Why plain ByteLevel BPE handles indentation

#136 flags GPT-2 BPE as bad for code because leading whitespace fragments. The
cause is GPT-2's *training data*, not its algorithm: ByteLevel BPE will happily
learn `"\\n    "` and `"\\n        "` as single tokens if it sees enough code.
Training on a mixed corpus is therefore the fix, and the `.py` tokens-per-line
number below is what proves it rather than assuming it.

Usage
-----
    python scripts/train_tokenizer.py --vocab-size 32000
    python scripts/train_tokenizer.py --measure-only
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CODE_TXT = "data_cache/code_corpus/code_all.txt"
ENGLISH_TXT = "data_cache/slimpajama_en.txt"
CHAT_TXT = "data_cache/oasst1.txt"


def build_mixed_corpus(out_path: str, code_mb: int, english_mb: int,
                       chat_mb: int) -> dict:
    """Concatenate a fixed-size sample of each source.

    Sizes are explicit MB rather than fractions because the sources differ by
    two orders of magnitude (SlimPajama is 5.4 GB, the code corpus 125 MB), and
    a fraction-based mix would hide how lopsided that is.
    """
    parts = [("code", CODE_TXT, code_mb), ("english", ENGLISH_TXT, english_mb),
             ("chat", CHAT_TXT, chat_mb)]
    written = {}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8", errors="ignore") as out:
        for name, src, mb in parts:
            if mb <= 0:
                continue
            if not os.path.exists(src):
                print(f"[tok] WARNING missing {src} — skipping {name}")
                continue
            budget = mb * 1024 ** 2
            got = 0
            with open(src, encoding="utf-8", errors="ignore") as fh:
                while got < budget:
                    chunk = fh.read(min(8 * 1024 ** 2, budget - got))
                    if not chunk:
                        break
                    out.write(chunk)
                    got += len(chunk)
            out.write("\n")
            written[name] = got
            print(f"[tok] {name:>8}: {got/1024**2:6.1f} MB from {src}")
    total = sum(written.values())
    for k, v in written.items():
        print(f"[tok] mix share {k:>8}: {100*v/max(total,1):5.1f}%")
    return written


def train(corpus_path: str, vocab_size: int, out_path: str):
    from tokenizers import ByteLevelBPETokenizer

    tok = ByteLevelBPETokenizer()
    print(f"[tok] training vocab_size={vocab_size} on {corpus_path}")
    tok.train(
        files=[corpus_path], vocab_size=vocab_size, min_frequency=2,
        # <|endoftext|> keeps EOS at a known id, matching the EOS-terminated
        # chat format every checkpoint here was trained with.
        special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>"],
        show_progress=True,
    )
    os.makedirs(out_path, exist_ok=True)
    tok.save_model(out_path)
    print(f"[tok] saved -> {out_path}")
    return tok


def sample_lines(path: str, n: int, seed: int = 0) -> list[str]:
    """Reservoir-sample n lines so the measurement is not biased to the head."""
    rng = random.Random(seed)
    out: list[str] = []
    with open(path, encoding="utf-8", errors="ignore") as fh:
        for i, line in enumerate(fh):
            line = line.rstrip("\n")
            if not line.strip():
                continue
            if len(out) < n:
                out.append(line)
            else:
                j = rng.randint(0, i)
                if j < n:
                    out[j] = line
    return out


def measure(tok_dir: str, n: int = 4000) -> dict:
    """Tokens per line, new tokenizer vs GPT-2. >1.0 means LONGER sequences."""
    from tokenizers import ByteLevelBPETokenizer
    from transformers import GPT2TokenizerFast

    ours = ByteLevelBPETokenizer(os.path.join(tok_dir, "vocab.json"),
                                 os.path.join(tok_dir, "merges.txt"))
    gpt2 = GPT2TokenizerFast.from_pretrained("gpt2")

    sources = {
        "code (all)": CODE_TXT,
        "english": ENGLISH_TXT,
        "chat": CHAT_TXT,
        "grounded_qa": "data/grounded_qa_expanded.jsonl",
    }
    rows = {}
    print(f"\n[tok] {'source':<16} {'gpt2 tok/line':>14} {'ours tok/line':>14} "
          f"{'ratio':>8}")
    for name, path in sources.items():
        if not os.path.exists(path):
            continue
        lines = sample_lines(path, n)
        if not lines:
            continue
        g = sum(len(gpt2.encode(ln)) for ln in lines) / len(lines)
        o = sum(len(ours.encode(ln).ids) for ln in lines) / len(lines)
        rows[name] = {"gpt2": g, "ours": o, "ratio": o / max(g, 1e-9),
                      "n_lines": len(lines)}
        print(f"[tok] {name:<16} {g:>14.2f} {o:>14.2f} {o/max(g,1e-9):>8.3f}")
    return rows


def verdict(rows: dict, vocab_size: int) -> dict:
    """Does the FLOP saving survive the longer sequences?"""
    from math import isclose
    d_model, n_layers, ffn = 512, 12, 3
    per_block = 2 * (6 * d_model ** 2 + 2 * ffn * d_model ** 2)
    blocks = per_block * n_layers

    def total(v):
        return blocks + 2 * d_model * v

    flop_ratio = total(vocab_size) / total(50257)
    # Cost per DOCUMENT is (FLOPs per token) x (tokens per document).
    weighted = sum(r["ratio"] for r in rows.values()) / max(len(rows), 1)
    net = flop_ratio * weighted
    print(f"\n[tok] per-token FLOPs at vocab {vocab_size}: {flop_ratio:.3f}x "
          f"({100*(1-flop_ratio):.1f}% cheaper)")
    # Phrase the direction explicitly. "-11.2% longer" is not a sentence anyone
    # can read correctly at a glance, and this repo has already shipped one
    # inverted-sign summary line today (#139's params audit).
    direction = "SHORTER" if weighted < 1 else "longer"
    print(f"[tok] mean tokens-per-line ratio: {weighted:.3f}x "
          f"({abs(100*(weighted-1)):.1f}% {direction} sequences)")
    print(f"[tok] NET cost per document: {net:.3f}x "
          f"-> {'a real win' if net < 0.97 else 'CANCELLED — memory win only' if net < 1.03 else 'WORSE than GPT-2'}")
    return {"flop_ratio": flop_ratio, "token_ratio": weighted, "net": net}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vocab-size", type=int, default=32000)
    ap.add_argument("--code-mb", type=int, default=125)
    ap.add_argument("--english-mb", type=int, default=100)
    ap.add_argument("--chat-mb", type=int, default=25)
    ap.add_argument("--corpus", default="data_cache/tokenizer_mixed.txt")
    ap.add_argument("--out", default="data_cache/tokenizer_32k")
    ap.add_argument("--measure-only", action="store_true")
    ap.add_argument("--results", default="results/vocab/tokenizer.json")
    args = ap.parse_args()

    mix = {}
    if not args.measure_only:
        mix = build_mixed_corpus(args.corpus, args.code_mb, args.english_mb,
                                 args.chat_mb)
        train(args.corpus, args.vocab_size, args.out)

    rows = measure(args.out)
    v = verdict(rows, args.vocab_size)

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as fh:
        json.dump({"vocab_size": args.vocab_size, "mix_bytes": mix,
                   "tokens_per_line": rows, "verdict": v}, fh, indent=2)
    print(f"[tok] wrote {args.results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
