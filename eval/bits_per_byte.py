"""Bits per byte — the only fair metric across different tokenizers (#139).

Written by: Rambarun Komaljeet

**Cross-entropy loss is not comparable between models with different
vocabularies.** Loss is per *token*, and a 32K model is choosing among 32,000
options where a 50,257 model chooses among 50,257 — a strictly easier problem.
The smaller-vocab model posts a lower number regardless of whether it is any
better, and the two also disagree on how many tokens a document even is.

Bits per byte removes both effects by normalising against the underlying TEXT:

    bits/byte = (sum of token NLL in nats) / (ln 2 * bytes of source text)

The numerator is the model's total surprise over a passage; the denominator is a
property of the passage alone. Two models can then be compared on the same
English, whatever their vocabularies.

This is the standard measure for exactly this comparison, and without it #139's
A/B would have produced a confident number that meant nothing.

Usage
-----
    python eval/bits_per_byte.py --checkpoint ckpt.pt --text data_cache/heldout.txt
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

from train_edge_conversational import (  # noqa: E402
    load_checkpoint_for_resume, load_tokenizer,
)


@torch.no_grad()
def bits_per_byte(model, tokenizer, text: str, seq_len: int = 512,
                  device=None, max_windows: int | None = None) -> dict:
    """Total NLL over ``text`` divided by its byte length.

    Scored in non-overlapping windows. The first token of each window has no
    context and is skipped -- otherwise a model would be charged for predicting
    a token it was given no information about, and the penalty would fall
    differently on the two tokenizers (more windows = more unconditioned tokens,
    which would quietly favour the vocabulary that produces FEWER tokens).
    """
    device = device or next(model.parameters()).device
    ids = tokenizer.encode(text)
    n_bytes = len(text.encode("utf-8"))

    total_nll = 0.0
    scored_tokens = 0
    n_windows = 0
    for start in range(0, len(ids) - 1, seq_len):
        window = ids[start:start + seq_len + 1]
        if len(window) < 2:
            break
        x = torch.tensor([window[:-1]], device=device)
        y = torch.tensor([window[1:]], device=device)
        with torch.amp.autocast("cuda", enabled=(device.type == "cuda"),
                                dtype=torch.bfloat16):
            logits = model(x)
        nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                              y.reshape(-1), reduction="sum")
        total_nll += float(nll)
        scored_tokens += y.numel()
        n_windows += 1
        if max_windows and n_windows >= max_windows:
            break

    bpb = total_nll / (math.log(2) * n_bytes) if n_bytes else float("nan")
    return {
        "bits_per_byte": bpb,
        "total_nll_nats": total_nll,
        "n_tokens": len(ids),
        "n_scored_tokens": scored_tokens,
        "n_bytes": n_bytes,
        "n_windows": n_windows,
        "tokens_per_byte": len(ids) / max(n_bytes, 1),
        # Reported for continuity with this repo's other numbers, and labelled
        # NOT-COMPARABLE because it is per-token and therefore vocabulary-bound.
        "mean_token_nll_NOT_COMPARABLE": total_nll / max(scored_tokens, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--text", required=True,
                    help="held-out RAW TEXT (not tokenized) — the whole point is "
                         "that both arms see identical bytes")
    ap.add_argument("--tokenizer", default=None,
                    help="defaults to the checkpoint's own vocab_size: 'gpt2' for "
                         "50257, else data_cache/tokenizer_32k")
    ap.add_argument("--max-bytes", type=int, default=2_000_000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, step, _ck = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()

    spec = args.tokenizer
    if spec is None:
        v = getattr(model, "vocab_size", 50257)
        spec = "gpt2" if v == 50257 else "data_cache/tokenizer_32k"
    tok = load_tokenizer(spec)
    if tok.vocab_size != getattr(model, "vocab_size", tok.vocab_size):
        raise SystemExit(
            f"tokenizer vocab {tok.vocab_size} != model vocab "
            f"{model.vocab_size}. Scoring with the wrong tokenizer produces a "
            f"number that looks fine and means nothing.")

    with open(args.text, encoding="utf-8", errors="replace") as fh:
        text = fh.read(args.max_bytes)

    res = bits_per_byte(model, tok, text, seq_len=args.seq_len, device=device)
    res.update(checkpoint=args.checkpoint.replace("\\", "/"), tokenizer=spec,
               vocab_size=tok.vocab_size, step=step)

    print(f"[bpb] {os.path.basename(args.checkpoint)} | vocab {tok.vocab_size}")
    print(f"[bpb]   bits/byte      : {res['bits_per_byte']:.4f}   <- COMPARABLE")
    print(f"[bpb]   tokens/byte    : {res['tokens_per_byte']:.4f}")
    print(f"[bpb]   mean token NLL : {res['mean_token_nll_NOT_COMPARABLE']:.4f}"
          f"   <- NOT comparable across vocabs")
    print(f"[bpb]   {res['n_scored_tokens']:,} tokens over {res['n_bytes']:,} bytes")

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=2)
        print(f"[bpb] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
