"""Add the three FIM sentinels to the 32K vocabulary as real tokens (#136).

Written by: Rambarun Komaljeet

With the stock 32K vocabulary each sentinel encodes to EIGHT tokens::

    '<|fim_prefix|>' -> [30, 94, 72, 329, 65, 7270, 94, 32]

That is 24 tokens of pure overhead on every FIM document, and worse, it asks the
model to learn that a particular 8-token sequence is structural punctuation
rather than text. Both problems disappear if the sentinels are single tokens,
which is what StarCoder and CodeLlama do.

**This changes vocab_size 32000 -> 32003**, and that is deliberate rather than
incidental:

  - The parameter cost is 3 x d_model. At d=896 that is 2,688 parameters out of
    157M -- far below the noise floor of any measurement in this repo.
  - It makes the FIM tokenizer a DIFFERENT tokenizer, which is correct. #139's
    `.tokenizer` sidecar on the .bin cache will refuse to reuse ids produced by
    the old one, exactly as intended -- a model trained on 32003 ids must never
    be served with the 32000 vocabulary.
  - EOS stays 0, so `resolve_eos_id` (#136) keeps working unchanged.

Written to a new directory rather than in place: the pre-FIM tokenizer produced
the measurements in docs/state.md (the code A/B, the mixed A/B, the size sweep),
and overwriting it would make those unreproducible.

Usage
-----
    python scripts/build_fim_tokenizer.py
    python scripts/build_fim_tokenizer.py --verify-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.fim import FIM_TOKENS  # noqa: E402

SRC = "data_cache/tokenizer_32k"
DST = "data_cache/tokenizer_32k_fim"


def verify(path: str) -> int:
    """Check each sentinel is exactly one token, and that EOS is unchanged."""
    from train_edge_conversational import load_tokenizer
    tok = load_tokenizer(path)
    print(f"[fim-tok] {path}: vocab_size={tok.vocab_size}, "
          f"eos={tok.eos_token!r} id={tok.eos_token_id}")
    bad = 0
    for t in FIM_TOKENS:
        ids = tok.encode(t)
        ok = len(ids) == 1
        print(f"[fim-tok]   {t:<16} -> {len(ids)} token(s) {ids}"
              f"{'' if ok else '   <-- NOT A SINGLE TOKEN'}")
        bad += not ok
    # A sentinel that round-trips to different text would corrupt every document
    # it appears in, silently.
    for t in FIM_TOKENS:
        back = tok.decode(tok.encode(t))
        if back.strip() != t:
            print(f"[fim-tok]   ROUND-TRIP BROKEN: {t!r} -> {back!r}")
            bad += 1
    if tok.eos_token_id != 0:
        print(f"[fim-tok]   EOS moved to {tok.eos_token_id} (was 0) -- every "
              f"hardcoded assumption about it is now wrong")
        bad += 1

    # The check that matters most, and the one this script originally missed.
    # HuggingFace excludes ADDED tokens from .vocab_size but emits their ids, so
    # .vocab_size reads 32000 while a FIM document contains 32002. A model sized
    # from .vocab_size gets a 32000-row embedding and dies on the first FIM
    # batch with a CUDA assert naming an embedding kernel -- minutes into a
    # multi-day run, saying nothing about tokenizers.
    from hvn.vocab import resolve_vocab_size
    base = tok.vocab_size
    effective = resolve_vocab_size(tokenizer=tok)
    max_id = max(max(tok.encode(t)) for t in FIM_TOKENS)
    print(f"[fim-tok]   vocab_size={base}  len={len(tok)}  "
          f"effective={effective}  max sentinel id={max_id}")
    if max_id >= effective:
        print(f"[fim-tok]   ID OVERFLOW: sentinel id {max_id} >= effective "
              f"vocab {effective}")
        bad += 1
    if effective <= base and len(tok) > base:
        print(f"[fim-tok]   resolve_vocab_size ignored the added tokens")
        bad += 1
    print(f"[fim-tok]   -> build the model with vocab_size={effective}, "
          f"NOT {base}")
    return bad


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=SRC)
    ap.add_argument("--dst", default=DST)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    if args.verify_only:
        return 1 if verify(args.dst) else 0

    if not os.path.isdir(args.src):
        raise SystemExit(f"source tokenizer not found: {args.src}")
    if os.path.isdir(args.dst):
        print(f"[fim-tok] {args.dst} exists, replacing")
        shutil.rmtree(args.dst)

    from tokenizers import ByteLevelBPETokenizer

    vocab = os.path.join(args.src, "vocab.json")
    merges = os.path.join(args.src, "merges.txt")
    if not (os.path.isfile(vocab) and os.path.isfile(merges)):
        raise SystemExit(f"{args.src} has no vocab.json + merges.txt")

    # The sentinels are registered as ADDED tokens, not merged into the BPE
    # vocabulary. That is the important distinction: the ByteLevel pre-tokenizer
    # would otherwise split "<|fim_prefix|>" into bytes before BPE ever saw it,
    # so an entry in vocab.json alone would never match. Added tokens bypass the
    # pre-tokenizer, which is exactly what a structural marker needs.
    backend = ByteLevelBPETokenizer(vocab, merges)
    tk = getattr(backend, "_tokenizer", backend)
    before = tk.get_vocab_size()
    added = tk.add_special_tokens(list(FIM_TOKENS))
    after = tk.get_vocab_size()
    print(f"[fim-tok] vocab {before} -> {after} (added {added} sentinel(s))")

    os.makedirs(args.dst, exist_ok=True)
    # Keep vocab.json/merges.txt so the directory still loads the old way, and
    # add tokenizer.json which carries the added tokens. load_tokenizer prefers
    # tokenizer.json when present.
    for name in os.listdir(args.src):
        s = os.path.join(args.src, name)
        if os.path.isfile(s):
            shutil.copy2(s, os.path.join(args.dst, name))
    tk.save(os.path.join(args.dst, "tokenizer.json"))
    print(f"[fim-tok] wrote {args.dst}")

    bad = verify(args.dst)
    if bad:
        print(f"[fim-tok] FAILED verification ({bad} problem(s))")
        return 1
    print("[fim-tok] verification passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
