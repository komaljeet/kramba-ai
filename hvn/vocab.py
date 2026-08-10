"""Vocabulary identity as configuration, not as a constant (#136, out of #139).

Written by: Rambarun Komaljeet

The problem
-----------
``EOS_ID = 50256`` and ``GPT2_VOCAB_SIZE = 50257`` were written into ~10 files
back when there was exactly one tokenizer. #139 added a second (the 32K mixed
vocabulary, whose EOS is **0**, not 50256) and measured in #139's mixed-corpus
follow-up that the right vocabulary depends on the corpus -- code-heavy corpora
want 32K, English-heavy corpora want GPT-2. So there is no correct global
default, and a hardcoded one is not a tuning choice but a latent bug.

**The specific failure it causes is silent.** A 32K model served with EOS 50256
can never sample its stop token, because 50256 is not in its vocabulary at all.
Generation then runs to ``max_new_tokens`` every single time. That reads as "the
model rambles" -- a quality problem -- while being a configuration one, and no
exception is ever raised.

The rule
--------
**EOS belongs to the model, and the model got it from its tokenizer.** Resolve it
at the point of use from whatever is in hand, and treat GPT-2's 50256 only as a
last-resort fallback for checkpoints written before ``eos_token_id`` was
persisted (#139 added that key; older files genuinely are GPT-2).

Resolution order is model first, then tokenizer, because the model is the thing
actually generating: a checkpoint that recorded its own EOS is authoritative even
if a caller passes a mismatched tokenizer, and ``check_vocab_match`` exists to
catch that mismatch loudly rather than let it produce plausible garbage.
"""

from __future__ import annotations

from typing import Any, Optional

# GPT-2's ids, kept as named constants so call sites stop writing the integers
# and a grep for the literal 50256 finds only this file.
GPT2_VOCAB_SIZE = 50257
GPT2_EOS_ID = 50256


def resolve_eos_id(model: Any = None, tokenizer: Any = None,
                   default: int = GPT2_EOS_ID) -> int:
    """Return the end-of-sequence id for this model/tokenizer pair.

    Order: the model's persisted ``eos_token_id``, then the tokenizer's, then
    ``default``. A value of 0 is a legitimate EOS (the 32K vocabulary's), so
    every check here is ``is not None`` and never truthiness -- ``or`` would
    silently discard it and hand back 50256, which is the exact bug this module
    exists to remove.
    """
    for src in (model, tokenizer):
        if src is None:
            continue
        eos = getattr(src, "eos_token_id", None)
        if eos is not None:
            return int(eos)
    return int(default)


def resolve_vocab_size(model: Any = None, tokenizer: Any = None,
                       default: int = GPT2_VOCAB_SIZE) -> int:
    """Return the vocabulary size for this model/tokenizer pair."""
    for src in (model, tokenizer):
        if src is None:
            continue
        v = getattr(src, "vocab_size", None)
        if v is not None:
            return int(v)
    return int(default)


def check_vocab_match(model: Any, tokenizer: Any, where: str = "") -> None:
    """Raise if a model and tokenizer disagree on vocabulary size.

    Serving a model with the wrong tokenizer does not crash -- ids are ints and
    both sides accept them. It produces fluent-looking nonsense, or silently
    truncates the id space. This turns that into an exception at setup time.
    """
    mv = getattr(model, "vocab_size", None)
    tv = getattr(tokenizer, "vocab_size", None)
    if mv is None or tv is None:
        return
    if int(mv) != int(tv):
        raise ValueError(
            f"vocab mismatch{' in ' + where if where else ''}: model has "
            f"{int(mv)}, tokenizer has {int(tv)}. Scoring or generating with a "
            f"mismatched tokenizer produces output that looks fine and means "
            f"nothing.")
