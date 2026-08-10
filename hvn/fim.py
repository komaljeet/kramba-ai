"""Fill-in-the-Middle: train the model to use the code AFTER the cursor (#136).

Written by: Rambarun Komaljeet

Why this is a prerequisite and not a later addition
---------------------------------------------------
A left-to-right model predicts the next token from everything before it. In an
editor that throws away half the context: when you are typing line 40, lines
41-90 already exist, and they say a great deal about what line 40 should be.

FIM (Bavarian et al., arXiv:2207.14255) is a **data transform**, not an
architecture change. Each document is cut into prefix / middle / suffix and
reordered so the suffix arrives BEFORE the span to predict::

    <PRE> prefix <SUF> suffix <MID> middle

The model still reads strictly left to right -- it simply sees both sides of the
gap before it has to fill it. At inference the editor sends prefix and suffix and
the model generates the middle.

Because it changes the training data, it is baked into the pretrain. A finished
left-to-right model cannot be given FIM afterwards without training again, which
is why #136 lists it beside the tokenizer as something to settle before the clock
starts.

Splitting on CHARACTERS, not tokens
-----------------------------------
The cut points are chosen in character space and the three pieces are tokenized
separately. Splitting in token space would put the boundary at a token edge the
tokenizer chose, so the model would only ever see gaps that begin at
"tokenizer-friendly" positions -- while a real cursor sits anywhere, frequently
mid-identifier. Bavarian et al. make the same point; the practical consequence is
a model that degrades exactly when the user is halfway through typing a name,
which is most of the time.

Reconstruction must be EXACT
----------------------------
``reconstruct()`` is the inverse, and ``tests/test_fim.py`` asserts
prefix+middle+suffix reproduces the original byte-for-byte. This matters more
than it sounds: a transform that silently drops one newline at a boundary still
trains, still converges, and still emits plausible completions. It would just
produce a model permanently and subtly wrong about indentation, with nothing
failing anywhere to say so. This repo has shipped that class of error enough
times (#117, #138) that the inverse gets written and tested first.
"""

from __future__ import annotations

import random
from typing import Sequence

# Sentinel strings. Added to the tokenizer as real single tokens by
# scripts/build_fim_tokenizer.py -- with the stock 32K vocabulary each of these
# encodes to EIGHT tokens, which is 24 tokens of overhead per document and asks
# the model to learn that an 8-token sequence is structural punctuation.
FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_TOKENS = (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE)


def split_document(text: str, rng: random.Random) -> tuple[str, str, str]:
    """Cut ``text`` into (prefix, middle, suffix) at two random character points.

    Uniform over character positions, so the middle span can be empty and can sit
    anywhere including the very start or end. Those degenerate cases are kept on
    purpose: a cursor at the top of a file or at EOF is a real editor state, and
    a transform that never produced them would leave the model unprepared for the
    moment a user opens a new file.
    """
    n = len(text)
    if n < 3:
        return text, "", ""
    a, b = sorted(rng.randint(0, n) for _ in range(2))
    return text[:a], text[a:b], text[b:]


def apply_fim(text: str, rng: random.Random, spm: bool = False) -> str:
    """Return the FIM-reordered form of ``text``.

    ``spm`` selects SPM mode (suffix first) instead of the default PSM. The paper
    trains on a mixture; SPM tends to help when the suffix is the more
    informative side. Both are exactly reversible by ``reconstruct``.
    """
    prefix, middle, suffix = split_document(text, rng)
    if spm:
        return (f"{FIM_PREFIX}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{prefix}{middle}")
    return (f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}{middle}")


def reconstruct(transformed: str) -> str:
    """Inverse of ``apply_fim``. Raises if the input is not well-formed.

    Exists so the transform can be TESTED, not because training needs it. An
    untestable data transform is one whose bugs surface as mysterious model
    behaviour weeks later.
    """
    if not transformed.startswith(FIM_PREFIX):
        raise ValueError("not a FIM document: missing prefix sentinel")
    body = transformed[len(FIM_PREFIX):]

    if body.startswith(FIM_SUFFIX):                      # SPM
        body = body[len(FIM_SUFFIX):]
        suffix, sep, rest = body.partition(FIM_MIDDLE)
        if not sep:
            raise ValueError("malformed SPM document: no middle sentinel")
        # SPM packs prefix+middle contiguously after <MID>, and the boundary
        # between them is not recorded -- it does not need to be, because the
        # original text is their concatenation followed by the suffix.
        return rest + suffix

    prefix, sep, rest = body.partition(FIM_SUFFIX)       # PSM
    if not sep:
        raise ValueError("malformed PSM document: no suffix sentinel")
    suffix, sep, middle = rest.partition(FIM_MIDDLE)
    if not sep:
        raise ValueError("malformed PSM document: no middle sentinel")
    return prefix + middle + suffix


def transform_stream(docs: Sequence[str], fim_rate: float = 0.5,
                     spm_rate: float = 0.5, seed: int = 0) -> list[str]:
    """Apply FIM to a fraction of ``docs``, leaving the rest left-to-right.

    ``fim_rate`` below 1.0 on purpose. The model still has to generate code
    forward from a cursor with nothing after it -- a new file, the end of a
    function -- and a corpus that is 100% FIM never shows it that case. Bavarian
    et al. report rates up to 0.9 are harmless to left-to-right ability; 0.5 is
    the conservative default here because this model is small and gets one
    pretrain, not a sweep.
    """
    rng = random.Random(seed)
    out = []
    for d in docs:
        if rng.random() < fim_rate:
            out.append(apply_fim(d, rng, spm=rng.random() < spm_rate))
        else:
            out.append(d)
    return out
