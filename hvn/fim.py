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

#: Bumped whenever the gap SHAPE changes. Part of the tokenized-cache key, so a
#: corpus built by an older transform can never be silently reused -- it has the
#: same rate/spm/seed and looks identical from outside (#164).
#:   v1 = two uniform cuts over the whole document (237-token median middle)
#:   v2 = short, 80% line-aligned middles, one gap per ~2000-char unit
FIM_TRANSFORM_VERSION = "v2"


#: Target size of a FIM unit, in characters. A document is cut into pieces of
#: about this size and each piece gets its OWN gap -- see ``iter_fim_units``.
#: 1200 chars is ~330 tokens of surrounding code per gap. Measured density on a
#: 4 MB corpus slice, against the v1 baseline of one jump-back per 4,813 tokens:
#:     unit 2000, rate 0.5 -> 1 per 1,354 ( 3.6x)
#:     unit 1200, rate 0.5 -> 1 per   849 ( 5.7x)
#:     unit 1200, rate 0.9 -> 1 per   472 (10.2x)   <- chosen
#:     unit  800, rate 0.5 -> 1 per   589 ( 8.2x)
#: Smaller units keep raising density but shrink the context the model gets to
#: reason from, and the gate supplies far more than 800 characters -- training on
#: less than we evaluate on would re-create a train/eval mismatch.
FIM_UNIT_CHARS = 1200

#: Fraction of gaps that start and end on a line boundary.
FIM_LINE_ALIGNED_RATE = 0.8

#: Middle length as a fraction of its unit, sampled uniformly in this range.
FIM_MIDDLE_FRAC = (0.02, 0.20)


def split_document(text: str, rng: random.Random) -> tuple[str, str, str]:
    """Cut ``text`` into (prefix, middle, suffix) around a SHORT gap.

    Why not two uniform cuts (the obvious implementation, used until #164)
    ---------------------------------------------------------------------
    ``a, b = sorted(rng.randint(0, n) for _ in range(2))`` gives E[middle] = n/3,
    which measured out at a 237-token median middle. That is the wrong shape for
    what FIM has to teach.

    Only ONE token in the whole document carries the FIM lesson: the first token
    after ``<|fim_middle|>``, where the model must jump back to the cursor instead
    of continuing the suffix it has just read. Every later middle token is
    ordinary left-to-right prediction. So a 237-token middle buys one jump-back
    and 236 tokens of what the model already does.

    Measured on the 48M run at step 12000 (#164): the jump-back token was 1 in
    4,813 corpus tokens, the model had NOT learned it (first-middle-token NLL
    3.554 with FIM context vs 0.576 with no suffix at all -- the suffix made it
    WORSE), and the aggregate loss curve could not see this because never learning
    it costs +0.0006 nats. Short middles raise the density of that one signal by
    ~50x at identical GPU cost per step.

    Line alignment
    --------------
    ``FIM_LINE_ALIGNED_RATE`` of gaps begin and end at a newline. The original
    character-level cut is deliberately KEPT for the rest: a real cursor sits
    mid-identifier most of the time, and a model trained only on tidy boundaries
    degrades exactly when the user is halfway through typing a name. But a
    mid-word BPE fragment is often genuinely unguessable -- measured
    first-middle-token NLL was 6.230 on character cuts against 3.749 on
    line-aligned ones -- so making every gap that hard drowns the signal in noise.
    The mixture teaches the skill on learnable cases and still shows the hard one.

    Degenerate cases (empty middle, gap at the very start or end) remain
    reachable: a cursor at the top of a file or at EOF is a real editor state.
    """
    n = len(text)
    if n < 3:
        return text, "", ""

    lo, hi = FIM_MIDDLE_FRAC
    span = max(1, int(n * rng.uniform(lo, hi)))
    a = rng.randint(0, max(0, n - span))
    b = min(n, a + span)

    if rng.random() < FIM_LINE_ALIGNED_RATE:
        # Snap outwards to line boundaries. Snapping OUT (not to the nearest
        # edge) keeps the gap non-empty, so the aligned arm always carries a
        # jump-back token.
        ls = text.rfind("\n", 0, a)
        a = 0 if ls < 0 else ls + 1
        le = text.find("\n", b)
        b = n if le < 0 else le + 1

    return text[:a], text[a:b], text[b:]


def iter_fim_units(text: str, rng: random.Random,
                   unit_chars: int | None = None):
    """Yield roughly ``unit_chars``-sized pieces of ``text`` to gap separately.

    The second half of the density fix. Even with a short middle, treating a
    30,000-character file as ONE document yields ONE jump-back for the whole
    file; the other ~7,000 tokens teach nothing new. Cutting it into units of
    ~2,000 characters yields ~15, from the same tokens and the same GPU time.

    Units are split at line boundaries so each one is coherent code rather than
    a fragment starting mid-statement. Short documents yield exactly one unit,
    which is the previous behaviour.

    ``unit_chars`` defaults to ``FIM_UNIT_CHARS`` at CALL time, not at import
    time -- a default argument would freeze the module constant when this
    function was defined, so tuning it (or a test setting it) would silently do
    nothing.
    """
    unit_chars = FIM_UNIT_CHARS if unit_chars is None else unit_chars
    n = len(text)
    if n <= unit_chars:
        yield text
        return

    start = 0
    while start < n:
        end = min(n, start + unit_chars)
        if end < n:
            nl = text.find("\n", end)
            end = n if nl < 0 else nl + 1
        yield text[start:end]
        start = end


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
        # Each unit gets its own independent coin flip, so a long file
        # contributes many gaps AND many plain left-to-right stretches rather
        # than being all-or-nothing (#164).
        for unit in iter_fim_units(d, rng):
            if rng.random() < fim_rate:
                out.append(apply_fim(unit, rng, spm=rng.random() < spm_rate))
            else:
                out.append(unit)
    return out
