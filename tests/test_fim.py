"""Fill-in-the-Middle: the reconstruction must be EXACT (#136).

Written by: Rambarun Komaljeet

#136's acceptance criterion 3 is "FIM reconstruction test passes exactly", and
the word doing the work is *exactly*. A transform that loses one newline at a
segment boundary still trains, still converges, and still emits plausible
completions -- it just produces a model permanently wrong about indentation,
with nothing anywhere failing to say so. Byte-for-byte or it is not a pass.

The properties tested are chosen so that plausible bugs cannot slip through:
whitespace at the boundaries, empty segments, cut points inside a token, and
documents that contain the sentinel strings themselves.
"""

from __future__ import annotations

import os
import random
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.fim import (  # noqa: E402
    FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, apply_fim, reconstruct,
    split_document, transform_stream,
)

SAMPLES = [
    "def f(x):\n    return x + 1\n",
    "class A:\n\tdef m(self):\n\t\tpass\n",
    "import os\n\n\n\nif __name__ == '__main__':\n    main()\n",
    "  \n\t\n   leading and trailing whitespace   \n\n",
    "x = 'quote\" mix'\n# comment with unicode: caf\u00e9 \u00fc\u00f1\n",
    "a",
    "ab",
    "abc",
    "",
    "\n",
    "line\r\nwindows\r\nendings\r\n",
]


@pytest.mark.parametrize("text", SAMPLES)
@pytest.mark.parametrize("spm", [False, True])
def test_reconstruction_is_byte_exact(text, spm):
    rng = random.Random(0)
    out = apply_fim(text, rng, spm=spm)
    assert reconstruct(out) == text


@pytest.mark.parametrize("spm", [False, True])
def test_reconstruction_exact_over_many_random_cuts(spm):
    """One cut point proves nothing -- the bug lives at a specific boundary."""
    text = "def compute(values):\n    total = 0\n    for v in values:\n" \
           "        total += v\n    return total\n"
    for seed in range(500):
        rng = random.Random(seed)
        assert reconstruct(apply_fim(text, rng, spm=spm)) == text


def test_whitespace_at_boundaries_survives():
    """Indentation is the thing a code model must not get wrong, and a stray
    strip() at a segment boundary is the most natural way to break it."""
    text = "if x:\n    y = 1\n    z = 2\n"
    for seed in range(300):
        rng = random.Random(seed)
        out = apply_fim(text, rng)
        assert reconstruct(out) == text
        assert "    y = 1" in reconstruct(out)


def test_empty_middle_is_allowed():
    """A cursor at EOF with nothing to fill is a real editor state."""
    rng = random.Random(0)
    p, m, s = "abc", "", "def"
    doc = f"{FIM_PREFIX}{p}{FIM_SUFFIX}{s}{FIM_MIDDLE}{m}"
    assert reconstruct(doc) == "abcdef"


def test_all_three_segments_can_be_empty():
    doc = f"{FIM_PREFIX}{FIM_SUFFIX}{FIM_MIDDLE}"
    assert reconstruct(doc) == ""


def test_split_covers_the_original():
    """The three pieces must partition the text with nothing added or lost."""
    text = "abcdefghij" * 7
    for seed in range(400):
        p, m, s = split_document(text, random.Random(seed))
        assert p + m + s == text


def test_cut_can_land_mid_identifier():
    """Character-level splitting is the whole point: a real cursor sits inside a
    name most of the time. If every cut landed on a token boundary this would
    never fire."""
    text = "some_long_identifier_name = 1\n"
    seen_mid_word = False
    for seed in range(200):
        p, m, s = split_document(text, random.Random(seed))
        if p and m and p[-1].isalnum() and m[0].isalnum():
            seen_mid_word = True
            break
    assert seen_mid_word, "no cut ever landed inside a word -- splitting is not " \
                          "character-level and the model will only see " \
                          "tokenizer-aligned gaps"


def test_document_containing_sentinels_does_not_silently_corrupt():
    """A file that literally contains '<|fim_prefix|>' must not be mistaken for
    structure. Reconstruction either round-trips or raises -- what it must never
    do is return different text while looking successful."""
    text = f"comment = '{FIM_PREFIX}'\nx = 1\n"
    rng = random.Random(3)
    out = apply_fim(text, rng)
    try:
        back = reconstruct(out)
    except ValueError:
        return                      # refusing is an acceptable outcome
    assert back == text, ("reconstruction succeeded but returned different text "
                          "-- a document containing a sentinel was misparsed as "
                          "structure, which is the silent-corruption case")


def test_malformed_input_raises():
    with pytest.raises(ValueError):
        reconstruct("no sentinels here")
    with pytest.raises(ValueError):
        reconstruct(f"{FIM_PREFIX}only a prefix")


def test_stream_leaves_some_documents_untouched():
    """fim_rate < 1 on purpose: the model still has to generate forward from a
    cursor with nothing after it. A 100% FIM corpus never shows it that."""
    docs = [f"def f{i}():\n    return {i}\n" for i in range(400)]
    out = transform_stream(docs, fim_rate=0.5, seed=1)
    touched = sum(1 for d in out if d.startswith(FIM_PREFIX))
    assert 0 < touched < len(docs)
    assert 0.4 < touched / len(docs) < 0.6


def test_stream_transformed_docs_all_reconstruct():
    docs = [f"def f{i}():\n    return {i}\n" for i in range(200)]
    out = transform_stream(docs, fim_rate=1.0, seed=2)
    for original, t in zip(docs, out):
        assert reconstruct(t) == original


def test_both_modes_actually_occur():
    """Guard against a stream that silently only ever emits PSM."""
    docs = ["abcdefghij"] * 400
    out = transform_stream(docs, fim_rate=1.0, spm_rate=0.5, seed=5)
    spm = sum(1 for d in out if d.startswith(FIM_PREFIX + FIM_SUFFIX))
    psm = len(out) - spm
    assert spm > 0 and psm > 0, f"only one mode emitted (spm={spm}, psm={psm})"


def test_the_suite_can_fail():
    """Fails if reconstruct is ever reduced to identity or a no-op.

    Every other test here compares reconstruct(apply_fim(x)) to x, which a
    reconstruct that simply returned its input would pass for any document
    without sentinels. This states the requirement directly: the transformed
    form must NOT equal the original, so the inverse has real work to do.
    """
    text = "def f(x):\n    return x\n"
    out = apply_fim(text, random.Random(0))
    assert out != text, "apply_fim did not transform anything"
    assert FIM_PREFIX in out and FIM_SUFFIX in out and FIM_MIDDLE in out
    assert reconstruct(out) == text


# --------------------------------------------------------------------------
# Added-token vocabulary sizing (#136)
# --------------------------------------------------------------------------

class _AddedTokenTokenizer:
    """Mimics HuggingFace: `vocab_size` EXCLUDES added tokens, `len` includes.

    Not a contrived case -- this is exactly what the FIM tokenizer does, and
    reading the wrong one gives an embedding table three rows too short.
    """
    vocab_size = 32000
    eos_token_id = 0

    def __len__(self):
        return 32003


def test_added_tokens_count_toward_the_embedding_table():
    from hvn.vocab import resolve_vocab_size
    got = resolve_vocab_size(tokenizer=_AddedTokenTokenizer())
    assert got == 32003, (
        f"got {got}; a model built with {got} rows would index out of bounds on "
        f"the first FIM document, whose sentinels are ids 32000-32002")


def test_plain_tokenizer_is_unaffected():
    class Plain:
        vocab_size = 32000
        def __len__(self):
            return 32000
    from hvn.vocab import resolve_vocab_size
    assert resolve_vocab_size(tokenizer=Plain()) == 32000


def test_model_vocab_size_still_wins():
    """A loaded checkpoint knows the table it actually has."""
    import types
    from hvn.vocab import resolve_vocab_size
    m = types.SimpleNamespace(vocab_size=12345)
    assert resolve_vocab_size(model=m, tokenizer=_AddedTokenTokenizer()) == 12345


def test_sizing_guard_is_not_vacuous():
    """Fails if resolve_vocab_size goes back to reading .vocab_size blindly."""
    from hvn.vocab import resolve_vocab_size
    tok = _AddedTokenTokenizer()
    assert resolve_vocab_size(tokenizer=tok) != tok.vocab_size, \
        "resolve_vocab_size returned the base vocab_size, ignoring added tokens"


# ---------------------------------------------------------------------------
# Gap DENSITY (#164).
#
# The 48M code run reached 63% of its token budget at exactly 0.0% exact-match
# because only ONE token per document -- the first after <|fim_middle|> -- teaches
# the model to jump back to the cursor, and with a 237-token median middle that
# token was 1 in 4,813 corpus tokens. Never learning it costs +0.0006 nats on the
# aggregate curve, so the training loss looked healthy throughout.
#
# Nothing in the round-trip tests above can fail when the density is wrong: a
# transform with one enormous gap per file reconstructs perfectly. These tests
# exist to fail in that case, because the loss curve will not.
# ---------------------------------------------------------------------------

def _jump_back_density(text, seed=0, **kw):
    """(jump-back tokens, total chars) for the real training path."""
    import re
    rng = random.Random(seed)
    out = transform_stream([text], fim_rate=1.0, seed=seed, **kw)
    joined = "".join(out)
    return joined.count(FIM_MIDDLE), len(text)


def test_middle_is_a_small_fraction_of_its_unit():
    """A gap must be a SMALL hole, not a third of the file.

    Guards the property directly: with two uniform cuts E[middle] = n/3, so a
    regression to that implementation drives this mean to ~0.33 and fails.
    """
    text = "".join(f"line_{i} = compute({i})\n" for i in range(400))
    fracs = []
    for seed in range(300):
        _, m, _ = split_document(text, random.Random(seed))
        fracs.append(len(m) / len(text))
    mean = sum(fracs) / len(fracs)
    assert mean < 0.25, (
        f"mean middle is {mean:.1%} of the document -- gaps this large give one "
        f"jump-back token per hundreds of ordinary ones, which is what made the "
        f"48M run score 0% (#164)")


def test_long_file_produces_many_gaps_not_one():
    """The unit split. One gap per file is too sparse to learn from."""
    text = "".join(f"def f_{i}(a, b):\n    return a + b + {i}\n" for i in range(900))
    gaps, n_chars = _jump_back_density(text, seed=0)
    assert n_chars > 20000, "test corpus too small to distinguish the cases"
    assert gaps >= 8, (
        f"a {n_chars:,}-character file produced {gaps} gap(s); the transform is "
        f"still gapping whole documents (#164)")


def test_density_beats_the_configuration_that_failed():
    """The quantitative gate: characters of code per jump-back token.

    The v1 transform delivered one jump-back per ~19,000 characters of this
    corpus (~4,800 tokens). This asserts a large margin below that, so a partial
    revert -- short middles but no unit split, or vice versa -- still fails.
    """
    text = "".join(f"class C{i}:\n    x = {i}\n    def m(self):\n        return self.x\n"
                   for i in range(600))
    gaps, n_chars = _jump_back_density(text, seed=1)
    chars_per_gap = n_chars / max(1, gaps)
    assert chars_per_gap < 4000, (
        f"one jump-back per {chars_per_gap:,.0f} characters; v1 was ~19,000 and "
        f"scored 0%. The density fix is not in effect (#164)")


def test_most_gaps_start_on_a_line_boundary():
    """Line-aligned starts are learnable; mid-word BPE fragments largely are not
    (measured first-middle-token NLL 3.749 aligned vs 6.230 character-cut)."""
    text = "".join(f"    value_{i} = helper({i})\n" for i in range(300))
    aligned = 0
    trials = 300
    for seed in range(trials):
        p, m, _ = split_document(text, random.Random(seed))
        if m and (not p or p.endswith("\n")):
            aligned += 1
    assert aligned / trials > 0.6, (
        f"only {aligned/trials:.0%} of gaps start on a line boundary")


def test_character_level_cuts_still_happen():
    """The counterpart. Aligning EVERY gap would leave the model helpless
    mid-identifier, which is where a real cursor usually sits -- so the fix must
    not become 'line-aligned only'."""
    text = "".join(f"some_identifier_{i} = {i}\n" for i in range(300))
    mid_word = 0
    for seed in range(300):
        p, m, _ = split_document(text, random.Random(seed))
        if p and m and p[-1].isalnum() and m[0].isalnum():
            mid_word += 1
    assert mid_word > 0, "no gap ever starts mid-word -- character cuts are gone"


def test_unit_split_preserves_every_byte():
    """The units must partition the text exactly; a dropped newline here would
    train the model on subtly wrong indentation with nothing failing."""
    from hvn.fim import iter_fim_units
    text = "".join(f"row {i}\n" for i in range(2000))
    assert "".join(iter_fim_units(text, random.Random(0))) == text


def test_density_test_would_catch_a_revert():
    """Meta-test: re-runs the OLD transform and asserts the gate above rejects it.

    Without this, the density gates could be silently satisfied by a change in
    the test corpus rather than in the code -- the 'test that cannot fail' the
    repo has shipped before (#138).
    """
    def old_split(text, rng):
        n = len(text)
        a, b = sorted(rng.randint(0, n) for _ in range(2))
        return text[:a], text[a:b], text[b:]

    text = "".join(f"line_{i} = compute({i})\n" for i in range(400))
    fracs = [len(old_split(text, random.Random(s))[1]) / len(text)
             for s in range(300)]
    assert sum(fracs) / len(fracs) > 0.25, \
        "the v1 transform now passes the density gate -- the gate is vacuous"
