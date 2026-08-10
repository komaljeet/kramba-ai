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
