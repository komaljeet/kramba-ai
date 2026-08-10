"""Vocabulary/EOS resolution is config, not a constant (#136).

Written by: Rambarun Komaljeet

**Every test here uses a NON-GPT-2 vocabulary on purpose.** A test written
against GPT-2 would pass whether or not the code reads the config at all --
50256 is the hardcoded value AND the correct answer, so the two are
indistinguishable. That is exactly the vacuous-test shape this repo has been
burned by (#138's CUDA graph compared a replay against an eager step doing the
same thing and passed at 0.000e+00 while the model was silently broken).

The 32K vocabulary's EOS is **0**, which is also falsy -- so these tests pin
down the two ways this breaks: reading a stale constant, and writing
``eos or default``, which throws away a perfectly valid id of 0.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.vocab import (  # noqa: E402
    GPT2_EOS_ID, GPT2_VOCAB_SIZE, check_vocab_match, resolve_eos_id,
    resolve_vocab_size,
)


class FakeTokenizer:
    """Minimal stand-in: a vocabulary that is NOT GPT-2's."""

    def __init__(self, vocab_size=32000, eos_token_id=0):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id

    def encode(self, text):
        # Deterministic, and never emits 0, so an EOS of 0 in the output can
        # only have come from the code under test appending it.
        return [(ord(c) % (self.vocab_size - 1)) + 1 for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class FakeModel:
    def __init__(self, vocab_size=32000, eos_token_id=0):
        self.vocab_size = vocab_size
        self.eos_token_id = eos_token_id


def test_the_fixture_is_not_gpt2():
    """Guard whose only job is to fail if these tests go vacuous.

    If someone 'fixes' the fixture to GPT-2's ids, every other test here would
    still pass while testing nothing. This one would fail first.
    """
    tok = FakeTokenizer()
    assert tok.eos_token_id != GPT2_EOS_ID
    assert tok.vocab_size != GPT2_VOCAB_SIZE


def test_resolves_eos_from_tokenizer_not_the_constant():
    assert resolve_eos_id(tokenizer=FakeTokenizer()) == 0


def test_zero_eos_survives_truthiness():
    """`eos or 50256` returns 50256 when eos is 0. That is the bug."""
    assert resolve_eos_id(tokenizer=FakeTokenizer(eos_token_id=0)) == 0
    assert resolve_eos_id(model=FakeModel(eos_token_id=0)) == 0


def test_model_wins_over_tokenizer():
    """The model is what generates, so its persisted id is authoritative."""
    got = resolve_eos_id(model=FakeModel(eos_token_id=7),
                         tokenizer=FakeTokenizer(eos_token_id=0))
    assert got == 7


def test_falls_back_to_gpt2_for_legacy_checkpoints():
    """Checkpoints written before eos_token_id was persisted really are GPT-2."""
    class Bare:
        pass

    assert resolve_eos_id(model=Bare(), tokenizer=Bare()) == GPT2_EOS_ID
    assert resolve_eos_id() == GPT2_EOS_ID


def test_resolve_vocab_size():
    assert resolve_vocab_size(tokenizer=FakeTokenizer()) == 32000
    assert resolve_vocab_size() == GPT2_VOCAB_SIZE


def test_vocab_mismatch_raises():
    """Serving a model with the wrong tokenizer does not crash on its own --
    it produces fluent nonsense. Make it an exception instead."""
    with pytest.raises(ValueError, match="vocab mismatch"):
        check_vocab_match(FakeModel(vocab_size=32000),
                          FakeTokenizer(vocab_size=50257))


def test_vocab_match_passes_when_equal():
    check_vocab_match(FakeModel(vocab_size=32000), FakeTokenizer(vocab_size=32000))


def test_instruction_dataset_appends_the_tokenizers_eos():
    """The dataset must append 0, not 50256.

    Training a 32K model on GPT-2's EOS teaches it to emit an id outside its own
    vocabulary -- so it would never produce a stop token that its own decoder
    can represent.
    """
    from finetune_edge_conversational import InstructionDataset

    tok = FakeTokenizer()
    ds = InstructionDataset(tok, [[("user", "hi"), ("assistant", "yo")]],
                            seq_len=32, pad=True)
    assert ds.eos_id == 0
    assert ds.eos_id != GPT2_EOS_ID

    ids, _labels = ds.examples[0]
    assert 0 in list(ids), "tokenizer EOS (0) never appears in the built row"
    assert GPT2_EOS_ID not in list(ids), "GPT-2 EOS leaked into a 32K row"
