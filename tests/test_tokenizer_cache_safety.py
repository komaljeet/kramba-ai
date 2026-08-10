"""The tokenized-corpus cache must not silently serve the wrong vocabulary (#139).

Written by: Rambarun Komaljeet

Before this, `_tokenize_text_to_bin` cached on "does the .bin exist" and recorded
nothing about which tokenizer produced it. Switching to a 32K vocab would have
reused GPT-2 ids in complete silence — training a 32K model on token id 45,000,
or worse on ids that are merely *wrong* rather than out of range, which raises
nothing at all.

That is the same shape as #117, where a cache served untrained weights
undetected because no test looked at the specific thing that changed. So these
tests check the identity mechanism directly rather than checking that training
"still works".
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_edge_conversational as tec  # noqa: E402


class FakeTokenizer:
    """Stands in for a custom 32K tokenizer without needing to train one."""

    def __init__(self, name="custom32k", vocab_size=32000):
        self.name_or_path = name
        self.vocab_size = vocab_size

    def encode(self, text):
        return [i % self.vocab_size for i in range(len(text) // 4)]


@pytest.fixture
def corpus(tmp_path):
    p = tmp_path / "corpus.txt"
    p.write_text("hello world " * 2000, encoding="utf-8")
    return str(p), str(tmp_path / "corpus.bin")


def gpt2():
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    tok.model_max_length = 10 ** 9
    return tok


# --------------------------------------------------------------------------


def test_signature_includes_name_and_vocab_size():
    assert tec._tokenizer_signature(FakeTokenizer()) == "custom32k/32000"
    assert tec._tokenizer_signature(gpt2()) == "gpt2/50257"


def test_first_tokenize_writes_a_sidecar(corpus):
    txt, binp = corpus
    n = tec._tokenize_text_to_bin(txt, binp, gpt2())
    assert n > 0
    with open(binp + ".tokenizer", encoding="utf-8") as fh:
        assert fh.read().strip() == "gpt2/50257"


def test_same_tokenizer_hits_the_cache(corpus):
    txt, binp = corpus
    n1 = tec._tokenize_text_to_bin(txt, binp, gpt2())
    mtime = os.path.getmtime(binp)
    n2 = tec._tokenize_text_to_bin(txt, binp, gpt2())
    assert n1 == n2
    assert os.path.getmtime(binp) == mtime      # reused, not rewritten


def test_different_tokenizer_refuses_rather_than_reusing(corpus):
    """The whole point: a silent reuse here poisons an entire training run."""
    txt, binp = corpus
    tec._tokenize_text_to_bin(txt, binp, gpt2())
    with pytest.raises(RuntimeError, match="wrong vocabulary|tokenized by"):
        tec._tokenize_text_to_bin(txt, binp, FakeTokenizer())


def test_legacy_cache_without_a_sidecar_is_adopted_as_gpt2(corpus):
    """Pre-#139 caches are real and multi-hour to rebuild.

    They can only be GPT-2 — it is the only tokenizer this pipeline ever used —
    so adopting them is correct and avoids a needless re-tokenize. It is done
    out loud, and only for GPT-2.
    """
    txt, binp = corpus
    tec._tokenize_text_to_bin(txt, binp, gpt2())
    os.remove(binp + ".tokenizer")
    n = tec._tokenize_text_to_bin(txt, binp, gpt2())   # must NOT raise
    assert n > 0
    with open(binp + ".tokenizer", encoding="utf-8") as fh:
        assert fh.read().strip() == "gpt2/50257"


def test_legacy_cache_with_a_custom_tokenizer_still_refuses(corpus):
    """Adopting an unlabelled cache must not become a loophole."""
    txt, binp = corpus
    tec._tokenize_text_to_bin(txt, binp, gpt2())
    os.remove(binp + ".tokenizer")
    with pytest.raises(RuntimeError, match="predates tokenizer tracking"):
        tec._tokenize_text_to_bin(txt, binp, FakeTokenizer())


def test_custom_tokenizer_writes_its_own_signature(tmp_path):
    txt = tmp_path / "c.txt"
    txt.write_text("abcd" * 1000, encoding="utf-8")
    binp = str(tmp_path / "c.bin")
    tec._tokenize_text_to_bin(str(txt), binp, FakeTokenizer())
    with open(binp + ".tokenizer", encoding="utf-8") as fh:
        assert fh.read().strip() == "custom32k/32000"
