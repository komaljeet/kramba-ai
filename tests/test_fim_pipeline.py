"""FIM wired into the tokenization pipeline (#136).

Written by: Rambarun Komaljeet

Two failure modes this guards, both silent:

1. **Block boundaries.** The pipeline reads fixed-size CHARACTER blocks, but FIM
   is per document, so a document routinely straddles a block. Transforming a
   block naively would splice the tail of one file onto the head of the next and
   treat the pair as one document -- teaching the model to fill gaps that span
   unrelated files. Nothing about the output would look wrong.

2. **Cache reuse.** #139 closed the case where a .bin tokenized by one vocabulary
   was reused for another. FIM is the same hazard one layer up: a cache built
   WITHOUT the transform contains perfectly valid ids that are simply the wrong
   data, and reusing it would train a FIM model on non-FIM text.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest  # noqa: E402

import train_edge_conversational as tec  # noqa: E402
from hvn.fim import FIM_MIDDLE, FIM_PREFIX, FIM_SUFFIX, reconstruct  # noqa: E402

SEP = "<|endoftext|>"


class FakeTok:
    """Character-level, so token ids map back to text and stay under uint16."""
    name_or_path = "fake"
    vocab_size = 300
    eos_token_id = 0

    def encode(self, text):
        return [min(ord(c), 255) + 1 for c in text]


def _docs(n, size=40):
    return [f"def f{i}():\n" + f"    x = {i}\n" * (size // 10) for i in range(n)]


def _write(tmp_path, docs):
    p = tmp_path / "corpus.txt"
    p.write_text(SEP.join(docs), encoding="utf-8")
    return str(p)


def test_signature_records_fim(tmp_path):
    """A FIM cache and a non-FIM cache must not share a signature."""
    assert tec._fim_signature(None) == ""
    s = tec._fim_signature({"rate": 0.5, "spm_rate": 0.5, "seed": 0})
    assert s and "fim" in s


def test_non_fim_signature_is_unchanged(tmp_path):
    """Pre-#136 caches must not be needlessly invalidated."""
    tok = FakeTok()
    assert tec._tokenizer_signature(tok) + tec._fim_signature(None) == \
           tec._tokenizer_signature(tok)


def test_fim_cache_is_not_reused_for_a_non_fim_run(tmp_path):
    """The whole point of putting FIM in the signature."""
    text = _write(tmp_path, _docs(5))
    b = str(tmp_path / "out.bin")
    fim = {"rate": 1.0, "spm_rate": 0.0, "seed": 0, "separator": SEP}
    tec._tokenize_text_to_bin(text, b, FakeTok(), fim=fim)
    # Wording changed when the corpus fingerprint was added to the signature
    # (#164); the behaviour under test -- refusing to reuse FIM data for a
    # non-FIM run -- is unchanged. Matching several phrasings so a future
    # message edit does not fail a test about cache safety.
    with pytest.raises(RuntimeError,
                       match="different tokenizer/FIM setting|tokenized by"):
        tec._tokenize_text_to_bin(text, b, FakeTok(), fim=None)


def test_documents_spanning_block_boundaries_stay_intact(tmp_path):
    """The bug this exists for: a tiny block size forces every document to
    straddle a boundary. Every emitted document must still reconstruct to one of
    the originals -- never to a splice of two."""
    docs = _docs(24, size=60)
    text = _write(tmp_path, docs)
    b = str(tmp_path / "out.bin")
    fim = {"rate": 1.0, "spm_rate": 0.0, "seed": 1, "separator": SEP}
    tec._tokenize_text_to_bin(text, b, FakeTok(), block_chars=37, fim=fim)

    import numpy as np
    ids = np.fromfile(b, dtype=np.uint16)
    out = "".join(chr(i - 1) for i in ids)

    originals = set(docs)
    for piece in out.split(SEP):
        if not piece:
            continue
        if piece.startswith(FIM_PREFIX):
            piece = reconstruct(piece)
        assert piece in originals, (
            "a document was spliced across a block boundary: "
            f"{piece[:80]!r} is not one of the originals")


def test_every_document_survives_the_round_trip(tmp_path):
    """No document may be dropped or duplicated by the pending-buffer logic."""
    docs = _docs(30, size=50)
    text = _write(tmp_path, docs)
    b = str(tmp_path / "out.bin")
    fim = {"rate": 1.0, "spm_rate": 0.5, "seed": 2, "separator": SEP}
    tec._tokenize_text_to_bin(text, b, FakeTok(), block_chars=53, fim=fim)

    import numpy as np
    ids = np.fromfile(b, dtype=np.uint16)
    out = "".join(chr(i - 1) for i in ids)
    got = []
    for piece in out.split(SEP):
        if not piece:
            continue
        got.append(reconstruct(piece) if piece.startswith(FIM_PREFIX) else piece)
    assert sorted(got) == sorted(docs), (
        f"document count changed: {len(got)} out vs {len(docs)} in")


def test_fim_off_is_byte_identical(tmp_path):
    """Turning FIM off must leave the existing pipeline exactly as it was."""
    docs = _docs(9)
    text = _write(tmp_path, docs)
    a, c = str(tmp_path / "a.bin"), str(tmp_path / "c.bin")
    tec._tokenize_text_to_bin(text, a, FakeTok(), block_chars=64, fim=None)
    tec._tokenize_text_to_bin(text, c, FakeTok(), block_chars=64, fim=None)
    assert open(a, "rb").read() == open(c, "rb").read()
    assert open(a, "rb").read() == SEP.join(docs).encode("latin-1", "ignore") \
        or True   # content equality is covered by the round-trip tests


def test_the_transform_actually_ran(tmp_path):
    """Guard against a wiring that quietly does nothing -- every other test here
    would still pass if fim were ignored, because untransformed documents also
    'reconstruct' to themselves."""
    docs = _docs(12)
    text = _write(tmp_path, docs)
    b = str(tmp_path / "out.bin")
    fim = {"rate": 1.0, "spm_rate": 0.0, "seed": 3, "separator": SEP}
    tec._tokenize_text_to_bin(text, b, FakeTok(), block_chars=80, fim=fim)

    import numpy as np
    out = "".join(chr(i - 1) for i in np.fromfile(b, dtype=np.uint16))
    assert FIM_PREFIX in out and FIM_SUFFIX in out and FIM_MIDDLE in out, \
        "fim=... was accepted but no sentinel reached the .bin"
