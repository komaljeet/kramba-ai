"""Tests for BM25 retrieval (Issue #137).

Written by: Rambarun Komaljeet

The retriever has to be right before any RAG result means anything — a bad
retriever and a model that cannot read produce the same failure, and #137's
whole design rests on being able to tell them apart.
"""

from __future__ import annotations

import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.retrieval import BM25Index, fit_context, tokenize  # noqa: E402


DOCS = [
    "The Zorvex is called Blenthar.",
    "The Kravilo is called Fimeth.",
    "The Wexarn is called Nalorix.",
    "Paris is the capital of France.",
    "The capital of Japan is Tokyo.",
]


def test_tokenize_lowercases_and_drops_punctuation():
    assert tokenize("The Zorvex is called Blenthar.") == [
        "the", "zorvex", "is", "called", "blenthar"]


def test_finds_the_document_containing_a_rare_term():
    idx = BM25Index(DOCS)
    assert idx.retrieve("What is the Zorvex?", k=1) == ["The Zorvex is called Blenthar."]


def test_rare_terms_outrank_common_ones():
    """The property the whole approach leans on.

    'the' and 'is' appear everywhere and must contribute almost nothing; a name
    appearing in one document out of many must dominate. Without this, a query
    like 'What is the Zorvex?' would just return the shortest document.
    """
    idx = BM25Index(DOCS)
    ranked = idx.search("What is the Kravilo?", k=len(DOCS))
    assert idx.documents[ranked[0][0]] == "The Kravilo is called Fimeth."
    assert ranked[0][1] > 2 * ranked[1][1]


def test_zero_score_documents_are_dropped():
    """Context is the scarcest resource here — padding it with noise is a cost."""
    idx = BM25Index(DOCS)
    assert idx.retrieve("Zorvex", k=5) == ["The Zorvex is called Blenthar."]


def test_a_query_of_only_unknown_terms_matches_nothing():
    idx = BM25Index(DOCS)
    assert idx.retrieve("Nonexistentword Otherfakeword", k=3) == []


def test_stopwords_alone_still_match_but_rank_below_the_real_hit():
    """A caveat worth pinning rather than discovering during a run.

    A natural query like "What is the Zorvex?" carries stopwords that occur in
    every document, so `k=3` returns the target PLUS two stopword matches — the
    context gets padded with irrelevant facts. That is inherent to top-k and is
    why #137's oracle arm exists: it isolates how much the padding costs.
    """
    idx = BM25Index(DOCS)
    got = idx.retrieve("What is the Zorvex?", k=3)
    assert got[0] == "The Zorvex is called Blenthar."
    assert len(got) > 1          # stopword matches follow — noise, but ranked below


def test_scales_to_a_realistic_haystack():
    """2,000 facts is the #137 probe's corpus size; the target must still win."""
    docs = [f"The Ent{i} is called Val{i}." for i in range(2000)]
    idx = BM25Index(docs)
    got = idx.retrieve("What is the Ent1742?", k=3)
    assert got[0] == "The Ent1742 is called Val1742."


def test_empty_corpus_is_rejected_loudly():
    with pytest.raises(ValueError):
        BM25Index([])


def test_search_k_larger_than_corpus_is_safe():
    idx = BM25Index(DOCS[:2])
    assert len(idx.search("The Zorvex", k=99)) <= 2


# --------------------------------------------------------------------------
# context budget — the binding constraint of the whole track
# --------------------------------------------------------------------------


class FakeTok:
    """One token per word — enough to test the budget logic without transformers."""
    def encode(self, s):
        return s.split()


def test_fit_context_stops_at_the_budget():
    passages = ["a b c", "d e f", "g h i"]
    assert fit_context(passages, FakeTok(), budget_tokens=6) == ["a b c", "d e f"]


def test_fit_context_keeps_best_first_order():
    """BM25 returns best-first; truncation must drop the WORST, not the best."""
    passages = ["best match here", "second one", "third"]
    kept = fit_context(passages, FakeTok(), budget_tokens=5)
    assert kept[0] == "best match here"


def test_fit_context_can_return_nothing_rather_than_overflow():
    """Silently truncating a passage would make a null result unattributable."""
    assert fit_context(["a b c d e"], FakeTok(), budget_tokens=3) == []
