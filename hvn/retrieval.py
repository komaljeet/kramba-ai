"""BM25 retrieval — knowledge in a database instead of in the weights (Issue #137).

Written by: Rambarun Komaljeet

Okapi BM25 over a chunked corpus. Pure Python + numpy, no new dependency and no
second neural model.

**Why lexical and not embeddings**, stated so the choice is not mistaken for
laziness: BM25 is the honest floor. If a dense retriever is proposed later it
has to beat this, not merely work — and adding an embedding model to an
experiment that has not yet shown retrieval helps at all would confound "is the
retriever good" with "can this model use retrieved text". #122's retraction came
from exactly that kind of entangled measurement.

BM25 also happens to be strong on the case that matters here: a rare entity name
appearing in one document out of thousands gets a very high IDF, which is
precisely the "find the passage containing this specific thing" query shape.

Scoring (Robertson/Sparck Jones):

    score(q, d) = sum_t IDF(t) * f(t,d) * (k1 + 1)
                  / (f(t,d) + k1 * (1 - b + b * |d| / avgdl))

    IDF(t) = ln(1 + (N - n_t + 0.5) / (n_t + 0.5))
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Sequence, Tuple

import numpy as np

__all__ = ["tokenize", "BM25Index"]

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> List[str]:
    """Lowercased alphanumeric tokens.

    Deliberately not the GPT-2 tokenizer: retrieval matches whole words, and
    BPE would split a rare entity name into fragments shared with unrelated
    words, which is the opposite of the high-IDF behaviour this relies on.
    """
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """An in-memory BM25 index over a fixed list of documents.

    Built once and queried many times. Sized for the corpora this project can
    hold in RAM; there is no on-disk index, because adding one before knowing
    whether retrieval helps at all would be premature.
    """

    def __init__(self, documents: Sequence[str], *, k1: float = 1.5, b: float = 0.75):
        if not documents:
            raise ValueError("BM25Index needs at least one document")
        self.documents: List[str] = list(documents)
        self.k1 = k1
        self.b = b

        tokenized = [tokenize(d) for d in self.documents]
        self.doc_len = np.array([len(t) for t in tokenized], dtype=np.float64)
        self.avgdl = float(self.doc_len.mean()) if len(self.doc_len) else 0.0
        n_docs = len(self.documents)

        # term -> (doc indices, term frequencies), as parallel arrays so scoring
        # a query touches only the postings of its own terms.
        postings: dict[str, dict[int, int]] = {}
        for i, toks in enumerate(tokenized):
            for t in toks:
                postings.setdefault(t, {})
                postings[t][i] = postings[t].get(i, 0) + 1

        self._postings: dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        self._idf: dict[str, float] = {}
        for term, docs in postings.items():
            idx = np.fromiter(docs.keys(), dtype=np.int64, count=len(docs))
            tf = np.fromiter(docs.values(), dtype=np.float64, count=len(docs))
            self._postings[term] = (idx, tf)
            n_t = len(docs)
            self._idf[term] = math.log(1.0 + (n_docs - n_t + 0.5) / (n_t + 0.5))

    def __len__(self) -> int:
        return len(self.documents)

    def scores(self, query: str) -> np.ndarray:
        """BM25 score of every document against ``query``."""
        out = np.zeros(len(self.documents), dtype=np.float64)
        denom_norm = 1.0 - self.b + self.b * (self.doc_len / max(self.avgdl, 1e-12))
        for term in tokenize(query):
            posting = self._postings.get(term)
            if posting is None:
                continue                      # term appears in no document
            idx, tf = posting
            contrib = (self._idf[term] * tf * (self.k1 + 1.0)
                       / (tf + self.k1 * denom_norm[idx]))
            out[idx] += contrib
        return out

    def search(self, query: str, k: int = 3) -> List[Tuple[int, float]]:
        """Top-``k`` ``(doc_index, score)``, best first. Zero-score docs dropped.

        A zero score means not one query term occurs in the document, so
        returning it would be padding the context with noise — and context here
        is the scarcest resource in the whole pipeline (see ``fit_context``).
        """
        s = self.scores(query)
        k = min(k, len(s))
        if k <= 0:
            return []
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top if s[i] > 0.0]

    def retrieve(self, query: str, k: int = 3) -> List[str]:
        return [self.documents[i] for i, _ in self.search(query, k)]


def fit_context(passages: Iterable[str], tokenizer, budget_tokens: int) -> List[str]:
    """Keep passages, best-first, while they fit inside ``budget_tokens``.

    The binding constraint of this whole track: every EdgeHVN checkpoint was
    trained at ``seq_len=512``, and sliding-window models do not reliably
    extrapolate past their trained length (#135). So retrieved context is a
    ~512-token budget shared with the question and the answer — one or two short
    passages, not ten. Truncating silently would make a "RAG doesn't help"
    result unattributable, so this returns only what actually fits and the
    caller reports the count.
    """
    kept: List[str] = []
    used = 0
    for p in passages:
        n = len(tokenizer.encode(p))
        if used + n > budget_tokens:
            break
        kept.append(p)
        used += n
    return kept
