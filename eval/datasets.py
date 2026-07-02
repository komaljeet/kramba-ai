"""Dataset loaders for HVN evaluation harnesses.

Each loader returns a ``(train_text, eval_text)`` tuple of raw strings.
Loaders try a local file path first (if provided), then fall back to
downloading from a public URL, then to a bundled synthetic fallback so
the harness always runs even without internet access.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Callable

# ---------------------------------------------------------------------------
# Public dataset sources
# ---------------------------------------------------------------------------

WIKITEXT_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/data/wikitext-2/"
)
WIKITEXT_TRAIN_FILE = "wiki.train.tokens"
WIKITEXT_EVAL_FILE = "wiki.valid.tokens"

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# A small Python code corpus (stdlib source snippets) for structured generation tests
CODE_CORPUS_URL = (
    "https://raw.githubusercontent.com/python/cpython/main/Lib/collections/__init__.py"
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache")


def _ensure_cache_dir() -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def _download(url: str, filename: str) -> str:
    """Download *url* to the local cache and return the file path."""
    cache_dir = _ensure_cache_dir()
    local_path = os.path.join(cache_dir, filename)
    if os.path.exists(local_path):
        return local_path
    print(f"  Downloading {url} ...")
    urllib.request.urlretrieve(url, local_path)
    print(f"  Cached to {local_path}")
    return local_path


def _read_text(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


# ---------------------------------------------------------------------------
# Dataset loaders — each returns (train_text, eval_text)
# ---------------------------------------------------------------------------

def load_wikitext(local_dir: str | None = None) -> tuple[str, str]:
    """Load WikiText-2 (a subset of WikiText-103).

    Parameters
    ----------
    local_dir : str, optional
        Path to a directory containing ``wiki.train.tokens`` and
        ``wiki.valid.tokens``.  If not provided, the files are downloaded.
    """
    if local_dir:
        train_path = os.path.join(local_dir, WIKITEXT_TRAIN_FILE)
        eval_path = os.path.join(local_dir, WIKITEXT_EVAL_FILE)
        if os.path.exists(train_path) and os.path.exists(eval_path):
            return _read_text(train_path), _read_text(eval_path)

    train_path = _download(WIKITEXT_URL + WIKITEXT_TRAIN_FILE, WIKITEXT_TRAIN_FILE)
    eval_path = _download(WIKITEXT_URL + WIKITEXT_EVAL_FILE, WIKITEXT_EVAL_FILE)
    return _read_text(train_path), _read_text(eval_path)


def load_shakespeare(local_path: str | None = None) -> tuple[str, str]:
    """Load Tiny Shakespeare.

    The full text is split 90/10 into train/eval.
    """
    if local_path and os.path.exists(local_path):
        text = _read_text(local_path)
    else:
        path = _download(SHAKESPEARE_URL, "tiny_shakespeare.txt")
        text = _read_text(path)

    split = int(len(text) * 0.9)
    return text[:split], text[split:]


def load_code_corpus(local_path: str | None = None) -> tuple[str, str]:
    """Load a Python code corpus for structured generation evaluation.

    Uses CPython's ``collections/__init__.py`` as a representative code
    sample.  The text is split 90/10 into train/eval.
    """
    if local_path and os.path.exists(local_path):
        text = _read_text(local_path)
    else:
        path = _download(CODE_CORPUS_URL, "python_code_corpus.py")
        text = _read_text(path)

    split = int(len(text) * 0.9)
    return text[:split], text[split:]


# ---------------------------------------------------------------------------
# Registry for CLI-driven dispatch
# ---------------------------------------------------------------------------

DATASET_REGISTRY: dict[str, Callable[..., tuple[str, str]]] = {
    "wikitext": load_wikitext,
    "shakespeare": load_shakespeare,
    "code": load_code_corpus,
}