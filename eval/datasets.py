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

# WikiText-103 (~103M tokens) — the real-scale corpus for scaling studies (Issue #35).
# The raw archive is ~181 MB zipped / ~517 MB unzipped. We do NOT bundle it; the loader
# reads pre-extracted files from a local directory (``--local-path``) and, only as a
# best-effort convenience, attempts a download when none is present.
WIKITEXT103_ZIP_URL = "https://wikitext.smerity.com/wikitext-103-v1.zip"
# Accepted filenames inside a local WikiText-103 directory (tokenized *or* raw variants).
WIKITEXT103_TRAIN_CANDIDATES = ("wiki.train.tokens", "wiki.train.raw")
WIKITEXT103_EVAL_CANDIDATES = ("wiki.valid.tokens", "wiki.valid.raw")

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)

# A small Python code corpus (stdlib source snippets) for structured generation tests
CODE_CORPUS_URL = (
    "https://raw.githubusercontent.com/python/cpython/main/Lib/collections/__init__.py"
)

# ---------------------------------------------------------------------------
# Coding-curriculum loaders (Issue #53) — HTML → CSS → JS → PHP → Python → DevOps.
# Implemented in eval/curriculum.py and re-exported here so callers (the #54
# orchestrator, the #55 eval) can use the same ``from eval.datasets import ...``
# entry point as the WikiText / Shakespeare loaders below.
# ---------------------------------------------------------------------------
from eval.curriculum import (  # noqa: E402
    CURRICULUM_STAGES,
    CURRICULUM_BUDGETS,
    CURRICULUM_EXTENSIONS,
    load_curriculum_stage,
    curriculum_stage_info,
    smoke_load_all as smoke_load_curriculum,
    is_permissive_source,
    dedup_documents,
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


def _read_text(path: str, max_chars: int | None = None) -> str:
    """Read a text file, optionally streaming only the first ``max_chars`` characters.

    Streaming (line-by-line, stopping early) keeps memory bounded on the ~517 MB
    WikiText-103 corpus: a reduced-budget scaling run needs only enough characters
    to cover its token budget, not the whole file.
    """
    if max_chars is None:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()

    chunks: list[str] = []
    total = 0
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            chunks.append(line)
            total += len(line)
            if total >= max_chars:
                break
    return "".join(chunks)[:max_chars]


def _find_first(directory: str, candidates: tuple[str, ...]) -> str | None:
    """Return the first existing ``directory/name`` for name in *candidates*."""
    for name in candidates:
        path = os.path.join(directory, name)
        if os.path.exists(path):
            return path
    return None


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


def load_wikitext103(
    local_dir: str | None = None,
    max_train_chars: int | None = None,
    max_eval_chars: int | None = None,
) -> tuple[str, str]:
    """Load WikiText-103 (~103M tokens) — the real-scale corpus for Issue #35.

    Parameters
    ----------
    local_dir : str, optional
        Directory containing extracted WikiText-103 files
        (``wiki.train.tokens``/``.raw`` and ``wiki.valid.tokens``/``.raw``).
        This is the recommended path: download + unzip the archive once, then
        point ``--local-path`` here.
    max_train_chars, max_eval_chars : int, optional
        Character caps for streaming reads. A reduced-budget scaling sweep only
        needs enough characters to cover its token budget, so these bound memory
        instead of loading the full ~517 MB train split.

    Notes
    -----
    The corpus is NOT bundled. If ``local_dir`` is not provided (or lacks the
    files), the loader attempts a best-effort download of the zip archive. In
    locked-down / offline environments that download will fail with a clear
    message instructing you to fetch the archive manually and pass ``--local-path``.
    """
    search_dirs: list[str] = []
    if local_dir:
        search_dirs.append(local_dir)
        # Common layout: the zip extracts into a ``wikitext-103`` subdirectory.
        search_dirs.append(os.path.join(local_dir, "wikitext-103"))
        search_dirs.append(os.path.join(local_dir, "wikitext-103-raw"))
    search_dirs.append(os.path.join(_ensure_cache_dir(), "wikitext-103"))

    for directory in search_dirs:
        if not os.path.isdir(directory):
            continue
        train_path = _find_first(directory, WIKITEXT103_TRAIN_CANDIDATES)
        eval_path = _find_first(directory, WIKITEXT103_EVAL_CANDIDATES)
        if train_path and eval_path:
            print(f"  WikiText-103: reading from {directory}")
            return (
                _read_text(train_path, max_train_chars),
                _read_text(eval_path, max_eval_chars),
            )

    # --- Not found locally: attempt a best-effort download ------------------
    try:
        extract_dir = _download_and_extract_wikitext103()
    except Exception as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            "WikiText-103 not found locally and automatic download failed "
            f"({type(exc).__name__}: {exc}).\n"
            "  Download it once (e.g. from a WikiText-103 mirror or "
            "`datasets.load_dataset('wikitext', 'wikitext-103-raw-v1')`), extract it, "
            "and pass the directory via --local-path. Expected files: "
            f"{WIKITEXT103_TRAIN_CANDIDATES} and {WIKITEXT103_EVAL_CANDIDATES}."
        ) from exc

    train_path = _find_first(extract_dir, WIKITEXT103_TRAIN_CANDIDATES)
    eval_path = _find_first(extract_dir, WIKITEXT103_EVAL_CANDIDATES)
    if not (train_path and eval_path):
        raise RuntimeError(
            f"Downloaded WikiText-103 archive but could not find expected files in {extract_dir}."
        )
    return (
        _read_text(train_path, max_train_chars),
        _read_text(eval_path, max_eval_chars),
    )


def _download_and_extract_wikitext103() -> str:
    """Download + unzip WikiText-103 into the cache, returning the extraction dir."""
    import zipfile

    cache_dir = _ensure_cache_dir()
    zip_path = os.path.join(cache_dir, "wikitext-103-v1.zip")
    if not os.path.exists(zip_path):
        print(f"  Downloading WikiText-103 (~181 MB) from {WIKITEXT103_ZIP_URL} ...")
        urllib.request.urlretrieve(WIKITEXT103_ZIP_URL, zip_path)
        print(f"  Cached to {zip_path}")

    print("  Extracting WikiText-103 ...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(cache_dir)
    # The archive extracts into a ``wikitext-103`` directory.
    extracted = os.path.join(cache_dir, "wikitext-103")
    return extracted if os.path.isdir(extracted) else cache_dir


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
    "wikitext103": load_wikitext103,
    "shakespeare": load_shakespeare,
    "code": load_code_corpus,
}

# Register each curriculum stage under ``curriculum:<stage>`` so the CLI-driven
# harnesses can dispatch to them like any other dataset.
for _stage in CURRICULUM_STAGES:
    DATASET_REGISTRY[f"curriculum:{_stage}"] = (
        lambda local_dir=None, _s=_stage: load_curriculum_stage(_s, local_dir=local_dir)
    )