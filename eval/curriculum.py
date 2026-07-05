"""Curriculum dataset loaders for the coding-curriculum milestone (Issue #53).

Milestone 2, Issue C of 5. Adds per-language code corpora for the target
curriculum order

    HTML → CSS → JS → PHP → Python → DevOps

so the #54 orchestrator can train the stages in order and the #55 eval harness
can score each one. The public entry point is :func:`load_curriculum_stage`,
which returns a ``(train_text, eval_text)`` tuple exactly like the WikiText /
Shakespeare loaders in :mod:`eval.datasets`.

Design goals (all inherited from the existing loaders + the issue brief)
-----------------------------------------------------------------------
* **License filtering — permissive only.** Every document is passed through
  :func:`is_permissive_source`, which rejects any file carrying a copyleft /
  share-alike marker (GPL/LGPL/AGPL, MPL, EPL, CC-BY-SA). Only MIT/Apache/BSD/
  ISC/Unlicense-style permissive code survives.
* **Near-dedup.** :func:`dedup_documents` removes exact duplicates and
  near-duplicates (MinHash-estimated Jaccard over word-shingles) so a stage's
  budget isn't wasted on boilerplate repeats — the same concern that motivates
  dedup on The-Stack / StarCoder-data subsets.
* **Streaming + local cache.** Mirrors the WikiText-103 loader: read a local
  directory first (``<local_dir>/<stage>/*``), then best-effort download a small
  curated set of permissively-licensed files into ``eval/_cache/curriculum``,
  then fall back to a bundled synthetic corpus so the harness **always** runs on
  the offline 6 GB box.
* **Compute-matched budgets.** Per-stage character caps (``CURRICULUM_BUDGETS``)
  bound memory and give the #54 scheduler compute-matched, unsaturated stages
  (§8.4 methodology). Train/valid split is a deterministic 90/10.

This module deliberately does **not** bundle multi-GB corpora. The synthetic
fallback exists so tests / CI and an offline laptop can smoke-load every stage;
for a real curriculum run, drop permissively-licensed files into a local dir and
pass ``local_dir=...`` (or let the best-effort download populate the cache).
"""

from __future__ import annotations

import os
import re
import random
import urllib.request
import zlib
from typing import Callable

# ---------------------------------------------------------------------------
# Stage configuration
# ---------------------------------------------------------------------------

# The curriculum order is significant — the #54 orchestrator trains it in this
# sequence (simple markup → styling → scripting → server → general-purpose →
# ops), carrying the checkpoint forward between stages.
CURRICULUM_STAGES: list[str] = ["html", "css", "js", "php", "python", "devops"]

# File extensions that belong to each stage (used when reading a local dir).
CURRICULUM_EXTENSIONS: dict[str, tuple[str, ...]] = {
    "html": (".html", ".htm"),
    "css": (".css", ".scss"),
    "js": (".js", ".mjs", ".jsx"),
    "php": (".php",),
    "python": (".py",),
    "devops": (".yml", ".yaml", ".sh", ".tf", "Dockerfile", ".dockerfile"),
}

# Per-stage character budgets (train, eval). Compute-matched, deliberately small
# so a full six-stage run fits the 6 GB / limited-disk box. These are documented
# defaults; callers may override via ``max_train_chars`` / ``max_eval_chars``.
CURRICULUM_BUDGETS: dict[str, tuple[int, int]] = {
    "html": (1_500_000, 150_000),
    "css": (1_500_000, 150_000),
    "js": (2_000_000, 200_000),
    "php": (2_000_000, 200_000),
    "python": (3_000_000, 300_000),
    "devops": (1_000_000, 100_000),
}

# Best-effort curated download sources — raw URLs of files from repositories with
# a well-known **permissive** license (MIT/Apache/BSD/PSF). Kept short and robust;
# each downloaded file is still license-filtered and deduped before use, so a
# stale URL just falls through to the synthetic fallback.
CURRICULUM_SOURCES: dict[str, tuple[str, ...]] = {
    "html": (
        # html5-boilerplate (MIT)
        "https://raw.githubusercontent.com/h5bp/html5-boilerplate/main/dist/index.html",
        "https://raw.githubusercontent.com/h5bp/html5-boilerplate/main/dist/404.html",
    ),
    "css": (
        # normalize.css (MIT)
        "https://raw.githubusercontent.com/necolas/normalize.css/master/normalize.css",
    ),
    "js": (
        # mithril.js (MIT) — single small readable file
        "https://raw.githubusercontent.com/lhorie/mithril.js/next/api/mithril.js",
    ),
    "php": (
        # Slim framework (MIT)
        "https://raw.githubusercontent.com/slimphp/Slim/4.x/Slim/App.php",
    ),
    "python": (
        # CPython stdlib (PSF, permissive) — reuse the existing #2 corpus source.
        "https://raw.githubusercontent.com/python/cpython/main/Lib/collections/__init__.py",
        "https://raw.githubusercontent.com/python/cpython/main/Lib/functools.py",
    ),
    "devops": (
        # Docker's official CLI (Apache-2.0) Dockerfile + a permissive GH Actions file.
        "https://raw.githubusercontent.com/docker/cli/master/Dockerfile",
    ),
}

# ---------------------------------------------------------------------------
# License filtering (permissive only)
# ---------------------------------------------------------------------------

# Copyleft / share-alike markers. If any of these appears in a document it is
# rejected — the curriculum trains on permissive code only.
_COPYLEFT_PATTERNS: tuple[str, ...] = (
    r"gnu\s+general\s+public\s+license",
    r"gnu\s+lesser\s+general\s+public",
    r"gnu\s+affero\s+general\s+public",
    r"\bg\.?p\.?l\b",                 # GPL / G.P.L
    r"\blgpl\b",
    r"\bagpl\b",
    r"mozilla\s+public\s+license",
    r"\bmpl-2\.0\b",
    r"eclipse\s+public\s+license",
    r"\bepl-\d",
    r"creative\s+commons\s+attribution[- ]sharealike",
    r"\bcc[- ]by[- ]sa\b",
    r"copyleft",
)
_COPYLEFT_RE = re.compile("|".join(_COPYLEFT_PATTERNS), re.IGNORECASE)


def is_permissive_source(text: str) -> bool:
    """Return ``True`` if *text* carries no copyleft / share-alike license marker.

    Conservative by design: a document that mentions a copyleft license (even in
    a header comment) is dropped so only permissive-licensed code is trained on.
    Permissive licenses (MIT/Apache/BSD/ISC/Unlicense/PSF) and unlicensed
    snippets pass through.
    """
    return _COPYLEFT_RE.search(text) is None


# ---------------------------------------------------------------------------
# Near-dedup (MinHash-estimated Jaccard over word-shingles)
# ---------------------------------------------------------------------------

_WORD_RE = re.compile(r"\S+")


def _shingles(text: str, k: int = 5) -> set[int]:
    """Return the set of hashed k-word shingles for *text* (whitespace-normalized)."""
    words = _WORD_RE.findall(text)
    if len(words) < k:
        # Short doc: hash the whole thing as a single shingle.
        return {zlib.crc32(" ".join(words).encode("utf-8", "replace"))}
    shingles: set[int] = set()
    for i in range(len(words) - k + 1):
        gram = " ".join(words[i:i + k])
        shingles.add(zlib.crc32(gram.encode("utf-8", "replace")))
    return shingles


def _minhash(shingles: set[int], num_hashes: int = 24) -> tuple[int, ...]:
    """Deterministic MinHash signature of a shingle set (empty → all-zero)."""
    if not shingles:
        return tuple([0] * num_hashes)
    sig = []
    for seed in range(num_hashes):
        salt = (seed + 1) * 0x9E3779B1
        sig.append(min((h ^ salt) & 0xFFFFFFFF for h in shingles))
    return tuple(sig)


def _sig_jaccard(a: tuple[int, ...], b: tuple[int, ...]) -> float:
    """Estimate Jaccard similarity from two MinHash signatures."""
    if not a:
        return 0.0
    matches = sum(1 for x, y in zip(a, b) if x == y)
    return matches / len(a)


def dedup_documents(
    docs: list[str],
    threshold: float = 0.8,
    shingle_k: int = 5,
    max_docs: int | None = None,
) -> list[str]:
    """Remove exact and near-duplicate documents.

    Exact duplicates (by normalized-whitespace hash) are dropped first; the
    remainder are greedily kept unless their MinHash-estimated Jaccard against an
    already-kept document is ``>= threshold``. Order is preserved.
    """
    kept: list[str] = []
    kept_sigs: list[tuple[int, ...]] = []
    seen_exact: set[int] = set()

    for doc in docs:
        if max_docs is not None and len(kept) >= max_docs:
            break
        norm = " ".join(doc.split())
        if not norm:
            continue
        exact_key = zlib.crc32(norm.encode("utf-8", "replace"))
        if exact_key in seen_exact:
            continue
        seen_exact.add(exact_key)

        sig = _minhash(_shingles(doc, shingle_k))
        if any(_sig_jaccard(sig, ks) >= threshold for ks in kept_sigs):
            continue
        kept.append(doc)
        kept_sigs.append(sig)

    return kept


# ---------------------------------------------------------------------------
# Cache / local-dir / download plumbing (mirrors eval/datasets.py)
# ---------------------------------------------------------------------------

_CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_cache", "curriculum")

_MIT_HEADER = (
    "/* SPDX-License-Identifier: MIT\n"
    " * Synthetic curriculum sample — permissively licensed, generated for the\n"
    " * HVN coding-curriculum (Issue #53). Not from any third-party corpus. */\n"
)
_PY_MIT_HEADER = (
    "# SPDX-License-Identifier: MIT\n"
    "# Synthetic curriculum sample (Issue #53) — permissively licensed.\n"
)


def _ensure_cache_dir() -> str:
    os.makedirs(_CACHE_DIR, exist_ok=True)
    return _CACHE_DIR


def _read_local_dir(directory: str, stage: str) -> list[str]:
    """Read every stage-matching file under *directory* as one document each."""
    exts = CURRICULUM_EXTENSIONS[stage]
    docs: list[str] = []
    if not os.path.isdir(directory):
        return docs
    for root, _dirs, files in os.walk(directory):
        for name in files:
            if name.endswith(exts) or name in exts:
                path = os.path.join(root, name)
                try:
                    with open(path, "r", encoding="utf-8", errors="replace") as f:
                        docs.append(f.read())
                except OSError:
                    continue
    return docs


def _download_stage_sources(stage: str) -> list[str]:
    """Best-effort download of curated permissive files for *stage* into the cache.

    Silently skips any URL that fails so the loader still falls back to the
    synthetic corpus in offline / locked-down environments.
    """
    cache_dir = _ensure_cache_dir()
    stage_dir = os.path.join(cache_dir, stage)
    os.makedirs(stage_dir, exist_ok=True)
    docs: list[str] = []
    for i, url in enumerate(CURRICULUM_SOURCES.get(stage, ())):
        fname = f"{stage}_{i}_" + os.path.basename(url).replace("/", "_")
        local_path = os.path.join(stage_dir, fname)
        if not os.path.exists(local_path):
            try:
                print(f"  [curriculum:{stage}] downloading {url} ...")
                urllib.request.urlretrieve(url, local_path)
            except Exception as exc:  # pragma: no cover - network dependent
                print(f"  [curriculum:{stage}] skip {url} ({type(exc).__name__})")
                continue
        try:
            with open(local_path, "r", encoding="utf-8", errors="replace") as f:
                docs.append(f.read())
        except OSError:  # pragma: no cover
            continue
    return docs


# ---------------------------------------------------------------------------
# Synthetic fallback corpora (offline-safe, license-clean, deduped by design)
# ---------------------------------------------------------------------------
#
# Each generator returns a list of *distinct* documents parametrized by a seeded
# RNG so the fallback always produces enough deduped, on-language material to
# smoke-load a stage on an offline box. These are original templated snippets
# (no third-party code), carrying an MIT header so the license filter passes.

_NOUNS = ["user", "order", "product", "invoice", "session", "widget", "account",
          "payment", "profile", "cart", "message", "report", "token", "vortex",
          "hamiltonian", "particle", "cluster", "channel", "record", "document"]
_ADJS = ["active", "pending", "archived", "primary", "secondary", "cached",
         "compact", "verbose", "nested", "flat", "async", "batched"]
_VERBS = ["create", "update", "delete", "fetch", "render", "compute", "validate",
          "serialize", "resolve", "dispatch", "encode", "decode", "merge", "flush"]


def _pick(rng: random.Random, seq):
    return seq[rng.randrange(len(seq))]


def _gen_html(rng: random.Random, n: int) -> list[str]:
    docs = []
    for _ in range(n):
        title = f"{_pick(rng, _ADJS).title()} {_pick(rng, _NOUNS).title()}"
        items = "\n".join(
            f"      <li class=\"{_pick(rng, _ADJS)}\">{_pick(rng, _VERBS)} {_pick(rng, _NOUNS)}</li>"
            for _ in range(rng.randint(3, 7))
        )
        field = _pick(rng, _NOUNS)
        docs.append(
            f"<!-- SPDX-License-Identifier: MIT -->\n"
            f"<!DOCTYPE html>\n<html lang=\"en\">\n<head>\n"
            f"  <meta charset=\"utf-8\">\n"
            f"  <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
            f"  <title>{title}</title>\n</head>\n<body>\n"
            f"  <header><h1>{title}</h1></header>\n"
            f"  <main>\n    <section id=\"{field}\">\n"
            f"      <p>Manage your {field} records below.</p>\n"
            f"      <ul>\n{items}\n      </ul>\n"
            f"      <form action=\"/{field}\" method=\"post\">\n"
            f"        <label for=\"{field}-name\">{field.title()} name</label>\n"
            f"        <input id=\"{field}-name\" name=\"name\" type=\"text\" required>\n"
            f"        <button type=\"submit\">{_pick(rng, _VERBS).title()}</button>\n"
            f"      </form>\n    </section>\n  </main>\n"
            f"  <footer><small>&copy; 2026 Example Co.</small></footer>\n"
            f"</body>\n</html>\n"
        )
    return docs


def _gen_css(rng: random.Random, n: int) -> list[str]:
    props = ["color", "background", "margin", "padding", "display", "font-size",
             "border", "flex", "gap", "width", "height", "position"]
    vals = ["1rem", "0.5rem", "flex", "block", "none", "auto", "100%", "2px solid #ccc",
            "#1a1a1a", "#f5f5f5", "center", "relative", "1.25rem", "700"]
    docs = []
    for _ in range(n):
        rules = []
        for _ in range(rng.randint(4, 9)):
            sel = f".{_pick(rng, _ADJS)}-{_pick(rng, _NOUNS)}"
            body = "\n".join(
                f"  {_pick(rng, props)}: {_pick(rng, vals)};"
                for _ in range(rng.randint(2, 5))
            )
            rules.append(f"{sel} {{\n{body}\n}}")
        docs.append(_MIT_HEADER + "\n" + "\n\n".join(rules) + "\n")
    return docs


def _gen_js(rng: random.Random, n: int) -> list[str]:
    docs = []
    for _ in range(n):
        verb, noun = _pick(rng, _VERBS), _pick(rng, _NOUNS)
        fn = f"{verb}{noun.title()}"
        prop = _pick(rng, _NOUNS)
        docs.append(
            _MIT_HEADER +
            f"\nexport function {fn}({prop}, options = {{}}) {{\n"
            f"  if (!{prop}) throw new Error('{prop} is required');\n"
            f"  const result = {{ ...options, {prop} }};\n"
            f"  return Object.keys(result).reduce((acc, key) => {{\n"
            f"    acc[key] = result[key];\n"
            f"    return acc;\n"
            f"  }}, {{}});\n}}\n\n"
            f"export class {noun.title()}Store {{\n"
            f"  constructor() {{ this.items = new Map(); }}\n"
            f"  {verb}(id, value) {{ this.items.set(id, value); return this; }}\n"
            f"  get(id) {{ return this.items.get(id); }}\n"
            f"  get size() {{ return this.items.size; }}\n}}\n"
        )
    return docs


def _gen_php(rng: random.Random, n: int) -> list[str]:
    docs = []
    for _ in range(n):
        noun = _pick(rng, _NOUNS).title()
        verb = _pick(rng, _VERBS)
        prop = _pick(rng, _NOUNS)
        docs.append(
            "<?php\n// SPDX-License-Identifier: MIT\n\n"
            f"declare(strict_types=1);\n\n"
            f"final class {noun}Service\n{{\n"
            f"    private array ${prop}s = [];\n\n"
            f"    public function {verb}(string $id, array $data): array\n    {{\n"
            f"        $this->{prop}s[$id] = $data;\n"
            f"        return $this->{prop}s[$id];\n    }}\n\n"
            f"    public function all(): array\n    {{\n"
            f"        return array_values($this->{prop}s);\n    }}\n\n"
            f"    public function count(): int\n    {{\n"
            f"        return count($this->{prop}s);\n    }}\n}}\n"
        )
    return docs


def _gen_python(rng: random.Random, n: int) -> list[str]:
    docs = []
    for _ in range(n):
        noun = _pick(rng, _NOUNS)
        verb = _pick(rng, _VERBS)
        cls = noun.title()
        docs.append(
            _PY_MIT_HEADER +
            f"from dataclasses import dataclass, field\n\n\n"
            f"@dataclass\nclass {cls}:\n"
            f"    name: str\n"
            f"    {noun}_id: int = 0\n"
            f"    tags: list[str] = field(default_factory=list)\n\n"
            f"    def {verb}(self, value):\n"
            f"        \"\"\"{verb.title()} the {noun} and return the new state.\"\"\"\n"
            f"        self.tags.append(str(value))\n"
            f"        return self\n\n\n"
            f"def {verb}_{noun}s(items):\n"
            f"    \"\"\"Return the {noun}s for which the predicate holds.\"\"\"\n"
            f"    return [it for it in items if it.{noun}_id >= 0]\n"
        )
    return docs


def _gen_devops(rng: random.Random, n: int) -> list[str]:
    docs = []
    for _ in range(n):
        kind = rng.randrange(3)
        svc = _pick(rng, _NOUNS)
        if kind == 0:  # Dockerfile
            docs.append(
                "# SPDX-License-Identifier: MIT\n"
                f"FROM python:3.12-slim AS {svc}\n"
                f"WORKDIR /app\n"
                f"COPY requirements.txt .\n"
                f"RUN pip install --no-cache-dir -r requirements.txt\n"
                f"COPY . .\n"
                f"EXPOSE {8000 + rng.randint(0, 99)}\n"
                f"CMD [\"python\", \"-m\", \"{svc}\"]\n"
            )
        elif kind == 1:  # GitHub Actions workflow
            docs.append(
                "# SPDX-License-Identifier: MIT\n"
                f"name: {svc}-ci\non:\n  push:\n    branches: [main]\n"
                f"jobs:\n  {svc}:\n    runs-on: ubuntu-latest\n    steps:\n"
                f"      - uses: actions/checkout@v4\n"
                f"      - uses: actions/setup-python@v5\n"
                f"        with:\n          python-version: '3.12'\n"
                f"      - run: pip install -r requirements.txt\n"
                f"      - run: pytest -q\n"
            )
        else:  # k8s manifest
            replicas = rng.randint(1, 5)
            docs.append(
                "# SPDX-License-Identifier: MIT\n"
                f"apiVersion: apps/v1\nkind: Deployment\nmetadata:\n  name: {svc}\n"
                f"spec:\n  replicas: {replicas}\n  selector:\n"
                f"    matchLabels:\n      app: {svc}\n  template:\n    metadata:\n"
                f"      labels:\n        app: {svc}\n    spec:\n      containers:\n"
                f"        - name: {svc}\n          image: registry.example.com/{svc}:latest\n"
                f"          ports:\n            - containerPort: 8080\n"
            )
    return docs


_SYNTHETIC_GENERATORS: dict[str, Callable[[random.Random, int], list[str]]] = {
    "html": _gen_html,
    "css": _gen_css,
    "js": _gen_js,
    "php": _gen_php,
    "python": _gen_python,
    "devops": _gen_devops,
}


def _synthetic_documents(stage: str, seed: int, target_chars: int) -> list[str]:
    """Generate enough distinct synthetic documents to cover *target_chars*."""
    rng = random.Random(seed ^ (zlib.crc32(stage.encode()) & 0xFFFFFFFF))
    gen = _SYNTHETIC_GENERATORS[stage]
    docs: list[str] = []
    total = 0
    # Generate in batches until the budget is covered (with headroom for dedup).
    while total < target_chars * 1.3:
        batch = gen(rng, 64)
        docs.extend(batch)
        total += sum(len(d) for d in batch)
        if len(docs) > 20000:  # hard safety cap
            break
    return docs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _gather_stage_documents(
    stage: str,
    local_dir: str | None,
    allow_download: bool,
    seed: int,
    target_chars: int,
) -> tuple[list[str], str]:
    """Return ``(documents, source_label)`` for *stage*.

    Resolution order (first non-empty wins): local dir → cache/download →
    synthetic fallback.
    """
    if local_dir:
        for cand in (os.path.join(local_dir, stage), local_dir):
            docs = _read_local_dir(cand, stage)
            if docs:
                return docs, f"local:{cand}"

    # Cached download dir (may already be populated from a prior run).
    cache_stage_dir = os.path.join(_CACHE_DIR, stage)
    docs = _read_local_dir(cache_stage_dir, stage)
    if not docs and allow_download:
        docs = _download_stage_sources(stage)
    if docs:
        return docs, "download-cache"

    return _synthetic_documents(stage, seed, target_chars), "synthetic"


def load_curriculum_stage(
    stage: str,
    local_dir: str | None = None,
    max_train_chars: int | None = None,
    max_eval_chars: int | None = None,
    seed: int = 42,
    allow_download: bool = True,
    dedup_threshold: float = 0.8,
) -> tuple[str, str]:
    """Load one curriculum stage as a ``(train_text, eval_text)`` pair.

    Parameters
    ----------
    stage : str
        One of :data:`CURRICULUM_STAGES` (``'html'|'css'|'js'|'php'|'python'|'devops'``).
    local_dir : str, optional
        Directory of pre-fetched code. Files are read from ``<local_dir>/<stage>/``
        (or ``<local_dir>`` directly) filtered by the stage's extensions. This is
        the recommended path for a real curriculum run — drop permissively
        licensed files there. When omitted, the loader best-effort downloads a
        small curated permissive set, then falls back to a synthetic corpus.
    max_train_chars, max_eval_chars : int, optional
        Character caps (default: the stage's :data:`CURRICULUM_BUDGETS` entry).
        Bound memory and give compute-matched stages.
    seed : int
        Deterministic shuffle / synthetic-generation seed.
    allow_download : bool
        If ``False``, never touch the network (offline / CI). Skips straight to
        the cache then the synthetic fallback.
    dedup_threshold : float
        MinHash Jaccard threshold above which a document is treated as a
        near-duplicate and dropped.

    Returns
    -------
    (train_text, eval_text) : tuple[str, str]
        License-filtered, deduped, budget-capped train/valid streams. The split
        is a deterministic 90/10 over the deduped documents.
    """
    if stage not in CURRICULUM_STAGES:
        raise ValueError(
            f"unknown curriculum stage {stage!r}; expected one of {CURRICULUM_STAGES}"
        )

    budget_train, budget_eval = CURRICULUM_BUDGETS[stage]
    if max_train_chars is None:
        max_train_chars = budget_train
    if max_eval_chars is None:
        max_eval_chars = budget_eval

    docs, source = _gather_stage_documents(
        stage, local_dir, allow_download, seed, max_train_chars + max_eval_chars
    )

    # 1. License filter — permissive only.
    docs = [d for d in docs if d.strip() and is_permissive_source(d)]
    # 2. Near-dedup.
    docs = dedup_documents(docs, threshold=dedup_threshold)
    if not docs:
        raise RuntimeError(
            f"curriculum stage {stage!r}: no permissively-licensed documents after "
            f"filtering (source={source})."
        )

    # 3. Deterministic shuffle + 90/10 split (by document, so train/valid are
    #    disjoint files rather than a mid-document cut).
    rng = random.Random(seed)
    rng.shuffle(docs)
    split = max(1, int(len(docs) * 0.9))
    train_docs, eval_docs = docs[:split], docs[split:]
    if not eval_docs:  # tiny corpus: hold out the last doc
        eval_docs = train_docs[-1:]
        train_docs = train_docs[:-1] or eval_docs

    train_text = "\n\n".join(train_docs)[:max_train_chars]
    eval_text = "\n\n".join(eval_docs)[:max_eval_chars]

    print(
        f"  [curriculum:{stage}] source={source} docs={len(docs)} "
        f"train={len(train_text):,} chars eval={len(eval_text):,} chars"
    )
    return train_text, eval_text


def curriculum_stage_info(
    stage: str,
    local_dir: str | None = None,
    allow_download: bool = False,
    seed: int = 42,
) -> dict:
    """Return a small dict of sizes/counts for *stage* (documentation / logging).

    Defaults to ``allow_download=False`` so it is cheap to call for a report.
    """
    train_text, eval_text = load_curriculum_stage(
        stage, local_dir=local_dir, allow_download=allow_download, seed=seed
    )
    return {
        "stage": stage,
        "extensions": CURRICULUM_EXTENSIONS[stage],
        "budget": CURRICULUM_BUDGETS[stage],
        "train_chars": len(train_text),
        "eval_chars": len(eval_text),
    }


def smoke_load_all(local_dir: str | None = None, allow_download: bool = False) -> dict:
    """Smoke-load every curriculum stage and return a ``{stage: info}`` map.

    Used by the #53 test and as a quick CLI (``python -m eval.curriculum``).
    """
    report = {}
    for stage in CURRICULUM_STAGES:
        report[stage] = curriculum_stage_info(
            stage, local_dir=local_dir, allow_download=allow_download
        )
    return report


if __name__ == "__main__":  # pragma: no cover - manual smoke CLI
    import argparse
    import json

    p = argparse.ArgumentParser(description="Smoke-load the coding curriculum (Issue #53)")
    p.add_argument("--local-dir", default=None, help="Directory of pre-fetched code")
    p.add_argument("--download", action="store_true", help="Allow best-effort downloads")
    p.add_argument("--stage", default=None, help="Only load this stage")
    args = p.parse_args()

    if args.stage:
        tr, ev = load_curriculum_stage(
            args.stage, local_dir=args.local_dir, allow_download=args.download
        )
        print(f"\n{args.stage}: {len(tr):,} train chars, {len(ev):,} eval chars")
        print("--- train sample ---")
        print(tr[:500])
    else:
        rep = smoke_load_all(local_dir=args.local_dir, allow_download=args.download)
        print(json.dumps(rep, indent=2, default=str))
