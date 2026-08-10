"""Download a code corpus for the mixed 32K tokenizer (Issues #139 / #136).

Written by: Rambarun Komaljeet

Mirrors ``scripts/download_fineweb_edu.py``'s acquisition pattern, which exists
because plain ``huggingface-cli download`` does not work on this machine:

  - **SSL** — Avast's HTTPS-scanning root certificate is malformed under strict
    X.509 rules, so every HTTPS call fails with
    ``CERTIFICATE_VERIFY_FAILED: Basic Constraints of CA cert not marked
    critical``. Verification is disabled for ``huggingface_hub``'s own client
    only, via ``set_client_factory`` — never system-wide.
  - **Auth** — the cached HF token is malformed (825 bytes, far larger than a
    real one) and produces ``401 OAuth token signature verification failed``.
    Everything here is public, so ``token=False`` is both correct and required.

Both are documented in ``docs/training.md`` (#120).

Why ``the-stack-dedup``: it is laid out per language, so a few GB of exactly the
languages that matter can be pulled instead of the full multi-TB corpus. A
tokenizer needs breadth of *syntax*, not volume — a few GB is ample. The full
download belongs to #136's pretrain, not here.

**Language mix is a real decision, not a default.** The tokenizer's merges are
spent on whatever it sees; a corpus that is 80% Java produces a tokenizer that
handles Python badly. The default list is weighted toward what this project
would plausibly generate.

Usage
-----
    python scripts/download_code_corpus.py --files-per-language 1
    python scripts/download_code_corpus.py --languages python javascript --files-per-language 3
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")

import httpx
from huggingface_hub import hf_hub_download, list_repo_files
from huggingface_hub.utils._http import set_client_factory

set_client_factory(lambda: httpx.Client(verify=False, follow_redirects=True,
                                        timeout=120))

REPO_ID = "bigcode/the-stack-dedup"

# Weighted toward what this project would realistically read and write. Markdown
# is included deliberately: it is the bridge between the prose and the code half
# of a mixed tokenizer, and it is what docstrings, READMEs and issue text look
# like — the text an agentic coding model actually sees most.
DEFAULT_LANGUAGES = [
    "python", "javascript", "typescript", "markdown", "shell",
    "java", "c", "c++", "go", "rust", "html", "json", "yaml", "sql",
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--languages", nargs="+", default=DEFAULT_LANGUAGES)
    ap.add_argument("--files-per-language", type=int, default=1,
                    help="parquet shards per language. 1 is usually enough for "
                         "a tokenizer; raise it only if the measured "
                         "tokens-per-line says the merges are starved.")
    ap.add_argument("--local-dir", default="data_cache/code_corpus")
    ap.add_argument("--list-only", action="store_true")
    args = ap.parse_args()

    print(f"[code] listing {REPO_ID} (anonymous)")
    all_files = list_repo_files(REPO_ID, repo_type="dataset", token=False)

    wanted: list[str] = []
    missing: list[str] = []
    for lang in args.languages:
        shards = sorted(f for f in all_files if f.startswith(f"data/{lang}/"))
        if not shards:
            missing.append(lang)
            continue
        wanted.extend(shards[:args.files_per_language])

    if missing:
        # Loudly, not silently: a typo'd language name would otherwise just
        # produce a tokenizer quietly missing that syntax.
        print(f"[code] WARNING no shards for: {missing} — check the names "
              f"against the repo's data/ directory")

    print(f"[code] {len(wanted)} shards across {len(args.languages) - len(missing)} "
          f"languages")
    for f in wanted:
        print(f"[code]   {f}")
    if args.list_only:
        return 0

    os.makedirs(args.local_dir, exist_ok=True)
    t0 = time.time()
    total = 0
    for i, fn in enumerate(wanted, 1):
        dest = os.path.join(args.local_dir, fn.replace("/", os.sep))
        if os.path.exists(dest):
            total += os.path.getsize(dest)
            print(f"[code] [{i}/{len(wanted)}] cached {fn}")
            continue
        try:
            path = hf_hub_download(repo_id=REPO_ID, repo_type="dataset",
                                   filename=fn, local_dir=args.local_dir,
                                   token=False)
            size = os.path.getsize(path)
            total += size
            print(f"[code] [{i}/{len(wanted)}] {fn} — {size/1024**2:.0f} MB "
                  f"({total/1024**3:.2f} GB total, {(time.time()-t0)/60:.1f} min)",
                  flush=True)
        except Exception as e:
            # One unavailable shard must not lose the whole download.
            print(f"[code] [{i}/{len(wanted)}] FAILED {fn}: "
                  f"{type(e).__name__}: {str(e)[:120]}", flush=True)

    print(f"[code] done: {total/1024**3:.2f} GB in {(time.time()-t0)/60:.1f} min "
          f"-> {args.local_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
