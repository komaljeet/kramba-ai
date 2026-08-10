"""Collect a code corpus by shallow-cloning real repositories (Issues #139/#136).

Written by: Rambarun Komaljeet

**Why not HuggingFace.** `bigcode/the-stack-dedup` is gated — it lists
anonymously but returns `401 GatedRepoError` on download, and this machine's
cached HF token is malformed (#120), so there is no anonymous path.
`codeparrot/github-code-clean` is ungated but ships 880 parquet shards; a single
shard was still downloading after 7 minutes here.

**A tokenizer does not need a curated dataset.** It needs breadth of *syntax*,
which a few dozen real repositories supply directly — free, unauthenticated,
ungated, and current rather than a 2022 snapshot. The full pretrain corpus is a
different problem and belongs to #136.

`--depth 1` and `--filter=blob:none` keep each clone to the working tree only,
no history.

**Licensing, stated because it matters for anything trained on this.** The
default list is permissive-licensed (MIT / Apache-2.0 / BSD / PSF) on purpose.
A tokenizer learns sub-word statistics rather than reproducing content, which is
the weakest possible form of use — but a model trained on the same corpus is a
stronger claim, so the constraint is applied here rather than left to be
remembered later.

Usage
-----
    python scripts/collect_code_corpus.py --out data_cache/code_corpus
    python scripts/collect_code_corpus.py --stats-only
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from collections import Counter

# Permissive-licensed, syntactically diverse, and large enough to matter.
# Weighted toward what this project would plausibly read and write.
REPOS = [
    # Python
    "https://github.com/psf/requests",
    "https://github.com/pallets/flask",
    "https://github.com/encode/httpx",
    "https://github.com/scikit-learn/scikit-learn",
    "https://github.com/numpy/numpy",
    "https://github.com/pytest-dev/pytest",
    # JS / TS
    "https://github.com/expressjs/express",
    "https://github.com/axios/axios",
    "https://github.com/microsoft/TypeScript",
    "https://github.com/vuejs/core",
    # Go
    "https://github.com/gin-gonic/gin",
    "https://github.com/spf13/cobra",
    # Rust
    "https://github.com/serde-rs/serde",
    "https://github.com/clap-rs/clap",
    # C / C++
    "https://github.com/nlohmann/json",
    "https://github.com/curl/curl",
    "https://github.com/redis/redis",
    # Java
    "https://github.com/google/gson",
    "https://github.com/square/retrofit",
    # Shell / config / docs-heavy
    "https://github.com/ohmyzsh/ohmyzsh",
    "https://github.com/kubernetes/website",
]

EXTENSIONS = {
    ".py", ".js", ".ts", ".tsx", ".jsx", ".go", ".rs", ".c", ".h", ".cpp",
    ".hpp", ".cc", ".java", ".sh", ".bash", ".zsh", ".sql", ".html", ".css",
    ".json", ".yaml", ".yml", ".toml", ".md", ".rst",
}
MAX_FILE_BYTES = 512 * 1024      # skip minified bundles, lockfiles, vendored blobs


# A clone is only "cached" if this marker exists. Directory presence is NOT
# enough: a clone that dies partway leaves a populated directory behind, and on
# Windows the shutil.rmtree meant to clean it up ALSO fails on the same
# over-260-character paths that killed the clone. The result was a 38 MB partial
# checkout of microsoft/TypeScript reported as "cached" on the retry — a
# silently incomplete corpus, which is worse than a loud failure.
_DONE_MARKER = ".collect_complete"


def clone(url: str, dest: str) -> bool:
    if os.path.exists(os.path.join(dest, _DONE_MARKER)):
        print(f"[code] cached {os.path.basename(dest)}")
        return True
    if os.path.isdir(dest):
        print(f"[code] incomplete clone found, removing {os.path.basename(dest)}")
        _force_rmtree(dest)
    # -c http.sslBackend=schannel: use the WINDOWS certificate store instead of
    # git's bundled OpenSSL one. Avast's HTTPS-scanning root cert is malformed
    # under strict X.509 rules but IS trusted by the Windows store (browsers
    # work; git and Python's certifi do not) -- so this keeps certificate
    # verification ON, unlike the sslVerify=false that would also "work".
    # Without it every clone fails with "unable to get local issuer certificate".
    # Scoped to this command; the user's global git config is untouched.
    # core.longpaths: Windows' 260-character MAX_PATH kills clones of repos with
    # deep test fixtures — microsoft/TypeScript fails on
    # tests/baselines/reference/config/... without it. Also scoped, not global.
    cmd = ["git", "-c", "http.sslBackend=schannel", "-c", "core.longpaths=true",
           "clone", "--depth", "1", "--filter=blob:none",
           "--quiet", url, dest]
    try:
        subprocess.run(cmd, check=True, timeout=600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        with open(os.path.join(dest, _DONE_MARKER), "w") as fh:
            fh.write(url)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[code] FAILED {url}: {e.stderr.decode()[:120]}")
    except subprocess.TimeoutExpired:
        print(f"[code] TIMEOUT {url}")
    # One unreachable repo must not lose the whole collection.
    _force_rmtree(dest)
    return False


def _force_rmtree(path: str) -> None:
    """Remove a tree that may contain paths over Windows' 260-char limit.

    ``shutil.rmtree`` fails silently-ish on those, which is how a partial clone
    survived a "cleanup" and was later mistaken for a complete one.
    """
    shutil.rmtree(path, ignore_errors=True)
    if os.path.isdir(path) and sys.platform == "win32":
        # \?\ prefix lifts MAX_PATH for the Win32 API.
        subprocess.run(["cmd", "/c", "rmdir", "/s", "/q",
                        "\\?\\" + os.path.abspath(path)],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if os.path.isdir(path):
        print(f"[code] WARNING could not fully remove {path} — "
              f"delete it by hand before re-running, or the corpus will be "
              f"silently incomplete")


def harvest(root: str, out_path: str, max_file_bytes: int,
            max_bytes_per_ext: int | None = None) -> dict:
    """Concatenate every source file into one text file, one document per file.

    Files are separated by a blank line and written in a stable order so the
    output is reproducible; a tokenizer trained twice on the same clones should
    produce the same merges.
    """
    by_ext: Counter = Counter()
    bytes_by_ext: Counter = Counter()
    skipped_large = skipped_binary = skipped_capped = 0
    n_files = 0

    with open(out_path, "w", encoding="utf-8", errors="ignore") as out:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in sorted(dirnames)
                           if d not in (".git", "node_modules", "vendor",
                                        "third_party", "dist", "build")]
            for fn in sorted(filenames):
                ext = os.path.splitext(fn)[1].lower()
                if ext not in EXTENSIONS:
                    continue
                path = os.path.join(dirpath, fn)
                try:
                    size = os.path.getsize(path)
                except OSError:
                    continue
                if size > max_file_bytes:
                    skipped_large += 1
                    continue
                try:
                    with open(path, encoding="utf-8") as fh:
                        text = fh.read()
                except (UnicodeDecodeError, OSError):
                    skipped_binary += 1
                    continue
                if not text.strip():
                    continue
                # Per-extension byte cap. Without it the mix is an accident of
                # which repos happen to be large: the first pass came out 36%
                # markdown (64 MB) against 17% Python (30 MB), because
                # kubernetes/website is a documentation site. A tokenizer spends
                # its merge budget on whatever it sees most, so an uncapped mix
                # silently buys markdown merges with the budget meant for code.
                if (max_bytes_per_ext is not None
                        and bytes_by_ext[ext] + len(text) > max_bytes_per_ext):
                    skipped_capped += 1
                    continue
                out.write(text)
                out.write("\n\n")
                by_ext[ext] += 1
                bytes_by_ext[ext] += len(text)
                n_files += 1

    return {"n_files": n_files, "by_ext": dict(by_ext.most_common()),
            "bytes_by_ext": dict(bytes_by_ext.most_common()),
            "skipped_large": skipped_large, "skipped_binary": skipped_binary,
            "skipped_capped": skipped_capped}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_cache/code_corpus")
    ap.add_argument("--text-out", default="data_cache/code_corpus/code_all.txt")
    ap.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES)
    ap.add_argument("--repos", nargs="+", default=REPOS)
    ap.add_argument("--max-mb-per-ext", type=int, default=25,
                    help="cap MB per file extension so the language mix is a "
                         "DECISION, not a side effect of repo sizes. 0 disables.")
    ap.add_argument("--stats-only", action="store_true",
                    help="re-harvest already-cloned repos without cloning")
    args = ap.parse_args()

    clones_dir = os.path.join(args.out, "repos")
    os.makedirs(clones_dir, exist_ok=True)

    if not args.stats_only:
        t0 = time.time()
        ok = 0
        for i, url in enumerate(args.repos, 1):
            name = url.rstrip("/").split("/")[-1]
            print(f"[code] [{i}/{len(args.repos)}] {name}", flush=True)
            ok += clone(url, os.path.join(clones_dir, name))
        print(f"[code] {ok}/{len(args.repos)} cloned in "
              f"{(time.time()-t0)/60:.1f} min")

    os.makedirs(os.path.dirname(args.text_out) or ".", exist_ok=True)
    stats = harvest(clones_dir, args.text_out, args.max_file_bytes,
                    args.max_mb_per_ext * 1024 ** 2 if args.max_mb_per_ext else None)
    total = os.path.getsize(args.text_out)
    print(f"\n[code] {stats['n_files']:,} files -> {args.text_out} "
          f"({total/1024**2:.0f} MB)")
    print(f"[code] skipped {stats['skipped_large']} oversized, "
          f"{stats['skipped_binary']} undecodable, "
          f"{stats['skipped_capped']} over the per-extension cap")
    print(f"[code] {'ext':>8} {'files':>8} {'MB':>8}")
    for ext, n in list(stats["by_ext"].items())[:14]:
        print(f"[code] {ext:>8} {n:>8,} {stats['bytes_by_ext'][ext]/1024**2:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
