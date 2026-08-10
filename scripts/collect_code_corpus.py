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


def clone(url: str, dest: str) -> bool:
    if os.path.isdir(dest):
        print(f"[code] cached {os.path.basename(dest)}")
        return True
    # -c http.sslBackend=schannel: use the WINDOWS certificate store instead of
    # git's bundled OpenSSL one. Avast's HTTPS-scanning root cert is malformed
    # under strict X.509 rules but IS trusted by the Windows store (browsers
    # work; git and Python's certifi do not) -- so this keeps certificate
    # verification ON, unlike the sslVerify=false that would also "work".
    # Without it every clone fails with "unable to get local issuer certificate".
    # Scoped to this command; the user's global git config is untouched.
    cmd = ["git", "-c", "http.sslBackend=schannel",
           "clone", "--depth", "1", "--filter=blob:none",
           "--quiet", url, dest]
    try:
        subprocess.run(cmd, check=True, timeout=600,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[code] FAILED {url}: {e.stderr.decode()[:120]}")
    except subprocess.TimeoutExpired:
        print(f"[code] TIMEOUT {url}")
    # One unreachable repo must not lose the whole collection.
    shutil.rmtree(dest, ignore_errors=True)
    return False


def harvest(root: str, out_path: str, max_file_bytes: int) -> dict:
    """Concatenate every source file into one text file, one document per file.

    Files are separated by a blank line and written in a stable order so the
    output is reproducible; a tokenizer trained twice on the same clones should
    produce the same merges.
    """
    by_ext: Counter = Counter()
    bytes_by_ext: Counter = Counter()
    skipped_large = skipped_binary = 0
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
                out.write(text)
                out.write("\n\n")
                by_ext[ext] += 1
                bytes_by_ext[ext] += len(text)
                n_files += 1

    return {"n_files": n_files, "by_ext": dict(by_ext.most_common()),
            "bytes_by_ext": dict(bytes_by_ext.most_common()),
            "skipped_large": skipped_large, "skipped_binary": skipped_binary}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data_cache/code_corpus")
    ap.add_argument("--text-out", default="data_cache/code_corpus/code_all.txt")
    ap.add_argument("--max-file-bytes", type=int, default=MAX_FILE_BYTES)
    ap.add_argument("--repos", nargs="+", default=REPOS)
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
    stats = harvest(clones_dir, args.text_out, args.max_file_bytes)
    total = os.path.getsize(args.text_out)
    print(f"\n[code] {stats['n_files']:,} files -> {args.text_out} "
          f"({total/1024**2:.0f} MB)")
    print(f"[code] skipped {stats['skipped_large']} oversized, "
          f"{stats['skipped_binary']} undecodable")
    print(f"[code] {'ext':>8} {'files':>8} {'MB':>8}")
    for ext, n in list(stats["by_ext"].items())[:14]:
        print(f"[code] {ext:>8} {n:>8,} {stats['bytes_by_ext'][ext]/1024**2:>8.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
