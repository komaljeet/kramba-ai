"""Fetch a budgeted, language-mixed slice of The Stack (dedup) for #136.

Written by: Rambarun Komaljeet

The Stack (dedup) is 0.91 TB across 358 "languages" and we need ~10 GB of it, so
**which** slice matters far more than download speed. Two facts from the manifest
decide the design:

1. **Half of it is not code.** csv (126 GB), text (117 GB), jupyter-notebook
   (112 GB) and json (100 GB) are 49% of the corpus; add svg, xml and html and it
   is over 60%. A naive "take the first N shards" slice would spend most of a
   3.4B-token budget teaching the model to predict CSV rows, which is worthless
   for line-level code autocomplete.
2. **Parquet is compressed ~3.09x** against the text inside it (measured on
   data/python/data-00000-of-00144.parquet: 195.7 MB on disk, ~605 MB of
   content). So a text budget must be converted before choosing shards, or the
   download overshoots by 3x.

Why git rather than huggingface_hub: this machine's Python TLS cannot reach the
hub -- Avast's HTTPS interception presents a CA certificate that OpenSSL rejects
("Basic Constraints of CA cert not marked critical"). Git works because Git for
Windows uses schannel, the Windows trust store, which accepts it. Hence a
pointer-only clone plus `git lfs pull --include`, and hence `-c
http.sslBackend=schannel` on every git call: **`git clone` does not inherit the
enclosing repo's local config**, so the setting has to be passed explicitly.

Resumable by construction: git-lfs skips objects it already has, so re-running
after an interruption costs one round-trip per already-present shard.

Usage
-----
    python scripts/download_the_stack.py --plan          # choose, print, no I/O
    python scripts/download_the_stack.py                 # choose and fetch
    python scripts/download_the_stack.py --target-tokens 3.4e9
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time

MANIFEST = "data_cache/the_stack_manifest"
POINTER_SIZE_RE = re.compile(rb"size (\d+)")

# Measured, not assumed -- see the module docstring.
TEXT_PER_PARQUET_BYTE = 3.09

# Tokens per byte for the 32K vocabulary on code, measured in the #139 mixed
# corpus A/B (results/vocab/ab_mixed.json, code held-out: 0.3312). Using the
# code-specific figure because this corpus is code.
TOKENS_PER_BYTE = 0.3312

# Share of the token budget per language. Weighted toward real code and away from
# the data formats that dominate the corpus. A flag, not a constant, because the
# right mix is a product decision -- see --mix.
DEFAULT_MIX = {
    "python": 0.30,
    "javascript": 0.15,
    "java": 0.10,
    "typescript": 0.08,
    "c": 0.07,
    "cpp": 0.07,
    "c-sharp": 0.05,
    "go": 0.05,
    "rust": 0.04,
    "php": 0.04,
    "ruby": 0.02,
    "shell": 0.02,
    "sql": 0.01,
}


def git(*args, cwd=MANIFEST, check=True, capture=True):
    """Run git with the TLS backend that works on this machine."""
    cmd = ["git", "-c", "http.sslBackend=schannel", *args]
    return subprocess.run(cmd, cwd=cwd, check=check,
                          capture_output=capture, text=True)


def shard_sizes(lang: str) -> list[tuple[str, int]]:
    """[(repo-relative path, parquet bytes)] for one language, from LFS pointers.

    The pointer file carries the object's size, so the whole 0.91 TB layout is
    measurable without fetching a byte of it.
    """
    d = os.path.join(MANIFEST, "data", lang)
    if not os.path.isdir(d):
        return []
    out = []
    for name in sorted(os.listdir(d)):
        p = os.path.join(d, name)
        try:
            with open(p, "rb") as fh:
                head = fh.read(300)
            disk = os.path.getsize(p)
        except OSError:
            continue
        m = POINTER_SIZE_RE.search(head)
        if m:
            out.append((f"data/{lang}/{name}", int(m.group(1))))
        elif disk > 10_000:
            # ALREADY PULLED. The pointer has been replaced by the real parquet,
            # so there is no "size NNN" line to parse -- fall back to the file on
            # disk. Without this branch a downloaded shard vanishes from the
            # catalogue entirely, and a resumed run does not skip it: it drops it
            # from the plan and selects a DIFFERENT shard instead, growing the
            # download every time. That is the opposite of resumable, and it is
            # invisible unless you notice the plan changed between runs.
            out.append((f"data/{lang}/{name}", disk))
    return out


def plan(mix: dict, target_tokens: float, tol: float = 0.02,
         max_overshoot: float = 2.0) -> tuple[list, dict]:
    """Choose whole shards per language to approximate the requested token mix.

    Shards are the unit git-lfs can fetch, so the mix is approximate -- a
    language needing less than one shard still gets one. The realised mix is
    reported rather than the requested one, because that is what training sees.
    """
    target_bytes = target_tokens / TOKENS_PER_BYTE
    target_parquet = target_bytes / TEXT_PER_PARQUET_BYTE

    selected, report = [], {}
    for lang, share in sorted(mix.items(), key=lambda kv: -kv[1]):
        shards = shard_sizes(lang)
        if not shards:
            report[lang] = {"error": "no shards in manifest"}
            continue
        want = target_parquet * share
        taken, got = [], 0
        for path, size in shards:
            if got >= want:
                break
            taken.append((path, size))
            got += size
        if not taken:                      # always take at least one
            taken, got = [shards[0]], shards[0][1]
        selected.extend(taken)
        report[lang] = {
            "shards": len(taken), "of": len(shards),
            "parquet_bytes": got,
            "est_text_bytes": got * TEXT_PER_PARQUET_BYTE,
            "est_tokens": got * TEXT_PER_PARQUET_BYTE * TOKENS_PER_BYTE,
            "requested_share": share,
            "_all": shards,
        }

    # Top-up. A shard is indivisible and every language takes at least one, so
    # small languages overshoot their share and large ones end up starved --
    # python landed at 22% against a requested 30% before this pass. Repeatedly
    # give one more shard to whichever language is furthest BELOW its request,
    # until nothing is more than `tol` under or the cap is hit.
    #
    # Overshooting the token target is deliberate and cheap: more unique data
    # than the budget means the run stays under one epoch, and multi-pass
    # training on a small corpus is the regime that got #132's verdict retracted.
    cap = target_parquet * max_overshoot
    for _ in range(400):
        total_tok = sum(r.get("est_tokens", 0) for r in report.values()) or 1
        deficits = []
        for lang, r in report.items():
            if "_all" not in r or r["shards"] >= len(r["_all"]):
                continue
            realised = r["est_tokens"] / total_tok
            deficits.append((r["requested_share"] - realised, lang))
        if not deficits:
            break
        worst, lang = max(deficits)
        if worst <= tol:
            break
        r = report[lang]
        if sum(s for _, s in selected) + r["_all"][r["shards"]][1] > cap:
            break
        path, size = r["_all"][r["shards"]]
        selected.append((path, size))
        r["shards"] += 1
        r["parquet_bytes"] += size
        r["est_text_bytes"] = r["parquet_bytes"] * TEXT_PER_PARQUET_BYTE
        r["est_tokens"] = r["est_text_bytes"] * TOKENS_PER_BYTE

    total_tok = sum(r.get("est_tokens", 0) for r in report.values())
    for lang, r in report.items():
        r.pop("_all", None)
        if "est_tokens" in r:
            r["realised_share"] = r["est_tokens"] / total_tok if total_tok else 0
    return selected, report


def print_report(report: dict, selected: list) -> None:
    """#136 acceptance criterion 1: report the distribution BEFORE training."""
    total_pq = sum(s for _, s in selected)
    total_tok = sum(r.get("est_tokens", 0) for r in report.values())
    print(f"\n[stack] {'language':<14}{'shards':>8}{'parquet GB':>12}"
          f"{'est tokens':>14}{'want':>7}{'get':>7}")
    for lang, r in sorted(report.items(),
                          key=lambda kv: -kv[1].get("est_tokens", 0)):
        if "error" in r:
            print(f"[stack] {lang:<14}  {r['error']}")
            continue
        print(f"[stack] {lang:<14}{r['shards']:>4}/{r['of']:<3}"
              f"{r['parquet_bytes']/1024**3:>12.2f}"
              f"{r['est_tokens']/1e6:>13.0f}M"
              f"{100*r['requested_share']:>6.1f}%"
              f"{100*r['realised_share']:>6.1f}%")
    print(f"[stack] {'TOTAL':<14}{len(selected):>8}"
          f"{total_pq/1024**3:>12.2f}{total_tok/1e9:>13.2f}B")
    print(f"[stack] est text {total_pq*TEXT_PER_PARQUET_BYTE/1024**3:.1f} GB "
          f"(parquet x{TEXT_PER_PARQUET_BYTE} measured)")


def fetch(selected: list, args) -> int:
    """Pull the chosen shards. git-lfs skips what is already present."""
    todo = [(p, s) for p, s in selected
            if not _already_have(os.path.join(MANIFEST, p))]
    have = len(selected) - len(todo)
    print(f"\n[stack] {have} shard(s) already present, {len(todo)} to fetch "
          f"({sum(s for _, s in todo)/1024**3:.2f} GB)", flush=True)

    t0 = time.time()
    for i, (path, size) in enumerate(todo, 1):
        el = time.time() - t0
        print(f"[stack] ({i}/{len(todo)}) {path}  {size/1024**2:.0f} MB"
              f"  [{el/60:.1f} min elapsed]", flush=True)
        try:
            git("lfs", "pull", f"--include={path}", capture=False)
        except subprocess.CalledProcessError as e:
            print(f"[stack] FAILED {path}: {e}", flush=True)
            if not args.keep_going:
                return 1
    print(f"\n[stack] done in {(time.time()-t0)/60:.1f} min", flush=True)
    return 0


def _already_have(path: str) -> bool:
    """A pulled shard is the real file; an unpulled one is a ~130-byte pointer."""
    try:
        return os.path.getsize(path) > 10_000
    except OSError:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-tokens", type=float, default=3.4e9,
                    help="Token budget. 3.4e9 is Chinchilla for ~170M params; "
                         "the #136 sizing recommends 157M -> ~3.14e9.")
    ap.add_argument("--mix", type=str, default=None,
                    help="JSON dict of language->share, overriding the default.")
    ap.add_argument("--plan", action="store_true",
                    help="Choose and report, fetch nothing.")
    ap.add_argument("--keep-going", action="store_true",
                    help="Continue past a failed shard.")
    ap.add_argument("--out", default="results/codecomplete/corpus_plan.json")
    args = ap.parse_args()

    if not os.path.isdir(os.path.join(MANIFEST, "data")):
        raise SystemExit(
            f"{MANIFEST}/data not found. Clone the manifest first:\n"
            f"  GIT_LFS_SKIP_SMUDGE=1 git -c http.sslBackend=schannel clone "
            f"--depth 1 https://huggingface.co/datasets/bigcode/the-stack-dedup "
            f"{MANIFEST}")

    mix = json.loads(args.mix) if args.mix else DEFAULT_MIX
    total = sum(mix.values())
    if abs(total - 1.0) > 0.01:
        print(f"[stack] note: mix sums to {total:.3f}, normalising")
        mix = {k: v / total for k, v in mix.items()}

    selected, report = plan(mix, args.target_tokens)
    print(f"[stack] target {args.target_tokens/1e9:.2f}B tokens "
          f"-> {args.target_tokens/TOKENS_PER_BYTE/1024**3:.1f} GB text "
          f"-> {args.target_tokens/TOKENS_PER_BYTE/TEXT_PER_PARQUET_BYTE/1024**3:.1f} GB parquet")
    print_report(report, selected)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"target_tokens": args.target_tokens, "mix": mix,
                   "tokens_per_byte": TOKENS_PER_BYTE,
                   "text_per_parquet_byte": TEXT_PER_PARQUET_BYTE,
                   "report": report,
                   "shards": [p for p, _ in selected]}, fh, indent=2)
    print(f"[stack] wrote {args.out}")

    if args.plan:
        print("[stack] --plan given, fetching nothing.")
        return 0
    return fetch(selected, args)


if __name__ == "__main__":
    raise SystemExit(main())
