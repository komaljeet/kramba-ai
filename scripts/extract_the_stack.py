"""Turn downloaded Stack shards into a filtered training corpus (#136).

Written by: Rambarun Komaljeet

Two jobs the raw parquet cannot do for us.

**1. Filtering, because "it is code" is not enough.** A line-level autocomplete
model is graded on finishing the line a human is typing. Minified JavaScript,
generated bindings, vendored blobs and base64 payloads are all technically code
and all actively harmful here: a single minified file can be one 40,000-character
line, and a model trained on those learns that lines do not end.

The schema hands us the discriminators precomputed -- ``avg_line_length``,
``max_line_length`` and ``alphanum_fraction`` -- so this is cheap and does not
need heuristics over the text.

**2. Licence verification, because the subset name is not evidence.** The Stack's
permissive filter is applied by *repository* licence, which is not the same claim
as every file in that repository being permissively licensed. ``max_stars_repo_licenses``
is per file, so it can be checked rather than trusted. Anything that is not on the
allow-list is dropped and counted, and the count is reported -- if it is ever
non-trivial, that is a finding about the dataset, not a rounding error.

Output is one document per file separated by a sentinel, matching what
`collect_code_corpus.py` produced, so the existing tokenizer/training path reads
it unchanged.

Usage
-----
    python scripts/extract_the_stack.py --plan results/codecomplete/corpus_plan.json
    python scripts/extract_the_stack.py --dry-run       # filter stats, no writing
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import time

MANIFEST = "data_cache/the_stack_manifest"

# SPDX identifiers accepted. Deliberately short: if a licence is not obviously
# permissive, it is dropped rather than argued about. A model trained on this
# corpus is a stronger use than a tokenizer, so the bar is set here, not later.
PERMISSIVE = {
    "mit", "apache-2.0", "bsd-3-clause", "bsd-2-clause", "isc",
    "unlicense", "cc0-1.0", "0bsd", "zlib", "postgresql", "python-2.0",
    "bsl-1.0", "mit-0", "apache-2.0 with llvm-exception",
}


class Filters:
    """Thresholds, with the reason each exists. Tuned to reject the file types
    that break a line-completion model, not to maximise corpus size."""
    MIN_BYTES = 64             # a stub or a licence header, no syntax to learn
    MAX_BYTES = 1_000_000      # vendored blob or generated table
    MIN_ALPHANUM = 0.25        # base64/binary payloads
    MAX_AVG_LINE = 100.0       # generated code; humans do not average 100 chars
    MAX_LINE = 1_000           # one long line = minified


def keep(row: dict) -> tuple[bool, str]:
    """Return (keep?, reason-if-dropped). Reasons are counted and reported."""
    size = row.get("size") or 0
    if size < Filters.MIN_BYTES:
        return False, "too_small"
    if size > Filters.MAX_BYTES:
        return False, "too_large"
    af = row.get("alphanum_fraction")
    if af is not None and af < Filters.MIN_ALPHANUM:
        return False, "low_alphanum"
    avg = row.get("avg_line_length")
    if avg is not None and avg > Filters.MAX_AVG_LINE:
        return False, "long_avg_line"
    mx = row.get("max_line_length")
    if mx is not None and mx > Filters.MAX_LINE:
        return False, "minified"
    lic = row.get("max_stars_repo_licenses") or []
    if isinstance(lic, str):
        lic = [lic]
    if not lic:
        return False, "no_licence"
    if not any(str(x).strip().lower() in PERMISSIVE for x in lic):
        return False, "non_permissive"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="results/codecomplete/corpus_plan.json")
    ap.add_argument("--out", default="data_cache/code_corpus/stack_code.txt")
    ap.add_argument("--report", default="results/codecomplete/corpus_stats.json")
    ap.add_argument("--dry-run", action="store_true",
                    help="Count and report, write nothing.")
    ap.add_argument("--separator", default="\n<|endoftext|>\n")
    args = ap.parse_args()

    import pyarrow.parquet as pq

    with open(args.plan, encoding="utf-8") as fh:
        shards = json.load(fh)["shards"]

    present = [s for s in shards if _have(os.path.join(MANIFEST, s))]
    print(f"[extract] {len(present)}/{len(shards)} shards downloaded", flush=True)
    if not present:
        raise SystemExit("nothing downloaded yet")

    cols = ["content", "size", "lang", "ext", "alphanum_fraction",
            "avg_line_length", "max_line_length", "max_stars_repo_licenses"]

    kept = dropped = 0
    kept_bytes = 0
    reasons = collections.Counter()
    per_lang = collections.Counter()
    per_lang_bytes = collections.Counter()
    licences = collections.Counter()

    out_fh = None
    if not args.dry_run:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        out_fh = open(args.out, "w", encoding="utf-8", errors="replace")

    t0 = time.time()
    for i, rel in enumerate(present, 1):
        path = os.path.join(MANIFEST, rel)
        pf = pq.ParquetFile(path)
        for rg in range(pf.metadata.num_row_groups):
            tbl = pf.read_row_group(rg, columns=cols)
            for row in tbl.to_pylist():
                ok, why = keep(row)
                if not ok:
                    dropped += 1
                    reasons[why] += 1
                    continue
                kept += 1
                lang = row.get("lang") or "unknown"
                per_lang[lang] += 1
                body = row["content"]
                per_lang_bytes[lang] += len(body.encode("utf-8"))
                kept_bytes += len(body.encode("utf-8"))
                for x in (row.get("max_stars_repo_licenses") or []):
                    licences[str(x).strip().lower()] += 1
                if out_fh is not None:
                    out_fh.write(body)
                    out_fh.write(args.separator)
        print(f"[extract] ({i}/{len(present)}) {rel} | kept {kept:,} "
              f"dropped {dropped:,} | {kept_bytes/1024**3:.2f} GB "
              f"[{(time.time()-t0)/60:.1f} min]", flush=True)

    if out_fh is not None:
        out_fh.close()

    total = kept + dropped
    print(f"\n[extract] ================ CORPUS ================")
    print(f"[extract] kept {kept:,} files ({100*kept/max(total,1):.1f}%), "
          f"{kept_bytes/1024**3:.2f} GB of code")
    print(f"[extract] dropped {dropped:,}:")
    for why, n in reasons.most_common():
        print(f"[extract]   {why:<16}{n:>10,}  ({100*n/max(total,1):.2f}%)")
    print(f"[extract] language mix by bytes:")
    tb = sum(per_lang_bytes.values()) or 1
    for lang, b in per_lang_bytes.most_common(20):
        print(f"[extract]   {lang:<14}{b/1024**3:>7.2f} GB  "
              f"{100*b/tb:>5.1f}%  {per_lang[lang]:>9,} files")

    # If this is ever large, it is a finding about the dataset, not a rounding
    # error -- the "permissive subset" would not be per-file permissive.
    np = reasons["non_permissive"] + reasons["no_licence"]
    print(f"[extract] licence-rejected: {np:,} "
          f"({100*np/max(total,1):.2f}% of all files)")

    os.makedirs(os.path.dirname(args.report) or ".", exist_ok=True)
    with open(args.report, "w", encoding="utf-8") as fh:
        json.dump({"shards": present, "kept": kept, "dropped": dropped,
                   "kept_bytes": kept_bytes, "drop_reasons": dict(reasons),
                   "files_per_lang": dict(per_lang),
                   "bytes_per_lang": dict(per_lang_bytes),
                   "licences": dict(licences.most_common(40)),
                   "filters": {k: v for k, v in vars(Filters).items()
                               if not k.startswith("_")}}, fh, indent=2)
    print(f"[extract] wrote {args.report}")
    if not args.dry_run:
        print(f"[extract] wrote {args.out}")
    return 0


def _have(path: str) -> bool:
    try:
        return os.path.getsize(path) > 10_000
    except OSError:
        return False


if __name__ == "__main__":
    raise SystemExit(main())
