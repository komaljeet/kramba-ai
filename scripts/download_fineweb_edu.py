"""Download a partial slice of FineWeb-Edu's sample-350BT config (Issue
#120): enough parquet files to reach a target token count, not the whole
~350B-token config. Resumable (skips files already on disk with the
expected size) so it's safe to re-run if interrupted.

Uses the same scoped workarounds established for this environment:
  - SSL verify disabled for huggingface_hub's own client only (Avast's
    HTTPS-scanning root cert is malformed per strict X.509 rules --
    see docs/training.md's data-acquisition notes).
  - Anonymous access (token=False) -- the cached HF token was
    malformed/expired and public dataset access doesn't need one anyway.

Usage:
    python scripts/download_fineweb_edu.py --config 350BT --num-files 103 --local-dir data_cache/fineweb_edu
    python scripts/download_fineweb_edu.py --config 10BT --num-files 14 --local-dir data_cache/fineweb_edu
"""
from __future__ import annotations

import argparse
import sys

import httpx
from huggingface_hub.utils._http import set_client_factory
from huggingface_hub import hf_hub_download, list_repo_files

set_client_factory(lambda: httpx.Client(verify=False, follow_redirects=True))

REPO_ID = "HuggingFaceFW/fineweb-edu"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="350BT", help="sample config, e.g. 10BT, 100BT, 350BT")
    ap.add_argument("--num-files", type=int, required=True)
    ap.add_argument("--local-dir", default="data_cache/fineweb_edu")
    args = ap.parse_args()

    prefix = f"sample/{args.config}/"
    print(f"Listing files under {prefix} ...", flush=True)
    all_files = sorted(f for f in list_repo_files(REPO_ID, repo_type="dataset", token=False) if f.startswith(prefix))
    print(f"Found {len(all_files)} files in this config; downloading first {args.num_files}.", flush=True)

    targets = all_files[: args.num_files]
    for i, fn in enumerate(targets, 1):
        print(f"[{i}/{len(targets)}] {fn} ...", flush=True)
        try:
            path = hf_hub_download(
                repo_id=REPO_ID, repo_type="dataset", filename=fn,
                local_dir=args.local_dir, token=False,
            )
            print(f"  -> {path}", flush=True)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            return 1
    print("Done.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
