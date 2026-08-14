"""Prune EdgeHVN pretrain checkpoints to bound disk usage.

Pretrains save every 500 steps (hardcoded in train_edge_conversational.py) → a 102K-step
run produces ~200 checkpoints × ~318 MB ≈ 64 GB, which is untenable on a laptop. Most are
redundant snapshots (each .pt is just model_state_dict; the latest subsumes the rest).

Policy: keep the **latest** (highest-step, for auto-resume via find_latest_checkpoint) +
every **--keep-every**-step milestone (default 5000 → includes the 50K grounded-QA probe,
its 45K/55K neighbors, and the final ~100K trunk). Delete the rest.

Safety:
  - Never deletes the latest .pt.
  - Skips any .pt modified within --safe-seconds (default 120) so a live save mid-write is
    never touched (torch.save is not atomic; a partial file would be left alone this round).
  - Dry-run by default; prints what it would delete + GB reclaimed. Pass --apply to delete.

Checkpoints are gitignored (local disk only) — pruning has zero git/repo impact.

Examples:
  python scripts/prune_checkpoints.py --dir checkpoints/edge_chinchilla83m            # dry-run
  python scripts/prune_checkpoints.py --dir checkpoints/edge_chinchilla83m --apply     # delete
  python scripts/prune_checkpoints.py --dir checkpoints/edge_chinchilla83m --keep-every 2500 --apply
"""
import argparse
import glob
import os
import re
import time


def step_of(path: str):
    m = re.search(r"_step(\d+)\.pt$", os.path.basename(path))
    return int(m.group(1)) if m else None


def gb(n: float) -> str:
    return f"{n / 1024 ** 3:.2f} GB"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="checkpoint directory to prune")
    ap.add_argument("--keep-every", type=int, default=5000,
                    help="keep .pt whose step is a multiple of this (default 5000)")
    ap.add_argument("--safe-seconds", type=int, default=120,
                    help="never delete a .pt modified within this many seconds (default 120)")
    ap.add_argument("--apply", action="store_true", help="actually delete (default: dry-run)")
    args = ap.parse_args()

    pts = glob.glob(os.path.join(args.dir, "*_step*.pt"))
    items = [(p, step_of(p)) for p in pts]
    items = [(p, s) for p, s in items if s is not None]
    if not items:
        print(f"no *_step*.pt checkpoints found in {args.dir}")
        return
    items.sort(key=lambda x: x[1])

    latest_step = items[-1][1]
    now = time.time()
    keep, delete = [], []
    for p, s in items:
        size = os.path.getsize(p)
        age = now - os.path.getmtime(p)
        if s == latest_step:
            keep.append((p, s, size, "latest"))
        elif s % args.keep_every == 0:
            keep.append((p, s, size, f"milestone (÷{args.keep_every})"))
        elif age <= args.safe_seconds:
            keep.append((p, s, size, "recent (within safe window)"))
        else:
            delete.append((p, s, size))

    keep_bytes = sum(s for _, _, s, _ in keep)
    del_bytes = sum(s for _, _, s in delete)

    print(f"checkpoint dir: {args.dir}")
    print(f"keep-every={args.keep_every}  safe-seconds={args.safe_seconds}  "
          f"mode={'APPLY (delete)' if args.apply else 'DRY-RUN'}")
    print(f"found {len(items)} checkpoints ({gb(sum(sz for _, _, sz, _ in keep) + del_bytes)})")
    print(f"\nKEEP {len(keep)}  ({gb(keep_bytes)}):")
    for p, s, sz, why in keep:
        print(f"  + step {s:>7}  {gb(sz):>9}  {why}  {os.path.basename(p)}")
    print(f"\nDELETE {len(delete)}  ({gb(del_bytes)}):")
    for p, s, sz in delete:
        print(f"  - step {s:>7}  {gb(sz):>9}  {os.path.basename(p)}")

    if not delete:
        print("\nnothing to delete.")
        return

    print(f"\nwould reclaim {gb(del_bytes)} "
          f"({len(delete)} of {len(items)} files).")
    if not args.apply:
        print("dry-run — re-run with --apply to delete.")
        return

    for p, s, sz in delete:
        try:
            os.remove(p)
        except OSError as e:
            print(f"  ERROR deleting {os.path.basename(p)}: {e}")
    print(f"\nDeleted {len(delete)} files, reclaimed {gb(del_bytes)}.")
    remaining = glob.glob(os.path.join(args.dir, "*_step*.pt"))
    print(f"Remaining checkpoints: {len(remaining)}.")


if __name__ == "__main__":
    main()