"""Real peak-VRAM measurement: vanilla full-AdamW vs BAdam-style block-
coordinate-descent (Issue #99), same config, isolated subprocesses (reuses
the eval/vram_envelope.py methodology so an OOM on one config can't poison
the other's CUDA allocator state).

Two comparisons:
  1. A config where BOTH fit (the documented ~270M ceiling, d=1024/24L,
     +grad_checkpoint) -- a clean apples-to-apples peak-VRAM number.
  2. A larger config that OOMs under vanilla full-AdamW -- does BCD actually
     unlock training at a size the card otherwise can't hold?

Usage:
    python scripts/bench_block_coordinate_vram.py --out results/bcd_vram.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

# repo-root path bootstrap, so `python scripts/bench_block_coordinate_vram.py`
# can `import hvn` regardless of cwd (matches the pattern used elsewhere, e.g.
# tests/test_issue33.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

VOCAB = 8000


def run_probe(cfg: dict) -> dict:
    import torch
    from hvn.edge.model import EdgeHVN
    from hvn.core.block_coordinate import BlockCoordinateScheduler

    result = dict(cfg)
    if not torch.cuda.is_available():
        result["verdict"] = "no-cuda"
        return result
    device = torch.device("cuda")
    torch.manual_seed(0)

    try:
        model = EdgeHVN(
            vocab_size=VOCAB, d_model=cfg["d_model"], num_layers=cfg["num_layers"],
            n_heads=cfg.get("n_heads", 8), dropout=0.0, use_deltanet=True,
            grad_checkpoint=cfg.get("grad_checkpoint", False),
        )
        result["params"] = sum(p.numel() for p in model.parameters())
        model.to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        crit = torch.nn.CrossEntropyLoss()

        sched = None
        if cfg.get("block_coordinate", False):
            sched = BlockCoordinateScheduler(
                model, opt, steps_per_block=cfg.get("steps_per_block", 2),
            )

        B, S = cfg["batch_size"], cfg["seq_len"]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(cfg.get("steps", 6)):
            x = torch.randint(0, VOCAB, (B, S), device=device)
            y = torch.randint(0, VOCAB, (B, S), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x).reshape(-1, VOCAB), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            if sched is not None:
                sched.step()
        torch.cuda.synchronize()
        result.update({
            "verdict": "ok",
            "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
            "sec_per_step": round((time.time() - t0) / cfg.get("steps", 6), 2),
        })
    except torch.OutOfMemoryError:
        result["verdict"] = "oom"
        result["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
    except RuntimeError as e:
        result["verdict"] = f"error: {str(e)[:200]}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=str, default=None, help="(internal)")
    ap.add_argument("--out", type=str, default="results/bcd_vram.json")
    args = ap.parse_args()

    if args.probe is not None:
        cfg = json.loads(args.probe)
        print(json.dumps(run_probe(cfg)))
        return 0

    grid = [
        # 1. Both should fit -- clean side-by-side comparison at the
        #    documented ~270M ceiling config.
        {"d_model": 1024, "num_layers": 24, "batch_size": 8, "seq_len": 128,
         "block_coordinate": False, "label": "ceiling_baseline"},
        {"d_model": 1024, "num_layers": 24, "batch_size": 8, "seq_len": 128,
         "block_coordinate": True, "steps_per_block": 2, "label": "ceiling_bcd"},
        # 2. Larger -- expected to OOM under vanilla full-AdamW; does BCD
        #    unlock it, and does stacking grad_checkpoint (Issue #100) get it
        #    to a CLEAN fit under this card's true 6141 MiB envelope?
        {"d_model": 2048, "num_layers": 24, "batch_size": 4, "seq_len": 128,
         "block_coordinate": False, "label": "large_baseline"},
        {"d_model": 2048, "num_layers": 24, "batch_size": 4, "seq_len": 128,
         "block_coordinate": True, "steps_per_block": 2, "label": "large_bcd"},
        {"d_model": 2048, "num_layers": 24, "batch_size": 4, "seq_len": 128,
         "block_coordinate": True, "steps_per_block": 2, "grad_checkpoint": True,
         "label": "large_bcd_ckpt"},
        {"d_model": 2048, "num_layers": 24, "batch_size": 2, "seq_len": 64,
         "block_coordinate": True, "steps_per_block": 2, "grad_checkpoint": True,
         "label": "large_bcd_ckpt_tuned"},
    ]

    results = []
    for cfg in grid:
        print(f"probing {cfg['label']}: d={cfg['d_model']} L={cfg['num_layers']} "
              f"bcd={cfg['block_coordinate']} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--probe", json.dumps(cfg)],
            capture_output=True, text=True, timeout=300,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        line = None
        for ln in proc.stdout.strip().splitlines():
            if ln.startswith("{"):
                line = ln
        if line is None:
            print(f"  FAILED (no JSON output). stderr tail:\n{proc.stderr[-1500:]}")
            results.append({**cfg, "verdict": "subprocess-failed"})
            continue
        r = json.loads(line)
        results.append(r)
        print(f"  -> {r.get('verdict')}  "
              f"peak_alloc={r.get('peak_alloc_mb')}MB  "
              f"peak_reserved={r.get('peak_reserved_mb')}MB  "
              f"params={r.get('params')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 78)
    print(f"{'label':20s} {'verdict':10s} {'params':>12s} {'peak_alloc_MB':>14s} {'peak_reserved_MB':>17s}")
    print("-" * 78)
    for r in results:
        print(f"{r.get('label',''):20s} {r.get('verdict',''):10s} "
              f"{r.get('params','-'):>12} {r.get('peak_alloc_mb','-'):>14} "
              f"{r.get('peak_reserved_mb','-'):>17}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
