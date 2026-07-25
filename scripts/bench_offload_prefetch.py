"""Real speed measurement: prefetch/double-buffer overlap for CPU-offloaded
BCD-inactive blocks (Issue #115) -- A/B against offload_prefetch=False (the
fully serial, pre-#115 transfer behavior), at the same configs used for
Issue #113's bf16-parking measurement.

Usage:
    python scripts/bench_offload_prefetch.py --out results/offload_prefetch.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

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
            grad_checkpoint=True,
        )
        result["params"] = sum(p.numel() for p in model.parameters())
        model.to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=cfg.get("steps_per_block", 2),
            offload_inactive_weights=True, device=device,
            offload_prefetch=cfg.get("prefetch", True),
        )

        crit = torch.nn.CrossEntropyLoss()
        B, S = cfg["batch_size"], cfg["seq_len"]
        # A few warmup steps so any one-time pinning cost (Issue #115: CPU
        # home tensors are pinned lazily, on first prefetch of a given
        # resting tensor) doesn't pollute the timed measurement below.
        for _ in range(cfg.get("warmup_steps", 3)):
            x = torch.randint(0, VOCAB, (B, S), device=device)
            y = torch.randint(0, VOCAB, (B, S), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x).reshape(-1, VOCAB), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()

        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(cfg.get("steps", 8)):
            x = torch.randint(0, VOCAB, (B, S), device=device)
            y = torch.randint(0, VOCAB, (B, S), device=device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(model(x).reshape(-1, VOCAB), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()
        torch.cuda.synchronize()
        result.update({
            "verdict": "ok",
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
            "sec_per_step": round((time.time() - t0) / cfg.get("steps", 8), 4),
        })
    except torch.OutOfMemoryError:
        result["verdict"] = "oom"
        result["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
    except RuntimeError as e:
        result["verdict"] = f"error: {str(e)[:300]}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=str, default=None, help="(internal)")
    ap.add_argument("--out", type=str, default="results/offload_prefetch.json")
    args = ap.parse_args()

    if args.probe is not None:
        cfg = json.loads(args.probe)
        print(json.dumps(run_probe(cfg)))
        return 0

    grid = []
    for d_model, num_layers, label in [
        (1024, 24, "268M"),
        (2048, 24, "1.04B"),
        (2048, 36, "1.54B"),
    ]:
        for prefetch in (False, True):
            grid.append({
                "d_model": d_model, "num_layers": num_layers,
                "batch_size": 4, "seq_len": 128, "steps_per_block": 2,
                "prefetch": prefetch,
                "label": f"{label}_{'prefetch' if prefetch else 'serial'}",
            })

    results = []
    for cfg in grid:
        print(f"probing {cfg['label']}: d={cfg['d_model']} L={cfg['num_layers']} "
              f"prefetch={cfg['prefetch']} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--probe", json.dumps(cfg)],
            capture_output=True, text=True, timeout=420,
            cwd=_REPO_ROOT,
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
        print(f"  -> {r.get('verdict')}  peak_reserved={r.get('peak_reserved_mb')}MB  "
              f"sec_per_step={r.get('sec_per_step')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 80)
    print(f"{'label':22s} {'verdict':8s} {'peak_reserved_MB':>17s} {'sec/step':>10s}")
    print("-" * 80)
    for r in results:
        print(f"{r.get('label',''):22s} {r.get('verdict',''):8s} "
              f"{r.get('peak_reserved_mb','-'):>17} {r.get('sec_per_step','-'):>10}")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
