"""Real peak-VRAM measurement: vanilla AdamW vs GaLore (Issue #84), and the
full stack (GaLore + block-coordinate-descent #99 + grad_checkpoint #100),
isolated subprocesses (reuses the eval/vram_envelope.py / bench_block_
coordinate_vram.py methodology).

Three comparisons:
  1. GaLore alone vs vanilla AdamW at a size both fit -- clean apples-to-
     apples optimizer-state reduction.
  2. The largest config #99+#100 alone reached cleanly (~1.04B, 4.6 GiB) --
     does adding GaLore on top shrink it further, or is #99+#100 already
     bottlenecked elsewhere (the "2M" bf16-weight floor, which GaLore does
     NOT touch -- it only compresses optimizer state, same as #99)?
  3. Push past 1.04B -- how far does the full 3-lever stack actually reach
     before hitting the 2M floor?

Usage:
    python scripts/bench_galore_vram.py --out results/galore_vram.json
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
    from hvn.optimizers import build_optimizer

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

        opt_name = cfg.get("optimizer", "adamw")
        opt, desc = build_optimizer(
            model, opt_name, lr=1e-4, weight_decay=0.0,
            fused=(opt_name == "adamw"),
            galore_rank=cfg.get("galore_rank", 128),
            galore_update_gap=cfg.get("galore_update_gap", 200),
        )
        result["optimizer_desc"] = desc

        sched = None
        if cfg.get("block_coordinate", False):
            sched = BlockCoordinateScheduler(
                model, opt, steps_per_block=cfg.get("steps_per_block", 2),
            )

        crit = torch.nn.CrossEntropyLoss()
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
    ap.add_argument("--out", type=str, default="results/galore_vram.json")
    args = ap.parse_args()

    if args.probe is not None:
        cfg = json.loads(args.probe)
        print(json.dumps(run_probe(cfg)))
        return 0

    grid = [
        # 1. GaLore alone vs vanilla AdamW, both fit -- clean comparison,
        #    same 268M ceiling config used for #99's own baseline.
        {"d_model": 1024, "num_layers": 24, "batch_size": 8, "seq_len": 128,
         "optimizer": "adamw", "label": "268M_adamw_baseline"},
        {"d_model": 1024, "num_layers": 24, "batch_size": 8, "seq_len": 128,
         "optimizer": "galore", "galore_rank": 128, "label": "268M_galore_alone"},
        # 2. The exact #99+#100 config that reached a clean 4.6 GiB fit --
        #    does GaLore on top shrink it further?
        {"d_model": 2048, "num_layers": 24, "batch_size": 4, "seq_len": 128,
         "optimizer": "adamw", "block_coordinate": True, "steps_per_block": 2,
         "grad_checkpoint": True, "label": "1.04B_bcd_ckpt_adamw"},
        {"d_model": 2048, "num_layers": 24, "batch_size": 4, "seq_len": 128,
         "optimizer": "galore", "galore_rank": 128, "block_coordinate": True,
         "steps_per_block": 2, "grad_checkpoint": True,
         "label": "1.04B_bcd_ckpt_galore"},
        # 3. Push past 1.04B with the full 3-lever stack -- how far?
        {"d_model": 3072, "num_layers": 32, "batch_size": 2, "seq_len": 64,
         "optimizer": "galore", "galore_rank": 128, "block_coordinate": True,
         "steps_per_block": 2, "grad_checkpoint": True, "label": "push_further"},
    ]

    results = []
    for cfg in grid:
        print(f"probing {cfg['label']}: d={cfg['d_model']} L={cfg['num_layers']} "
              f"opt={cfg.get('optimizer')} bcd={cfg.get('block_coordinate', False)} "
              f"ckpt={cfg.get('grad_checkpoint', False)} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--probe", json.dumps(cfg)],
            capture_output=True, text=True, timeout=300,
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
        print(f"  -> {r.get('verdict')}  "
              f"peak_alloc={r.get('peak_alloc_mb')}MB  "
              f"peak_reserved={r.get('peak_reserved_mb')}MB  "
              f"params={r.get('params')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 78)
    print(f"{'label':22s} {'verdict':10s} {'params':>12s} {'peak_alloc_MB':>14s} {'peak_reserved_MB':>17s}")
    print("-" * 78)
    for r in results:
        print(f"{r.get('label',''):22s} {r.get('verdict',''):10s} "
              f"{r.get('params','-'):>12} {r.get('peak_alloc_mb','-'):>14} "
              f"{r.get('peak_reserved_mb','-'):>17}")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
