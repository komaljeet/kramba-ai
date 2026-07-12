"""Real peak-VRAM + speed measurement: CPU weight offload for BCD-inactive
blocks (Issue #101), isolated subprocesses (reuses the eval/vram_envelope.py
methodology).

Two questions:
  1. Does offload actually unlock configs that #99+#100 alone could not fit
     cleanly? (The ~1.38-1.54B boundary measured in results/bcd_ckpt_vram.json
     -- test something clearly past it, e.g. 2.05B.)
  2. What is the real speed cost? (Estimated ~50-100%+ overhead from the
     PCIe-transfer math in docs/training.md -- measure it directly instead
     of trusting the estimate.)

Usage:
    python scripts/bench_offload_vram.py --out results/offload_vram.json
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
            grad_checkpoint=True,  # required by offload
        )
        result["params"] = sum(p.numel() for p in model.parameters())

        # Issue #102 (extended finding): a blanket model.to(device) here
        # requires the WHOLE model resident on GPU at once (in fp32 --
        # EdgeHVN's default parameter dtype), transiently, during the copy
        # -- independent of anything the offload scheduler does afterward,
        # since it hasn't been constructed yet. For large+offloaded configs
        # this is now the actual bottleneck (confirmed: a 3.35B model needs
        # ~13.4 GiB just to exist on GPU once, causing a hard OOM at
        # construction, even though steady-state offloaded training only
        # needs ~1 GiB). skip_initial_to_device leaves the model CPU-
        # resident; BlockCoordinateScheduler's existing eviction logic
        # already handles a CPU-resident starting state correctly (a
        # `.to("cpu")` on an already-CPU tensor is a no-op), so no scheduler
        # changes were needed -- only skipping this one caller-side call.
        # Measured: a 3.35B config that hard-OOM'd (12.7 GiB demand) at
        # construction fits in ~1 GiB peak reserved with this flag on.
        skip_to_device = cfg.get("offload", False) and cfg.get("skip_initial_to_device", False)
        if not skip_to_device:
            model.to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=cfg.get("steps_per_block", 2),
            offload_inactive_weights=cfg.get("offload", False),
            device=device if cfg.get("offload", False) else None,
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
            sched.step()
        torch.cuda.synchronize()
        result.update({
            "verdict": "ok",
            "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
            "sec_per_step": round((time.time() - t0) / cfg.get("steps", 6), 3),
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
    ap.add_argument("--out", type=str, default="results/offload_vram.json")
    args = ap.parse_args()

    if args.probe is not None:
        cfg = json.loads(args.probe)
        print(json.dumps(run_probe(cfg)))
        return 0

    grid = [
        # 1. Speed cost at a size BOTH fit (1.04B) -- clean apples-to-apples
        #    step-time comparison, isolates offload's pure PCIe overhead.
        {"d_model": 2048, "num_layers": 24, "batch_size": 4, "seq_len": 128,
         "steps_per_block": 2, "offload": False, "label": "1.04B_bcd_ckpt_no_offload"},
        {"d_model": 2048, "num_layers": 24, "batch_size": 4, "seq_len": 128,
         "steps_per_block": 2, "offload": True, "label": "1.04B_bcd_ckpt_offload"},
        # 2. Does offload unlock what #99+#100 alone could not? (1.54B spilled
        #    at 6.62 GiB without offload; 2.05B needed 8.4-8.6 GiB.)
        {"d_model": 2048, "num_layers": 36, "batch_size": 4, "seq_len": 128,
         "steps_per_block": 2, "offload": True, "label": "1.54B_offload"},
        {"d_model": 2048, "num_layers": 48, "batch_size": 4, "seq_len": 128,
         "steps_per_block": 2, "offload": True, "label": "2.05B_offload"},
        {"d_model": 2048, "num_layers": 72, "batch_size": 4, "seq_len": 128,
         "steps_per_block": 2, "offload": True, "label": "3.07B_offload"},
    ]

    results = []
    for cfg in grid:
        print(f"probing {cfg['label']}: d={cfg['d_model']} L={cfg['num_layers']} "
              f"offload={cfg['offload']} ...", flush=True)
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
        print(f"  -> {r.get('verdict')}  "
              f"peak_alloc={r.get('peak_alloc_mb')}MB  "
              f"peak_reserved={r.get('peak_reserved_mb')}MB  "
              f"params={r.get('params')}  sec_per_step={r.get('sec_per_step')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 92)
    print(f"{'label':26s} {'verdict':10s} {'params':>12s} {'peak_reserved_MB':>17s} {'sec/step':>10s}")
    print("-" * 92)
    for r in results:
        print(f"{r.get('label',''):26s} {r.get('verdict',''):10s} "
              f"{r.get('params','-'):>12} {r.get('peak_reserved_mb','-'):>17} "
              f"{r.get('sec_per_step','-'):>10}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
