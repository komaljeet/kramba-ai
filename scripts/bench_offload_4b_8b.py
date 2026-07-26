"""Real step-time + VRAM measurement at 4B and 8B params, using every
offload lever landed so far: CPU weight offload + LRU cache (#101/#102/
#103), the #117 correctness fix, bf16 parking (#113), and prefetch
overlap (#115, on by default). Also requires `skip_initial_to_device`
(#102's extended finding) -- a blanket `model.to(device)` needs the WHOLE
model transiently resident in fp32 on GPU, which alone exceeds this
card's 6 GiB well before 4B params.

Usage:
    python scripts/bench_offload_4b_8b.py --out results/offload_4b_8b.json
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
        # skip_initial_to_device (#102 extended finding): construct
        # CPU-resident, let BlockCoordinateScheduler's own eviction/
        # activation logic handle ALL placement from the start -- a
        # blanket model.to(device) here would need ~4x(params) bytes
        # transiently resident in fp32, hard-OOMing well before 4B.
        model = EdgeHVN(
            vocab_size=VOCAB, d_model=cfg["d_model"], num_layers=cfg["num_layers"],
            n_heads=cfg.get("n_heads", 8), dropout=0.0, use_deltanet=True,
            grad_checkpoint=True,
        )
        result["params"] = sum(p.numel() for p in model.parameters())
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)

        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=cfg.get("steps_per_block", 2),
            offload_inactive_weights=True, device=device,
            offload_park_dtype=torch.bfloat16,
            offload_prefetch=True,
        )

        crit = torch.nn.CrossEntropyLoss()
        B, S = cfg["batch_size"], cfg["seq_len"]
        for _ in range(cfg.get("warmup_steps", 2)):
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
        result["verdict"] = f"error: {str(e)[:300]}"
    return result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=str, default=None, help="(internal)")
    ap.add_argument("--out", type=str, default="results/offload_4b_8b.json")
    args = ap.parse_args()

    if args.probe is not None:
        cfg = json.loads(args.probe)
        print(json.dumps(run_probe(cfg)))
        return 0

    grid = [
        {"d_model": 2048, "num_layers": 95, "batch_size": 2, "seq_len": 128,
         "steps_per_block": 2, "label": "4.02B"},
        {"d_model": 2048, "num_layers": 190, "batch_size": 2, "seq_len": 128,
         "steps_per_block": 2, "label": "8.01B"},
    ]

    results = []
    for cfg in grid:
        print(f"probing {cfg['label']}: d={cfg['d_model']} L={cfg['num_layers']} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, __file__, "--probe", json.dumps(cfg)],
            capture_output=True, text=True, timeout=600,
            cwd=_REPO_ROOT,
        )
        line = None
        for ln in proc.stdout.strip().splitlines():
            if ln.startswith("{"):
                line = ln
        if line is None:
            print(f"  FAILED (no JSON output). stderr tail:\n{proc.stderr[-2000:]}")
            results.append({**cfg, "verdict": "subprocess-failed"})
            continue
        r = json.loads(line)
        results.append(r)
        print(f"  -> {r.get('verdict')}  params={r.get('params')}  "
              f"peak_alloc={r.get('peak_alloc_mb')}MB  "
              f"peak_reserved={r.get('peak_reserved_mb')}MB  "
              f"sec_per_step={r.get('sec_per_step')}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nwrote {args.out}")

    print("\n" + "=" * 92)
    print(f"{'label':10s} {'verdict':10s} {'params':>12s} {'peak_reserved_MB':>17s} {'sec/step':>10s}")
    print("-" * 92)
    for r in results:
        print(f"{r.get('label',''):10s} {r.get('verdict',''):10s} "
              f"{r.get('params','-'):>12} {r.get('peak_reserved_mb','-'):>17} "
              f"{r.get('sec_per_step','-'):>10}")
    print("=" * 92)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
