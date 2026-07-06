"""
GPU training-envelope probe (Issues #69/#70)
============================================

Measures the largest HVN/DeltaNet model that actually **trains** on this GPU,
config by config, and records peak VRAM — so nobody plans a run the card
can't hold. Companion to the §1.1 "wind tunnel, not the aircraft" framing:
this quantifies the wind tunnel's walls.

Method
------
Each config runs in a **fresh subprocess** (own CUDA context): build model +
AdamW, run a few real optimizer steps on random token batches, record
``torch.cuda.max_memory_allocated/reserved``; a CUDA OOM is caught and
recorded as the config's verdict. Subprocess isolation means an OOM (or
allocator fragmentation) can't poison the next probe.

Axes (from the issue): params (d_model x layers) x batch x seq_len,
fp32 vs bf16, +/- ``--grad_checkpoint`` (#49). DeltaNet block (#39
recommended default), BPE-scale vocab.

Usage
-----
    python -m eval.vram_envelope --out results/vram_envelope_rtx4050.json
    python -m eval.vram_envelope --quick        # smaller grid (~5 min)

    # one probe (what the parent spawns internally):
    python -m eval.vram_envelope --probe '{"d_model":512,"num_layers":8,...}'

The training-state memory model this validates (bf16 + fp32 AdamW moments):
    bytes/param ~ 2 (bf16 weights) + 2 (grads) + 4+4 (moments) + 4 (master)
    ~ 16 B/param, + activations ~ O(B*S*d*L) (halved-ish by #49).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

VOCAB = 8000  # BPE-scale vocab (the #51 pipeline regime)


# ---------------------------------------------------------------------------
# Single-config probe (runs in its own subprocess)
# ---------------------------------------------------------------------------

def run_probe(cfg: dict) -> dict:
    import torch
    from hvn.core import HamiltonianVortexNetwork

    result = dict(cfg)
    if not torch.cuda.is_available():
        result["verdict"] = "no-cuda"
        return result
    device = torch.device("cuda")
    torch.manual_seed(0)

    try:
        model = HamiltonianVortexNetwork(
            vocab_size=VOCAB,
            d_model=cfg["d_model"],
            num_layers=cfg["num_layers"],
            n_heads=cfg.get("n_heads", 8),
            dropout=0.0,
            use_deltanet=True,
            grad_checkpoint=cfg.get("grad_checkpoint", False),
        )
        result["params"] = sum(p.numel() for p in model.parameters())
        model.to(device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4, fused=True)
        crit = torch.nn.CrossEntropyLoss()
        use_bf16 = cfg.get("precision", "bf16") == "bf16"

        B, S = cfg["batch_size"], cfg["seq_len"]
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        for _ in range(cfg.get("steps", 2)):
            x = torch.randint(0, VOCAB, (B, S), device=device)
            y = torch.randint(0, VOCAB, (B, S), device=device)
            with torch.amp.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16):
                loss = crit(model(x).reshape(-1, VOCAB), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
        torch.cuda.synchronize()
        result.update({
            "verdict": "ok",
            "peak_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20),
            "peak_reserved_mb": round(torch.cuda.max_memory_reserved() / 2**20),
            "sec_per_step": round((time.time() - t0) / cfg.get("steps", 2), 2),
        })
    except torch.OutOfMemoryError:
        result["verdict"] = "oom"
        result["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
    except RuntimeError as e:  # non-OOM CUDA failures stay visible
        result["verdict"] = f"error: {str(e)[:160]}"
    return result


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

def build_grid(quick: bool) -> list[dict]:
    base = {"batch_size": 8, "seq_len": 128, "precision": "bf16",
            "grad_checkpoint": False}
    sizes = [  # (d_model, layers) ladder — ~6M .. ~500M params
        (512, 8), (768, 12), (1024, 12), (1024, 24),
        (1536, 24), (2048, 24), (2048, 48),
    ]
    if quick:
        sizes = [(512, 8), (1024, 12), (1536, 24), (2048, 24)]
    grid: list[dict] = []
    for d, L in sizes:
        grid.append({**base, "d_model": d, "num_layers": L})
    # Knobs on the interesting sizes: checkpointing rescue + fp32 cost +
    # batch/seq headroom on the largest fitting config.
    for d, L in ([(1024, 24), (1536, 24), (2048, 24), (2048, 48)]
                 if not quick else [(1536, 24), (2048, 24)]):
        grid.append({**base, "d_model": d, "num_layers": L, "grad_checkpoint": True})
    if not quick:
        grid.append({**base, "d_model": 1024, "num_layers": 12, "precision": "fp32"})
        grid.append({**base, "d_model": 1024, "num_layers": 24, "precision": "fp32"})
        grid.append({**base, "d_model": 1024, "num_layers": 24, "batch_size": 16})
        grid.append({**base, "d_model": 1024, "num_layers": 24, "batch_size": 16,
                     "seq_len": 256, "grad_checkpoint": True})
    return grid


def main() -> int:
    ap = argparse.ArgumentParser(description="GPU training-envelope probe (#69/#70)")
    ap.add_argument("--probe", type=str, default=None,
                    help="(internal) run one config, print one JSON line")
    ap.add_argument("--out", type=str, default="results/vram_envelope.json")
    ap.add_argument("--quick", action="store_true", help="smaller grid")
    args = ap.parse_args()

    if args.probe:
        print("PROBE_RESULT " + json.dumps(run_probe(json.loads(args.probe))))
        return 0

    import torch
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    total_mb = (round(torch.cuda.get_device_properties(0).total_memory / 2**20)
                if torch.cuda.is_available() else 0)
    print(f"GPU: {gpu} ({total_mb} MiB)")

    rows = []
    grid = build_grid(args.quick)
    for i, cfg in enumerate(grid, 1):
        tag = (f"d={cfg['d_model']} L={cfg['num_layers']} B={cfg['batch_size']} "
               f"S={cfg['seq_len']} {cfg['precision']}"
               f"{' +ckpt' if cfg['grad_checkpoint'] else ''}")
        print(f"[{i}/{len(grid)}] {tag} ...", flush=True)
        proc = subprocess.run(
            [sys.executable, "-m", "eval.vram_envelope", "--probe", json.dumps(cfg)],
            capture_output=True, text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )
        row = dict(cfg)
        for line in (proc.stdout or "").splitlines():
            if line.startswith("PROBE_RESULT "):
                row = json.loads(line[len("PROBE_RESULT "):])
                break
        else:
            row["verdict"] = f"probe-crash: {(proc.stderr or '')[-200:]}"
        rows.append(row)
        print(f"    -> {row['verdict']}"
              + (f" | params {row.get('params', 0)/1e6:.1f}M"
                 f" | peak {row.get('peak_reserved_mb', '?')} MiB reserved"
                 f" | {row.get('sec_per_step', '?')} s/step"
                 if row.get("verdict") == "ok" else ""), flush=True)

    payload = {"gpu": gpu, "total_mb": total_mb,
               "vocab_size": VOCAB, "block": "deltanet", "rows": rows}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=1)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
