"""Where do the 4.02 s of a 1B training step actually go? (2026-08-27)

Optimising before profiling is how this repo got #158's "4.25x regression" that
turned out to be a missing kernel flag. So: measure the decomposition first.

Splits a real step at the 1B config (d=2048, 24L, seq 4096, batch 1,
BCD+offload, grad-checkpoint, chunked loss) into:

  forward / backward / optimizer / block-switch

and separately reports **GPU busy fraction**, because that single number picks
the class of fix:

  * GPU ~100% busy  -> compute-bound. Only fewer FLOPs or a better kernel helps.
  * GPU well under  -> stalled on something. With CPU offload the first suspect
                       is PCIe transfer, and the fix is bytes moved or overlap,
                       not arithmetic.

#153 measured 97.2% GPU-busy at 83M with everything resident. Nobody has
measured it at 1B with weights streaming from CPU, which is a different regime.

Usage: python scripts/profile_1b_step.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN                       # noqa: E402
from hvn.edge.chunked_loss import chunked_cross_entropy  # noqa: E402
from hvn.optimizers import build_optimizer               # noqa: E402
from scripts.ab_prodigy import enable_scan_flags         # noqa: E402


def plan(num_layers: int):
    p = ["deltanet"] * num_layers
    p[3] = "transformer"
    p[10] = "transformer"
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=2048)
    ap.add_argument("--n-heads", type=int, default=32)
    ap.add_argument("--num-layers", type=int, default=24)
    ap.add_argument("--seq", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--loss-chunk", type=int, default=512)
    ap.add_argument("--bcd-steps", type=int, default=50)
    ap.add_argument("--iters", type=int, default=4)
    ap.add_argument("--vocab-size", type=int, default=50257)
    ap.add_argument("--offload", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--out", default="results/profile_1b_step.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required."); return 2

    model = EdgeHVN(
        vocab_size=args.vocab_size, d_model=args.d_model,
        num_layers=args.num_layers, n_heads=args.n_heads,
        block_plan=plan(args.num_layers), attn_window=4096,
        tie_embeddings=not args.offload, grad_checkpoint=True,
    )
    n = sum(p.numel() for p in model.parameters())
    model = model.to("cpu" if args.offload else "cuda")
    enable_scan_flags(model)
    opt, desc = build_optimizer(model, "muon", lr=2e-5,
                               weight_decay=0.01, muon_lr=0.02)
    sched = None
    if args.offload:
        from hvn.core.block_coordinate import BlockCoordinateScheduler
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=args.bcd_steps, device="cuda",
            offload_inactive_weights=True, offload_park_dtype=torch.bfloat16,
            offload_prefetch=True,
        )
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"1B step profile: {n:,} params, offload={'ON' if args.offload else 'OFF'}")
    print(f"  {desc}")

    x = torch.randint(0, args.vocab_size, (args.batch, args.seq), device="cuda")
    y = torch.randint(0, args.vocab_size, (args.batch, args.seq), device="cuda")

    def sync():
        torch.cuda.synchronize()

    acc = {"fwd": [], "bwd": [], "opt": [], "total": []}
    for i in range(args.iters + 2):
        sync(); t_start = time.perf_counter()

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(x, return_hidden=True)
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
            loss = chunked_cross_entropy(hidden, y, model.decoder,
                                         chunk=args.loss_chunk).mean()
        sync(); t_fwd = time.perf_counter()

        loss.backward()
        sync(); t_bwd = time.perf_counter()

        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if sched is not None:
            sched.step()
        sync(); t_opt = time.perf_counter()

        if i >= 2:                       # discard warmup
            acc["fwd"].append(t_fwd - t_start)
            acc["bwd"].append(t_bwd - t_fwd)
            acc["opt"].append(t_opt - t_bwd)
            acc["total"].append(t_opt - t_start)

    med = {k: sorted(v)[len(v) // 2] for k, v in acc.items()}
    total = med["total"]
    print(f"\n{'phase':<12}{'seconds':>10}{'share':>9}")
    for k in ("fwd", "bwd", "opt"):
        print(f"{k:<12}{med[k]:>10.3f}{100*med[k]/total:>8.1f}%")
    print(f"{'TOTAL':<12}{total:>10.3f}{'100.0%':>9}")
    print(f"throughput: {args.batch*args.seq/total:.0f} tok/s")

    # ---- GPU busy fraction --------------------------------------------------
    # Sampled from the driver while a step runs. A low number under offload
    # points at PCIe/stalls; a high one says the arithmetic is the cost and only
    # fewer FLOPs will help.
    import subprocess, threading
    samples = []
    stop = threading.Event()

    def poll():
        while not stop.is_set():
            try:
                r = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=2)
                samples.append(int(r.stdout.strip().split("\n")[0]))
            except Exception:
                pass
            time.sleep(0.05)

    t = threading.Thread(target=poll, daemon=True); t.start()
    for _ in range(3):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model(x, return_hidden=True)
            h = h[0] if isinstance(h, tuple) else h
            l = chunked_cross_entropy(h, y, model.decoder, chunk=args.loss_chunk).mean()
        l.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        if sched is not None:
            sched.step()
        sync()
    stop.set(); t.join(timeout=1)

    busy = sum(samples) / len(samples) if samples else float("nan")
    print(f"\nGPU busy: {busy:.1f}% mean over {len(samples)} samples "
          f"(max {max(samples) if samples else 0}%)")
    if busy == busy and busy < 85:
        print("  -> STALLED. Under offload the first suspect is PCIe transfer:")
        print("     attack bytes moved or overlap, NOT arithmetic.")
    else:
        print("  -> compute-bound. Only fewer FLOPs or a better kernel helps;")
        print("     transfer/scheduling tuning has nothing to reclaim.")

    out = dict(params=n, offload=args.offload, seq=args.seq, batch=args.batch,
               phases_sec=med, tok_per_sec=args.batch * args.seq / total,
               gpu_busy_mean=busy, gpu_busy_samples=len(samples),
               peak_reserved_gb=torch.cuda.max_memory_reserved() / 2**30)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
