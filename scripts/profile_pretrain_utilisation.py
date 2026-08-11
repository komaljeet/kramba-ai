"""Is the GPU actually busy during a real pretrain? (Issue #153)

Author: Rambarun Komaljeet

Every throughput number in this repo is **tokens/sec** -- a rate, not a
utilisation. It cannot distinguish "the GPU is saturated" from "the GPU idles
40% of the time and still produces that rate". This measures the second thing.

The only idle gap ever measured is the dataloader's 0.2-0.3% (#141), and that
was a lean A/B harness with pre-materialised batches, no checkpoint saves, no
inline evals and no audit machinery. It says nothing about a real run.

**Why the number matters:** a 9-day pretrain is the plan. At 60% utilisation,
~3.6 of those days are idle GPU, and recovering them is plumbing -- no quality
risk, no A/B, no scale dependence. That is a larger prize than any of the nine
architecture ideas measured in this session, all of which returned null.

Deliberately includes a **checkpoint save** and an **eval** inside the profiled
window, because those are the suspects a steady-state loop measurement would
miss by construction.

**The profiler adds overhead**, so the utilisation reported here is a LOWER
BOUND on the un-profiled run. Stated up front so a mediocre number is not
over-read -- and a wall-clock control (identical steps, profiler off) is run
alongside so the overhead itself is quantified rather than hand-waved.

Usage::

    python scripts/profile_pretrain_utilisation.py --steps 60
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ab_prodigy import as_pair, enable_scan_flags, require_cuda  # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model                      # noqa: E402
from hvn.optimizers import build_optimizer                               # noqa: E402

GPT2_VOCAB_SIZE = 50257


def build(args, device):
    from hvn.edge.model import EdgeHVN
    plan = ["transformer" if i in (3, 10) else "deltanet" for i in range(12)]
    model = EdgeHVN(
        vocab_size=args.vocab_size, d_model=args.d_model, num_layers=12,
        n_heads=args.n_heads, block_plan=plan, attn_window=4096,
    ).to(device)
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(
        model, target_names=[n for n in linears
                             if not any(p in n for p in exclude)])
    enable_scan_flags(model)
    return model


def train_window(model, opt, batches, args, device, *, save_path=None,
                 do_eval=False, probes=None):
    """One measured window: `--steps` micro-batches, optionally a save + an eval.

    Mirrors production: bf16 autocast, grad_accum, clipping, and (unlike the A/B
    harnesses) the checkpoint write and inline eval that a real run performs.
    """
    crit = nn.CrossEntropyLoss()
    model.train()
    micro = 0
    phase_wall = defaultdict(float)
    t_start = time.perf_counter()

    for i, (bx, by) in enumerate(batches, start=1):
        t0 = time.perf_counter()
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)
        phase_wall["h2d"] += time.perf_counter() - t0

        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = crit(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss = loss / args.grad_accum
        loss.backward()
        phase_wall["fwd_bwd"] += time.perf_counter() - t0

        micro += 1
        if micro >= args.grad_accum:
            t0 = time.perf_counter()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            phase_wall["optimizer"] += time.perf_counter() - t0
            micro = 0

    if do_eval and probes:
        t0 = time.perf_counter()
        model.eval()
        with torch.no_grad():
            for bx, by in probes:
                x, y = bx.to(device), by.to(device)
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    lg = model(x)
                nn.functional.cross_entropy(
                    lg.reshape(-1, lg.size(-1)).float(), y.reshape(-1))
        model.train()
        phase_wall["inline_eval"] += time.perf_counter() - t0

    if save_path:
        t0 = time.perf_counter()
        from train_edge_conversational import _save_edge_checkpoint
        _save_edge_checkpoint(model, save_path, global_step=0)
        phase_wall["checkpoint_save"] += time.perf_counter() - t0

    torch.cuda.synchronize()
    phase_wall["_total_wall"] = time.perf_counter() - t_start
    return phase_wall


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=60, help="micro-batches profiled")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--probe-batches", type=int, default=8)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--vocab-size", type=int, default=GPT2_VOCAB_SIZE)
    p.add_argument("--fast", action=argparse.BooleanOptionalAction, default=True,
                   help="TF32 + torch.compile, matching production defaults "
                        "(#142/#143). Profiling without it would measure a "
                        "configuration nobody runs.")
    p.add_argument("--scratch", default=None)
    p.add_argument("--out", default="results/gpu_utilisation/profile.json")
    args = p.parse_args()

    device = require_cuda()
    if args.fast:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    import train_edge_conversational as tec
    tok = tec.load_tokenizer(args.tokenizer)
    if args.vocab_size != tok.vocab_size:
        args.vocab_size = tok.vocab_size
    torch.manual_seed(7)
    loader, _v, _s = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=2)
    pairs = []
    for b in loader:
        x, y = as_pair(b)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= args.probe_batches + args.warmup + args.steps:
            break
    probes = pairs[:args.probe_batches]
    warm = pairs[args.probe_batches:args.probe_batches + args.warmup]
    batches = pairs[args.probe_batches + args.warmup:]

    model = build(args, device)
    if args.fast:
        model = torch.compile(model)
    opt, desc = build_optimizer(model, "muon", lr=2e-5, weight_decay=0.01,
                                muon_lr=args.muon_lr)
    print(f"[prof] {desc}\n[prof] --fast={args.fast}, "
          f"{len(batches)} profiled micro-batches, grad_accum={args.grad_accum}")

    train_window(model, opt, warm, args, device)          # compile + allocator warm

    scratch = args.scratch or os.path.join(
        os.environ.get("TEMP", "."), "hvn_prof_ckpt.pt")

    # ---- control: identical work, profiler OFF (quantifies profiler overhead)
    ctrl = train_window(model, opt, batches, args, device,
                        save_path=scratch, do_eval=True, probes=probes)

    # ---- profiled window
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False, with_stack=False) as prof:
        prof_wall = train_window(model, opt, batches, args, device,
                                 save_path=scratch, do_eval=True, probes=probes)

    # GPU-busy = union of CUDA kernel time. key_averages() double-counts across
    # op/kernel rows (the #131 trap), so sum only device-side kernel events.
    gpu_ns = 0
    for e in prof.events():
        if getattr(e, "device_type", None) is not None and \
                str(getattr(e, "device_type", "")).endswith("CUDA") and \
                getattr(e, "self_device_time_total", 0):
            gpu_ns += e.self_device_time_total
    gpu_busy_s = gpu_ns / 1e6
    wall = prof_wall["_total_wall"]
    util = gpu_busy_s / wall * 100 if wall else float("nan")
    overhead = (prof_wall["_total_wall"] / ctrl["_total_wall"] - 1) * 100

    print(f"\n[prof] {'phase':<18}{'seconds':>10}{'% wall':>9}   (profiler OFF control)")
    for k in ("h2d", "fwd_bwd", "optimizer", "inline_eval", "checkpoint_save"):
        v = ctrl.get(k, 0.0)
        print(f"[prof] {k:<18}{v:>10.3f}{v / ctrl['_total_wall'] * 100:>8.1f}%")
    print(f"[prof] {'TOTAL WALL':<18}{ctrl['_total_wall']:>10.3f}{100.0:>8.1f}%")

    print(f"\n[prof] GPU-busy (kernel time): {gpu_busy_s:.3f}s of {wall:.3f}s "
          f"profiled wall = {util:.1f}%")
    print(f"[prof] profiler overhead: {overhead:+.1f}% wall vs control -> the "
          f"{util:.1f}% is a LOWER BOUND on the un-profiled run")

    if util > 90:
        verdict = (f"GPU is saturated ({util:.1f}%). The pipeline is not the "
                   f"bottleneck; there is no idle time to reclaim.")
    elif util > 80:
        verdict = (f"{util:.1f}% busy -- modest headroom. Worth attributing before "
                   f"acting; the phase table above says where.")
    else:
        verdict = (f"ONLY {util:.1f}% BUSY -- roughly {100-util:.0f}% of wall clock "
                   f"is idle GPU. On a 9-day pretrain that is ~{(100-util)/100*9:.1f} "
                   f"days. Fixing it is plumbing, not research.")
    print(f"\n[prof] VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 153, "config": vars(args),
                   "phases_control_s": {k: v for k, v in ctrl.items()},
                   "profiled_wall_s": wall, "gpu_busy_s": gpu_busy_s,
                   "utilisation_pct": util,
                   "profiler_overhead_pct": overhead,
                   "verdict": verdict}, fh, indent=2)
    print(f"[prof] wrote {args.out}")
    try:
        os.remove(scratch)
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
