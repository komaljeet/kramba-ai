"""1B speed A/B for dropout p=0.1 vs p=0 (2026-10-15).

Author: Rambarun Komaljeet

Build and step are scripts/ab_fused_rmsnorm_1b.py's -- the --fast stack exactly
as main() passes it today (triton + megakernel scan, TF32, fused ternary,
ternary step cache, fused RMSNorm; hvn-hybrid 2048x24, attn [6,18], tied
embeddings, chunked loss 512, Muon, grad-ckpt, balanced BCD + offload + bf16
parking + prefetch) -- with only ``dropout`` (passed through
build_model_and_config, the same path as the trainer's --dropout) differing.

Per run: 2 warmup steps, then ``--cycles`` full BCD cycles (13 steps each,
every block active once; only whole cycles are comparable), timed unprofiled.
With ``--profile``, one further full cycle under the profiler gives CUDA
kernel rows (#131-safe: device_type == CUDA, annotation ranges excluded), so
the wall delta can be checked against the dropout kernels it should remove.

This measures SPEED only. Inputs are random tokens: the losses it prints say
nothing about model quality.

Falsifiability: ``F.dropout`` is wrapped with a counter of calls that would do
work (training and p > 0). The p=0.1 arm must show such calls and the p=0 arm
none, and the p=0 arm's profiled cycle must show zero dropout kernels, or the
run is VOID (exit 4) rather than reporting a number.

Usage::

    python -u scripts/ab_dropout_1b.py --dropout 0.0 --profile --out results/dropout_1b/off_1.json
"""
import argparse
import collections
import gc
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", required=True)
ap.add_argument("--dropout", type=float, required=True)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--cycles", type=int, default=2)
ap.add_argument("--profile", action="store_true")
args = ap.parse_args()

sys.path.insert(0, args.repo)
import pyarrow  # noqa: F401,E402  -- before CUDA (ed4777d)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

import train_edge_conversational as tec  # noqa: E402
import hvn.optimizers as hopt  # noqa: E402
import hvn.edge.chunked_loss as hcl  # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402

CALLS = [0]
_orig = F.dropout


def _counted(input, p=0.5, training=True, inplace=False):
    if training and p > 0:
        CALLS[0] += 1
    return _orig(input, p, training, inplace)


F.dropout = _counted

GA, SEQ, VOCAB, CHUNK = 8, 4096, 50257, 512
dev = torch.device("cuda")
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
settings = {"batch_size": 1, "grad_accum": GA, "seq_len": SEQ, "d_model": 2048,
            "num_layers": 24, "n_heads": 64}
model, _ = tec.build_model_and_config(
    settings, None, dev, checkpoint_dir=os.path.splitext(args.out)[0] + "_ckpt",
    arch="hvn-hybrid", apply_ternary=True, attn_layers=[6, 18], attn_window=4096,
    grad_checkpoint=True, tie_embeddings=True, chunked_loss=CHUNK, optimizer="muon",
    precision="bf16", lr=3e-4, vocab_size=VOCAB, offload_weights=True,
    dropout=args.dropout)
model = tec.apply_fast_stack(model, triton_scan=True, megakernel_scan=True, tf32=True,
                             compile_model=False, verbose=False, fused_ternary=True,
                             ternary_step_cache=True, fused_rmsnorm=True)
model.train()
dropout_ps = collections.Counter(m.p for m in model.modules() if isinstance(m, torch.nn.Dropout))
print(f"[build] dropout module p counts {dict(dropout_ps)}", flush=True)
opt, _ = hopt.build_optimizer(model, "muon", lr=3e-4, weight_decay=0.01,
                              fused=False, muon_lr=0.02)
sched = BlockCoordinateScheduler(
    model, opt, steps_per_block=1, offload_inactive_weights=True, device=dev,
    offload_park_dtype=torch.bfloat16, offload_prefetch=True,
    offload_cache_blocks=1, partition="balanced")
NB = sched.num_blocks
g = torch.Generator().manual_seed(0)
XS = torch.randint(0, VOCAB, (GA, 1, SEQ), generator=g)
YS = torch.randint(0, VOCAB, (GA, 1, SEQ), generator=g)


def one_step():
    loss_acc = torch.zeros((), device=dev)
    for mb in range(GA):
        x = XS[mb].to(dev, non_blocking=True)
        y = YS[mb].to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(x, return_hidden=True)
            loss = hcl.chunked_cross_entropy(hidden, y, model.decoder, chunk=CHUNK).mean() / GA
        loss.backward()
        loss_acc += loss.detach()
    gn = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    opt.zero_grad()
    sched.step()
    return loss_acc, gn


tag = f"p={args.dropout}"
out = {"config": vars(args), "dropout_module_p_counts": {str(k): v for k, v in dropout_ps.items()},
       "steps": []}
for s in range(args.warmup):
    a = sched.active_block_name
    lv, gn = one_step()
    out["steps"].append({"phase": "warm", "active": a, "loss": lv.item(), "grad_norm": gn.item()})
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
calls0 = CALLS[0]
for c in range(args.cycles):
    for s in range(NB):
        a = sched.active_block_name
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        lv, gn = one_step()
        lv = lv.item()
        torch.cuda.synchronize()
        wall = time.perf_counter() - w0
        out["steps"].append({"phase": f"cycle{c}", "active": a, "wall": wall, "loss": lv,
                             "grad_norm": gn.item()})
        print(f"[{tag} c{c} {s:2d}] {a:<34} wall {wall:6.2f} loss {lv:.6f} gn {gn.item():.3f}",
              flush=True)
calls = CALLS[0] - calls0
out["active_dropout_calls_timed"] = calls
out["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
out["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
walls = [r["wall"] for r in out["steps"] if "wall" in r]
out["mean_s_per_step"] = sum(walls) / len(walls)
out["cycle_means"] = [sum(walls[i * NB:(i + 1) * NB]) / NB for i in range(args.cycles)]
print(f"[{tag}] mean {out['mean_s_per_step']:.3f} s/step, cycles "
      f"{[round(v, 3) for v in out['cycle_means']]}, active dropout calls {calls}, "
      f"peak reserved {out['peak_reserved_mib']:.0f} MiB", flush=True)

void = (args.dropout > 0) != (calls > 0)
if args.profile:
    KT = collections.Counter()
    annot = 0.0
    dev_tot = 0.0
    for s in range(NB):
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            one_step()[0].item()
            torch.cuda.synchronize()
        for e in prof.key_averages():
            d = getattr(e, "self_device_time_total", 0) or 0
            if d <= 0 or getattr(e, "device_type", None) != DeviceType.CUDA:
                continue
            if "#" in e.key or e.key.startswith("Optimizer."):
                annot += d
                continue
            KT[e.key] += d
            dev_tot += d
        del prof
        gc.collect()
    drop = {k: v / NB / 1e6 for k, v in KT.items() if "ropout" in k}
    out["device_s_per_step"] = dev_tot / NB / 1e6
    out["dropout_kernels_s_per_step"] = drop
    out["dropout_kernel_s_per_step_total"] = sum(drop.values())
    out["top_kernels"] = [(k, v / NB / 1e6) for k, v in KT.most_common(40)]
    print(f"[{tag}] profiled cycle: device {out['device_s_per_step']:.3f} s/step, dropout "
          f"kernels {out['dropout_kernel_s_per_step_total']:.3f} s/step "
          f"({len(drop)} kernel names), annotation excluded {annot/NB/1e6:.3f}", flush=True)
    for k, v in sorted(drop.items(), key=lambda kv: -kv[1]):
        print(f"[{tag}]   {v:7.3f}s  {k[:110]}", flush=True)
    if args.dropout == 0 and drop:
        void = True
with open(args.out, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
if void:
    print(f"[VOID] arm {tag}: {calls} active dropout calls / profile disagrees with the arm",
          flush=True)
    sys.exit(4)
