"""1B speed + memory bench for the 2026-10-17 fusion sweep.

Author: Rambarun Komaljeet

Build and step are scripts/ab_dropout_1b.py's (the --fast stack exactly as
main() passes it: triton + megakernel scan, TF32, fused ternary, ternary step
cache, fused RMSNorm; hvn-hybrid 2048x24, attn [6,18], tied embeddings,
chunked loss 512, Muon, grad-ckpt, balanced BCD + offload + bf16 parking +
prefetch), at ``--dropout 0`` (the 2026-10-17 decision). Levers under test are
toggled with ``--disable NAME`` so both arms run the same code tree.

Per run: 2 warmup steps, then ``--cycles`` full BCD cycles (13 steps each,
every block active once), timed unprofiled. Records peak reserved/allocated
VRAM and the process's PEAK working set and peak commit (host RAM) via
GetProcessMemoryInfo -- RAM is measured, not estimated (2026-10-17 val-path
RAM overflow). With ``--profile``, one more cycle under the profiler gives
CUDA kernel rows (device_type == CUDA, annotation ranges excluded, #131).

Inputs are random tokens: losses say nothing about quality, but they ARE
compared bit-for-bit across arms (``loss_hex``) -- a lever claimed
bit-identical must reproduce the baseline's per-step losses exactly.

Usage::

    python -u scripts/bench_1b_fusion_sweep.py --out results/fusion_sweep_1b/base.json --profile
"""
import argparse
import collections
import ctypes
import gc
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", required=True)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--cycles", type=int, default=2)
ap.add_argument("--profile", action="store_true")
LEVERS = ("read_gate", "sigmoid", "residual", "ternary_bf16")
ap.add_argument("--disable", action="append", default=[], choices=LEVERS + ("all",),
                help="2026-10-17 lever to turn OFF (repeatable; 'all' = pre-sweep main)")
args = ap.parse_args()
if "all" in args.disable:
    args.disable = list(LEVERS)

sys.path.insert(0, args.repo)
import pyarrow  # noqa: F401,E402  -- before CUDA (ed4777d)
import torch  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

import train_edge_conversational as tec  # noqa: E402
import hvn.optimizers as hopt  # noqa: E402
import hvn.edge.chunked_loss as hcl  # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402


class PMC(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t)]


def proc_mem():
    c = PMC()
    c.cb = ctypes.sizeof(PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = ctypes.c_void_p
    k32.K32GetProcessMemoryInfo.argtypes = [ctypes.c_void_p, ctypes.POINTER(PMC), ctypes.c_ulong]
    if not k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(c), c.cb):
        raise OSError("GetProcessMemoryInfo failed")
    g = 2 ** 30
    return {"ws": c.WorkingSetSize / g, "peak_ws": c.PeakWorkingSetSize / g,
            "commit": c.PagefileUsage / g, "peak_commit": c.PeakPagefileUsage / g}


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
    precision="bf16", lr=3e-4, vocab_size=VOCAB, offload_weights=True, dropout=0.0)
model = tec.apply_fast_stack(model, triton_scan=True, megakernel_scan=True, tf32=True,
                             compile_model=False, verbose=False, fused_ternary=True,
                             ternary_step_cache=True, fused_rmsnorm=True)
model.train()
from hvn.core.deltanet import DeltaNetBlock  # noqa: E402
from hvn.edge.ternary import TernaryLinear  # noqa: E402
for m in model.modules():
    if isinstance(m, DeltaNetBlock):
        m.use_fused_read_gate = "read_gate" not in args.disable
        m.use_fused_sigmoid = "sigmoid" not in args.disable
        m.use_fused_residual = "residual" not in args.disable
TernaryLinear.use_autocast_dtype = "ternary_bf16" not in args.disable
print(f"[cfg] disabled: {args.disable or 'none'}", flush=True)
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


tag = "+".join(args.disable) or "all-on"
out = {"config": vars(args), "steps": [], "mem_after_build": proc_mem()}
for s in range(args.warmup):
    a = sched.active_block_name
    lv, gn = one_step()
    out["steps"].append({"phase": "warm", "active": a, "loss": lv.item(),
                         "loss_hex": float(lv.item()).hex(), "grad_norm": gn.item()})
    print(f"[{tag} warm {s}] {a:<34} loss {lv.item():.6f}", flush=True)
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
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
                             "loss_hex": float(lv).hex(), "grad_norm": gn.item(),
                             "mem": proc_mem()})
        print(f"[{tag} c{c} {s:2d}] {a:<34} wall {wall:6.2f} loss {lv:.6f} gn {gn.item():.3f}",
              flush=True)
import hvn.core.fused_gates as _fg  # noqa: E402
import hvn.core.fused_read_norm as _frn  # noqa: E402
out["fast_path_calls"] = {**_fg.STATS, **_frn.STATS}
print(f"[{tag}] fast-path forward calls (whole run): {out['fast_path_calls']}", flush=True)
out["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
out["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
out["mem_end"] = proc_mem()
walls = [r["wall"] for r in out["steps"] if "wall" in r]
out["mean_s_per_step"] = sum(walls) / len(walls)
out["cycle_means"] = [sum(walls[i * NB:(i + 1) * NB]) / NB for i in range(args.cycles)]
print(f"[{tag}] mean {out['mean_s_per_step']:.3f} s/step, cycles "
      f"{[round(v, 3) for v in out['cycle_means']]}, peak reserved "
      f"{out['peak_reserved_mib']:.0f} MiB alloc {out['peak_alloc_mib']:.0f} MiB, "
      f"peak WS {out['mem_end']['peak_ws']:.2f} GiB peak commit "
      f"{out['mem_end']['peak_commit']:.2f} GiB", flush=True)
with open(args.out, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)

if args.profile:
    KT = collections.Counter()
    NCALL = collections.Counter()
    annot = 0.0
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
            NCALL[e.key] += int(e.count)
        del prof
        gc.collect()
    out["device_s_per_step"] = sum(KT.values()) / NB / 1e6
    out["top_kernels"] = [(k, v / NB / 1e6, NCALL[k] / NB) for k, v in KT.most_common(80)]
    print(f"[{tag}] profiled cycle: device {out['device_s_per_step']:.3f} s/step", flush=True)
    for k, v, n in out["top_kernels"][:60]:
        print(f"[{tag}]   {v:7.3f}s {n:8.0f}x  {k[:120]}", flush=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
print("[done]", flush=True)
