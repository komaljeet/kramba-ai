"""Lean 1B A/B: offload prefetch overlap + pre-cast padded loss head (2026-10-12).

One arm per process. Builds the real 2048x24 hvn-hybrid model through the
trainer's own constructors (build_model_and_config + apply_fast_stack +
build_optimizer + BlockCoordinateScheduler) with the exact arguments of the
pretrain-sizing 2048x24 offload rung. No DataLoader and no pre-training audit:
synthetic token ids of the real shape. Full-trainer runs were hitting the
low-RAM reaper today; this keeps the process to the model itself.

Measures one FULL BCD cycle (balanced partition, steps_per_block=1 -> every
block active once) after `--warmup` steps, because step time depends heavily
on which block is active (12-30 s/step measured today). Reports per step:
wall, forward GPU time, loss (hex, for bit-level comparison), VRAM.

`--repo` selects the checkout to import (main vs the branch), so the baseline
is the real old code rather than a toggle that imitates it.

Usage:
  python scripts/ab_offload_losshead_1b.py --repo <checkout> --out arm.json
      [--no-prefetch] [--no-fast-head]
"""
import argparse
import ctypes
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--no-prefetch", action="store_true")
ap.add_argument("--no-fast-head", action="store_true",
                help="branch only: force the per-slice loss head")
ap.add_argument("--min-free-ram-gib", type=float, default=1.5)
ap.add_argument("--blanket-wait", action="store_true",
                help="branch only: restore the old whole-stream prefetch wait (lever 1 off)")
ap.add_argument("--chunk", type=int, default=512,
                help="loss chunk; 256 on main = control for accepted bf16 drift")
args = ap.parse_args()

sys.path.insert(0, args.repo)
import pyarrow  # noqa: F401,E402  -- before CUDA (ed4777d heap-corruption fix)
import torch  # noqa: E402


def free_ram_gib() -> float:
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS()
    m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


out = {"repo": args.repo, "prefetch": not args.no_prefetch,
       "fast_head": not args.no_fast_head, "chunk": args.chunk, "blanket_wait": args.blanket_wait, "steps": [], "aborted": None}


def dump():
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


def guard(where):
    r = free_ram_gib()
    if r < args.min_free_ram_gib:
        out["aborted"] = f"free RAM {r:.2f} GiB < {args.min_free_ram_gib} at {where}"
        dump()
        print("[abort]", out["aborted"], flush=True)
        sys.exit(3)
    return r


import train_edge_conversational as tec  # noqa: E402
import hvn.optimizers as hopt  # noqa: E402
import hvn.edge.chunked_loss as hcl  # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler, GPUBufferPool  # noqa: E402

if args.blanket_wait:
    # The pre-2026-10-12 ordering: every prefetch waits for the whole main
    # stream. Changes only WHEN copies run, so values must equal the default.
    GPUBufferPool._order_prefetch_after = lambda self, ready:         self._prefetch_stream.wait_stream(torch.cuda.current_stream(self.device))

GA, SEQ, VOCAB, CHUNK = 8, 4096, 50257, args.chunk
dev = torch.device("cuda")
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
guard("start")
t0 = time.time()
settings = {"batch_size": 1, "grad_accum": GA, "seq_len": SEQ, "d_model": 2048,
            "num_layers": 24, "n_heads": 64}
model, _ = tec.build_model_and_config(
    settings, None, dev, checkpoint_dir=os.path.splitext(args.out)[0] + "_ckpt",
    arch="hvn-hybrid", apply_ternary=True, attn_layers=[6, 18], attn_window=4096,
    grad_checkpoint=True, tie_embeddings=True, chunked_loss=CHUNK, optimizer="muon",
    precision="bf16", lr=3e-4, vocab_size=VOCAB, offload_weights=True)
model = tec.apply_fast_stack(model, triton_scan=True, megakernel_scan=True, tf32=True,
                             compile_model=False, verbose=False, fused_ternary=True)
model.train()
opt, desc = hopt.build_optimizer(model, "muon", lr=3e-4, weight_decay=0.01,
                                 fused=False, muon_lr=0.02)
sched = BlockCoordinateScheduler(
    model, opt, steps_per_block=1, offload_inactive_weights=True, device=dev,
    offload_park_dtype=torch.bfloat16, offload_prefetch=not args.no_prefetch,
    offload_cache_blocks=1, partition="balanced")
out["params"] = sum(p.numel() for p in model.parameters())
out["bcd_blocks"] = sched.num_blocks
out["build_s"] = time.time() - t0
out["card_bytes"] = torch.cuda.get_device_properties(0).total_memory
print(f"[build] {out['params']:,} params, {sched.num_blocks} BCD blocks, "
      f"{out['build_s']:.0f}s, ram {free_ram_gib():.1f}G", flush=True)
guard("after build")

loss_kw = {"fast_head": False} if args.no_fast_head else {}
g = torch.Generator().manual_seed(0)
XS = torch.randint(0, VOCAB, (GA, 1, SEQ), generator=g)
YS = torch.randint(0, VOCAB, (GA, 1, SEQ), generator=g)

out["build_peak_reserved"] = torch.cuda.max_memory_reserved()
n_steps = args.warmup + sched.num_blocks
for s in range(n_steps):
    torch.cuda.reset_peak_memory_stats()     # per-step peaks: which block is the peak
    active = sched.active_block_name
    loss_acc = torch.zeros((), device=dev)
    fwd_gpu = 0.0
    min_free_dev = None
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    for mb in range(GA):
        x = XS[mb].to(dev, non_blocking=True)
        y = YS[mb].to(dev, non_blocking=True)
        e0 = torch.cuda.Event(enable_timing=True)
        e1 = torch.cuda.Event(enable_timing=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            e0.record()
            hidden = model(x, return_hidden=True)
            e1.record()
            loss = hcl.chunked_cross_entropy(hidden, y, model.decoder, chunk=CHUNK,
                                             **loss_kw).mean() / GA
        loss.backward()
        loss_acc += loss.detach()
        e1.synchronize()
        fwd_gpu += e0.elapsed_time(e1) / 1e3
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    opt.zero_grad()
    sched.step()
    torch.cuda.synchronize()
    wall = time.perf_counter() - w0
    free_dev, _ = torch.cuda.mem_get_info()
    r = {"step": s, "warmup": s < args.warmup, "active": active, "wall": wall,
         "fwd_gpu": fwd_gpu, "loss": loss_acc.item(), "loss_hex": loss_acc.float().item().hex(),
         "peak_alloc": torch.cuda.max_memory_allocated(),
         "peak_reserved": torch.cuda.max_memory_reserved(),
         "driver_free_after": free_dev, "ram_free_gib": guard(f"step {s}")}
    out["steps"].append(r)
    dump()
    print("[%s %2d] %-34s wall %6.2f fwd %5.2f loss %.7f (%s) peak-rsv %.2fG (%.1f%%) ram %.1fG" % (
        "warm" if r["warmup"] else "step", s, active, wall, fwd_gpu, r["loss"], r["loss_hex"],
        r["peak_reserved"] / 2 ** 30, 100 * r["peak_reserved"] / out["card_bytes"],
        r["ram_free_gib"]), flush=True)
print("[done]", flush=True)
