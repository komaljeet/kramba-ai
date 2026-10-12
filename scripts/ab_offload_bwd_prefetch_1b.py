"""Lean 1B A/B for the backward recompute prefetch (2026-10-12).

Derived from scripts/ab_offload_losshead_1b.py (same model build: the trainer's
own constructors, synthetic tokens, one full 13-block BCD cycle per arm) plus:
  * every GPUBufferPool.load() MISS is timed with CUDA events on the main stream
    and split forward / backward -- the quantity the lever removes, so the
    validation run on main and the A/B both report it per step;
  * H1B_BWD=0/1 toggles model._offload_bwd_prefetch; H1B_GLUE=0/1 forces the
    fused DeltaNet glue kernels off/on (feat/fused-l2norm-glue); H1B_STEPS caps
    the step count;
  * prints the offload pool's buffer sets per shape signature at the end (the
    VRAM claim: backward prefetch must not add a set).

Usage: H1B_BWD=1 H1B_STEPS=14 python scripts/ab_offload_bwd_prefetch_1b.py            --repo <checkout> --out arm.json --warmup 1
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

# ---- lever 1b instrumentation: time every load() MISS on the main stream ----
PHASE = {"bwd": False}
MISS = {"fwd_n": 0, "bwd_n": 0, "fwd_ev": [], "bwd_ev": [], "bwd_host": 0.0, "fwd_host": 0.0,
        "bwd_bytes": 0}
_orig_load = GPUBufferPool.load


def _timed_load(self, key, params):
    miss = key not in self._resident and bool(self._offloadable(params))
    if not miss:
        return _orig_load(self, key, params)
    ph = "bwd" if PHASE["bwd"] else "fwd"
    a = torch.cuda.Event(enable_timing=True)
    b = torch.cuda.Event(enable_timing=True)
    h0 = time.perf_counter()
    a.record()
    _orig_load(self, key, params)
    b.record()
    MISS[ph + "_host"] += time.perf_counter() - h0
    MISS[ph + "_n"] += 1
    MISS[ph + "_ev"].append((a, b))
    if ph == "bwd":
        MISS["bwd_bytes"] += sum(self._cpu_home[id(q)].nbytes for q in self._offloadable(params))


GPUBufferPool.load = _timed_load
MAX_STEPS = int(os.environ.get("H1B_STEPS", "0"))

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
if "H1B_GLUE" in os.environ:
    _glue = os.environ["H1B_GLUE"] == "1"
    _n = 0
    for _m in model.modules():
        if hasattr(_m, "use_fused_l2norm"):
            _m.use_fused_l2norm = _glue
            _m.use_fused_read_norm = _glue
            _n += 1
    out["glue_fused"] = _glue
    print("[cfg] fused glue", _glue, "on", _n, "DeltaNet blocks", flush=True)
    assert _n > 0, "no DeltaNet block has the fused-glue flags -- wrong checkout?"
model._offload_bwd_prefetch = os.environ.get("H1B_BWD", "1") == "1"
out["bwd_prefetch"] = model._offload_bwd_prefetch
print("[cfg] bwd_prefetch", model._offload_bwd_prefetch, flush=True)
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
if MAX_STEPS:
    n_steps = min(n_steps, MAX_STEPS)
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
        PHASE["bwd"] = True
        loss.backward()
        PHASE["bwd"] = False
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
    for ph in ("fwd", "bwd"):
        r[ph + "_miss_n"] = MISS[ph + "_n"]
        r[ph + "_miss_gpu"] = sum(a.elapsed_time(b) for a, b in MISS[ph + "_ev"]) / 1e3
        r[ph + "_miss_host"] = MISS[ph + "_host"]
        MISS[ph + "_n"] = 0
        MISS[ph + "_ev"] = []
        MISS[ph + "_host"] = 0.0
    r["bwd_miss_bytes"] = MISS["bwd_bytes"]
    MISS["bwd_bytes"] = 0
    print("        misses: fwd %d (gpu %.3fs host %.3fs)  bwd %d (gpu %.3fs host %.3fs, %.2f GB)" % (
        r["fwd_miss_n"], r["fwd_miss_gpu"], r["fwd_miss_host"], r["bwd_miss_n"], r["bwd_miss_gpu"],
        r["bwd_miss_host"], r["bwd_miss_bytes"] / 1e9), flush=True)
    out["steps"].append(r)
    dump()
    print("[%s %2d] %-34s wall %6.2f fwd %5.2f loss %.7f (%s) peak-rsv %.2fG (%.1f%%) ram %.1fG" % (
        "warm" if r["warmup"] else "step", s, active, wall, fwd_gpu, r["loss"], r["loss_hex"],
        r["peak_reserved"] / 2 ** 30, 100 * r["peak_reserved"] / out["card_bytes"],
        r["ram_free_gib"]), flush=True)
_pool = sched.buffer_pool
_sets = {}
for sig, lst in _pool._free_buffers.items():
    _sets[sig] = _sets.get(sig, 0) + len(lst)
for e in _pool._resident.values():
    _sets[e["sig"]] = _sets.get(e["sig"], 0) + 1
out["pool_sets"] = [{"n_params": len(sig), "sets": n,
                     "mib_per_set": sum(torch.Size(s).numel() * torch.empty((), dtype=dt).element_size()
                                        for s, dt in sig) / 2**20} for sig, n in _sets.items()]
dump()
print("[pool]", out["pool_sets"], flush=True)
print("[done]", flush=True)
