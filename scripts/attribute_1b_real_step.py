"""Where does the REAL 1B step go now? (re-attribution after the 2026-10-12 levers)

Author: Rambarun Komaljeet

Why a new script and not `attribute_1b_step.py`
------------------------------------------------
The table everyone quotes (elementwise 36 / gemm 33 / scan 15 / other 15 /
optimizer 2, at 3.67 s/step) came from `attribute_1b_step.py`, which builds a
plain `EdgeHVN`: **no TernaryLinear at all**, untied embeddings, ONE micro-batch,
`steps_per_block=50`, a non-balanced partition. The real 1B config -- the one
the 22% levers were measured on -- is the trainer's own build: hvn-hybrid,
ternary + fused quantizer, tied embeddings, grad_accum 8, balanced BCD with 13
blocks, offload + prefetch, ~17 s/step. A number from a different harness is not
a measurement of this one, so this reuses `scripts/ab_offload_bwd_prefetch_1b.py`'s
build exactly and keeps `attribute_1b_step.py`'s METHOD:

* attribute CUDA device time by kernel name, **kernel rows only**
  (`device_type == CUDA`) -- op-level rows carry the same time (#131 double count);
* `other` is printed with its top kernels, never folded away;
* device time per step must match the unprofiled wall clock (ratio check).

What is new
-----------
* One FULL BCD cycle (13 steps, every block active once) is measured twice:
  cycle A unprofiled (wall + CUDA-event phase split fwd / bwd / optimizer), then
  cycle B with a fresh profiler per step (bounded RAM). Step time ranges 10-30 s
  by active block, so one block is not a sample of the step.
* Categories split `other` into `pcie` (Memcpy HtoD/DtoH) and the rest, and
  name the ternary quantizer kernel on its own.
* Weight-shaped `aten::copy_` (2-D, >= 64K elements, no sequence dim) is split
  by the KERNEL it launched (PCIe memcpy vs on-device cast) and by its CPU
  parent op. The old probe attributed the [2048,2048] row to "autocast cast and
  the BitNet STE" on a harness that had no STE, and an op-level copy_ row also
  includes the H2D memcpy it launches -- so that reading is checked, not reused.

Falsifiability: if GEMM had NOT grown, or the copy row were PCIe rather than
casts, the kernel split shows it directly; the ratio check voids the run if the
profiler misses or double counts work.

Usage::

    python -u scripts/attribute_1b_real_step.py --out results/attribution_1b_real/main.json
"""
import argparse
import collections
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", required=True)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--min-free-ram-gib", type=float, default=1.5)
ap.add_argument("--no-profile", action="store_true", help="cycle A only (timing)")
ap.add_argument("--cycles-a", type=int, default=1, help="unprofiled cycles")
ap.add_argument("--step-cache", action="store_true",
                help="enable the within-step TernaryLinear quantized-weight cache")
args = ap.parse_args()

sys.path.insert(0, args.repo)
import pyarrow  # noqa: F401,E402  -- before CUDA (ed4777d heap-corruption fix)
import torch  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402

import train_edge_conversational as tec  # noqa: E402
import hvn.optimizers as hopt  # noqa: E402
import hvn.edge.chunked_loss as hcl  # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402


def free_ram_gib() -> float:
    import ctypes

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


CATS = [
    ("pcie", ("memcpy htod", "memcpy dtoh")),
    ("scan", ("gated_delta", "gd_fwd", "gd_bwd", "_mk_", "mk_fwd", "mk_bwd",
              "loggram", "delta_rule")),
    ("ternary_q", ("_quantize_kernel", "_absmean_partial")),
    ("gemm", ("gemm", "cutlass", "ampere", "sm80", "sm89", "sm90", "dot_kernel",
              "s16816", "implicit", "cublas", "nn_128x", "tn_128x", "nt_128x",
              "splitk", "xmma")),
    ("optimizer", ("newton", "schulz", "adam", "muon", "orthogon", "foreach")),
    ("elementwise", ("elementwise", "vectorized", "pointwise", "reduce",
                     "norm", "copy", "cast", "fill", "index", "embedding",
                     "softmax", "cat_", "unrolled", "direct_copy", "_l2n_",
                     "_rn_", "memset", "memcpy dtod", "scatter", "gather",
                     "cross_entropy", "nll", "log_softmax", "where", "sort")),
]


def categorize(name):
    low = name.lower()
    for cat, keys in CATS:
        if any(k in low for k in keys):
            return cat
    return "other"


out = {"config": vars(args), "steps_a": [], "steps_b": [], "aborted": None}


def dump():
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)


def guard(where):
    r = free_ram_gib()
    if r < args.min_free_ram_gib:
        out["aborted"] = f"free RAM {r:.2f} GiB < {args.min_free_ram_gib} at {where}"
        dump()
        print("[abort]", out["aborted"], flush=True)
        sys.exit(3)
    return r


GA, SEQ, VOCAB, CHUNK = 8, 4096, 50257, 512
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
                             compile_model=False, verbose=False, fused_ternary=True,
                             fused_rmsnorm=True)
model.train()
if args.step_cache:
    from hvn.edge.ternary import enable_ternary_step_cache
    out["step_cache_modules"] = enable_ternary_step_cache(model, True)
    print(f"[cfg] within-step ternary cache ON for {out['step_cache_modules']} modules", flush=True)
opt, desc = hopt.build_optimizer(model, "muon", lr=3e-4, weight_decay=0.01,
                                 fused=False, muon_lr=0.02)
sched = BlockCoordinateScheduler(
    model, opt, steps_per_block=1, offload_inactive_weights=True, device=dev,
    offload_park_dtype=torch.bfloat16, offload_prefetch=True,
    offload_cache_blocks=1, partition="balanced")
NB = sched.num_blocks
out["params"] = sum(p.numel() for p in model.parameters())
out["bcd_blocks"] = NB
print(f"[build] {out['params']:,} params, {NB} BCD blocks, {time.time()-t0:.0f}s, "
      f"ram {free_ram_gib():.1f}G", flush=True)

g = torch.Generator().manual_seed(0)
XS = torch.randint(0, VOCAB, (GA, 1, SEQ), generator=g)
YS = torch.randint(0, VOCAB, (GA, 1, SEQ), generator=g)


class _Null:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def one_step(phases=None, seg=None):
    """seg: optional (ctx factory, consumer). Each micro-batch and the optimizer
    phase run inside their own ctx, consumed right after -- one profiler per
    segment holds 1/8 of a step's events, which is what keeps host RAM bounded
    (a whole-step profile of the heaviest block took free RAM below 1.2 GiB)."""
    mk, eat = seg if seg is not None else (_Null, lambda c: None)
    loss_acc = torch.zeros((), device=dev)
    for mb in range(GA):
      with (mk() if mb == 0 else _Null()) as ctx:
        x = XS[mb].to(dev, non_blocking=True)
        y = YS[mb].to(dev, non_blocking=True)
        ev = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
        ev[0].record()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(x, return_hidden=True)
            loss = hcl.chunked_cross_entropy(hidden, y, model.decoder, chunk=CHUNK).mean() / GA
        ev[1].record()
        loss.backward()
        ev[2].record()
        loss_acc += loss.detach()
        if phases is not None:
            phases.append(("fwd", ev[0], ev[1]))
            phases.append(("bwd", ev[1], ev[2]))
        if seg is not None and mb == 0:
            torch.cuda.synchronize()
      if mb == 0:
          eat(ctx)
      del ctx
    e0 = torch.cuda.Event(enable_timing=True)
    e1 = torch.cuda.Event(enable_timing=True)
    e0.record()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    opt.zero_grad()
    sched.step()
    e1.record()
    if phases is not None:
        phases.append(("opt", e0, e1))
    return loss_acc


for s in range(args.warmup):
    a = sched.active_block_name
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    lv = one_step().item()
    print(f"[warm {s}] {a:<34} wall {time.perf_counter()-w0:6.2f} loss {lv:.6f}", flush=True)
    guard("warm")
torch.cuda.reset_peak_memory_stats()

for c in range(args.cycles_a):
    for s in range(NB):
        a = sched.active_block_name
        ph = []
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        lv = one_step(ph).item()
        torch.cuda.synchronize()
        wall = time.perf_counter() - w0
        split = collections.Counter()
        for k, e0, e1 in ph:
            split[k] += e0.elapsed_time(e1) / 1e3
        import hvn.edge.ternary as _T
        r = {"cycle": c, "active": a, "wall": wall, "loss": lv, **dict(split),
             "loss_hex": float(lv).hex(), "cache_stats": dict(_T.STEP_CACHE_STATS),
             "peak_reserved": torch.cuda.max_memory_reserved(),
             "peak_alloc": torch.cuda.max_memory_allocated(), "ram": guard("A")}
        out["steps_a"].append(r)
        dump()
        print(f"[A{c} {s:2d}] {a:<34} wall {wall:6.2f} fwd {split['fwd']:5.2f} "
              f"bwd {split['bwd']:5.2f} opt {split['opt']:5.2f} loss {lv:.6f} "
              f"rsv {r['peak_reserved']/2**30:.2f}G", flush=True)
        torch.cuda.reset_peak_memory_stats()

wall_a = [r["wall"] for r in out["steps_a"]]
print(f"[A] cycle-mean wall {sum(wall_a)/len(wall_a):.3f} s/step over {len(wall_a)} steps; "
      f"max peak reserved {max(r['peak_reserved'] for r in out['steps_a'])/2**20:.0f} MiB",
      flush=True)
if args.no_profile:
    print("[done]", flush=True)
    sys.exit(0)

KT = collections.Counter()          # kernel name -> us (all profiled steps)
OPS = collections.Counter()         # aten op -> us (separate view, never summed with KT)
COPY_SHAPE = collections.Counter()  # shape -> us, aten::copy_ only
COPY_CALLS = collections.Counter()
WCOPY = collections.Counter()       # (shape, parent chain, kernel class) -> us
WCOPY_N = collections.Counter()
dropped = 0.0
annot = {}


def kclass(n):
    low = n.lower()
    if "memcpy htod" in low:
        return "H2D"
    if "memcpy dtoh" in low:
        return "D2H"
    if "memcpy dtod" in low:
        return "D2D-memcpy"
    return "device-kernel"


def chain(e, k=4):
    names = []
    p = e.cpu_parent
    while p is not None and len(names) < k:
        names.append(p.name)
        p = p.cpu_parent
    return " <- ".join(names) if names else "(top-level)"


import gc  # noqa: E402

def consume_shapes(prof):
    for e in prof.key_averages(group_by_input_shape=True):
        if e.key != "aten::copy_":
            continue
        d = getattr(e, "self_device_time_total", 0) or 0
        if d <= 0:
            continue
        shp = tuple(e.input_shapes[0]) if e.input_shapes else ()
        COPY_SHAPE[shp] += d
        COPY_CALLS[shp] += int(e.count)
    for e in prof.events():
        if e.name != "aten::copy_" or e.device_type == DeviceType.CUDA:
            continue
        shp = e.input_shapes[0] if e.input_shapes else []
        n = 1
        for x_ in shp:
            n *= x_
        if len(shp) != 2 or n < 65536 or SEQ in shp:
            continue
        for kk in e.kernels:
            key = (str(list(shp)), chain(e), kclass(kk.name))
            WCOPY[key] += kk.duration
            WCOPY_N[key] += 1
    del prof
    gc.collect()


# ---- cycle B: whole-step profiles, kernel rows only, no shapes (RAM) ----
for s in range(NB):
    a = sched.active_block_name
    cur = collections.Counter()
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        lv = one_step().item()
        torch.cuda.synchronize()
    wall = time.perf_counter() - w0
    for e in prof.key_averages():
        d = getattr(e, "self_device_time_total", 0) or 0
        if d <= 0:
            continue
        if getattr(e, "device_type", None) != DeviceType.CUDA:
            dropped += d
            if e.key.startswith("aten::"):
                OPS[e.key[6:]] += d
            continue
        if "#" in e.key or e.key.startswith("Optimizer."):
            # GPU-side USER ANNOTATION ("Optimizer.step#Muon.step"): a range
            # spanning the optimizer's kernels, which are already counted as
            # gemm/elementwise rows. Same double count as #131, one level up:
            # it carries device_type CUDA, so the kernel filter alone lets it
            # in (0.25 s/step here; likely all of the old table's "optimizer").
            annot[e.key] = annot.get(e.key, 0.0) + d
            continue
        KT[e.key] += d
        cur[categorize(e.key)] += d
    del prof
    gc.collect()
    tot = sum(cur.values()) / 1e6
    r = {"active": a, "wall_profiled": wall, "device_s": tot, "loss": lv,
         "cats": {k: v / 1e6 for k, v in cur.items()}, "ram": guard("B")}
    out["steps_b"].append(r)
    out["agg_kernels"] = dict(KT)
    out["agg_annotations"] = dict(annot)
    out["agg_ops"] = dict(OPS)
    print(f"[B {s:2d}] {a:<34} dev {tot:6.2f}s (profiled wall {wall:6.2f}) " +
          " ".join(f"{k}={v/1e6:.2f}" for k, v in sorted(cur.items(), key=lambda kv: -kv[1])) +
          f" ram {r['ram']:.1f}G", flush=True)
    dump()


def mk_prof():
    return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                   record_shapes=True)


# ---- cycle C: copy_ shapes, micro-batch 0 of each step only (x GA below) ----
for s in range(NB):
    a = sched.active_block_name
    one_step(seg=(mk_prof, consume_shapes))
    guard("C")
    out["agg_copy"] = {"n_steps": s + 1,
                       "copy_shape": {str(list(k)): v for k, v in COPY_SHAPE.items()},
                       "copy_calls": {str(list(k)): v for k, v in COPY_CALLS.items()},
                       "wcopy": {" | ".join(k): v for k, v in WCOPY.items()},
                       "wcopy_n": {" | ".join(k): v for k, v in WCOPY_N.items()}}
    print(f"[C {s:2d}] {a} copy pass done", flush=True)
    dump()

# ---------------- summary ----------------
ref = sum(r["wall"] for r in out["steps_a"] if r["cycle"] == 0) / NB
tot_cat = collections.Counter()
for r in out["steps_b"]:
    for k, v in r["cats"].items():
        tot_cat[k] += v
dev_step = sum(tot_cat.values()) / NB
print(f"\n[attr] unprofiled step (cycle mean): {ref:.3f} s; device/step {dev_step:.3f} s; "
      f"ratio {dev_step/ref:.2f}")
ph = collections.Counter()
for r in out["steps_a"]:
    if r["cycle"] == 0:
        for k in ("fwd", "bwd", "opt"):
            ph[k] += r[k] / NB
print("[attr] phase split (CUDA events, cycle mean): " +
      " ".join(f"{k} {v:.3f}s" for k, v in ph.items()))
print(f"[attr] {'category':<14}{'s/step':>10}{'share':>9}")
rows = {}
for k, v in sorted(tot_cat.items(), key=lambda kv: -kv[1]):
    s_ = v / NB
    rows[k] = {"s_per_step": s_, "share_pct": 100 * s_ / dev_step}
    print(f"[attr] {k:<14}{s_:>10.3f}{100*s_/dev_step:>8.1f}%")
print(f"[attr] (excluded {dropped/NB/1e6:.3f} s/step of op-level launcher rows, #131, and "
      f"{sum(annot.values())/NB/1e6:.3f} s/step of GPU user-annotation ranges)")
print("\n[attr] top kernels:")
for k, v in KT.most_common(25):
    print(f"[attr]  {v/NB/1e6:7.3f}s [{categorize(k):<11}] {k[:90]}")
print("\n[attr] top 'other' kernels:")
for k, v in [kv for kv in KT.most_common() if categorize(kv[0]) == "other"][:10]:
    print(f"[attr]  {v/NB/1e6:7.3f}s {k[:100]}")
print("\n[attr] op-level view (separate; do NOT add to the kernel table):")
for k, v in OPS.most_common(16):
    print(f"[attr]  {v/NB/1e6:7.3f}s {k}")
print("\n[copy] aten::copy_ by shape (ms/step, calls/step; micro-batch 0 x GA, no optimizer phase):")
for k, v in COPY_SHAPE.most_common(12):
    print(f"[copy]  {GA*v/NB/1e3:8.1f} ms {GA*COPY_CALLS[k]/NB:7.1f}  {list(k)}")
print("\n[copy] weight-shaped copy_ split by launched kernel and caller:")
for k, v in WCOPY.most_common(20):
    print(f"[copy]  {GA*v/NB/1e3:8.1f} ms {GA*WCOPY_N[k]/NB:7.1f}  {k[2]:<13} {k[0]:<14} {k[1][:80]}")
out["summary"] = {
    "ref_step_s": ref, "device_s_per_step": dev_step, "ratio": dev_step / ref,
    "phase_s": dict(ph), "categories": rows,
    "top_kernels": [(k, v / NB / 1e6, categorize(k)) for k, v in KT.most_common(40)],
    "ops": [(k, v / NB / 1e6) for k, v in OPS.most_common(40)],
    "copy_by_shape": [(list(k), v / NB / 1e3, COPY_CALLS[k] / NB)
                      for k, v in COPY_SHAPE.most_common(30)],
    "weight_copy": [(list(k), v / NB / 1e3, WCOPY_N[k] / NB) for k, v in WCOPY.most_common(40)],
}
dump()
print("[done]", flush=True)
