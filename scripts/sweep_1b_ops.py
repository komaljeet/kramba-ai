"""Final 1B speed sweep (2026-10-15): flag audit + per-op attribution of the real step.

Author: Rambarun Komaljeet

Same build as scripts/attribute_1b_real_step.py with the --fast step cache
(hvn-hybrid 2048x24, attn [6,18], ternary + fused quantizer + step cache, tied
embeddings, chunked loss 512, Muon, grad-ckpt, balanced BCD + offload + bf16
parking + prefetch, strided-IO megakernel scan). Three things it adds:

1. Flag audit at RUNTIME, not by reading defaults: every fused entry point in
   hvn.core.deltanet is wrapped with a call counter, and the ternary cache
   counters are read. A flag whose "None = on" default failed to resolve shows
   up as zero calls on its fused path and nonzero on the eager fallback.
2. Kernel-row attribution (#131-safe, device_type == CUDA, GPU annotation
   ranges excluded) over one full BCD cycle, top kernels listed by name.
3. Per-op attribution: each CUDA kernel's time is charged to the CPU op that
   launched it (FunctionEvent.kernels belongs to the innermost op, so nothing is
   counted twice), keyed by (owning module or autograd node, op, input shape).
   Module context comes from forward pre/post hooks that open a
   record_function range "mod:<Class>.<attr>"; backward kernels are keyed by
   the autograd node that launched them. Run per segment (micro-batch 0 and
   the optimizer phase of every step) to bound host RAM; micro-batch rows are
   scaled x GA.

Falsifiability: the kernel-row device total must match the unprofiled wall
(ratio printed), and the per-op table's total must match the kernel-row total
for the same segment (printed) -- if attribution drops or double counts, the
two disagree.
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
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--min-free-ram-gib", type=float, default=1.5)
ap.add_argument("--skip-ops", action="store_true")
args = ap.parse_args()

sys.path.insert(0, args.repo)
import pyarrow  # noqa: F401,E402  -- before CUDA (ed4777d)
import torch  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import profile, ProfilerActivity, record_function  # noqa: E402

import train_edge_conversational as tec  # noqa: E402
import hvn.optimizers as hopt  # noqa: E402
import hvn.edge.chunked_loss as hcl  # noqa: E402
import hvn.core.deltanet as hdn  # noqa: E402
import hvn.edge.ternary as hter  # noqa: E402
import hvn.core.deltanet_triton as hdt  # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402
from hvn.core.transformer_block import TransformerBlock  # noqa: E402


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


# ---- 1. runtime call counters on every fused / fallback entry point ----
CALLS = collections.Counter()


def _count(modname, mod, attr):
    fn = getattr(mod, attr)
    if fn is None:
        CALLS[f"{modname}.{attr}=None(unavailable)"] += 0
        return

    def wrapped(*a, **k):
        CALLS[f"{modname}.{attr}"] += 1
        return fn(*a, **k)
    setattr(mod, attr, wrapped)


for _a in ("_triton_megakernel_scan_bshd", "_triton_megakernel_scan",
           "_triton_gated_delta_scan", "_fused_causal_conv", "_fused_l2norm",
           "_l2norm", "_fused_read_norm"):
    _count("deltanet", hdn, _a)
_orig_rn = hdn.DeltaNetBlock._read_normalize
_orig_sdpa = torch.nn.functional.scaled_dot_product_attention


def _rn(self, o):
    CALLS["deltanet.DeltaNetBlock._read_normalize(eager)"] += 1
    return _orig_rn(self, o)


hdn.DeltaNetBlock._read_normalize = _rn


def _sdpa(*a, **k):
    key = "sdpa(is_causal=True)" if k.get("is_causal") else (
        "sdpa(explicit mask)" if k.get("attn_mask") is not None else "sdpa(other)")
    CALLS[key] += 1
    return _orig_sdpa(*a, **k)


torch.nn.functional.scaled_dot_product_attention = _sdpa

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
# Exactly the --fast preset's call (main() passes these five).
model = tec.apply_fast_stack(model, triton_scan=True, megakernel_scan=True, tf32=True,
                             compile_model=False, verbose=False, fused_ternary=True,
                             ternary_step_cache=True)
model.train()

# static flag audit (what each block will resolve to)
flag_rows = collections.Counter()
for i, blk in enumerate(model.blocks):
    if isinstance(blk, hdn.DeltaNetBlock):
        for f in ("use_triton_scan", "use_megakernel_scan", "use_fused_l2norm",
                  "use_fused_read_norm", "use_fused_conv", "use_fused_scan_io",
                  "adaptive_chunk", "train_chunk"):
            flag_rows[f"deltanet.{f}={getattr(blk, f, 'MISSING')}"] += 1
    else:
        flag_rows[f"{type(blk).__name__}(window={getattr(blk, 'window', None)})"] += 1
tl = [m for m in model.modules() if isinstance(m, hter.TernaryLinear)]
flag_rows[f"TernaryLinear n={len(tl)} use_fused={sum(m.use_fused for m in tl)} "
          f"use_step_cache={sum(m.use_step_cache for m in tl)}"] += 1
dropouts = collections.Counter(m.p for m in model.modules() if isinstance(m, torch.nn.Dropout))
flag_rows[f"dropout p counts={dict(dropouts)}"] += 1
out["static_flags"] = dict(flag_rows)
print("[flags]", json.dumps(dict(flag_rows), indent=1), flush=True)

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


def one_step(seg=None):
    """seg: (ctx factory, consumer): micro-batch 0 and the optimizer phase each
    run inside their own profiler, consumed right after (bounded host RAM)."""
    mk, eat = seg if seg is not None else (_Null, lambda c, w: None)
    loss_acc = torch.zeros((), device=dev)
    for mb in range(GA):
        with (mk() if mb == 0 else _Null()) as ctx:
            x = XS[mb].to(dev, non_blocking=True)
            y = YS[mb].to(dev, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                hidden = model(x, return_hidden=True)
                loss = hcl.chunked_cross_entropy(hidden, y, model.decoder, chunk=CHUNK).mean() / GA
            loss.backward()
            loss_acc += loss.detach()
            if seg is not None and mb == 0:
                torch.cuda.synchronize()
        if mb == 0:
            eat(ctx, "mb")
        del ctx
    with mk() as ctx:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        opt.step()
        opt.zero_grad()
        sched.step()
        if seg is not None:
            torch.cuda.synchronize()
    eat(ctx, "opt")
    del ctx
    return loss_acc


for s in range(args.warmup):
    a = sched.active_block_name
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    lv = one_step().item()
    print(f"[warm {s}] {a:<34} wall {time.perf_counter()-w0:6.2f} loss {lv:.6f}", flush=True)
    guard("warm")

# ---- cycle A: unprofiled timing + runtime flag counters over exactly one cycle ----
CALLS.clear()
for k in hter.STEP_CACHE_STATS:
    hter.STEP_CACHE_STATS[k] = 0
for k in hter.FROZEN_SCALE_STATS:
    hter.FROZEN_SCALE_STATS[k] = 0
fb0 = hdt.SCAN_IO_STATS["fallback_copies"]
torch.cuda.reset_peak_memory_stats()
for s in range(NB):
    a = sched.active_block_name
    torch.cuda.synchronize()
    w0 = time.perf_counter()
    lv = one_step().item()
    torch.cuda.synchronize()
    wall = time.perf_counter() - w0
    out["steps_a"].append({"active": a, "wall": wall, "loss": lv, "ram": guard("A")})
    print(f"[A {s:2d}] {a:<34} wall {wall:6.2f} loss {lv:.6f}", flush=True)
out["runtime_calls_per_cycle"] = dict(CALLS)
out["step_cache"] = dict(hter.STEP_CACHE_STATS)
out["frozen_scale"] = dict(hter.FROZEN_SCALE_STATS)
out["scan_io_fallback_copies"] = hdt.SCAN_IO_STATS["fallback_copies"] - fb0
out["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
ref = sum(r["wall"] for r in out["steps_a"]) / NB
out["ref_step_s"] = ref
print(f"[A] cycle-mean wall {ref:.3f} s/step, peak reserved {out['peak_reserved_mib']:.0f} MiB",
      flush=True)
print("[calls/cycle]", json.dumps(dict(CALLS), indent=1), flush=True)
print("[cache]", out["step_cache"], out["frozen_scale"],
      "scan_io_fallback_copies", out["scan_io_fallback_copies"], flush=True)
dump()

# ---- cycle B: whole-step kernel rows (#131-safe) ----
KT = collections.Counter()
annot = 0.0
for s in range(NB):
    a = sched.active_block_name
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        lv = one_step().item()
        torch.cuda.synchronize()
    tot = 0.0
    for e in prof.key_averages():
        d = getattr(e, "self_device_time_total", 0) or 0
        if d <= 0 or getattr(e, "device_type", None) != DeviceType.CUDA:
            continue
        if "#" in e.key or e.key.startswith("Optimizer."):
            annot += d
            continue
        KT[e.key] += d
        tot += d
    del prof
    gc.collect()
    out["steps_b"].append({"active": a, "device_s": tot / 1e6, "ram": guard("B")})
    print(f"[B {s:2d}] {a:<34} dev {tot/1e6:6.2f}s", flush=True)
dev_step = sum(r["device_s"] for r in out["steps_b"]) / NB
out["device_s_per_step"] = dev_step
out["top_kernels"] = [(k, v / NB / 1e6) for k, v in KT.most_common(60)]
print(f"\n[attr] unprofiled {ref:.3f} s/step; device {dev_step:.3f} s/step; ratio {dev_step/ref:.2f}"
      f" (annotation ranges excluded: {annot/NB/1e6:.3f} s/step)")
print("[attr] top kernels (s/step, share of device time):")
for k, v in KT.most_common(40):
    print(f"[attr]  {v/NB/1e6:7.3f}s {100*v/NB/1e6/dev_step:5.1f}%  {k[:110]}")
dump()
if args.skip_ops:
    print("[done]", flush=True)
    sys.exit(0)

# ---- cycle C: per-op attribution with module context ----
STACK = []


def _pre(name):
    def h(mod, inp):
        rf = record_function(name)
        rf.__enter__()
        STACK.append(rf)
    return h


def _post(mod, inp, outp):
    if STACK:
        STACK.pop().__exit__(None, None, None)


hooks = []
for bi, blk in enumerate(model.blocks):
    bt = "attn" if isinstance(blk, TransformerBlock) else "dn"
    for n, m in blk.named_modules():
        if n == "":
            nm = f"mod:{bt}"
        else:
            nm = f"mod:{bt}.{n}"
        hooks.append(m.register_forward_pre_hook(_pre(nm)))
        hooks.append(m.register_forward_hook(_post))
for n, m in (("final_norm", model.final_norm), ("decoder", model.decoder),
             ("token_embedding", model.token_embedding)):
    hooks.append(m.register_forward_pre_hook(_pre(f"mod:{n}")))
    hooks.append(m.register_forward_hook(_post))

OPK = collections.Counter()   # (phase, context, op, shape) -> us
OPN = collections.Counter()
SEGK = collections.Counter()  # phase -> kernel-row total, for the cross-check
SEGO = collections.Counter()  # phase -> per-op total


def ctx_of(e):
    p = e.cpu_parent
    mod = None
    node = None
    while p is not None:
        nm = p.name
        if mod is None and nm.startswith("mod:"):
            mod = nm
        if node is None and nm.startswith("autograd::engine::evaluate_function:"):
            node = nm.split(":", 4)[-1].strip()
        p = p.cpu_parent
    if node is not None:
        return f"bwd[{node}]" + (f"<{mod}>" if mod else "")
    return mod or "(no module)"


def consume(prof, phase):
    for e in prof.key_averages():
        d = getattr(e, "self_device_time_total", 0) or 0
        if d > 0 and getattr(e, "device_type", None) == DeviceType.CUDA \
                and "#" not in e.key and not e.key.startswith("Optimizer."):
            SEGK[phase] += d
    for e in prof.events():
        if e.device_type == DeviceType.CUDA or not e.kernels:
            continue
        shp = e.input_shapes[0] if e.input_shapes else []
        dt = 0.0
        for kk in e.kernels:
            dt += kk.duration
        key = (phase, ctx_of(e), e.name, str(list(shp)))
        OPK[key] += dt
        OPN[key] += 1
        SEGO[phase] += dt
    del prof
    gc.collect()


def mk_prof():
    return profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True)


# Non-reentrant checkpoint stops the recompute early by default, which would
# skip the post-hooks of the block's tail modules and leave ranges open. Off
# for this attribution-only pass (it may recompute a few extra tail ops; cycles
# A/B above are untouched).
from torch.utils.checkpoint import set_checkpoint_early_stop  # noqa: E402
for s in range(NB):
    a = sched.active_block_name
    with set_checkpoint_early_stop(False):
        one_step(seg=(mk_prof, lambda c, w: consume(c, w)))
    if STACK:
        print(f"[warn] {len(STACK)} unclosed module ranges", flush=True)
        STACK.clear()
    guard("C")
    print(f"[C {s:2d}] {a} op pass done; mb kernels {SEGK['mb']/1e6:.2f}s ops {SEGO['mb']/1e6:.2f}s",
          flush=True)

scale = {"mb": GA, "opt": 1}
rows = [(k, v * scale[k[0]] / NB / 1e6, OPN[k] * scale[k[0]] / NB) for k, v in OPK.items()]
rows.sort(key=lambda r: -r[1])
out["op_rows"] = [(list(k), s_, n) for k, s_, n in rows[:300]]
out["op_cross_check"] = {"kernel_rows_mb": SEGK["mb"], "op_rows_mb": SEGO["mb"],
                         "kernel_rows_opt": SEGK["opt"], "op_rows_opt": SEGO["opt"]}
print(f"\n[ops] cross-check mb: kernel rows {SEGK['mb']/1e6:.3f}s vs op rows {SEGO['mb']/1e6:.3f}s;"
      f" opt: {SEGK['opt']/1e6:.3f}s vs {SEGO['opt']/1e6:.3f}s")
print("[ops] top (phase, context, op, shape): s/step calls/step")
for k, s_, n in rows[:80]:
    print(f"[ops] {s_:7.3f}s {n:7.1f}  {k[0]:<3} {k[1][:60]:<60} {k[2]:<32} {k[3]}")
# rollup by op name
byop = collections.Counter()
for k, s_, n in rows:
    byop[k[2]] += s_
print("\n[ops] by op name:")
for k, v in byop.most_common(30):
    print(f"[ops]  {v:7.3f}s {k}")
bymod = collections.Counter()
for k, s_, n in rows:
    c = k[1]
    m = c[c.find("<") + 1:c.find(">")] if "<" in c else c
    bymod[m] += s_
print("\n[ops] by module context (fwd+recompute+bwd where known):")
for k, v in bymod.most_common(40):
    print(f"[ops]  {v:7.3f}s {k}")
out["by_op"] = byop.most_common(60)
out["by_module"] = bymod.most_common(80)
dump()
print("[done]", flush=True)
