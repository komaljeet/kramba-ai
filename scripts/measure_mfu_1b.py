"""Real MFU of the 1B pretrain step (2026-10-15): counted FLOPs + measured wall.

Author: Rambarun Komaljeet

Build: identical to scripts/ab_scan_strided_io_1b.py with --step-cache (the
--fast stack: hvn-hybrid 2048x24, attn [6,18], ternary + fused quantizer +
step cache, tied embeddings, chunked loss 512, Muon, grad-ckpt, balanced BCD
with offload + bf16 parking + prefetch, strided-IO megakernel scan).

1. cycles A: unprofiled full BCD cycles -> wall s/step (cycle mean).
2. cycle F: one full BCD cycle, EVERY micro-batch (8) and the optimizer phase
   under torch.utils.flop_counter.FlopCounterMode, run AFTER timing (the
   counter decomposes unregistered ops, so it must never be timed).
   - Each counted op is bucketed exactly once (fwd / ckpt-recompute / bwd / opt)
     in an override of _count_flops. The per-module table of FlopCounterMode
     repeats every op under each parent module (Global + every ancestor) --
     the #131-style double count -- so only the Global total is used, and the
     bucket sum must equal it (asserted).
   - Recompute is tagged by wrapping the checkpoint function that
     hvn.edge.model and hvn.edge.chunked_loss call: the wrapped fn's 2nd+
     invocation is recompute.
   - The Triton scan is invisible to FlopCounterMode (no ATen op). Its calls
     are counted by wrapping _MegaScanBSHD.forward/backward (shape per call);
     FLOPs per call come from the kernel source op count (SCAN_* below), and
     the forward constant is checked against an op-level count of the eager
     reference scan.
"""
import argparse, collections, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", required=True)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--cycles-a", type=int, default=2)
ap.add_argument("--skip-count", action="store_true")
ap.add_argument("--min-free-ram-gib", type=float, default=1.5)
args = ap.parse_args()
sys.path.insert(0, args.repo)
import pyarrow  # noqa: F401  -- before CUDA (ed4777d)
import torch
from torch.utils.flop_counter import FlopCounterMode
from torch.utils._python_dispatch import TorchDispatchMode

import train_edge_conversational as tec
import hvn.optimizers as hopt
import hvn.edge.chunked_loss as hcl
import hvn.edge.model as hem
import hvn.core.deltanet as hdn
import hvn.core.deltanet_triton as hdt
from hvn.core.block_coordinate import BlockCoordinateScheduler


def free_ram_gib():
    import ctypes
    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MS(); m.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m.ullAvailPhys / 2 ** 30


out = {"config": vars(args), "steps_a": [], "steps_f": [], "aborted": None}


def dump():
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1, default=str)


def guard(where):
    r = free_ram_gib()
    if r < args.min_free_ram_gib:
        out["aborted"] = f"free RAM {r:.2f} GiB < {args.min_free_ram_gib} at {where}"
        dump(); print("[abort]", out["aborted"], flush=True); sys.exit(3)
    return r


dev = torch.device("cuda")

# ---------------------------------------------------------------- scan check
# Op-level count of the eager reference scan (DeltaNetBlock._scan_sequential):
# every elementwise op = numel(out) FLOPs, a sum = numel(in). Checks the
# kernel-source constant (8 d^2 per token-head forward) by execution.
class EltCount(TorchDispatchMode):
    def __init__(self):
        super().__init__(); self.n = 0; self.ops = collections.Counter()

    def __torch_dispatch__(self, func, types, a=(), kw=None):
        r = func(*a, **(kw or {}))
        name = func._overloadpacket.__name__
        if name in ("mul", "add", "sub", "div"):
            self.n += r.numel(); self.ops[name] += r.numel()
        elif name == "sum":
            self.n += a[0].numel(); self.ops[name] += a[0].numel()
        return r


Nr, Sr, Dr = 4, 16, 32
qr, kr, vr = (torch.randn(Nr, Sr, Dr, device=dev) for _ in range(3))
ar = torch.rand(Nr, Sr, Dr, device=dev); br = torch.rand(Nr, Sr, 1, device=dev)
with EltCount() as ec:
    hdn.DeltaNetBlock._scan_sequential(qr, kr, vr, ar, br)
ref_per_tok = ec.n / (Nr * Sr)
out["scan_ref_fwd_flops_per_token_head"] = ref_per_tok
out["scan_ref_fwd_ops"] = dict(ec.ops)
print(f"[scan] eager reference fwd, op-level count: {ref_per_tok:.0f} FLOPs/token/head "
      f"= {ref_per_tok/Dr**2:.3f} d^2  (roofline formula: 8 d^2 = {8*Dr*Dr})", flush=True)

# Kernel-source counts per token per head, state D x D, _MegaScan.block = Bk:
#  fwd (_gated_delta_fwd_kernel_s): decay D^2, S k 2D^2, rank-1 3D^2, read 2D^2 = 8 D^2
#  bwd: ckpt pass 6 D^2 + in-block replay (Bk-1)/2 * 6 D^2 + adjoint body 21 D^2
Bk = hdt._MegaScan.block
SCAN_FWD = 8
SCAN_BWD_ADJ = 21
SCAN_BWD_RECOMP = 6 + (Bk - 1) / 2 * 6
out["scan_consts_d2"] = {"fwd": SCAN_FWD, "bwd_adjoint": SCAN_BWD_ADJ,
                         "bwd_internal_recompute": SCAN_BWD_RECOMP, "block": Bk}

# ---------------------------------------------------------------- phase tags
PHASE = ["none"]
RECOMP = [False]


def make_tagged(orig):
    def tagged_checkpoint(fn, *a, **kw):
        calls = [0]

        def wrapped(*aa, **kk):
            calls[0] += 1
            prev = RECOMP[0]
            RECOMP[0] = prev or calls[0] > 1
            try:
                return fn(*aa, **kk)
            finally:
                RECOMP[0] = prev
        return orig(wrapped, *a, **kw)
    return tagged_checkpoint


hem.checkpoint = make_tagged(hem.checkpoint)
hcl.checkpoint = make_tagged(hcl.checkpoint)


def bucket():
    if PHASE[0] == "opt":
        return "opt"
    if RECOMP[0]:
        return "recompute"
    return PHASE[0]


SCAN = collections.Counter()          # (direction, bucket) -> sum of N*S*D^2
SCAN_CALLS = collections.Counter()
COUNTING = [False]
_f, _b = hdt._MegaScanBSHD.forward, hdt._MegaScanBSHD.backward


def _scan_fwd(ctx, q, k, v, alpha, beta):
    Bt, S_, H, D = q.shape
    ctx._mfu_nsd2 = Bt * H * S_ * D * D   # saved_tensors may be unpacked only once
    if COUNTING[0]:
        SCAN[("fwd", bucket())] += Bt * H * S_ * D * D
        SCAN_CALLS[("fwd", bucket())] += 1
    return _f(ctx, q, k, v, alpha, beta)


def _scan_bwd(ctx, do):
    if COUNTING[0]:
        SCAN[("bwd", bucket())] += ctx._mfu_nsd2
        SCAN_CALLS[("bwd", bucket())] += 1
    return _b(ctx, do)


hdt._MegaScanBSHD.forward = staticmethod(_scan_fwd)
hdt._MegaScanBSHD.backward = staticmethod(_scan_bwd)


class BucketCounter(FlopCounterMode):
    def __init__(self):
        super().__init__(display=False)
        self.buckets = collections.Counter()
        self.by_op = collections.Counter()
        self.unreg = collections.Counter()

    # ModuleTracker (per-module attribution) is NOT entered: under grad
    # checkpointing + offload it drove allocated memory to 12 GiB and OOMed
    # (first attempt, 2026-10-15). Without it every op lands only in "Global".
    def __enter__(self):
        from torch.utils.flop_counter import _FlopCounterMode
        self.flop_counts.clear()
        self.mode = _FlopCounterMode(self)
        self.mode.__enter__()
        return self

    def __exit__(self, *a):
        b = self.mode.__exit__(*a)
        self.mode = None
        return b

    def _count_flops(self, func_packet, out_, a, kw):
        if func_packet in self.flop_registry:
            f = self.flop_registry[func_packet](*a, **kw, out_val=out_)
            b = bucket()
            self.buckets[b] += f
            self.by_op[(b, func_packet.__name__)] += f
        else:
            self.unreg[func_packet.__name__] += 1
        return super()._count_flops(func_packet, out_, a, kw)

# ---------------------------------------------------------------- build
GA, SEQ, VOCAB, CHUNK = 8, 4096, 50257, 512
torch.manual_seed(1234); torch.cuda.manual_seed_all(1234)
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
from hvn.edge.ternary import enable_ternary_step_cache
out["step_cache_modules"] = enable_ternary_step_cache(model, True)
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


def one_step():
    loss_acc = torch.zeros((), device=dev)
    for mb in range(GA):
        x = XS[mb].to(dev, non_blocking=True); y = YS[mb].to(dev, non_blocking=True)
        PHASE[0] = "fwd"
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(x, return_hidden=True)
            loss = hcl.chunked_cross_entropy(hidden, y, model.decoder, chunk=CHUNK).mean() / GA
        PHASE[0] = "bwd"
        loss.backward()
        loss_acc += loss.detach()
    PHASE[0] = "opt"
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step(); opt.zero_grad(); sched.step()
    PHASE[0] = "none"
    return loss_acc


for s in range(args.warmup):
    a = sched.active_block_name
    torch.cuda.synchronize(); w0 = time.perf_counter()
    lv = one_step().item()
    print(f"[warm {s}] {a:<34} wall {time.perf_counter()-w0:6.2f} loss {lv:.6f}", flush=True)
    guard("warm")
torch.cuda.reset_peak_memory_stats()

for c in range(args.cycles_a):
    for s in range(NB):
        a = sched.active_block_name
        torch.cuda.synchronize(); w0 = time.perf_counter()
        lv = one_step().item()
        torch.cuda.synchronize(); wall = time.perf_counter() - w0
        r = {"cycle": c, "active": a, "wall": wall, "loss": lv,
             "peak_reserved": torch.cuda.max_memory_reserved(), "ram": guard("A")}
        out["steps_a"].append(r); dump()
        print(f"[A{c} {s:2d}] {a:<34} wall {wall:6.2f} loss {lv:.6f} "
              f"rsv {r['peak_reserved']/2**30:.2f}G", flush=True)
        torch.cuda.reset_peak_memory_stats()
wall_a = [r["wall"] for r in out["steps_a"]]
out["wall_mean"] = sum(wall_a) / max(1, len(wall_a))
print(f"[A] cycle-mean wall {out['wall_mean']:.3f} s/step over {len(wall_a)} steps", flush=True)

# ---------------------------------------------------------------- count cycle
if args.skip_count:
    print("[done]", flush=True); sys.exit(0)
tot = collections.Counter(); scan_tot = collections.Counter()
by_op_tot = collections.Counter(); unreg_tot = collections.Counter()
for s in range(NB):
    a = sched.active_block_name
    SCAN.clear(); SCAN_CALLS.clear()
    fc = BucketCounter()
    COUNTING[0] = True
    torch.cuda.reset_peak_memory_stats()
    w0 = time.perf_counter()
    with fc:
        lv = one_step().item()
    torch.cuda.synchronize()
    COUNTING[0] = False
    glob = fc.get_total_flops()
    bsum = sum(fc.buckets.values())
    assert glob == bsum, f"bucket sum {bsum} != Global {glob}"
    # summing the per-module table double counts (#131 analog) -- recorded to show it
    permod = sum(sum(v.values()) for v in fc.get_flop_counts().values())
    r = {"active": a, "wall_counted": time.perf_counter() - w0, "loss": lv,
         "aten_global": glob, "aten_buckets": dict(fc.buckets),
         "per_module_table_sum": permod, "peak_alloc": torch.cuda.max_memory_allocated(),
         "scan_NSD2": {f"{k[0]}|{k[1]}": v for k, v in SCAN.items()},
         "scan_calls": {f"{k[0]}|{k[1]}": v for k, v in SCAN_CALLS.items()},
         "ram": guard("F")}
    out["steps_f"].append(r)
    tot.update(fc.buckets); scan_tot.update(SCAN)
    by_op_tot.update(fc.by_op); unreg_tot.update(fc.unreg)
    out["by_op"] = {f"{k[0]}|{k[1]}": v for k, v in by_op_tot.items()}
    out["unregistered_ops"] = dict(unreg_tot)
    dump()
    print(f"[F {s:2d}] {a:<34} aten {glob/1e12:7.2f} TF ("
          + " ".join(f"{k}={v/1e12:.2f}" for k, v in sorted(fc.buckets.items()))
          + f") peak alloc {torch.cuda.max_memory_allocated()/2**30:.2f}G; scan calls "
          + " ".join(f"{k}={v}" for k, v in sorted(r["scan_calls"].items())), flush=True)
    del fc

# ---------------------------------------------------------------- summary
scan_exec = scan_model = 0.0
for (direction, b), nsd2 in scan_tot.items():
    if direction == "fwd":
        scan_exec += SCAN_FWD * nsd2
        if b != "recompute":
            scan_model += SCAN_FWD * nsd2
    else:
        scan_exec += (SCAN_BWD_ADJ + SCAN_BWD_RECOMP) * nsd2
        scan_model += SCAN_BWD_ADJ * nsd2
sdpa = sum(v for k, v in by_op_tot.items() if "scaled_dot_product" in k[1])
sdpa_model = sum(v for k, v in by_op_tot.items()
                 if "scaled_dot_product" in k[1] and k[0] != "recompute")
aten_exec = sum(tot.values())
aten_model = aten_exec - tot.get("recompute", 0)
summ = {
    "aten_exec_per_step": aten_exec / NB,
    "aten_model_per_step": aten_model / NB,
    "sdpa_counted_full_square_per_step": sdpa / NB,
    "scan_exec_per_step": scan_exec / NB,
    "scan_model_per_step": scan_model / NB,
    "exec_total_per_step_causal_halved": (aten_exec + scan_exec - sdpa / 2) / NB,
    "model_total_per_step_causal_halved": (aten_model + scan_model - sdpa_model / 2) / NB,
    "aten_buckets_per_step": {k: v / NB for k, v in tot.items()},
    "scan_NSD2_cycle": {f"{k[0]}|{k[1]}": v for k, v in scan_tot.items()},
}
out["summary"] = summ
dump()
print("\n[sum] per step, averaged over one full BCD cycle:")
for k, v in summ.items():
    if isinstance(v, (int, float)):
        print(f"[sum]  {k:<40} {v/1e12:9.3f} TFLOP")
print("[sum]  aten buckets: " + " ".join(f"{k}={v/1e12:.3f}" for k, v in summ["aten_buckets_per_step"].items()))
print("[sum] top counted ops:")
for k, v in sorted(by_op_tot.items(), key=lambda kv: -kv[1])[:15]:
    print(f"[sum]   {k[0]:<10} {k[1]:<50} {v/NB/1e12:8.3f} TF/step")
sus = {k: v for k, v in unreg_tot.items()
       if any(t in k for t in ("mm", "conv", "matmul", "dot", "attention", "einsum", "linear"))}
print(f"[sum] unregistered ops that look matmul-like: {sus}")
print("[done]", flush=True)
