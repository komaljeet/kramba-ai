"""#203: what does ONE frozen DeltaNet layer cost at the 1B shapes, and why?

Author: Rambarun Komaljeet

Builds the trainer's own 1B layer (``build_model_and_config`` + ``apply_fast_stack``
with the ``--fast --fp8-linear`` stack: ternary, fused ternary, step cache, fused
RMSNorm, Triton + megakernel scan, fp8 GEMMs), d 2048, 64 heads, seq 4096, batch 1,
dropout 0, bf16-parked weights streamed from CPU by the real BCD offload pool, grad
checkpointing, ``--deterministic`` settings. Only the layer COUNT differs from
production: 6 layers (L5 = attention, as hybrid needs one), 2 layers per BCD block:

    blocks: [emb] [L0 L1] [L2 L3] [L4 L5 tail]
    --active 1 : L0 active (no input grad), L1 active (input grad),
                 L2 L3 L4 frozen AFTER the active block (forward, recompute, dX only)
    --active 3 : L0..L3 frozen BEFORE the active block (forward only)

The measured layers are interior ones (frozen neighbours on both sides), so the
weight prefetch pattern is production's.

Timing: CUDA events at block boundaries (as scripts/profile_step_202.py), plus a
pre-hook on ``ffn[3]`` that marks the end of a frozen block's checkpoint recompute
(non-reentrant checkpoint stops early after the last saved tensor, which ffn[3]
saves, so the block's own post-hook never fires there). Every event is paired with
a one-thread marker kernel ``_phase_marker_203`` on the same stream, so a profiler
trace can be cut into the same phases and each phase's kernels classed.

ANCHOR (falsifier for the whole harness): the production #202 profile gives, per
step of 8 micro-batches, frozen layer forward 0.156 s, frozen-after recompute +
backward 0.400 s, active layer recompute + backward 0.534 s (0.522 s for the first
active layer). If this harness's per-layer numbers are not within ~5% of those, its
breakdown does not describe production and must not be used.
"""
import argparse, collections, hashlib, json, os, sys, time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", required=True)
ap.add_argument("--active", type=int, default=1, help="BCD block index to activate")
ap.add_argument("--steps", type=int, default=3, help="optimizer steps of GA micro-batches")
ap.add_argument("--ga", type=int, default=8)
ap.add_argument("--cache-blocks", type=int, default=1,
                help="offload cache size; >= #blocks keeps frozen weights resident (no streaming)")
ap.add_argument("--profile", action="store_true", help="kernel trace of the last step's micro-batch 1")
ap.add_argument("--no-skip-unread", action="store_true", help="disable #195 recompute GEMM skip")
ap.add_argument("--hash", action="store_true", help="sha256 of active grads and loss per step")
ap.add_argument("--set", default="", help="extra knobs, e.g. frozen_dw=1")
args = ap.parse_args()

sys.path.insert(0, args.repo)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
import pyarrow  # noqa: F401  (before CUDA, ed4777d)
import torch
import triton
import triton.language as tl

import train_edge_conversational as tec
import hvn.optimizers as hopt
import hvn.edge.fp8_fused as F8
from hvn.core.block_coordinate import BlockCoordinateScheduler
from hvn.core.transformer_block import set_deterministic_attention

assert os.path.abspath(tec.__file__).startswith(os.path.abspath(args.repo)), tec.__file__
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=False)
from torch.utils import deterministic as _det
_det.fill_uninitialized_memory = False
set_deterministic_attention(True)


@triton.jit
def _phase_marker_203(p, v):
    tl.store(p, v)


dev = torch.device("cuda")
MARKBUF = torch.zeros(1, device=dev, dtype=torch.int32)
EV, LABELS = [], []
S = {"bwd": False, "on": False}


def mark(label):
    if not S["on"]:
        return
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    _phase_marker_203[(1,)](MARKBUF, len(LABELS))
    EV.append(e)
    LABELS.append(label)


torch.manual_seed(0)
SEQ, D, NL = 4096, 2048, 6
settings = {"batch_size": 1, "grad_accum": args.ga, "seq_len": SEQ, "d_model": D,
            "num_layers": NL, "n_heads": 64}
model, _ = tec.build_model_and_config(
    settings, None, dev, checkpoint_dir=os.path.join(os.path.dirname(os.path.abspath(args.out)), "ckpt"),
    arch="hvn-hybrid", apply_ternary=True, attn_layers=[5], attn_window=4096,
    grad_checkpoint=True, tie_embeddings=True, chunked_loss=512, optimizer="muon",
    precision="bf16", lr=3e-4, vocab_size=50257, offload_weights=True, dropout=0.0)
model = tec.apply_fast_stack(model, triton_scan=True, megakernel_scan=True, tf32=True,
                             compile_model=False, verbose=False, fused_ternary=True,
                             ternary_step_cache=True, fused_rmsnorm=True, fp8_linear=True)
model.train()
opt, _ = hopt.build_optimizer(model, "muon", lr=3e-4, weight_decay=0.01, fused=False, muon_lr=0.02)
sched = BlockCoordinateScheduler(
    model, opt, steps_per_block=10 ** 9, offload_inactive_weights=True, device=dev,
    offload_park_dtype=torch.bfloat16, offload_prefetch=True,
    offload_cache_blocks=args.cache_blocks, partition="balanced", layers_per_block=2,
    initial_active_idx=args.active)
print(f"[cfg] blocks {[n for n, _ in sched.blocks]} active {sched.active_block_name}", flush=True)
for kv in filter(None, args.set.split(",")):
    k, v = kv.split("=")
    setattr(model, k, int(v))

active_names = set(sched.active_block_members) if hasattr(sched, "active_block_members") else set()
info = {}
for i, blk in enumerate(model.blocks):
    name = f"blocks.{i}"
    rg = [p.requires_grad for p in blk.parameters()]
    info[name] = {"type": type(blk).__name__, "params": sum(p.numel() for p in blk.parameters()),
                  "requires_grad": f"{sum(rg)}/{len(rg)}"}
print("[cfg] layers", json.dumps(info), flush=True)

if args.no_skip_unread:
    import contextlib
    import hvn.edge.model as EM
    F8.skip_unread_outputs = lambda block: contextlib.nullcontext()

for i, blk in enumerate(model.blocks):
    def bpre(mod, inp, i=i):
        if mod.training:
            mark(("bwd.L%d" if S["bwd"] else "fwd.L%d") % i)

    def bpost(mod, inp, out, i=i):
        if mod.training and not S["bwd"]:
            mark("fwd.glue")
    blk.register_forward_pre_hook(bpre)
    blk.register_forward_hook(bpost)
    if hasattr(blk, "ffn") and isinstance(blk.ffn, torch.nn.Sequential):
        def fpre(mod, inp, i=i):
            if S["bwd"]:
                mark("bwdgrad.L%d" % i)     # recompute of L_i reached ffn[3]: grad part follows
        blk.ffn[3].register_forward_pre_hook(fpre)

g = torch.Generator().manual_seed(0)
XS = torch.randint(0, 50257, (args.ga, 1, SEQ), generator=g)
GR = torch.randn(1, SEQ, D, generator=g).to(dev) * 1e-3


def micro(mb):
    x = XS[mb].to(dev)
    mark("embed")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        h = model(x, return_hidden=True)
    mark("head")
    loss = (h.float() * GR).sum()
    S["bwd"] = True
    loss.backward()
    S["bwd"] = False
    mark("end")
    return loss.detach()


rows = []
peak_frac = 0.0
total = torch.cuda.get_device_properties(0).total_memory
for step in range(args.steps):
    last = step == args.steps - 1
    S["on"] = step >= 1          # step 0 is warm-up (Triton compile, pool buffers)
    EV.clear(); LABELS.clear()
    torch.cuda.synchronize(); t0 = time.perf_counter()
    losses = []
    prof = None
    for mb in range(args.ga):
        if args.profile and last and mb == 1:
            from torch.profiler import profile, ProfilerActivity
            torch.cuda.synchronize()
            prof = profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]); prof.__enter__()
            n0 = len(LABELS)
        losses.append(micro(mb))
        if prof is not None and mb == 1:
            torch.cuda.synchronize(); prof.__exit__(None, None, None)
            prof_labels = LABELS[n0:]
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    gh = None
    if args.hash:
        h_ = hashlib.sha256()
        for n_, p in model.named_parameters():
            if p.grad is not None:
                h_.update(n_.encode()); h_.update(p.grad.detach().float().cpu().numpy().tobytes())
        h_.update(torch.stack(losses).float().cpu().numpy().tobytes())
        gh = h_.hexdigest()
    opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); wall = time.perf_counter() - t0
    ph = collections.Counter()
    for (lab, e0), e1 in zip(zip(LABELS[:-1], EV[:-1]), EV[1:]):
        if lab == "end":
            continue
        ph[lab] += e0.elapsed_time(e1) / 1e3
    pk = torch.cuda.max_memory_reserved() / total
    peak_frac = max(peak_frac, pk)
    r = {"step": step, "wall": wall, "phases": dict(ph), "peak_reserved_frac": pk,
         "loss": float(torch.stack(losses).sum()), "grad_sha": gh}
    rows.append(r)
    print(f"[step {step}] wall {wall:.3f}s peak {pk:.1%} loss {r['loss']:.6f} "
          + (f"sha {gh[:16]} " if gh else "")
          + " ".join(f"{k}={v:.4f}" for k, v in sorted(ph.items())), flush=True)
    torch.cuda.reset_peak_memory_stats()
    if pk > 0.85:
        print("[abort] peak reserved over 85%", flush=True)
        break

out = {"args": vars(args), "info": info, "rows": rows, "peak_reserved_frac": peak_frac}
if prof is not None:
    tp = os.path.splitext(args.out)[0] + "_trace.json"
    prof.export_chrome_trace(tp)
    out["trace"] = tp
    out["trace_labels"] = prof_labels
json.dump(out, open(args.out, "w"), indent=1)
print("[done]", flush=True)
