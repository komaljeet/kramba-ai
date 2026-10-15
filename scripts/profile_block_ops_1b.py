"""Per-op kernel time of ONE 1B DeltaNet / attention block, fwd + ckpt-recompute + bwd.

Author: Rambarun Komaljeet

Clean, single-block companion to scripts/sweep_1b_ops.py (whose per-op pass is
confounded by micro-batch 0 effects and BCD loads). Shape and execution mode
are the real 1B step's: d 2048, 64 heads, seq 4096, batch 1, bf16 autocast,
fp32 residual stream, ternary + fused quantizer + step cache, --fast scan flags,
gradient checkpointing (so the forward runs twice). Every CUDA kernel's time is
charged to the innermost CPU op that launched it (plus the autograd node for
backward), so rows sum to the kernel total -- printed as a cross-check.
"""
import argparse
import collections
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--block", choices=["dn", "attn"], default="dn")
ap.add_argument("--iters", type=int, default=5)
args = ap.parse_args()
sys.path.insert(0, args.repo)
import pyarrow  # noqa: F401,E402
import torch  # noqa: E402
from torch.autograd import DeviceType  # noqa: E402
from torch.profiler import profile, ProfilerActivity  # noqa: E402
from torch.utils.checkpoint import checkpoint  # noqa: E402

from hvn.core.deltanet import DeltaNetBlock  # noqa: E402
from hvn.core.transformer_block import TransformerBlock  # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model, enable_ternary_step_cache, TernaryLinear  # noqa: E402

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
D, H, S = 2048, 64, 4096
torch.manual_seed(0)
if args.block == "dn":
    blk = DeltaNetBlock(D, n_heads=H, dropout=0.1, use_triton_scan=True, use_megakernel_scan=True)
else:
    blk = TransformerBlock(D, n_heads=H, dropout=0.1, ffn_expansion=3, window=4096)
wrap = torch.nn.Sequential(blk)
apply_ternary_to_model(wrap, [n for n, m in wrap.named_modules()
                              if isinstance(m, torch.nn.Linear)
                              and not any(t in n for t in ("beta_proj", "alpha_proj"))])
for m in wrap.modules():
    if isinstance(m, TernaryLinear):
        m.use_fused = True
enable_ternary_step_cache(wrap, True)
blk = wrap[0].cuda().train()
x = torch.randn(1, S, D, device="cuda", requires_grad=True)
pos = torch.arange(S, device="cuda").unsqueeze(0)


def step():
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        y = checkpoint(blk, x, pos, use_reentrant=False)
    y.float().pow(2).mean().backward()


for _ in range(3):
    step()
torch.cuda.synchronize()
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
    for _ in range(args.iters):
        step()
    torch.cuda.synchronize()

KT = collections.Counter()
for e in prof.key_averages():
    d = getattr(e, "self_device_time_total", 0) or 0
    if d > 0 and e.device_type == DeviceType.CUDA and "#" not in e.key:
        KT[e.key] += d
tot = sum(KT.values())


def node(e):
    p = e.cpu_parent
    while p is not None:
        if p.name.startswith("autograd::engine::evaluate_function:"):
            return "bwd:" + p.name.split(":", 4)[-1].strip()
        p = p.cpu_parent
    return "fwd"


OP = collections.Counter()
N = collections.Counter()
for e in prof.events():
    if e.device_type == DeviceType.CUDA or not e.kernels:
        continue
    k = (node(e), e.name, str(e.input_shapes[:2]) if e.input_shapes else "")
    OP[k] += sum(kk.duration for kk in e.kernels)
    N[k] += 1
print(f"kernel total {tot/args.iters/1e3:.2f} ms/iter; op-attributed {sum(OP.values())/args.iters/1e3:.2f} ms/iter")
print("top kernels (ms/iter):")
for k, v in KT.most_common(30):
    print(f"  {v/args.iters/1e3:7.3f}  {k[:120]}")
print("top ops (ms/iter, calls/iter):")
for k, v in OP.most_common(60):
    print(f"  {v/args.iters/1e3:7.3f} {N[k]/args.iters:4.0f}  {k[0][:34]:<34} {k[1]:<34} {k[2][:70]}")
