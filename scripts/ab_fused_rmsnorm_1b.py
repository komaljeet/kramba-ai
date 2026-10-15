"""1B A/B for the fused block RMSNorm (2026-10-15).

Author: Rambarun Komaljeet

Build and step are scripts/sweep_1b_ops.py's (the --fast stack exactly as
main() passes it: triton + megakernel scan, TF32, fused ternary, ternary step
cache; hvn-hybrid 2048x24, attn [6,18], tied embeddings, chunked loss 512,
Muon, grad-ckpt, balanced BCD + offload + bf16 parking + prefetch), with only
``fused_rmsnorm`` differing between arms.

Per run: 2 warmup steps, then ``--cycles`` full BCD cycles (13 steps each, every
block active once -- step time ranges ~8-22 s by active block, so only whole
cycles are comparable). Records per-step wall, loss (and its float hex, for an
exact on/off comparison), peak reserved VRAM.

Falsifiability: ``F.rms_norm`` is wrapped with a call counter; the "on" arm
must show calls and the "off" arm none, so an arm that silently took the other
path voids the run instead of reporting a number.

Usage::

    python -u scripts/ab_fused_rmsnorm_1b.py --rmsnorm on --out results/final_sweep_1b/ab_on_1.json
"""
import argparse
import json
import os
import sys
import time

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", required=True)
ap.add_argument("--rmsnorm", choices=["on", "off"], required=True)
ap.add_argument("--warmup", type=int, default=2)
ap.add_argument("--cycles", type=int, default=2)
args = ap.parse_args()

sys.path.insert(0, args.repo)
import pyarrow  # noqa: F401,E402  -- before CUDA (ed4777d)
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

import train_edge_conversational as tec  # noqa: E402
import hvn.optimizers as hopt  # noqa: E402
import hvn.edge.chunked_loss as hcl  # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402

CALLS = [0]
_orig = F.rms_norm


def _counted(*a, **k):
    CALLS[0] += 1
    return _orig(*a, **k)


F.rms_norm = _counted

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
    precision="bf16", lr=3e-4, vocab_size=VOCAB, offload_weights=True)
model = tec.apply_fast_stack(model, triton_scan=True, megakernel_scan=True, tf32=True,
                             compile_model=False, verbose=False, fused_ternary=True,
                             ternary_step_cache=True,
                             fused_rmsnorm=(args.rmsnorm == "on"))
model.train()
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
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    opt.zero_grad()
    sched.step()
    return loss_acc


out = {"config": vars(args), "steps": []}
for s in range(args.warmup):
    a = sched.active_block_name
    lv = one_step().item()
    out["steps"].append({"phase": "warm", "active": a, "loss": lv, "loss_hex": float(lv).hex()})
torch.cuda.synchronize()
torch.cuda.reset_peak_memory_stats()
calls0 = CALLS[0]
for c in range(args.cycles):
    for s in range(NB):
        a = sched.active_block_name
        torch.cuda.synchronize()
        w0 = time.perf_counter()
        lv = one_step().item()
        torch.cuda.synchronize()
        wall = time.perf_counter() - w0
        out["steps"].append({"phase": f"cycle{c}", "active": a, "wall": wall, "loss": lv,
                             "loss_hex": float(lv).hex()})
        print(f"[{args.rmsnorm} c{c} {s:2d}] {a:<34} wall {wall:6.2f} loss {lv:.6f}", flush=True)
calls = CALLS[0] - calls0
out["rms_norm_calls_timed"] = calls
out["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2 ** 20
out["peak_alloc_mib"] = torch.cuda.max_memory_allocated() / 2 ** 20
walls = [r["wall"] for r in out["steps"] if "wall" in r]
out["mean_s_per_step"] = sum(walls) / len(walls)
out["cycle_means"] = [sum(walls[i * NB:(i + 1) * NB]) / NB for i in range(args.cycles)]
with open(args.out, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(f"[{args.rmsnorm}] mean {out['mean_s_per_step']:.3f} s/step, cycles "
      f"{[round(v, 3) for v in out['cycle_means']]}, rms_norm calls {calls}, "
      f"peak reserved {out['peak_reserved_mib']:.0f} MiB", flush=True)
if (args.rmsnorm == "on") != (calls > 0):
    print(f"[VOID] arm {args.rmsnorm} saw {calls} fused rms_norm calls", flush=True)
    sys.exit(4)
