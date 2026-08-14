"""
Fair steady-state HVN-vs-Transformer throughput, now with the Design B
fused blockwise-recompute backward (#96, --megakernel-scan) in the mix.

Three arms, SAME warmup-excluded methodology (so they are directly
comparable):
  - transformer : param-matched TransformerBlock (the goalpost)
  - hvn-seq     : DeltaNetBlock + triton-scan (--fast path, sequential bwd)
  - hvn-mega    : same, but blk.use_megakernel_scan = True (Design B bwd)

This answers two questions at once:
  1. the current HVN/Transformer gap (hvn-seq vs transformer)
  2. the gap AFTER Design B (hvn-mega vs transformer) -- did the ~1.05x
     step win from #96 actually move the end-to-end ratio?

Fairness (matches the A/B #73/#77 + scripts/measure_steadystate_ab.py):
  - same EdgeHVN shell (embedding, final_norm, LM head), only the block differs
  - both bf16 + TF32 + torch.compile; grad_accum=8; AdamW(lr=3e-4, wd=0.01)
  - random GPU-resident tokens (loader proven free; identical path for all arms)
  - warmup EXCLUDED: 10 warmup steps (compile + settle), then time 30 steps,
    report per-step median + a flatness check (settled, not still declining).

    python scripts/measure_steadystate_ab_mega.py
"""
from __future__ import annotations

import os
import sys
import time

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from hvn.edge.model import EdgeHVN  # noqa: E402

D_MODEL, NUM_LAYERS, N_HEADS = 512, 12, 8
BATCH, SEQ, GRAD_ACCUM, VOCAB = 4, 512, 8, 50257
DEV = torch.device("cuda")


def _enable_fast_matmuls():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _build(arch: str) -> EdgeHVN:
    torch.manual_seed(0)
    if arch == "transformer":
        return EdgeHVN(
            vocab_size=VOCAB, d_model=D_MODEL, num_layers=NUM_LAYERS, n_heads=N_HEADS,
            use_deltanet=False, block_type="transformer",
        ).to(DEV)
    m = EdgeHVN(
        vocab_size=VOCAB, d_model=D_MODEL, num_layers=NUM_LAYERS, n_heads=N_HEADS,
        use_deltanet=True, deltanet_conv_kernel=4, use_triton_scan=True,
    ).to(DEV)
    if arch == "hvn-mega":
        for blk in m.blocks:
            if hasattr(blk, "use_megakernel_scan"):
                blk.use_megakernel_scan = True
    return m


def measure(arch: str, n_warmup=10, n_time=30):
    print("\n" + "=" * 72)
    tag = {
        "transformer": "--fast: bf16+TF32+compile",
        "hvn-seq": "--fast: bf16+TF32+compile+triton-scan (sequential bwd)",
        "hvn-mega": "--fast: bf16+TF32+compile+triton-scan+megakernel-scan (Design B bwd)",
    }[arch]
    print(f"STEADY-STATE -- arch={arch}  ({tag})")
    print("=" * 72)
    model = _build(arch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}")
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    inps = [torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV, dtype=torch.long)
            for _ in range(GRAD_ACCUM)]
    tgts = [torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV, dtype=torch.long)
            for _ in range(GRAD_ACCUM)]

    def step():
        opt.zero_grad(set_to_none=True)
        for i in range(GRAD_ACCUM):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inps[i])
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB), tgts[i].reshape(-1))
            (loss / GRAD_ACCUM).backward()
        opt.step()

    print(f"  warming up {n_warmup} steps (compile + settle)...")
    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()

    per_step = []
    for _ in range(n_time):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        per_step.append((time.perf_counter() - t0) * 1e3)

    per_step.sort()
    med = per_step[len(per_step) // 2]
    mn = per_step[0]
    mx = per_step[-1]
    sps_med = 1000.0 / med
    early = sorted(per_step[:5])[2]
    late = sorted(per_step[-5:])[2]
    flat = late / early
    print(f"  per-step ms over {n_time} timed steps: min={mn:.0f}  median={med:.0f}  max={mx:.0f}")
    print(f"  steady-state flatness (late/early median): {flat:.3f}  "
          f"({'settled' if 0.9 <= flat <= 1.1 else 'NOT settled -- increase warmup'})")
    print(f"  -> {med:.0f} ms/step   =   {sps_med:.2f} steps/s   (steady-state)")
    del model, opt
    torch.cuda.empty_cache()
    return sps_med, med, flat


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    _enable_fast_matmuls()
    print(f"config: d={D_MODEL} L={NUM_LAYERS} H={N_HEADS} B={BATCH} S={SEQ} "
          f"grad_accum={GRAD_ACCUM}  (effective batch {BATCH*GRAD_ACCUM})")

    res = {}
    for arch in ("hvn-seq", "hvn-mega", "transformer"):
        res[arch] = measure(arch)

    print("\n" + "=" * 72)
    print("RESULT -- fair steady-state HVN vs Transformer (warmup excluded)")
    print("=" * 72)
    for arch in ("hvn-seq", "hvn-mega", "transformer"):
        sps, ms, flat = res[arch]
        print(f"  {arch:12s}: {sps:.2f} steps/s  ({ms:.0f} ms/step)  flat={flat:.3f}")
    tx = res["transformer"][0]
    seq = res["hvn-seq"][0]
    mega = res["hvn-mega"][0]
    print()
    print(f"  HVN-seq  / Transformer = {seq / tx:.2f}x   (current --fast gap)")
    print(f"  HVN-mega / Transformer = {mega / tx:.2f}x   (after Design B)")
    print(f"  Design B step lift     = {mega / seq:.3f}x   (hvn-mega / hvn-seq)")
    print("=" * 72)