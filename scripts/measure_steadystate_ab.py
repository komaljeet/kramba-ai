"""
Fair steady-state HVN-vs-Transformer throughput (issue #95 follow-up).

The 200-step run reported HVN ~0.4 vs Transformer ~1.6 steps/s (a "4x gap"),
but the dataloader probe showed HVN steady-state is really ~0.81 steps/s --
the 0.4 was compile-warmup amortized over a short run. This script measures
BOTH arms at true steady state with the SAME warmup-excluded methodology, so
the real gap is known before any #95 kernel investment.

Fairness (matches the A/B #73/#77):
  - same EdgeHVN shell (embedding, final_norm, LM head), only the block differs
  - HVN: DeltaNetBlock + triton-scan (--fast path); Transformer: TransformerBlock
  - both bf16 + TF32 + torch.compile; grad_accum=8; AdamW(lr=3e-4, wd=0.01)
  - random GPU-resident tokens (loader proven free for HVN; identical path
    for both arms, so no data confound)
  - warmup EXCLUDED: 10 warmup steps (compile + settle), then time 30 steps,
    and report per-step min/median to confirm steady state (flat, not still
    declining).

    python scripts/measure_steadystate_ab.py
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
    if arch == "hvn":
        return EdgeHVN(
            vocab_size=VOCAB, d_model=D_MODEL, num_layers=NUM_LAYERS, n_heads=N_HEADS,
            use_deltanet=True, deltanet_conv_kernel=4, use_triton_scan=True,
        ).to(DEV)
    return EdgeHVN(
        vocab_size=VOCAB, d_model=D_MODEL, num_layers=NUM_LAYERS, n_heads=N_HEADS,
        use_deltanet=False, block_type="transformer",
    ).to(DEV)


def measure(arch: str, n_warmup=10, n_time=30):
    print("\n" + "=" * 72)
    print(f"STEADY-STATE — arch={arch}  (--fast: bf16+TF32+compile"
          f"{'+triton-scan' if arch == 'hvn' else ''})")
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
    # flatness: ratio of last-5 median to first-5 median (1.0 == settled)
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

    hvn_sps, hvn_ms, hvn_flat = measure("hvn")
    tx_sps, tx_ms, tx_flat = measure("transformer")

    print("\n" + "=" * 72)
    print("RESULT — fair steady-state HVN vs Transformer (warmup excluded)")
    print("=" * 72)
    print(f"  HVN         : {hvn_sps:.2f} steps/s  ({hvn_ms:.0f} ms/step)  flat={hvn_flat:.3f}")
    print(f"  Transformer : {tx_sps:.2f} steps/s  ({tx_ms:.0f} ms/step)  flat={tx_flat:.3f}")
    ratio = hvn_sps / tx_sps
    print(f"  HVN / Transformer = {ratio:.2f}x")
    print(f"  (the 200-step run reported ~0.25x; this is the real steady-state ratio.)")
    print("=" * 72)