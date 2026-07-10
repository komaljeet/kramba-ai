"""
#96 Design B end-to-end A/B: real EdgeHVN training-step throughput, sequential
backward vs the fused blockwise-recompute backward (megakernel_scan), under the
real --fast config (compile + triton-scan, bf16, TF32).

The FORWARD is the identical `_gated_delta_fwd_kernel` in both paths, so the
loss is bit-identical — this is purely a SPEED comparison (the backward grads
differ ~1e-7, negligible). The micro-bench (scripts/bench_megakernel_bwd.py)
showed ~1.18x on the isolated scan at d=64; this measures the real STEP-level
win (scan is ~60% of the micro-batch wall-clock, so the step win is smaller).

Builds the real A/B pretraining model (d512 L12 H8, B4 S512, grad_accum=8),
compiles each variant, times full training steps (8 fwd+bwd + opt).

    python scripts/ab_megakernel.py
"""
from __future__ import annotations

import os, sys, time
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from hvn.edge.model import EdgeHVN

D_MODEL, NUM_LAYERS, N_HEADS = 512, 12, 8
BATCH, SEQ, GRAD_ACCUM, VOCAB = 4, 512, 8, 50257
DEV = torch.device("cuda")


def _build(megakernel: bool) -> EdgeHVN:
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=VOCAB, d_model=D_MODEL, num_layers=NUM_LAYERS,
                n_heads=N_HEADS, use_deltanet=True, deltanet_conv_kernel=4,
                use_triton_scan=True).to(DEV)
    for blk in m.blocks:
        if hasattr(blk, "use_megakernel_scan"):
            blk.use_megakernel_scan = megakernel
    return m


def _time(megakernel: bool, n_warmup=6, n_time=15):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    m = _build(megakernel)
    m = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=0.01)
    inps = [torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV, dtype=torch.long) for _ in range(GRAD_ACCUM)]
    tgts = [torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV, dtype=torch.long) for _ in range(GRAD_ACCUM)]

    def train_step():
        opt.zero_grad(set_to_none=True)
        for i in range(GRAD_ACCUM):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = m(inps[i])
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), tgts[i].reshape(-1))
            (loss / GRAD_ACCUM).backward()
        opt.step()

    for _ in range(n_warmup):
        train_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_time):
        train_step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_time * 1e3
    del m, opt; torch.cuda.empty_cache()
    return ms, 1000.0 / ms


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    print(f"  config: d={D_MODEL} L={NUM_LAYERS} H={N_HEADS} B={BATCH} S={SEQ} "
          f"bf16+TF32 grad_accum={GRAD_ACCUM} --fast\n")
    print("  [A] sequential backward (current --fast)...")
    ms_a, sps_a = _time(megakernel=False)
    print(f"      {ms_a:7.1f} ms/step  ->  {sps_a:5.2f} steps/s")
    print("  [B] Design B blockwise-recompute backward...")
    ms_b, sps_b = _time(megakernel=True)
    print(f"      {ms_b:7.1f} ms/step  ->  {sps_b:5.2f} steps/s")
    print()
    print(f"  step speedup: {sps_b/sps_a:.3f}x  ({ms_a:.1f} -> {ms_b:.1f} ms/step)")
    if sps_b > sps_a * 1.03:
        print("  VERDICT: Design B wins end-to-end (>=3% faster) — wire it in behind --fast.")
    elif sps_b > sps_a:
        print("  VERDICT: Design B marginally faster end-to-end (<3%) — modest real win.")
    else:
        print("  VERDICT: Design B does NOT win end-to-end (compile/overhead eats the scan win).")


if __name__ == "__main__":
    main()