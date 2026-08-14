"""
End-to-end training-speed A/B vs the param-matched transformer, 500 steps,
with the Design B fused blockwise-recompute backward (#96 --megakernel-scan)
in the mix.

Three arms, identical real-training loop (fwd+bwd+opt, grad_accum=8):
  - transformer : param-matched TransformerBlock (the goalpost)
  - hvn-seq     : DeltaNetBlock + triton-scan (--fast, sequential bwd)
  - hvn-mega    : same + blk.use_megakernel_scan = True (Design B bwd)

Reports the REALIZED end-to-end steps/s over the full 500 steps (incl. the
one-time compile/warmup amortized -- the number a real training launch sees),
plus a steady-state tail (last 100 steps) for reference. Same EdgeHVN shell;
only the block differs (matches A/B #73/#77).

    python scripts/measure_500step_ab.py
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
N_STEPS = 500
TAIL = 100  # steady-state reference window (last N steps)


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


def measure(arch: str):
    print("\n" + "=" * 72)
    tag = {
        "transformer": "--fast: bf16+TF32+compile",
        "hvn-seq": "--fast: bf16+TF32+compile+triton-scan (sequential bwd)",
        "hvn-mega": "--fast: bf16+TF32+compile+triton-scan+megakernel-scan (Design B bwd)",
    }[arch]
    print(f"END-TO-END {N_STEPS} STEPS -- arch={arch}  ({tag})")
    print("=" * 72)
    model = _build(arch)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  params: {n_params:,}  compiling...")
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

    step_ms = []
    torch.cuda.synchronize()
    t_start = time.perf_counter()
    for s in range(N_STEPS):
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        step_ms.append((time.perf_counter() - t0) * 1e3)
        if (s + 1) % 100 == 0:
            done = s + 1
            el = time.perf_counter() - t_start
            print(f"  step {done:4d}/{N_STEPS}  {el:.0f}s elapsed  "
                  f"({done / el:.2f} steps/s so far)")

    total = time.perf_counter() - t_start
    overall_sps = N_STEPS / total
    tail_ms = sorted(step_ms[-TAIL:])
    tail_med = tail_ms[len(tail_ms) // 2]
    tail_sps = 1000.0 / tail_med
    first_med = sorted(step_ms[:20])[10]
    print(f"  done: {N_STEPS} steps in {total:.0f}s")
    print(f"  end-to-end (incl compile amortized): {overall_sps:.2f} steps/s")
    print(f"  steady-state tail (last {TAIL}, median):  {tail_sps:.2f} steps/s "
          f"({tail_med:.0f} ms/step)  [first-20 median {first_med:.0f} ms]")
    del model, opt
    torch.cuda.empty_cache()
    return {"overall_sps": overall_sps, "tail_sps": tail_sps,
            "tail_ms": tail_med, "total_s": total}


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    _enable_fast_matmuls()
    print(f"config: d={D_MODEL} L={NUM_LAYERS} H={N_HEADS} B={BATCH} S={SEQ} "
          f"grad_accum={GRAD_ACCUM}  (effective batch {BATCH*GRAD_ACCUM})  "
          f"steps={N_STEPS}")

    res = {}
    for arch in ("hvn-seq", "hvn-mega", "transformer"):
        res[arch] = measure(arch)

    print("\n" + "=" * 72)
    print(f"RESULT -- end-to-end {N_STEPS}-step HVN vs Transformer")
    print("=" * 72)
    print(f"  {'arch':12s}  {'end-to-end':>10s}  {'tail':>8s}   {'wall':>6s}")
    for arch in ("hvn-seq", "hvn-mega", "transformer"):
        r = res[arch]
        print(f"  {arch:12s}  {r['overall_sps']:8.2f} sps  "
              f"{r['tail_sps']:6.2f} sps   {r['total_s']:5.0f}s")
    tx_o = res["transformer"]["overall_sps"]
    seq_o = res["hvn-seq"]["overall_sps"]
    mega_o = res["hvn-mega"]["overall_sps"]
    tx_t = res["transformer"]["tail_sps"]
    seq_t = res["hvn-seq"]["tail_sps"]
    mega_t = res["hvn-mega"]["tail_sps"]
    print()
    print(f"  HVN-seq  / Transformer  = {seq_o / tx_o:.2f}x  "
          f"(tail {seq_t / tx_t:.2f}x)   [current --fast]")
    print(f"  HVN-mega / Transformer  = {mega_o / tx_o:.2f}x  "
          f"(tail {mega_t / tx_t:.2f}x)   [after Design B]")
    print(f"  Design B step lift      = {mega_o / seq_o:.3f}x overall  "
          f"({mega_t / seq_t:.3f}x tail)")
    print("=" * 72)