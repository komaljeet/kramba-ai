"""
Isolate ternary BitNet STE as the cause of the 0.81 -> 0.4 steps/s drop.

Real train_edge_conversational.py applies ternary to 84 Linear layers (STE,
quantizes weights every forward). My earlier minimal loop did NOT -> it measured
0.81 steps/s while the real script measures 0.4. This script runs the SAME
minimal loop with ternary ON vs OFF (everything else identical: --fast, compile,
triton-scan, bf16, TF32, grad_accum=8) to pin the overhead on ternary.

    python scripts/isolate_ternary.py
"""
from __future__ import annotations

import os, sys, time
import torch
import torch.nn as nn

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from hvn.edge.model import EdgeHVN
from hvn.edge.ternary import apply_ternary_to_model

D_MODEL, NUM_LAYERS, N_HEADS = 512, 12, 8
BATCH, SEQ, GRAD_ACCUM, VOCAB = 4, 512, 8, 50257
DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _build(with_ternary: bool) -> EdgeHVN:
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=VOCAB, d_model=D_MODEL, num_layers=NUM_LAYERS,
                n_heads=N_HEADS, use_deltanet=True, deltanet_conv_kernel=4,
                use_triton_scan=True)
    if with_ternary:
        all_lin = {n: mod for n, mod in m.named_modules() if isinstance(mod, nn.Linear)}
        exclude = ("beta_proj", "alpha_proj", "decoder")
        targets = [n for n in all_lin if not any(p in n for p in exclude)]
        replaced = apply_ternary_to_model(m, target_names=targets)
        print(f"  ternary ON: replaced {len(replaced)} Linear layers (STE)")
    else:
        print("  ternary OFF (fp32 weights)")
    return m.to(DEV)


def measure(with_ternary: bool, n_warmup=8, n_time=20):
    label = "HVN + ternary STE" if with_ternary else "HVN (no ternary)"
    print("\n" + "=" * 72)
    print(f"ISOLATE — {label}  (--fast: compile+triton-scan+bf16+TF32, grad_accum=8)")
    print("=" * 72)
    model = _build(with_ternary)
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    inps = [torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV, dtype=torch.long) for _ in range(GRAD_ACCUM)]
    tgts = [torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV, dtype=torch.long) for _ in range(GRAD_ACCUM)]

    def step():
        opt.zero_grad(set_to_none=True)
        for i in range(GRAD_ACCUM):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inps[i])
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), tgts[i].reshape(-1))
            (loss / GRAD_ACCUM).backward()
        opt.step()

    print(f"  warming up {n_warmup} steps (compile)...")
    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_time):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_time * 1e3
    sps = 1000.0 / ms
    print(f"  -> {ms:.0f} ms/step  =  {sps:.2f} steps/s")
    del model, opt
    torch.cuda.empty_cache()
    return sps, ms


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    off_sps, off_ms = measure(with_ternary=False)
    on_sps, on_ms = measure(with_ternary=True)
    print("\n" + "=" * 72)
    print("RESULT — ternary isolation (same loop, only ternary differs)")
    print("=" * 72)
    print(f"  HVN (no ternary) : {off_sps:.2f} steps/s  ({off_ms:.0f} ms/step)")
    print(f"  HVN + ternary STE: {on_sps:.2f} steps/s  ({on_ms:.0f} ms/step)")
    print(f"  ternary overhead : {on_ms - off_ms:+.0f} ms/step  ({on_ms/off_ms:.2f}x slower)")
    print(f"  real script reports 0.4 steps/s with ternary -> matches ON arm if {on_sps:.2f} ~ 0.4")
    print("=" * 72)