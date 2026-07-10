"""
Decisive: is triton-scan the 0.4 vs 0.8 lever?

All other suspects ruled out (ternary +7ms; per-source sync / clip / DataLoader
all 0.81). Remaining hypothesis: the real script's forward is NOT actually using
the triton scan (falls back to eager chunk-32 scan -> ~0.4), while my replications
DO use triton -> 0.81. Same compiled minimal loop, only use_triton_scan differs.

Also prints _HAS_TRITON_SCAN so we know whether triton is even importable here.

    python scripts/isolate_triton.py
"""
from __future__ import annotations
import os, sys, time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import hvn.core.deltanet as _dn
from hvn.edge.model import EdgeHVN

D_MODEL, NUM_LAYERS, N_HEADS = 512, 12, 8
BATCH, SEQ, GRAD_ACCUM, VOCAB = 4, 512, 8, 50257
DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

print(f"_HAS_TRITON_SCAN = {_dn._HAS_TRITON_SCAN}")
print(f"triton scan fn = {_dn._triton_gated_delta_scan}")


def _build(triton: bool):
    torch.manual_seed(0)
    return EdgeHVN(vocab_size=VOCAB, d_model=D_MODEL, num_layers=NUM_LAYERS, n_heads=N_HEADS,
                   use_deltanet=True, deltanet_conv_kernel=4, use_triton_scan=triton).to(DEV)


def measure(triton: bool, n_warmup=8, n_time=20):
    label = "triton-scan ON" if triton else "triton-scan OFF (eager chunk-32)"
    print("\n" + "=" * 72)
    print(f"ISOLATE — {label}  (compiled, bf16, grad_accum=8)")
    print("=" * 72)
    model = torch.compile(_build(triton))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, fused=True)
    inps = [torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV) for _ in range(GRAD_ACCUM)]
    tgts = [torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV) for _ in range(GRAD_ACCUM)]

    def step():
        opt.zero_grad(set_to_none=True)
        for i in range(GRAD_ACCUM):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inps[i])
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), tgts[i].reshape(-1))
            (loss / GRAD_ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

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
    on_s, on_m = measure(True)
    off_s, off_m = measure(False)
    print("\n" + "=" * 72)
    print("RESULT — triton-scan isolation (compiled, same loop)")
    print("=" * 72)
    print(f"  triton-scan ON  : {on_s:.2f} steps/s  ({on_m:.0f} ms/step)")
    print(f"  triton-scan OFF : {off_s:.2f} steps/s  ({off_m:.0f} ms/step)")
    print(f"  triton speedup  : {on_m/off_m:.2f}x  (OFF should match the real script's ~0.4 if triton is the cause)")
    print("=" * 72)