"""
Reproduce the reported ~0.4 steps/s HVN 200-step figure, to settle whether
the 4x gap is real or a compile-warmup artifact.

Runs HVN (--fast: compile+triton-scan+bf16+TF32, grad_accum=8) for 200 steps and
reports:
  - WHOLE-RUN average steps/s  (incl. compile warmup) -- the short-benchmark view
  - LAST-50-steps steps/s      -- the true steady-state view
If warmup is the cause: whole-run ~0.4, last-50 ~0.8  -> the 4x was a short-run
artifact and the real gap is 2x. If both ~0.8: the 0.4 came from something else
and the 4x stands.

    python scripts/reproduce_200step.py
"""
from __future__ import annotations

import os, sys, time
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
from hvn.edge.model import EdgeHVN

D_MODEL, NUM_LAYERS, N_HEADS = 512, 12, 8
BATCH, SEQ, GRAD_ACCUM, VOCAB = 4, 512, 8, 50257
DEV = torch.device("cuda")
N_STEPS = 200

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(0)
model = EdgeHVN(vocab_size=VOCAB, d_model=D_MODEL, num_layers=NUM_LAYERS,
                n_heads=N_HEADS, use_deltanet=True, deltanet_conv_kernel=4,
                use_triton_scan=True).to(DEV)
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

print(f"Running {N_STEPS} HVN steps (--fast), timing EVERY step incl. compile warmup...")
t_start = time.perf_counter()
step_times = []
for s in range(N_STEPS):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    step()
    torch.cuda.synchronize()
    step_times.append(time.perf_counter() - t0)
    if (s + 1) % 50 == 0:
        recent = step_times[-50:]
        print(f"  step {s+1:>3}: last-50 steps/s = {1.0/(sum(recent)/len(recent)):.2f}  "
              f"({sum(recent)/len(recent)*1000:.0f} ms/step)")

whole = time.perf_counter() - t_start
last50 = step_times[-50:]
print("\n" + "=" * 72)
print(f"WHOLE-RUN  ({N_STEPS} steps, incl compile warmup): "
      f"{N_STEPS/whole:.2f} steps/s  ({whole/N_STEPS*1000:.0f} ms/step)")
print(f"LAST-50   (steady state):                  "
      f"{1.0/(sum(last50)/len(last50)):.2f} steps/s  ({sum(last50)/len(last50)*1000:.0f} ms/step)")
print("=" * 72)
print("If WHOLE-RUN ~0.4 and LAST-50 ~0.8 -> the 0.4 was compile warmup; real gap is 2x.")
print("If both ~0.8 -> the reported 0.4 came from something else; 4x stands.")