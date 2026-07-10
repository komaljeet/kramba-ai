"""
Thermal/power-throttling probe — is the 0.4 vs 0.81 gap a run-length artifact?

Every component isolates fast (0.81-0.82): model build, ternary, triton-scan,
compile, DataLoader, per-source, clip, optimizer. Yet the real script is 0.40.
The profiler diff shows IDENTICAL kernel counts but each kernel ~1.2-1.35x
slower per call in the real script, and repro_realtrain.log shows the rate
RAMPING DOWN over time (step 50=0.3 -> step 100=0.4 -> flat). That is the
signature of GPU thermal/power throttling under sustained load.

My isolates time ~20 steps after ~8 warmup (~40s total, cool, high boost clock).
The real script runs 2-9 min sustained (warm, decayed boost clock). This probe
runs the FAST isolate loop (real-build ternary + compile, minimal loop) for 60
CONTINUOUS steps with per-step timing + GPU temp/clock sampling, to see whether
steps/s degrades from ~0.82 toward ~0.4 as the GPU heats up.

  - If it degrades toward 0.4  -> thermal/power throttling; the 0.4-vs-0.81 gap
    is a run-length/measurement artifact, NOT an optimization opportunity. The
    transformer's 1.6 steps/s then needs the same long-run test for a fair ratio.
  - If it stays flat at ~0.82   -> throttling is NOT it; the cause is elsewhere.

    python scripts/thermal_probe.py
"""
from __future__ import annotations

import os, sys, time, subprocess
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import train_edge_conversational as T

D_MODEL, NUM_LAYERS, N_HEADS = 512, 12, 8
BATCH, SEQ, GRAD_ACCUM, VOCAB = 4, 512, 8, T.GPT2_VOCAB_SIZE
DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
SETTINGS = {"d_model": D_MODEL, "num_layers": NUM_LAYERS, "n_heads": N_HEADS,
            "batch_size": BATCH, "grad_accum": GRAD_ACCUM, "seq_len": SEQ}

N_WARMUP = 8
N_TIMED = 60


def _gpu_state():
    """Sample GPU temp + clocks + power via nvidia-smi (one line, parseable)."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.sm,clocks.max.sm,power.draw",
             "--format=csv,noheader,nounits"], text=True, timeout=5).strip()
        temp, sm, smmax, pwr = (float(x) for x in out.split(","))
        return temp, sm, smmax, pwr
    except Exception as e:
        return None, None, None, None


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    model, _cfg = T.build_model_and_config(
        settings=SETTINGS, tokenizer=None, device=DEV,
        checkpoint_dir=os.path.join(_ROOT, "checkpoints", "thermal_probe"),
        arch="hvn", apply_ternary=True)
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
    model = torch.compile(model)
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

    print(f"\nWarming {N_WARMUP} steps (compile)...")
    for _ in range(N_WARMUP):
        step()
    torch.cuda.synchronize()

    print(f"\nTimed run — {N_TIMED} continuous steps, per-step ms + GPU state every 5 steps")
    print(f"{'step':>5} {'ms/step':>8} {'steps/s':>8}   {'tempC':>6} {'SMclk':>6} {'SMmax':>6} {'pwrW':>6}")
    print("-" * 60)
    t_start = time.perf_counter()
    for s in range(1, N_TIMED + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) * 1e3
        if s % 5 == 0 or s <= 3 or s == N_TIMED:
            temp, sm, smmax, pwr = _gpu_state()
            tclk = f"{int(sm)}" if sm else "  -"
            tmax = f"{int(smmax)}" if smmax else "  -"
            tp = f"{temp:.0f}" if temp is not None else "  -"
            pw = f"{pwr:.1f}" if pwr is not None else "  -"
            print(f"{s:>5} {ms:>8.0f} {1000/ms:>8.2f}   {tp:>6} {tclk:>6} {tmax:>6} {pw:>6}")
        else:
            print(f"{s:>5} {ms:>8.0f} {1000/ms:>8.2f}")

    total = time.perf_counter() - t_start
    print("-" * 60)
    print(f"Overall: {N_TIMED} steps in {total:.1f}s = {N_TIMED/total:.2f} steps/s avg")
    print(f"First 10 avg: {sum(0 for _ in [])}  Last 10 vs first 10 shows the throttling trend.")
    print("=" * 60)
    print("VERDICT:")
    print("  if last-10 ms/step >> first-10 ms/step  -> THERMAL/POWER throttling confirmed")
    print("  if flat                              -> throttling NOT it; cause is elsewhere")
    print("=" * 60)


if __name__ == "__main__":
    main()