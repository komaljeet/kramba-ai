"""
DataLoader-sustain probe — does the loader fall behind over a long run?

Evidence chain:
  - thermal_probe.py: 60 continuous steps with GPU-resident inputs (NO loader) =
    FLAT 0.82 steps/s, no thermal throttling (clock steady 2550 MHz, 60-75C).
  - So GPU compute is genuinely ~1.22s/step stable. The real script's 0.40
    (2.48s/step) must add ~1.26s/step of NON-GPU time.
  - Profiler: ProfilerStep* captured only 1.82s/step of GPU work out of 2.48s
    wall-clock -> ~0.66s/step the GPU is IDLE (not running captured kernels).
  - isolate_trainloop used the real DataLoader but only ran ~23 steps; the
    prefetch buffer (prefetch_factor=2 x 4 workers = 8 batches) covered it ->
    0.81. Over 50+ steps the buffer may drain and the host blocks on
    next(data_iter), idling the GPU -> 0.4.

This probe: real build (ternary + post-hoc triton + compile) + REAL DataLoader
(4 workers), 60 steps, NO per-step cuda.synchronize (matches the real script's
time.time() measurement), reporting:
  - steps/s per 10-step window (watch for degradation as the buffer drains),
  - per-micro-batch next(data_iter) latency p50/p90/max per window.

  - If steps/s degrades 0.82 -> ~0.4 and next() latency grows -> DataLoader
    can't sustain the rate; the fix is a loader-throughput lever (more workers,
    bigger prefetch, or the pipeline is genuinely too slow). REAL optimization.
  - If flat 0.82 and next() stays <1ms -> loader is fine; cause is elsewhere.

    python scripts/dataloader_long_probe.py
"""
from __future__ import annotations

import os, sys, time, statistics
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PRIMARY = os.path.dirname(os.path.dirname(os.path.dirname(_ROOT)))
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
WINDOW = 10


def _loader():
    loader, _, _ = T.build_dataset_pipeline(
        tokenizer=None, seq_len=SEQ, batch_size=BATCH, smoke=False,
        data_dir=os.path.join(_PRIMARY, "data_cache"), val_fraction=0.0, num_workers=4)
    return loader


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    loader = _loader()
    model, _cfg = T.build_model_and_config(
        settings=SETTINGS, tokenizer=None, device=DEV,
        checkpoint_dir=os.path.join(_ROOT, "checkpoints", "dl_long_probe"),
        arch="hvn", apply_ternary=True)
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, fused=True)
    it = iter(loader)

    def micro():
        t_get = time.perf_counter()
        batch = next(it)
        dt_get = (time.perf_counter() - t_get) * 1e3  # ms, host-only (no GPU sync)
        bx, by, _bsrc = batch
        bx = bx.to(DEV, non_blocking=True)
        by = by.to(DEV, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(bx)
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), by.reshape(-1))
        (loss / GRAD_ACCUM).backward()
        return dt_get

    def step(latencies):
        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            latencies.append(micro())
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"\nWarming {N_WARMUP} steps (compile)...")
    for _ in range(N_WARMUP):
        step([])
    torch.cuda.synchronize()

    print(f"\nTimed — {N_TIMED} steps, NO per-step sync (matches real script), window={WINDOW}")
    print(f"{'win':>4} {'steps/s':>8} {'next p50':>9} {'p90':>7} {'max':>8}  (ms)")
    print("-" * 50)
    win_start = time.perf_counter()
    win_lats = []
    for s in range(1, N_TIMED + 1):
        step(win_lats)
        if s % WINDOW == 0:
            elapsed = time.perf_counter() - win_start
            sps = WINDOW / elapsed
            p50 = statistics.median(win_lats)
            p90 = sorted(win_lats)[int(0.9 * len(win_lats)) - 1] if win_lats else 0
            mx = max(win_lats)
            print(f"{s//WINDOW:>4} {sps:>8.2f} {p50:>9.2f} {p90:>7.2f} {mx:>8.2f}")
            win_start = time.perf_counter()
            win_lats = []
    print("=" * 50)
    print("VERDICT:")
    print("  if steps/s drops toward 0.4 and next() latency grows -> LOADER can't sustain")
    print("  if steps/s stays ~0.8 and next() <1ms               -> loader fine; cause elsewhere")
    print("=" * 50)


if __name__ == "__main__":
    main()