"""
DataLoader stall probe for the HVN 2x wall-clock goal (issue #95 follow-up).

The scan-share benchmark found GPU compute = 0.81 steps/s but the reported
real-loop = ~0.4 steps/s, implying ~1.26 s/step of data/overhead. That was
INFERRED (GPU-only vs reported). This probe measures it DIRECTLY:

  (A) REAL training loop with the real DataLoader (--fast, workers=4,
      persistent_workers, pin_memory, prefetch=2) vs the GPU-only 0.81
      steps/s. If real-loop ~0.4 -> loader stalls the GPU -> data-pipeline
      fix is a real lever. If real-loop ~0.8 -> the loader overlaps fine and
      the 0.4 figure came from something else (warmup bleed / measurement).

  (B) Loader raw throughput: ms/batch for workers=0 vs 4 (CPU + pin-memory
      transfer to GPU, no model), to characterize the loader's own cost and
      whether workers=4 can keep up with the ~143 ms/micro-batch GPU work.

Uses the REAL MixedLocalDataset (tokenized memmap bins in the primary tree,
all cache hits -> no network). Model = real EdgeHVN d512/L12/H8, --fast.

    python scripts/measure_dataloader.py
"""
from __future__ import annotations

import os
import sys
import time

# Force offline so GPT2TokenizerFast.from_pretrained never hits the network
# (bins exist -> tokenizer is never actually invoked; cache-hit path only).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PRIMARY = os.path.dirname(os.path.dirname(os.path.dirname(_ROOT)))
_DATA_DIR = os.path.join(_PRIMARY, "data_cache")

from hvn.edge.model import EdgeHVN  # noqa: E402
import train_edge_conversational as T  # noqa: E402

D_MODEL, NUM_LAYERS, N_HEADS = 512, 12, 8
BATCH, SEQ, GRAD_ACCUM, VOCAB = 4, 512, 8, 50257
DEV = torch.device("cuda")


def _enable_fast_matmuls():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def _build_loader(num_workers: int):
    """Real mixed-source DataLoader, identical kwargs to build_dataset_pipeline."""
    # tokenizer=None is fine: bins exist -> _tokenize_text_to_bin cache-hits and
    # never invokes the tokenizer; GPT2TokenizerFast falls back to None offline.
    loader, _, stats = T.build_dataset_pipeline(
        tokenizer=None,
        seq_len=SEQ,
        batch_size=BATCH,
        smoke=False,
        data_dir=_DATA_DIR,
        val_fraction=0.0,
        num_workers=num_workers,
    )
    return loader, stats


def measure_real_loop():
    print("\n" + "=" * 72)
    print("PROBE A — real training loop with real DataLoader (--fast, workers=4)")
    print("=" * 72)
    _enable_fast_matmuls()
    loader, stats = _build_loader(num_workers=4)
    print(f"  dataset: {stats.get('source')}  samples={stats.get('num_samples'):,}  "
          f"batches={stats.get('num_batches'):,}")

    model = EdgeHVN(
        vocab_size=VOCAB, d_model=D_MODEL, num_layers=NUM_LAYERS, n_heads=N_HEADS,
        use_deltanet=True, deltanet_conv_kernel=4, use_triton_scan=True,
    ).to(DEV)
    model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)

    it = iter(loader)

    def train_step():
        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            x, y, _src = next(it)              # real batch from the loader
            x = x.to(DEV, non_blocking=True)
            y = y.to(DEV, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB), y.reshape(-1))
            (loss / GRAD_ACCUM).backward()
        opt.step()

    n_warm, n_time = 6, 20
    print(f"  warming up {n_warm} steps (compile + worker spin-up)...")
    for _ in range(n_warm):
        train_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_time):
        train_step()
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    ms = dt / n_time * 1e3
    sps = 1000.0 / ms
    print(f"\n  REAL loop (--fast + DataLoader workers=4):")
    print(f"    {ms:7.1f} ms/step  ->  {sps:5.2f} steps/s   (WITH real data loader)")
    print(f"    GPU-only reference (no loader)           :   0.81 steps/s (1236 ms/step)")
    print(f"    reported 200-step steady-state           : ~0.40 steps/s (2500 ms/step)")
    if sps > 0.70:
        print("  -> loader OVERLAPS the GPU (no stall). The 0.4 figure is NOT the loader;")
        print("     likely compile-warmup bleed or measurement. Data-pipeline fix = low value.")
    elif sps < 0.55:
        stall = ms - 1236
        print(f"  -> loader STALLS the GPU by ~{stall:.0f} ms/step ({100*stall/ms:.0f}% of wall-clock).")
        print("     Data-pipeline fix is a REAL lever for the 2x goal.")
    else:
        print("  -> partial stall; loader is a meaningful but not dominant share.")
    return sps


def measure_loader_raw():
    print("\n" + "=" * 72)
    print("PROBE B — loader raw throughput (CPU + pin-memory transfer, no model)")
    print("=" * 72)
    for w in (0, 4):
        loader, _ = _build_loader(num_workers=w)
        it = iter(loader)
        # warmup
        for _ in range(5):
            x, y, _s = next(it)
            x = x.to(DEV, non_blocking=True)
        torch.cuda.synchronize()
        n = 60
        t0 = time.perf_counter()
        for _ in range(n):
            x, y, _s = next(it)
            x = x.to(DEV, non_blocking=True)
            y = y.to(DEV, non_blocking=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        ms_b = dt / n * 1e3
        print(f"  workers={w}:  {ms_b:6.2f} ms/batch  ({1000/ms_b:6.1f} batches/s)  "
              f"-> 8-batch step = {ms_b*GRAD_ACCUM:6.1f} ms  "
              f"{'(keeps up with 143 ms GPU micro-batch)' if ms_b < 143 else '(SLOWER than 143 ms GPU -> stalls)'}")
        del loader, it
        torch.cuda.empty_cache()


if __name__ == "__main__":
    assert torch.cuda.is_available(), "CUDA required"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"data_dir: {_DATA_DIR}")
    measure_loader_raw()
    measure_real_loop()
    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)