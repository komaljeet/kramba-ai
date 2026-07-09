"""
Isolate the train-loop hot-path op that drops HVN 0.81 -> 0.4 steps/s.

Ternary is ruled out (+7ms). The real loop's extra per-micro-batch work vs my
minimal loop: per-source loss tracking via a SYNCHRONOUS batch_src.to(device)
(non_blocking=False, line ~1394) + scatter_add, plus clip_grad_norm_ per step.
A sync H2D every micro-batch forces a device sync, serializing the 8 micro-
batches and killing async overlap -> a clean ~2x slowdown.

Uses the REAL DataLoader. Three variants, same compiled model + --fast:
  full         : replicate the real hot path (sync batch_src.to + scatter + clip) -> expect ~0.4
  no_src       : skip the per-source block entirely (don't touch batch_src)      -> expect ~0.8 if sync is the cause
  async_src    : per-source block but batch_src.to(non_blocking=True)            -> expect ~0.8 if the FIX works

    python scripts/isolate_trainloop.py
"""
from __future__ import annotations

import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1"); os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
_PRIMARY = os.path.dirname(os.path.dirname(os.path.dirname(_ROOT)))
from hvn.edge.model import EdgeHVN
import train_edge_conversational as T

D_MODEL, NUM_LAYERS, N_HEADS = 512, 12, 8
BATCH, SEQ, GRAD_ACCUM, VOCAB = 4, 512, 8, 50257
DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


def _loader():
    loader, _, _ = T.build_dataset_pipeline(
        tokenizer=None, seq_len=SEQ, batch_size=BATCH, smoke=False,
        data_dir=os.path.join(_PRIMARY, "data_cache"), val_fraction=0.0, num_workers=4)
    return loader


def _build():
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=VOCAB, d_model=D_MODEL, num_layers=NUM_LAYERS, n_heads=N_HEADS,
                use_deltanet=True, deltanet_conv_kernel=4, use_triton_scan=True).to(DEV)
    return torch.compile(m)


def measure(variant: str, n_warmup=8, n_time=15):
    print("\n" + "=" * 72)
    print(f"TRAINLOOP ISOLATE — variant={variant}")
    print("=" * 72)
    loader = _loader()
    model = _build()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, fused=True)
    it = iter(loader)
    src_loss_t = torch.zeros(3, device=DEV)
    src_count_t = torch.zeros(3, device=DEV)

    def step():
        opt.zero_grad(set_to_none=True)
        for _ in range(GRAD_ACCUM):
            batch = next(it)
            bx, by, bsrc = batch
            bx = bx.to(DEV, non_blocking=True)
            by = by.to(DEV, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(bx)
                tok_loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, VOCAB), by.reshape(-1), reduction="none")
                loss = tok_loss.mean() / GRAD_ACCUM
            loss.backward()
            if variant == "full":
                with torch.no_grad():
                    per_sample = tok_loss.detach().reshape(bx.shape[0], -1).mean(dim=1)
                    sids = bsrc.to(device=DEV, dtype=torch.long)  # SYNC H2D (non_blocking=False)
                    src_loss_t.scatter_add_(0, sids, per_sample.to(src_loss_t.dtype))
                    src_count_t.scatter_add_(0, sids, torch.ones_like(sids, dtype=src_count_t.dtype))
            elif variant == "async_src":
                with torch.no_grad():
                    per_sample = tok_loss.detach().reshape(bx.shape[0], -1).mean(dim=1)
                    sids = bsrc.to(device=DEV, dtype=torch.long, non_blocking=True)  # ASYNC fix
                    src_loss_t.scatter_add_(0, sids, per_sample.to(src_loss_t.dtype))
                    src_count_t.scatter_add_(0, sids, torch.ones_like(sids, dtype=src_count_t.dtype))
            # variant == "no_src": skip per-source entirely
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
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
    del model, opt, loader, it
    torch.cuda.empty_cache()
    return sps, ms


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    full_s, _ = measure("full")
    no_s, _ = measure("no_src")
    async_s, _ = measure("async_src")
    print("\n" + "=" * 72)
    print("RESULT — train-loop hot-path isolation (real DataLoader, compiled)")
    print("=" * 72)
    print(f"  full (sync batch_src.to + scatter + clip) : {full_s:.2f} steps/s   (real script behaviour)")
    print(f"  no_src (skip per-source block)            : {no_s:.2f} steps/s")
    print(f"  async_src (batch_src.to non_blocking=True): {async_s:.2f} steps/s   (the fix)")
    print("=" * 72)