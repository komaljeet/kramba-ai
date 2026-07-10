"""
Verbatim T.train() probe — run the REAL train() loop, nothing replicated.

Every component isolates fast (0.81-0.82): build, ternary, triton, compile,
DataLoader (sustained 60 steps @ 0.81, next() <0.5ms), no-sync measurement,
thermal (flat 0.82, no throttle). Yet the real script = 0.40.

The one thing I have NOT done is call the REAL T.train() verbatim with the real
build + real loader + T.train()'s OWN build_optimizer + the full loop
(loss_accum_t, per-source scatter, warmup LR, scheduler, cross_entropy
reduction="none"). This runs the EXACT code path the real script runs.

  - If T.train() gives ~0.4 -> the loop CODE is the cause; bisect by toggling
    loss_accum_t / per-source / warmup-LR / scheduler / build_optimizer / reduction.
  - If T.train() gives ~0.8 -> something in main() OUTSIDE train() (the audit's
    pre-compile eager forward+backward, Check 2) is the cause.

log_every_steps=10 so steady-state is visible across multiple windows.
max_steps=60 so step 50 includes compile but steps 20-40+ are steady-state.

    python scripts/train_verbatim_probe.py
"""
from __future__ import annotations

import os, sys
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
BATCH, SEQ, GRAD_ACCUM = 4, 512, 8
DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
SETTINGS = {"d_model": D_MODEL, "num_layers": NUM_LAYERS, "n_heads": N_HEADS,
            "batch_size": BATCH, "grad_accum": GRAD_ACCUM, "seq_len": SEQ}

CKPT = os.path.join(_ROOT, "checkpoints", "train_verbatim_probe")


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Real loader (same as main).
    loader, _, _ = T.build_dataset_pipeline(
        tokenizer=None, seq_len=SEQ, batch_size=BATCH, smoke=False,
        data_dir=os.path.join(_PRIMARY, "data_cache"), val_fraction=0.0, num_workers=4)

    # Real build (ternary inside).
    model, config = T.build_model_and_config(
        settings=SETTINGS, tokenizer=None, device=DEV,
        checkpoint_dir=CKPT, arch="hvn", apply_ternary=True)

    # Post-hoc triton-scan (mirrors main).
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
    print(f"\n  [probe] post-hoc triton-scan set on {sum(hasattr(b,'use_triton_scan') for b in model.blocks)} blocks")

    # Compile (mirrors main).
    print("  [probe] compiling...")
    model = torch.compile(model)

    # Verbatim real loop. log every 10 so steady-state is visible; no checkpoint save.
    config["log_every_steps"] = 10
    config["save_every_steps"] = 100_000  # don't save during the probe
    config["checkpoint_dir"] = CKPT
    config["log_file"] = os.path.join(CKPT, "training_log.csv")
    os.makedirs(CKPT, exist_ok=True)

    print("\n" + "=" * 70)
    print("VERBATIM T.train() — the REAL loop, log_every=10, max_steps=60")
    print("=" * 70)
    T.train(
        model=model,
        dataloader=loader,
        config=config,
        device=DEV,
        checkpoint_dir=CKPT,
        max_steps=60,
        resume_step=0,
        val_loader=None,
    )
    print("\n" + "=" * 70)
    print("Read the per-10-step steps/s above:")
    print("  steps 1-10   : includes compile warmup (slow, ignore)")
    print("  steps 20-60  : STEADY-STATE. ~0.4 -> loop code is the cause; ~0.8 -> main()'s audit is the cause")
    print("=" * 70)


if __name__ == "__main__":
    main()