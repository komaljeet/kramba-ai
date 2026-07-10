"""
Verify the audit-before-compile hypothesis.

train_verbatim_probe.py (NO audit) = 0.8 steps/s. The real script main() = 0.4.
The only difference: main() runs pre_training_audit() (Step 4) — which does an
EAGER forward+backward on the model — BEFORE torch.compile. This script mirrors
main()'s EXACT order (build -> audit -> triton-scan -> compile -> train) by
inserting pre_training_audit() before compile.

  - If this drops to ~0.4  -> the audit's pre-compile eager fwd+bwd is CONFIRMED
    as the cause. Fix: move the audit after compile, or make it not perturb the
    compiled path (e.g. run on a throwaway copy, or skip the backward), or just
    re-order so compile happens before any eager backward.
  - If this stays ~0.8     -> the audit is NOT it; keep looking.

    python scripts/verify_audit_cause.py
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

CKPT = os.path.join(_ROOT, "checkpoints", "verify_audit_cause")


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    loader, _, _ = T.build_dataset_pipeline(
        tokenizer=None, seq_len=SEQ, batch_size=BATCH, smoke=False,
        data_dir=os.path.join(_PRIMARY, "data_cache"), val_fraction=0.0, num_workers=4)

    model, config = T.build_model_and_config(
        settings=SETTINGS, tokenizer=None, device=DEV,
        checkpoint_dir=CKPT, arch="hvn", apply_ternary=True)

    # === THE DIFFERENCE: run the audit BEFORE triton-scan + compile (mirrors main) ===
    audit_passed = T.pre_training_audit(
        model=model, dataloader=loader, config=config, device=DEV, checkpoint_dir=CKPT)
    if not audit_passed:
        print("  [probe] audit failed; continuing anyway for the timing test")

    # Post-hoc triton-scan (mirrors main).
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
    print(f"\n  [probe] post-hoc triton-scan set on {sum(hasattr(b,'use_triton_scan') for b in model.blocks)} blocks")

    # Compile (mirrors main).
    print("  [probe] compiling...")
    model = torch.compile(model)

    config["log_every_steps"] = 50
    config["save_every_steps"] = 100_000
    config["checkpoint_dir"] = CKPT
    config["log_file"] = os.path.join(CKPT, "training_log.csv")
    os.makedirs(CKPT, exist_ok=True)

    print("\n" + "=" * 70)
    print("WITH-AUDIT T.train() — audit BEFORE compile, log_every=50, max_steps=200 (matches real)")
    print("=" * 70)
    T.train(
        model=model, dataloader=loader, config=config, device=DEV,
        checkpoint_dir=CKPT, max_steps=200, resume_step=0, val_loader=None,
    )
    print("\n" + "=" * 70)
    print("Compare steps 100/150/200 to the real script (0.4 steps/s):")
    print("  ~0.4 -> reproduced; probe == real script (audit + long run is the cause)")
    print("  ~0.7 -> a remaining difference in main() still to find")
    print("=" * 70)


if __name__ == "__main__":
    main()