"""
Verify the audit-fix: compile BEFORE the audit (no eager backward before compile).

verify_audit_cause.py (audit BEFORE compile) = 0.7 steps/s.
train_verbatim_probe.py (NO audit)          = 0.8 steps/s.
GPU-only bare loop (no audit, synthetic)    = 0.81 steps/s.

Hypothesis: the audit's eager forward+backward (Check 2) BEFORE torch.compile
leaves autograd/allocator state that makes the compiled graph ~0.1 s/step slower
(an HVN-specific tax — heavier graph perturbed more). Fix: compile FIRST, so the
audit runs on the compiled model and no eager backward perturbs the trace.

This probe mirrors verify_audit_cause.py EXACTLY except the audit and compile
order is swapped: build -> triton-scan -> COMPILE -> audit -> train.

  - ~0.8 steps/s  -> fix CONFIRMED; apply to main() (compile before audit).
  - ~0.7 steps/s  -> the audit-before-compile ordering is NOT the cause; the
    0.81->0.7 gap is something else the audit does (e.g. allocator fragmentation
    that persists regardless of order). Keep digging.

    python scripts/verify_audit_fix.py
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

CKPT = os.path.join(_ROOT, "checkpoints", "verify_audit_fix")


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    loader, _, _ = T.build_dataset_pipeline(
        tokenizer=None, seq_len=SEQ, batch_size=BATCH, smoke=False,
        data_dir=os.path.join(_PRIMARY, "data_cache"), val_fraction=0.0, num_workers=4)

    model, config = T.build_model_and_config(
        settings=SETTINGS, tokenizer=None, device=DEV,
        checkpoint_dir=CKPT, arch="hvn", apply_ternary=True)

    # Post-hoc triton-scan (mirrors main).
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
    print(f"\n  [probe] post-hoc triton-scan set on {sum(hasattr(b,'use_triton_scan') for b in model.blocks)} blocks")

    # === THE FIX: compile BEFORE the audit (no eager backward perturbs the trace) ===
    print("  [probe] compiling BEFORE audit (the proposed fix)...")
    model = torch.compile(model)

    # Audit runs on the compiled model (no eager backward happened before compile).
    audit_passed = T.pre_training_audit(
        model=model, dataloader=loader, config=config, device=DEV, checkpoint_dir=CKPT)
    if not audit_passed:
        print("  [probe] audit failed; continuing anyway for the timing test")

    config["log_every_steps"] = 50
    config["save_every_steps"] = 100_000
    config["checkpoint_dir"] = CKPT
    config["log_file"] = os.path.join(CKPT, "training_log.csv")
    os.makedirs(CKPT, exist_ok=True)

    print("\n" + "=" * 70)
    print("COMPILE-BEFORE-AUDIT — log_every=50, max_steps=200 (matches real)")
    print("=" * 70)
    T.train(
        model=model, dataloader=loader, config=config, device=DEV,
        checkpoint_dir=CKPT, max_steps=200, resume_step=0, val_loader=None,
    )
    print("\n" + "=" * 70)
    print("Compare steps 100/150/200 to audit-before-compile (0.7) and no-audit (0.8):")
    print("  ~0.8 -> fix CONFIRMED (compile-before-audit removes the audit tax)")
    print("  ~0.7 -> audit ordering is NOT the cause; keep digging")
    print("=" * 70)


if __name__ == "__main__":
    main()