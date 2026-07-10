"""
Isolate WHAT in the audit perturbs torch.compile: the eager FORWARD or the BACKWARD?

Evidence so far:
  no audit (train_verbatim)              = 0.8 steps/s
  full audit fwd+bwd BEFORE compile       = 0.7  (verify_audit_cause)
  compile BEFORE audit                   = 0.8  (verify_audit_fix)  <- perturbation avoided

This isolates the perturber. Mirrors verify_audit_cause.py EXCEPT Check 2 does the
forward only (eval-mode forward + train-mode forward, NO backward, no loss) — i.e.
the audit stays BEFORE compile (so no charmap bug), but the eager backward is gone.

  - ~0.8 -> the BACKWARD is the perturber. Clean fix: drop the backward from
    Check 2 (forward-only audit, audit-before-compile preserved). One-line change,
    no charmap bug, no env changes, no audit-semantics change for shapes.
  - ~0.7 -> the eager FORWARD / eval() train() toggling perturbs too. Fix needs
    audit-on-a-copy or compile-before-audit + PYTHONUTF8=1.

    python scripts/verify_audit_fwdonly.py
"""
from __future__ import annotations

import os, sys
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch
import torch.nn as nn

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
CKPT = os.path.join(_ROOT, "checkpoints", "verify_audit_fwdonly")


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    loader, _, _ = T.build_dataset_pipeline(
        tokenizer=None, seq_len=SEQ, batch_size=BATCH, smoke=False,
        data_dir=os.path.join(_PRIMARY, "data_cache"), val_fraction=0.0, num_workers=4)

    model, config = T.build_model_and_config(
        settings=SETTINGS, tokenizer=None, device=DEV,
        checkpoint_dir=CKPT, arch="hvn", apply_ternary=True)

    # === Check 2 FORWARD-ONLY (mirrors pre_training_audit but NO backward) ===
    batch = next(iter(loader))
    batch_x, batch_y, _ = batch
    x = batch_x.to(DEV)
    print("\n  [probe] FORWARD-ONLY audit (eval forward + train forward, NO backward)...")
    model.eval()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = model(x)
    model.train()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        _ = model(x)
    model.zero_grad()
    print("  [probe] forward-only done (no backward, no loss)")

    # Post-hoc triton-scan + compile (mirrors main; audit stays BEFORE compile).
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
    print(f"  [probe] post-hoc triton-scan set on {sum(hasattr(b,'use_triton_scan') for b in model.blocks)} blocks")
    print("  [probe] compiling (audit was forward-only, BEFORE compile)...")
    model = torch.compile(model)

    config["log_every_steps"] = 50
    config["save_every_steps"] = 100_000
    config["checkpoint_dir"] = CKPT
    config["log_file"] = os.path.join(CKPT, "training_log.csv")
    os.makedirs(CKPT, exist_ok=True)

    print("\n" + "=" * 70)
    print("FORWARD-ONLY-AUDIT — log_every=50, max_steps=200")
    print("=" * 70)
    T.train(
        model=model, dataloader=loader, config=config, device=DEV,
        checkpoint_dir=CKPT, max_steps=200, resume_step=0, val_loader=None,
    )
    print("\n" + "=" * 70)
    print("  ~0.8 -> BACKWARD is the perturber; clean fix = drop backward from Check 2")
    print("  ~0.7 -> forward/toggling also perturbs; need audit-on-copy or PYTHONUTF8")
    print("=" * 70)


if __name__ == "__main__":
    main()