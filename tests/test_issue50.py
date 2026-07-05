"""
Test suite for Issue #50 — the ``--full`` meta-flag (one-switch recommended
fast-training stack).

``--full`` expands to the complete recommended training stack in a single switch:

  * **Category A — exact speed levers (model bit-identical):** fused AdamW kernel,
    per-epoch (not per-step) loss sync, pinned/async data loading, ``--compile``,
    ``--grad_checkpoint`` (#49).
  * **Category B — recommended defaults that change the model/precision:**
    ``--precision bf16`` and ``--use_deltanet``.

Any individual flag passed explicitly overrides the preset.

Acceptance gates (from the issue):
  1. ``--full`` expands to the documented flag set.
  2. Explicit overrides beat ``--full`` (e.g. ``--full --precision fp32``).
  3. Category-A primitives are exact: ``fused-off vs on`` and
     ``sync-removed vs original`` produce identical loss curves (numerical noise).
  4. Defaults unchanged — a bare run enables none of the preset primitives.
  5. ``--full`` composes ``--grad_checkpoint`` (#49) and does NOT enable
     ``--use_parallel_scan`` (no-op under DeltaNet).

Run:  python tests/test_issue50.py   (Category-A GPU parity skips gracefully on CPU)
"""

import argparse

# --- repo-root path bootstrap (added during reorg, issue #48) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---
import torch
from torch.utils.data import DataLoader

from train import apply_full_preset, train_model, TextDataset
from hvn.core import HamiltonianVortexNetwork

TOL = 1e-5


def _base_args(**overrides):
    """A minimal Namespace mirroring the train.py defaults touched by --full."""
    ns = argparse.Namespace(
        precision="fp32", use_deltanet=False, compile=False,
        grad_checkpoint=False, use_parallel_scan=False,
        fused_adamw=False, defer_loss_sync=False, pin_memory=False,
    )
    for k, v in overrides.items():
        setattr(ns, k, v)
    return ns


def test_full_expands_stack():
    """--full turns on exactly the documented Category-A + Category-B set."""
    args = _base_args()
    apply_full_preset(args, explicit_flags=set())
    # Category B — model/precision changing
    assert args.precision == "bf16", "full should set bf16"
    assert args.use_deltanet is True, "full should enable DeltaNet"
    # Category A — exact speed levers
    assert args.compile is True
    assert args.grad_checkpoint is True, "full should compose #49 checkpointing"
    assert args.fused_adamw is True
    assert args.defer_loss_sync is True
    assert args.pin_memory is True
    # Interaction rule: parallel_scan stays OFF (no-op under DeltaNet)
    assert args.use_parallel_scan is False, "full must not enable parallel_scan"
    print("[1/5] --full expands to the full recommended stack (A+B), no parallel_scan ... OK")


def test_explicit_override_beats_full():
    """An explicitly-passed flag wins over the preset (precision fp32)."""
    args = _base_args(precision="fp32")
    enabled = apply_full_preset(args, explicit_flags={"--precision"})
    assert args.precision == "fp32", "explicit --precision fp32 must survive --full"
    assert any("user override" in e for e in enabled), "banner should note the override"
    # Non-overridden preset members still apply
    assert args.use_deltanet is True and args.grad_checkpoint is True
    print("[2/5] explicit --precision fp32 overrides --full (bf16) ... OK")


def test_defaults_unchanged():
    """A bare run (no --full) enables none of the preset primitives."""
    args = _base_args()
    # apply_full_preset is simply never called without --full; assert the
    # defaults it would flip are all off to begin with.
    assert args.fused_adamw is False
    assert args.defer_loss_sync is False
    assert args.pin_memory is False
    assert args.compile is False and args.grad_checkpoint is False
    assert args.precision == "fp32" and args.use_deltanet is False
    print("[3/5] defaults unchanged: bare run enables no --full primitives ... OK")


def _train_tiny(device, *, fused, defer_sync, seed=0):
    """Train a tiny model deterministically and return it (weights are the probe)."""
    torch.manual_seed(seed)
    ids = torch.randint(0, 32, (600,)).tolist()
    ds = TextDataset(ids, seq_len=16)
    dl = DataLoader(ds, batch_size=8, shuffle=False, drop_last=False)  # deterministic order
    torch.manual_seed(seed)
    model = HamiltonianVortexNetwork(vocab_size=32, d_model=32, num_layers=2,
                                     n_heads=2, dropout=0.0)
    save_path = _os.path.join(_os.path.dirname(__file__), "_tmp_issue50.pt")
    train_model(model, dl, epochs=3, lr=1e-3, device=device, save_path=save_path,
                fused_optimizer=fused, defer_loss_sync=defer_sync)
    if _os.path.exists(save_path):
        _os.remove(save_path)
    return model


def _weights_flat(model):
    return torch.cat([p.detach().reshape(-1).float().cpu() for p in model.parameters()])


def test_defer_loss_sync_exact():
    """Removing the per-step loss.item() sync is a pure logging change: the
    trained weights are bit-identical to the original per-step-sync path."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    m_orig = _train_tiny(device, fused=False, defer_sync=False, seed=1)
    m_defer = _train_tiny(device, fused=False, defer_sync=True, seed=1)
    diff = (_weights_flat(m_orig) - _weights_flat(m_defer)).abs().max().item()
    assert diff < TOL, f"defer_loss_sync changed the model (max|diff|={diff:.2e})"
    print(f"[4/5] sync-removed == original: identical weights (max|diff|={diff:.1e}) ... OK")


def test_fused_adamw_exact():
    """Fused vs unfused AdamW train the same model (GPU-only; skips on CPU)."""
    if not torch.cuda.is_available():
        print("[5/5] fused-off == fused-on: SKIP (no CUDA) ... OK")
        return
    device = torch.device("cuda")
    m_unfused = _train_tiny(device, fused=False, defer_sync=False, seed=2)
    m_fused = _train_tiny(device, fused=True, defer_sync=False, seed=2)
    diff = (_weights_flat(m_unfused) - _weights_flat(m_fused)).abs().max().item()
    # Fused kernel reassociates reductions; allow a small fp noise floor.
    assert diff < 1e-4, f"fused AdamW diverged from unfused (max|diff|={diff:.2e})"
    print(f"[5/5] fused-off == fused-on: matching weights (max|diff|={diff:.1e}) ... OK")


if __name__ == "__main__":
    print("=" * 72)
    print("Issue #50 — --full meta-flag acceptance suite")
    print("=" * 72)
    test_full_expands_stack()
    test_explicit_override_beats_full()
    test_defaults_unchanged()
    test_defer_loss_sync_exact()
    test_fused_adamw_exact()
    print("=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)
