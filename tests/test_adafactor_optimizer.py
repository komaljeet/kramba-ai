"""Tests for the Adafactor optimizer wiring (Issue #85): factored second
moment (O(m+n) instead of O(m*n) per weight matrix). Uses torch's own native
`torch.optim.Adafactor` -- bitsandbytes (8-bit Adam) has no reliable prebuilt
wheel path on native Windows (this project's platform), so Adafactor is the
practical choice for this issue. No GPU needed -- runs on CPU with tiny dims.

Covers:
  1. Factored state: row_var/col_var are O(m+n)-shaped, not a full m*n
     exp_avg_sq -- the actual memory-savings mechanism, verified directly.
  2. build_optimizer("adafactor") wires weight-decay groups correctly (decay
     vs no-decay, matching every other optimizer in this module).
  3. Unknown optimizer name still raises (regression guard: adding a new
     branch didn't break the "expected adamw|muon|..." error path).
  4. Loss decreases over a real short training run through the actual
     build_optimizer("adafactor") + EdgeHVN path (correctness smoke).

Run:  python -m pytest tests/test_adafactor_optimizer.py -q
  or:  python tests/test_adafactor_optimizer.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.optimizers import build_optimizer
from hvn.edge.model import EdgeHVN


def test_factored_state_smaller_than_full():
    """A 64x32 matrix's Adafactor state is row_var [64,1] + col_var [1,32]
    (96 elements total) vs AdamW's exp_avg + exp_avg_sq at 64*32*2 = 4096
    elements -- the actual O(m+n) vs O(m*n) claim, verified directly."""
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(64, 32))
    opt = torch.optim.Adafactor([p], lr=0.01)
    (p ** 2).sum().backward()
    opt.step()
    st = opt.state[p]
    assert st["row_var"].shape == (64, 1)
    assert st["col_var"].shape == (1, 32)
    factored_elems = st["row_var"].numel() + st["col_var"].numel()
    full_adamw_elems = 2 * p.numel()  # exp_avg + exp_avg_sq
    assert factored_elems < full_adamw_elems


def test_build_optimizer_adafactor_groups():
    torch.manual_seed(0)
    model = EdgeHVN(vocab_size=64, d_model=32, num_layers=2, n_heads=4,
                     use_deltanet=True, chunk_size=8, dropout=0.0)
    opt, desc = build_optimizer(model, "adafactor", lr=1e-3, weight_decay=0.01)
    assert "Adafactor" in desc
    assert len(opt.param_groups) == 2
    wds = {pg["weight_decay"] for pg in opt.param_groups}
    assert wds == {0.01, 0.0}  # decay group + no-decay group, same pattern as adamw/muon/galore


def test_unknown_optimizer_still_raises():
    torch.manual_seed(0)
    model = EdgeHVN(vocab_size=64, d_model=32, num_layers=2, n_heads=4,
                     use_deltanet=True, chunk_size=8, dropout=0.0)
    try:
        build_optimizer(model, "nonexistent", lr=1e-3, weight_decay=0.01)
        raise AssertionError("expected ValueError for unknown optimizer name")
    except ValueError as e:
        assert "adafactor" in str(e)  # new choice listed in the updated error message


def test_training_converges_with_adafactor():
    torch.manual_seed(0)
    model = EdgeHVN(vocab_size=64, d_model=32, num_layers=2, n_heads=4,
                     use_deltanet=True, chunk_size=8, dropout=0.0)
    opt, _ = build_optimizer(model, "adafactor", lr=1e-2, weight_decay=0.0)

    torch.manual_seed(1)
    ids = torch.randint(0, 64, (4, 16))
    losses = []
    for _ in range(60):
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    early_mean = sum(losses[:10]) / 10
    late_mean = sum(losses[-10:]) / 10
    assert late_mean < early_mean, (
        f"loss did not decrease with Adafactor: early={early_mean:.4f} late={late_mean:.4f}"
    )


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
