"""Parity tests for EdgeHVN gradient (activation) checkpointing (Issue #100).
No GPU needed — runs on CPU with tiny dims.

Mirrors the core HamiltonianVortexNetwork's #49 implementation, which
EdgeHVN never inherited (it has its own forward()). Checkpointing must be
EXACT — it changes only where activations are recomputed, not what the model
computes — so this asserts bit-identical logits and gradients with
grad_checkpoint on vs off, not just "it runs."

Run:  python -m pytest tests/test_edgehvn_grad_checkpoint.py -q
  or:  python tests/test_edgehvn_grad_checkpoint.py
"""
from __future__ import annotations

import os
import sys

import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.edge.model import EdgeHVN


def _make_pair(num_layers=4, block_plan=None, attn_window=None):
    """Two identically-initialized models, differing only in grad_checkpoint.

    dropout=0.0: a bit-identical parity test must not depend on matching two
    separately-drawn dropout masks across two forward calls (torch.utils.
    checkpoint also re-draws RNG state during its backward recompute, which
    makes seed-resetting tricks fragile) -- eliminating dropout sidesteps the
    issue entirely and isolates exactly what this test is checking: does
    checkpointing change the computed result.
    """
    torch.manual_seed(0)
    kwargs = dict(vocab_size=64, d_model=32, num_layers=num_layers, n_heads=4,
                  dropout=0.0)
    if block_plan is not None:
        kwargs["block_plan"] = block_plan
        kwargs["attn_window"] = attn_window
    else:
        kwargs["use_deltanet"] = True
        kwargs["chunk_size"] = 8

    torch.manual_seed(1)
    m_off = EdgeHVN(**kwargs, grad_checkpoint=False)
    torch.manual_seed(1)
    m_on = EdgeHVN(**kwargs, grad_checkpoint=True)

    # Same init (same seed) -- assert the weights actually match before
    # trusting any downstream comparison.
    for (n1, p1), (n2, p2) in zip(m_off.named_parameters(), m_on.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2), f"init mismatch at {n1}"
    return m_off, m_on


def test_forward_logits_bit_identical():
    m_off, m_on = _make_pair()
    m_off.train()
    m_on.train()
    ids = torch.randint(0, 64, (2, 16))

    torch.manual_seed(42)
    out_off = m_off(ids)
    torch.manual_seed(42)
    out_on = m_on(ids)

    assert torch.equal(out_off, out_on), "checkpointed forward changed the output"


def test_backward_gradients_bit_identical():
    m_off, m_on = _make_pair()
    m_off.train()
    m_on.train()
    ids = torch.randint(0, 64, (2, 16))

    out_off = m_off(ids)
    loss_off = out_off.float().pow(2).mean()
    loss_off.backward()

    out_on = m_on(ids)
    loss_on = out_on.float().pow(2).mean()
    loss_on.backward()

    assert torch.equal(loss_off, loss_on)
    for (n1, p1), (n2, p2) in zip(m_off.named_parameters(), m_on.named_parameters()):
        assert n1 == n2
        assert p1.grad is not None and p2.grad is not None, f"missing grad at {n1}"
        assert torch.equal(p1.grad, p2.grad), f"gradient mismatch at {n1}"


def test_checkpoint_inactive_in_eval_mode():
    """grad_checkpoint should not fire during eval() -- checkpoint() requires a
    live autograd graph, and eval-mode inference has no reason to pay the
    recompute cost."""
    m_on = EdgeHVN(vocab_size=64, d_model=32, num_layers=3, n_heads=4,
                   use_deltanet=True, chunk_size=8, grad_checkpoint=True)
    m_on.eval()
    ids = torch.randint(0, 64, (2, 8))
    with torch.no_grad():
        out = m_on(ids)  # must not raise / must not attempt checkpointing
    assert out.shape == (2, 8, 64)


def test_hybrid_block_plan_with_checkpoint():
    """The DeltaNet/attention hybrid also checkpoints correctly (mixed block
    types, matching Issue #99's partition being block-plan-aware too)."""
    plan = ["deltanet", "transformer", "deltanet", "transformer"]
    m_off, m_on = _make_pair(num_layers=len(plan), block_plan=plan, attn_window=16)
    m_off.train()
    m_on.train()
    ids = torch.randint(0, 64, (2, 16))

    out_off = m_off(ids)
    out_on = m_on(ids)
    assert torch.equal(out_off, out_on)

    loss_off = out_off.float().pow(2).mean()
    loss_off.backward()
    loss_on = out_on.float().pow(2).mean()
    loss_on.backward()
    for (n1, p1), (n2, p2) in zip(m_off.named_parameters(), m_on.named_parameters()):
        assert torch.equal(p1.grad, p2.grad), f"gradient mismatch at {n1}"


def test_default_grad_checkpoint_is_false():
    """Back-compat: existing callers that don't pass grad_checkpoint get the
    old (no-checkpointing) behaviour."""
    m = EdgeHVN(vocab_size=32, d_model=16, num_layers=2, n_heads=2, use_deltanet=True)
    assert m.grad_checkpoint is False


# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
