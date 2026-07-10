"""Unit tests for the hybrid token-recall architecture (Phase A, issue: agentic
coding, VRAM-flat). No GPU needed — runs on CPU with tiny dims.

Covers:
  A1. per-layer block_plan dispatch in EdgeHVN (block types match the plan).
  A2. sliding-window attention mask in TransformerBlock (the mask actually
      bites: a token beyond the window has zero influence on a query inside it).
  A3. filtered hybrid trunk-init (matching trunk keys copied, attention-position
      keys left at fresh init, orphaned DeltaNet keys skipped — no crash).

Run:  python -m pytest tests/test_hybrid_forward.py -q
  or:  python tests/test_hybrid_forward.py
"""
from __future__ import annotations

import os
import sys
import tempfile

import torch

# Make the repo root importable when run directly (python tests/test_*.py).
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.core.transformer_block import TransformerBlock
from hvn.edge.model import EdgeHVN


# ---------------------------------------------------------------------------
# A1 — block_plan dispatch
# ---------------------------------------------------------------------------

def test_block_plan_dispatch_matches_plan():
    """Each layer's block type matches block_plan[i]."""
    plan = ["deltanet", "transformer", "deltanet", "transformer", "deltanet"]
    model = EdgeHVN(
        vocab_size=64, d_model=32, num_layers=len(plan), n_heads=4,
        block_plan=plan, attn_window=16,
    )
    assert model.block_plan == plan
    assert model.attn_window == 16
    types = [type(b).__name__ for b in model.blocks]
    expected = ["DeltaNetBlock" if t == "deltanet" else "TransformerBlock" for t in plan]
    assert types == expected, (types, expected)


def test_block_plan_wrong_length_raises():
    try:
        EdgeHVN(vocab_size=64, d_model=32, num_layers=4, n_heads=4,
                block_plan=["deltanet", "transformer"], attn_window=16)
    except ValueError:
        return
    raise AssertionError("block_plan of wrong length should raise ValueError")


def test_legacy_single_type_back_compat():
    """No block_plan → legacy broadcast (all DeltaNet / all Transformer)."""
    m_hvn = EdgeHVN(vocab_size=64, d_model=32, num_layers=3, n_heads=4, use_deltanet=True)
    assert m_hvn.block_plan == ["deltanet", "deltanet", "deltanet"]
    assert all(type(b).__name__ == "DeltaNetBlock" for b in m_hvn.blocks)

    m_tf = EdgeHVN(vocab_size=64, d_model=32, num_layers=3, n_heads=4, block_type="transformer")
    assert m_tf.block_plan == ["transformer"] * 3
    assert all(type(b).__name__ == "TransformerBlock" for b in m_tf.blocks)


def test_hybrid_forward_shape():
    """Mixed block types share the forward contract → hybrid forward just works."""
    plan = ["deltanet", "transformer", "deltanet", "transformer"]
    model = EdgeHVN(vocab_size=50, d_model=32, num_layers=4, n_heads=4,
                   block_plan=plan, attn_window=16)
    model.eval()
    ids = torch.randint(0, 50, (2, 24))
    with torch.no_grad():
        y = model(ids)
    assert y.shape == (2, 24, 50), y.shape


# ---------------------------------------------------------------------------
# A2 — sliding-window mask
# ---------------------------------------------------------------------------

def test_sliding_window_masks_beyond_window():
    """A key farther than `window` from a query has zero effect on that query.

    Perturb token 0 only. Output at query position j should be unchanged for
    j >= window (key 0 is masked: j - 0 >= window). Position window-1 still
    attends to token 0, so it must change. FFN + residual are per-position, so
    the only cross-token path is attention — this isolates the mask.
    """
    torch.manual_seed(0)
    blk = TransformerBlock(d_model=64, n_heads=4, dropout=0.0, window=4)
    blk.eval()
    x = torch.randn(1, 10, 64)
    with torch.no_grad():
        y1 = blk(x)
        x2 = x.clone()
        x2[0, 0] = torch.randn(64) * 10.0  # strongly perturb token 0
        y2 = blk(x2)

    # positions >= window: key 0 masked (j - 0 = j >= window) → unchanged
    assert torch.allclose(y1[0, 4:], y2[0, 4:], atol=1e-5), \
        "token 0 leaked into positions >= window (mask failed)"
    # position window-1 (3): 3 - 0 = 3 < window → attends to token 0 → changes
    assert not torch.allclose(y1[0, 3:4], y2[0, 3:4], atol=1e-5), \
        "position window-1 did not attend to token 0 (mask too aggressive)"


def test_window_none_is_full_causal():
    """window=None (or S <= window) keeps the fast is_causal=True path — no mask.

    Equivalent to full causal: every position attends to all past positions, so
    perturbing token 0 changes every later position.
    """
    torch.manual_seed(1)
    blk = TransformerBlock(d_model=64, n_heads=4, dropout=0.0, window=None)
    blk.eval()
    x = torch.randn(1, 10, 64)
    with torch.no_grad():
        y1 = blk(x)
        x2 = x.clone()
        x2[0, 0] = torch.randn(64) * 10.0
        y2 = blk(x2)
    # full causal: token 0 affects all later positions
    assert not torch.allclose(y1[0, 5:6], y2[0, 5:6], atol=1e-5), \
        "window=None should be full causal (token 0 must affect position 5)"


def test_window_larger_than_seq_is_full_causal():
    """S <= window → no sliding mask, full causal (training path: seq 512 < W 4096)."""
    torch.manual_seed(2)
    blk = TransformerBlock(d_model=64, n_heads=4, dropout=0.0, window=4096)
    blk.eval()
    x = torch.randn(1, 8, 64)
    with torch.no_grad():
        y1 = blk(x)
        x2 = x.clone()
        x2[0, 0] = torch.randn(64) * 10.0
        y2 = blk(x2)
    # window 4096 > S 8 → full causal → token 0 affects later positions
    assert not torch.allclose(y1[0, 5:6], y2[0, 5:6], atol=1e-5)


# ---------------------------------------------------------------------------
# A3 — filtered hybrid trunk-init
# ---------------------------------------------------------------------------

def _save_pure_hvn_ckpt(path, d_model=64, num_layers=4, n_heads=4, vocab=64):
    """Build a pure-HVN (all-DeltaNet) EdgeHVN and save a minimal ckpt dict."""
    trunk = EdgeHVN(vocab_size=vocab, d_model=d_model, num_layers=num_layers,
                   n_heads=n_heads, use_deltanet=True)
    ckpt = {"model_state_dict": trunk.state_dict(), "global_step": 1234,
            "vocab_size": vocab, "d_model": d_model, "num_layers": num_layers,
            "n_heads": n_heads, "use_deltanet": True, "block_type": None,
            "block_plan": None, "attn_window": None}
    torch.save(ckpt, path)
    return trunk


def test_filtered_trunk_init_key_accounting():
    """load_hybrid_trunk_into_model copies trunk keys, skips attn positions, re-inits attn."""
    import train_edge_conversational as T

    attn_positions = [1, 3]
    plan = ["deltanet" if i not in attn_positions else "transformer"
            for i in range(4)]

    with tempfile.TemporaryDirectory() as tmp:
        trunk_path = os.path.join(tmp, "trunk.pt")
        trunk = _save_pure_hvn_ckpt(trunk_path)

        # Build the hybrid target (same dims) and snapshot fresh attn weights.
        hybrid = EdgeHVN(vocab_size=64, d_model=64, num_layers=4, n_heads=4,
                        block_plan=plan, attn_window=16)
        # Capture fresh-init attention weights before the load.
        fresh_attn_w = {
            i: {k: v.clone() for k, v in hybrid.blocks[i].state_dict().items()}
            for i in attn_positions
        }
        # Capture a trunk (DeltaNet) block weight at a non-attn position to verify copy.
        trunk_block0_w = trunk.blocks[0].state_dict()
        # The first key tensor of block 0 in the trunk.
        first_key0 = next(iter(trunk_block0_w))

        step = T.load_hybrid_trunk_into_model(hybrid, trunk_path, torch.device("cpu"))

        # 1. Trunk (DeltaNet) block at a non-attn position must match the trunk.
        hybrid_block0_w = hybrid.blocks[0].state_dict()
        k = first_key0
        assert torch.equal(hybrid_block0_w[k], trunk_block0_w[k]), \
            "trunk DeltaNet block 0 was not copied from the pure-HVN checkpoint"

        # 2. Attention-position blocks must still be at fresh init (unchanged).
        for i in attn_positions:
            for k, v in hybrid.blocks[i].state_dict().items():
                assert torch.equal(v, fresh_attn_w[i][k]), \
                    f"attn block {i} weight {k} was overwritten by trunk-init (should be fresh)"

        # 3. Token embedding + final_norm + decoder copied (trunk keys).
        assert torch.equal(hybrid.token_embedding.weight, trunk.token_embedding.weight), \
            "token_embedding not copied from trunk"

        assert step == 1234


def test_resume_hybrid_checkpoint_roundtrip():
    """A hybrid checkpoint saves block_plan and resumes into the same arch."""
    import train_edge_conversational as T

    plan = ["deltanet", "transformer", "deltanet", "transformer"]
    model = EdgeHVN(vocab_size=64, d_model=64, num_layers=4, n_heads=4,
                    block_plan=plan, attn_window=16)
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "hybrid_step500.pt")
        # Minimal save mirroring _save_edge_checkpoint's hybrid fields.
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab_size": 64, "d_model": 64, "num_layers": 4, "n_heads": 4,
            "use_deltanet": False, "block_type": None,
            "block_plan": plan, "attn_window": 16,
            "ternary": False, "global_step": 500,
        }, path)
        resumed, step, ckpt = T.load_checkpoint_for_resume(path, torch.device("cpu"))
        assert resumed.block_plan == plan, resumed.block_plan
        assert resumed.attn_window == 16
        assert step == 500
        # State dict must load with zero missing/unexpected (matched arch).
        types = [type(b).__name__ for b in resumed.blocks]
        assert types == ["DeltaNetBlock", "TransformerBlock",
                         "DeltaNetBlock", "TransformerBlock"], types


# ---------------------------------------------------------------------------
# Runner for `python tests/test_hybrid_forward.py`
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