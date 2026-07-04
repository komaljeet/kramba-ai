"""
Test suite for Issue #40 — hybrid weightless (seed-regenerated) projections.

Verifies the ``WeightlessDeltaNetBlock`` variant that freezes the DeltaNet
``q/k/v/out`` projections as PRNG-seed-regenerated random features (stored as one
integer, not d² floats each) while keeping the learned data-dependent dynamics
(decay/write/output gates + causal conv).  Checks that:

  * the frozen projections are excluded from the checkpoint (the file-size win),
  * a checkpoint round-trips *exactly* (regenerated from the seed on load),
  * O(1) inference and forward↔incremental parity are preserved (dynamics intact),
  * defaults are untouched (weightless is opt-in and requires --use_deltanet).

Run on GPU (RTX 4050):  python test_issue40.py
"""

import io

import torch

from hvn.core import HamiltonianVortexNetwork, DeltaNetBlock
from hvn.core.weightless import WeightlessDeltaNetBlock, SeededLinear


def _device():
    assert torch.cuda.is_available(), "CUDA required — this project runs on GPU only"
    return torch.device("cuda")


def _build(weightless, dev, **kw):
    return HamiltonianVortexNetwork(
        vocab_size=97, d_model=128, num_layers=2, n_heads=4,
        use_deltanet=True, weightless=weightless, dropout=0.0, **kw
    ).to(dev).eval()


def test_build_and_frozen_projections():
    """Weightless model builds, is NaN-free, and freezes q/k/v/out as SeededLinear."""
    dev = _device()
    m = _build(True, dev)
    assert m.weightless is True
    blk = m.blocks[0]
    assert isinstance(blk, WeightlessDeltaNetBlock), "expected WeightlessDeltaNetBlock"
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        proj = getattr(blk, name)
        assert isinstance(proj, SeededLinear), f"{name} not seeded"
        # No trainable parameters in a frozen projection.
        assert len(list(proj.parameters())) == 0, f"{name} still has parameters"
    # The learned high-leverage core is still trainable.
    assert blk.alpha_proj.weight.requires_grad and blk.gate_proj.weight.requires_grad
    ids = torch.randint(0, 97, (2, 64), device=dev)
    with torch.no_grad():
        logits = m(ids)
    assert logits.shape == (2, 64, 97) and not torch.isnan(logits).any()
    print("[1/6] build + frozen q/k/v/out as SeededLinear ... OK")


def test_projections_excluded_from_state_dict():
    """Frozen projection weights are non-persistent → absent from the checkpoint."""
    dev = _device()
    m = _build(True, dev)
    keys = set(m.state_dict().keys())
    banned = [k for k in keys if any(
        k.endswith(f"{p}.weight") for p in ("q_proj", "k_proj", "v_proj", "out_proj"))]
    assert not banned, f"frozen projections leaked into state_dict: {banned}"
    # But the learned gates ARE present.
    assert any("alpha_proj.weight" in k for k in keys)
    print("[2/6] frozen projections excluded from state_dict . OK")


def test_checkpoint_size_delta():
    """Weightless checkpoint is smaller by exactly 4·d²·num_layers floats."""
    dev = _device()
    dense = _build(False, dev)
    light = _build(True, dev)

    def nbytes(model):
        buf = io.BytesIO()
        torch.save(model.state_dict(), buf)
        return buf.getbuffer().nbytes

    dense_params = sum(v.numel() for v in dense.state_dict().values())
    light_params = sum(v.numel() for v in light.state_dict().values())
    expected_saved = 4 * (128 ** 2) * 2  # q/k/v/out × d² × num_layers
    saved = dense_params - light_params
    assert saved == expected_saved, f"expected {expected_saved} fewer floats, got {saved}"
    db, lb = nbytes(dense), nbytes(light)
    print(f"[3/6] checkpoint size {db/1024:.1f}KB -> {lb/1024:.1f}KB "
          f"(-{(db-lb)/1024:.1f}KB, -{4*128*128*2:,} floats) . OK")


def test_save_load_roundtrip_exact():
    """Reloading (regenerating projections from the seed) reproduces outputs exactly."""
    dev = _device()
    torch.manual_seed(0)
    m = _build(True, dev, weightless_seed=777)
    ids = torch.randint(0, 97, (2, 40), device=dev)
    with torch.no_grad():
        ref = m(ids)

    # Round-trip through a checkpoint dict like train.py writes.
    ckpt = {
        "model_state_dict": m.state_dict(),
        "weightless": True, "weightless_seed": 777,
    }
    buf = io.BytesIO(); torch.save(ckpt, buf); buf.seek(0)
    loaded = torch.load(buf, weights_only=False)

    m2 = _build(True, dev, weightless_seed=loaded["weightless_seed"])
    missing, unexpected = m2.load_state_dict(loaded["model_state_dict"], strict=False)
    # Only the non-persistent frozen buffers may be "missing"; nothing unexpected.
    assert not unexpected, f"unexpected keys on load: {unexpected}"
    with torch.no_grad():
        out = m2(ids)
    max_diff = (ref - out).abs().max().item()
    assert max_diff < 1e-5, f"reload not exact: {max_diff}"
    # A different seed must give different frozen features (sanity).
    m3 = _build(True, dev, weightless_seed=778)
    m3.load_state_dict(loaded["model_state_dict"], strict=False)
    with torch.no_grad():
        out3 = m3(ids)
    assert (ref - out3).abs().max().item() > 1e-3, "seed had no effect"
    print(f"[4/6] save/load exact via seed (diff={max_diff:.1e}) .. OK")


def test_forward_matches_incremental():
    """Freezing projections doesn't change the recurrence: forward == forward_step."""
    dev = _device()
    torch.manual_seed(0)
    m = _build(True, dev)
    seq = torch.randint(0, 97, (1, 48), device=dev)
    with torch.no_grad():
        full = m(seq)
        st = m.init_generation_state(1, device=dev, dtype=torch.float32)
        step = []
        for t in range(seq.shape[1]):
            lo, st = m.forward_step(seq[:, t:t+1], st, position_idx=t)
            step.append(lo)
        step = torch.cat(step, dim=1)
    max_diff = (full - step).abs().max().item()
    assert max_diff < 1e-4, f"forward vs forward_step diverge: {max_diff}"
    print(f"[5/6] forward == incremental (diff={max_diff:.1e}) . OK")


def test_defaults_and_no_op_without_deltanet():
    """weightless defaults off, and is a no-op unless --use_deltanet is set."""
    dev = _device()
    # Default: not weightless, HVN path.
    m = HamiltonianVortexNetwork(vocab_size=97, d_model=64, num_layers=1, n_heads=4).to(dev)
    assert m.weightless is False
    # weightless=True without deltanet must NOT engage (needs DeltaNet dynamics).
    m2 = HamiltonianVortexNetwork(vocab_size=97, d_model=64, num_layers=1, n_heads=4,
                                  weightless=True, use_deltanet=False).to(dev)
    assert m2.weightless is False, "weightless silently engaged without DeltaNet"
    assert not isinstance(m2.blocks[0], WeightlessDeltaNetBlock)
    print("[6/6] defaults off + no-op without --use_deltanet . OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Issue #40 — hybrid weightless projections test suite")
    print("=" * 60)
    test_build_and_frozen_projections()
    test_projections_excluded_from_state_dict()
    test_checkpoint_size_delta()
    test_save_load_roundtrip_exact()
    test_forward_matches_incremental()
    test_defaults_and_no_op_without_deltanet()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
