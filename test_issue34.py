"""
Test suite for Issue #34 — Gated-DeltaNet readout/update dynamics.

Verifies the DeltaNetBlock fix that closes HVN's PPL gap by repairing the
readout/update dynamics (per-channel decay gate + readout normalizer +
separate-k/v DeltaNet update + output gate/conv), while preserving O(1)
inference and full backward compatibility with the default HVN path.

Run on GPU (RTX 4050):  python test_issue34.py
"""

import math
import time

import torch

from hvn.core import HamiltonianVortexNetwork, DeltaNetBlock


def _device():
    assert torch.cuda.is_available(), "CUDA required — this project runs on GPU only"
    return torch.device("cuda")


def test_forward_shapes_and_no_nan():
    dev = _device()
    m = HamiltonianVortexNetwork(vocab_size=97, d_model=128, num_layers=2,
                                 n_heads=4, use_deltanet=True, dropout=0.0).to(dev).eval()
    ids = torch.randint(0, 97, (2, 64), device=dev)
    with torch.no_grad():
        logits = m(ids)
    assert logits.shape == (2, 64, 97), logits.shape
    assert not torch.isnan(logits).any(), "NaN in forward logits"
    print("[1/5] forward shapes + NaN-free .......... OK")


def test_forward_matches_incremental():
    """Training forward and incremental forward_step share the same recurrence."""
    dev = _device()
    torch.manual_seed(0)
    m = HamiltonianVortexNetwork(vocab_size=97, d_model=128, num_layers=2,
                                 n_heads=4, use_deltanet=True, dropout=0.0).to(dev).eval()
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
    print(f"[2/5] forward == incremental (diff={max_diff:.1e}) . OK")


def test_o1_inference_scaling():
    """Per-token cost is flat in N (no KV cache, fixed-size state)."""
    dev = _device()
    m = HamiltonianVortexNetwork(vocab_size=97, d_model=128, num_layers=2,
                                 n_heads=4, use_deltanet=True, dropout=0.0).to(dev).eval()
    tok = torch.randint(0, 97, (1, 1), device=dev)
    times = {}
    for N in (256, 4096):
        st = m.init_generation_state(1, device=dev, dtype=torch.float32)
        for _ in range(5):
            _, st = m.forward_step(tok, st, position_idx=0)
        torch.cuda.synchronize(); t0 = time.time()
        for i in range(N):
            _, st = m.forward_step(tok, st, position_idx=i)
        torch.cuda.synchronize()
        times[N] = (time.time() - t0) / N * 1000
    ratio = times[4096] / times[256]
    # Fixed-size state ⇒ per-token cost independent of N (allow launch-noise slack)
    assert ratio < 1.5, f"per-token cost grew with N (ratio={ratio:.2f}) — not O(1)"
    print(f"[3/5] O(1) inference ({times[256]:.2f}→{times[4096]:.2f} ms/tok, "
          f"ratio={ratio:.2f}) . OK")


def test_long_context_stable_no_clamp():
    """1000+ token stability WITHOUT the Frobenius clamp (decay gate keeps state bounded)."""
    dev = _device()
    m = HamiltonianVortexNetwork(vocab_size=97, d_model=128, num_layers=2,
                                 n_heads=4, use_deltanet=True, dropout=0.0).to(dev).eval()
    st = m.init_generation_state(1, device=dev, dtype=torch.float32)
    with torch.no_grad():
        for t in range(1500):
            tok = torch.randint(0, 97, (1, 1), device=dev)
            lo, st = m.forward_step(tok, st, position_idx=t)
    smax = st[0]["S"].abs().max().item()
    assert not torch.isnan(lo).any(), "NaN after 1500 tokens"
    assert smax < 1e3, f"state blew up without clamp: |S|max={smax}"
    print(f"[4/5] 1500-token stability (|S|max={smax:.2f}, no clamp) . OK")


def test_backward_compat_default_path():
    """use_deltanet defaults to False — original HVN path is untouched."""
    dev = _device()
    m = HamiltonianVortexNetwork(vocab_size=97, d_model=128, num_layers=2, n_heads=4).to(dev)
    assert m.use_deltanet is False
    assert not isinstance(m.blocks[0], DeltaNetBlock), "default must be HVNBlock"
    ids = torch.randint(0, 97, (2, 32), device=dev)
    logits = m(ids)
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 97),
                                             torch.randint(0, 97, (64,), device=dev))
    loss.backward()  # trainable
    assert logits.shape == (2, 32, 97)
    print("[5/5] backward-compat default HVN path ..... OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Issue #34 — Gated-DeltaNet dynamics test suite")
    print("=" * 60)
    test_forward_shapes_and_no_nan()
    test_forward_matches_incremental()
    test_o1_inference_scaling()
    test_long_context_stable_no_clamp()
    test_backward_compat_default_path()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
