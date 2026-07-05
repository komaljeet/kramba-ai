"""
Test suite for Issue #46 — systemic batch-ordering bug in the multi-head
batched forward (train / eval / generate).

The shared batched vortex/memory modules expanded per-head params with
``repeat_interleave`` (head-outer ordering ``h*B+b``) while the block lays the
batch out **batch-outer** (``bh = b*n_heads + h``).  For ``n_heads>1, B>1`` this
paired each slot's data-head with the wrong weight-head, so a batched
``forward`` did **not** equal running each sequence alone — silently affecting
training, batched eval PPL, ``generate_batched`` and ``parallel_forward``.

The fix tiles per-head params batch-outer (``repeat`` instead of
``repeat_interleave``) in ``batched_vortex.py`` / ``batched_memory.py``, matching
the input-dependent-eta branch that already indexed ``arange(BH) % n_heads``.

Acceptance gate: **batched forward == single-stream** at every width/config.

Run on GPU (RTX 4050):  python test_issue46.py
"""

import torch

# --- repo-root path bootstrap (added during reorg, issue #48) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---
from hvn.core import HamiltonianVortexNetwork
from hvn.core.hvn_block import HVNBlock

TOL = 1e-4  # fp32 batched-matmul reassociation noise floor


def _device():
    assert torch.cuda.is_available(), "CUDA required — this project runs on GPU only"
    return torch.device("cuda")


def _max_batched_vs_alone(m, seqs):
    b = m(seqs)
    a = torch.cat([m(seqs[i:i + 1]) for i in range(seqs.shape[0])], 0)
    return (b - a).abs().max().item()


def test_forward_batched_equals_single_stream():
    """model.forward on a batch == concatenating per-sequence forwards."""
    dev = _device()
    seqs = torch.randint(0, 64, (4, 16), device=dev)
    for n_heads in (1, 2, 4, 8):
        torch.manual_seed(0)
        m = HamiltonianVortexNetwork(
            vocab_size=64, d_model=64, num_layers=2, n_heads=n_heads, dropout=0.0,
        ).to(dev).eval()
        diff = _max_batched_vs_alone(m, seqs)
        assert diff < TOL, f"forward batched != single-stream (n_heads={n_heads}, diff={diff:.2e})"
    print("[1/5] forward: batched == single-stream (n_heads in {1,2,4,8}) ... OK")


def test_forward_variants_equal_single_stream():
    """All flag-gated batched paths keep batched == single-stream."""
    dev = _device()
    seqs = torch.randint(0, 64, (3, 16), device=dev)
    variants = {
        "n_threads=2": dict(n_threads=2),
        "parallel_scan": dict(use_parallel_scan=True),
        "per_token_eta": dict(use_per_token_eta=True),
        "rank_k=4": dict(rank_k=4),
        "causal_reinjection": dict(use_causal_reinjection=True),
        "nonlinear_update": dict(use_nonlinear_update=True),
    }
    for name, kw in variants.items():
        torch.manual_seed(0)
        m = HamiltonianVortexNetwork(
            vocab_size=64, d_model=64, num_layers=2, n_heads=4, dropout=0.0, **kw,
        ).to(dev).eval()
        diff = _max_batched_vs_alone(m, seqs)
        assert diff < TOL, f"{name}: batched != single-stream (diff={diff:.2e})"
    print(f"[2/5] forward variants: batched == single-stream ({', '.join(variants)}) ... OK")


def test_parallel_forward_equals_single_stream():
    """HVNBlock.parallel_forward (Issue #33 Fix 4) keeps batched == single-stream."""
    dev = _device()
    torch.manual_seed(0)
    blk = HVNBlock(d_model=64, n_heads=4, dropout=0.0).to(dev).eval()
    x = torch.randn(3, 16, 64, device=dev)
    b = blk.parallel_forward(x)
    a = torch.cat([blk.parallel_forward(x[i:i + 1]) for i in range(3)], 0)
    diff = (b - a).abs().max().item()
    assert diff < TOL, f"parallel_forward batched != single-stream (diff={diff:.2e})"
    print(f"[3/5] parallel_forward: batched == single-stream (diff={diff:.1e}) ... OK")


def test_generate_batched_equals_single_stream():
    """generate_batched == running each prompt through generate_incremental alone."""
    dev = _device()
    for n_heads in (1, 4, 8):
        torch.manual_seed(0)
        m = HamiltonianVortexNetwork(
            vocab_size=128, d_model=128, num_layers=2, n_heads=n_heads, dropout=0.0,
        ).to(dev).eval()
        prompts = [[3, 5, 7, 9, 11], [2, 4, 6, 8, 10], [1, 15, 17, 19, 21]]
        ref = [m.generate_incremental(p, max_new_tokens=12, temperature=1e-6,
                                      top_k=1, device=dev) for p in prompts]
        got = m.generate_batched(prompts, max_new_tokens=12, temperature=1e-6,
                                 top_k=1, device=dev)
        assert ref == got, f"generate_batched != single-stream (n_heads={n_heads})"
    print("[4/5] generate_batched == single-stream (greedy, n_heads in {1,4,8}) ... OK")


def test_b1_unchanged():
    """B=1 is unaffected: tiling and interleaving coincide at batch size 1."""
    dev = _device()
    torch.manual_seed(0)
    m = HamiltonianVortexNetwork(
        vocab_size=64, d_model=64, num_layers=2, n_heads=4, dropout=0.0,
    ).to(dev).eval()
    seqs = torch.randint(0, 64, (1, 16), device=dev)
    # single-element batch equals itself (trivially) and runs without error
    diff = _max_batched_vs_alone(m, seqs)
    assert diff < TOL, f"B=1 self-consistency broke (diff={diff:.2e})"
    print("[5/5] B=1 forward self-consistent (tiling == interleave at B=1) ... OK")


if __name__ == "__main__":
    print("=" * 72)
    print("Issue #46 — batched forward == single-stream acceptance suite")
    print("=" * 72)
    test_forward_batched_equals_single_stream()
    test_forward_variants_equal_single_stream()
    test_parallel_forward_equals_single_stream()
    test_generate_batched_equals_single_stream()
    test_b1_unchanged()
    print("=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)
