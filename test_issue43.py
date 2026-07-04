"""
Test suite for Issue #43 — correct, fused concurrent-serving decode.

Serving many users means running N independent streams (each with its own
Hamiltonian) through the model together.  The batched decode feeds a length-1
sequence per stream, so per-step wall-time is dominated by **kernel-launch /
Python overhead**, not compute — it is essentially flat in the stream count.
The lever for concurrent throughput is therefore **fewer launches**, not a bigger
matmul: routing the batched step through the Issue #33 ``FusedHVNStep`` Triton
kernel (2 launches/block instead of ~25 PyTorch ops) gives ~3× sustained tok/s.

This also fixes a **pre-existing correctness bug**: the legacy ``generate_batched``
expands per-head params with ``repeat_interleave`` (head-outer ordering ``h*B+b``)
while the stream batch is laid out batch-outer (``b*n_heads+h``), so for
``n_heads>1, B>1`` each concurrent stream silently used the *wrong head's*
projection weights — ``generate_batched`` did NOT equal running each stream alone.
``generate_served`` uses batch-outer ordering in both backends, so a concurrent
batch is equivalent to single-stream generation.

Two backends (``generate_served(backend=...)``):
  * ``torch`` — portable grouped-GEMM step; matches single-stream to ~1e-6.
  * ``fused`` — Triton single-launch step; ~3× tok/s; matches single-stream to
    the fused kernel's ~1e-2 numerics (same path as ``generate_incremental_fused``).

Run on GPU (RTX 4050):  python test_issue43.py
"""

import time

import torch

from hvn.core import HamiltonianVortexNetwork


def _device():
    assert torch.cuda.is_available(), "CUDA required — this project runs on GPU only"
    return torch.device("cuda")


def _build(dev, n_heads=4, d_model=128, num_layers=2, vocab=128, **kw):
    torch.manual_seed(0)
    return HamiltonianVortexNetwork(
        vocab_size=vocab, d_model=d_model, num_layers=num_layers,
        n_heads=n_heads, dropout=0.0, **kw,
    ).to(dev).eval()


def test_served_torch_single_stream_identical_to_batched():
    """B=1: the torch served step is bit-identical to forward_step_batched."""
    dev = _device()
    m = _build(dev, n_heads=4)
    toks = torch.randint(0, 128, (1, 6), device=dev)
    sa = m.init_generation_state_batched(1, device=dev, dtype=torch.float32)
    sb = m.init_generation_state_batched(1, device=dev, dtype=torch.float32)
    max_diff = 0.0
    for t in range(6):
        la, sa = m.forward_step_served(toks[:, t:t + 1], sa, position_idx=t, use_fused=False)
        lb, sb = m.forward_step_batched(toks[:, t:t + 1], sb, position_idx=t)
        max_diff = max(max_diff, (la - lb).abs().max().item())
    assert max_diff == 0.0, f"served(torch) != batched at B=1 (diff={max_diff})"
    print("[1/7] served(torch, B=1) == forward_step_batched(B=1) exactly ... OK")


def test_served_torch_concurrent_matches_single_stream_greedy():
    """Torch served concurrent batch == running each stream alone (greedy parity)."""
    dev = _device()
    for n_heads in (1, 4, 8):
        m = _build(dev, n_heads=n_heads)
        prompts = [[3, 5, 7, 9, 11, 13], [2, 4, 6, 8, 10, 12],
                   [1, 15, 17, 19, 21, 23], [9, 8, 7, 6, 5, 4]]
        ref = [m.generate_incremental(p, max_new_tokens=15, temperature=1e-6,
                                      top_k=1, device=dev) for p in prompts]
        srv = m.generate_served(prompts, max_new_tokens=15, temperature=1e-6,
                                top_k=1, device=dev, backend="torch")
        assert ref == srv, f"served(torch) != single-stream (n_heads={n_heads})"
    print("[2/7] served(torch) concurrent == single-stream (greedy, n_heads in {1,4,8}) ... OK")


def test_served_torch_concurrent_matches_single_stream_logits():
    """Per-step torch served logits match per-stream logits to float tolerance."""
    dev = _device()
    m = _build(dev, n_heads=4)
    B, T = 4, 6
    toks = torch.randint(0, 128, (B, T), device=dev)
    sb = m.init_generation_state_batched(B, device=dev, dtype=torch.float32)
    ss = [m.init_generation_state_batched(1, device=dev, dtype=torch.float32) for _ in range(B)]
    max_diff = 0.0
    for t in range(T):
        lg_srv, sb = m.forward_step_served(toks[:, t:t + 1], sb, position_idx=t, use_fused=False)
        for b in range(B):
            lg_b, ss[b] = m.forward_step_batched(toks[b:b + 1, t:t + 1], ss[b], position_idx=t)
            max_diff = max(max_diff, (lg_srv[b] - lg_b[0]).abs().max().item())
    assert max_diff < 1e-4, f"served(torch) concurrent logits diverge: {max_diff}"
    print(f"[3/7] served(torch) concurrent logits == single-stream (diff={max_diff:.1e}) ... OK")


def test_served_fused_concurrent_matches_single_stream():
    """Fused served concurrent batch matches single-stream to the kernel's ~1e-2 numerics."""
    dev = _device()
    m = _build(dev, n_heads=8, d_model=512, num_layers=2)  # d_head=64 >= 16
    B, T = 4, 6
    toks = torch.randint(0, 128, (B, T), device=dev)
    sb = m.init_generation_state_batched(B, device=dev, dtype=torch.float32)
    ss = [m.init_generation_state_batched(1, device=dev, dtype=torch.float32) for _ in range(B)]
    max_diff = 0.0
    for t in range(T):
        lg_srv, sb = m.forward_step_served(toks[:, t:t + 1], sb, position_idx=t, use_fused=True)
        for b in range(B):
            lg_b, ss[b] = m.forward_step_batched(toks[b:b + 1, t:t + 1], ss[b], position_idx=t)
            max_diff = max(max_diff, (lg_srv[b] - lg_b[0]).abs().max().item())
    # Fused kernel is the established fast path (#33): approximate, not bit-exact.
    assert max_diff < 5e-2, f"served(fused) concurrent logits diverge too far: {max_diff}"
    print(f"[4/7] served(fused) concurrent == single-stream (diff={max_diff:.1e}, kernel numerics) ... OK")


def test_legacy_batched_bug_is_documented():
    """Sanity: the legacy batched path DOES diverge from single-stream for n_heads>1.

    This pins the pre-existing head-misrouting bug that the served path fixes.
    """
    dev = _device()
    m = _build(dev, n_heads=4)
    prompts = [[3, 5, 7, 9, 11], [2, 4, 6, 8, 10]]
    ref = [m.generate_incremental(p, max_new_tokens=10, temperature=1e-6,
                                  top_k=1, device=dev) for p in prompts]
    legacy = m.generate_batched(prompts, max_new_tokens=10, temperature=1e-6,
                                top_k=1, device=dev)
    diverges = any(ref[i] != legacy[i] for i in range(len(prompts)))
    assert diverges, "legacy generate_batched unexpectedly matched single-stream"
    print("[5/7] legacy generate_batched confirmed to mis-route heads for B>1 (bug) ... OK")


def test_o1_scaling_preserved():
    """Served decode is O(1): ms/token flat as the generated length grows."""
    dev = _device()
    m = _build(dev, n_heads=8, d_model=512)
    prompts = [[1, 2, 3, 4] for _ in range(8)]

    def ms_per_tok(n_new):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        m.generate_served(prompts, max_new_tokens=n_new, temperature=1e-6, top_k=1, device=dev)
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / n_new

    _ = ms_per_tok(16)  # warmup
    short = ms_per_tok(64)
    long = ms_per_tok(256)
    ratio = long / short
    assert ratio < 1.5, f"decode not O(1): {short:.3f} -> {long:.3f} ms/tok (ratio {ratio:.2f})"
    print(f"[6/7] O(1) decode: {short:.3f} -> {long:.3f} ms/tok (ratio {ratio:.2f}) ... OK")


def _bench_tok_s(m, dev, B, max_new_tokens, gen_fn, prompt_len=8):
    prompts = [[(i + j) % 100 + 1 for j in range(prompt_len)] for i in range(B)]
    gen_fn(prompts, 8)  # warmup
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    gen_fn(prompts, max_new_tokens)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return B * max_new_tokens / dt  # total tokens/sec across all streams


def test_concurrent_throughput_benchmark():
    """Benchmark sustained tok/s vs concurrent-stream count: legacy / served-torch / served-fused."""
    dev = _device()
    m = _build(dev, n_heads=8, d_model=512, num_layers=2)

    def legacy(p, n):
        return m.generate_batched(p, max_new_tokens=n, temperature=1e-6, top_k=1, device=dev)

    def s_torch(p, n):
        return m.generate_served(p, max_new_tokens=n, temperature=1e-6, top_k=1, device=dev, backend="torch")

    def s_fused(p, n):
        return m.generate_served(p, max_new_tokens=n, temperature=1e-6, top_k=1, device=dev, backend="fused")

    print("[7/7] concurrent throughput (total tok/s across streams, 128 new tokens):")
    print(f"       {'streams':>8} | {'legacy(buggy)':>13} | {'served-torch':>12} | {'served-fused':>12} | {'fused vs legacy':>15}")
    for B in (1, 8, 32, 64):
        lg = _bench_tok_s(m, dev, B, 128, legacy)
        st = _bench_tok_s(m, dev, B, 128, s_torch)
        sf = _bench_tok_s(m, dev, B, 128, s_fused)
        print(f"       {B:>8} | {lg:>13.1f} | {st:>12.1f} | {sf:>12.1f} | {sf / lg:>14.2f}x")
    print("       (served backends keep every stream's output equal to single-stream; legacy does not)")


if __name__ == "__main__":
    print("=" * 72)
    print("Issue #43 — correct, fused concurrent-serving decode test suite")
    print("=" * 72)
    test_served_torch_single_stream_identical_to_batched()
    test_served_torch_concurrent_matches_single_stream_greedy()
    test_served_torch_concurrent_matches_single_stream_logits()
    test_served_fused_concurrent_matches_single_stream()
    test_legacy_batched_bug_is_documented()
    test_o1_scaling_preserved()
    test_concurrent_throughput_benchmark()
    print("=" * 72)
    print("ALL TESTS PASSED ✓")
    print("=" * 72)
