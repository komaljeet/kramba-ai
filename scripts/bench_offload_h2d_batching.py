"""Would batching the offload H2D copies (one contiguous copy per block instead
of one per parameter tensor) make them faster? (2026-10-22: no -- ~1%.)

The 1B --profile run showed `Memcpy HtoD (Pinned -> Device)` at 2.46 s/step and
6,528 calls/step. This script answers the two questions that decide whether
batching is worth building:

1. Predicted count and bytes per optimizer step from the real model's parameter
   list and the offload code path (GPUBufferPool.prefetch copies one tensor at a
   time; each offloaded block is fetched once in forward and once more for its
   backward recompute, on every micro-batch). If this does not reproduce the
   profiled 6,528 calls and ~2.46 s, the explanation is wrong.
2. Measured H2D time for one real block's tensors (park dtypes as
   GPUBufferPool.park_dtype_for assigns them) copied the way prefetch does it
   (one copy per tensor, pinned source, side stream) vs ONE contiguous copy of
   the same bytes. The gap is the most batching could ever save.

Model shapes come from a meta-device build, so nothing large is allocated on the
host. GPU use is ~170 MB for a few seconds.

Usage:
    python scripts/bench_offload_h2d_batching.py
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
import torch.nn as nn

# The profiled config (results/profile_1b_hunt_v2/train.log).
D, L, H, VOCAB, ATTN, GRAD_ACCUM = 2048, 24, 64, 50257, (6, 18), 8


def build_meta():
    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model
    plan = ["transformer" if i in ATTN else "deltanet" for i in range(L)]
    with torch.device("meta"):
        m = EdgeHVN(vocab_size=VOCAB, d_model=D, num_layers=L, n_heads=H,
                    use_deltanet=False, deltanet_conv_kernel=4, block_plan=plan,
                    attn_window=4096, grad_checkpoint=True, dropout=0.0)
        lin = [n for n, mod in m.named_modules() if isinstance(mod, nn.Linear)]
        apply_ternary_to_model(m, target_names=[
            n for n in lin if not any(p in n for p in ("beta_proj", "alpha_proj", "decoder"))])
    return m


def park_dtype(p):
    # GPUBufferPool.park_dtype_for with park_dtype=bf16 and fp32 native.
    return torch.float32 if p.dim() <= 1 else torch.bfloat16


def nbytes(params):
    return sum(p.numel() * (4 if park_dtype(p) == torch.float32 else 2) for p in params)


def predict(m):
    """Active block = token_embedding (the profiled one): every layer block is
    downstream of it, so every block is recomputed in backward."""
    blocks = [list(b.parameters()) for b in m.blocks]
    tail = list(m.final_norm.parameters()) + list(m.decoder.parameters())
    # Forward prefetch: issued after block i for block i+1, and after the last
    # block for the tail. Block 0 is loaded (not prefetched) -- pageable, not
    # counted in the Pinned row.
    fwd = blocks[1:] + [tail]
    # Backward recompute prefetch (behind=True): block 23 at the end of
    # forward, then block i-1 at the start of block i's recompute.
    bwd = blocks
    calls = sum(len(b) for b in fwd + bwd)
    byts = sum(nbytes(b) for b in fwd + bwd)
    return calls * GRAD_ACCUM, byts * GRAD_ACCUM


def bench_block(params, reps=40, trials=3):
    src = [torch.zeros(p.shape, dtype=park_dtype(p)).pin_memory() for p in params]
    dst = [torch.empty(p.shape, dtype=park_dtype(p), device="cuda") for p in params]
    total = sum(t.numel() * t.element_size() for t in src)
    flat_src = torch.empty(total, dtype=torch.uint8).pin_memory()
    flat_dst = torch.empty(total, dtype=torch.uint8, device="cuda")
    side = torch.cuda.Stream()

    def per_tensor():
        for s, d in zip(src, dst):
            d.copy_(s, non_blocking=True)

    def one_copy():
        flat_dst.copy_(flat_src, non_blocking=True)

    # Falsifiability: the per-tensor path must really move data, else its time
    # would be meaningless.
    for i, s in enumerate(src):
        s.fill_(i + 1)
    with torch.cuda.stream(side):
        per_tensor()
    torch.cuda.synchronize()
    assert all(torch.equal(s.cuda(), d) for s, d in zip(src, dst)), "per-tensor copy did not land"

    def timeit(fn):
        with torch.cuda.stream(side):
            for _ in range(3):
                fn()
        torch.cuda.synchronize()
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.stream(side):
            a.record(side)
            for _ in range(reps):
                fn()
            b.record(side)
        torch.cuda.synchronize()
        return a.elapsed_time(b) / reps

    rows = [(timeit(per_tensor), timeit(one_copy)) for _ in range(trials)]
    return total, len(src), rows


def main():
    if not torch.cuda.is_available():
        raise SystemExit("needs CUDA")
    m = build_meta()
    calls, byts = predict(m)
    print(f"predicted Pinned->Device per optimizer step: {calls} calls, {byts / 1e9:.2f} GB")
    print("  (profile: 6528.3 calls incl. the DataLoader's 2 pinned batch copies per micro-batch, 2.460 s)")
    total, n, rows = bench_block(list(m.blocks[0].parameters()))
    for tp, to in rows:
        print(f"one DeltaNet block, {total / 1e6:.1f} MB in {n} tensors: per-tensor {tp:.3f} ms "
              f"({total / tp / 1e6:.1f} GB/s) | one contiguous copy {to:.3f} ms "
              f"({total / to / 1e6:.1f} GB/s) | batching saves {100 * (tp - to) / tp:+.1f}%")
    bw = total / min(r[1] for r in rows)  # bytes per ms
    print(f"predicted memcpy time/step at measured link rate: {byts / bw / 1000:.3f} s (profile: 2.460 s)")


if __name__ == "__main__":
    main()
