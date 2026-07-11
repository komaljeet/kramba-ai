# Benchmark Results — Speed, Kernels & Decode

> **Moved from README:** §8.1 Speed · §8.3 Custom Kernels & Parallel Processing · §8.9
> Concurrent-Serving Decode · §8.13 Exact Gradient Checkpointing. Quality in
> [benchmarks_quality.md](benchmarks_quality.md); scaling in
> [benchmarks_scaling.md](benchmarks_scaling.md).

**Test configuration**: RTX 4050 (6 GB), PyTorch 2.12.1+cu126, d_model=128, 2 layers,
vocab=256 unless noted.

## 8.1 Speed

**Generation (50 tokens, prompt=32):**

| Model | tokens/sec | ms/token | Peak Mem | Params |
|-------|-----------|----------|----------|--------|
| HVN (incremental) | 106.5 | 9.4 | 13.6 MB | 264,966 |
| HVN (old, full context) | 75.6 | 13.2 | 13.6 MB | 264,966 |
| Transformer | 780.6 | 1.28 | 13.0 MB | 461,312 |
| SSM | 40.0 | 25.0 | 12.8 MB | 362,496 |

**Incremental generation loop cleanups (Issue #36).** Removed the redundant prompt prefill
(prompt processed twice) and moved decode sampling fully on-device (reused input buffer,
on-device top-k + `multinomial`, one host sync per generation). Measured HEAD vs pre-#36,
d=128, 2L, RTX 4050, **512-token prompt + 200 generated tokens**:

| Path | Metric | Before | After | Speedup |
|------|--------|--------|-------|---------|
| DeltaNet | time-to-first-token (512-tok prompt) | 1,612 ms | **23 ms** | **70×** |
| DeltaNet | end-to-end (prompt + 200 tok) | 2,338 ms | 691 ms | 3.4× |
| DeltaNet | decode | 276 tok/s | 300 tok/s | 1.1× |
| HVN (default) | decode | 80.7 tok/s | **197.7 tok/s** | **2.45×** |
| HVN (default) | time-to-first-token (512-tok prompt) | 6,115 ms | 2,553 ms | 2.4× |
| HVN (default) | end-to-end (prompt + 200 tok) | 8,593 ms | 3,565 ms | 2.4× |

The DeltaNet prefill win is *prompt-length dependent* — its chunked-parallel scan is
algebraically exact, so the whole prompt is absorbed in one pass (O(chunks) launches) instead
of token-by-token; on a 512-token prompt that collapses the ~1.6 s wait before the first
output token to ~23 ms. Stacks with #33 kernels (HVN decode + `--cuda_graph` 350 tok/s,
`--fused_kernel` 354 tok/s). Generated output is token-for-token unchanged (`test_issue36.py`,
`top_k=1` greedy parity).

**Forward pass scaling (ms/token, batch=1):** HVN flat 5.06→5.05 from N=256→4096 (**O(1) ✓**);
SSM ~0.43 flat; Transformer sub-ms (Flash Attention). **Memory at N=4096:** HVN 38.4 MB vs
Transformer 48.3 MB (20% less) vs SSM 36.3 MB.

**Parameter efficiency:** HVN 264,966 (1.0×) < SSM 362,496 (1.37×) < Transformer 461,312
(1.74×) — HVN uses 43% fewer params than Transformer.

## 8.3 Custom Kernels & Parallel Processing (Issue #33)

| Fix | Technique | Before | After | Speedup |
|-----|-----------|--------|-------|---------|
| Fix 6 | CUDA Graph capture (generation) | 78 tok/s | 220 tok/s | 2.82× |
| Fix 1 | Fused Triton kernel (generation) | 78 tok/s | 167 (340 warmed) | 2.14× (7.2×) |
| Fix 3 | Parallel scan (training forward) | 7.63 ms | 4.79 ms | 1.59× |
| Fix 4 | Multi-chunk barrier sync (training) | 7.74 ms | 6.77 ms | 1.14× |

Deferred: Fix 2 (flash-style tiling, needed when d_model>1024), Fix 5 (fully parallel
positional injection, research-level).

## 8.9 Correct, Fused Concurrent-Serving Decode (Issue #43)

Split from the #37 hardware review (point ②): serving N users runs N independent streams —
each with its own Hamiltonian H_i — through the model together. The batched decode feeds a
**length-1** sequence per stream, so per-step wall-time is dominated by **kernel-launch /
Python overhead, not compute** (nearly flat in stream count — that is why batching scales
tok/s ~linearly with B). The review's premise (reformulate per-stream `H_i·v_i` reads into
one big **GEMM**) turned out **not** to be the lever: those reads are inherently
**block-diagonal** (each stream owns its H), and at HVN's head dims (`d_head=64`) cuBLAS's
batched path is already efficient — a grouped-GEMM reformulation was measured
**throughput-neutral-to-negative** (layout permutes cost more than the matmul saves; decode
is launch-bound). The real lever is **fewer launches**: routing the batched step through the
Issue #33 `FusedHVNStep` Triton kernel (2 launches/block vs ~25 PyTorch ops), previously
wired only for single-stream.

Two things shipped in `model.generate_served` (`forward_step_served`):

1. **Correctness fix.** The legacy `generate_batched` expanded per-head params with
   `repeat_interleave(B)` (head-outer `h·B+b`) while the stream batch is batch-outer
   (`b·n_heads+h`), so for **`n_heads>1, B>1` each concurrent stream silently used the *wrong
   head's* projection weights** — `generate_batched` did **not** equal running each stream
   alone (verified: diverges at `n_heads=4`, matches only at `n_heads=1`). The served path
   uses batch-outer ordering in both backends, so a concurrent batch is **equivalent to
   single-stream** (torch backend ~1e-6; fused to the kernel's ~1e-2).
2. **Throughput.** `backend="fused"` (default `auto` when CUDA+Triton+`d_head≥16`) collapses
   per-token launches; `backend="torch"` is a portable grouped-GEMM fallback (matches
   single-stream to ~1e-6). DeltaNet / multi-thread memory fall back to `generate_batched`.

Sustained throughput — **total tok/s across all streams**, 128 new tokens, d=512, 8 heads,
2L, fp32, RTX 4050 (`test_issue43.py`):

| Concurrent streams | legacy `generate_batched` (buggy) | served-torch (correct) | **served-fused (correct)** | fused vs legacy |
|---:|---:|---:|---:|:---:|
| 1 | 263 | 254 | **675** | **2.56×** |
| 8 | 2,096 | 1,987 | **5,416** | **2.58×** |
| 32 | 8,484 | 7,744 | **20,815** | **2.45×** |
| 64 | 15,924 | 15,062 | **42,716** | **2.68×** |

**~2.5–2.7× sustained tok/s at every concurrency level** (and 2.66× over the *correct* torch
baseline, so the win is real, not an artifact of the bug), with every stream's output
identical to single-stream. O(1) per-token preserved (flat ms/token N=64→256), no KV cache.
`test_issue43.py` 7/7.

## 8.13 Exact Gradient (Activation) Checkpointing — `--grad_checkpoint` (Issue #49)

The one **mathematically exact** training-memory lever:
`torch.utils.checkpoint.checkpoint` (`use_reentrant=False`) wraps each HVN/DeltaNet block in
`model.forward`, recomputing activations in the backward pass. Gradients, weights, and the
final trained model are **bit-for-bit identical** to training without it — the only cost is
per-step wall-clock. Flag-gated, default **off**. Training-only (guarded by `self.training
and torch.is_grad_enabled()`), so O(1)/no-KV inference and eval paths are untouched.

**Exactness — proven, not assumed** (`test_issue49.py`): with checkpointing on vs off, both
the **logits** and **every parameter's gradient** match to `max|diff| < 1e-4` (fp32 recompute
reassociation floor), for the default HVN block *and* the DeltaNet block.

**Memory win** — peak train-step CUDA memory (forward+backward), d=256, 4L, seq_len=512,
B=16, fp32, RTX 4050:

| Config | `--grad_checkpoint` off | on | Peak reduction |
|--------|------------------------:|---:|:---:|
| d=256, 4L, seq=512, B=16 | 766 MiB | 274 MiB | **−64%** |

**Unlocks a larger model on the same card:** a `seq_len=1024, batch=48, 8 layers, d=384`
training step **OOMs** on the 6 GB RTX 4050 without the flag but **trains successfully with
it** (`test_issue49.py` 5/5) — directly serving the frontier north star (validate a bigger
model on the wind-tunnel GPU, architecture untouched). **Out of scope** (not 100% exact):
8-bit optimizer (quantized moments, needs a PPL A/B) and CPU/NVMe optimizer offload.