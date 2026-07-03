# Execution Plan: Issues #32 & #33

> Created: 2026-07-03
> Status: **#33 closed (2026-07-03)** — Phases 1-3 done (4/6 fixes). Phase 4 (#32) not yet started. Phase 5 (Fixes 2 & 5) deferred.
> Decision: **#33 first**, then #32

---

## Issue Summary

| Issue | Title | Goal | Scope |
|-------|-------|------|-------|
| **#33** | Custom Kernels & Parallel Processing | 106 → 500+ tok/s | 6 fixes, mostly modifying existing files |
| **#32** | Extreme Weight Compression | 10B → <2GB, 1T → <200GB | 15 techniques, 5 phases, 6 new files |

---

## Why #33 First

1. **Speed unlocks faster iteration** — at 500+ tok/s, benchmarking and evaluating #32's compression techniques is 5× faster
2. **4 of 6 fixes are weight-representation-independent** — no rework needed when #32 changes weight formats
3. **Smaller scope** — days of work vs weeks for #32
4. **Result is a fast, usable model** to test compression against

## The One Dependency

```
#32 changes weight representation (fp16 → ternary/2-bit/tensor-decomposed)
#33's Triton kernel operates on weights
```

**Mitigation**: Design the Triton kernel (Fix 1) to be dtype-agnostic from the start:
```python
if weight_dtype == fp16:
    result = matmul(H, omega)
elif weight_dtype == ternary:
    result = ternary_matmul(H, omega)  # add/subtract only
elif weight_dtype == int8:
    result = int8_matmul(H, omega)     # integer matmul + scale
```

This way kernel infrastructure is built once; #32 just adds new dtype branches.

---

## Execution Order

### Phase 1: Quick Win (~1 day) — ✅ Done (2026-07-03)
**#33 Fix 6: CUDA Graph Capture for Incremental Generation**
- **What**: Use `torch.cuda.CUDAGraph` to capture the per-token computation graph, replay it for each generated token
- **Why first**: Zero weight dependency, easy to implement, 2-3× immediate speedup
- **Files**: `hvn/core/model.py` — `CUDAGraphRunner` class, `enable_cuda_graph()` and `generate_incremental_graph()` methods
- **Expected**: 106 → 250-300 tok/s
- **Result**: 78 → 220 tok/s (**2.82×**) — CLI flag `--cuda_graph` (generate.py)
- **Difficulty**: Easy-Medium
- **Verified**: `benchmark.py` updated with kernel-enabled vs baseline comparison; output identical to non-graph version

### Phase 2: Biggest Impact (~2-3 days) — ✅ Done (2026-07-03)
**#33 Fix 1: Fused Triton Kernel for Vortex Injection + Hamiltonian Update**
- **What**: Single Triton kernel that fuses the entire per-token HVN computation (vorticity → velocity → deflection → projector → Hamiltonian update → SVD)
- **Why second**: Eliminates ~800 kernel launches → 1 per token, 5-10× speedup
- **Files**: `hvn/kernels/hamiltonian_kernel.py` (new: `fused_vortex_step`, `fused_hamiltonian_update`, `FusedHVNStep`), `hvn/core/hvn_block.py` (`forward_step_fused()`), `hvn/core/model.py` (`generate_incremental_fused()`)
- **Expected**: 250-300 → 500-1000 tok/s
- **Result**: 78 → 167 tok/s (**2.14×**), up to 340 tok/s with warmup (**7.2×**) — CLI flag `--fused_kernel` (generate.py)
- **Difficulty**: Medium-Hard
- **Critical**: Kernel accepts dtype parameter (fp16/fp32/int8/ternary) for #32 forward-compatibility — done
- **Dependencies**: `triton-windows` (already installed)
- **Verified**: `test_issue33.py` correctness tests pass on GPU; within 0.06 numerical tolerance vs reference PyTorch implementation

**#33 Fix 3: Parallel Scan with Associative Hamiltonian Update**
- **What**: Reformulate Hamiltonian update as associative scan (parallel prefix sum) to eliminate sequential loop for training
- **Why third**: 10-50× training speedup, algorithmic change (no weight dependency)
- **Files**: `hvn/core/hamiltonian_memory.py` and `hvn/core/batched_memory.py` — `parallel_update()`, `hvn/core/hvn_block.py` — `use_parallel_scan` flag
- **Expected**: Training 4.0 → 0.5-1.0 s/epoch
- **Result**: Forward pass 7.63 → 4.79 ms/forward (**1.59×**) — CLI flag `--use_parallel_scan` (train.py, quality_benchmark.py)
- **Difficulty**: Medium
- **Risk**: Linearized approximation drops H·P·H term — **did not materialize**: correction pass re-applies it, zero logit diff vs sequential (exact match)
- **Verified**: PPL/logits exact match vs sequential version, training speed benchmark

### Phase 3: Complementary (~1-2 days) — ✅ Done (2026-07-03)
**#33 Fix 4: Multi-Chunk Parallel with Barrier Sync**
- **What**: Split sequence into K chunks, process in parallel, merge Hamiltonian deltas at barrier
- **Why fourth**: Complements parallel scan, additional K× training speedup
- **Files**: `hvn/core/hvn_block.py` — `parallel_forward()` method, `hvn/core/model.py` — `forward_parallel()`
- **Expected**: Additional 2-4× training speedup
- **Result**: Forward pass 7.74 → 6.77 ms/forward (**1.14×**)
- **Difficulty**: Medium
- **Verified**: PPL within tolerance of sequential, training speed benchmark

**Verification summary**: `test_issue33.py` — 5/5 tests pass on GPU (RTX 4050). Backward compatible — `demo.py` and `test_quantize.py` still pass.

### Phase 4: Start #32 (~weeks) — 🔄 In progress (Phase 1 done 2026-07-03)
**#32 Phase 1: Foundational Compression (10B → 5 GB → 2.5 GB)** — ✅ Done (2026-07-03)
- **4.1** GPTQ-style Hessian-based 4-bit quantization with error compensation
  - File: `hvn/compress.py` — `GPTQQuantizer`
  - Sequential column-wise quantization with inverse-Hessian error compensation
    (Frantar et al. 2022). Reduces Hessian-weighted output distortion by 50%
    vs naive round-to-nearest at the same 4-bit width.
- **4.2** Hadamard pre-rotation before quantization (QuIP#)
  - File: `hvn/compress.py` — `HadamardQuantizer`
  - Random-sign Hadamard rotation spreads outlier weights before 2-bit
    quantization; reduces outlier-driven error by 50% vs unrotated 2-bit.
- **4.3** Per-layer sensitivity-based bit allocation (rate-distortion)
  - File: `hvn/compress.py` — `RateDistortionAllocator`
  - Water-filling allocation from calibration-Hessian-derived sensitivity;
    verified to allocate more bits to high-sensitivity layers while holding
    the requested average bit budget.
- **4.4** 2:4 block-sparse weights with NVIDIA tensor core support
  - File: `hvn/sparse.py` — `BlockSparseLinear`
  - Exact 50% structured sparsity (2 of every 4 weights), drop-in for
    `nn.Linear`, with `convert_model_to_block_sparse()` for whole-model use.
- **Verified**: `test_issue32_phase1.py` — 5/5 tests pass on GPU (RTX 4050).
  End-to-end: GPTQ 4-bit compression of all 11 exercised `nn.Linear` layers
  in a trained HVN model gave **5.33× weight-storage compression with only
  +0.02% perplexity degradation** (target: <5%).
- **Deferred within Phase 1**: int8/2-bit Triton kernel dtype branch (kernel
  already dtype-agnostic per #33 Fix 1; wiring compressed weights through it
  is Phase 2+ work once ternary/2-bit training lands).
- **Next**: Phase 2 (ternary BitNet training, low-rank+sparse decomposition,
  tensor-train Hamiltonian, weight sharing) — not started.

**#32 Phase 2: Advanced Compression (10B → 2 GB)**
- **2.1** Ternary (1.58-bit) weight training (BitNet b1.58)
  - File: `hvn/ternary.py` — `TernaryLinear`, `TernaryHamiltonian`
  - Requires training from scratch with straight-through estimator
- **2.2** Low-rank + sparse weight decomposition (L+S)
  - File: `hvn/compress.py` — `LowRankSparseWeights`
- **2.3** Tensor Train decomposition for Hamiltonian matrices
  - File: `hvn/tensor_train.py` — `TTMatrix`
- **2.4** Weight sharing across HVN layers (ALBERT-style)
  - File: `hvn/compress.py` — `SharedWeightHVN`
- **Update Triton kernel**: Add ternary dtype branch (add/subtract only, no multiplications)
- **Verify**: 10B model at 2 GB, PPL within 5% of fp16

### Phase 5: Scale (ongoing)
**#33 Fix 2: Flash-Style Tiled Hamiltonian Computation** — ⬜ Deferred
- **When**: Needed when d_model > 1024 (large models)
- **What**: Tile the Hamiltonian matmul to fit in SRAM, minimize HBM traffic
- **Files**: `hvn/kernels/flash_hamiltonian.py` (new)
- **Expected**: 2-3× speedup at large d_model, enables d=4096+ without OOM

**#32 Phase 4: Scalability to 1T+**
- **4.1** Layer-by-layer weight streaming from NVMe
  - File: `hvn/streaming.py` — `StreamingHVNInference`
  - VRAM = 1 layer's weights at a time
- **4.2** Memory-mapped compressed weight files (zero-copy)
  - File: `hvn/streaming.py` — `MMapWeightStore`
- **4.3** Succinct data structures for sparse weight indexing
- **4.4** MERA decomposition for ultra-large matrices
- **4.5** On-demand weight loading (task-specific subsets)
- **Verify**: 1T model at <200 GB storage, <16 GB VRAM with streaming

**#33 Fix 5: Fully Parallel with Learned Positional Injection** — ⬜ Deferred
- **When**: Long-term research moonshot
- **What**: Eliminate sequential dependency via learned positional Hamiltonian
- **Files**: `hvn/core/vortex_injection.py`, `hvn/core/hvn_block.py`
- **Expected**: 100×+ speedup, Transformer-level parallelism
- **Risk**: Loses causal structure, needs correction pass, quality unproven

---

## Expected Outcome Timeline

| Phase | Generation Speed (Expected) | Generation Speed (Measured) | Training Speed | 10B Memory | 1T Memory |
|-------|------------------|------------------|----------------|------------|-----------|
| **Current (pre-#33)** | 106 tok/s | 78 tok/s* | 4.0 s/epoch | 20 GB (fp16) | 2 TB |
| **After Phase 1** (Fix 6) | 250-300 tok/s | **220 tok/s (2.82×)** | 4.0 s/epoch | 20 GB | 2 TB |
| **After Phase 2** (Fix 1, 3) | 500-1000 tok/s | **167 tok/s (340 warmed, 2.14-7.2×)** | **4.79 ms/fwd (1.59×)** | 20 GB | 2 TB |
| **After Phase 3** (Fix 4) | 500-1000 tok/s | (gen unaffected) | **6.77 ms/fwd (1.14×, stacks with Fix 3)** | 20 GB | 2 TB |
| **After Phase 4.1** | 500-1000 tok/s | not started | 0.3-0.5 s/epoch | 2.5 GB | 250 GB |
| **After Phase 4.2** | 500-1000 tok/s | not started | 0.3-0.5 s/epoch | **2.0 GB** | **200 GB** |
| **After Phase 5** | 1000+ tok/s | deferred | 0.1-0.2 s/epoch | 2.0 GB | **<200 GB storage, <16 GB VRAM** |

*Baseline measured at 78 tok/s on the benchmark harness used for Issue #33 (differs from the earlier 106.5 tok/s figure in `docs/BENCHMARK_RESULTS.md`, which used a different prompt/token-count configuration).

---

## GitLab Issue Links

- **#32**: https://gitlab.com/komalbarun/kramba-ai/-/issues/32 — Extreme Weight Compression
- **#33**: https://gitlab.com/komalbarun/kramba-ai/-/issues/33 — Custom Kernels & Parallel Processing

## Workflow

For each fix/phase:
1. Create a branch: `git checkout -b issue-33-fix-N-description`
2. Implement the change
3. Verify with a real test or runnable command on GPU
4. Commit with reference: `git commit -m "Fix N: description (issue #33)"`
5. Push: `git push origin <branch-name>`
6. Merge to main
7. Close issue or update progress via `glab issue note`

## Key Constraints

- **GPU required**: All work on RTX 4050 (6 GB VRAM), CUDA 12.6
- **O(1) inference must be preserved**: No technique may introduce a KV cache or N-dependent memory
- **Backward compatible**: All fixes enabled via CLI flags, defaults preserve current behavior
- **PPL must not regress >2%** per fix (except ternary training which may have its own PPL)
- **Triton kernel must be dtype-agnostic**: Support fp16, int8, ternary from the start