# Execution Plan: Issues #32 & #33

> **Moved from README:** §10 Execution Plan (Issues #32 & #33). Roadmap phases are in
> [roadmap.md](roadmap.md); measured kernel/compression numbers in
> [benchmarks_speed.md](benchmarks_speed.md).
>
> Created 2026-07-03. **#33 closed** (4/6 fixes). Decision: **#33 first, then #32.**

**Why #33 first:** speed unlocks 5× faster iteration for #32 benchmarking; 4/6 fixes are
weight-representation-independent; smaller scope; result is a fast usable model. **The one
dependency:** #32 changes weight representation (fp16 → ternary/2-bit/tensor-decomposed) and
#33's Triton kernel operates on weights — mitigated by making the kernel dtype-agnostic from
the start (fp16/int8/ternary branches).

## #33 — Custom Kernels & Parallel Processing (goal 106 → 500+ tok/s)

- **Phase 1 (Fix 6):** CUDA Graph capture → 78 → 220 tok/s (2.82×), `--cuda_graph`.
- **Phase 2 (Fix 1, 3):** Fused Triton kernel → 167–340 tok/s (2.14–7.2×), `--fused_kernel`;
  parallel scan → 4.79 ms/fwd (1.59×), zero logit diff, `--use_parallel_scan`.
- **Phase 3 (Fix 4):** Multi-chunk barrier sync → 6.77 ms/fwd (1.14×).
- Verified: `test_issue33.py` 5/5 on GPU; `demo.py`/`test_quantize.py` regressions pass.

## #32 — Extreme Weight Compression (10B → <2GB, 1T → <200GB)

- **Phase 1 — Foundational (done):** `hvn/compress.py` — `GPTQQuantizer` (Hessian-based
  4-bit, error compensation, Frantar et al. 2022), `HadamardQuantizer` (QuIP#-style random-sign
  rotation), `RateDistortionAllocator` (water-filling bit allocation); `hvn/sparse.py` —
  `BlockSparseLinear` (2:4 structured sparsity). End-to-end: GPTQ 4-bit gave **5.33× weight
  compression at +0.02% PPL**. `test_issue32_phase1.py` 5/5.
- **Phases 2–5 — `edge-devices` branch (done):** separate `hvn/edge/` sub-package (kept off
  `hvn/core` because ternary training / weight sharing change training dynamics). Modules:
  `ternary.py` (BitNet b1.58 STE), `decomposition.py` (LowRankSparse, TTMatrix, MERAMatrix,
  Kronecker/Circulant/Toeplitz), `shared.py` (ALBERT-style sharing), `entropy.py` (Huffman;
  LDPC documented alias), `transform.py` (Haar wavelet, DCT), `holographic.py`, `streaming.py`
  (NVMe layer streaming, mmap, succinct), `recovery.py` (pruners, Lloyd-Max, QLoRA recovery),
  `model.py` (`EdgeHVN`). End-to-end: EdgeHVN (4 shared layers, 3.62× param reduction) + GPTQ
  4-bit → **2.33× additional compression at +0.37% PPL**. `test_edge_devices.py` 12/12.
  *Scope:* some techniques are simplified-but-functioning versions (e.g. TopologicalPruner
  is a union-find connectivity sweep, not full persistent homology).

## Key Constraints (all issues)

GPU required (RTX 4050, CUDA 12.6); **O(1) inference must be preserved** (no KV cache, no
N-dependent memory); backward compatible (CLI-flag gated, defaults unchanged); PPL must not
regress >2% per fix; Triton kernel dtype-agnostic.
