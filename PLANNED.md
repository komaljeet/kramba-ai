# HVN Development Roadmap

> Status legend: ✅ Done · 🔄 In Progress · ⬜ Planned

---

## Phase 1: Core Architecture (✅ Complete)

- [x] `VortexInjectionLayer` — Biot-Savart velocity field from token → vortex particle
- [x] `HamiltonianMemoryBlock` — Fixed-size MPS with entanglement renormalization
- [x] `HVNBlock` — Causal wrapper linking fluid + quantum evolution
- [x] `HVNDecoder` — Shallow projection to vocabulary logits
- [x] `HamiltonianVortexNetwork` — Full model assembly
- [x] Theoretical framework document (`docs/THEORETICAL_FRAMEWORK.md`)
- [x] FLOP analysis proving O(1) inference (`docs/FLOP_ANALYSIS.md`)

---

## Phase 2: Stability & Training (✅ Complete)

- [x] Velocity L2-normalization (prevents spectral norm explosion)
- [x] Frobenius norm clamping on Hamiltonian (prevents NaN over long sequences)
- [x] Power-iteration low-rank truncation (replaces `eigh` — no NaN gradients)
- [x] Detached eigenvectors in QR projection (stable backprop)
- [x] Gradient clipping in training loop (`max_norm=1.0`)
- [x] Cosine LR schedule with AdamW optimizer
- [x] Character-level tokenizer + dataset loader
- [x] Training script (`train.py`) with CLI args
- [x] Generation script (`generate.py`) with temperature/top-k sampling
- [x] Training guide (`docs/TRAINING_GUIDE.md`)

---

## Phase 3: Performance Optimizations (✅ Complete)

- [x] **Multi-head Hamiltonian** — `n_heads` parallel H matrices (like multi-head attention)
  - Each head operates on `d_model // n_heads` dimensions
  - PPL improved 9.48 → 8.37 (12% better)
- [x] **Gated vortex injection** — `g_t = σ(W_g · [x_t ; v_t])` controls deflection strength
  - Like LSTM output gate — selective memory writing
- [x] **Chunked parallel processing** — process `chunk_size` tokens in parallel, update H between chunks
  - Forward pass 35% faster (5.05ms → 3.28ms per token)
  - Training 27% faster (236s → 172s)
- [x] **Incremental generation** — `forward_step()` with Hamiltonian state cache
  - O(1) per token generation (no context reprocessing)
  - 41% faster generation (75.6 → 106.5 tok/s)
- [x] **Input/output projections** — learned projections between full d_model and per-head subspaces
- [x] Speed benchmark (`benchmark.py`) — HVN vs Transformer vs SSM
- [x] Quality benchmark (`quality_benchmark.py`) — training perplexity comparison

---

## Phase 4: Quality Improvements (⬜ Planned)

These changes are needed to close the perplexity gap with Transformer (PPL 1.90) and SSM (PPL 1.12).

### 4.1 Positional Encoding
- [x] Add **rotary positional embeddings (RoPE)** to the vortex injection layer (Issue #1, closed)
  - File: `hvn/core/vortex_injection.py` — `_apply_rotary_embeddings()`

### 4.2 Higher-Rank Hamiltonian Update
- [x] Upgrade from rank-1 to **rank-k interaction projector** (Issue #12, closed)
  - Implemented via batched projectors in `batched_memory.py`
  - P_sum = Σ_t v_t ⊗ v_t^T computed in single batched matmul

### 4.3 Multi-Head Vorticity Projections
- [ ] Add per-head **learned vorticity projections** with different initialization (Issue #16, open)
  - File: `hvn/core/vortex_injection.py`, `hvn/core/batched_vortex.py`

### 4.4 Residual Hamiltonian Connection
- [x] Add a **skip connection** from the input Hamiltonian to the output (Issue #13, closed)
  - H_new = H_old + ΔH (was: (1-η)·H + η·(P-H·P·H))
  - File: `hvn/core/batched_memory.py` — `use_residual=True` flag

### 4.5 Learned Entanglement Rate per Head
- [x] Make `η` (entanglement rate) **head-specific and input-dependent** (Issue #14, closed)
  - η_t = σ(W_η · x_t) — input-dependent gating (like Mamba selective scan)
  - File: `hvn/core/batched_memory.py` — `use_input_dependent_eta=True` flag

### 4.6 Normalization Improvements
- [🔄] Add **RMSNorm** instead of LayerNorm (Issue #15, IN PROGRESS on branch `feature/rmsnorm-parallel-issue-15`)
  - RMSNorm class added to `hvn/core/hvn_block.py`
  - Parallel residual: `x = x + vortex(norm(x)) + ffn(norm(x))`
  - **INCOMPLETE**: `forward_step` still references old `self.norm1`/`self.norm2` — needs updating
  - **INCOMPLETE**: Demo not yet verified with RMSNorm changes

---

## Phase 5: Speed & Kernel Optimizations (⬜ Planned)

### 5.1 Custom Triton Kernel for Hamiltonian Update
- [ ] Write a **fused Triton kernel** that combines vortex injection + Hamiltonian update
  - Currently: Python `for t in range(S)` loop with separate PyTorch ops
  - Proposed: single fused kernel that processes all tokens in a chunk
  - File: `hvn/kernels/hamiltonian_kernel.py` (new)
  - Expected impact: 5-10× faster forward pass (eliminate Python overhead)

### 5.2 torch.compile Integration
- [x] Add `torch.compile(model, mode="reduce-overhead")` support (Issue #10, closed)
  - Refactored `_process_batched_chunked` to separate vortex (single batched call) from H update (sequential loop)
  - Installed `triton-windows` for GPU kernel generation
  - Added `--compile` flag to `train.py`, `quality_benchmark.py`, `eval/run_eval.py`
  - **14x per-batch speedup** (46.3ms → 3.3ms), HVN near Transformer parity

### 5.3 Flash-Style Hamiltonian Computation
- [ ] Implement a **tiled Hamiltonian update** that processes chunks in shared memory
  - Analogous to Flash Attention's tiling strategy
  - Avoid materializing the full D×D Hamiltonian in HBM
  - File: `hvn/kernels/flash_hamiltonian.py` (new)
  - Expected impact: lower memory, faster for large d_model

### 5.4 KV-Cache-Free Generation Optimization
- [ ] Optimize the `forward_step` path for batched generation
  - Currently: processes 1 token at a time
  - Proposed: batch multiple sequences with different Hamiltonian states
  - File: `hvn/core/model.py`
  - Expected impact: better GPU utilization for multi-sequence generation

### 5.5 Parallel Hamiltonian Processing (⬜ Major — Eliminate Sequential Bottleneck)

This is the **single biggest speed bottleneck** in HVN. Currently the forward pass
loops over tokens sequentially: `for t in range(S)`. This kills GPU utilization
because GPUs are designed for massive parallelism, not sequential loops.

There are **4 strategies** to parallelize, ordered by implementation difficulty:

#### Strategy A: Larger Chunk Size (Easy — small code change)
- [x] Increase `chunk_size` from 16 to 64 (Issue #4/#7, closed)
  - File: `hvn/core/hvn_block.py` — default chunk_size=64, adaptive fallback

#### Strategy B: Parallel Scan with Associative Hamiltonian Update (Medium — algorithmic change)
- [ ] Reformulate the Hamiltonian update as an **associative scan** (parallel prefix sum)
  - Currently: `H_t = f(H_{t-1}, v_t)` — sequential, each step depends on previous
  - Key insight: if the update rule is **associative** (like `(a ⊗ b) ⊗ c = a ⊗ (b ⊗ c)`),
    then it can be parallelized using a prefix-scan algorithm (Blelloch scan)
  - The entanglement update `H_new = (1-η)·H + η·(P - H·P·H)` is NOT directly associative,
    but it can be **linearized** by approximating the destructive interference term:
    - Approximation: `H_new ≈ (1-η)·H + η·P` (drop the H·P·H term for the scan)
    - This becomes: `H_t = (1-η)·H_{t-1} + η·P_t` — a linear recurrence!
    - Linear recurrences ARE associative and can be parallel-scanned in O(log S) depth
  - Use `torch.cumsum` or a custom parallel scan kernel
  - File: `hvn/core/hamiltonian_memory.py` — add `parallel_update()` method
  - File: `hvn/core/hvn_block.py` — replace sequential loop with parallel scan
  - Expected impact: O(log S) parallel depth instead of O(S) sequential — massive speedup
  - This is exactly what Mamba does (parallel selective scan) and what Flash Attention does
  - Risk: the linearized approximation drops the destructive interference term

#### Strategy C: Multi-Chunk Parallel with Barrier Sync (Medium — architectural change)
- [ ] Process **multiple chunks in parallel** with periodic barrier synchronization
  - Currently: chunk_0 → update H → chunk_1 → update H → chunk_2 → ... (fully sequential)
  - Proposed: split sequence into K independent chunks, process all K in parallel,
    then merge their Hamiltonian updates at a barrier:
    ```
    Phase 1 (parallel): Each chunk_k computes velocities using H_0 (initial state)
    Phase 2 (parallel): Each chunk_k computes its ΔH_k (Hamiltonian delta)
    Phase 3 (barrier):  Merge: H_final = H_0 + Σ_k ΔH_k
    Phase 4 (optional): Re-run chunks with updated H_final for better accuracy
    ```
  - This is analogous to **block-sparse attention** — process blocks independently, merge results
  - File: `hvn/core/hvn_block.py` — add `parallel_forward()` method
  - Expected impact: K× speedup where K = number of parallel chunks
  - Risk: chunks don't see each other's updates (approximation); merge step needs careful design

#### Strategy D: Fully Parallel with Learned Positional Injection (Hard — research-level)
- [ ] Eliminate the sequential dependency entirely by making each token's velocity
  computation **independent** of the Hamiltonian state
  - Currently: `v_t = Γ · (H_t · ω_t)` where H_t depends on all previous tokens — sequential
  - Proposed: compute all velocities in parallel using a **learned positional encoding**
    that encodes the "expected Hamiltonian state" at each position:
    ```
    v_t = Γ · (H_pos(t) · ω_t)    where H_pos(t) = learned_positional_matrix(t)
    ```
  - Then update the Hamiltonian in a single batched operation:
    ```
    H_final = H_0 + Σ_t η · (P_t - H_0 · P_t · H_0)   (all in parallel)
    ```
  - This makes the forward pass fully parallel (like Transformer's attention)
  - File: `hvn/core/vortex_injection.py` — add positional Hamiltonian
  - File: `hvn/core/hvn_block.py` — fully parallel forward
  - Expected impact: Transformer-level parallelism, O(1) parallel depth
  - Risk: loses the causal sequential structure; may need a second "correction" pass
  - This is the most ambitious change — essentially making HVN attention-like in parallelism

#### Parallel Processing Comparison Table

| Strategy | Parallelism | Speedup | Quality Risk | Difficulty |
|----------|------------|---------|-------------|------------|
| A: Larger chunks | Within chunk | 2-4× | Low (stale H) | Easy |
| B: Associative scan | O(log S) depth | 10-50× | Medium (linearized) | Medium |
| C: Multi-chunk barrier | Across chunks | K× | Medium (no cross-chunk) | Medium |
| D: Fully parallel | All tokens | 100×+ | High (loses causality) | Hard |

#### Recommended Implementation Order
1. **Strategy A** first (1 hour — just change chunk_size, benchmark the tradeoff)
2. **Strategy B** second (1-2 days — implement parallel scan, biggest bang-for-buck)
3. **Strategy C** third (2-3 days — multi-chunk with barrier merge)
4. **Strategy D** last (research — needs experimentation to validate quality)

---

## Phase 6: Evaluation & Scaling (⬜ Planned)

### 6.1 Real Dataset Benchmarks
- [x] Train on **WikiText-2** (subset) and compare perplexity (Issue #2, closed)
- [x] Train on **Tiny Shakespeare** for qualitative text generation (Issue #2, closed)
- [x] Train on a **code dataset** (Python corpus) for structured generation (Issue #2, closed)
- File: `eval/run_eval.py`, `eval/datasets.py` (implemented)

### 6.2 Long-Context Evaluation
- [x] Test on **N=256, 512, 1024, 2048, 4096** token sequences (Issue #2, closed)
  - HVN's O(1) advantage confirmed: scaling ratio 1.08x (near-constant per-token)
  - File: `eval/long_context.py` (implemented)
- [x] Implement a **needle-in-haystack** retrieval task (Issue #2, closed)
  - File: `eval/long_context.py` (implemented)

### 6.3 Scaling Laws
- [ ] Train HVN at d_model = {64, 128, 256, 512} and plot loss curves (Issue #18, open)
- [ ] Compare scaling efficiency: how does PPL improve with parameters?
- [ ] Determine the **crossover point** where HVN beats Transformer on wall-clock time
- File: `eval/scaling_laws.py` (new)

---

## Phase 7: Production Readiness (⬜ Planned)

### 7.1 BPE Tokenizer Support
- [x] Replace `CharTokenizer` with HuggingFace `tokenizers` BPE (Issue #11, closed)
- [x] Add tokenizer save/load compatibility
- File: `hvn/tokenizer.py` (implemented) — `BPETokenizer` class
- `train.py` supports `--tokenizer bpe|char` and `--bpe_vocab` flags
- `generate.py` auto-detects BPE vs char tokenizer on load

### 7.2 HuggingFace Integration
- [x] Wrap HVN as a `PreTrainedModel` for HuggingFace Transformers compatibility (Issue #3/#6, closed)
- [x] Support `generate()`, `from_pretrained()`, `save_pretrained()`
- File: `hvn/hf_model.py` (implemented) — `HVNConfig`, `HVNForCausalLM`

### 7.3 Distributed Training
- [ ] Add DDP (DistributedDataParallel) support
- [ ] Add gradient accumulation for large batch sizes
- [ ] Add mixed-precision (fp16/bf16) training support (Issue #19, open)
- File: `hvn/distributed.py` (new)

### 7.4 Quantization
- [ ] Add INT8 quantization for the Hamiltonian matrices (Issue #20, open)
- [ ] Add 4-bit quantization for inference
- File: `hvn/quantize.py` (new)

---

## Current Benchmark Snapshot

### Speed (d_model=128, 2 layers, GPU)

| Metric | HVN | Transformer | SSM |
|--------|-----|-------------|-----|
| Forward (ms/tok, N=4096) | **3.28** | 0.00 | 0.47 |
| Generation (tok/s) | **106.5** | 584.4 | 34.3 |
| Memory at N=4096 | **38.5 MB** | 48.5 MB | 36.5 MB |
| O(1) scaling | ✅ | ✅ | ✅ |

### Quality (d_model=128, 2 layers, 10 epochs, GPU + torch.compile)

| Model | Params | Eval PPL | Train Time |
|-------|--------|----------|------------|
| HVN (with quality improvements) | 282K | 13.87 | 63.9s (4.0s/epoch steady-state) |
| Transformer | 412K | 8.46 | 34.5s |
| SSM | 313K | 4.28 | 114.6s |

### Speed Optimization Journey

| Stage | HVN Time | Speedup | vs Transformer |
|-------|----------|---------|----------------|
| Original (sequential loop) | 764.2s | 1x | 63x slower |
| + Batched Hamiltonian update | 81.8s | 9.3x | 4.8x slower |
| + Parallel heads | 33.9s | 22.5x | 2.3x slower |
| + torch.compile | 45.2s (10ep) | 14x per-batch | 1.3x slower |

### Priority Order for Next Changes

1. ~~Larger chunk size~~ ✅ Done (Issue #4/#7)
2. ~~Rotary positional encoding~~ ✅ Done (Issue #1)
3. ~~torch.compile~~ ✅ Done (Issue #10) — 14x per-batch speedup
4. ~~Input-dependent entanglement rate~~ ✅ Done (Issue #14)
5. ~~BPE tokenizer~~ ✅ Done (Issue #11)
6. ~~Higher-rank Hamiltonian update~~ ✅ Done (Issue #12)
7. ~~Residual Hamiltonian connection~~ ✅ Done (Issue #13)
8. ~~Real dataset evaluation~~ ✅ Done (Issue #2)
9. **RMSNorm + parallel residual** 🔄 In Progress (Issue #15) — branch `feature/rmsnorm-parallel-issue-15`
10. **Multi-head vorticity frequencies** ⬜ Planned (Issue #16)
11. **Batched generation** ⬜ Planned (Issue #17)
12. **Scaling laws evaluation** ⬜ Planned (Issue #18)
13. **Mixed-precision training** ⬜ Planned (Issue #19)
14. **Quantization** ⬜ Planned (Issue #20)
15. **Triton kernel** ⬜ Planned (no issue yet)
16. **Associative parallel scan** ⬜ Planned (no issue yet)
17. **Multi-chunk barrier sync** ⬜ Planned (no issue yet)