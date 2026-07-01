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
- [ ] Add **rotary positional embeddings (RoPE)** to the vortex injection layer
  - Currently HVN has no position awareness — tokens are position-agnostic
  - RoPE rotates the vorticity vectors based on position, giving the Hamiltonian spatial structure
  - File: `hvn/core/vortex_injection.py`
  - Implementation: apply rotation matrix `R_θ(pos)` to `ω_t` before Biot-Savart velocity computation
  - Expected impact: significant PPL improvement (position info is critical for language)

### 4.2 Higher-Rank Hamiltonian Update
- [ ] Upgrade from rank-1 to **rank-k interaction projector**
  - Currently: `P_t = v_t ⊗ v_t^T` (rank-1) — limited expressiveness
  - Proposed: `P_t = Σ_{i=1}^{k} v_t^{(i)} ⊗ v_t^{(i)T}` (rank-k)
  - Generate k velocity vectors per token via k parallel vorticity projections
  - File: `hvn/core/hamiltonian_memory.py`
  - Expected impact: more expressive memory updates, better associative recall

### 4.3 Multi-Head Vorticity Projections
- [ ] Add per-head **learned vorticity projections** with different initialization
  - Currently all heads share the same vorticity projection structure
  - Proposed: each head has a distinct vorticity "frequency" (like Fourier features)
  - File: `hvn/core/vortex_injection.py`
  - Expected impact: heads capture different frequency patterns (local vs global)

### 4.4 Residual Hamiltonian Connection
- [ ] Add a **skip connection** from the input Hamiltonian to the output
  - Currently the Hamiltonian is fully overwritten by the entanglement update
  - Proposed: `H_new = H_old + ΔH` where `ΔH` is the renormalization update
  - This preserves long-range information more effectively
  - File: `hvn/core/hamiltonian_memory.py`
  - Expected impact: better long-range recall, less catastrophic forgetting

### 4.5 Learned Entanglement Rate per Head
- [ ] Make `η` (entanglement rate) **head-specific and input-dependent**
  - Currently: single global `η` for all heads and all tokens
  - Proposed: `η_t = σ(W_η · x_t)` — input-dependent gating of memory update strength
  - Like Mamba's selective scan — the model decides how much to remember per token
  - File: `hvn/core/hamiltonian_memory.py`
  - Expected impact: selective forgetting, better handling of irrelevant tokens

### 4.6 Normalization Improvements
- [ ] Add **RMSNorm** instead of LayerNorm (faster, works better with no-bias)
- [ ] Add **parallel residual** architecture (like GPT-J / PaLM) instead of sequential
  - `x = x + vortex(norm(x)) + ffn(norm(x))` — both branches in parallel
  - File: `hvn/core/hvn_block.py`
  - Expected impact: faster forward pass, slightly better gradient flow

---

## Phase 5: Speed & Kernel Optimizations (⬜ Planned)

### 5.1 Custom Triton Kernel for Hamiltonian Update
- [ ] Write a **fused Triton kernel** that combines vortex injection + Hamiltonian update
  - Currently: Python `for t in range(S)` loop with separate PyTorch ops
  - Proposed: single fused kernel that processes all tokens in a chunk
  - File: `hvn/kernels/hamiltonian_kernel.py` (new)
  - Expected impact: 5-10× faster forward pass (eliminate Python overhead)

### 5.2 torch.compile Integration
- [ ] Add `torch.compile(model, mode="reduce-overhead")` support
  - Currently: the Python loop prevents torch.compile from optimizing
  - Proposed: restructure the chunked loop to be compile-friendly
  - File: `hvn/core/hvn_block.py`
  - Expected impact: 2-3× faster training and inference

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
- [ ] Increase `chunk_size` from 16 to 64 or 128
  - Currently: chunk_size=16 means 16 tokens share the same H, then H updates
  - Larger chunks = more parallel tokens per H state = better GPU utilization
  - Tradeoff: tokens within a chunk see a "stale" H (not updated by earlier tokens in the same chunk)
  - File: `hvn/core/hvn_block.py` — change `chunk_size` default
  - Expected impact: 2-4× faster forward pass, small quality tradeoff
  - Risk: quality degrades if chunk_size is too large (H becomes too stale)

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
- [ ] Train on **WikiText-103** (subset) and compare perplexity
- [ ] Train on **Tiny Shakespeare** for qualitative text generation
- [ ] Train on a **code dataset** (e.g., Python functions) for structured generation
- [ ] Compare against Mamba and RWKV (not just Transformer + simple SSM)
- File: `eval/run_wikitext.py`, `eval/run_shakespeare.py` (new)

### 6.2 Long-Context Evaluation
- [ ] Test on **N=10K, 50K, 100K** token sequences
  - HVN's O(1) advantage should dominate here
  - Measure: perplexity, memory, and speed vs Transformer with KV cache
- [ ] Implement a **needle-in-haystack** retrieval task
  - Place a key fact at position N and query it at position N+1000
  - Tests whether the Hamiltonian actually stores long-range associations
- File: `eval/long_context.py` (new)

### 6.3 Scaling Laws
- [ ] Train HVN at d_model = {64, 128, 256, 512} and plot loss curves
- [ ] Compare scaling efficiency: how does PPL improve with parameters?
- [ ] Determine the **crossover point** where HVN beats Transformer on wall-clock time
- File: `eval/scaling_laws.py` (new)

---

## Phase 7: Production Readiness (⬜ Planned)

### 7.1 BPE Tokenizer Support
- [ ] Replace `CharTokenizer` with HuggingFace `tokenizers` BPE
- [ ] Add tokenizer save/load compatibility
- File: `hvn/tokenizer.py` (new)

### 7.2 HuggingFace Integration
- [ ] Wrap HVN as a `PreTrainedModel` for HuggingFace Transformers compatibility
- [ ] Support `generate()`, `from_pretrained()`, `save_pretrained()`
- File: `hvn/hf_model.py` (new)

### 7.3 Distributed Training
- [ ] Add DDP (DistributedDataParallel) support
- [ ] Add gradient accumulation for large batch sizes
- [ ] Add mixed-precision (fp16/bf16) training support
- File: `hvn/distributed.py` (new)

### 7.4 Quantization
- [ ] Add INT8 quantization for the Hamiltonian matrices
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

### Quality (d_model=128, 2 layers, 20 epochs)

| Model | Params | Eval PPL | Train Time |
|-------|--------|----------|------------|
| HVN (multi-head) | 241K | 8.37 | 468s |
| Transformer | 405K | 1.90 | 3.4s |
| SSM | 306K | 1.12 | 69s |

### Priority Order for Next Changes

1. **Larger chunk size** (Phase 5.5-A) — easiest speed win, just change a parameter
2. **Rotary positional encoding** (Phase 4.1) — highest expected impact on quality
3. **Associative parallel scan** (Phase 5.5-B) — biggest speed win, eliminates sequential loop
4. **Input-dependent entanglement rate** (Phase 4.5) — selective memory, Mamba-like
5. **BPE tokenizer** (Phase 7.1) — fewer tokens = faster + better quality simultaneously
6. **Higher-rank Hamiltonian update** (Phase 4.2) — more expressive memory
7. **torch.compile** (Phase 5.2) — easy speed win after parallel scan
8. **Real dataset evaluation** (Phase 6.1) — needed to prove real-world viability
9. **Multi-chunk barrier sync** (Phase 5.5-C) — further parallelism after scan
10. **Triton kernel** (Phase 5.1) — ultimate speed, most engineering effort