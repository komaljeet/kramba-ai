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
- [x] Add per-head **learned vorticity projections** with different initialization (Issue #16, closed)
  - File: `hvn/core/vortex_injection.py`, `hvn/core/batched_vortex.py`
  - Per-head `vorticity_frequency` parameter (geometric init: 2^h) scales rotary embeddings
  - High-frequency heads capture local patterns; low-frequency heads capture global structure

### 4.4 Residual Hamiltonian Connection
- [x] Add a **skip connection** from the input Hamiltonian to the output (Issue #13, closed)
  - H_new = H_old + ΔH (was: (1-η)·H + η·(P-H·P·H))
  - File: `hvn/core/batched_memory.py` — `use_residual=True` flag

### 4.5 Learned Entanglement Rate per Head
- [x] Make `η` (entanglement rate) **head-specific and input-dependent** (Issue #14, closed)
  - η_t = σ(W_η · x_t) — input-dependent gating (like Mamba selective scan)
  - File: `hvn/core/batched_memory.py` — `use_input_dependent_eta=True` flag

### 4.6 Normalization Improvements
- [x] Add **RMSNorm** instead of LayerNorm (Issue #15, closed)
  - RMSNorm class in `hvn/core/hvn_block.py` — no mean subtraction, faster than LayerNorm
  - Parallel residual: `x = x + vortex(norm(x)) + ffn(norm(x))` (GPT-J / PaLM style)
  - `forward_step()` updated to use single `self.norm` (was `norm1`/`norm2`)
  - Demo verified, benchmark: HVN PPL 14.74 vs Transformer PPL 8.49 (Shakespeare, 10ep)

---

## Phase 5: Speed & Kernel Optimizations (🔄 In Progress — 4/6 fixes done, Issue #33 closed)

### 5.1 Custom Triton Kernel for Hamiltonian Update
- [x] Write a **fused Triton kernel** that combines vortex injection + Hamiltonian update (Issue #33 Fix 1, closed)
  - File: `hvn/kernels/hamiltonian_kernel.py` — `fused_vortex_step`, `fused_hamiltonian_update`, `FusedHVNStep`
  - Fuses vorticity projection + RoPE + Biot-Savart + gate + output + LayerNorm into 1 kernel
  - Fuses velocity normalization + projector + H·P·H + entanglement + Frobenius clamp into 1 kernel
  - 25 PyTorch ops → 2 Triton kernel launches per token
  - Dtype-agnostic (fp16/fp32/int8/ternary) for Issue #32 forward-compatibility
  - CLI flag: `--fused_kernel` (generate.py)
  - Result: 78 → 167 tok/s (2.14×), up to 340 tok/s with warmup (7.2×)

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
- [x] Optimize the `forward_step` path for batched generation
  - Currently: processes 1 token at a time
  - Proposed: batch multiple sequences with different Hamiltonian states
  - File: `hvn/core/model.py`
  - Expected impact: better GPU utilization for multi-sequence generation

### 5.6 CUDA Graph Capture for Incremental Generation
- [x] Use `torch.cuda.CUDAGraph` to capture the per-token computation graph and replay it (Issue #33 Fix 6, closed)
  - File: `hvn/core/model.py` — `CUDAGraphRunner` class, `generate_incremental_graph()` method
  - Eliminates ~800 kernel launches → 1 graph replay per token
  - CLI flag: `--cuda_graph` (generate.py)
  - Result: 78 → 220 tok/s (2.82×)

### 5.5 Parallel Hamiltonian Processing (⬜ Major — Eliminate Sequential Bottleneck)

This is the **single biggest speed bottleneck** in HVN. Currently the forward pass
loops over tokens sequentially: `for t in range(S)`. This kills GPU utilization
because GPUs are designed for massive parallelism, not sequential loops.

There are **4 strategies** to parallelize, ordered by implementation difficulty:

#### Strategy A: Larger Chunk Size (Easy — small code change)
- [x] Increase `chunk_size` from 16 to 64 (Issue #4/#7, closed)
  - File: `hvn/core/hvn_block.py` — default chunk_size=64, adaptive fallback

#### Strategy B: Parallel Scan with Associative Hamiltonian Update (Medium — algorithmic change)
- [x] Reformulate the Hamiltonian update as an **associative scan** (parallel prefix sum) (Issue #33 Fix 3, closed)
  - Implemented in `parallel_update()`, `hvn/core/hamiltonian_memory.py` and `hvn/core/hvn_block.py`
  - Correction pass re-applies the `H·P·H` destructive interference term after the linear scan
  - **Zero logit diff vs sequential** (exact match) — the linearization risk below did not materialize
  - CLI flag: `--use_parallel_scan` (train.py, quality_benchmark.py)
  - Result: training forward 7.63 → 4.79 ms (1.59×)
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
- [x] Process **multiple chunks in parallel** with periodic barrier synchronization (Issue #33 Fix 4, closed)
  - Implemented in `parallel_forward()` (`hvn/core/hvn_block.py`) and `forward_parallel()` (`hvn/core/model.py`)
  - Result: training forward 7.74 → 6.77 ms (1.14×)
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

| Strategy | Parallelism | Speedup (expected) | Speedup (measured) | Quality Risk | Difficulty | Status |
|----------|------------|---------|---------|-------------|------------|--------|
| A: Larger chunks | Within chunk | 2-4× | — | Low (stale H) | Easy | ✅ Done (#4/#7) |
| B: Associative scan | O(log S) depth | 10-50× | 1.59× (training fwd) | Medium (linearized) | Medium | ✅ Done (#33 Fix 3) — zero logit diff |
| C: Multi-chunk barrier | Across chunks | K× | 1.14× (training fwd) | Medium (no cross-chunk) | Medium | ✅ Done (#33 Fix 4) |
| D: Fully parallel | All tokens | 100×+ | — | High (loses causality) | Hard | ⬜ Deferred (research) |

#### Recommended Implementation Order
1. ~~**Strategy A** first (1 hour — just change chunk_size, benchmark the tradeoff)~~ ✅ Done (#4/#7)
2. ~~**Strategy B** second (1-2 days — implement parallel scan, biggest bang-for-buck)~~ ✅ Done (#33 Fix 3)
3. ~~**Strategy C** third (2-3 days — multi-chunk with barrier merge)~~ ✅ Done (#33 Fix 4)
4. **Strategy D** last (research — needs experimentation to validate quality) — still deferred

---

## Phase 7: Inference Optimization (✅ Complete)

### 7.4 Hamiltonian Quantization
- [x] Add **INT8 and 4-bit NF4 quantization** for Hamiltonian matrices (Issue #20, closed)
  - File: `hvn/quantize.py` (new)
  - INT8 symmetric quantization (per-tensor and per-channel): ~4× memory reduction, <1% relative error
  - NF4 (NormalFloat 4-bit) block-wise quantization: ~7-8× memory reduction, <10% relative error
  - `QuantizedHamiltonian` wrapper for storing/dequantizing Hamiltonian state tensors
  - `QuantizedHVNInference` wrapper for quantized incremental + batched generation
  - `quantize_model_hamiltonians()` for quantizing model parameters
  - `estimate_memory_savings()` for computing expected compression ratios
  - Test suite: `test_quantize.py` — 8 tests, all passing on GPU
  - Results: INT8 73.4% memory savings, NF4 85.9% memory savings (d_model=256, 4 layers, 8 heads)

---

## Phase 6: Evaluation & Scaling (✅ Complete)

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
- [x] Train HVN at d_model = {64, 128, 256, 512} and plot loss curves (Issue #18, closed)
- [x] Compare scaling efficiency: how does PPL improve with parameters?
- [x] Determine the **crossover point** where HVN beats Transformer on wall-clock time
  - **Result**: No crossover found within tested sizes — HVN slower at all d_model values
  - HVN scaling exponent: PPL ∝ params^-0.007 (nearly flat — poor scaling)
  - Transformer scaling exponent: PPL ∝ params^-0.090 (steep improvement)
  - Key finding: HVN's rank-1 entanglement update lacks expressiveness to benefit from scale
- File: `eval/scaling_laws.py` (implemented, with matplotlib plots + JSON output)

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
- [x] Add mixed-precision (fp16/bf16) training support (Issue #19, closed)
  - `--precision fp32|fp16|bf16` flag in train.py, quality_benchmark.py, eval/run_eval.py, eval/scaling_laws.py
  - Uses `torch.amp.autocast` + `GradScaler` (fp16) or autocast-only (bf16)
  - `--grad_accum N` flag for gradient accumulation (effective batch = batch_size × N)
  - Fixed `torch.linalg.qr` dtype issue in batched_memory.py and hamiltonian_memory.py (upcast to fp32 inside autocast-disabled context)
- File: `hvn/distributed.py` (new)

### 7.4 Quantization
- [ ] Add INT8 quantization for the Hamiltonian matrices (Issue #20, open)
- [ ] Add 4-bit quantization for inference
- File: `hvn/quantize.py` (new)

---

## Phase 8: Fine-Tuning & Instruction Tuning (✅ Complete)

### 8.1 LoRA (Low-Rank Adaptation)
- [x] Add **LoRA adapters** to all nn.Linear layers in HVN (Issue #29, closed)
  - `LoRAAdapter`: ΔW = B@A, zero-init B, scaling = α/r
  - `LoRALinear`: frozen base + LoRA, y = W_frozen(x) + lora(x)
  - `apply_lora()`: injects LoRA into model, freezes base weights
  - `merge_lora_weights()`: folds LoRA back into base (zero inference overhead)
  - File: `hvn/lora.py` (new)

### 8.2 QLoRA (Quantized LoRA)
- [x] Add **QLoRA** — NF4 quantized base + LoRA adapters (Issue #29, closed)
  - `QLoRALinear`: stores NF4/INT8 quantized weight, dequantizes on-the-fly
  - `QLoRAConfig`: configuration container
  - `apply_qlora()`: quantizes base model + applies LoRA
  - Enables fine-tuning larger models on 6GB VRAM

### 8.3 Instruction Tuning
- [x] Add **instruction-formatted training data** (Issue #29, closed)
  - `generate_instruction_training_data()`: synthetic Alpaca-format examples
  - `format_instruction_example()`: ### Instruction / ### Input / ### Response format
  - `train.py`: `--instruction` and `--instruction_examples` CLI flags
  - `eval/instruction.py`: instruction evaluation harness (PPL + following accuracy)

### 8.4 LoRA Save/Load
- [x] Add **LoRA checkpoint save/load** (Issue #29, closed)
  - `save_lora_checkpoint()`: saves only trainable LoRA weights (small checkpoint)
  - `load_lora_checkpoint()`: loads LoRA weights into model with LoRA applied
  - `train.py`: `--merge_lora` flag to merge after training

### 8.5 Demo
- [x] `demo_lora.py`: 11-test demo verifying all components (all pass on GPU)
  - Tests: LoRAAdapter, LoRALinear, apply_lora, LoRA training, merge, QLoRA,
    instruction data, instruction tuning, save/load, O(1) inference, VRAM usage

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

### Custom Kernels & Parallel Processing Results (Issue #33, closed 2026-07-03)

| Fix | Technique | Before | After | Speedup |
|-----|-----------|--------|-------|---------|
| Fix 6 | CUDA Graph capture (generation) | 78 tok/s | 220 tok/s | 2.82× |
| Fix 1 | Fused Triton kernel (generation) | 78 tok/s | 167 tok/s (340 warmed) | 2.14× (up to 7.2×) |
| Fix 3 | Parallel scan (training forward) | 7.63 ms/fwd | 4.79 ms/fwd | 1.59× |
| Fix 4 | Multi-chunk barrier sync (training forward) | 7.74 ms/fwd | 6.77 ms/fwd | 1.14× |

See `docs/BENCHMARK_RESULTS.md` § 5 for full details. Fixes 2 and 5 deferred (Phase 5 of `EXECUTION_PLAN.md`).

### Priority Order for Next Changes

1. ~~Larger chunk size~~ ✅ Done (Issue #4/#7)
2. ~~Rotary positional encoding~~ ✅ Done (Issue #1)
3. ~~torch.compile~~ ✅ Done (Issue #10) — 14x per-batch speedup
4. ~~Input-dependent entanglement rate~~ ✅ Done (Issue #14)
5. ~~BPE tokenizer~~ ✅ Done (Issue #11)
6. ~~Higher-rank Hamiltonian update~~ ✅ Done (Issue #12)
7. ~~Residual Hamiltonian connection~~ ✅ Done (Issue #13)
8. ~~Real dataset evaluation~~ ✅ Done (Issue #2)
9. ~~RMSNorm + parallel residual~~ ✅ Done (Issue #15)
10. ~~Multi-head vorticity frequencies~~ ✅ Done (Issue #16)
11. ~~Batched generation~~ ✅ Done (Issue #17)
12. ~~Scaling laws evaluation~~ ✅ Done (Issue #18)
13. ~~Mixed-precision training~~ ✅ Done (Issue #19)
14. ~~Quantization~~ ✅ Done (Issue #20)
15. **Multi-strategy quality upgrade** ✅ Done (Issue #21)
    - Strategy 1: Rank-k entanglement update (`--rank_k 4|8`)
    - Strategy 2: Multi-chunk causal re-injection (`--causal_reinjection`)
    - Strategy 3: Cross-head Hamiltonian mixing (`--cross_head_mixing`)
    - Strategy 4: Nonlinear Hamiltonian update (`--nonlinear_update`)
    - Strategy 5: Per-token entanglement rate (`--per_token_eta`)
    - Strategy 6: SwiGLU FFN (`--swiglu`)
16. ~~**Triton kernel**~~ ✅ Done (Issue #33, Fix 1) — 78 → 167 tok/s (up to 340 warmed)
17. ~~**Associative parallel scan**~~ ✅ Done (Issue #33, Fix 3) — training fwd 1.59×, zero logit diff
18. ~~**Multi-chunk barrier sync**~~ ✅ Done (Issue #33, Fix 4) — training fwd 1.14×
19. ~~**CUDA Graph capture**~~ ✅ Done (Issue #33, Fix 6) — 78 → 220 tok/s
20. **Flash-style tiled Hamiltonian computation** ⬜ Deferred (Issue #33, Fix 2 — needed only when d_model > 1024)
21. **Extreme weight compression** ⬜ Open (Issue #32 — 10B → <2GB, 1T → <200GB)
22. **Fix readout/update dynamics for PPL** ⬜ Open (Issue #34) — the real quality lever
23. **Real-scale scaling re-measurement** ⬜ Open (Issue #35) — WikiText-103
24. **Generation-speed cleanups** ⬜ Open (Issue #36) — redundant prefill + host-device syncs

---

## Phase 9: PPL Root-Cause Correction (⬜ Planned — Issues #34/#35/#36, 2026-07-04)

> **Diagnosis update**: The long-standing "d×d matrix is a fundamental compression
> bottleneck" explanation for HVN's PPL gap is contradicted by our own benchmark —
> the SSM baseline reaches PPL ~1.1 with a state *smaller* than HVN's d×d-per-head
> Hamiltonian. Fixed-size O(1) state is **not** the limiter. HVN is really a
> linear-attention fast-weight network (`v=Γ·Hω` read, `H←(1-η)H+η(P−HPH)` delta-rule
> update), so the fix is to borrow proven dynamics from DeltaNet / GLA / Mamba-2 / RWKV-7,
> not to add more capacity (which #21/#22 already showed moves PPL <3%).

### 9.1 Readout & update dynamics (Issue #34)
- [ ] Replace velocity L2-norm + Frobenius clamp with a **per-channel data-dependent
      decay gate** + a **normalizer/denominator** on the readout (prime suspect for the 8→1 gap)
- [ ] Adopt **canonical DeltaNet update** with **separate k/v projections** (drop symmetric `HPH`)
- [ ] Add **output gate + short causal depthwise conv** on the read path
- Target: eval PPL ≥30% better than the 8.13 baseline at equal/lower FLOPs, O(1) preserved

### 9.2 Real-scale scaling validation (Issue #35)
- [ ] Wire WikiText-103 (≥100M tokens) into `eval/scaling_laws.py`; compute-matched sweep
- [ ] Refit HVN/Transformer/SSM exponents; the current -0.007 is from 34K chars and untrustworthy

### 9.3 Generation-speed cleanups (Issue #36)
- [ ] Remove the dead full-`forward()` prompt prefill in `generate_incremental`
- [ ] Batch/chunk the prompt prefill; cut per-token tensor alloc + `.item()` syncs