# Related Issues & Future Work

> Issues that are related to HVN's quality improvement journey.
> Created for context continuity across sessions.

> **⚠️ Workflow note**: Always create a new branch for each issue
> (e.g. `git checkout -b issue-NN-short-description`).  Never work directly
> on `main`.  After implementation, push the branch, merge to `main, and
> close the issue via `glab issue close NN --repo komalbarun/kramba-ai`.

---

## Open Issues

### #25 — Agentic Mode: Multi-Step Reasoning, Planning, and Self-Correction
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/25
- **Status**: Open
- **Goal**: ReAct-style agentic loop with planning, tool chaining, and self-correction.
- **Depends on**: #24 (Tool Calling)

### #32 — Extreme Weight Compression: 10B → <2GB, 1T → <200GB
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/issues/32
- **Status**: Open
- **Goal**: 15 compression techniques across 5 phases (GPTQ-style quantization, Hadamard
  pre-rotation, ternary weights, tensor-train decomposition, NVMe streaming for 1T+ models).
- **Depends on**: #33 (dtype-agnostic Triton kernel, closed — ready for #32's new dtype branches)
- **See**: `EXECUTION_PLAN.md` for the full phased plan.

### #26 — Training Dashboard: Interactive HTML GUI
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/26
- **Status**: Open
- **Goal**: Single-page HTML dashboard for training config, results visualization, and documentation.
- **Key idea**: No server required — pure HTML/CSS/JS with Chart.js via CDN.
- **Note**: Initial dashboard.html created in docs/ — issue covers refinement and full feature set.

### #27 — Multimodal Fusion: Joint Image-Text Training
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/27
- **Status**: Closed (2026-07-03)
- **Goal**: Joint training on image-text pairs with cross-modal retrieval.
- **Depends on**: #23 (Vision)
- **What was done**: Added multimodal fusion components for joint image-text training:
  - `GatedFusionLayer`: Learned gate mixes image and text representations before HVN processing
  - `ModalityEmbedding`: Learned [IMAGE]/[TEXT] type embeddings for modality distinction
  - `CrossModalRetriever`: Shared embedding space with InfoNCE contrastive loss for cross-modal retrieval
  - `MultimodalFusionHVN`: Full multimodal model with gated fusion, modality embeddings, retrieval head, VQA, and captioning
  - `ImagePreprocessor`: Image preprocessing utilities (resize, normalize)
  - `generate_synthetic_image_text_pairs()`: Synthetic correlated image-text dataset generator
  - `train.py`: `--image_text` flag for joint multimodal training (LM + contrastive loss)
  - `eval/multimodal.py`: Full multimodal evaluation harness (captioning, VQA, retrieval, O(1) verification, memory)
  - `demo_multimodal.py`: 11-test demo verifying all components (all pass on GPU)
  - `hvn/hf_model.py`: `generate_multimodal()` and `answer_question_multimodal()` methods on HVNForCausalLM
  - O(1) inference preserved: Hamiltonian memory is fixed d×d, constant in num_patches and seq_len
  - All tests pass on GPU (RTX 4050)

### #28 — Conversation Memory: Multi-Turn Chat with Persistent State
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/28
- **Status**: Closed (2026-07-03)
- **Goal**: ChatSession with persistent Hamiltonian state, conversation formatting, state checkpointing.
- **Depends on**: #24 (Tool Calling), #25 (Agentic Mode)
- **What was done**: Added multi-turn chat with persistent Hamiltonian state:
  - `ChatSession`: multi-turn chat leveraging `model.chat_step()` — Hamiltonian state
    persists across turns at O(1) per-token cost (no KV cache)
  - `ConversationFormatter`: [SYSTEM]/[USER]/[ASSISTANT] special token formatting
  - `StateCheckpoint`: save/load/fork conversation state (Hamiltonian matrices) to disk
  - Context window management: truncate old turns when exceeding `max_context_tokens`
  - Interactive terminal chat mode with `/save`, `/load`, `/reset`, `/history`, `/fork` commands
  - `generate_chat_training_data()`: synthetic chat-formatted training examples
  - `hvn/tokenizer.py`: BPE trainer includes chat special tokens
  - `train.py`: `CharTokenizer` gains `CHAT_SPECIAL_TOKENS` (6 tokens)
  - `hvn/core/model.py`: `chat()` method (alias for `chat_step`)
  - `generate.py`: `--chat`, `--system_prompt`, `--save_chat`, `--load_chat` CLI flags
  - `demo_chat.py`: 10-test demo verifying all components (all pass on GPU)
  - O(1) inference preserved: Hamiltonian memory is fixed d×d, constant in conversation length

### #29 — Fine-Tuning: LoRA, QLoRA, and Instruction Tuning
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/29
- **Status**: Closed (2026-07-03)
- **Goal**: Parameter-efficient fine-tuning (LoRA/QLoRA) on 6GB VRAM, instruction tuning for agentic features.
- **What was done**: Added LoRA, QLoRA, and instruction tuning support:
  - `LoRAAdapter`: Low-rank adapter ΔW = B@A with zero-init B, scaling = α/r
  - `LoRALinear`: Frozen nn.Linear + LoRA adapter, computes y = W_frozen(x) + lora(x)
  - `QLoRALinear`: NF4/INT8 quantized base weight + LoRA adapter (QLoRA recipe)
  - `apply_lora()`: Injects LoRA into all nn.Linear layers in an HVN model, freezes base
  - `apply_qlora()`: Quantizes base model to NF4 + applies LoRA adapters
  - `merge_lora_weights()`: Folds LoRA deltas back into base weights (zero inference overhead)
  - `freeze_base_model()`, `get_lora_params()`, `count_lora_params()`: utility functions
  - `save_lora_checkpoint()` / `load_lora_checkpoint()`: save/load only LoRA adapter weights
  - `generate_instruction_training_data()`: Synthetic instruction examples (Alpaca format)
  - `format_instruction_example()`: Alpaca prompt formatting (### Instruction / ### Input / ### Response)
  - `hvn/core/model.py`: `apply_lora()`, `apply_qlora()`, `merge_lora()`, `save_lora()`, `load_lora()` methods
  - `train.py`: `--lora`, `--qlora`, `--lora_rank`, `--lora_alpha`, `--lora_dropout`, `--instruction`, `--instruction_examples`, `--merge_lora` CLI flags
  - `eval/instruction.py`: Instruction evaluation harness (PPL + instruction following accuracy)
  - `demo_lora.py`: 11-test demo verifying all components (all pass on GPU)
  - O(1) inference preserved: LoRA adds O(r·d) per layer, dominated by O(d²) Hamiltonian
  - All tests pass on GPU (RTX 4050)

### #33 — Custom Kernels & Parallel Processing
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/issues/33
- **Status**: Closed (2026-07-03)
- **Goal**: 106 → 500+ tok/s via custom CUDA/Triton kernels and parallel Hamiltonian processing.
- **What was done**: 4 of 6 planned fixes implemented and verified on GPU (RTX 4050):
  - **Fix 6** — CUDA Graph capture: `CUDAGraphRunner` class, `generate_incremental_graph()` in
    `hvn/core/model.py`. Eliminates ~800 kernel launches → 1 graph replay per token.
    Result: 78 → 220 tok/s (2.82×). CLI flag: `--cuda_graph` (generate.py)
  - **Fix 1** — Fused Triton kernel: `hvn/kernels/hamiltonian_kernel.py` (new) —
    `fused_vortex_step`, `fused_hamiltonian_update`, `FusedHVNStep`. Fuses 25 PyTorch ops
    into 2 Triton kernel launches per token. Dtype-agnostic (fp16/fp32/int8/ternary) for
    #32 forward-compatibility. Result: 78 → 167 tok/s (2.14×, up to 340 tok/s / 7.2× warmed
    up). CLI flag: `--fused_kernel` (generate.py)
  - **Fix 3** — Parallel scan: `parallel_update()` in `HamiltonianMemoryBlock` and
    `MultiHeadHamiltonianMemory` — linearized recurrence `H_t = α·H_{t-1} + η·P_t` is
    associative (O(log S) depth), with a correction pass re-applying the `H·P·H`
    destructive interference term. Zero logit diff vs sequential. Result: training forward
    7.63 → 4.79 ms (1.59×). CLI flag: `--use_parallel_scan` (train.py, quality_benchmark.py)
  - **Fix 4** — Multi-chunk parallel with barrier sync: `parallel_forward()` in `HVNBlock`,
    `forward_parallel()` in `model.py` — K chunks processed in parallel, ΔH merged at
    barrier. Result: training forward 7.74 → 6.77 ms (1.14×)
  - `test_issue33.py`: 5-test demo verifying all components (all pass on GPU)
  - Backward compatible: `demo.py` and `test_quantize.py` still pass
  - Deferred to Phase 5: Fix 2 (Flash-style tiling, needed only when d_model > 1024) and
    Fix 5 (fully parallel positional injection, research-level moonshot)
  - See `EXECUTION_PLAN.md` and `docs/BENCHMARK_RESULTS.md` § 5 for full details

---

## Closed Issues (Quality Journey)

### #24 — Tool Calling: Structured Output and Function Invocation (CLOSED)
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/24
- **Status**: Closed (2026-07-02)
- **What was done**: Added tool-calling protocol with special tokens, constrained
  decoding, and multi-turn agentic loop.
  - New module: `hvn/tools.py` — `ToolRegistry`, `ToolCallParser`, `ToolExecutor`,
    `ConstrainedJSONDecoder`, `ToolCallingSession`
  - `hvn/tokenizer.py`: BPE trainer includes tool special tokens
  - `train.py`: `CharTokenizer` gains `add_tool_tokens` param (6 special tokens:
    `[TOOL_CALL]`, `[/TOOL_CALL]`, `[TOOL_RESULT]`, `[/TOOL_RESULT]`, `[THINK]`, `[/THINK]`)
  - `train.py`: `--tool_training` and `--tool_examples` CLI flags for tool-calling training data
  - `hvn/core/model.py`: `generate_with_tools()` method with `allowed_token_ids`
    (JSON-constrained decoding) and `stop_token_ids` (stop at `[/TOOL_CALL]`)
  - `generate.py`: `--tools` and `--max_turns` CLI flags for tool-augmented generation
  - `demo_tool_calling.py`: 8-test demo verifying all components (all pass on GPU)
  - Built-in demo tools: `calculator`, `web_search`, `string_reverse`, `word_count`
  - `generate_tool_training_data()`: synthetic tool-call training examples
  - O(1) inference preserved — tool calling is a prompting/training feature,
    not an architectural change

### #23 — Vision: Image Encoding via Vortex-Projected Patch Embeddings (CLOSED)
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/23
- **Status**: Closed (2026-07-02)
- **What was done**: Image patches treated as vortex particles — split image
  into grid, project to d_model, process through HVN blocks.
  - New module: `hvn/vision.py` — `PatchEmbedding`, `PositionalEmbedding2D`,
    `ImageEncoder`, `CrossModalProjector`, `MultimodalHVN`
  - `model.py`: `forward_multimodal()` for image patches + optional text
  - `hf_model.py`: `HVNConfig` gains vision fields, `HVNForCausalLM` gains
    `vision_encoder` + `forward_multimodal()`
  - `train.py`: `--vision` flag for image-text training mode with captioning demo
  - `demo_vision.py`: standalone demo verifying all success criteria
  - O(1) inference preserved: Hamiltonian memory is fixed d×d, constant in
    num_patches and seq_len
  - All tests pass on GPU (RTX 4050)
- **Result**: Image [B,C,H,W] + text [B,S] → logits [B, num_patches+S, V].
  Image captioning (image → text generation) works with O(1) inference.

### #22 — Multi-Thread Hamiltonian Memory (CLOSED)
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/22
- **Status**: Closed (2026-07-02)
- **What was done**: K independent Hamiltonians per head, each updated with a
  different projection of the input.  K "threads" of memory = K different
  views of the sequence.
  - New classes: `MultiThreadHamiltonianMemory`, `MultiThreadVortexInjection`
  - `--n_threads K` CLI flag (default 1 for backward compat)
  - Per-thread input projections, vorticity weights, coupling constants
  - Learned thread mixing weights (like MHA output projection)
  - All existing strategies (rank_k, nonlinear, per_token_eta, etc.) work
    within each thread
- **Result**: PPL improved 8.80 → 8.41 (4.4% better) with n_threads=4 alone.
  With all strategies combined: PPL 8.13 (7.6% better).  Scaling exponent
  improved from -0.0126 to -0.0123.  O(1) inference preserved — memory is
  O(K·d²) per head, still 100× less than KV cache at N=100K.

### #21 — Multi-Strategy Quality Upgrade (CLOSED)
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/21
- **Status**: Closed (2026-07-02)
- **What was done**: 6 strategies implemented:
  1. Rank-k entanglement update (`--rank_k 4|8`)
  2. Multi-chunk causal re-injection (`--causal_reinjection`)
  3. Cross-head Hamiltonian mixing (`--cross_head_mixing`)
  4. Nonlinear Hamiltonian update (`--nonlinear_update`)
  5. Per-token entanglement rate (`--per_token_eta`)
  6. SwiGLU FFN (`--swiglu`)
- **Result**: Scaling exponent improved -0.007 → -0.0126 (1.8× better).
  PPL at d_model=128 unchanged (8.37). PPL at d_model=256 improved 13.82 → 13.67.
  The gap to Transformer remains large — fundamental compression bottleneck.

### #1 — Rotary Positional Embeddings (CLOSED)
- RoPE added to vortex injection layer
- Per-head frequency specialization (Issue #16): geometric 2^h frequencies

### #12 — Higher-Rank Hamiltonian Update (CLOSED)
- Batched projectors in batched_memory.py

### #13 — Residual Hamiltonian Connection (CLOSED)
- H_new = H_old + ΔH (use_residual=True)

### #14 — Learned Input-Dependent Entanglement Rate (CLOSED)
- η_t = σ(W_η · x_t)

### #15 — RMSNorm + Parallel Residual (CLOSED)
- x = x + vortex(norm(x)) + ffn(norm(x))

### #16 — Per-Head Vorticity Frequency Projections (CLOSED)
- Geometric init 2^h, Fourier-style specialization

---

## The Quality Gap Summary

| Model | PPL (Shakespeare, d_model=128, 20ep) | Scaling Exponent |
|-------|--------------------------------------|------------------|
| SSM | ~1.16 | N/A |
| Transformer | ~2.0 | -0.090 |
| HVN (Issue #22 n_threads=4 + all strategies) | ~8.13 | -0.0123 |
| HVN (Issue #22 n_threads=4) | ~8.41 | -0.0123 |
| HVN (Issue #21 strategies) | ~8.37 | -0.0126 |
| HVN (original) | ~8.37 | -0.007 |

**Root cause**: The Hamiltonian compresses all sequence history into a single
d×d matrix per head.  Multi-thread memory (K matrices per head) increases
capacity from O(d²) to O(K·d²), improving PPL by 4-8%.

**Next step**: Further quality improvements may require parallel scan (Phase 5.5
Strategy B) or custom Triton kernels (Phase 5.1, 5.3) for speed.