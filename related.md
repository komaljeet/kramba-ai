# Related Issues & Future Work

> Issues that are related to HVN's quality improvement journey.
> Created for context continuity across sessions.

> **⚠️ Workflow note**: Always create a new branch for each issue
> (e.g. `git checkout -b issue-NN-short-description`).  Never work directly
> on `main`.  After implementation, push the branch, merge to `main, and
> close the issue via `glab issue close NN --repo komalbarun/kramba-ai`.

---

## Open Issues

### #34 — Close the PPL gap by fixing readout/update dynamics (not state capacity)
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/34
- **Status**: Open (created 2026-07-04)
- **Goal**: Close the PPL gap (HVN ~8.1 vs SSM ~1.1, Transformer ~1.9) by fixing the
  **readout/update dynamics**, not state capacity. Preserve O(1) inference.
- **Key insight (why prior work stalled)**: The "fixed d×d matrix is a fundamental
  compression bottleneck" narrative is contradicted by the project's own benchmark —
  the SSM reaches PPL ~1.1 with a state *smaller* than HVN's d×d-per-head Hamiltonian.
  Fixed-size O(1) state is NOT the limiter. Every capacity-oriented strategy (#21 rank-k,
  #22 multi-thread, multi-head) moved PPL <3% — optimizing the wrong axis.
- **What HVN really is**: a linear-attention fast-weight net. `v_t = Γ·(H·ω_t)` is a
  linear-attention read; `H ← (1-η)H + η(P − HPH)` is a delta-rule update. Borrow proven
  fixes from DeltaNet / GLA / Mamba-2 / RWKV-7.
- **Three levers (highest impact first)**:
  1. Replace the velocity L2-norm + Frobenius clamp (a NaN band-aid that discards all
     magnitude info) with a **per-channel data-dependent decay gate** + a
     **normalizer/denominator** on the readout. *(Prime suspect for the 8→1 gap.)*
  2. Adopt the **canonical DeltaNet update** with **separate k/v projections** (decouple
     read from write; drop the ill-conditioned symmetric `HPH`).
  3. Add an **output gate + short causal depthwise conv** on the read path.
- **Acceptance**: eval PPL improves ≥30% vs 8.13 baseline at equal/lower FLOPs; O(1) preserved.
- **Depends on**: #35 (real-scale validation — gates whether scaling is real).

### #35 — Re-measure HVN scaling on real-scale data (WikiText-103)
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/35
- **Status**: Open (created 2026-07-04)
- **Goal**: Re-measure the scaling exponent on a ≥100M-token corpus before trusting the
  "flat scaling" conclusion.
- **Key insight**: The exponent `PPL ∝ params^-0.007` was measured on **~34K characters**
  of Tiny-Shakespeare — deep in the overfit/saturation regime, so it measures
  regularization noise, not architectural scaling. A scaling law from toy data is not a
  scaling law.
- **Tasks**: wire WikiText-103 into `eval/scaling_laws.py`; compute-matched sweep
  d_model={128,256,512,768} on GPU; refit exponents for HVN/Transformer/SSM; report
  whether "flat scaling" survives.

### #36 — Speed up incremental generation (redundant prefill + host-device syncs)
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/36
- **Status**: Open (created 2026-07-04)
- **Goal**: Remove implementation-level generation overhead (complements #33's kernel work).
- **Concrete bugs/inefficiencies found**:
  1. `generate_incremental` (`hvn/core/model.py` ~L586) runs a **full `self.forward()` over
     the prompt and throws the result away**, then re-processes the prompt token-by-token
     to build states — the prompt is processed twice per call.
  2. Prompt prefill is sequential; should use the batched/chunked forward.
  3. Per-token `torch.tensor([[id]], device=...)` alloc + `.item()` sync stalls the pipeline.
- **Acceptance**: dead forward removed (identical seeded output), prefill batched,
  per-token syncs cut, tok/s improvement reported on GPU and shown stackable with
  `--cuda_graph`/`--fused_kernel`.

### #31 — OpenAI-Compatible API Server: Serve HVN via REST API for ecosystem integration
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/31
- **Status**: Closed (2026-07-03)
- **Goal**: REST API matching the OpenAI schema so HVN works with LangChain, OpenWebUI,
  LibreChat, the OpenAI SDK, curl, etc. Ollama/llama.cpp ruled out (GGUF is
  Transformer-only); custom FastAPI server chosen over vLLM/TGI/sglang (also
  Transformer-only) and LocalAI (unnecessary complexity).
- **What was done**:
  - `hvn/server.py` (new) — FastAPI app with `/v1/chat/completions`,
    `/v1/completions`, `/v1/models`, `/v1/health`
  - `stream_token_ids()` / `generate_text()` / `stream_text_chunks()` — token-by-token
    generator built directly on `forward_step` + the Hamiltonian state cache (same
    algorithm as `model.generate_incremental`, refactored to yield per token for SSE)
  - Streaming responses use Server-Sent Events (`data: {...}\n\n`, terminated by
    `data: [DONE]\n\n`), matching the OpenAI streaming chunk format exactly
  - Chat formatting reuses `hvn.chat.ConversationFormatter` ([SYSTEM]/[USER]/[ASSISTANT])
  - Tool calling (Issue #24): `tools`/`functions` request field is injected into the
    prompt as schema text; any `[TOOL_CALL]` blocks the model emits are parsed via
    `hvn.tools.ToolCallParser` and returned as OpenAI-format `tool_calls` — execution is
    left to the client, matching the OpenAI API's own contract (the API never executes
    functions itself)
  - `--lora` CLI flag applies a LoRA checkpoint (Issue #29) at server startup via
    `apply_lora()` + `load_lora_checkpoint()`
  - `requirements.txt`: added `fastapi`, `uvicorn`
  - `README.md`: new "OpenAI-Compatible API Server" section with usage examples
  - `test_server.py` (new): boots the server as a subprocess against the bundled
    checkpoint and exercises all 4 endpoints (health, models, completions, chat
    completions) plus SSE streaming and tool-call requests — **6/6 checks pass**
  - Manually verified against the real `openai` Python SDK (non-streaming chat,
    streaming chat, text completions) and `curl`
  - O(1) inference preserved — server uses the same Hamiltonian-state-cache generation
    path as `generate_incremental`, no KV cache introduced
  - **Scope note**: single model loaded per server process (no dynamic multi-checkpoint
    hot-swapping) — `/v1/models` lists that one model, which satisfies the OpenAI
    models-list contract without the added complexity

### #25 — Agentic Mode: Multi-Step Reasoning, Planning, and Self-Correction
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/25
- **Status**: Open
- **Goal**: ReAct-style agentic loop with planning, tool chaining, and self-correction.
- **Depends on**: #24 (Tool Calling)

### #32 — Extreme Weight Compression: 10B → <2GB, 1T → <200GB
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/issues/32
- **Status**: Open (Phase 1 of 5 done, 2026-07-03)
- **Goal**: 15 compression techniques across 5 phases (GPTQ-style quantization, Hadamard
  pre-rotation, ternary weights, tensor-train decomposition, NVMe streaming for 1T+ models).
- **Depends on**: #33 (dtype-agnostic Triton kernel, closed — ready for #32's new dtype branches)
- **See**: `EXECUTION_PLAN.md` for the full phased plan.
- **Phase 1 — Foundational Compression (done 2026-07-03)**:
  - `hvn/compress.py` (new) — `GPTQQuantizer` (Hessian-based 4-bit quantization
    with sequential error compensation, Frantar et al. 2022), `HadamardQuantizer`
    (QuIP#-style random-sign Hadamard pre-rotation for 2-bit quantization),
    `RateDistortionAllocator` (water-filling per-layer bit allocation from
    calibration-Hessian sensitivity), plus calibration utilities
    (`gather_layer_inputs`, `collect_calibration_hessian`) and a model-level
    convenience (`compress_linear_layers_gptq`)
  - `hvn/sparse.py` (new) — `BlockSparseLinear` (2:4 structured sparsity,
    `nn.Linear` drop-in), `apply_2_4_sparsity`, `convert_model_to_block_sparse`
  - `test_issue32_phase1.py`: 5-test suite verifying all components on GPU —
    GPTQ reduces Hessian-weighted output distortion 50% vs naive round-to-nearest;
    Hadamard rotation reduces 2-bit outlier error 50% vs unrotated; rate-distortion
    allocator correctly ranks layers by sensitivity while holding the bit budget;
    `BlockSparseLinear` achieves exact 50% 2:4 sparsity
  - **End-to-end result**: GPTQ 4-bit compression of a trained HVN model's
    `nn.Linear` layers gives **5.33× weight-storage compression with only
    +0.02% perplexity degradation** (acceptance criterion: <5%)
  - All quantization is inference-only (training stays full precision), per
    the issue's acceptance criteria
  - **Next**: Phase 2 (ternary BitNet training, low-rank+sparse decomposition,
    tensor-train Hamiltonian, ALBERT-style weight sharing) — not started
- **Full remaining scope (Phases 2-5) — `edge-devices` branch (done 2026-07-03)**:
  - Implemented as a separate `hvn/edge/` sub-package (edge-deployment
    architecture variant) rather than layered onto `hvn/core`, since several
    techniques (ternary training, weight sharing) change training dynamics
    and would risk the reference model's benchmarked quality
  - `hvn/edge/ternary.py` (`TernaryLinear`/`TernaryHamiltonian`, BitNet
    b1.58 STE), `hvn/edge/decomposition.py` (`LowRankSparseWeights`,
    `TTMatrix`, `MERAMatrix`, `KroneckerLinear`, `CirculantLinear`,
    `ToeplitzLinear`), `hvn/edge/shared.py` (`SharedWeightHVN`),
    `hvn/edge/entropy.py` (`EntropyCoder`, `LDPCCoder`),
    `hvn/edge/transform.py` (`WaveletCompressor`, `DCTCompressor`),
    `hvn/edge/holographic.py` (`HolographicWeights`, `SuperpositionWeights`,
    `PhaseEncodedComplexWeights`), `hvn/edge/streaming.py`
    (`StreamingHVNInference`, `MMapWeightStore`, `SuccinctWeights`,
    `TaskSpecificLoader`), `hvn/edge/recovery.py` (`AnnealingPruner`,
    `TopologicalPruner`, `FreeEnergyQuantizer`, `PopulationCoding`,
    `recovery_finetune_step`), `hvn/edge/model.py` (`EdgeHVN`)
  - `test_edge_devices.py`: 12/12 test groups pass on GPU (RTX 4050)
  - End-to-end: `EdgeHVN` (4 shared layers, 3.62x parameter reduction) +
    GPTQ 4-bit gives **2.33x additional weight compression at +0.37% PPL**
  - See `EXECUTION_PLAN.md` for the full technique list and honest scope
    notes on which techniques are simplified vs. literature-exact

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

**Root cause (SUPERSEDED — see #34)**: Previously attributed to the d×d matrix being a
"fundamental compression bottleneck." This is now believed to be **wrong**: the SSM
baseline reaches PPL ~1.1 with a *smaller* fixed state than HVN's d×d-per-head
Hamiltonian, proving O(1) state capacity is not the limiter. The real culprit is the
**readout/update dynamics** — specifically the velocity L2-norm + Frobenius clamp that
discard magnitude information, and the non-standard symmetric `HPH` update. This is why
every capacity-oriented strategy (#21, #22) moved PPL <3%.

**Next step**: See **#34** (fix readout normalization + adopt DeltaNet-style update —
the real lever) and **#35** (validate scaling on real data before trusting the flat
exponent). #36 handles generation-speed cleanups independently.