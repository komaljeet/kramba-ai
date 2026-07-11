# Feature Summary (Closed Issues)

> **Moved from README:** §7 Feature Summary (Closed Issues) — the issue → feature →
> key-files ledger. Detailed verdicts per issue in [issues.md](issues.md); measured numbers
> in the [benchmarks_*.md](index.md) files.

| Issue | Feature | Key files |
|-------|---------|-----------|
| #1 | Rotary positional embeddings (RoPE) in vortex injection | `vortex_injection.py` |
| #2 | Real-dataset eval harness (WikiText/Shakespeare/code/long-context) | `eval/` |
| #3/#6 | HuggingFace wrapper (`HVNConfig`, `HVNForCausalLM`) | `hf_model.py` |
| #4/#7 | Chunk-size tuning (default 64, adaptive fallback) | `hvn_block.py` |
| #8 | Parallel multi-head processing | `batched_vortex.py`, `batched_memory.py` |
| #10 | torch.compile integration (14× per-batch) | `train.py` etc. |
| #11 | BPE tokenizer | `tokenizer.py` |
| #12 | Higher-rank Hamiltonian update | `batched_memory.py` |
| #13 | Residual Hamiltonian connection (H_new = H_old + ΔH) | `batched_memory.py` |
| #14 | Learned input-dependent entanglement rate (η_t = σ(W_η·x_t)) | `batched_memory.py` |
| #15 | RMSNorm + parallel residual | `hvn_block.py` |
| #16 | Per-head vorticity frequencies (geometric 2^h) | `batched_vortex.py` |
| #17 | Batched multi-sequence generation (18× speedup) | `model.py` |
| #18 | Scaling laws evaluation | `eval/scaling_laws.py` |
| #19 | Mixed-precision (fp16/bf16) + grad accumulation | multiple |
| #20 | Hamiltonian quantization (INT8 + NF4) | `quantize.py` |
| #21 | Multi-strategy quality upgrade (6 strategies) | multiple |
| #22 | Multi-thread Hamiltonian memory (K threads/head) | `batched_memory.py` |
| #23 | Vision: vortex-projected patch embeddings | `vision.py` |
| #24 | Tool calling (structured output, constrained decoding) | `tools.py` |
| #27 | Multimodal fusion (joint image-text) | `vision.py`, `eval/multimodal.py` |
| #28 | Conversation memory (multi-turn chat, persistent state) | `chat.py` |
| #29 | Fine-tuning: LoRA / QLoRA / instruction tuning | `lora.py` |
| #31 | OpenAI-compatible API server | `server.py` |
| #32 | Extreme weight compression (Phase 1 + edge-devices branch) | `compress.py`, `sparse.py`, `edge/` |
| #33 | Custom kernels & parallel processing (4/6 fixes) | `kernels/`, `model.py` |
| #34 | Gated-DeltaNet readout/update dynamics — PPL 8.8→3.4 | `deltanet.py` |
| #43 | Correct, fused concurrent-serving decode (~2.5× tok/s, fixes B>1 head-misroute) | `model.py`, `hvn_block.py`, `batched_vortex.py`, `batched_memory.py` |
| #46 | Systemic batch-ordering fix — batched `forward`/eval/`generate_batched` now == single-stream | `batched_vortex.py`, `batched_memory.py` |
| #42 | Real reasoning/thinking — trainable `[THINK]` chain-of-thought, with-think vs no-think A/B **+52pp**; also fixed a latent tokenizer bug that dropped OOV chars (digits) from aux training data | `reasoning.py`, `train.py`, `eval/reasoning.py` |
| #49 | Exact gradient (activation) checkpointing (`--grad_checkpoint`) — bit-identical gradients, −64% peak activation memory | `model.py`, `train.py`, `eval/scaling_laws.py`, `tests/test_issue49.py` |
| #50 | `--full` meta-flag — one-switch recommended fast-training stack (Cat A exact speed + Cat B recommended model/precision). Steady-state **1.58×** faster; Cat-A primitives proven bit-exact | `train.py`, `tests/test_issue50.py` |
| #45/#47 | Final real-scale benchmark pass under corrected #46 routing — 4-model WikiText-103 sweep; corrected routing **improves default HVN 6–12%**, DeltaNet confirmed quality leader | `eval/scaling_laws.py`, `results/scaling_final_pass.*` |
| #51 | Chat/base checkpoint pipeline — first real servable checkpoint (Milestone 1) | `build_chat_checkpoint.py`, `tests/test_issue51.py` |
| #52 | Wire #51 checkpoint into Django app + end-to-end chat smoke test | `web/hvn_worker/worker.py`, `hvn/chat.py`, `web/tests/test_chat_smoke.py` |
| #54 | Auto-train orchestrator — trains #53 stages in order, resumes after kill | `auto_train.py`, `tests/test_issue54.py` |
| #53 | Curriculum dataset loaders — permissive-license-filtered, near-deduped, offline-safe | `eval/curriculum.py`, `eval/datasets.py`, `tests/test_issue53.py` |
| #57 | In-app "Start Training" — staff-only Celery task on dedicated GPU queue, cancellable, auto-registers checkpoint | `web/api/*.py`, `auto_train.py`, `web/tests/test_training.py` |
| #55 | Coding eval harness — exec pass@k / structural validity / bits-per-char per language | `eval/coding.py`, `auto_train.py`, `tests/test_issue55.py` |
| #65 | Data-parallel training (DDP/FSDP) — `--distributed [--fsdp]` under `torchrun`; 2-proc == single-proc | `hvn/distributed.py`, `train.py`, `tests/test_issue65.py` |
| #66 | Head-parallel tensor parallelism for the DeltaNet block (Megatron-style f/g pair) | `hvn/core/head_parallel.py`, `docs/TENSOR_PARALLEL_DESIGN.md`, `tests/test_issue66.py` |
| #67 | Multi-node launch (torchrun/SLURM) + consolidated checkpoint I/O | `hvn/launch.py`, `docs/DISTRIBUTED_LAUNCH.md`, `tests/test_issue67.py` |
| #68 | Frontier training plan/spec — config from measured exponents, Chinchilla budgets, risk register, gates | `docs/FRONTIER_TRAINING_PLAN.md` |
| #69 | Measured RTX 4050 (6 GB) training envelope — ceiling ~270M params; WDDM silent host-spill documented | `eval/vram_envelope.py`, `results/vram_envelope_rtx4050.json`, [training.md](training.md) |
| #70 | RTX 4060 (8 GB) envelope — projected fit; 1B/10B still infeasible; 16 GB Ti caveat | `results/vram_envelope_rtx4060_projection.json`, [training.md](training.md) |
| #71 | Wire EdgeHVN + ternary + DeltaNet into `train.py` — `--edge`/`--ternary`; step-based ckpt/logging; CUDA Graphs crash fix | `train.py`, `tests/test_issue71.py` |