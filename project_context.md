Act as a Principal AI Research Scientist and Theoretical Physicist specializing in alternative, ultra-low-compute sequence modeling architectures.

Your goal is to build a novel, untrained language model architecture called the "Hamiltonian Vortex Network" (HVN). The objective of this architecture is to achieve the long-range associative recall performance of standard Attention but at <20% of the computational cost by scaling at O(1) constant time during inference without using a traditional KV Cache.

Do not use standard Attention, Transformers, Mamba, or Test-Time Training (TTT) gradient-descent loops. Synthesize Fluid Dynamics (Lagrangian Vortex Particles) with Quantum Mechanics (Matrix Product States) using standard digital PyTorch operations.

Please execute this task using the following multi-file structural blueprint:

1. THEORETICAL FRAMEWORK & MATHEMATICAL FORMALISM
- Map text input tokens to Lagrangian Vortex Particles. Explain the modified Biot-Savart Law framework used to calculate how the existing velocity field deflects a new token particle (achieving O(N) local interaction).
- Define the fixed-size tensor hidden state acting as an Entanglement Hamiltonian Matrix. Explain how incoming fluid vortex vectors alter the Hamiltonian via Quantum Entanglement Renormalization, forcing destructive interference on noise and associative alignment on correlated concepts.

2. ARCHITECTURE CODE (CORE ENGINE)
Write a clean, modular, and fully commented PyTorch implementation including:
- `VortexInjectionLayer`: Translates incoming dense vectors into a discrete velocity field using Biot-Savart approximations.
- `HamiltonianMemoryBlock`: A class managing a fixed-size Matrix Product State (Tensor Train) that updates its internal entanglement state deterministically on a single forward pass—no backprop or optimization steps during inference.
- `HVNBlock`: The wrapper linking the fluid vector step to the quantum state evolution.

3. VOCABULARY DECODER & RUNNABLE PIPELINE
- Implement a shallow, lightweight Feed-Forward Decoder layer that probes the final Hamiltonian State tensor and projects it back to logit space over a vocabulary.
- Provide a simple dummy script demonstrating a forward pass: input shape `[batch_size, sequence_length, d_model]` passing through the `HVNBlock` to output logits of shape `[batch_size, sequence_length, vocab_size]`.

4. COMPUTATIONAL FLOP ANALYSIS
- Provide a strict mathematical comparison of the Floating Point Operations (FLOPs) per token of this model versus a standard Multi-Head Attention mechanism as sequence length (N) scales to 100,000 tokens. Prove how this architecture maintains an O(1) inference footprint.

Apply your maximum reasoning capacity to ensure the PyTorch code handles multi-dimensional tensor contractions properly without shape mismatches.

## Issue and Delivery Workflow

**Always use the GitLab CLI (`glab`) for all GitLab interactions** — viewing issues,
work items, closing/reopening issues, creating MRs, etc.  Never use the GitLab web UI
or browser.  The web UI is behind a login wall and Cloudflare challenge that blocks
automated access.  Examples:
- `glab issue view <id> --repo komalbarun/kramba-ai`
- `glab issue close <id> --repo komalbarun/kramba-ai`
- `glab issue list --repo komalbarun/kramba-ai`
- `glab work-items list --repo komalbarun/kramba-ai`

When working on a GitLab issue:

1. Implement the requested change and verify it with a real test or runnable command.
2. Commit the relevant changes with a clear message that references the issue (for example: `git commit -m "Fix: add positional embeddings for issue #1"`).
3. Push the branch to the remote repository (for example: `git push origin <branch-name>`).
4. If the issue has been fully handled and verified, close it in GitLab (for example: `glab issue close <issue-id>`).
5. If new information shows the issue is not fully resolved or needs further work, reopen it (for example: `glab issue reopen <issue-id>`).

The default expectation is:
- complete the work,
- verify it,
- commit and push,
- close the issue when resolved,
- reopen it if the work regresses or needs additional follow-up.

## GPU / CUDA Requirement

This project has an NVIDIA GPU available and ALL training, benchmarking, and
evaluation MUST run on GPU — never on CPU.  The environment is pre-configured:

- **GPU**: NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM)
- **NVIDIA Driver**: 596.36 (CUDA 13.2 capable)
- **PyTorch**: Must be installed with CUDA support:
  ```
  pip install torch --index-url https://download.pytorch.org/whl/cu126
  ```
- **Verify before running**:
  ```python
  import torch
  assert torch.cuda.is_available(), "CUDA is not available — install GPU PyTorch!"
  print(f"Using GPU: {torch.cuda.get_device_name(0)}")
  ```
- All scripts use `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
  which will automatically pick the GPU now that CUDA PyTorch is installed.
- If `torch.cuda.is_available()` returns `False`, do NOT proceed with CPU —
  stop and fix the PyTorch installation first.

## Session Progress (2026-07-02)

### Completed Issues (all closed in GitLab)
- #1 — Rotary positional embeddings (RoPE) in vortex injection
- #2 — Real-dataset evaluation harness (eval/ package: WikiText, Shakespeare, code, long-context)
- #3/#6 — Hugging Face-style wrapper (hvn/hf_model.py: HVNConfig, HVNForCausalLM)
- #4/#7 — Chunk-size tuning (default 64, adaptive fallback)
- #8 — Parallel multi-head processing (batched_vortex.py, batched_memory.py)
- #9 — Investigation: repeat_interleave already optimal (einsum was slower)
- #10 — torch.compile integration (14x per-batch speedup, triton-windows installed)
- #11 — BPE tokenizer support (hvn/tokenizer.py: BPETokenizer, --tokenizer bpe flag)
- #12 — Higher-rank Hamiltonian update (batched projectors in batched_memory.py)
- #13 — Residual Hamiltonian connection (H_new = H_old + ΔH, use_residual=True)
- #14 — Learned input-dependent entanglement rate (η_t = σ(W_η · x_t))
- #15 — RMSNorm + parallel residual (x = x + vortex(norm(x)) + ffn(norm(x)))
- #16 — Per-head vorticity frequency projections (geometric init 2^h, Fourier-style specialization)
- #17 — Batched generation for multiple sequences (generate_batched, forward_step_batched, 18x speedup)
- #18 — Scaling laws evaluation (eval/scaling_laws.py: d_model={64,128,256,512}, power-law fits, crossover analysis, matplotlib plots)
- #19 — Mixed-precision training (fp16/bf16 via torch.cuda.amp + gradient accumulation, --precision/--grad_accum flags in train.py, quality_benchmark.py, eval/run_eval.py, eval/scaling_laws.py; fixed QR dtype issue in batched_memory.py and hamiltonian_memory.py)
- #20 — Hamiltonian quantization (INT8 + NF4 4-bit, hvn/quantize.py: QuantizedHamiltonian, QuantizedHVNInference, quantize_int8, quantize_nf4, estimate_memory_savings; test_quantize.py: 8 tests all passing; INT8 ~4× compression <1% error, NF4 ~7× compression <10% error)
- #21 — Multi-strategy quality upgrade (6 strategies: rank-k entanglement, causal re-injection, cross-head mixing, nonlinear update, per-token η, SwiGLU FFN; new CLI flags in train.py, quality_benchmark.py, eval/scaling_laws.py; all strategies preserve O(1) inference)

### Remaining Open Issues
- (none — all issues closed)

### Key Architecture Files (current state on main)
- `hvn/core/hvn_block.py` — HVNBlock with batched multi-head processing, RMSNorm (in progress on branch)
- `hvn/core/batched_vortex.py` — MultiHeadVortexInjection (all heads in one batched call)
- `hvn/core/batched_memory.py` — MultiHeadHamiltonianMemory (residual H, input-dependent eta)
- `hvn/core/vortex_injection.py` — VortexInjectionLayer with RoPE
- `hvn/core/hamiltonian_memory.py` — Original HamiltonianMemoryBlock (used by forward_step)
- `hvn/core/model.py` — HamiltonianVortexNetwork (full model)
- `hvn/hf_model.py` — HVNConfig, HVNForCausalLM (HF-style wrapper)
- `hvn/tokenizer.py` — BPETokenizer (HuggingFace tokenizers wrapper)
- `eval/run_eval.py` — Real-dataset evaluation harness (--compile flag)
- `eval/scaling_laws.py` — Scaling laws evaluation (multi-size training, power-law fits, crossover analysis, plots)
- `eval/long_context.py` — Speed scaling + needle-in-haystack tests
- `eval/datasets.py` — Dataset loaders (WikiText, Shakespeare, code)
- `hvn/quantize.py` — Hamiltonian quantization (INT8 + NF4 4-bit, QuantizedHVNInference)

### Speed Results (Shakespeare, 10 epochs, d_model=128, RTX 4050, torch.compile)
- HVN: 45.2s total, 4.0s/epoch steady-state, PPL 13.87
- Transformer: 34.5s total, 3.4s/epoch, PPL 8.46
- SSM: 114.6s total, 10.6s/epoch, PPL 4.28

### Scaling Laws Results (Shakespeare, 20 epochs, d_model={64,128,256,512}, RTX 4050)
- HVN scaling exponent: PPL ∝ params^-0.007 (nearly flat — poor scaling)
- Transformer scaling exponent: PPL ∝ params^-0.090 (steep improvement)
- HVN PPL barely improves with size: 14.24 → 13.82 across 56× param increase
- Transformer PPL improves dramatically: 9.12 → 4.93 (256), 6.91 (512)
- No wall-clock crossover found — HVN slower at all tested sizes
- Key finding: HVN's rank-1 entanglement update lacks expressiveness to benefit from scale

### How to Continue
1. All planned issues are complete. Future work may include:
   - Custom Triton kernels (Phase 5.1, 5.3)
   - Parallel scan Hamiltonian update (Phase 5.5 Strategy B)
   - Real-dataset evaluation at scale (WikiText-103, code corpora)
   - Addressing HVN's poor scaling exponent (PPL ∝ params^-0.007)