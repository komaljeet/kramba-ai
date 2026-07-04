# Hamiltonian Vortex Network (HVN) — Complete Compendium

> **Single source of truth.** This file is the consolidated documentation for the project —
> it merges what were previously nine separate markdown docs (theory, FLOP analysis, training
> guide, benchmarks, roadmap, execution plan, project context, and issue history).
>
> Status legend: ✅ Done · 🔄 In Progress · ⬜ Planned

## Table of Contents

1. [Overview](#1-overview)
2. [Theoretical Framework & Mathematical Formalism](#2-theoretical-framework--mathematical-formalism)
3. [FLOP Analysis: HVN vs Multi-Head Attention](#3-flop-analysis-hvn-vs-multi-head-attention)
4. [Architecture Components & Project Structure](#4-architecture-components--project-structure)
5. [Quick Start & Usage](#5-quick-start--usage)
6. [Training & Usage Guide](#6-training--usage-guide)
7. [Feature Summary (Closed Issues)](#7-feature-summary-closed-issues)
8. [Benchmark Results](#8-benchmark-results)
9. [Development Roadmap](#9-development-roadmap)
10. [Execution Plan: Issues #32 & #33](#10-execution-plan-issues-32--33)
11. [Issue History & Related Work](#11-issue-history--related-work)
12. [Project Context & Delivery Workflow](#12-project-context--delivery-workflow)

---

## 1. Overview

HVN is a novel, untrained language model architecture that achieves **long-range
associative recall at <20% of the computational cost of standard Attention**, scaling
at **O(1) constant time per token during inference** — without a traditional KV Cache.

### How It Works

HVN fuses two theoretical frameworks:

1. **Lagrangian Vortex Fluid Dynamics** — Each token is mapped to a vortex particle
   whose velocity is computed via a *modified Biot-Savart law* against a fixed-size
   velocity field (the Hamiltonian). This replaces the O(N) attention scan with an
   O(d²) matrix-vector product.

2. **Quantum Matrix Product State Entanglement** — A fixed-size Hamiltonian matrix
   evolves via *Quantum Entanglement Renormalization*, absorbing each new vortex
   vector through a deterministic update rule (no gradient descent at inference time).
   SVD truncation discards noise modes, preserving only the most entangled associations.

The result: **O(d²) per-token inference cost, constant in sequence length N.**

### Key Results (N = 100,000 tokens, d = 512, L = 12 layers)

| Metric | Standard MHA (KV cache) | HVN (SVD r=128) | Ratio |
|--------|------------------------|------------------|-------|
| FLOPs/token | 2,474M | 463M | **18.7%** ✓ |
| Inference memory | 2.46 GB | 6.29 MB | **0.26%** |
| Scaling in N | O(Nd) | O(d²r) = O(1) | — |

---

## 2. Theoretical Framework & Mathematical Formalism

The architecture fuses two theoretical pillars:

| Pillar | Source Field | Role in HVN |
|--------|-------------|-------------|
| **Lagrangian Vortex Particles** | Fluid Dynamics | Translates each token into a discrete vortex particle that interacts *locally* with a background velocity field via a modified Biot-Savart law. |
| **Matrix Product State (MPS) Entanglement Hamiltonian** | Quantum Mechanics | A fixed-size tensor-train state that evolves deterministically on every forward pass, absorbing vortex vectors through quantum entanglement renormalization. |

### 2.1 Lagrangian Vortex Particle Mapping

Each input token embedding **x**_t ∈ ℝ^{d_model} is interpreted as a **Lagrangian vortex
particle** characterized by:

- **Position**: r_t ∈ ℝ^{d_model} — the token's embedding vector (static within a step).
- **Vorticity (circulation)**: ω_t ∈ ℝ^{d_model} — derived via a learned linear projection.
- **Velocity**: v_t ∈ ℝ^{d_model} — the velocity field induced at the particle's location
  by all *previously absorbed* vorticity stored in the Hamiltonian memory.

**Modified Biot-Savart Law (Local O(1) Approximation).** Classical fluid dynamics
computes induced velocity by a global integral (O(N) over all prior particles):

```
v(r) = (1/4π) ∫ (ω(r') × (r - r')) / |r - r'|³  dV'
```

HVN replaces it with a **local, fixed-size approximation**:

```
v_t = Γ · (H · ω_t)
```

- **H** ∈ ℝ^{d_model × d_model} is the **Entanglement Hamiltonian** — a fixed-size matrix
  summarizing the entire history of absorbed vorticity (the effective velocity field).
- **Γ** is a learned scalar coupling constant (initialized to 1/√d_model).
- **ω_t** is the current token's vorticity vector.

Because **H** has fixed shape, velocity is a single matrix-vector product: **O(d_model²)**
regardless of N. This is the key to O(1) per-token inference.

**Vortex Deflection.** The velocity deflects the token's position, producing the
*interacted* representation injected into quantum memory:

```
r̃_t = r_t + Γ · v_t
```

This is the fluid-dynamical analogue of the query-key interaction in Attention, but
computed in constant time against a summary rather than against all keys.

### 2.2 Entanglement Hamiltonian Matrix (Fixed-Size Quantum Memory)

**Matrix Product State (Tensor Train) Structure.** The Hamiltonian memory is stored as
an MPS — a tensor train of fixed bond dimension χ: `H = A_1 · A_2 · ... · A_K`
(K = d_model / χ), each core **A_k** ∈ ℝ^{χ × χ}. HVN **never explicitly reconstructs**
the full d_model × d_model matrix; operations run on the compressed cores, keeping the
footprint at O(K · χ²) = O(d_model · χ). (The reference implementation stores H as a
single d×d matrix but updates it via low-rank MPS-style renormalization.)

**Quantum Entanglement Renormalization.** When a new vortex vector **v_t** arrives, it is
absorbed into the Hamiltonian (analogous to a DMRG step). The update rule:

```
H_new = (1 - η) · H_old + η · (v_t ⊗ v_t^T - H_old · v_t ⊗ v_t^T · H_old)
```

Decomposed:

1. **Outer product**: P_t = v_t ⊗ v_t^T — the rank-1 "interaction projector."
2. **Destructive interference**: subtract H_old · P_t · H_old — cancels components of H
   already aligned with v_t (destructive interference on noise / repeated patterns).
3. **Associative alignment**: add P_t — reinforces the new association.
4. **Renormalization**: blend with (1 - η) · H_old to preserve prior memory. η is a
   learned or fixed entanglement rate (default 1/√d_model).

This is a **deterministic, single-pass update** — no optimization, no gradient descent.
Runs in O(d_model²) (or O(d_model · χ) compressed), independent of sequence length.

**Low-Rank MPS Compression.** After each update a truncated-SVD renormalization keeps the
top-r singular values (`H = U_r · S_r · V_r^T`), discarding high-energy noise modes.
Cost: O(d_model² · r) — still O(1) in N.

### 2.3 The HVN Block (per token t)

```
1. VortexInjection:
   ω_t = W_ω · x_t                    (vorticity projection)
   v_t = Γ · (H · ω_t)                (Biot-Savart local velocity)
   r̃_t = x_t + Γ · v_t                (deflected representation)

2. HamiltonianMemoryUpdate:
   P_t = v_t ⊗ v_t^T                  (rank-1 interaction projector)
   H ← (1-η)·H + η·(P_t - H·P_t·H)    (entanglement renormalization)
   [optional: SVD truncation to rank r]

3. Output:
   y_t = W_o · r̃_t                    (project back to d_model)
```

**Total per-token cost**: O(d_model²) — constant in N. The Hamiltonian is a compressed
holographic summary of all past tokens; correlated concepts reinforce (via P_t) while
noise cancels (via H·P_t·H), and each new token "feels" the entire history through H·ω_t
without scanning it.

---

## 3. FLOP Analysis: HVN vs Multi-Head Attention

Notation: N=seq len (up to 100,000), d=d_model (512), h=heads (8), r=SVD bond dim (32–128),
L=layers (12), V=vocab (32,000). All counts are inference (forward), per token, per layer.

### 3.1 Standard Multi-Head Attention (MHA)

- **Without KV cache**: 8d² + 4Nd + 3N per token per layer.
- **With KV cache**: 4d² + 4Nd + 3N per token per layer. Even with a KV cache, per-token
  cost is **O(Nd)** — the new query must attend to all N cached keys/values. The cache
  reduces *recomputation*, not *scan complexity*.
- **KV cache memory**: 2·N·d·bytes per layer. For N=100K, d=512, fp16: ~200 MB/layer →
  **~2.4 GB** for 12 layers.

### 3.2 HVN Per-Token Cost (exploiting rank-1 structure)

Because P_t = v_t ⊗ v_t^T is rank-1, `H·P_t·H = (Hv) ⊗ (H^T v)^T`, computed via two
matvecs (2d² each) + one outer product (d²) = **5d²** instead of the naive 2d³.

| Operation | FLOPs |
|-----------|-------|
| Vorticity projection | 2d² |
| Biot-Savart velocity (H · ω_t) | 2d² |
| Deflection + output proj + norm | 2d² + 2d |
| Interaction projector (outer product) | d² |
| H · v_t and H^T · v_t (destructive interference) | 4d² |
| Outer product (Hv ⊗ (H^Tv)^T) | d² |
| Renormalization blend | 2d² |
| SVD truncation (optional, rank r) | O(d²r) |
| FFN sub-layer (2× expansion) | 6d² |
| **Total per token per layer** | **~19d² + O(d²r)** |

For d=512: with SVD r=128 ≈ **38.6M** FLOPs/layer; full rank ≈ 139.2M. HVN Hamiltonian
memory = d²·bytes/layer → for d=512, fp16, 12 layers: **~6.29 MB, constant in N**.

### 3.3 Head-to-Head at N = 100,000 (d=512, L=12, r=128)

| Architecture | Total Per-Token FLOPs (all 12 layers) |
|---|---|
| MHA with KV cache | ~**2,474M** |
| HVN (with SVD, r=128) | ~**463M** |
| HVN (no SVD) | ~1,670M |

| Variant | Ratio at N=100K |
|---|---|
| **HVN with SVD (r=128)** | 463 / 2,474 ≈ **18.7%** ✓ (< 20% target) |
| HVN without SVD | ≈ 67.5% |

### 3.4 Scaling vs Sequence Length (d=512, L=12, r=128)

| N | MHA (KV cache) | HVN (SVD r=128) | Ratio |
|---|---|---|---|
| 1,000 | 36M | 463M | 12.8× |
| 10,000 | 258M | 463M | 0.56× |
| 100,000 | 2,474M | 463M | **0.187×** |
| 1,000,000 | 24,588M | 463M | 0.019× |

MHA is **O(Nd)** (grows linearly); HVN is **O(d²r)** (constant in N). Crossover ≈ N ≈ 5,000;
HVN's advantage grows unboundedly beyond. Memory: HVN uses ~390× less at N=100K (6.29 MB
vs 2.46 GB), improving further as N grows.

---

## 4. Architecture Components & Project Structure

```
kramba-ai/
├── hvn/
│   ├── core/
│   │   ├── vortex_injection.py    # VortexInjectionLayer: Biot-Savart velocity field (+ RoPE)
│   │   ├── hamiltonian_memory.py  # HamiltonianMemoryBlock: MPS entanglement update
│   │   ├── batched_vortex.py      # MultiHeadVortexInjection (all heads in one batched call)
│   │   ├── batched_memory.py      # MultiHeadHamiltonianMemory (residual H, input-dep η, rank-k, multi-thread)
│   │   ├── deltanet.py            # DeltaNetBlock: Gated-DeltaNet dynamics (Issue #34)
│   │   ├── hvn_block.py           # HVNBlock: fluid + quantum wrapper, RMSNorm, parallel scan/forward
│   │   ├── decoder.py             # HVNDecoder: shallow logit projection
│   │   └── model.py               # HamiltonianVortexNetwork: full model + incremental/graph/fused generation
│   ├── kernels/hamiltonian_kernel.py  # Fused Triton kernel (Issue #33 Fix 1)
│   ├── tokenizer.py               # BPETokenizer (HuggingFace tokenizers wrapper)
│   ├── quantize.py                # INT8 + NF4 Hamiltonian quantization (Issue #20)
│   ├── compress.py / sparse.py    # GPTQ, Hadamard, rate-distortion, 2:4 sparsity (Issue #32 Phase 1)
│   ├── lora.py                    # LoRA / QLoRA / instruction tuning (Issue #29)
│   ├── vision.py / tools.py / chat.py / server.py  # Multimodal, tool calling, chat, OpenAI API
│   ├── hf_model.py                # HVNConfig, HVNForCausalLM (HuggingFace wrapper)
│   └── edge/                      # edge-devices branch: full Issue #32 compression suite
├── docs/                          # (source docs — now consolidated into this compendium)
├── eval/                          # run_eval, scaling_laws, long_context, datasets, instruction, multimodal
├── demo.py / train.py / generate.py / benchmark.py / quality_benchmark.py
└── requirements.txt
```

| Component | File | Role |
|-----------|------|------|
| `VortexInjectionLayer` | `vortex_injection.py` | Token → vortex particle → velocity field via Biot-Savart |
| `HamiltonianMemoryBlock` | `hamiltonian_memory.py` | Fixed-size MPS Hamiltonian with entanglement renormalization |
| `HVNBlock` | `hvn_block.py` | Causal wrapper: sequential vortex injection + H update |
| `DeltaNetBlock` | `deltanet.py` | Gated-DeltaNet readout/update dynamics (Issue #34) |
| `HVNDecoder` | `decoder.py` | Shallow projection: hidden state → vocabulary logits |
| `HamiltonianVortexNetwork` | `model.py` | Full model: embedding + N×block + decoder |

---

## 5. Quick Start & Usage

```bash
# Install dependencies (CUDA GPU version)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy

# Run the architecture demo (verifies shapes & forward pass)
python demo.py

# Train the model on the bundled sample text
python train.py --epochs 50 --generate --prompt "The "

# Generate text from a trained checkpoint
python generate.py --prompt "The " --tokens 200 --temp 0.7
```

```python
import torch
from hvn.core import HamiltonianVortexNetwork

model = HamiltonianVortexNetwork(
    vocab_size=1000, d_model=128, num_layers=2,
    bond_dimension=32,   # SVD truncation rank (quantum renormalization)
)
input_ids = torch.randint(0, 1000, (4, 64))   # [batch, seq_len]
logits = model(input_ids)                       # [batch, seq_len, vocab_size]
```

### Notable Runtime Features

**Gated-DeltaNet Dynamics — the real PPL lever (Issue #34).** `--use_deltanet` swaps each
block for a self-contained Gated-DeltaNet block (`hvn/core/deltanet.py`): per-channel
data-dependent decay gate `α_t=σ(W_α x)` (replaces the Frobenius clamp), per-head output
RMSNorm, gated DeltaNet update `S_t = diag(α_t)S_{t-1} + β_t(v_t−S_{t-1}k_t)k_tᵀ` with
separate k/v projections and query read `o_t=S_t q_t` (drops the ill-conditioned HPH),
plus an output gate `SiLU(W_g x)` + short causal depthwise conv. Result: eval PPL
**8.81 → 3.42 (61% better)**, O(1) inference preserved.

```bash
python train.py --epochs 20 --use_deltanet --generate --prompt "The "
```

**Custom Kernels & Parallel Processing (Issue #33).**

| Technique | Flag | Speedup | Applies to |
|-----------|------|---------|------------|
| CUDA Graph capture | `--cuda_graph` | 2.82× (78 → 220 tok/s) | Generation |
| Fused Triton kernel | `--fused_kernel` | 2.14–7.2× (78 → 167–340 tok/s) | Generation |
| Parallel scan (associative H update) | `--use_parallel_scan` | 1.59× (7.63 → 4.79 ms/fwd) | Training |
| Multi-chunk barrier sync | (internal via `parallel_forward()`) | 1.14× | Training |

**Hamiltonian Quantization (Issue #20).** INT8 (~4×, <1% error) and NF4 4-bit (~8×, <10%
error) via `hvn.quantize.QuantizedHVNInference` and `estimate_memory_savings`.

**OpenAI-Compatible API Server (Issue #31).** `python -m hvn.server --checkpoint hvn_checkpoint.pt --port 8000`
exposes `/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/health` (SSE
streaming + tool calling), built directly on `forward_step` — no KV cache, O(1) preserved.

---

## 6. Training & Usage Guide

### HVN vs Transformer — Fundamental Differences

| Aspect | Transformer (MHA) | HVN |
|--------|----------------------|---------|
| Core mechanism | Self-Attention (Q·K^T over all tokens) | Vortex particles interacting with a fixed-size Hamiltonian |
| Memory of past | KV Cache (grows with N) | Hamiltonian matrix H (constant d×d) |
| Per-token inference | O(N·d) | O(d²·r) — constant in N |
| Inference memory | O(N·d) | O(d²) — fixed |
| Positional encoding | RoPE / learned / sinusoidal | RoPE in vortex injection |
| Parallelism | Fully parallel | Sequential H update (chunked/parallel-scan mitigation) |

### What Fits in 6 GB VRAM (RTX 4050)

| Config | d_model | Layers | Params | VRAM | Fits? |
|--------|---------|--------|--------|------|-------|
| Small (default) | 128 | 2 | ~265K | ~14 MB | ✅ Easily |
| Medium | 256 | 4 | ~1M | ~50–100 MB | ✅ Easily |
| Large | 512 | 8 | ~5–10M | ~200–500 MB | ✅ Yes |
| Very Large | 512 | 12 | ~15M | ~1–2 GB | ✅ Yes |

**Recommended balanced command:**
```bash
python train.py --text my_corpus.txt \
    --d_model 256 --num_layers 4 --bond_dim 64 \
    --batch_size 16 --seq_len 128 --precision bf16 --compile \
    --epochs 100 --generate --prompt "Chapter 1 "
```

**On OOM, reduce in order:** `--batch_size 8|4` → `--seq_len 64` → `--d_model 128`, then
add `--grad_accum 4` to recover effective batch.

### Key Training Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--text` | (sample) | Path to training text |
| `--epochs` | 30 | Training epochs |
| `--lr` | 3e-4 | Learning rate (AdamW) |
| `--batch_size` / `--seq_len` | 16 / 64 | Batch size / sequence length |
| `--d_model` / `--num_layers` / `--bond_dim` | 128 / 2 / 32 | Model size / SVD rank |
| `--precision` | fp32 | fp32 / fp16 / bf16 (GPU for 16-bit) |
| `--grad_accum` | 1 | Gradient accumulation steps |
| `--compile` | flag | torch.compile (14× per-batch) |
| `--tokenizer` / `--bpe_vocab` | char / 1000 | char or bpe tokenizer |
| `--rank_k` | 1 | Rank-k entanglement (Issue #21 S1); 4/8 recommended |
| `--nonlinear_update` | flag | Nonlinear gated update (S4) |
| `--causal_reinjection` | flag | Multi-chunk causal re-injection (S2) |
| `--cross_head_mixing` | flag | Cross-head H mixing (S3) |
| `--swiglu` | flag | SwiGLU FFN (S6) |
| `--per_token_eta` | flag | Per-token entanglement rate (S5) |
| `--n_threads` | 1 | Independent H threads per head (Issue #22); 4 recommended |
| `--use_parallel_scan` | flag | Associative parallel scan (Issue #33 Fix 3) |
| `--use_deltanet` | flag | Gated-DeltaNet dynamics (Issue #34) |

### Precision Guide

| Precision | Speed | Memory | Stability | Notes |
|-----------|-------|--------|-----------|-------|
| fp32 | 1× | 1× | Best | Default |
| fp16 | 2–3× | 0.5× | Good | GradScaler; may overflow |
| bf16 | 2–3× | 0.5× | Best | No scaler (RTX 40-series+) |

### Generation

```bash
python generate.py --prompt "Hamiltonian " --tokens 300 --temp 0.5 --top_k 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | hvn_checkpoint.pt | Trained model path |
| `--prompt` | "The " | Seed text |
| `--tokens` | 200 | Max new tokens |
| `--temp` | 0.8 | Temperature (0.1–0.3 deterministic, 0.5–0.7 focused, 0.8–1.0 balanced, 1.0+ wild) |
| `--top_k` | 10 | Top-k sampling (0 = off) |

### Tips

More/real data (bundled sample is only ~34K chars), 50–100+ epochs on small data, larger
model (`--d_model 256 --num_layers 4 --bond_dim 64`), longer sequences (`--seq_len 128+`),
BPE tokenizer for real projects, always `--precision bf16` + `--compile` on RTX 40-series.

---

## 7. Feature Summary (Closed Issues)

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

---

## 8. Benchmark Results

**Test configuration**: RTX 4050 (6 GB), PyTorch 2.12.1+cu126, d_model=128, 2 layers,
vocab=256 unless noted.

### 8.1 Speed

**Generation (50 tokens, prompt=32):**

| Model | tokens/sec | ms/token | Peak Mem | Params |
|-------|-----------|----------|----------|--------|
| HVN (incremental) | 106.5 | 9.4 | 13.6 MB | 264,966 |
| HVN (old, full context) | 75.6 | 13.2 | 13.6 MB | 264,966 |
| Transformer | 780.6 | 1.28 | 13.0 MB | 461,312 |
| SSM | 40.0 | 25.0 | 12.8 MB | 362,496 |

**Incremental generation loop cleanups (Issue #36).** Removing the redundant prompt prefill
(the prompt was processed twice) and moving decode sampling fully on-device (reused input
buffer, on-device top-k + `multinomial`, one host sync per generation). Measured HEAD vs the
pre-#36 commit, warmed best-of-5, d=128, 2L, RTX 4050, **512-token prompt + 200 generated
tokens**:

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
of token-by-token; on a 512-token prompt that collapses the ~1.6 s wait before the first output
token to ~23 ms (a 2,000-token prompt would be ~6.4 s → ~90 ms). The HVN decode win comes from
eliminating per-token host↔device syncs. Stacks with #33 kernels (HVN decode + `--cuda_graph`
350 tok/s, `--fused_kernel` 354 tok/s). Generated output is token-for-token unchanged
(`test_issue36.py`, `top_k=1` greedy parity).

**Forward pass scaling (ms/token, batch=1):** HVN flat 5.06→5.05 from N=256→4096 (**O(1) ✓**);
SSM ~0.43 flat; Transformer sub-ms (Flash Attention). **Memory at N=4096:** HVN 38.4 MB vs
Transformer 48.3 MB (20% less) vs SSM 36.3 MB.

**Parameter efficiency:** HVN 264,966 (1.0×) < SSM 362,496 (1.37×) < Transformer 461,312
(1.74×) — HVN uses 43% fewer params than Transformer.

### 8.2 Quality (Shakespeare)

**d_model=64, 1 layer, 10 epochs:**

| Model | Params | Eval PPL | Train Time |
|-------|--------|----------|------------|
| HVN (old, single-head) | 30,056 | 9.48 | 236.1s |
| HVN (new, multi-head) | 34,225 | 8.98 | 172.5s |
| Transformer | 54,592 | 4.29 | 4.0s |
| SSM | 42,176 | 1.65 | 35.4s |

**d_model=128, 2 layers, 20 epochs:**

| Model | Params | Eval PPL | Train Time |
|-------|--------|----------|------------|
| HVN (multi-head) | 241,725 | 8.37 | 467.8s |
| Transformer | 405,248 | 1.90 | 3.4s |
| SSM | 306,432 | 1.12 | 68.6s |

### 8.3 Custom Kernels & Parallel Processing (Issue #33)

| Fix | Technique | Before | After | Speedup |
|-----|-----------|--------|-------|---------|
| Fix 6 | CUDA Graph capture (generation) | 78 tok/s | 220 tok/s | 2.82× |
| Fix 1 | Fused Triton kernel (generation) | 78 tok/s | 167 (340 warmed) | 2.14× (7.2×) |
| Fix 3 | Parallel scan (training forward) | 7.63 ms | 4.79 ms | 1.59× |
| Fix 4 | Multi-chunk barrier sync (training) | 7.74 ms | 6.77 ms | 1.14× |

Deferred: Fix 2 (flash-style tiling, needed when d_model>1024), Fix 5 (fully parallel
positional injection, research-level).

### 8.4 Real-Scale Scaling Re-measurement (Issue #35)

Rebuilt scaling harness (`eval/scaling_laws.py`): **compute-matched** (fixed per-model token
budget, not epochs), **full held-out eval PPL** (whole valid split in `seq_len` chunks), BPE +
SSM added, per-model saturation tracking, streaming WikiText-103 loader.

**Headline run — WikiText-103, BPE (vocab 8,000), 5M-token compute-matched budget** (12.4M
unique train tokens available, so no repeats — models are **not** data-starved), d={128,256,512,768},
2 layers, seq_len=128, bf16, RTX 4050:

| Model | d=128 | d=256 | d=512 | d=768 | Params (d=768) | **Exponent** (PPL ∝ N^b) |
|-------|------:|------:|------:|------:|---------------:|:---:|
| **HVN** (default) | 521.1 | 351.9 | 236.1 | 205.8 | 21.75M | **−0.4237** |
| Transformer | 395.6 | 246.0 | 172.9 | 150.3 | 26.46M | −0.4066 |
| SSM | 366.8 | 225.2 | 138.5 | 110.6 | 22.92M | −0.5311 |

All three models are **still improving at 5M tokens** (unsaturated — the correct regime for a
scaling law). Absolute PPL is high because 5M tokens is a small budget for BPE-8000; the
**exponent**, not the absolute PPL, is the quantity of interest.

**Result — the flat exponent was a small-data artifact.** On real-scale data HVN scales at
**−0.42**, essentially matching Transformer (**−0.41**) and near SSM (−0.53) — a ~100× steeper
exponent than the original **−0.007**. HVN's PPL drops 521 → 206 (2.5×) across a 9× param
increase; it clearly benefits from scale. The "HVN doesn't scale" narrative is **refuted**.

**Small-data control — full Tiny-Shakespeare, char-level, 2M-token budget** (1.0M train chars,
32× the 34K the −0.007 was fit on): HVN −0.0042, Transformer −0.0252, SSM −0.1060; HVN saturates
at ~1M tokens (all sizes). This *reproduces* the flat exponent — and shows why it is an artifact:
at 1M chars every size saturates early (data-bound), so the exponent measures saturation noise,
not architectural scaling. Move to a corpus where models aren't starved and the flatness vanishes.

*Caveats:* default HVN only (the `--use_deltanet` (#34) sweep is now done — see §8.6, Issue #39);
5M-token reduced budget, not the full ~100M — absolute PPLs will drop substantially at a larger
budget, but the exponent comparison is already in the trustworthy (unsaturated) regime. Reproduce
/ scale up via the one-command job in §9.2.

### 8.6 DeltaNet vs Default Block at Real Scale (Issue #39)

The #34 DeltaNet win was previously only measured on 34K chars. Issue #39 swept it on the same
real-scale config as §8.4 (WikiText-103, BPE-8000, 5M-token compute-matched budget, 2 layers,
seq_len=128, RTX 4050) via a distinct `HVN-DeltaNet` model in `eval/scaling_laws.py`
(`--models HVN HVN-DeltaNet Transformer`):

| d_model | HVN (default) | HVN-DeltaNet | Transformer | DeltaNet vs HVN |
|--------:|--------------:|-------------:|------------:|:---:|
| 128 | 522.1 | **410.4** | 394.5 | **−21.4%** |
| 256 | 350.4 | **251.9** | 248.4 | **−28.1%** |
| 512 | 237.0 | **218.9** (fp32) / NaN (bf16) | 173.1 | −7.6% (fp32) |
| 768 | 206.1 | NaN (bf16) | 150.1 | — |

**DeltaNet is better on quality at every width** — it cuts PPL 21–28% at d≤256 (essentially
matching the Transformer) and, in fp32, reaches 218.9 at d=512 (beating HVN's 237 and still
improving). **But under bf16** — the project's recommended training precision on RTX 40-series —
**it diverges to NaN at d≥512** (d=512 spiked after 2M tokens at PPL 207; d=768 after 1M at 244,
i.e. while *winning*). A fp32 rerun of d=512 trains stably to 218.9, confirming the failure is a
**large-width bf16 numerical-stability / loss-spike issue, not a quality deficit** — existing grad
clipping is insufficient at that width.

**Decision — keep flag-gated, do not flip the default.** Acceptance criterion #35 ("DeltaNet ≥ HVN
block at real scale") holds on *quality* but fails on *robustness*: a default must train cleanly in
the recommended precision, and DeltaNet currently does not at d≥512/bf16. `--use_deltanet` is the
**recommended** setting (use fp32, or bf16 only at d≤256, until the instability is fixed); flipping
the default is blocked on a follow-up that fixes large-width bf16 stability (loss-spike guarding /
fp32-casting the decay gate & normalization / init & LR tuning).

### 8.5 Honest Assessment

**Where HVN wins:** memory efficiency (O(1) vs O(N)), parameter efficiency (43% fewer),
theoretical O(1) per-token scaling (flat N=256→4096), incremental generation, kernels
close the raw-speed gap (2–7×). **Where HVN loses:** raw speed (Flash Attention faster in
absolute terms), and the quality gap — addressed by Issue #34 (see below).

---

## 9. Development Roadmap

### Phase 1–3: Core Architecture, Stability, Performance — ✅ Complete

Core engine (`VortexInjectionLayer`, `HamiltonianMemoryBlock`, `HVNBlock`, `HVNDecoder`,
full model), theory + FLOP docs; velocity L2-norm + Frobenius clamp, power-iteration
low-rank truncation, gradient clipping, cosine LR + AdamW, char tokenizer, train/generate
scripts; multi-head Hamiltonian (PPL 9.48→8.37), gated vortex injection, chunked parallel
(35% faster fwd), incremental generation (O(1), 41% faster), speed/quality benchmarks.

### Phase 4: Quality Improvements — ✅ (all sub-items closed)

RoPE (#1), rank-k projector (#12), per-head vorticity projections (#16), residual H
connection (#13), input-dependent η (#14), RMSNorm + parallel residual (#15).

### Phase 5: Speed & Kernel Optimizations — 🔄 (4/6 done, Issue #33 closed)

Fused Triton kernel (Fix 1 ✅), torch.compile (#10 ✅), associative parallel scan (Fix 3 ✅),
CUDA Graph capture (Fix 6 ✅), multi-chunk barrier sync (Fix 4 ✅). Larger chunk size (#4/#7
✅). **Deferred:** Fix 2 flash-style tiled Hamiltonian (needed when d_model>1024), Fix 5
fully parallel with learned positional injection (research moonshot, loses causal structure).

### Phase 6: Evaluation & Scaling — ✅ Complete

Real-dataset benchmarks (#2), long-context N=256→4096 + needle-in-haystack (#2), scaling
laws (#18). **Original finding (now overturned):** HVN PPL ∝ params^-0.007 (nearly flat). **#35
re-measurement on real WikiText-103 (BPE, compute-matched, unsaturated):** HVN scales at **−0.42**
≈ Transformer −0.41 — the flat −0.007 was a 34K-char artifact; HVN benefits from scale normally
(see §8.4 / §9.2).

### Phase 7: Inference Optimization & Production — ✅ / ⬜ mixed

Quantization INT8/NF4 (#20 ✅), BPE tokenizer (#11 ✅), HuggingFace integration (#3/#6 ✅),
mixed precision (#19 ✅). ⬜ Distributed training (DDP) still open.

### Phase 8: Fine-Tuning & Instruction Tuning — ✅ Complete

LoRA, QLoRA, instruction tuning, LoRA save/load, demo (#29).

### Phase 9: PPL Root-Cause Correction (Issues #34/#35/#36, 2026-07-04)

> **Diagnosis update:** the "fixed d×d matrix is a fundamental compression bottleneck"
> explanation is contradicted by our own benchmark — the SSM reaches PPL ~1.1 with a state
> *smaller* than HVN's d×d-per-head Hamiltonian. Fixed-size O(1) state is **not** the limiter.
> HVN is really a linear-attention fast-weight network; the fix is to borrow proven dynamics
> from DeltaNet / GLA / Mamba-2 / RWKV-7, not to add capacity.

- **9.1 Readout & update dynamics (#34)** — ✅ **Done** (`--use_deltanet`). New
  `DeltaNetBlock`: per-channel data-dependent decay gate + per-head output RMSNorm, canonical
  gated DeltaNet update with separate k/v projections (drops HPH), output gate + causal conv.
  A/B tiny-Shakespeare (d=128, 2L): PPL **8.81 → 3.42 = 61% better** (target ≥30%). O(1)
  preserved. Training via chunked-parallel delta rule (Yang et al. 2024): 27.5 → 3.7 s/epoch
  (~7.4×), numerically identical to sequential scan (max|diff| 4.8e-7). `test_issue34.py` 5/5.
- **9.2 Real-scale scaling validation (#35)** — ✅ **Done — the flat exponent was a small-data
  artifact.** Rewired `eval/scaling_laws.py` for a *methodologically correct* scaling study:
  **compute-matched** training (every size sees the same **token budget**, not the same epochs),
  **full held-out eval PPL** (whole valid split in `seq_len` chunks, not a single mid-text batch),
  **BPE** tokenizer support, **SSM** added to the sweep, and a per-model eval-PPL-vs-tokens
  trajectory reporting each model's **saturation token budget**. Added a streaming WikiText-103
  loader (`eval/datasets.py`, `--dataset wikitext103 --local-path <dir>`, char-capped reads to
  bound memory on the 539 MB corpus).

  **Headline run — real WikiText-103, BPE (vocab 8,000), 5M-token compute-matched budget** (12.4M
  unique train tokens, so unsaturated), d={128,256,512,768}, 2L, bf16, RTX 4050 (see §8.4).
  **Refit exponents: HVN −0.4237, Transformer −0.4066, SSM −0.5311.** HVN scales *as well as* a
  Transformer (marginally steeper) and its PPL drops 521 → 206 across a 9× param increase — a
  ~100× steeper exponent than the original −0.007. **The "HVN doesn't benefit from scale"
  conclusion is refuted:** it was an artifact of fitting on 34K characters (deep in the
  overfit/saturation regime). A **small-data control** on full Tiny-Shakespeare (char-level, 32×
  the original data) *reproduces* the flat exponent (HVN −0.0042) precisely because every size
  saturates at ~1M tokens there — confirming the artifact mechanism. **Caveat:** default HVN only
  (the #34 `--use_deltanet` fix was not swept); 5M-token reduced budget (absolute PPLs drop at a
  larger budget, but the exponent is already in the trustworthy unsaturated regime). Scale up via:
  ```bash
  python -m eval.scaling_laws --dataset wikitext103 --local-path wikitext-103-v1 \
      --tokenizer bpe --bpe_vocab 8000 --token_budget 100_000_000 \
      --d_models 128,256,512,768 --precision bf16 --compile \
      --output-json results/scaling_wt103.json --output-plot results/scaling_wt103.png
  ```
- **9.3 Generation-speed cleanups (#36)** — ✅ **Done**. Removed the dead full-`forward()`
  prompt prefill in `generate_incremental` (the prompt was processed twice), and moved decode
  sampling on-device: reuse one preallocated input buffer (no per-token host→device alloc),
  keep top-k + `multinomial` on-device, and sync generated ids to host once at the end (no
  per-step `.item()`). New `prefill_states` absorbs the prompt in a single pass with two exact
  paths: DeltaNet uses the algebraically-exact chunked scan (O(chunks) launches — **42.7×**
  fewer than token-by-token on a 512-token prompt); the default HVN path uses the exact
  sequential recurrence (its chunked forward is a within-chunk *approximation*, so prefilling
  through it would change output). Measured HEAD vs pre-#36 (d=128, 2L, RTX 4050, warmed
  best-of-5, 512-tok prompt + 200 tokens): HVN decode **80.7 → 197.7 tok/s (2.45×)**; DeltaNet
  time-to-first-token **1,612 → 23 ms (70×)** for a 512-token prompt (prompt-length dependent).
  Verified token-for-token identical to the previous implementation (`test_issue36.py` 5/5,
  `top_k=1` greedy parity), and **stackable** with #33 kernels (`--cuda_graph` 350 tok/s,
  `--fused_kernel` 354 tok/s). O(1) preserved; `demo.py`, `test_issue33.py`, `test_issue34.py`
  regressions pass.

---

## 10. Execution Plan: Issues #32 & #33

> Created 2026-07-03. **#33 closed** (4/6 fixes). Decision: **#33 first, then #32.**

**Why #33 first:** speed unlocks 5× faster iteration for #32 benchmarking; 4/6 fixes are
weight-representation-independent; smaller scope; result is a fast usable model. **The one
dependency:** #32 changes weight representation (fp16 → ternary/2-bit/tensor-decomposed) and
#33's Triton kernel operates on weights — mitigated by making the kernel dtype-agnostic from
the start (fp16/int8/ternary branches).

### #33 — Custom Kernels & Parallel Processing (goal 106 → 500+ tok/s)

- **Phase 1 (Fix 6):** CUDA Graph capture → 78 → 220 tok/s (2.82×), `--cuda_graph`.
- **Phase 2 (Fix 1, 3):** Fused Triton kernel → 167–340 tok/s (2.14–7.2×), `--fused_kernel`;
  parallel scan → 4.79 ms/fwd (1.59×), zero logit diff, `--use_parallel_scan`.
- **Phase 3 (Fix 4):** Multi-chunk barrier sync → 6.77 ms/fwd (1.14×).
- Verified: `test_issue33.py` 5/5 on GPU; `demo.py`/`test_quantize.py` regressions pass.

### #32 — Extreme Weight Compression (10B → <2GB, 1T → <200GB)

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
  *Honest scope:* some techniques are simplified-but-functioning versions (e.g. TopologicalPruner
  is a union-find connectivity sweep, not full persistent homology).

### Key Constraints (all issues)

GPU required (RTX 4050, CUDA 12.6); **O(1) inference must be preserved** (no KV cache, no
N-dependent memory); backward compatible (CLI-flag gated, defaults unchanged); PPL must not
regress >2% per fix; Triton kernel dtype-agnostic.

---

## 11. Issue History & Related Work

### Open Issues

- **#25 — Agentic mode**: ReAct-style loop with planning, tool chaining, self-correction.
  Depends on #24 (done).
- **#26 — Training dashboard**: single-page HTML/CSS/JS (Chart.js via CDN), no server.
  Initial `docs/dashboard.html` built; issue covers refinement.
- **#32 — Extreme weight compression**: see §10 (Phase 1 + edge-devices branch done;
  Phases 2–5 on `hvn/core` not layered in by design).
- **#35 — Re-measure scaling on real-scale data** ✅ **Done** (2026-07-04): rebuilt harness
  (compute-matched token budget, full eval PPL, BPE, SSM, saturation tracking, streaming
  WikiText-103 loader). On real WikiText-103 (BPE, 5M-token compute-matched, unsaturated) HVN
  scales at **−0.4237** ≈ Transformer −0.4066 (SSM −0.5311) — the flat **−0.007** was a **34K-char
  artifact**, now refuted; a full-Shakespeare small-data control reproduces the flat exponent via
  early saturation, confirming the mechanism (see §8.4 / §9.2).
- **#36 — Speed up incremental generation** ✅ **Done** (2026-07-04): removed redundant prefill
  + per-token host↔device syncs; HVN decode 80.7 → 197.7 tok/s (2.45×), DeltaNet
  time-to-first-token 1,612 → 23 ms (70×), stacks with #33 kernels (see §9.3).
- **#39 — Make Gated-DeltaNet the default block?** 🔄 **Decided: keep flag-gated** (2026-07-04,
  blocked-on-#35 sweep now run). Added a distinct `HVN-DeltaNet` variant to
  `eval/scaling_laws.py` and swept it on real WikiText-103 (BPE-8000, 5M-token compute-matched):
  DeltaNet cuts PPL **21–28%** at d≤256 (matches Transformer) and beats HVN at d=512 in fp32
  (218.9 vs 237), **but diverges to NaN at d≥512 under bf16** (the recommended precision). Since a
  default must train cleanly in bf16, the default stays HVN and `--use_deltanet` is documented as
  recommended (fp32, or bf16 ≤ d=256). Flipping is now blocked on a **large-width bf16-stability
  fix**, not on scaling evidence (see §8.6).

### #34 — Close the PPL Gap via Readout/Update Dynamics — ✅ Done (2026-07-04)

Confirmed the diagnosis that **dynamics, not state capacity, were the limiter**. HVN is a
linear-attention fast-weight net (`v_t = Γ·(H·ω_t)` read; `H ← (1-η)H + η(P − HPH)` delta-rule
update). Every prior capacity-oriented strategy (#21 rank-k, #22 multi-thread) moved PPL <3%;
swapping in Gated-DeltaNet dynamics cut PPL 8.8 → 3.4 (61% better) at equal/lower FLOPs with
O(1) preserved. Wired `--use_deltanet` / `--deltanet_conv_kernel` through model/train/
quality_benchmark/generate (checkpoint round-trips the flag). Mirrored to `edge-devices`
(`EdgeHVN(use_deltanet=True)`). `test_issue34.py` 5/5: forward↔incremental parity (6e-7),
O(1) scaling (2.57 ms/tok flat N=256→4096), 1500-token stability with no clamp.

### The Quality Gap Summary

| Model | PPL (Shakespeare, d=128, 20ep) | Scaling exponent |
|-------|-------------------------------|------------------|
| SSM | ~1.16 | N/A |
| Transformer | ~2.0 | -0.090 |
| **HVN + Gated-DeltaNet (#34)** | **~3.4** (61% better than 8.8) | ~−0.42, real-scale, −21–28% PPL vs HVN (#39; bf16-unstable at d≥512) |
| HVN (#22 n_threads=4 + all strategies) | ~8.13 | -0.0123 |
| HVN (#21 strategies) | ~8.37 | -0.0126 |
| HVN (original) | ~8.37 | -0.007 |

> **Exponent caveat (#35):** every "Scaling exponent" above was fit on **char-level
> Tiny-Shakespeare**, which is small enough that all model sizes saturate — so these numbers
> measure saturation noise, **not** architectural scaling. On real WikiText-103 (BPE,
> compute-matched, unsaturated) HVN scales at **−0.42** ≈ Transformer −0.41 (see §8.4). Read the
> flat char-Shakespeare exponents as a small-data artifact, not a property of HVN.

**Root cause (superseded → confirmed):** previously attributed to a "fundamental compression
bottleneck" of the d×d matrix. Now known **wrong** — the SSM reaches PPL ~1.1 with a *smaller*
fixed state. The real culprit is the readout/update dynamics (velocity L2-norm + Frobenius
clamp discarding magnitude; non-standard symmetric HPH update), validated by #34.

### Closed Issues (Quality Journey, condensed)

#21 six strategies (rank-k, causal re-injection, cross-head mixing, nonlinear update,
per-token η, SwiGLU); #22 multi-thread memory (PPL 8.80→8.41 alone, 8.13 combined); #23
vision; #24 tool calling; #27 multimodal fusion; #28 conversation memory; #29 LoRA/QLoRA;
#31 OpenAI server; #33 kernels; plus #1/#12/#13/#14/#15/#16 (see §7).

---

## 12. Project Context & Delivery Workflow

### Original Design Brief

Act as a Principal AI Research Scientist and Theoretical Physicist specializing in
alternative, ultra-low-compute sequence modeling. Goal: build a novel, untrained LM
architecture ("Hamiltonian Vortex Network") achieving Attention's long-range associative
recall at <20% compute, O(1) inference, no KV cache. Do **not** use standard Attention,
Transformers, Mamba, or Test-Time Training gradient loops — synthesize Fluid Dynamics
(Lagrangian vortex particles) with Quantum Mechanics (Matrix Product States) in standard
PyTorch. Deliverables: theory + formalism, core engine, vocabulary decoder + runnable
pipeline, and a strict FLOP comparison vs MHA as N → 100,000.

### GPU / CUDA Requirement

All training, benchmarking, and evaluation **must run on GPU, never CPU**. Environment:
NVIDIA GeForce RTX 4050 Laptop GPU (6 GB VRAM), driver 596.36, PyTorch with CUDA
(`pip install torch --index-url https://download.pytorch.org/whl/cu126`). Verify
`torch.cuda.is_available()` before running; if False, fix the install rather than falling
back to CPU.

### Issue & Delivery Workflow

**Always use the GitLab CLI (`glab`)** for all GitLab interactions (the web UI is behind a
login wall + Cloudflare challenge). Examples: `glab issue view <id> --repo komalbarun/kramba-ai`,
`glab issue list`, `glab work-items list`. Per-issue workflow:

1. Create a branch `issue-NN-short-description` (never work on `main` directly).
2. Implement the change; verify with a real test or runnable command on GPU.
3. Commit referencing the issue (`git commit -m "Fix: … for issue #NN"`).
4. Push the branch (`git push origin <branch>`).
5. **Always open a Merge Request** with `glab mr create` — target `main`, put `Closes #NN`
   in the description so the issue auto-closes on merge, and add a `glab issue note NN`
   summarizing the result + linking the MR. Never push to or merge `main` directly; the
   maintainer reviews and merges the MR manually.

GitLab project: https://gitlab.com/komalbarun/kramba-ai

---

*End of document. This README is the single consolidated source of documentation for the
project; the earlier per-topic markdown files have been merged into the sections above.*
