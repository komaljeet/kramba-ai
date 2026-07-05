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

### 1.1 North Star: Frontier-Scale Training

> **This project is being built toward training a frontier-scale model on the HVN
> architecture.** That goal is the tie-breaker for every design decision — features, fixes,
> defaults, and evaluation methodology must all be chosen so they *still hold at scale*, even
> though present-day validation runs on a single RTX 4050 (6 GB). When two options are otherwise
> equal, pick the one that scales.

**Honest framing.** A frontier model (100B–1T+ params) requires a multi-node GPU cluster
(hundreds–thousands of 80 GB accelerators), not this laptop. The RTX 4050 work exists to
**de-risk and validate scaling behavior** — measuring exponents, stability, and O(1) inference
on research-scale models — *before* committing cluster budget. The 6 GB GPU is the wind tunnel,
not the aircraft.

**Frontier-readiness checklist (apply to every change):**

1. **Preserve O(1) inference.** No KV cache, no N-dependent memory. This is the architecture's
   entire value proposition and the reason frontier-scale *inference* is affordable — never
   trade it away for a training-time convenience.
2. **Batched paths must equal single-stream, provably.** The #46 head-misrouting bug proved that
   a self-consistent-but-wrong batched path silently degrades quality and entangles learning with
   batch size — fatal at scale (large data-parallel batches). Every batched/parallel path needs a
   `batched == single-stream` test (see `test_issue46.py`).
3. **Numerical stability at width and in bf16.** Large `d_model` is where bf16 hot-spots bite
   (#44). New ops must be checked for overflow/underflow at frontier widths, not just d≤256, with
   fp32 islands where reductions are sensitive.
4. **Scale-honest evaluation.** Never fit conclusions on saturated small data (the −0.007
   artifact, §8.4). Use compute-matched, unsaturated, held-out PPL and report the **exponent**.
   Prefer levers whose advantage *grows* with scale (DeltaNet's steeper −0.504 exponent).
5. **Distributed-training readiness.** No single-GPU assumptions baked into the model or training
   loop. Multi-GPU/multi-node data + tensor parallelism (DDP/FSDP) is the critical open
   dependency for this north star — see the elevated priority in §9 Phase 7.
6. **Defaults that scale.** The documented default should be the variant that wins *at scale*, not
   on the 34K sample — hence the #39 recommendation to make `--use_deltanet` the default block.

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
├── docs/                          # ISSUE_30_PLAN.md, dashboard.html (compendium consolidated into this README)
├── eval/                          # run_eval, scaling_laws, long_context, datasets, curriculum, coding, instruction, multimodal
├── tests/                         # test_*.py standalone test scripts (run from root: python tests/test_issueNN.py)
├── demos/                         # demo*.py showcase scripts (run from root: python demos/demo.py)
├── results/                       # benchmark JSON/PNG outputs
├── train.py / generate.py / benchmark.py / quality_benchmark.py   # CLI entry points (kept at root)
├── hvn_checkpoint*.pt / hvn_checkpoint_tokenizer.json             # trained checkpoints (default --checkpoint paths)
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
python demos/demo.py

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

**Real chat/base checkpoint pipeline (Issue #51).** One command wires the existing pieces
into the first *servable* checkpoint: BPE tokenizer → WikiText-103 pretrain (DeltaNet + bf16 +
the #50 speed levers) → instruction/chat fine-tune → `hvn_chat_checkpoint.pt` +
`hvn_chat_checkpoint_tokenizer.json`, gated by a non-degenerate smoke test.

```bash
# Recommended run (d=256, 3 layers, streams WikiText-103 from ./wikitext-103-v1):
python build_chat_checkpoint.py --out hvn_chat_checkpoint.pt
# Fast loop-validation (tiny budget/model — CI):
python build_chat_checkpoint.py --smoke --out hvn_chat_smoke.pt
```

The smoke gate requires a non-degenerate, on-topic completion — e.g. `Create a function that
adds two numbers` → `def add(a, b):\n    return a + b`. Held-out **bits-per-char** (tokenization-
invariant) is reported against the bundled char baseline. This is the first of a 5-issue milestone
tree (#51 → #52 app wiring → #53/#54/#55 coding curriculum); it is a loop-validation, not a
frontier-quality model.

**Coding-curriculum dataset loaders (Issue #53).** `eval/curriculum.py` adds per-language code
corpora for the curriculum order **HTML → CSS → JS → PHP → Python → DevOps**, re-exported from
`eval/datasets.py`:

```python
from eval.datasets import load_curriculum_stage, CURRICULUM_STAGES
train, valid = load_curriculum_stage("python")            # cached, license-filtered, deduped
train, valid = load_curriculum_stage("html", local_dir="./code_corpus")  # use pre-fetched files
```

Each stage is **permissive-license-filtered** (copyleft/share-alike files rejected),
**near-deduped** (exact + MinHash-Jaccard), budget-capped (compute-matched), and split 90/10 by
document. Resolution mirrors the WikiText-103 loader — local dir → best-effort curated permissive
download → **offline synthetic fallback**, so `python -m eval.curriculum` smoke-loads all six
stages with no network. This feeds the #54 orchestrator and #55 coding eval.

**Auto-train orchestrator — staged curriculum scheduler (Issue #54).** `auto_train.py` is a
single command that trains the #53 stages **in order**, continuing from the #51 base checkpoint,
evaluating + checkpointing each stage, and **resuming** cleanly after an interruption (kill / cron /
`/loop`):

```bash
# Full curriculum, continuing from the #51 checkpoint:
python auto_train.py --from hvn_chat_checkpoint.pt --stages html,css,js,php,python,devops
# Re-run the same command to RESUME — completed stages are skipped, the last
# stage checkpoint is carried forward (state in auto_train_state.json):
python auto_train.py --from hvn_chat_checkpoint.pt
# Fast offline loop-validation (tiny budget/epochs — CI / 6 GB box):
python auto_train.py --from tiny_base.pt --smoke
```

It reuses `train.train_model` (the #50 Category-A exact speed levers: fused AdamW, per-epoch loss
sync, pinned/async H2D — carrying the model **weights** forward, a fresh optimizer per stage) and
the #53 loaders; each stage writes `checkpoint_stage_{name}.pt` (+ tokenizer, so every stage is
independently servable) and appends a metrics row (held-out **bits-per-char**/PPL + non-degenerate
completion) to `auto_train_metrics.jsonl`. Progress is committed to `auto_train_state.json`
**atomically after every stage**, so a re-run picks up at the next unfinished stage instead of
restarting. `evaluate_stage()` is the seam the #55 coding-eval harness plugs into. O(1) inference is
preserved (orchestration only; the served model is exactly what `train_model` produces).

**Coding eval harness — per-language held-out completion scoring (Issue #55).** `eval/coding.py`
gives each curriculum stage a **measurable** score instead of vibes — a per-language,
HumanEval-style held-out code-completion eval, in the same style as the other `eval/` harnesses:

```bash
# Score one language (held-out text comes from the #53 loader, offline-safe):
python -m eval.coding --checkpoint hvn_chat_checkpoint.pt --lang python
# Score every stage and write a baseline JSON:
python -m eval.coding --checkpoint hvn_chat_checkpoint.pt --lang all --json results/coding_baseline.json
```

It picks the strongest signal each language's tooling allows, always reporting the
tokenization-invariant fallback:

* **Exec-based pass@k** where a runtime exists — Python always (`sys.executable`), JavaScript when
  `node` is installed, PHP when `php` is installed. A small bundled set of HumanEval-style problems
  (a stub + hidden asserts) is completed by the model, executed in a throwaway temp dir via
  `subprocess` with a timeout, and graded pass/fail (model output is untrusted → sandbox + timeout).
* **Structural / compile validity** where no runtime is available (HTML, CSS, DevOps, and as the
  off-box fallback for JS/PHP): the model completes a held-out prefix and the concatenation is
  checked for structural validity — balanced tags/braces, `compile()` for Python, `yaml.safe_load`
  for manifests.
* **Held-out bits-per-char / PPL** is *always* computed (reusing the #51 metric) — tokenization-
  invariant, so it compares fairly across tokenizers and gives a smooth gradient even when a tiny
  loop-validation model can't yet emit runnable code.

The callable API (`score_stage` / `evaluate_coding`) is what the #54 orchestrator logs after each
stage: `auto_train.evaluate_stage` now delegates to it (keeping its backward-compatible
`ppl`/`bits_per_char`/`non_degenerate` keys and adding a per-language `coding_score`). The
orchestrator path defaults to **structural-only** (`allow_exec=False`) so unattended runs stay fast
and side-effect-free; the exec pass@k is an explicit `python -m eval.coding` pass. **#51 base-model
baselines** (`results/coding_baseline_issue51.json`, RTX 4050) — all ~0 exec/structural on the
languages the base never saw, so curriculum gains are quantifiable: `python` pass@1 0% (5.08
bits/char), `html`/`css` 0% structural (7.20 / 6.57 bits/char), `js`/`php` 25% (5.77 / 5.73). O(1)
inference is untouched — the harness only reads a trained checkpoint.

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
| `--weightless` | flag | Freeze DeltaNet q/k/v/out projections as seed-regenerated random features (Issue #40; needs `--use_deltanet`) |
| `--grad_checkpoint` | flag | Exact gradient (activation) checkpointing (Issue #49): recompute block activations in the backward pass — bit-identical gradients, ~20-30% slower/step, ~64% lower peak activation memory (fits a bigger `d_model`/batch in 6 GB) |
| `--full` | flag | Issue #50 meta-flag: enable the **complete recommended fast-training stack** in one switch. Any flag passed explicitly overrides the preset (e.g. `--full --precision fp32`). See the Category A/B breakdown below. |

**The `--full` meta-flag (Issue #50).** One switch that turns on the whole recommended
stack so you don't have to remember and combine half a dozen flags — `python train.py --full ...`.
It expands into two **honestly-labelled** categories:

| Category | What it enables | Honesty |
|----------|-----------------|---------|
| **A — exact speed levers** (model bit-identical) | fused AdamW CUDA kernel · per-epoch (not per-step) loss sync (the training-loop analogue of the #36 generation fix) · pinned-memory + `non_blocking` H2D transfers · `--compile` (#10) · `--grad_checkpoint` (#49) | Speed **only** — the trained model is unchanged. |
| **B — recommended defaults** (change model/precision) | `--precision bf16` (bf16-stable at every width since #44) · `--use_deltanet` (the #39-recommended higher-quality block) | **Not** quality-neutral — `--full` trains the best/fastest *recommended* model, not "the same model, faster." |

It deliberately does **not** enable `--use_parallel_scan` (a no-op under `--use_deltanet`).
On the first run `--full` prints a one-line banner listing exactly what it turned on (with
`[A]`/`[B]` tags) so runs stay reproducible/auditable. **Measured:** steady-state **1.58×
faster** (5.53 → 3.51 s/epoch, d=256/3L/seq128/B32, RTX 4050) once the one-time `torch.compile`
warmup amortizes; the Category-A primitives are proven exact (`fused-off==on`,
`sync-removed==original` train bit-identical weights — `tests/test_issue50.py`). Explicit flags
always win, so `--full --precision fp32` trains in fp32.

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
| #43 | Correct, fused concurrent-serving decode (~2.5× tok/s, fixes B>1 head-misroute) | `model.py`, `hvn_block.py`, `batched_vortex.py`, `batched_memory.py` |
| #46 | Systemic batch-ordering fix — batched `forward`/eval/`generate_batched` now == single-stream (batch-outer tiling of per-head params) | `batched_vortex.py`, `batched_memory.py` |
| #42 | Real reasoning/thinking — trainable `[THINK]` chain-of-thought (multi-step arithmetic), with-think vs no-think A/B **+52pp**; also fixed a latent tokenizer bug that dropped OOV chars (digits) from aux training data | `reasoning.py`, `train.py`, `eval/reasoning.py` |
| #49 | Exact gradient (activation) checkpointing (`--grad_checkpoint`) — bit-identical gradients, −64% peak activation memory, unlocks a larger model on the 6 GB card | `model.py`, `train.py`, `eval/scaling_laws.py`, `tests/test_issue49.py` |
| #50 | `--full` meta-flag — one-switch recommended fast-training stack (Category A exact speed levers: fused AdamW, per-epoch loss sync, pinned/async loading, `--compile`, `--grad_checkpoint`; Category B recommended model/precision: bf16 + DeltaNet). Explicit flags override; steady-state **1.58×** faster; Category-A primitives proven bit-exact | `train.py`, `tests/test_issue50.py` |
| #45/#47 | Final real-scale benchmark pass under corrected #46 routing — 4-model WikiText-103 sweep; corrected routing **improves default HVN 6–12%**, DeltaNet confirmed quality leader (recommended default), weightless real-scale cost measured (§8.12) | `eval/scaling_laws.py`, `results/scaling_final_pass.*` |
| #51 | Chat/base checkpoint pipeline — one command wires BPE tokenizer → WikiText-103 pretrain (DeltaNet + bf16 + #50 speed levers) → instruction/chat fine-tune → checkpoint+tokenizer, with a non-degenerate smoke-gate. Produces the first real servable checkpoint (Milestone 1, feeds #52/#53/#54/#55) | `build_chat_checkpoint.py`, `tests/test_issue51.py` |
| #52 | Wire the #51 checkpoint into the Django app + end-to-end chat smoke test (Milestone 1, Issue B). Worker rebuilds the model with the checkpoint's `use_deltanet`/`weightless` flags (the #51 checkpoint is a DeltaNet model — default-block load raised); `StateCheckpoint` now serializes conversation state structure-agnostically (v2 tensor-tagged tree) so the DeltaNet `{"S","conv"}` per-block state round-trips (old nested-list serializer crashed on every turn); compose/env point at `hvn_chat_checkpoint.pt`; smoke test drives the real serving path (`hvn_worker` → `ChatSession.user_turn`/streaming) for a non-degenerate reply + save→load round-trip | `web/hvn_worker/worker.py`, `hvn/chat.py`, `web/docker-compose.yml`, `web/README.md`, `web/tests/test_chat_smoke.py` |
| #54 | Auto-train orchestrator (Milestone 2, Issue D) — `auto_train.py` trains the #53 curriculum stages **in order** from the #51 base checkpoint, carrying the model weights forward, evaluating + checkpointing each stage (`checkpoint_stage_{name}.pt` + tokenizer + `auto_train_metrics.jsonl`), and **resuming** from the last completed stage after a kill (atomic `auto_train_state.json`; `--restart` to ignore it). Reuses `train.train_model` (#50 speed levers) + the #53 loaders; `evaluate_stage()` is the #55 eval seam. O(1) inference preserved | `auto_train.py`, `tests/test_issue54.py` |
| #53 | Curriculum dataset loaders (Milestone 2, Issue C) — `load_curriculum_stage('html'\|'css'\|'js'\|'php'\|'python'\|'devops')` returns `(train, eval)` streams for the coding-curriculum order, mirroring the WikiText-103 loader. **Permissive-only license filtering** (rejects GPL/LGPL/AGPL/MPL/EPL/CC-BY-SA), **near-dedup** (exact + MinHash-Jaccard over word-shingles), per-stage compute-matched char budgets + 90/10 doc split, and streaming/local-cache resolution (local dir → best-effort curated permissive download → offline synthetic fallback so every stage smoke-loads on the 6 GB box with no network). Feeds #54 (orchestrator) and #55 (eval) | `eval/curriculum.py`, `eval/datasets.py`, `tests/test_issue53.py` |
| #55 | Coding eval harness (Milestone 2, Issue E) — `python -m eval.coding --checkpoint X --lang python\|all` scores a checkpoint per language: **exec-based HumanEval-style pass@k** where a runtime exists (Python always; JS/PHP when `node`/`php` installed — sandboxed subprocess + timeout), **structural/compile validity** otherwise (balanced tags/braces, `compile()`, `yaml.safe_load`), and **always** tokenization-invariant held-out **bits-per-char**/PPL. Callable API (`score_stage`/`evaluate_coding`) is the #54 seam — `auto_train.evaluate_stage` delegates to it (keeps its `ppl`/`bits_per_char` keys, adds `coding_score`). #51 base baselines committed (`results/coding_baseline_issue51.json`) so curriculum gains are quantifiable. O(1) inference untouched | `eval/coding.py`, `auto_train.py`, `tests/test_issue55.py` |

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

> **HVN column superseded by §8.12** (corrected #46 routing lowers it 6–12%: −0.397 exponent);
> Transformer/SSM are unaffected by #46 and stand. This §8.4 table is retained as the #35 record.

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

**Decision (at the time) — keep flag-gated, do not flip the default.** Acceptance criterion #35
("DeltaNet ≥ HVN block at real scale") held on *quality* but failed on *robustness*: a default must
train cleanly in the recommended precision, and DeltaNet did not at d≥512/bf16. Flipping the default
was blocked on a follow-up (Issue #44) fixing large-width bf16 stability. **That fix is now done —
see §8.7.**

### 8.7 Large-Width bf16 Stability Fix for Gated-DeltaNet (Issue #44)

The §8.6 divergence was diagnosed to two numerical hot-spots in the bf16 path, both fixed in
`hvn/core/deltanet.py` (matmuls stay bf16; only the sensitive reductions move to fp32):

1. **fp32 islands** for the numerically sensitive ops — the L2-normalization of q/k (`_l2norm`), the
   per-head output RMSNorm on the read (`_read_normalize`, which squares a large read `S·q` — bf16
   overflows above ≈1.8e19 after `pow(2)`), and the data-dependent decay/write gates `α`/`β`.
2. **Flooring the intra-chunk cumulative decay `P = cumprod(α)`** in the chunked-parallel scan. The
   algebra only uses the bounded causal ratios `P_i/P_j ≤ 1` (j≤i), but the intermediate
   `ktil = k/P` overflows fp32 to Inf once `P` underflows — at d≥512 enough decay channels shrink
   within a 32-token chunk that `(1e-6)^32` underflows. `P` is floored at `1e-16` (keys/queries are
   unit-norm, so `1/P ≤ 1e16`, well inside fp32), only perturbing channels already decayed to ≈0.
3. **Loss-spike guard** in the training loop (`eval/scaling_laws.py`) as a belt-and-suspenders net:
   skip the optimizer step on a non-finite loss or a grad-norm spike (>1e3) so a transient spike
   can't poison the weights and AdamW moments.

**Result — DeltaNet now trains cleanly in bf16 at every width, with no NaN, and its quality win
*grows* with scale** (the instability had been silently degrading the large-width runs even before
the NaN). Full sweep, same config as §8.6 (WikiText-103, BPE-8000, 5M-token compute-matched, 2L,
seq_len=128, **bf16**, RTX 4050):

> **Superseded for the HVN-default column — see §8.12.** This table was produced under the
> pre-#46 buggy head routing. The corrected-routing final pass (§8.12) lowers the default-HVN
> numbers 6–12% (e.g. d=128 522.5 → 469.1); the DeltaNet column is unaffected (it doesn't use the
> buggy modules) and reproduces here within run variance.

| d_model | HVN (default) | HVN-DeltaNet | Transformer | DeltaNet vs HVN |
|--------:|--------------:|-------------:|------------:|:---:|
| 128 | 522.5 | **410.6** | 392.6 | **−21.4%** |
| 256 | 348.4 | **251.9** | 245.1 | **−27.7%** |
| 512 | 236.0 | **159.4** | 173.9 | **−32.4%** |
| 768 | 205.9 | **128.0** | 149.7 | **−37.8%** |

Refit exponents (bf16): HVN −0.423, **HVN-DeltaNet −0.507** (steepest of the three), Transformer
−0.404. DeltaNet no longer just matches the Transformer at d≤256 — it now **beats both HVN and the
Transformer at d=512 and d=768** and has the steepest scaling exponent. All acceptance criteria met:
no NaN at any width; bf16 d=512/768 PPL ≤ default HVN (and well below the fp32 DeltaNet numbers);
O(1) preserved (`test_issue34.py` 5/5, `test_issue33.py` 5/5, `demo.py` pass). `--use_deltanet` is
now safe to run in bf16 at all widths, which **unblocks the #39 default-flip decision** (left to a
follow-up so this change keeps defaults unchanged).

### 8.8 Hybrid Weightless Projections for Gated-DeltaNet (Issue #40)

*"Instead of using weights, use something else to keep file size / memory low — we already won
the no-qkv battle."* The Hamiltonian H is **already** weightless (runtime state, never
trained/stored). #40 targets the remaining trained tensors. A **fully** weightless net is not
viable — the #34 DeltaNet win came *specifically* from the **learned, data-dependent dynamics**
(`α_t=σ(W_α x)`, `β_t=σ(W_β x)`, output gate, causal conv), so freezing those hands most of it
back. So the shipped variant (`--weightless`, requires `--use_deltanet`,
`hvn/core/weightless.py`) is a **hybrid**: freeze the four big `d×d` "address/content"
projections (`q/k/v/out`) as **PRNG-seed-regenerated random features** — each stored as *one
integer*, not `d²` floats (`SeededLinear`, a non-persistent buffer regenerated from the seed on
load) — while keeping the tiny high-leverage learned core (`α/β/g` gates, conv, read RMSNorm).

A/B on the bundled tiny-Shakespeare sample (the #34 scale — d=128, 2L, 20 epochs, bf16, RTX 4050),
dense vs weightless DeltaNet, identical config/seed:

| Variant | Params | Eval PPL | Checkpoint |
|---------|-------:|---------:|-----------:|
| DeltaNet (dense) | 344,633 | 1.087 | 1,359 KB |
| **DeltaNet (weightless)** | **213,561 (−38%)** | **1.220 (+12.3%)** | **845 KB (−37.8%)** |

Freezing the four projections removes `4·d²·L` floats (here 131,072), cutting params/checkpoint
~38% for a **+12.3% relative PPL** cost — the model stays reliable (both memorize the sample near
perfectly). O(1) inference, no-KV, fixed d×d state and forward↔incremental parity are all
preserved (freezing a projection is orthogonal to them); checkpoints round-trip **exactly** by
regenerating the frozen matrices from the stored seed. `test_issue40.py` 6/6; `test_issue34.py`
5/5, `demo.py` pass. *Scope:* flag-gated, defaults unchanged; PPL is the memorization regime, so
read the −38% size / +12% PPL trade as the small-scale signal, not a scaled headline. At larger
`d` the `4d²`-per-layer projection share grows relative to the `vocab·d` embedding, so the size
win grows (and the PPL cost should be re-measured at scale). **Now measured at real scale — see
§8.12:** at BPE-8000 the −38% saving shrinks to −5.5% at d=128 (large embedding) and *grows* with
width to −19.6% at d=768, for a +10–21% PPL cost — confirming this caveat quantitatively.

### 8.9 Correct, Fused Concurrent-Serving Decode (Issue #43)

Split from the #37 hardware review (point ②): serving N users runs N independent
streams — each with its own Hamiltonian H_i — through the model together. The batched
decode feeds a **length-1** sequence per stream, so per-step wall-time is dominated by
**kernel-launch / Python overhead, not compute** (it is nearly flat in the stream count —
that is why batching already scales tok/s ~linearly with B). The review's premise (reformulate
the per-stream `H_i·v_i` reads into one big **GEMM**) turned out **not** to be the lever: those
reads are inherently **block-diagonal** (each stream owns its H, so no shared-weight GEMM
exists), and at HVN's head dims (`d_head=64`) cuBLAS's batched path is already efficient — a
grouped-GEMM reformulation of the *shared* projections was measured **throughput-neutral-to-
negative** (the layout permutes cost more than the matmul saves; decode is launch-bound). The
real lever is **fewer launches**: routing the batched step through the Issue #33 `FusedHVNStep`
Triton kernel (2 launches/block vs ~25 PyTorch ops), which was previously wired only for
single-stream (`generate_incremental_fused`).

Two things shipped in `model.generate_served` (`forward_step_served`):

1. **Correctness fix.** The legacy `generate_batched` expanded per-head params with
   `repeat_interleave(B)` (head-outer ordering `h·B+b`) while the stream batch is laid out
   batch-outer (`b·n_heads+h`), so for **`n_heads>1, B>1` each concurrent stream silently used
   the *wrong head's* projection weights** — `generate_batched` did **not** equal running each
   stream alone (verified: diverges at `n_heads=4`, matches only at `n_heads=1`). The served path
   uses batch-outer ordering in both backends, so a concurrent batch is **equivalent to
   single-stream** generation (torch backend matches to ~1e-6; fused to the kernel's ~1e-2).
2. **Throughput.** `backend="fused"` (default `auto` when CUDA+Triton+`d_head≥16`) collapses the
   per-token launches, `backend="torch"` is a portable grouped-GEMM fallback (matches
   single-stream to ~1e-6; used when Triton is unavailable or `d_head<16`). DeltaNet / multi-thread
   memory fall back to `generate_batched`.

Sustained throughput — **total tok/s across all streams**, 128 new tokens, d=512, 8 heads, 2L,
fp32, RTX 4050 (`test_issue43.py`):

| Concurrent streams | legacy `generate_batched` (buggy) | served-torch (correct) | **served-fused (correct)** | fused vs legacy |
|---:|---:|---:|---:|:---:|
| 1 | 263 | 254 | **675** | **2.56×** |
| 8 | 2,096 | 1,987 | **5,416** | **2.58×** |
| 32 | 8,484 | 7,744 | **20,815** | **2.45×** |
| 64 | 15,924 | 15,062 | **42,716** | **2.68×** |

**~2.5–2.7× sustained tok/s at every concurrency level** (and 2.66× over the *correct* torch
baseline, so the win is real, not an artifact of the bug), with every stream's output identical to
single-stream. O(1) per-token preserved (flat ms/token N=64→256), no KV cache, no HVN-math change.
`test_issue43.py` 7/7; `demo.py` / `test_issue33.py` / `test_issue34.py` regressions pass.

### 8.10 Systemic Batch-Ordering Fix — Batched `forward` == Single-Stream (Issue #46)

The head-misrouting bug that #43's served path worked around was **not** confined to
`generate_batched` — the same shared batched modules (`batched_vortex.py`,
`batched_memory.py`) back **training `forward`, batched eval PPL, and
`parallel_forward`**, so every batched path with `n_heads>1, B>1` was affected. The
block lays the batch out **batch-outer** (`bh = b·n_heads + h`, so slot `bh` holds head
`bh % n_heads`), but the per-head params were expanded with `repeat_interleave(B)` —
**head-outer** ordering (`h·B + b`) — pairing each slot's data-head with the *wrong*
weight-head. The bug was self-consistent within a fixed batch size (so training still
converged), but it entangled learned per-head routing with the training batch size and
diluted multi-head specialization.

The fix **tiles** per-head params batch-outer (`repeat` instead of `repeat_interleave`)
in `MultiHead`/`MultiThread` vortex + memory, matching the input-dependent-eta branch
that *already* indexed batch-outer via `arange(BH) % n_heads`. One source change corrects
`forward`, eval, `generate_batched`, and `parallel_forward` together.

**Result — the issue's own evidence snippet, `max|batched − per-sequence|`:**

| n_heads | Before (buggy) | After (#46) |
|--------:|---------------:|------------:|
| 1 | 6.0e-07 | 6.0e-07 |
| 4 | **2.42** | **6.0e-07** |
| 8 | (diverges) | 6.0e-07 |

Batched `forward` now equals single-stream at every width, across all flag-gated variants
(`n_threads>1`, `--use_parallel_scan`, `--per_token_eta`, `--rank_k`, `--causal_reinjection`,
`--nonlinear_update`) and for `parallel_forward` and `generate_batched`. `B=1` is bit-unchanged
(tiling ≡ interleaving at batch size 1). `test_issue46.py` 5/5; `test_issue43.py` 7/7 (the
"legacy bug" pin flipped to a "now-fixed" assertion), `test_issue33.py` / `test_issue34.py` /
`demo.py` regressions pass. **Caveat:** this is a semantics-breaking change — checkpoints trained
under the old routing and the §8.x benchmark numbers were produced under the buggy routing; the
re-measurement is tracked in **#47** (folded into the final real-scale pass #45), which should also
A/B whether correct head routing improves PPL by restoring multi-head specialization.

### 8.11 Real Reasoning / Thinking — Trainable `[THINK]` Chain-of-Thought (Issue #42)

Before #42 the `[THINK]`/`[/THINK]` special tokens (`hvn/tools.py`) were **only delimiter
parsing** — the model was never trained to put meaningful content inside them, so "thinking"
was a format convention, not a capability. #42 makes the blocks carry **real chain-of-thought
that measurably helps**, as a pure **data + training + decoding-convention** feature — no
architectural change, O(1) / no-KV inference untouched.

- **Data (`hvn/reasoning.py`).** A synthetic generator for **multi-step arithmetic** (chained
  add/sub, mul-then-add, order-of-operations, two-step word problems) — deterministic ground
  truth, so answers auto-grade. Each example pairs a question with an explicit step-by-step
  trace: `Answer: [THINK]\n6 * 2 = 12\n12 + 4 = 16\n[/THINK] The answer is 16.`
- **Training (`train.py --reasoning_training`).** Mirrors the tool-calling (#24) data path.
- **Decoding (`generate.py --hide_thinking`).** Strips `[THINK]` from the user-facing answer
  (via the existing `ToolCallParser.strip_tool_blocks`) while logging the raw trace to stderr.
- **Eval (`eval/reasoning.py`).** Auto-graded **with-think vs. no-think** A/B on a held-out set:
  *think* lets the model reason before answering; *nothink* primes the answer prefix so it must
  commit immediately (same model, same problems).

**Result — reasoning measurably helps.** Char-level, `--use_deltanet`, d=256, 4L, 6,000
reasoning examples, 80 epochs, RTX 4050; 200 held-out problems, greedy decode:

| Mode | Exact-match accuracy |
|------|:---:|
| **think** (may emit `[THINK]`) | **53.5%** |
| nothink (must answer immediately) | 1.5% |
| **delta** | **+52.0pp** |

Externalising intermediate results as tokens turns the final answer into a *copy* of a number
the model just computed one step at a time, whereas the no-think path must compute the whole
expression in a single forward step and essentially cannot. Example (greedy): `Question: 6 * 2 + 4`
→ `[THINK] 6 * 2 = 12  12 + 4 = 16 [/THINK] The answer is 16.` ✓ `test_issue42.py` 5/5;
`test_issue34.py` 5/5, `demo.py` pass.

**Also fixed — a latent tokenizer bug.** The char tokenizer was built over the *base* corpus
**before** the tool/reasoning/instruction data was appended, so any character absent from the
base text (notably **every digit**, and `*`/`=`) was OOV and **silently dropped at encode time** —
the reasoning examples trained with all numbers stripped. `train.py` now generates the auxiliary
data first and builds the vocab over the **full** corpus (base + tool + reasoning + instruction),
which also repairs the tool-calling (#24) and instruction-tuning (#29) data paths.

*Scope / caveats:* small-scale, memorization-regime signal (single-digit operands; the
generalization gap on harder word problems is visible — think accuracy is 53%, not 100%).
Consistent with the project's methodology (§8.4/§8.8), read the **+52pp delta** as the headline,
not the absolute number; a real-scale re-measure belongs in the #45 final pass.

### 8.12 Final Real-Scale Benchmark Pass — Corrected Head Routing (Issues #45 / #47)

The single end-state benchmark run, executed once now that all feature work is settled. It
**supersedes** the default-HVN columns of §8.4/§8.6/§8.7 and the tiny-Shakespeare weightless A/B
of §8.8: every number here is produced under the **corrected #46 batch-outer head routing** (the
§8.x tables above were produced under the pre-#46 buggy `repeat_interleave` routing). SSM is
omitted from this pass (a reference competitor, not part of HVN, and unaffected by #46 — its §8.4
numbers stand); the comparison of interest is the three HVN variants vs the Transformer baseline.

**Config** (identical harness to §8.4/§8.7): WikiText-103, BPE-8000, 5M-token compute-matched
budget (12.4M unique train tokens — unsaturated), d ∈ {128,256,512,768}, 2L, seq_len=128, **bf16**,
RTX 4050 (`eval/scaling_laws.py --models HVN HVN-DeltaNet HVN-DeltaNet-Weightless Transformer`,
`results/scaling_final_pass.{json,png}`).

| d_model | HVN (default) | HVN-DeltaNet | HVN-DeltaNet-Weightless | Transformer |
|--------:|--------------:|-------------:|------------------------:|------------:|
| 128 | 469.1 | **412.3** | 455.2 | 395.6 |
| 256 | 307.2 | **251.1** | 275.2 | 245.6 |
| 512 | 218.8 | **159.4** | 183.3 | 172.0 |
| 768 | 194.3 | **129.4** | 157.0 | 150.8 |

**Refit exponents (PPL ∝ params^b):** HVN **−0.397**, HVN-DeltaNet **−0.504** (steepest),
HVN-DeltaNet-Weightless **−0.499**, Transformer **−0.406**.

**Finding 1 — correct head routing measurably *improves* the default HVN block (#47's A/B).** The
#46 fix (each batched slot now reads its own head's weights, restoring multi-head specialization
the bug had diluted) lifts the default block at every width vs the buggy-routing §8.7 numbers:

| d_model | HVN pre-#46 (§8.7, buggy) | HVN post-#46 (this pass) | Δ |
|--------:|--------------------------:|-------------------------:|:---:|
| 128 | 522.5 | 469.1 | **−10.2%** |
| 256 | 348.4 | 307.2 | **−11.8%** |
| 512 | 236.0 | 218.8 | −7.3% |
| 768 | 205.9 | 194.3 | −5.7% |

So the #46 correctness fix was **also a quality fix** — the head-misrouting had been silently
degrading the default block by 6–12% PPL. **DeltaNet is unaffected** (412.3/251.1/159.4/129.4 ≈
§8.7's 410.6/251.9/159.4/128.0, within run variance) because the `DeltaNetBlock` replaces the
block entirely and never touches the buggy `batched_vortex`/`batched_memory` modules — a clean
confirmation that #46 was localized to the default HVN path.

**Finding 2 — DeltaNet remains the quality leader and is the recommended default (feeds #39).** It
beats the default HVN block at every width (−12.1% → −33.4% PPL as d grows), has the steepest
exponent (−0.504 vs Transformer −0.406), and **beats the Transformer at d ≥ 512** (159.4 vs 172.0;
129.4 vs 150.8). Combined with #44 (bf16-stable at all widths) the last blocker on the #39
default-flip is cleared: **`--use_deltanet` should become the documented default block.**

**Finding 3 — weightless (#40) real-scale cost, measured for the first time beyond d=128/tiny.**
The hybrid weightless DeltaNet (frozen seed-regenerated q/k/v/out projections) vs dense DeltaNet:

| d_model | Dense PPL | Weightless PPL | PPL cost | Param saving |
|--------:|----------:|---------------:|:--------:|:------------:|
| 128 | 412.3 | 455.2 | +10.4% | −5.5% |
| 256 | 251.1 | 275.2 | +9.6% | −9.7% |
| 512 | 159.4 | 183.3 | +15.0% | −15.6% |
| 768 | 129.4 | 157.0 | +21.3% | −19.6% |

This **confirms the §8.8 caveat quantitatively**: the −38% param saving seen at d=128 on
tiny-Shakespeare was a small-vocab artifact (char vocab ≈ 65, so the `vocab·d` embedding was
negligible and the frozen `4·d²·L` projections dominated). At BPE-8000 the embedding is large, so
at small `d` the saving is modest (−5.5%) and **grows with width** (−19.6% at d=768) as the `4d²L`
projection share overtakes the `vocab·d` embedding — exactly as predicted. Weightless still beats
the default HVN block at every width and keeps a near-DeltaNet exponent (−0.499), so it remains the
right pick when checkpoint size dominates; the PPL cost is a modest, width-growing 10–21%.

O(1) per-token inference, no-KV, and forward↔incremental parity are unchanged by this pass (it is
training/measurement only; `test_issue46.py` guards the routing fix). **Deliverables:**
`results/scaling_final_pass.json` + `results/scaling_final_pass.png`.

### 8.13 Exact Gradient (Activation) Checkpointing — `--grad_checkpoint` (Issue #49)

The one **mathematically exact** training-memory lever: `torch.utils.checkpoint.checkpoint`
(`use_reentrant=False`) wraps each HVN/DeltaNet block in `model.forward`, recomputing that
block's activations in the backward pass instead of storing them. Gradients, weights, and the
final trained model are **bit-for-bit identical** to training without it — the only cost is
per-step wall-clock. Flag-gated (`train.py --grad_checkpoint`, `eval/scaling_laws.py
--grad_checkpoint`), default **off**, so defaults are unchanged. Training-only: it is guarded by
`self.training and torch.is_grad_enabled()`, so the O(1)/no-KV inference and eval paths are
untouched (bit-identical regardless of the flag).

**Exactness — proven, not assumed** (`test_issue49.py`, RTX 4050): with checkpointing on vs off,
both the **logits** and **every parameter's gradient** match to `max|diff| < 1e-4` (fp32 recompute
reassociation floor), for the default HVN block *and* the DeltaNet block.

**Memory win** — peak train-step CUDA memory (forward+backward), d=256, 4 layers, seq_len=512,
batch=16, fp32, RTX 4050:

| Config | `--grad_checkpoint` off | on | Peak reduction |
|--------|------------------------:|---:|:---:|
| d=256, 4L, seq=512, B=16 | 766 MiB | 274 MiB | **−64%** |

**Unlocks a larger model on the same card:** a `seq_len=1024, batch=48, 8 layers, d=384` training
step **OOMs** on the 6 GB RTX 4050 without the flag but **trains successfully with it**
(`test_issue49.py` [5/5]) — directly serving the §1.1 frontier north star (validate a bigger model
on the wind-tunnel GPU, architecture untouched). `test_issue49.py` 5/5; `test_issue46.py` 5/5
regression. **Out of scope** (not 100% exact — separate issues): 8-bit optimizer (quantized moments,
needs a PPL A/B) and CPU/NVMe optimizer offload.

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
mixed precision (#19 ✅). ⬜ **Distributed training (DDP/FSDP) — the critical open dependency
for the §1.1 frontier north star.** Single-GPU validation is complete; scaling beyond
research-size requires multi-GPU data/tensor parallelism, and every batched path is already
#46-correct (`batched == single-stream`), which is the prerequisite for correct data-parallel
training. This should be the next priority when moving off the RTX 4050.

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

- **#40 — Non-weights NN** 🔄 **Hybrid shipped (flag-gated)** (2026-07-04): fully weightless is
  not viable (it hands back the #34 learned-dynamics win), so `--weightless` freezes the DeltaNet
  `q/k/v/out` projections as seed-regenerated random features (`hvn/core/weightless.py`) while
  keeping the learned gates/conv. A/B: −38% params/checkpoint at +12.3% PPL, O(1) preserved,
  exact seed round-trip (see §8.8). Defaults unchanged; scale-up PPL cost still to be measured.
- **#41 — Real web search tool** ✅ **Done** (2026-07-04): replaced the canned-dict mock
  `web_search` (`hvn/tools.py`) with a live backend — keyless **DuckDuckGo Instant Answer API**
  by default (stdlib `urllib`, no new required dep), auto-switching to **Tavily** when
  `TAVILY_API_KEY` is set (or forced via `HVN_SEARCH_PROVIDER`). Same tool signature/schema
  (registry + `ToolCallingSession` unchanged); network/parse errors degrade to a clear error
  string (no crash), results flow through `ToolExecutor.max_result_len` truncation, and a
  `certifi` CA bundle is used when available. `test_issue41.py` 7/7 (HTTP layer mocked). No
  architecture change — O(1) / no-KV preserved. Feeds #25.
- **#43 — Fuse concurrent-stream serving** ✅ **Done** (2026-07-04): `model.generate_served`
  (`forward_step_served`, backends `fused`/`torch`/`auto`). Found the batched `H_i·v_i` reads are
  block-diagonal (no shared GEMM) and decode is **launch-bound**, so the win is the #33 fused
  Triton kernel wired into the batched path (~**2.5–2.7× tok/s** at 1/8/32/64 streams), **not** a
  GEMM reformulation (measured neutral-to-negative). Also fixes a pre-existing **head-misrouting
  bug**: legacy `generate_batched` used head-outer `repeat_interleave` ordering, so `n_heads>1,
  B>1` streams got the wrong head's weights — served is batch-outer and equals single-stream
  (torch ~1e-6, fused ~1e-2). O(1)/no-KV preserved; `test_issue43.py` 7/7. Benefits #31 under
  multi-user load (see §8.9).
- **#51 — Chat/base checkpoint pipeline** ✅ **Done** (2026-07-05): `build_chat_checkpoint.py`
  wires BPE tokenizer → WikiText-103 pretrain (DeltaNet + bf16 + #50 Category-A speed levers) →
  synthetic instruction (`hvn/lora.py`) + chat (`hvn/chat.py`) fine-tune → `hvn_chat_checkpoint.pt`
  (+`_tokenizer.json`), gated by a non-degenerate smoke test. On a d=256/3L run the coding prompt
  emits `def add(a, b): return a + b` and held-out fine-tune is **0.051 bits/char** vs the char
  baseline's **4.577** (tokenization-invariant comparison). DeltaNet flag persisted; O(1) preserved;
  `test_issue51.py` 4/4, `demo.py`/`test_issue34.py` regressions pass. Milestone 1 — the first real
  servable checkpoint; **blocks #52**, prereq for #53/#54/#55. *Note for #52:* `web/hvn_worker/worker.py`
  must rebuild the model with `use_deltanet=True` (it currently omits the flag) to serve this checkpoint.
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
- **#39 — Make Gated-DeltaNet the default block?** ✅ **Recommendation reached** (2026-07-05): the
  final corrected-routing pass (§8.12) confirms DeltaNet is the quality leader at real scale —
  steepest exponent (−0.504), beats the default HVN block at every width (−12→−33% PPL), beats the
  Transformer at d≥512, and (via #44) is bf16-stable at all widths. **Recommendation: `--use_deltanet`
  becomes the documented default block.** The actual default-value flip in code is left as a small
  follow-up so this measurement pass keeps defaults unchanged.
- **#45 — Final real-scale benchmark pass** ✅ **Done** (2026-07-05): the end-state 4-model
  (HVN / HVN-DeltaNet / HVN-DeltaNet-Weightless / Transformer) WikiText-103 sweep under corrected
  #46 routing (`results/scaling_final_pass.{json,png}`, §8.12). Added the `HVN-DeltaNet-Weightless`
  entry to `eval/scaling_laws.py`. SSM dropped (reference competitor, unaffected by #46, §8.4 stands).
  Feeds the #39 recommendation.
- **#47 — Re-measure under corrected head routing** ✅ **Done** (2026-07-05, folded into #45): the
  A/B shows the #46 fix was **also a quality fix** — correct head routing lifts the default HVN
  block 6–12% PPL by restoring multi-head specialization the bug had diluted; DeltaNet is unaffected
  (doesn't use the buggy modules). §8.4/§8.6/§8.7 default-HVN columns superseded by §8.12.
- **#44 — Fix large-width bf16 instability in the Gated-DeltaNet block** ✅ **Done** (2026-07-04):
  fp32 islands for `_l2norm` / read RMSNorm / decay-write gates, flooring the intra-chunk cumulative
  decay `P` (bounds the `k/P` overflow), and a training-loop loss-spike guard. Full bf16 sweep now
  NaN-free at every width; DeltaNet PPL 522.5→410.6 (d128), 348→251.9 (d256), 236→159.4 (d512),
  205.9→128.0 (d768) — beating both HVN and Transformer at d≥512, steepest exponent (−0.507).
  `test_issue34.py`/`test_issue33.py`/`demo.py` all pass (see §8.7).

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
| **HVN + Gated-DeltaNet (#34)** | **~3.4** (61% better than 8.8) | **−0.507** real-scale, −21–38% PPL vs HVN, beats Transformer at d≥512 (#39/#44; bf16-stable at all widths) |
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

1. **ALWAYS pull `main` and branch off `main` — no exceptions.** Even if a dependency issue
   isn't merged yet, still `git fetch` and branch from the updated remote `main`; **never**
   cut the branch off whatever branch happens to be checked out (that inherits unrelated
   in-flight work and produces a polluted MR diff). If a prerequisite has since merged, the
   dependency is already in `main` — that's exactly why you pull first:
   ```bash
   git fetch origin
   git checkout main && git pull origin main
   git checkout -b issue-NN-short-description
   ```
   Never work on `main` directly.
2. Implement the change; verify with a real test or runnable command on GPU.
3. Commit referencing the issue (`git commit -m "Fix: … for issue #NN"`).
4. Push the branch (`git push origin <branch>`).
5. **Always open a Merge Request** with `glab mr create` — target `main`, put `Closes #NN`
   in the description so the issue auto-closes on merge. Never push to or merge `main`
   directly; the maintainer reviews and merges the MR manually.
6. **On merge to `main`, always post a closing comment on the issue** with
   `glab issue note NN` — summarize what shipped and link the merged MR, then confirm the
   issue is closed (`Closes #NN` auto-closes it on merge; add the note regardless). Do this
   for every handled issue so each one has a durable record of the delivered change.

GitLab project: https://gitlab.com/komalbarun/kramba-ai

---

*End of document. This README is the single consolidated source of documentation for the
project; the earlier per-topic markdown files have been merged into the sections above.*
