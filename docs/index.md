# HVN Documentation Index

> **Status legend:** ✅ Done · 🔄 In Progress · ⬜ Planned

This directory splits the former single-file `README.md` ("Complete Compendium") into
topic-focused files. The original `README.md` is retained unchanged as the historical
source of truth; these files are the reorganized, linkable view.

## Table of Contents

| File | Covers (original README section) |
|------|-----------------------------------|
| [theory.md](theory.md) | §2 Theoretical Framework & Math Formalism · §3 FLOP Analysis |
| [architecture.md](architecture.md) | §4 Architecture Components & Project Structure · Hybrid token-recall design (§5) |
| [setup.md](setup.md) | §5 Quick Start — install, demo, train, generate, library usage, bg-benchmark rule |
| [usage.md](usage.md) | §5 Notable Runtime Features — DeltaNet, kernels, quant, server, chat/curriculum pipeline, coding eval |
| [training.md](training.md) | §6 Training & Usage Guide — flags, VRAM envelopes (4050/4060), precision, generation, tips |
| [state.md](state.md) | §5 EdgeHVN 83M PPL picture · speed-ceiling investigation · 50K probe (current measured state) |
| [features.md](features.md) | §7 Feature Summary (Closed Issues) |
| [benchmarks_speed.md](benchmarks_speed.md) | §8.1 Speed · §8.3 Kernels · §8.9 Concurrent decode · §8.13 Grad checkpointing |
| [benchmarks_quality.md](benchmarks_quality.md) | §8.2 Quality · §8.5 Honest assessment · §8.11 Reasoning/`[THINK]` |
| [benchmarks_scaling.md](benchmarks_scaling.md) | §8.4 Real-scale re-measure · §8.10 Batch-ordering fix · §8.12 Final corrected-routing pass |
| [benchmarks_deltanet.md](benchmarks_deltanet.md) | §8.6 DeltaNet vs default · §8.7 bf16 fix · §8.8 Weightless |
| [roadmap.md](roadmap.md) | §9 Development Roadmap (phases 1–9) |
| [execution_plan.md](execution_plan.md) | §10 Execution Plan (Issues #32 & #33) |
| [issues.md](issues.md) | §11 Issue History & Related Work |
| [ai_rules.md](ai_rules.md) | §12 Project Context, Delivery Workflow, Maintainer note, GPU requirement |

## 1. Overview

HVN is a novel, untrained language model architecture that achieves **long-range
associative recall at <20% of the computational cost of standard Attention**, scaling
at **O(1) constant time per token during inference** — without a traditional KV Cache.

### How It Works

HVN fuses two theoretical frameworks (see [theory.md](theory.md)):

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
> defaults, and evaluation methodology must all be chosen so they *still hold at scale*,
> even though present-day validation runs on a single RTX 4050 (6 GB). When two options are
> otherwise equal, pick the one that scales.

**Honest framing.** A frontier model (100B–1T+ params) requires a multi-node GPU cluster
(hundreds–thousands of 80 GB accelerators), not this laptop. The RTX 4050 work exists to
**de-risk and validate scaling behavior** — measuring exponents, stability, and O(1)
inference on research-scale models — *before* committing cluster budget. The 6 GB GPU is
the wind tunnel, not the aircraft.

**Frontier-readiness checklist (apply to every change):**

1. **Preserve O(1) inference.** No KV cache, no N-dependent memory. The architecture's
   entire value proposition — never trade it away for a training-time convenience.
2. **Batched paths must equal single-stream, provably.** The #46 head-misrouting bug
   proved a self-consistent-but-wrong batched path silently degrades quality and entangles
   learning with batch size (fatal at scale). Every batched/parallel path needs a
   `batched == single-stream` test (see [benchmarks_scaling.md](benchmarks_scaling.md)).
3. **Numerical stability at width and in bf16.** Large `d_model` is where bf16 hot-spots
   bite (#44). Check new ops for overflow/underflow at frontier widths, with fp32 islands.
4. **Scale-honest evaluation.** Never fit conclusions on saturated small data (the −0.007
   artifact). Use compute-matched, unsaturated, held-out PPL and report the **exponent**.
5. **Distributed-training readiness.** No single-GPU assumptions baked into the model or
   training loop. DDP/FSDP + tensor parallelism is the critical open dependency (see
   [roadmap.md](roadmap.md) Phase 7).
6. **Defaults that scale.** The documented default should be the variant that wins *at
   scale*, hence the #39 recommendation to make `--use_deltanet` the default block.