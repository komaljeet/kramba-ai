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
| [vscode.md](vscode.md) | Using EdgeHVN from VS Code — OpenAI-compatible local server, Continue setup, RAG |

> **Start here if you are planning a training run:** [`training.md`](training.md) §
> "The compute wall — why *fits in VRAM* was never the binding constraint" (2026-08-02).
> Training cost grows as **N²** (`6·params·tokens`, Chinchilla scaling tokens with params),
> and this project's memory work — impressive as it is — addresses only ~2× of the gap.
> The local ceiling on the 6 GB card is **~142M** (3.6 days to Chinchilla); 6B is
> out of reach by ~30×. Size decisions should start from `6·N·D ÷ throughput`, never from
> whether the model fits.

## 1. Overview

> **Corrected 2026-08-16.** This section previously opened with "HVN is a novel,
> untrained language model architecture". That was written on 2026-07-02 and was true
> of the ORIGINAL block. It stopped being true on 2026-07-04, when
> `Fix PPL gap via Gated-DeltaNet readout/update dynamics (closes #34)` replaced that
> block, and it was never revised. Every session since read "novel" off this line and
> repeated it for six weeks. The measurements underneath were honest; this headline
> was not.

**What this project actually is today:** a training and evaluation stack that runs
modern linear-attention models on a single 6 GB consumer GPU, plus the measurements
that back that up.

**The architecture in use is not original.** `hvn/core/deltanet.py` is a
reimplementation of published work and says so in its own docstring — a canonical
**gated DeltaNet** update, a **GLA / Mamba-2** style per-channel decay gate, and an
output gate plus short causal conv from **Mamba / RWKV-7**. The default model
(`--arch hvn-hybrid`) is 10 such blocks with 2 sliding-window attention layers, which
places it in the same family as published hybrids.

**The original block (`HVNBlock`) is original, and it lost.** It framed tokens as
Lagrangian vortex particles read out against a fixed-size "entanglement Hamiltonian"
(see [theory.md](theory.md)). Measured at d=768 it reached PPL 194.3 against a
transformer's 150.8, and #34 replaced it rather than repairing it. `deltanet.py`'s own
analysis is blunter: *"stripped of the fluid/quantum framing, HVN is a linear-attention
fast-weight net"* whose update rule was "ill-conditioned". So the original was novel in
**framing**, not in **class**.

**Constant-memory inference is not a differentiator.** O(1) state and no KV cache are
the defining property of the whole linear-attention / SSM family — Mamba, RWKV, RetNet,
DeltaNet. Comparisons against a vanilla transformer (see state.md) show that property
working; they do not distinguish this work from that family, and no comparison against
Mamba or RWKV has ever been run.

### What IS measured and defensible

| claim | status |
|---|---|
| 1.07B-parameter model **trains in 4.05 GB** on a 6 GB card, 875 tok/s | measured 2026-08-14 |
| flat ms/token from N=256→4096 (O(1) in time) | measured, benchmarks_speed.md |
| beats a **matched vanilla transformer** at equal compute, 12M body | measured 2026-08-16, single seed |
| ~30 tokens/param, not Chinchilla's 20 | measured 2026-08-16, size trend unresolved |
| 6.29 MB vs 2.46 GB of context memory at 100K tokens | **CALCULATED, never measured** |

The stack — block-coordinate descent, CPU weight offload, ternary quantisation, tied
embeddings, and the fixes making them work together — is assembled from published
components. The working, measured integration on consumer hardware is the contribution.

### Open question worth naming

`HVNBlock` was benchmarked **once**, early, at a 5M-token budget, before the data-recipe
fix (#164), the LR-resume fix (#165), tied embeddings, and the tokens-per-parameter
work. Nobody retested it after the stack matured. Whether it would still lose is
**unknown**.

### 1.1 North Star: Frontier-Scale Training

> **This project is being built toward training a frontier-scale model on the HVN
> architecture.** That goal is the tie-breaker for every design decision — features, fixes,
> defaults, and evaluation methodology must all be chosen so they *still hold at scale*,
> even though present-day validation runs on a single RTX 4050 (6 GB). When two options are
> otherwise equal, pick the one that scales.

**Framing.** A frontier model (100B–1T+ params) requires a multi-node GPU cluster
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
6. **`torch.compile` is a net loss on this architecture — do not assume it helps.** Measured
   2.4x SLOWER in the real training loop (20.72 vs 8.66 s/step, GPU 35% vs 98%), with the fused
   kernels on OR off; whole-model compile also MemoryErrors at 16 layers because Inductor's
   `compute_ancestors` is O(N²) in graph nodes. Four independent losses now (#129, #112, #158).
   The speed here comes from the hand-written Triton scan + megakernel, not from Inductor.
7. **Defaults that scale.** The documented default should be the variant that wins *at
   scale*, hence the #39 recommendation to make `--use_deltanet` the default block.