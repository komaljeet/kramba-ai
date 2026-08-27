# kramba-ai — training linear-attention LMs on one 6 GB GPU

> **On the name.** The `hvn/` package, `--arch hvn-hybrid`, and this repository are named
> after the **Hamiltonian Vortex Network**, the original block this project started from.
> That block was replaced on 2026-07-04 and, retested properly on 2026-08-17, **lost**. The
> name is kept because it is wired through the code; it is not a description of the model in
> use, and it is not what this project is about. See
> [What this is, concretely](#what-this-is-concretely).

> ## 🌟 The Dream — frontier-class LLMs on edge devices
>
> **Run frontier-grade language-model capability on a 6 GB laptop GPU — eventually a phone.**
> The bet has two halves that this repo is built around:
> - **Fit the weights in edge VRAM** → **ternary (BitNet b1.58)** quantization shrinks a
>   10B-class model to ~2 GB, a 20B-class model to ~4 GB.
> - **Keep inference VRAM flat over long, agentic-length context** → a **recurrence +
>   windowed-attention hybrid** (no KV cache that grows with the conversation).
>
> **The path is native, from-scratch training of a ternary-hybrid model — not distillation of
> an existing one.** BitNet b1.58 shows ternary trained from scratch matches fp16 at scale, so
> the recipe is real, not a compromise. Distillation of an existing strong model into this
> format remains an optional accelerator for later, not the primary road.
>
> **This is a bet on the training recipe, not on a novel architecture.** The block in use is
> published work (see below). Whatever advantage exists here is in getting the recipe to run
> on hardware this small.
>
> **⚠️ Reality check (2026-08-02) — read before planning around the sequencing below.**
> Training cost is `~6 · params · tokens`, and Chinchilla scales tokens with params, so it grows
> as **N²**. Decomposing this project's own measured throughput (142M at 9,102 tok/s vs 6B at
> 116 tok/s), the 3,300× gap is 42× more params × 42× more tokens × **only ~2× offload
> inefficiency** (that ~2× is now stale and the real figure is *smaller* — see
> [`docs/training.md`](docs/training.md); it makes this conclusion stronger, not weaker).
> The extensive memory engineering (#99–#118) genuinely got a 6B model to run
> forward+backward on a 6 GB card at ~1 GiB peak — and that is **not the same as being able to
> train it**: 6B at Chinchilla is ~33 years on this hardware, ~30× out of reach even with every
> known lever stacked. **The honest local ceiling is ~142M** (3.6 days). Everything above that
> needs rented compute, and no architecture — this one or any other — changes that. See
> [`docs/training.md`](docs/training.md) § "The compute wall". The *inference* half of the dream
> below is intact and measured; the *training-it-yourself* half is not achievable on this
> hardware.
>
> **Sequencing (this is a staged bet, not a next-hour target):**
> 1. **Done, and it changed the subject — the sandbox phase.** The recurrence + windowed-attention
>    hybrid works, and beats a matched vanilla transformer at equal compute. But the block doing
>    the work turned out to be a reimplementation of **published** gated DeltaNet (see the
>    correction below), so that is a reproduction, not a finding. What survives as this project's
>    own contribution is the **training stack** — the thing that puts a 1B model on a 6 GB card.
>    The research claim was retired on 2026-08-17; the engineering was not.
> 2. **Next — ~20B-class native ternary-hybrid (~4 GB).** Already good enough for real agentic
>    coding on a single edge GPU. This is the first genuinely useful milestone.
> 3. **Later — push toward frontier scale.** Harder, and gated on real compute — a separate,
>    funded phase after (2) is proven, not a precondition for shipping something useful.

kramba-ai is a **training and evaluation stack for linear-attention language models on a
single 6 GB consumer GPU** — measured, not projected: a 1.07B-parameter model trains in
4.00 GB, and ~1.6B is the measured ceiling on the card.

> **Correction (2026-08-16).** This line previously read "HVN is a novel, untrained
> language-model architecture". True when written on 2026-07-02; false from 2026-07-04,
> when #34 replaced the original block with a reimplementation of published gated-DeltaNet
> dynamics. The architecture in use is **not original**, and the original block, retested
> properly on 2026-08-17 at matched parameters and matched tokens, **lost**. See
> [docs/index.md](docs/index.md) §1 for the full correction and for what IS measured and
> defensible.

> **📚 The full documentation now lives in [`docs/`](docs/index.md).** This README is a short
> landing page; the former single-file "Complete Compendium" has been split into topic-focused
> files. **Start at [`docs/index.md`](docs/index.md)**, then
> [`docs/handoff.md`](docs/handoff.md) if you are picking the project up cold.

> **Maintainer note — keep the docs current.** Whenever a session reaches a conclusion worth
> recording (a benchmark result, a confirmed/dead-end optimization, a parity or speed gate
> outcome, an architecture decision, or a closed issue's verdict), update the relevant file in
> [`docs/`](docs/index.md) **before ending the session**. Treat those files as the durable log of
> what was tried and concluded — don't leave a settled result only in a commit message or a
> session transcript. Full rules: [`docs/ai_rules.md`](docs/ai_rules.md).

## What this is, concretely

The default model (`--arch hvn-hybrid`) is **10 gated-DeltaNet blocks with 2 sliding-window
attention layers**. `hvn/core/deltanet.py` reimplements published work and says so in its own
docstring: a canonical **gated DeltaNet** update, a **GLA / Mamba-2** per-channel decay gate, and
an output gate plus short causal conv from **Mamba / RWKV-7**. That places it in the same family
as other published hybrids. Constant-memory inference — O(1) state, no KV cache — is the defining
property of that whole family, not a differentiator of this work; no comparison against Mamba or
RWKV has ever been run.

The original block (`HVNBlock`) *is* original. It framed tokens as Lagrangian vortex particles
read out against a fixed-size "entanglement Hamiltonian" — the fluid-dynamics /
matrix-product-state synthesis described in [`docs/theory.md`](docs/theory.md). It is still in the
tree (`--arch hvn-original-hybrid`) and it is kept as history: retested on 2026-08-17 at matched
body params, same seed, same corpus, it lost to gated DeltaNet by 8–20× the noise floor at both 4
and 10 tokens/parameter. It was novel in **framing**, not in **class**.

What is left — and what the work is actually about — is the stack that makes these models
trainable on consumer hardware: block-coordinate descent, CPU weight offload, ternary
quantisation, tied embeddings, chunked cross-entropy, a hand-written Triton scan and megakernel,
and the fixes that make them work together.

### What is measured, and what is not

| claim | status |
|---|---|
| 1.07B-parameter model **trains in 4.00 GB** on a 6 GB card, 875 tok/s | measured 2026-08-14/17 |
| **~1.6B is the size ceiling** on the card (2.45B spills to host RAM) | measured 2026-08-17 |
| the stack is **not architecture-specific** — a plain 1.07B transformer also trains in 4.04 GB | measured 2026-08-17 |
| flat ms/token from N = 256 → 4096 (O(1) in time) | measured, [benchmarks_speed.md](docs/benchmarks_speed.md) |
| beats a **matched vanilla transformer** at equal compute, 12M body (+0.016 nats) | measured 2026-08-16, single seed |
| **~30 tokens/param**, not Chinchilla's 20 — the size trend is unresolved | measured 2026-08-16 |
| chunked cross-entropy saves **1436 MB at 48M**, nothing at 1B | measured 2026-08-17 |
| **This architecture is not cheaper to train** — a transformer is 1.41× faster per step | measured; the compute win is at inference only |
| 6.29 MB vs 2.46 GB of context memory at 100K tokens | **CALCULATED for the retired block; never measured, and not carried forward** |

### The "18.7% of MHA FLOPs" table is gone, deliberately

Earlier versions of this README led with an analytic table — 463M vs 2,474M FLOPs/token,
6.29 MB vs 2.46 GB of memory at N = 100,000. Two problems, and both are disqualifying for a
headline number:

1. **It was never measured.** Those are hand-derived counts, not a benchmark.
2. **It describes the wrong model.** The `SVD r=128` rank truncation it assumes lives in
   `hvn/core/hamiltonian_memory.py` — the **original** block, the one that lost. The block
   actually in use, `hvn/core/deltanet.py`, contains no SVD at all.

The analysis is kept in [`docs/theory.md`](docs/theory.md), attached to the block it actually
describes. What *is* measured about inference cost — flat ms/token from N = 256 to 4096 — is in
the table above and in [`benchmarks_speed.md`](docs/benchmarks_speed.md).

## Quick start

```bash
# Install dependencies (CUDA GPU version)
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy

# Verify shapes & forward pass, then train + generate
python demos/demo.py
python train.py --epochs 50 --generate --prompt "The "
python generate.py --prompt "The " --tokens 200 --temp 0.7
```

Full install / usage / training details: [`docs/setup.md`](docs/setup.md),
[`docs/usage.md`](docs/usage.md), [`docs/training.md`](docs/training.md).

**Two traps that have cost hours:** `torch.compile` is **2.4× slower** on this architecture (four
independent losses now — use `--fast`, which skips it), and bare `--no-fast` silently reverts to
fp32. Both are documented in [`docs/training.md`](docs/training.md).

## Documentation map

| File | Covers |
|------|--------|
| [index.md](docs/index.md) | Overview · what is and isn't original · north-star checklist · doc map |
| [handoff.md](docs/handoff.md) | **Read first when picking the project up cold** — what is running, what is settled |
| [state.md](docs/state.md) | Current measured state — every result, dated, including the retractions |
| [theory.md](docs/theory.md) | Theoretical framework & math formalism · FLOP analysis vs MHA (original block) |
| [architecture.md](docs/architecture.md) | Components & project structure · hybrid token-recall design |
| [setup.md](docs/setup.md) | Install, demo, train, generate, library usage |
| [usage.md](docs/usage.md) | Runtime features — DeltaNet, kernels, quantization, server, pipelines |
| [training.md](docs/training.md) | Training flags · RTX 4050/4060 VRAM envelopes · precision · the compute wall |
| [vscode.md](docs/vscode.md) | Using EdgeHVN from VS Code — local OpenAI-compatible server, Continue, RAG |
| [features.md](docs/features.md) | Feature summary (closed issues) |
| [benchmarks_speed.md](docs/benchmarks_speed.md) | Speed · kernels · concurrent decode · grad checkpointing |
| [benchmarks_quality.md](docs/benchmarks_quality.md) | Quality · honest assessment · reasoning/`[THINK]` |
| [benchmarks_scaling.md](docs/benchmarks_scaling.md) | Real-scale scaling · batch-ordering fix · final routing pass |
| [benchmarks_deltanet.md](docs/benchmarks_deltanet.md) | DeltaNet vs default · bf16 stability fix · weightless |
| [roadmap.md](docs/roadmap.md) | Development roadmap (phases 1–9) |
| [DISTRIBUTED_LAUNCH.md](docs/DISTRIBUTED_LAUNCH.md) · [TENSOR_PARALLEL_DESIGN.md](docs/TENSOR_PARALLEL_DESIGN.md) | Multi-GPU launch · tensor-parallel design notes |
| [execution_plan.md](docs/execution_plan.md) · [ISSUE_30_PLAN.md](docs/ISSUE_30_PLAN.md) · [FRONTIER_TRAINING_PLAN.md](docs/FRONTIER_TRAINING_PLAN.md) | Historical plans |
| [issues.md](docs/issues.md) | Issue history & related work |
| [ai_rules.md](docs/ai_rules.md) | Project context, delivery workflow, GPU requirement |

## Dead ends, recorded on purpose

Negative results live in [`docs/state.md`](docs/state.md) with the same weight as the wins,
including results this project later **retracted** after finding the measurement itself was
wrong. The largest ones:

- **Looped / weight-shared models lose** (#122, closed 2026-08-21). Every variant was tried:
  looping a pretrained stack, loop-aware fine-tuning, a gated loop, fixed-N graded on held-out
  loss, and finally a small stack trained from scratch to be run 4×. Three layers × 4 loses to
  twelve layers by 0.101 nats at matched steps (11× the noise floor), and still loses on a
  matched wall clock.
- **`torch.compile` is a net loss here** — 2.4× slower in the real loop, measured four times.
- **MoE FFN loses to simply scaling the dense model** (#123).
- **Weight-delta extrapolation is measurably worse than doing nothing** (GHOST, #126).
- **Selective token training (Rho-1) gives no gain at 83M**, and the "phase transition" it was
  built on does not exist.

## GPU requirement

All training, benchmarking, and evaluation **must run on GPU, never CPU** (RTX 4050, CUDA 12.6).
Verify `torch.cuda.is_available()` before running. See [`docs/ai_rules.md`](docs/ai_rules.md).

GitLab project: https://gitlab.com/komalbarun/kramba-ai

## Something to try on your own rig

One question this project never answered, deliberately left for whoever runs this on their
own hardware: **is the effective batch size right, or just what fit in VRAM?**

`hardware_scan()` picks it to fit memory, and it became a constant nobody revisited. A
critical batch size exists and is well studied, but where it sits depends on your model and
your data, so it cannot be looked up — and the default optimiser here (Muon) is much less
characterised for large batches than Adam is.

Varying `--grad-accum` costs **no extra VRAM**, so a gaming GPU can answer it: three arms at
8 / 16 / 32, graded at matched *tokens* rather than matched steps, scored with
`eval/heldout_loss.py`. Full design, including the fixed-LR confound to state in any
writeup, is in [issue #152](https://gitlab.com/komalbarun/kramba-ai/-/issues/152).

If you measure it, I would like to know the number.

## License

[`LICENSE`](LICENSE) — **Apache License 2.0**. Open source, OSI-approved. Use it, modify
it, redistribute it, build a product on it, sell it. Two conditions only: keep the
attribution and the [`NOTICE`](NOTICE) file, and state what you changed. The licence also
carries an explicit patent grant from every contributor.

> **Changed 2026-08-21.** This repository was previously under the *Rambarun Komaljeet
> License v2.0*, a source-available licence that forbade commercial use. It has been
> relicensed to Apache-2.0 by the sole copyright holder. Commercial use no longer needs
> a separate licence and no longer needs to ask.

Reproduction and criticism were always explicitly permitted for anyone, and still are,
because a result nobody may check is worth nothing.

The licence covers this implementation, not the published algorithms it implements —
gated DeltaNet, GLA/Mamba-2 gating, Mamba/RWKV-7 conv and output gate, BitNet b1.58,
BAdam, ALBERT weight sharing, Chinchilla, and Bavarian et al.'s fill-in-the-middle are
other people's work, attributed in the files that implement them and listed in
[`NOTICE`](NOTICE).

## Read the summary page

A single measured-numbers write-up of the whole result — what fits, what it costs, and
what failed — is published at **<https://ai-research.rambarun.com>** and lives in
[`site/index.html`](site/index.html).
