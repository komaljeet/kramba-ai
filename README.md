# Hamiltonian Vortex Network (HVN)

HVN is a novel, untrained language-model architecture that achieves **long-range
associative recall at <20% of the computational cost of standard Attention**, scaling at
**O(1) constant time per token during inference** — without a traditional KV cache.

> **📚 The full documentation now lives in [`docs/`](docs/index.md).** This README is a short
> landing page; the former single-file "Complete Compendium" has been split into topic-focused
> files. **Start at [`docs/index.md`](docs/index.md).**

> **Maintainer note — keep the docs current.** Whenever a session reaches a conclusion worth
> recording (a benchmark result, a confirmed/dead-end optimization, a parity or speed gate
> outcome, an architecture decision, or a closed issue's verdict), update the relevant file in
> [`docs/`](docs/index.md) **before ending the session**. Treat those files as the durable log of
> what was tried and concluded — don't leave a settled result only in a commit message or a
> session transcript. Full rules: [`docs/ai_rules.md`](docs/ai_rules.md).

## How it works

HVN fuses two theoretical frameworks (see [`docs/theory.md`](docs/theory.md)):

1. **Lagrangian Vortex Fluid Dynamics** — each token is mapped to a vortex particle whose
   velocity is computed via a *modified Biot-Savart law* against a fixed-size velocity field
   (the Hamiltonian), replacing the O(N) attention scan with an O(d²) matrix-vector product.
2. **Quantum Matrix Product State Entanglement** — a fixed-size Hamiltonian matrix evolves via
   *quantum entanglement renormalization*, absorbing each new vortex vector through a
   deterministic update rule (no gradient descent at inference). SVD truncation discards noise
   modes, keeping only the most entangled associations.

The result: **O(d²) per-token inference cost, constant in sequence length N.**

### Key results (N = 100,000 tokens, d = 512, L = 12 layers)

| Metric | Standard MHA (KV cache) | HVN (SVD r=128) | Ratio |
|--------|------------------------|------------------|-------|
| FLOPs/token | 2,474M | 463M | **18.7%** ✓ |
| Inference memory | 2.46 GB | 6.29 MB | **0.26%** |
| Scaling in N | O(Nd) | O(d²r) = O(1) | — |

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

## Documentation map

| File | Covers |
|------|--------|
| [index.md](docs/index.md) | Overview · North-star / frontier-readiness checklist · doc map |
| [theory.md](docs/theory.md) | Theoretical framework & math formalism · FLOP analysis vs MHA |
| [architecture.md](docs/architecture.md) | Components & project structure · hybrid token-recall design |
| [setup.md](docs/setup.md) | Install, demo, train, generate, library usage |
| [usage.md](docs/usage.md) | Runtime features — DeltaNet, kernels, quantization, server, pipelines |
| [training.md](docs/training.md) | Training flags · RTX 4050/4060 VRAM envelopes · precision · tips |
| [state.md](docs/state.md) | Current measured state — EdgeHVN 83M PPL, speed-ceiling, 50K probe |
| [features.md](docs/features.md) | Feature summary (closed issues) |
| [benchmarks_speed.md](docs/benchmarks_speed.md) | Speed · kernels · concurrent decode · grad checkpointing |
| [benchmarks_quality.md](docs/benchmarks_quality.md) | Quality · honest assessment · reasoning/`[THINK]` |
| [benchmarks_scaling.md](docs/benchmarks_scaling.md) | Real-scale scaling · batch-ordering fix · final routing pass |
| [benchmarks_deltanet.md](docs/benchmarks_deltanet.md) | DeltaNet vs default · bf16 stability fix · weightless |
| [roadmap.md](docs/roadmap.md) | Development roadmap (phases 1–9) |
| [execution_plan.md](docs/execution_plan.md) | Execution plan (Issues #32 & #33) |
| [issues.md](docs/issues.md) | Issue history & related work |
| [ai_rules.md](docs/ai_rules.md) | Project context, delivery workflow, GPU requirement |

## GPU requirement

All training, benchmarking, and evaluation **must run on GPU, never CPU** (RTX 4050, CUDA 12.6).
Verify `torch.cuda.is_available()` before running. See [`docs/ai_rules.md`](docs/ai_rules.md).

GitLab project: https://gitlab.com/komalbarun/kramba-ai
