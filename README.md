# Hamiltonian Vortex Network (HVN)

A novel, untrained language model architecture that achieves **long-range associative
recall at <20% of the computational cost of standard Attention**, scaling at **O(1)
constant time per token during inference** — without a traditional KV Cache.

## How It Works

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

## Project Structure

```
kramba-ai/
├── hvn/
│   ├── __init__.py
│   └── core/
│       ├── __init__.py
│       ├── vortex_injection.py    # VortexInjectionLayer: Biot-Savart velocity field
│       ├── hamiltonian_memory.py  # HamiltonianMemoryBlock: MPS entanglement update
│       ├── hvn_block.py           # HVNBlock: fluid + quantum wrapper
│       ├── decoder.py             # HVNDecoder: shallow logit projection
│       └── model.py               # HamiltonianVortexNetwork: full model
├── docs/
│   ├── THEORETICAL_FRAMEWORK.md   # Mathematical formalism
│   ├── FLOP_ANALYSIS.md           # Computational cost comparison vs MHA
│   └── TRAINING_GUIDE.md          # How to train & use the model
├── demo.py                        # Runnable demo: forward pass + shape verification
├── train.py                       # Training script (GPU-supported)
├── generate.py                    # Text generation from trained checkpoint
├── requirements.txt
└── README.md
```

## Quick Start

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

📖 **Full training & usage guide**: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)

## Usage

```python
import torch
from hvn.core import HamiltonianVortexNetwork

model = HamiltonianVortexNetwork(
    vocab_size=1000,
    d_model=128,
    num_layers=2,
    bond_dimension=32,   # SVD truncation rank (quantum renormalization)
)

input_ids = torch.randint(0, 1000, (4, 64))   # [batch, seq_len]
logits = model(input_ids)                       # [batch, seq_len, vocab_size]
```

## Key Results (N = 100,000 tokens, d = 512, L = 12 layers)

| Metric | Standard MHA (KV cache) | HVN (SVD r=128) | Ratio |
|--------|------------------------|------------------|-------|
| FLOPs/token | 2,474M | 463M | **18.7%** ✓ |
| Inference memory | 2.46 GB | 6.29 MB | **0.26%** |
| Scaling in N | O(Nd) | O(d²r) = O(1) | — |

See [docs/FLOP_ANALYSIS.md](docs/FLOP_ANALYSIS.md) for the full derivation.

## Architecture Components

| Component | File | Role |
|-----------|------|------|
| `VortexInjectionLayer` | `vortex_injection.py` | Token → vortex particle → velocity field via Biot-Savart |
| `HamiltonianMemoryBlock` | `hamiltonian_memory.py` | Fixed-size MPS Hamiltonian with entanglement renormalization |
| `HVNBlock` | `hvn_block.py` | Causal wrapper: sequential vortex injection + H update |
| `HVNDecoder` | `decoder.py` | Shallow projection: hidden state → vocabulary logits |
| `HamiltonianVortexNetwork` | `model.py` | Full model: embedding + N×HVNBlock + decoder |

## Hamiltonian Quantization (Inference)

HVN supports INT8 and 4-bit NF4 quantization for the Hamiltonian matrices —
the main memory consumers during inference.  This reduces the Hamiltonian
state memory by **4× (INT8)** or **8× (NF4)** with minimal quality loss.

```python
from hvn.quantize import QuantizedHVNInference, estimate_memory_savings

# Estimate memory savings
savings = estimate_memory_savings(model, batch_size=1, bits=4)
print(f"Compression: {savings['compression']:.1f}×, "
      f"Savings: {savings['savings_pct']:.1f}%")

# Quantized inference (drop-in replacement for model.generate_incremental)
q_infer = QuantizedHVNInference(model, bits=4)  # or bits=8 for INT8
output = q_infer.generate_incremental(
    prompt_ids=[1, 5, 10], max_new_tokens=200, temperature=0.8, top_k=10,
)
```

| Quantization | Memory Reduction | Relative Error | Use Case |
|-------------|-----------------|----------------|----------|
| INT8 | ~4× (75% saved) | <1% | General inference |
| NF4 4-bit | ~8× (87% saved) | <10% | Memory-constrained inference |

## Custom Kernels & Parallel Processing (Issue #33)

HVN supports optional CUDA/Triton kernels for faster generation and parallel
Hamiltonian processing for faster training. All are opt-in via CLI flags,
backward compatible, and preserve O(1) inference.

```bash
# Faster generation: CUDA Graph capture (2.8× speedup)
python generate.py --prompt "The " --cuda_graph

# Faster generation: fused Triton kernel (2.1-7.2× speedup)
python generate.py --prompt "The " --fused_kernel

# Faster training: parallel scan Hamiltonian update (1.6× speedup, zero logit diff)
python train.py --epochs 100 --precision bf16 --use_parallel_scan --generate --prompt "The "
```

| Technique | Flag | Speedup | Applies to |
|-----------|------|---------|------------|
| CUDA Graph capture | `--cuda_graph` | 2.82× (78 → 220 tok/s) | Generation |
| Fused Triton kernel | `--fused_kernel` | 2.14-7.2× (78 → 167-340 tok/s) | Generation |
| Parallel scan (associative Hamiltonian update) | `--use_parallel_scan` | 1.59× (7.63 → 4.79 ms/fwd) | Training |
| Multi-chunk barrier sync | (enabled internally via `parallel_forward()`) | 1.14× (7.74 → 6.77 ms/fwd) | Training |

The fused Triton kernel is dtype-agnostic (fp16/fp32/int8/ternary), designed
to support the extreme weight compression techniques planned in Issue #32.

📖 **Full results**: [docs/BENCHMARK_RESULTS.md](docs/BENCHMARK_RESULTS.md) § 5

## OpenAI-Compatible API Server (Issue #31)

HVN can be served over a standard OpenAI-compatible REST API — `/v1/chat/completions`,
`/v1/completions`, `/v1/models`, `/v1/health` — so it works with the OpenAI Python SDK,
curl, LangChain, OpenWebUI, LibreChat, etc. Ollama/llama.cpp is not an option (GGUF only
supports Transformer-family architectures), so this is a lightweight FastAPI server built
directly on HVN's existing `forward_step` / Hamiltonian-state-cache inference — no KV
cache, O(1) per-token cost is preserved end to end.

```bash
# Install server dependencies
pip install fastapi uvicorn

# Start the server
python -m hvn.server --checkpoint hvn_checkpoint.pt --port 8000

# With a LoRA adapter loaded at startup
python -m hvn.server --checkpoint hvn_checkpoint.pt --lora hvn_checkpoint_lora.pt --port 8000

# Use with the OpenAI Python SDK
from openai import OpenAI
client = OpenAI(base_url="http://localhost:8000/v1", api_key="hvn")
response = client.chat.completions.create(
    model="hvn",
    messages=[{"role": "user", "content": "Hello!"}],
    stream=True,  # SSE token streaming also supported
)

# Or with curl
curl http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{"model": "hvn", "messages": [{"role": "user", "content": "Hello!"}]}'
```

Tool calling (Issue #24) is exposed via the standard `tools`/`functions` request field —
the server injects tool schemas into the prompt and returns any `[TOOL_CALL]` blocks the
model emits as OpenAI-format `tool_calls` (execution is left to the client, matching the
OpenAI API's own contract).

**Scope note**: a single model is loaded per server process — `/v1/models` lists that one
model, no dynamic multi-checkpoint hot-swapping. Verified with `test_server.py` (6/6
checks: health, models, completions, chat completions, SSE streaming, tool calling).

## Gated-DeltaNet Dynamics — the real PPL lever (Issue #34)

HVN's long-standing perplexity gap (≈8.1 vs SSM ≈1.1) was **not** a state-capacity
problem — the SSM reaches PPL ≈1.1 with a *smaller* fixed state, and every
capacity-oriented strategy (#21, #22) moved PPL <3%. The culprit was the **readout/update
dynamics**: the velocity L2-norm + Frobenius clamp (a NaN band-aid that discards all
magnitude) and the non-standard symmetric `H·P·H` update.

`--use_deltanet` swaps each block for a self-contained **Gated-DeltaNet** block
(`hvn/core/deltanet.py`) that borrows proven linear-attention fast-weight dynamics:

| Lever | What it does |
|-------|--------------|
| Per-channel data-dependent decay gate `α_t=σ(W_α x)` | Replaces the Frobenius clamp — stable over 1500+ tokens with no magnitude clamping |
| Per-head output RMSNorm normalizer | Makes the read a proper scale-normalized readout instead of an unnormalized matmul |
| Gated DeltaNet update `S_t = diag(α_t)S_{t-1} + β_t(v_t−S_{t-1}k_t)k_tᵀ` | Separate learned **k/v** projections + separate query read `o_t=S_t q_t` (drops the ill-conditioned `HPH`) |
| Output gate `SiLU(W_g x)` + short causal depthwise conv | Cheap output-side conditioning (Mamba / RWKV-7 style) |

```bash
# Train with the DeltaNet dynamics fix (backward compatible; default is the original HVN path)
python train.py --epochs 20 --use_deltanet --generate --prompt "The "
```

**Result** (tiny-Shakespeare, d_model=128, 2 layers, controlled A/B, same seed/data):
eval PPL **8.81 → 3.42 — 61% better** (target ≥30%), reaching PPL <4 by epoch 3.
**O(1) inference is fully preserved** (fixed `d_head×d_head` state, no KV cache — flat
2.6 ms/token from N=256→4096). Verified by `test_issue34.py` (5/5 on GPU). The delta-rule
training forward is a sequential scan (~9× slower/epoch — *training only*); the
chunked-parallel delta rule is a drop-in follow-up speedup.

## Quality Improvement Strategies (Issue #21)

HVN's perplexity can be improved by enabling one or more quality strategies.
All strategies preserve O(1) inference — the Hamiltonian remains fixed-size,
no KV cache is introduced, and FLOPs stay below 20% of Transformer at long
sequences.

| Strategy | Flag | Impact | Description |
|---------|------|--------|-------------|
| Rank-k entanglement | `--rank_k 4` or `--rank_k 8` | HIGH | Rank-k projector instead of rank-1 — k× more expressive per token |
| Causal re-injection | `--causal_reinjection` | MEDIUM | 2-pass chunked: re-compute deflections with updated H per chunk |
| Cross-head mixing | `--cross_head_mixing` | MEDIUM | Learned mixing of Hamiltonian states across heads |
| Nonlinear update | `--nonlinear_update` | MEDIUM-HIGH | Gated nonlinearity in the projector (like attention's softmax) |
| Per-token η | `--per_token_eta` | MEDIUM | Per-token entanglement rate (like Mamba's selective scan) |
| SwiGLU FFN | `--swiglu` | LOW-MEDIUM | SwiGLU gated FFN (used by LLaMA) instead of standard GELU FFN |
| Multi-thread memory | `--n_threads 4` | MEDIUM-HIGH | K independent Hamiltonians per head — K different views of the sequence (Issue #22) |

```bash
# Train with all quality strategies + multi-thread memory enabled
python train.py --epochs 100 --precision bf16 \
    --n_threads 4 --rank_k 8 --nonlinear_update --causal_reinjection \
    --cross_head_mixing --swiglu --per_token_eta \
    --generate --prompt "The "
```

📖 **Full training guide**: [docs/TRAINING_GUIDE.md](docs/TRAINING_GUIDE.md)

## License

Research / educational use.