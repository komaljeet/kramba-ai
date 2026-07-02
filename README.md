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