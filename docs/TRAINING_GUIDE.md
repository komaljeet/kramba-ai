# HVN Training & Usage Guide

This guide covers everything you need to train and use the Hamiltonian Vortex Network,
including a detailed comparison with standard Transformers and practical guidance for
training on consumer GPUs with limited VRAM (e.g. 6 GB).

---

## HVN vs Transformer: Architecture Comparison

HVN is **not** a Transformer. It replaces Self-Attention with a fixed-size
quantum-fluid dynamical system. Here is how the two architectures differ:

### Fundamental Differences

| Aspect | **Transformer (MHA)** | **HVN** |
|--------|----------------------|---------|
| **Core mechanism** | Self-Attention — every token attends to all others via Q·K^T | Fluid dynamics + quantum mechanics — tokens become vortex particles interacting with a fixed-size Hamiltonian memory matrix |
| **Memory of past tokens** | KV Cache — stores Key/Value for every token (grows with N) | Hamiltonian Matrix H — a fixed d×d matrix summarizing all past tokens (constant size) |
| **Per-token inference cost** | O(N·d) — must scan all N cached keys | O(d²·r) — single matrix-vector product, **constant in N** |
| **Inference memory** | O(N·d) — KV cache grows linearly | O(d²) — Hamiltonian is fixed-size, never grows |
| **Positional encoding** | RoPE / learned / sinusoidal | RoPE (rotary) in vortex injection layer |
| **Parallelism** | Fully parallel across all tokens | Sequential Hamiltonian update (chunked parallelism mitigates this) |

### How HVN Works (Plain Language)

Instead of attention's "every token looks at every other token," HVN does:

1. **Token → Vortex Particle**: Each token embedding is projected into a vorticity vector ω_t
2. **Biot-Savart Velocity**: `v_t = Γ · (H · ω_t)` — the token "feels" the entire history
   through the Hamiltonian matrix H in one matmul (replaces the O(N) attention scan)
3. **Deflection**: `r̃_t = x_t + Γ · v_t` — the token's representation is deflected by the velocity field
4. **Quantum Entanglement Update**: `H ← (1-η)·H + η·(P_t - H·P_t·H)` — the Hamiltonian absorbs
   the new token via DMRG-style renormalization (destructive interference cancels noise,
   associative alignment reinforces patterns)
5. **SVD Truncation**: H is truncated to rank-r to discard noise modes (quantum renormalization)

### Performance Comparison (Shakespeare, d_model=128, RTX 4050)

| Metric | Transformer | HVN | Verdict |
|--------|------------|-----|---------|
| **FLOPs/token @ N=100K** | 2,474M | 463M | HVN = **18.7%** of Transformer ✅ |
| **Inference memory @ N=100K** | 2.46 GB (KV cache) | 6.29 MB (Hamiltonian) | HVN = **0.26%** ✅ |
| **Parameters** | 461K | 265K | HVN uses **43% fewer** ✅ |
| **Forward pass scaling** | O(N) (Flash Attention) | **O(1) — flat from N=256 to N=4096** ✅ | HVN wins at long sequences |
| **Quality (PPL)** | **1.90** | **8.37** | ❌ Transformer is 4.4× better |
| **Training speed** | 3.4s/epoch | 4.0s/epoch (with torch.compile) | Roughly comparable |
| **Raw generation speed** | 780 tok/s (Flash Attention) | 106 tok/s | Transformer 7× faster ❌ |
| **Scaling exponent** | PPL ∝ params^-0.090 | PPL ∝ params^-0.007 | ❌ HVN barely improves with scale |

### Where HVN Wins

1. **Memory efficiency** — lowest memory at long sequences (O(1) Hamiltonian vs O(N) KV cache)
2. **Parameter efficiency** — 43% fewer params than Transformer
3. **Theoretical scaling** — true O(1) per-token, confirmed flat from N=256 to N=4096
4. **Incremental generation** — no context reprocessing needed

### Where HVN Loses

1. **Quality** — PPL 8.37 vs Transformer's 1.90 (rank-1 Hamiltonian update lacks expressiveness)
2. **Scaling** — making the model bigger barely helps (exponent -0.007 vs -0.090)
3. **Raw speed at short sequences** — Transformer's Flash Attention is 7× faster
4. **No custom kernels** — HVN uses Python loops + standard PyTorch ops

> **Key finding**: HVN's rank-1 entanglement update lacks the expressiveness to benefit
> from scale. This is the primary open research challenge.

---

## Prerequisites

### 1. GPU Setup (Recommended)

You need an NVIDIA GPU with CUDA support. Install the CUDA-enabled PyTorch:

```bash
# For CUDA 12.6 (compatible with driver 13.2+)
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Verify GPU access:
```bash
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### 2. CPU-Only (Fallback)

If you don't have a GPU, the CPU version works but is significantly slower:
```bash
pip install torch numpy
```

---

## Training on a 6 GB VRAM GPU (e.g. RTX 4050 Laptop)

HVN is **very memory-efficient** because it has no KV cache — the Hamiltonian is a
fixed d×d matrix. Even a d_model=512, 12-layer model uses only ~6 MB for the
Hamiltonian state. This makes HVN ideal for consumer GPUs with limited VRAM.

### What Fits in 6 GB VRAM

| Config | d_model | Layers | Approx. Params | VRAM Usage | Fits 6 GB? |
|--------|---------|--------|----------------|------------|------------|
| **Small (default)** | 128 | 2 | ~265K | ~14 MB | ✅ Easily |
| **Medium** | 256 | 4 | ~1M | ~50–100 MB | ✅ Easily |
| **Large** | 512 | 8 | ~5–10M | ~200–500 MB | ✅ Yes |
| **Very Large** | 512 | 12 | ~15M | ~1–2 GB | ✅ Yes |

The bulk of VRAM usage comes from:
- **Model weights**: a few MB to a few hundred MB
- **Training activations & gradients**: the main consumer
- **Optimizer states** (AdamW stores 2 extra copies of weights): 3× model size

### Recommended Commands for 6 GB VRAM

**Best balance (medium model, mixed precision, compiled):**
```bash
python train.py \
    --text my_corpus.txt \
    --d_model 256 --num_layers 4 --bond_dim 64 \
    --batch_size 16 --seq_len 128 \
    --precision bf16 --compile \
    --epochs 100 --generate --prompt "Chapter 1 "
```

**Maximum model size that fits comfortably:**
```bash
python train.py \
    --d_model 512 --num_layers 8 --bond_dim 128 \
    --batch_size 4 --seq_len 128 \
    --precision bf16 --grad_accum 8 \
    --compile --epochs 100
```

**If you hit OOM (out of memory), reduce these in order:**
1. `--batch_size 8` (or 4) — biggest single VRAM saver
2. `--seq_len 64` — shorter sequences = less activation memory
3. `--d_model 128` — smaller model dimension
4. Add `--grad_accum 4` to recover effective batch size after reducing `--batch_size`

### Memory-Saving Techniques Summary

| Technique | Flag | Memory Saved | Speed Gain |
|-----------|------|-------------|------------|
| Mixed precision (bf16) | `--precision bf16` | ~50% | 2–3× |
| Gradient accumulation | `--grad_accum N` | Use smaller batch | — |
| torch.compile | `--compile` | — | 14× per-batch |
| Smaller batch size | `--batch_size 4` | Up to 4× | — |
| Shorter sequences | `--seq_len 64` | ~2× | Slightly faster |

---

## Training

### Quick Start (Bundled Sample Text)

```bash
python train.py --epochs 30 --generate --prompt "The "
```

This trains on a built-in sample text about the HVN architecture itself, then generates text.

### Training on Your Own Text

```bash
python train.py --text my_corpus.txt --epochs 50 --generate --prompt "Once upon "
```

### All Training Options

| Flag | Default | Description |
|------|---------|-------------|
| `--text` | (sample) | Path to your training text file |
| `--epochs` | 30 | Number of training epochs |
| `--lr` | 3e-4 | Learning rate (AdamW) |
| `--batch_size` | 16 | Batch size |
| `--seq_len` | 64 | Sequence length per training sample |
| `--d_model` | 128 | Model hidden dimension |
| `--num_layers` | 2 | Number of HVN blocks |
| `--bond_dim` | 32 | SVD truncation rank (quantum renormalization) |
| `--save` | hvn_checkpoint.pt | Checkpoint save path |
| `--generate` | (flag) | Generate text after training |
| `--prompt` | "The " | Prompt for post-training generation |
| `--compile` | (flag) | Use torch.compile for faster training (GPU recommended) |
| `--precision` | fp32 | Mixed-precision mode: fp32, fp16, or bf16 (GPU required for fp16/bf16) |
| `--grad_accum` | 1 | Gradient accumulation steps (effective batch = batch_size × grad_accum) |
| `--tokenizer` | char | Tokenizer type: char or bpe |
| `--bpe_vocab` | 1000 | BPE vocabulary size (only used with --tokenizer bpe) |
| `--rank_k` | 1 | Rank-k entanglement update (Issue #21 Strategy 1). 1=rank-1 (original), 4 or 8 recommended for better quality |
| `--nonlinear_update` | (flag) | Enable nonlinear gated Hamiltonian update (Issue #21 Strategy 4) |
| `--causal_reinjection` | (flag) | Enable multi-chunk causal re-injection (Issue #21 Strategy 2) |
| `--cross_head_mixing` | (flag) | Enable cross-head Hamiltonian mixing (Issue #21 Strategy 3) |
| `--swiglu` | (flag) | Use SwiGLU FFN instead of GELU FFN (Issue #21 Strategy 6) |
| `--per_token_eta` | (flag) | Use per-token entanglement rate instead of chunk-averaged (Issue #21 Strategy 5) |
| `--n_threads` | 1 | Number of independent Hamiltonian threads per head (Issue #22). 1=single Hamiltonian (backward-compatible), 4 recommended for better quality |

### Example: Larger Model

```bash
python train.py \
    --text my_book.txt \
    --epochs 100 \
    --lr 1e-3 \
    --batch_size 32 \
    --seq_len 128 \
    --d_model 256 \
    --num_layers 4 \
    --bond_dim 64 \
    --generate --prompt "Chapter 1 "
```

### Quality Improvement Strategies (Issue #21)

HVN's quality (perplexity) can be improved by enabling one or more of the
following strategies.  All strategies preserve O(1) inference — the Hamiltonian
remains fixed-size, no KV cache is introduced, and FLOPs stay below 20% of
Transformer at long sequences.

| Strategy | Flag | Impact | Description |
|---------|------|--------|-------------|
| Rank-k entanglement | `--rank_k 4` or `--rank_k 8` | HIGH | Replaces rank-1 projector with rank-k, making each token's update k× more expressive |
| Causal re-injection | `--causal_reinjection` | MEDIUM | 2-pass chunked approach: re-computes deflected representations with updated H per chunk |
| Cross-head mixing | `--cross_head_mixing` | MEDIUM | Learned mixing of Hamiltonian states across heads (like MHA output projection) |
| Nonlinear update | `--nonlinear_update` | MEDIUM-HIGH | Gated nonlinearity in the projector (like attention's softmax on QK^T) |
| Per-token η | `--per_token_eta` | MEDIUM | Per-token entanglement rate instead of chunk-averaged (like Mamba's selective scan) |
| SwiGLU FFN | `--swiglu` | LOW-MEDIUM | SwiGLU gated FFN (used by LLaMA) instead of standard GELU FFN |

**Recommended combination** (all strategies enabled):

```bash
python train.py \
    --epochs 100 --precision bf16 \
    --rank_k 8 --nonlinear_update --causal_reinjection \
    --cross_head_mixing --swiglu --per_token_eta \
    --generate --prompt "The "
```

**Strategy 1 (rank-k) is the single most impactful change** — it addresses the
root cause of HVN's flat scaling exponent by making the Hamiltonian update
more expressive per token.

### Multi-Thread Hamiltonian Memory (Issue #22)

Instead of 1 Hamiltonian per head, maintain **K independent Hamiltonians per
head**, each updated with a different projection of the input.  Different
threads learn to track different types of relationships (syntax, semantics,
repetition, etc.).

| Flag | Default | Description |
|------|---------|-------------|
| `--n_threads` | 1 | Number of independent Hamiltonian threads per head (K). 1=single (backward-compatible), 4 recommended |

**Key insight**: One d×d matrix can only store one view of the sequence. K
matrices can store K different views — like having K people each take notes
on the same conversation from different angles.

**Memory cost**: O(K·d²) per head — still O(1) in sequence length N.  With
K=4 and d=512: 4 × 6.29 MB = 25.2 MB — still 100× less than the KV cache.

**Recommended combination** (all strategies + multi-thread):

```bash
python train.py \
    --epochs 100 --precision bf16 \
    --n_threads 4 --rank_k 8 --nonlinear_update \
    --causal_reinjection --cross_head_mixing --swiglu --per_token_eta \
    --generate --prompt "The "
```

### What Happens During Training

1. **Tokenizer**: A character-level tokenizer is built from your text (each unique character → integer ID)
2. **Dataset**: Text is split into non-overlapping chunks of `seq_len` tokens
3. **Training**: Next-token prediction with cross-entropy loss, AdamW optimizer, cosine LR schedule
4. **Checkpoint**: Model weights + hyperparameters saved to `hvn_checkpoint.pt`
5. **Tokenizer**: Saved to `hvn_checkpoint_tokenizer.json`

### Training Output Example

```
Device: cuda
Vocabulary size: 37 characters
Dataset: 536 samples | DataLoader: 34 batches
Model parameters: 208,683

Training for 15 epochs | lr=0.0003 | device=cuda
------------------------------------------------------------
Epoch   1/15  |  Loss: 3.1836  |  Perplexity: 24.13
Epoch   5/15  |  Loss: 2.2536  |  Perplexity: 9.52
Epoch  10/15  |  Loss: 2.2087  |  Perplexity: 9.10
Epoch  15/15  |  Loss: 2.2027  |  Perplexity: 9.05

✓ Model saved to hvn_checkpoint.pt
✓ Tokenizer saved to hvn_checkpoint_tokenizer.json
```

- **Loss**: Cross-entropy loss (lower is better)
- **Perplexity**: exp(loss) — effective vocabulary size the model is choosing from (lower is better)

### Mixed-Precision Training (fp16/bf16)

Mixed-precision training uses 16-bit floats for forward/backward passes, giving
**2-3× speedup** and **half the memory** on modern NVIDIA GPUs.  The RTX 4050
supports both fp16 and bf16.

```bash
# BF16 (recommended for RTX 40-series — no GradScaler needed, more stable)
python train.py --epochs 50 --precision bf16

# FP16 (older GPUs — uses GradScaler for numerical stability)
python train.py --epochs 50 --precision fp16

# Combine with gradient accumulation for larger effective batch sizes
python train.py --epochs 50 --precision bf16 --grad_accum 4 --batch_size 8
# → effective batch size = 8 × 4 = 32
```

| Precision | Speed | Memory | Stability | Notes |
|-----------|-------|--------|-----------|-------|
| fp32 | 1× | 1× | Best | Default, maximum precision |
| fp16 | 2-3× | 0.5× | Good | Uses GradScaler, may overflow on very small values |
| bf16 | 2-3× | 0.5× | Best | Same dynamic range as fp32, no scaler needed (RTX 40-series+) |

**Gradient accumulation** (`--grad_accum N`) splits a large batch into N
micro-batches, accumulating gradients before each optimizer step.  This allows
training with effective batch sizes larger than GPU memory would normally allow.

---

## Generation (Inference)

### Using the Generate Script

```bash
# Basic generation
python generate.py --prompt "The "

# Custom parameters
python generate.py --prompt "Hamiltonian " --tokens 300 --temp 0.5 --top_k 5

# Load a specific checkpoint
python generate.py --checkpoint hvn_checkpoint.pt --prompt "The vortex "
```

### Generation Options

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | hvn_checkpoint.pt | Path to trained model |
| `--prompt` | "The " | Seed text for generation |
| `--tokens` | 200 | Max new tokens to generate |
| `--temp` | 0.8 | Temperature (lower = more deterministic) |
| `--top_k` | 10 | Only sample from top-k next tokens (0 = disabled) |

### Temperature Guide

| Temperature | Effect |
|-------------|--------|
| 0.1 - 0.3 | Very deterministic, repetitive |
| 0.5 - 0.7 | Focused, coherent |
| 0.8 - 1.0 | Balanced creativity |
| 1.0+ | Wild, random |

---

## Using the Model in Python

### Load a Trained Model

```python
import torch
from hvn.core import HamiltonianVortexNetwork
from train import CharTokenizer, generate_text

# Load tokenizer
tokenizer = CharTokenizer.load("hvn_checkpoint_tokenizer.json")

# Load model
ckpt = torch.load("hvn_checkpoint.pt", weights_only=False)
model = HamiltonianVortexNetwork(
    vocab_size=ckpt["vocab_size"],
    d_model=ckpt["d_model"],
    num_layers=ckpt["num_layers"],
)
model.load_state_dict(ckpt["model_state_dict"])
model.eval()

# Generate text
text = generate_text(
    model, tokenizer,
    prompt="The ",
    max_new_tokens=200,
    temperature=0.7,
    top_k=10,
)
print(text)
```

### Build a New Model from Scratch

```python
import torch
from hvn.core import HamiltonianVortexNetwork

model = HamiltonianVortexNetwork(
    vocab_size=1000,
    d_model=256,
    num_layers=4,
    bond_dimension=64,   # SVD truncation rank
    dropout=0.1,
)

# Forward pass
input_ids = torch.randint(0, 1000, (4, 128))  # [batch, seq_len]
logits = model(input_ids)                       # [batch, seq_len, vocab_size]
```

### Custom Training Loop

```python
import torch
import torch.nn as nn
from hvn.core import HamiltonianVortexNetwork

model = HamiltonianVortexNetwork(vocab_size=100, d_model=128, num_layers=2, bond_dimension=32)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
criterion = nn.CrossEntropyLoss()

for epoch in range(50):
    for batch_x, batch_y in dataloader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        logits = model(batch_x)
        loss = criterion(logits.reshape(-1, vocab_size), batch_y.reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    print(f"Epoch {epoch}: loss={loss.item():.4f}")
```

---

## Tips for Better Results

1. **More data**: The bundled sample is tiny (~34K chars). Use a real corpus (books, articles, code) for meaningful results.
2. **More epochs**: Character-level models need 50-100+ epochs on small data.
3. **Larger model**: Increase `--d_model 256`, `--num_layers 4`, `--bond_dim 64` for more capacity.
4. **Longer sequences**: Increase `--seq_len 128` or `256` so the model sees more context per sample.
5. **Positional awareness**: The vortex injection layer now uses rotary positional embeddings, which improves sequence-order sensitivity compared with the earlier position-agnostic version.
6. **Learning rate**: Start with `3e-4`, try `1e-3` for faster (but less stable) training.
7. **Subword tokenizer**: For real projects, swap the `CharTokenizer` for a BPE tokenizer (e.g., HuggingFace `tokenizers`) for better vocabulary efficiency.
8. **Mixed precision**: Always use `--precision bf16` on RTX 40-series GPUs — halves memory and gives 2-3× speedup with no stability loss.
9. **torch.compile**: Add `--compile` for a 14× per-batch speedup (requires `triton-windows` on Windows).
10. **Gradient accumulation**: If VRAM is tight, use `--batch_size 8 --grad_accum 4` to get an effective batch of 32 without OOM.

---

## Honest Assessment & Known Limitations

HVN is a **research architecture**, not a production-ready Transformer replacement.
Understanding its limitations helps set realistic expectations:

### The Quality Gap

On the Shakespeare benchmark (d_model=128, 20 epochs):

| Model | Eval PPL | Train Time |
|-------|----------|------------|
| HVN (multi-head) | 8.37 | 467.8s |
| Transformer | 1.90 | 3.4s |
| SSM | 1.12 | 68.6s |

The gap is significant. The main reasons:
1. **Rank-1 entanglement update** — less expressive than attention's full QK^T or SSM's gated recurrence
2. **Chunked approximation** — tokens within a chunk share the same H state (small causal approximation)
3. **Poor scaling** — HVN PPL barely improves with size: 14.24 → 13.82 across 56× param increase
4. **Small training data** — 34K characters; all models would improve with more data

### The Scaling Problem

Scaling laws evaluation (d_model={64,128,256,512}, 20 epochs):

| Metric | HVN | Transformer |
|--------|-----|------------|
| Scaling exponent | PPL ∝ params^-0.007 | PPL ∝ params^-0.090 |
| PPL at smallest | 14.24 | 9.12 |
| PPL at largest | 13.82 | 6.91 |
| Improvement | ~3% | ~24% |

HVN's perplexity barely improves when the model is made larger. This is the
fundamental open research challenge — the rank-1 Hamiltonian update does not
have enough expressiveness to benefit from additional parameters.

### What Would Make HVN Competitive

1. **Custom Triton/CUDA kernel** for the Hamiltonian update (eliminate Python loop overhead)
2. **Parallel scan** formulation of the Hamiltonian update (O(log S) depth instead of O(S))
3. **Higher-rank entanglement updates** (beyond rank-1 projectors)
4. **Real-dataset evaluation** at scale (WikiText-103, code corpora)
5. **Flash-style tiled Hamiltonian** computation (avoid materializing full D×D in HBM)

See `PLANNED.md` for the full roadmap of planned optimizations.