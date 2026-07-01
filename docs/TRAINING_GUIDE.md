# HVN Training & Usage Guide

This guide covers everything you need to train and use the Hamiltonian Vortex Network.

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
5. **Learning rate**: Start with `3e-4`, try `1e-3` for faster (but less stable) training.
6. **Subword tokenizer**: For real projects, swap the `CharTokenizer` for a BPE tokenizer (e.g., HuggingFace `tokenizers`) for better vocabulary efficiency.