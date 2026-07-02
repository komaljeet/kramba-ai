"""
HVN Training Script
====================

A minimal but complete training pipeline for the Hamiltonian Vortex Network.

This script demonstrates:
  1. Building a simple character-level tokenizer
  2. Preparing a dataset from a text file
  3. Training the HVN model with next-token prediction (autoregressive LM)
  4. Saving / loading model checkpoints
  5. Generating text with the trained model

Usage:
    python train.py                          # Train on the bundled sample text
    python train.py --text my_corpus.txt      # Train on your own text file
    python train.py --epochs 50 --lr 3e-4     # Custom hyperparameters
"""

import argparse
import math
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hvn.core import HamiltonianVortexNetwork


# ---------------------------------------------------------------------------
# 1. Simple Character-Level Tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """A minimal character-level tokenizer.

    Maps each unique character in the training text to an integer ID.
    This keeps the vocabulary small so we can train quickly on CPU.
    """

    def __init__(self, text: str):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

    def encode(self, text: str) -> list[int]:
        return [self.char_to_id[ch] for ch in text if ch in self.char_to_id]

    def decode(self, ids: list[int]) -> str:
        return "".join(self.id_to_char[i] for i in ids)

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({"char_to_id": self.char_to_id, "id_to_char": {str(k): v for k, v in self.id_to_char.items()}}, f)

    @classmethod
    def load(cls, path: str):
        import json
        with open(path) as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        tok.vocab_size = len(tok.char_to_id)
        return tok


# ---------------------------------------------------------------------------
# 2. Dataset
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Autoregressive next-token-prediction dataset.

    Each sample is a chunk of `seq_len` tokens.  The input is tokens[0:seq_len-1]
    and the target is tokens[1:seq_len] (shifted by one).
    """

    def __init__(self, token_ids: list[int], seq_len: int):
        self.data = token_ids
        self.seq_len = seq_len

    def __len__(self):
        # Number of non-overlapping chunks
        return max(1, (len(self.data) - 1) // self.seq_len)

    def __getitem__(self, idx):
        start = idx * self.seq_len
        end = start + self.seq_len + 1  # +1 for the shifted target
        chunk = self.data[start:end]
        # Pad if needed (last chunk may be short)
        if len(chunk) < self.seq_len + 1:
            chunk = chunk + [0] * (self.seq_len + 1 - len(chunk))
        x = torch.tensor(chunk[:-1], dtype=torch.long)      # [seq_len]
        y = torch.tensor(chunk[1:], dtype=torch.long)       # [seq_len]
        return x, y


# ---------------------------------------------------------------------------
# 3. Training Loop
# ---------------------------------------------------------------------------

def train_model(
    model: HamiltonianVortexNetwork,
    dataloader: DataLoader,
    epochs: int,
    lr: float,
    device: torch.device,
    save_path: str = "hvn_checkpoint.pt",
    compile: bool = False,
    precision: str = "fp32",
    grad_accum_steps: int = 1,
):
    """Train the HVN model with next-token prediction (cross-entropy loss).

    Parameters
    ----------
    precision : str
        Mixed-precision mode: "fp32" (default), "fp16", or "bf16".
        fp16 uses torch.cuda.amp with a GradScaler (older GPUs).
        bf16 uses autocast without a scaler (RTX 40-series and newer).
    grad_accum_steps : int
        Number of micro-batches to accumulate gradients over before each
        optimizer step.  Effective batch size = batch_size × grad_accum_steps.
    """

    if compile:
        print("Compiling model with torch.compile (first epoch will be slower)...")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    # Cosine LR schedule for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    # --- Mixed-precision setup ---
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and precision == "fp16") else None

    if use_amp:
        print(f"Mixed-precision training enabled: {precision.upper()}")
    if grad_accum_steps > 1:
        print(f"Gradient accumulation: {grad_accum_steps} steps "
              f"(effective batch size = {dataloader.batch_size * grad_accum_steps})")

    print(f"\nTraining for {epochs} epochs | lr={lr} | device={device}")
    print("-" * 60)

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device)  # [B, S]
            batch_y = batch_y.to(device)  # [B, S]

            # Forward pass (with autocast for mixed precision)
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(batch_x)  # [B, S, V]
                loss = criterion(
                    logits.reshape(-1, model.vocab_size),   # [B*S, V]
                    batch_y.reshape(-1),                      # [B*S]
                )
                # Scale loss by accumulation steps so gradients average correctly
                loss = loss / grad_accum_steps

            # Backward pass
            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * grad_accum_steps  # restore unscaled loss for logging
            n_batches += 1

            # Optimizer step every grad_accum_steps micro-batches
            if (step + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                optimizer.zero_grad()

        # Handle any remaining accumulated gradients at epoch end
        if n_batches % grad_accum_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))  # clamp to avoid overflow

        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs}  |  Loss: {avg_loss:.4f}  |  Perplexity: {perplexity:.2f}  |  LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save checkpoint
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": model.vocab_size,
        "d_model": model.d_model,
        "num_layers": model.num_layers,
        "n_heads": model.n_heads,
        "n_threads": model.n_threads,
        "chunk_size": model.blocks[0].chunk_size if hasattr(model.blocks[0], 'chunk_size') else 64,
    }, save_path)
    print(f"\n✓ Model saved to {save_path}")


# ---------------------------------------------------------------------------
# 4. Text Generation (Inference)
# ---------------------------------------------------------------------------

@torch.no_grad()
def generate_text(
    model: HamiltonianVortexNetwork,
    tokenizer: CharTokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: int = 10,
    device: torch.device = None,
) -> str:
    """Generate text autoregressively from a prompt.

    Parameters
    ----------
    prompt : str
        The seed text to start generation from.
    max_new_tokens : int
        How many new tokens to generate.
    temperature : float
        Controls randomness. Lower = more deterministic, higher = more random.
    top_k : int
        Only sample from the top-k most likely next tokens (0 = no restriction).
    """
    if device is None:
        device = next(model.parameters()).device

    model.eval()

    # Encode the prompt
    ids = tokenizer.encode(prompt)
    if not ids:
        ids = [0]

    generated = list(ids)

    for _ in range(max_new_tokens):
        # Prepare input — use a limited context window for speed.
        # The HVN's Hamiltonian memory compresses history into a fixed-size
        # matrix, so we don't need a huge context window.  128 tokens is
        # a good balance between speed and context.
        context = generated[-128:]
        x = torch.tensor([context], dtype=torch.long, device=device)  # [1, seq_len]

        # Forward pass → logits for the LAST position
        logits = model(x)  # [1, seq_len, V]
        next_logits = logits[0, -1, :] / temperature  # [V]

        # Top-k filtering
        if top_k > 0:
            top_k = min(top_k, next_logits.size(-1))
            values, _ = torch.topk(next_logits, top_k)
            next_logits[next_logits < values[-1]] = float("-inf")

        # Sample
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()

        generated.append(next_id)

    return tokenizer.decode(generated)


# ---------------------------------------------------------------------------
# 5. Main Entry Point
# ---------------------------------------------------------------------------

SAMPLE_TEXT = """
The Hamiltonian Vortex Network is a novel architecture for sequence modeling.
It combines fluid dynamics with quantum mechanics to achieve efficient long range
recall. Each token is a vortex particle moving through a velocity field. The
Hamiltonian memory stores entangled associations in a fixed size tensor. This
allows the model to scale in constant time per token during inference. The
architecture does not use attention or a key value cache. Instead it relies on
a Biot Savart approximation and entanglement renormalization. Destructive
interference cancels noise while associative alignment reinforces correlated
concepts. The result is a model that is both fast and expressive.
"""


def main():
    parser = argparse.ArgumentParser(description="Train the Hamiltonian Vortex Network")
    parser.add_argument("--text", type=str, default=None, help="Path to training text file")
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of HVN blocks")
    parser.add_argument("--bond_dim", type=int, default=32, help="SVD bond dimension (quantum renormalization rank)")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of parallel Hamiltonian heads")
    parser.add_argument("--chunk_size", type=int, default=64, help="Chunk size for parallel processing (larger values improve throughput)")
    parser.add_argument("--save", type=str, default="hvn_checkpoint.pt", help="Checkpoint save path")
    parser.add_argument("--generate", action="store_true", help="Generate text after training")
    parser.add_argument("--prompt", type=str, default="The ", help="Prompt for generation")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (GPU recommended)")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Mixed-precision training mode (fp16/bf16 require CUDA)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps (effective batch size = batch_size × grad_accum)")
    parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"], help="Tokenizer type: char or bpe")
    parser.add_argument("--bpe_vocab", type=int, default=1000, help="BPE vocabulary size (only used with --tokenizer bpe)")
    # --- Issue #21 quality strategies ---
    parser.add_argument("--rank_k", type=int, default=1, help="Rank-k entanglement update (Strategy 1). 1=rank-1 (original), 4 or 8 recommended.")
    parser.add_argument("--nonlinear_update", action="store_true", help="Enable nonlinear gated Hamiltonian update (Strategy 4)")
    parser.add_argument("--causal_reinjection", action="store_true", help="Enable multi-chunk causal re-injection (Strategy 2)")
    parser.add_argument("--cross_head_mixing", action="store_true", help="Enable cross-head Hamiltonian mixing (Strategy 3)")
    parser.add_argument("--swiglu", action="store_true", help="Use SwiGLU FFN instead of GELU FFN (Strategy 6)")
    parser.add_argument("--per_token_eta", action="store_true", help="Use per-token entanglement rate instead of chunk-averaged (Strategy 5)")
    # --- Issue #22: Multi-thread Hamiltonian memory ---
    parser.add_argument("--n_threads", type=int, default=1, help="Number of independent Hamiltonian threads per head (Issue #22). 1=single Hamiltonian (backward-compatible), 4 recommended for better quality.")
    # --- Issue #23: Vision / multimodal ---
    parser.add_argument("--vision", action="store_true", help="Enable vision/multimodal mode (Issue #23). Trains image+text with patch embeddings.")
    parser.add_argument("--image_size", type=int, default=64, help="Image size (square, default 64 for fast demo)")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for image encoding (default 16)")
    parser.add_argument("--in_channels", type=int, default=3, help="Image channels (3=RGB, 1=grayscale)")
    parser.add_argument("--num_images", type=int, default=32, help="Number of synthetic images for demo vision training")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load training text ---
    if args.text and os.path.exists(args.text):
        with open(args.text, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded {len(text)} characters from {args.text}")
    else:
        text = SAMPLE_TEXT * 50  # repeat to have enough data
        print(f"Using bundled sample text ({len(text)} characters)")

    # --- Build tokenizer ---
    if args.tokenizer == "bpe":
        from hvn.tokenizer import BPETokenizer
        tokenizer = BPETokenizer.train(text, vocab_size=args.bpe_vocab)
        print(f"Vocabulary size: {tokenizer.vocab_size} BPE tokens")
    else:
        tokenizer = CharTokenizer(text)
        print(f"Vocabulary size: {tokenizer.vocab_size} characters")

    # --- Prepare dataset ---
    token_ids = tokenizer.encode(text)
    dataset = TextDataset(token_ids, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)
    print(f"Dataset: {len(dataset)} samples | DataLoader: {len(dataloader)} batches")

    # --- Build model ---
    model = HamiltonianVortexNetwork(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        bond_dimension=args.bond_dim,
        chunk_size=args.chunk_size,
        dropout=0.1,
        rank_k=args.rank_k,
        use_nonlinear_update=args.nonlinear_update,
        use_causal_reinjection=args.causal_reinjection,
        use_cross_head_mixing=args.cross_head_mixing,
        use_swiglu=args.swiglu,
        use_per_token_eta=args.per_token_eta,
        n_threads=args.n_threads,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # --- Train ---
    train_model(
        model, dataloader, args.epochs, args.lr, device, args.save,
        compile=args.compile, precision=args.precision, grad_accum_steps=args.grad_accum,
    )

    # --- Save tokenizer ---
    tok_path = args.save.replace(".pt", "_tokenizer.json")
    tokenizer.save(tok_path)
    print(f"✓ Tokenizer saved to {tok_path}")

    # --- Vision / multimodal training (Issue #23) ---
    if args.vision:
        print("\n" + "=" * 60)
        print("VISION / MULTIMODAL TRAINING (Issue #23)")
        print("=" * 60)
        from hvn.vision import ImageEncoder, CrossModalProjector, MultimodalHVN

        # Build vision encoder
        vision_encoder = ImageEncoder(
            image_size=args.image_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            d_model=args.d_model,
        )
        multimodal = MultimodalHVN(
            vision_encoder=vision_encoder,
            hvn=model,
            use_cross_modal_proj=True,
            dropout=0.1,
        ).to(device)

        num_patches = vision_encoder.num_patches
        print(f"Vision encoder: {args.image_size}×{args.image_size} → "
              f"{num_patches} patches of {args.d_model}D")

        # Generate synthetic image-text pairs for demo
        # Each image is a random tensor; the "caption" is a short random text
        print(f"Generating {args.num_images} synthetic image-text pairs...")
        images = torch.randn(args.num_images, args.in_channels,
                             args.image_size, args.image_size, device=device)
        captions = []
        for i in range(args.num_images):
            # Random caption from the training text
            start = torch.randint(0, max(1, len(token_ids) - args.seq_len - 1), (1,)).item()
            cap_ids = token_ids[start:start + args.seq_len]
            captions.append(cap_ids)

        # Quick multimodal forward pass demo
        optimizer = torch.optim.AdamW(multimodal.parameters(), lr=args.lr, weight_decay=0.01)
        criterion = nn.CrossEntropyLoss()

        multimodal.train()
        vision_epochs = min(args.epochs, 10)  # fewer epochs for demo
        print(f"\nTraining multimodal for {vision_epochs} epochs...")
        print("-" * 60)

        for epoch in range(1, vision_epochs + 1):
            total_loss = 0.0
            n_batches = 0
            # Mini-batch over image-text pairs
            batch_size = min(args.batch_size, args.num_images)
            for start in range(0, args.num_images, batch_size):
                end = min(start + batch_size, args.num_images)
                img_batch = images[start:end]
                cap_batch = captions[start:end]
                # Pad captions to same length
                max_len = max(len(c) for c in cap_batch)
                text_ids = torch.zeros(len(cap_batch), max_len, dtype=torch.long, device=device)
                for i, c in enumerate(cap_batch):
                    text_ids[i, :len(c)] = torch.tensor(c, dtype=torch.long, device=device)

                # Forward: image patches + text → logits
                logits = multimodal(img_batch, text_ids)  # [B, num_patches + S_text, V]
                # Target: shifted text tokens (predict next token in text portion)
                # We only compute loss on the text portion (skip image patch logits)
                text_logits = logits[:, num_patches:, :]  # [B, S_text, V]
                # Shift: predict token t+1 from token t
                pred_logits = text_logits[:, :-1, :]  # [B, S_text-1, V]
                targets = text_ids[:, 1:]  # [B, S_text-1]
                loss = criterion(pred_logits.reshape(-1, model.vocab_size), targets.reshape(-1))

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(multimodal.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            ppl = math.exp(min(avg_loss, 20))
            if epoch % 2 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{vision_epochs}  |  Loss: {avg_loss:.4f}  |  PPL: {ppl:.2f}")

        # Demo: image captioning
        print("\n--- Image Captioning Demo ---")
        test_image = torch.randn(1, args.in_channels, args.image_size,
                                  args.image_size, device=device)
        generated_ids = multimodal.generate_caption(
            test_image, max_new_tokens=50, temperature=0.8, top_k=10, device=device,
        )
        caption = tokenizer.decode(generated_ids)
        print(f"Generated caption: {caption}")
        print("=" * 60)

    # --- Generate text ---
    if args.generate:
        print("\n" + "=" * 60)
        print("GENERATED TEXT")
        print("=" * 60)
        output = generate_text(
            model, tokenizer,
            prompt=args.prompt,
            max_new_tokens=200,
            temperature=0.7,
            top_k=10,
            device=device,
        )
        print(output)
        print("=" * 60)


if __name__ == "__main__":
    main()