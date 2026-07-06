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
import sys
from contextlib import nullcontext

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hvn.core import HamiltonianVortexNetwork
from hvn import distributed as hvn_dist


# ---------------------------------------------------------------------------
# 1. Simple Character-Level Tokenizer
# ---------------------------------------------------------------------------

class CharTokenizer:
    """A minimal character-level tokenizer.

    Maps each unique character in the training text to an integer ID.
    This keeps the vocabulary small so we can train quickly on CPU.

    Special tool-calling tokens (Issue #24) are added automatically
    if ``add_tool_tokens=True`` (default).  These tokens are multi-character
    strings that are treated as single tokens.
    """

    # Special tool-calling tokens (Issue #24)
    TOOL_SPECIAL_TOKENS = [
        "[TOOL_CALL]", "[/TOOL_CALL]",
        "[TOOL_RESULT]", "[/TOOL_RESULT]",
        "[THINK]", "[/THINK]",
    ]

    # Special chat tokens (Issue #28)
    CHAT_SPECIAL_TOKENS = [
        "[SYSTEM]", "[/SYSTEM]",
        "[USER]", "[/USER]",
        "[ASSISTANT]", "[/ASSISTANT]",
    ]

    def __init__(self, text: str, add_tool_tokens: bool = True):
        chars = sorted(set(text))
        self.char_to_id = {ch: i for i, ch in enumerate(chars)}
        self.id_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab_size = len(chars)

        # Add special tool tokens (Issue #24) + chat tokens (Issue #28)
        self.special_tokens = {}
        if add_tool_tokens:
            all_special = self.TOOL_SPECIAL_TOKENS + self.CHAT_SPECIAL_TOKENS
            for token in all_special:
                # Assign a new ID for each special token
                new_id = self.vocab_size
                self.char_to_id[token] = new_id
                self.id_to_char[new_id] = token
                self.special_tokens[token] = new_id
                self.vocab_size += 1

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs.

        Handles multi-character special tokens (e.g. [TOOL_CALL]) by
        scanning for them before falling back to single characters.
        """
        ids = []
        i = 0
        while i < len(text):
            # Check for special tokens (longest match first)
            matched = False
            if self.special_tokens:
                for token in sorted(self.special_tokens.keys(), key=len, reverse=True):
                    if text[i:i+len(token)] == token:
                        ids.append(self.char_to_id[token])
                        i += len(token)
                        matched = True
                        break
            if matched:
                continue
            # Single character
            ch = text[i]
            if ch in self.char_to_id:
                ids.append(self.char_to_id[ch])
            i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        """Decode a list of token IDs back to text."""
        return "".join(self.id_to_char.get(i, "") for i in ids)

    def save(self, path: str):
        import json
        with open(path, "w") as f:
            json.dump({
                "char_to_id": self.char_to_id,
                "id_to_char": {str(k): v for k, v in self.id_to_char.items()},
                "special_tokens": self.special_tokens,
            }, f)

    @classmethod
    def load(cls, path: str):
        import json
        with open(path) as f:
            data = json.load(f)
        tok = cls.__new__(cls)
        tok.char_to_id = data["char_to_id"]
        tok.id_to_char = {int(k): v for k, v in data["id_to_char"].items()}
        tok.vocab_size = len(tok.char_to_id)
        tok.special_tokens = data.get("special_tokens", {})
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
    use_lora: bool = False,
    fused_optimizer: bool = False,
    defer_loss_sync: bool = False,
    non_blocking: bool = False,
    distributed: bool = False,
    fsdp: bool = False,
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
    use_lora : bool
        If True, only train LoRA parameters (the base model is frozen).
        The optimizer will only receive LoRA parameters.
    fused_optimizer : bool
        Use the fused CUDA AdamW kernel (Issue #50, Category-A exact speed lever).
        Model-bit-identical to the unfused kernel; only valid on CUDA.
    defer_loss_sync : bool
        Accumulate the epoch loss on-device and copy to host once per epoch
        instead of a per-micro-batch ``loss.item()`` host↔device sync (Issue #50,
        the training-loop analogue of the #36 generation fix).  Purely a logging
        optimisation — gradients and the trained model are unaffected.
    non_blocking : bool
        Issue async (``non_blocking=True``) host→device transfers of the input
        batches.  Only effective with a pinned-memory DataLoader (Issue #50).
    distributed : bool
        Data-parallel training (Issue #65).  Requires an initialised process
        group (``hvn.distributed.init_distributed`` under ``torchrun``).  The
        model is wrapped in DDP (or FSDP with ``fsdp=True``); each rank must be
        fed a disjoint shard via ``DistributedSampler``.  Gradient averaging
        across equal-size rank micro-batches is provably equal to the
        single-process batch (``tests/test_issue65.py``).  Checkpoints, loss
        logging, and prints are rank-0-only; the loss all-reduce is per-epoch
        (extends the #50 per-epoch sync).  The saved checkpoint layout is
        identical to a single-GPU run.
    fsdp : bool
        Shard params/grads/optimizer state with FSDP instead of replicating
        (DDP) — for models that outgrow one card.  Checkpoint is consolidated
        to the single-file layout on rank 0.
    """

    if compile:
        print("Compiling model with torch.compile (first epoch will be slower)...")
        model = torch.compile(model, mode="reduce-overhead")

    # Move params to the device BEFORE building the optimizer — the fused AdamW
    # kernel requires its params to already live on CUDA at construction time.
    model.to(device)
    model.train()

    # --- Data-parallel wrapping (Issue #65) ---
    is_main = hvn_dist.is_main_process() if distributed else True
    if distributed:
        model = hvn_dist.wrap_model(model, device, fsdp=fsdp)
        if is_main:
            mode = "FSDP (sharded)" if fsdp else "DDP (replicated)"
            print(f"Distributed data-parallel training: {mode}, "
                  f"world_size={hvn_dist.get_world_size()} (Issue #65)")
    # The unwrapped module: config attrs + single-GPU checkpoint layout.
    unwrapped = hvn_dist.unwrap_model(model)

    # --- Optimizer: use only trainable (LoRA) params if use_lora ---
    # The fused kernel is CUDA-only; silently fall back on CPU.
    use_fused = fused_optimizer and device.type == "cuda"
    if use_lora:
        trainable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01,
                                      fused=use_fused)
        print(f"Optimizer: training only {len(trainable_params)} LoRA parameter tensors")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                      fused=use_fused)
    if use_fused:
        print("Optimizer: fused AdamW CUDA kernel enabled (Issue #50)")
    # Cosine LR schedule for smooth convergence
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    def clip_gradients():
        # FSDP shards gradients, so the norm must be computed collectively via
        # its own method; DDP/plain models use the standard utility.
        if fsdp and hasattr(model, "clip_grad_norm_"):
            model.clip_grad_norm_(max_norm=1.0)
        else:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

    # --- Mixed-precision setup ---
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and precision == "fp16") else None

    if use_amp:
        print(f"Mixed-precision training enabled: {precision.upper()}")
    if grad_accum_steps > 1:
        print(f"Gradient accumulation: {grad_accum_steps} steps "
              f"(effective batch size = {dataloader.batch_size * grad_accum_steps})")

    if is_main:
        print(f"\nTraining for {epochs} epochs | lr={lr} | device={device}")
        print("-" * 60)

    n_loader_batches = len(dataloader)
    for epoch in range(1, epochs + 1):
        # DistributedSampler reshuffles per epoch only if told the epoch number;
        # without this every epoch would reuse the epoch-0 shard order.
        if distributed and hasattr(getattr(dataloader, "sampler", None), "set_epoch"):
            dataloader.sampler.set_epoch(epoch)

        # Accumulate the running loss on-device when defer_loss_sync is set so we
        # avoid a per-micro-batch host↔device sync; copy to host once at epoch end.
        total_loss = torch.zeros((), device=device) if defer_loss_sync else 0.0
        n_batches = 0
        optimizer.zero_grad()

        for step, (batch_x, batch_y) in enumerate(dataloader):
            batch_x = batch_x.to(device, non_blocking=non_blocking)  # [B, S]
            batch_y = batch_y.to(device, non_blocking=non_blocking)  # [B, S]

            # DDP syncs gradients on every backward by default; inside a
            # gradient-accumulation window we defer the all-reduce to the
            # boundary micro-batch (one collective per optimizer step, and
            # exact equivalence with the single-process schedule).
            sync_now = ((step + 1) % grad_accum_steps == 0
                        or step == n_loader_batches - 1)
            sync_ctx = (hvn_dist.grad_sync_context(model, sync_now)
                        if distributed else nullcontext())

            with sync_ctx:
                # Forward pass (with autocast for mixed precision)
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    logits = model(batch_x)  # [B, S, V]
                    loss = criterion(
                        logits.reshape(-1, unwrapped.vocab_size),   # [B*S, V]
                        batch_y.reshape(-1),                          # [B*S]
                    )
                    # Scale loss by accumulation steps so gradients average correctly
                    loss = loss / grad_accum_steps

                # Backward pass
                if scaler is not None:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # restore unscaled loss for logging
            if defer_loss_sync:
                total_loss += loss.detach() * grad_accum_steps  # on-device, no sync
            else:
                total_loss += loss.item() * grad_accum_steps
            n_batches += 1

            # Optimizer step every grad_accum_steps micro-batches
            if (step + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    clip_gradients()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    clip_gradients()
                    optimizer.step()
                optimizer.zero_grad()

        # Handle any remaining accumulated gradients at epoch end
        if n_batches % grad_accum_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                clip_gradients()
                scaler.step(optimizer)
                scaler.update()
            else:
                clip_gradients()
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        if defer_loss_sync:
            if distributed:
                hvn_dist.all_reduce_mean_(total_loss)  # one collective per epoch
            total_loss = total_loss.item()  # single host sync per epoch
        elif distributed:
            # Per-epoch metric all-reduce (Issue #65) — extends the #50
            # per-epoch loss sync; never a per-step collective.
            t = torch.tensor(total_loss, device=device)
            total_loss = hvn_dist.all_reduce_mean_(t).item()
        avg_loss = total_loss / max(n_batches, 1)
        perplexity = math.exp(min(avg_loss, 20))  # clamp to avoid overflow

        if is_main and (epoch % 5 == 0 or epoch == 1):
            print(f"Epoch {epoch:3d}/{epochs}  |  Loss: {avg_loss:.4f}  |  Perplexity: {perplexity:.2f}  |  LR: {scheduler.get_last_lr()[0]:.6f}")

    # Save checkpoint — rank-0 only under distribution, but the (collective)
    # FSDP consolidation must run on every rank. Layout is identical to a
    # single-GPU checkpoint so serving (#51/#52 contract) is unchanged.
    state_dict = (hvn_dist.consolidated_state_dict(model) if distributed
                  else model.state_dict())
    if is_main:
        torch.save({
            "model_state_dict": state_dict,
            "vocab_size": unwrapped.vocab_size,
            "d_model": unwrapped.d_model,
            "num_layers": unwrapped.num_layers,
            "n_heads": unwrapped.n_heads,
            "n_threads": unwrapped.n_threads,
            "chunk_size": unwrapped.blocks[0].chunk_size if hasattr(unwrapped.blocks[0], 'chunk_size') else 64,
            "use_deltanet": getattr(unwrapped, "use_deltanet", False),
            "deltanet_conv_kernel": getattr(unwrapped.blocks[0], "conv_kernel", 4),
            "weightless": getattr(unwrapped, "weightless", False),
            "weightless_seed": getattr(unwrapped, "weightless_seed", 1234),
        }, save_path)
        print(f"\n✓ Model saved to {save_path}")
    if distributed:
        hvn_dist.barrier()  # everyone waits until the checkpoint exists


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


# ---------------------------------------------------------------------------
# Issue #50: `--full` meta-flag — one-switch recommended fast-training stack
# ---------------------------------------------------------------------------

def apply_full_preset(args, explicit_flags):
    """Expand the ``--full`` meta-flag into the recommended fast-training stack.

    ``--full`` is a single switch that turns on the complete recommended
    training stack so you don't have to remember and combine half a dozen flags.
    It is split into two honestly-labelled categories:

      * **Category A — exact speed levers (model bit-identical):** fused AdamW
        kernel, per-epoch (not per-step) loss sync, and pinned/async data
        loading.  These change *speed only*, never the trained model.
      * **Category B — recommended defaults that change the model/precision:**
        ``--precision bf16`` (bf16-stable at every width since #44) and
        ``--use_deltanet`` (the #39-recommended higher-quality block).  These
        are *not* quality-neutral — ``--full`` trains the best/fastest
        *recommended* model, not the same model faster.

    Any individual flag the user passed explicitly overrides the preset
    (``explicit_flags`` is the set of ``--flag`` strings seen on the command
    line).  Returns a list of human-readable strings describing exactly what was
    enabled, for the reproducibility banner.

    Note ``--use_parallel_scan`` is deliberately *not* enabled: it targets the
    default HVN block and is a no-op under ``--use_deltanet`` (which ``--full``
    turns on), so enabling it would be misleading.
    """
    enabled = []

    # --- Category B: model/precision-changing recommended defaults ---
    if "--precision" in explicit_flags:
        enabled.append(f"precision={args.precision} (kept — user override)")
    else:
        args.precision = "bf16"
        enabled.append("precision=bf16  [B]")
    args.use_deltanet = True
    enabled.append("use_deltanet  [B]")

    # --- Category A: exact speed levers (model bit-identical) ---
    args.compile = True
    enabled.append("compile  [A]")
    args.grad_checkpoint = True
    enabled.append("grad_checkpoint (#49)  [A]")
    args.fused_adamw = True
    enabled.append("fused AdamW kernel  [A]")
    args.defer_loss_sync = True
    enabled.append("per-epoch loss sync (no per-step host sync)  [A]")
    args.pin_memory = True
    enabled.append("pinned-memory + async H2D transfers  [A]")

    return enabled


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
    parser.add_argument("--grad_checkpoint", action="store_true", help="Gradient (activation) checkpointing (Issue #49): recompute block activations in the backward pass instead of storing them. Mathematically exact (bit-identical gradients/model), ~20-30% slower per step, but frees activation memory so a larger --d_model / --batch_size fits in the same VRAM. Training-only; inference unchanged.")
    parser.add_argument("--full", action="store_true", help="Issue #50 meta-flag: enable the complete recommended fast-training stack in one switch. Category A (exact, model-bit-identical speed levers): fused AdamW, per-epoch loss sync, pinned/async data loading, --compile, --grad_checkpoint. Category B (recommended defaults that CHANGE the model/precision): --precision bf16 + --use_deltanet. So --full trains the best/fastest RECOMMENDED model, not the same model faster. Any flag passed explicitly overrides the preset (e.g. --full --precision fp32).")
    # --- Issue #65: distributed data-parallel training ---
    parser.add_argument("--distributed", action="store_true", help="Data-parallel training (Issue #65). Launch with torchrun --nproc_per_node=K train.py --distributed ... — each rank trains a disjoint data shard (DistributedSampler) with gradient averaging provably equal to the single-process batch (tests/test_issue65.py). NCCL on GPU, gloo on CPU. Default single-GPU path is unchanged without this flag.")
    parser.add_argument("--fsdp", action="store_true", help="With --distributed: shard params/grads/optimizer state via FSDP instead of replicating (DDP) — for models that outgrow one card. Checkpoint is consolidated to the standard single-file layout on rank 0.")
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
    parser.add_argument("--use_parallel_scan", action="store_true", help="Use parallel scan for Hamiltonian update (Issue #33 Fix 3). Eliminates sequential loop for training.")
    # --- Issue #34: Gated-DeltaNet readout/update dynamics ---
    parser.add_argument("--use_deltanet", action="store_true", help="Use Gated-DeltaNet blocks (Issue #34): per-channel decay gate + readout normalizer + separate k/v DeltaNet update + output gate/conv. Fixes the PPL gap. Preserves O(1) inference. Overrides HVN-specific strategy flags.")
    parser.add_argument("--deltanet_conv_kernel", type=int, default=4, help="Kernel size of the short causal depthwise conv in DeltaNet blocks (Issue #34). 0/1 disables it.")
    # --- Issue #40: hybrid weightless projections ---
    parser.add_argument("--weightless", action="store_true", help="Issue #40 (requires --use_deltanet): freeze the q/k/v/out DeltaNet projections as seed-regenerated random features (stored as one integer, not d² floats each) while keeping the learned data-dependent gates/conv. Shrinks the checkpoint; A/B the PPL cost. Preserves O(1) inference.")
    parser.add_argument("--weightless_seed", type=int, default=1234, help="Base PRNG seed for the frozen weightless projections (Issue #40).")
    # --- Issue #23: Vision / multimodal ---
    parser.add_argument("--vision", action="store_true", help="Enable vision/multimodal mode (Issue #23). Trains image+text with patch embeddings.")
    parser.add_argument("--image_size", type=int, default=64, help="Image size (square, default 64 for fast demo)")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size for image encoding (default 16)")
    parser.add_argument("--in_channels", type=int, default=3, help="Image channels (3=RGB, 1=grayscale)")
    parser.add_argument("--num_images", type=int, default=32, help="Number of synthetic images for demo vision training")
    # --- Issue #27: Multimodal fusion ---
    parser.add_argument("--image_text", action="store_true",
                        help="Enable multimodal fusion training (Issue #27). Joint image-text training with gated fusion, modality embeddings, and contrastive retrieval.")
    parser.add_argument("--retrieval_weight", type=float, default=0.5,
                        help="Weight for contrastive retrieval loss in multimodal training (0=LM only, 1=retrieval only)")
    parser.add_argument("--use_gated_fusion", action="store_true", default=True,
                        help="Use gated cross-modal fusion layer (Issue #27)")
    parser.add_argument("--use_modality_embed", action="store_true", default=True,
                        help="Add learned modality type embeddings [IMAGE]/[TEXT] (Issue #27)")
    parser.add_argument("--use_retrieval_head", action="store_true", default=True,
                        help="Add cross-modal retrieval head for contrastive learning (Issue #27)")
    # --- Issue #24: Tool calling ---
    parser.add_argument("--tool_training", action="store_true",
                        help="Enable tool-calling training data (Issue #24). Adds synthetic tool-call examples to the training data.")
    parser.add_argument("--tool_examples", type=int, default=50,
                        help="Number of tool-calling training examples to generate (only used with --tool_training)")
    # --- Issue #42: Reasoning / chain-of-thought training ---
    parser.add_argument("--reasoning_training", action="store_true",
                        help="Enable reasoning training data (Issue #42). Adds synthetic multi-step arithmetic examples with [THINK] chain-of-thought traces.")
    parser.add_argument("--reasoning_examples", type=int, default=200,
                        help="Number of reasoning training examples to generate (only used with --reasoning_training)")
    # --- Issue #29: LoRA / QLoRA / Instruction tuning ---
    parser.add_argument("--lora", action="store_true",
                        help="Enable LoRA fine-tuning (Issue #29). Freezes base model, trains low-rank adapters.")
    parser.add_argument("--qlora", action="store_true",
                        help="Enable QLoRA fine-tuning (Issue #29). Quantizes base model to NF4 + LoRA adapters.")
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank (r). Higher = more expressive but more parameters. Default 8.")
    parser.add_argument("--lora_alpha", type=float, default=16.0,
                        help="LoRA scaling factor. Effective LR scaling = alpha / rank. Default 16.")
    parser.add_argument("--lora_dropout", type=float, default=0.0,
                        help="LoRA dropout rate. Default 0.")
    parser.add_argument("--instruction", action="store_true",
                        help="Enable instruction tuning (Issue #29). Adds instruction-formatted training data.")
    parser.add_argument("--instruction_examples", type=int, default=50,
                        help="Number of instruction-tuning examples to generate (only used with --instruction).")
    parser.add_argument("--merge_lora", action="store_true",
                        help="Merge LoRA weights back into base model after training (for zero-overhead inference).")
    args = parser.parse_args()

    # Internal (non-CLI) Category-A loop primitives, off unless --full turns them on.
    args.fused_adamw = False
    args.defer_loss_sync = False
    args.pin_memory = False

    # --- Issue #50: expand the --full meta-flag (explicit flags win) ---
    if args.full:
        explicit_flags = {tok.split("=")[0] for tok in sys.argv[1:] if tok.startswith("--")}
        enabled = apply_full_preset(args, explicit_flags)
        print("=" * 60)
        print("--full (Issue #50): recommended fast-training stack enabled")
        print("  [A]=exact speed lever (model bit-identical)  [B]=changes model/precision")
        for item in enabled:
            print(f"    • {item}")
        print("=" * 60)

    # --- Issue #65: initialise the process group before touching the device ---
    if args.distributed or hvn_dist.distributed_env_present():
        args.distributed = True
        rank, world_size, local_rank = hvn_dist.init_distributed()
        device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    is_main = hvn_dist.is_main_process()
    if is_main:
        print(f"Device: {device}")

    # --- Load training text ---
    if args.text and os.path.exists(args.text):
        with open(args.text, "r", encoding="utf-8") as f:
            text = f.read()
        print(f"Loaded {len(text)} characters from {args.text}")
    else:
        text = SAMPLE_TEXT * 50  # repeat to have enough data
        print(f"Using bundled sample text ({len(text)} characters)")

    # --- Generate auxiliary training data (tool / reasoning / instruction) ---
    # Generated BEFORE the tokenizer is built so their characters (e.g. digits,
    # arithmetic operators) are included in the char vocabulary — otherwise they
    # are OOV and silently dropped at encode time (which previously stripped every
    # digit out of the reasoning/tool examples).
    aux_segments = []  # list of (label, text) appended to the training stream

    if args.tool_training:
        from hvn.tools import generate_tool_training_data
        print(f"\nGenerating {args.tool_examples} tool-calling training examples...")
        tool_examples = generate_tool_training_data(num_examples=args.tool_examples)
        tool_text = "".join(ex["prompt"] + ex["response"] for ex in tool_examples)
        aux_segments.append(("tool-calling", tool_text))

    if args.reasoning_training:
        from hvn.reasoning import generate_reasoning_training_data
        print(f"\nGenerating {args.reasoning_examples} reasoning (chain-of-thought) training examples...")
        reasoning_examples = generate_reasoning_training_data(num_examples=args.reasoning_examples)
        reasoning_text = "".join(ex["prompt"] + ex["response"] for ex in reasoning_examples)
        aux_segments.append(("reasoning", reasoning_text))

    if args.instruction:
        from hvn.lora import generate_instruction_training_data, format_instruction_example
        print(f"\nGenerating {args.instruction_examples} instruction-tuning examples...")
        instruction_examples = generate_instruction_training_data(num_examples=args.instruction_examples)
        instruction_text = "".join(format_instruction_example(ex) for ex in instruction_examples)
        aux_segments.append(("instruction-tuning", instruction_text))

    # --- Build tokenizer (vocab spans base text + all auxiliary data) ---
    vocab_text = text + "".join(seg for _, seg in aux_segments)
    if args.tokenizer == "bpe":
        from hvn.tokenizer import BPETokenizer
        tokenizer = BPETokenizer.train(vocab_text, vocab_size=args.bpe_vocab)
        print(f"Vocabulary size: {tokenizer.vocab_size} BPE tokens")
    else:
        tokenizer = CharTokenizer(
            vocab_text, add_tool_tokens=args.tool_training or args.reasoning_training
        )
        print(f"Vocabulary size: {tokenizer.vocab_size} characters")
        if args.tool_training or args.reasoning_training:
            print(f"  (includes {len(tokenizer.special_tokens)} tool-calling/reasoning special tokens)")

    # --- Prepare dataset ---
    token_ids = tokenizer.encode(text)
    for label, seg in aux_segments:
        seg_ids = tokenizer.encode(seg)
        token_ids = token_ids + seg_ids
        print(f"  Added {len(seg_ids)} {label} tokens to training data")
    if aux_segments:
        print(f"  Total training tokens: {len(token_ids)}")

    dataset = TextDataset(token_ids, seq_len=args.seq_len)
    # Pinned-memory loading (Issue #50 --full) only helps CUDA async transfers.
    pin_memory = args.pin_memory and torch.cuda.is_available()
    if args.distributed:
        # Each rank sees a disjoint shard; shuffling lives in the sampler
        # (reshuffled per epoch via set_epoch inside train_model).
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, shuffle=True, seed=42)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                drop_last=False, pin_memory=pin_memory)
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                drop_last=False, pin_memory=pin_memory)
    if is_main:
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
        use_parallel_scan=args.use_parallel_scan,
        use_deltanet=args.use_deltanet,
        deltanet_conv_kernel=args.deltanet_conv_kernel,
        weightless=args.weightless,
        weightless_seed=args.weightless_seed,
        grad_checkpoint=args.grad_checkpoint,
    )
    if args.grad_checkpoint:
        print("Gradient (activation) checkpointing enabled (Issue #49): "
              "exact gradients, ~20-30% slower/step, lower activation memory.")
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")

    # --- Apply LoRA / QLoRA (Issue #29) ---
    if args.qlora:
        from hvn.lora import QLoRAConfig, apply_qlora, count_lora_params
        qlora_config = QLoRAConfig(
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            quant_bits=4,
        )
        model = apply_qlora(model, config=qlora_config)
        param_info = count_lora_params(model)
        print(f"QLoRA: {param_info['trainable']:,} trainable / {param_info['total']:,} total "
              f"({param_info['trainable_pct']:.1f}%)")
    elif args.lora:
        from hvn.lora import apply_lora, count_lora_params
        model = apply_lora(model, rank=args.lora_rank, alpha=args.lora_alpha,
                           dropout=args.lora_dropout)
        param_info = count_lora_params(model)
        print(f"LoRA: {param_info['trainable']:,} trainable / {param_info['total']:,} total "
              f"({param_info['trainable_pct']:.1f}%)")

    # --- Train ---
    train_model(
        model, dataloader, args.epochs, args.lr, device, args.save,
        compile=args.compile, precision=args.precision, grad_accum_steps=args.grad_accum,
        use_lora=(args.lora or args.qlora),
        fused_optimizer=args.fused_adamw,
        defer_loss_sync=args.defer_loss_sync,
        non_blocking=pin_memory,
        distributed=args.distributed,
        fsdp=args.fsdp,
    )

    # --- Save tokenizer (rank-0 only under distribution) ---
    if is_main:
        tok_path = args.save.replace(".pt", "_tokenizer.json")
        tokenizer.save(tok_path)
        print(f"✓ Tokenizer saved to {tok_path}")

    # Non-main ranks are done: skip the single-process demo paths below.
    if args.distributed and not is_main:
        hvn_dist.cleanup()
        return

    # --- Merge LoRA weights if requested (Issue #29) ---
    if (args.lora or args.qlora) and args.merge_lora:
        print("\nMerging LoRA weights into base model...")
        model = model.merge_lora()
        # Re-save the merged model
        torch.save({
            "model_state_dict": model.state_dict(),
            "vocab_size": model.vocab_size,
            "d_model": model.d_model,
            "num_layers": model.num_layers,
            "n_heads": model.n_heads,
            "n_threads": model.n_threads,
            "chunk_size": model.blocks[0].chunk_size if hasattr(model.blocks[0], 'chunk_size') else 64,
            "use_deltanet": getattr(model, "use_deltanet", False),
            "deltanet_conv_kernel": getattr(model.blocks[0], "conv_kernel", 4),
            "weightless": getattr(model, "weightless", False),
            "weightless_seed": getattr(model, "weightless_seed", 1234),
        }, args.save)
        print(f"✓ Merged model saved to {args.save}")
    elif args.lora or args.qlora:
        # Save LoRA adapter separately
        lora_path = args.save.replace(".pt", "_lora.pt")
        model.save_lora(lora_path, extra_metadata={
            "lora_rank": args.lora_rank,
            "lora_alpha": args.lora_alpha,
            "qlora": args.qlora,
        })
        print(f"✓ LoRA adapter saved to {lora_path}")

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

    # --- Multimodal fusion training (Issue #27) ---
    if args.image_text:
        print("\n" + "=" * 60)
        print("MULTIMODAL FUSION TRAINING (Issue #27)")
        print("=" * 60)
        from hvn.vision import (
            ImageEncoder,
            MultimodalFusionHVN,
            generate_synthetic_image_text_pairs,
        )

        # Build vision encoder
        vision_encoder = ImageEncoder(
            image_size=args.image_size,
            patch_size=args.patch_size,
            in_channels=args.in_channels,
            d_model=args.d_model,
        )
        multimodal_fusion = MultimodalFusionHVN(
            vision_encoder=vision_encoder,
            hvn=model,
            use_cross_modal_proj=True,
            use_gated_fusion=args.use_gated_fusion,
            use_modality_embed=args.use_modality_embed,
            use_retrieval_head=args.use_retrieval_head,
            retrieval_dim=args.d_model,
            dropout=0.1,
        ).to(device)

        num_patches = vision_encoder.num_patches
        print(f"Vision encoder: {args.image_size}×{args.image_size} → "
              f"{num_patches} patches of {args.d_model}D")
        print(f"Gated fusion: {args.use_gated_fusion} | Modality embed: {args.use_modality_embed} | "
              f"Retrieval head: {args.use_retrieval_head}")
        print(f"Retrieval weight: {args.retrieval_weight}")

        # Generate synthetic image-text pairs with correlated patterns
        print(f"Generating {args.num_images} synthetic image-text pairs (correlated patterns)...")
        images, text_ids_batch = generate_synthetic_image_text_pairs(
            num_pairs=args.num_images,
            image_size=args.image_size,
            in_channels=args.in_channels,
            text_vocab_size=model.vocab_size,
            text_seq_len=args.seq_len,
            seed=42,
        )
        images = images.to(device)
        text_ids_batch = text_ids_batch.to(device)
        print(f"Images: {images.shape} | Text: {text_ids_batch.shape}")

        # Training loop: joint LM + contrastive retrieval loss
        optimizer = torch.optim.AdamW(multimodal_fusion.parameters(), lr=args.lr, weight_decay=0.01)

        multimodal_fusion.train()
        fusion_epochs = min(args.epochs, 10)
        print(f"\nTraining multimodal fusion for {fusion_epochs} epochs...")
        print("-" * 60)

        for epoch in range(1, fusion_epochs + 1):
            total_loss = 0.0
            total_lm_loss = 0.0
            total_ret_loss = 0.0
            n_batches = 0
            batch_size = min(args.batch_size, args.num_images)

            for start in range(0, args.num_images, batch_size):
                end = min(start + batch_size, args.num_images)
                img_batch = images[start:end]
                txt_batch = text_ids_batch[start:end]

                loss_dict = multimodal_fusion.compute_loss(
                    img_batch, txt_batch,
                    retrieval_weight=args.retrieval_weight,
                )

                loss = loss_dict["total"]
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(multimodal_fusion.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                total_lm_loss += loss_dict["lm"].item()
                total_ret_loss += loss_dict["retrieval"].item()
                n_batches += 1

            avg_loss = total_loss / max(n_batches, 1)
            avg_lm = total_lm_loss / max(n_batches, 1)
            avg_ret = total_ret_loss / max(n_batches, 1)
            ppl = math.exp(min(avg_lm, 20))
            if epoch % 2 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d}/{fusion_epochs}  |  Total: {avg_loss:.4f}  |  "
                      f"LM: {avg_lm:.4f} (PPL {ppl:.2f})  |  Ret: {avg_ret:.4f}")

        # Demo 1: Image captioning
        print("\n--- Image Captioning Demo ---")
        test_image = images[:1]
        generated_ids = multimodal_fusion.generate_caption(
            test_image, max_new_tokens=50, temperature=0.8, top_k=10, device=device,
        )
        caption = tokenizer.decode(generated_ids)
        print(f"Generated caption: {caption}")

        # Demo 2: Visual Question Answering
        print("\n--- Visual QA Demo ---")
        question_ids = token_ids[:args.seq_len]  # Use first seq_len tokens as "question"
        answer_ids = multimodal_fusion.answer_question(
            test_image, question_ids, max_new_tokens=30, temperature=0.8, top_k=10, device=device,
        )
        answer = tokenizer.decode(answer_ids)
        print(f"Q: {tokenizer.decode(question_ids)}")
        print(f"A: {answer}")

        # Demo 3: Cross-modal retrieval
        if args.use_retrieval_head:
            print("\n--- Cross-Modal Retrieval Demo ---")
            # Text-to-image retrieval
            query_ids = text_ids_batch[0].tolist()
            candidate_images = images[:8]
            indices, sims = multimodal_fusion.retrieve_images(
                query_ids, candidate_images, top_k=3, device=device,
            )
            print(f"Text-to-image retrieval (query=text[0]):")
            for rank, (idx, sim) in enumerate(zip(indices.tolist(), sims.tolist())):
                print(f"  Rank {rank+1}: image[{idx}] (sim={sim:.4f})")

            # Image-to-text retrieval
            query_img = images[:1]
            text_candidates = [text_ids_batch[i].tolist() for i in range(8)]
            indices, sims = multimodal_fusion.retrieve_text(
                query_img, text_candidates, top_k=3, device=device,
            )
            print(f"Image-to-text retrieval (query=image[0]):")
            for rank, (idx, sim) in enumerate(zip(indices.tolist(), sims.tolist())):
                print(f"  Rank {rank+1}: text[{idx}] (sim={sim:.4f})")

        # Save multimodal model
        mm_save_path = args.save.replace(".pt", "_multimodal.pt")
        torch.save({
            "model_state": multimodal_fusion.state_dict(),
            "hvn_config": {
                "vocab_size": model.vocab_size,
                "d_model": args.d_model,
                "num_layers": args.num_layers,
                "n_heads": args.n_heads,
                "bond_dimension": args.bond_dim,
                "chunk_size": args.chunk_size,
                "rank_k": args.rank_k,
                "use_nonlinear_update": args.nonlinear_update,
                "use_causal_reinjection": args.causal_reinjection,
                "use_cross_head_mixing": args.cross_head_mixing,
                "use_swiglu": args.swiglu,
                "use_per_token_eta": args.per_token_eta,
                "n_threads": args.n_threads,
            },
            "vision_config": {
                "image_size": args.image_size,
                "patch_size": args.patch_size,
                "in_channels": args.in_channels,
                "use_cross_modal_proj": True,
                "use_gated_fusion": args.use_gated_fusion,
                "use_modality_embed": args.use_modality_embed,
                "use_retrieval_head": args.use_retrieval_head,
            },
        }, mm_save_path)
        print(f"\n✓ Multimodal model saved to {mm_save_path}")
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

    if args.distributed:
        hvn_dist.cleanup()


if __name__ == "__main__":
    main()