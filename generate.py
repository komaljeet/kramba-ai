"""
HVN Inference Script
====================

Load a trained HVN checkpoint and generate text.

Usage:
    python generate.py --prompt "The " --tokens 300
    python generate.py --checkpoint hvn_checkpoint.pt --prompt "Hamiltonian " --temp 0.5
"""

import argparse
import json
import time
import torch

from hvn.core import HamiltonianVortexNetwork
from train import CharTokenizer, generate_text


def load_model(checkpoint_path: str, tokenizer_path: str, device: torch.device):
    """Load a trained HVN model from checkpoint."""

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    vocab_size = ckpt["vocab_size"]
    d_model = ckpt["d_model"]
    num_layers = ckpt["num_layers"]
    n_heads = ckpt.get("n_heads", 4)  # backward compat
    chunk_size = ckpt.get("chunk_size", 64)  # backward compat

    # Rebuild model with saved hyperparameters
    model = HamiltonianVortexNetwork(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        n_heads=n_heads,
        chunk_size=chunk_size,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    # Load tokenizer — try BPE first, fall back to CharTokenizer
    try:
        from hvn.tokenizer import BPETokenizer
        tokenizer = BPETokenizer.load(tokenizer_path)
        print(f"✓ Loaded BPE tokenizer from {tokenizer_path}")
    except Exception:
        tokenizer = CharTokenizer.load(tokenizer_path)
        print(f"✓ Loaded char tokenizer from {tokenizer_path}")

    print(f"✓ Loaded model: vocab={vocab_size}, d_model={d_model}, layers={num_layers}")
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with a trained HVN model")
    parser.add_argument("--checkpoint", type=str, default="hvn_checkpoint.pt")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer JSON (default: auto-detect from checkpoint name)")
    parser.add_argument("--prompt", type=str, default="The ", help="Generation prompt")
    parser.add_argument("--prompts", type=str, default=None,
                        help="Multiple prompts separated by '||' for batched generation")
    parser.add_argument("--batch", type=int, default=1,
                        help="Number of sequences to generate in parallel (batched generation)")
    parser.add_argument("--tokens", type=int, default=200, help="Max new tokens to generate")
    parser.add_argument("--temp", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k sampling (0 = disabled)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Auto-detect tokenizer path
    tok_path = args.tokenizer or args.checkpoint.replace(".pt", "_tokenizer.json")

    # Load model + tokenizer
    model, tokenizer = load_model(args.checkpoint, tok_path, device)

    # --- Batched generation mode ---
    if args.prompts or args.batch > 1:
        if args.prompts:
            prompt_strs = [p.strip() for p in args.prompts.split("||") if p.strip()]
        else:
            prompt_strs = [args.prompt] * args.batch

        print(f"\nBatched generation: {len(prompt_strs)} sequences")
        print("-" * 60)
        for i, p in enumerate(prompt_strs):
            print(f"  Seq {i}: \"{p}\"")
        print("-" * 60)

        prompt_ids_list = [tokenizer.encode(p) or [0] for p in prompt_strs]

        t0 = time.time()
        generated_list = model.generate_batched(
            prompt_ids_list,
            max_new_tokens=args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            device=device,
        )
        elapsed = time.time() - t0

        total_tokens = sum(len(g) - len(p) for g, p in zip(generated_list, prompt_ids_list))
        for i, (gen_ids, prompt_str) in enumerate(zip(generated_list, prompt_strs)):
            output = tokenizer.decode(gen_ids)
            print(f"\n--- Sequence {i} ---")
            print(output)

        print("-" * 60)
        print(f"Generated {total_tokens} tokens across {len(prompt_strs)} sequences "
              f"in {elapsed:.2f}s ({total_tokens/elapsed:.1f} tokens/sec)")
        print(f"Method: batched incremental O(1) per-token ({len(prompt_strs)} parallel sequences)")

    # --- Single-sequence generation (default) ---
    else:
        # Generate using fast incremental generation (O(1) per token)
        print(f"\nPrompt: \"{args.prompt}\"")
        print("-" * 60)

        prompt_ids = tokenizer.encode(args.prompt)
        if not prompt_ids:
            prompt_ids = [0]

        t0 = time.time()
        generated_ids = model.generate_incremental(
            prompt_ids,
            max_new_tokens=args.tokens,
            temperature=args.temp,
            top_k=args.top_k,
            device=device,
        )
        elapsed = time.time() - t0
        output = tokenizer.decode(generated_ids)

        print(output)
        print("-" * 60)
        print(f"Generated {args.tokens} tokens in {elapsed:.2f}s ({args.tokens/elapsed:.1f} tokens/sec)")
        print(f"Method: incremental O(1) per-token (Hamiltonian state cache)")


if __name__ == "__main__":
    main()