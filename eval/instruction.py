"""
Instruction Evaluation Harness (Issue #29)
==========================================

Evaluates an instruction-tuned HVN model on instruction-following tasks.
Tests whether the model can follow commands in the Alpaca/Dolly format:

    ### Instruction:
    {instruction}

    ### Input:
    {input}

    ### Response:
    {expected output}

Metrics:
  - Instruction following accuracy (exact match / substring match)
  - Perplexity on instruction-formatted text
  - Response coherence (loss on the response portion)

Usage:
    python -m eval.instruction --checkpoint hvn_checkpoint.pt
    python -m eval.instruction --checkpoint hvn_checkpoint.pt --lora_checkpoint hvn_checkpoint_lora.pt
"""

from __future__ import annotations

import argparse
import math
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hvn.core import HamiltonianVortexNetwork
from hvn.lora import (
    generate_instruction_training_data,
    format_instruction_example,
    apply_lora,
    load_lora_checkpoint,
    count_lora_params,
)
from train import CharTokenizer


# ---------------------------------------------------------------------------
# Instruction Dataset
# ---------------------------------------------------------------------------

class InstructionDataset(Dataset):
    """Dataset for instruction-tuning evaluation.

    Each sample is a formatted instruction example (Alpaca format).
    The input is the full formatted text up to (but not including) the response,
    and the target is the response text.

    Parameters
    ----------
    examples : list[dict]
        Instruction examples with keys "instruction", "input", "output".
    tokenizer : CharTokenizer
        Tokenizer for encoding text.
    seq_len : int
        Maximum sequence length.
    """

    def __init__(self, examples: list[dict], tokenizer: CharTokenizer, seq_len: int = 256):
        self.samples = []
        self.tokenizer = tokenizer
        self.seq_len = seq_len

        for ex in examples:
            formatted = format_instruction_example(ex)
            ids = tokenizer.encode(formatted)
            # Split into chunks of seq_len
            for start in range(0, len(ids), seq_len):
                end = min(start + seq_len + 1, len(ids))
                chunk = ids[start:end]
                if len(chunk) < 2:
                    continue
                # Pad if needed
                if len(chunk) < seq_len + 1:
                    chunk = chunk + [0] * (seq_len + 1 - len(chunk))
                x = torch.tensor(chunk[:-1], dtype=torch.long)
                y = torch.tensor(chunk[1:], dtype=torch.long)
                self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# ---------------------------------------------------------------------------
# Evaluation Functions
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_instruction_ppl(
    model: HamiltonianVortexNetwork,
    dataloader: DataLoader,
    device: torch.device,
) -> dict:
    """Evaluate perplexity on instruction-formatted text.

    Returns
    -------
    dict with keys:
        ``ppl`` : float — perplexity
        ``loss`` : float — average cross-entropy loss
        ``n_samples`` : int — number of evaluation samples
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_samples = 0

    for batch_x, batch_y in dataloader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)
        logits = model(batch_x)  # [B, S, V]
        loss = criterion(
            logits.reshape(-1, model.vocab_size),
            batch_y.reshape(-1),
        )
        total_loss += loss.item() * batch_x.size(0)
        n_samples += batch_x.size(0)

    avg_loss = total_loss / max(n_samples, 1)
    ppl = math.exp(min(avg_loss, 20))
    return {"ppl": ppl, "loss": avg_loss, "n_samples": n_samples}


@torch.no_grad()
def evaluate_instruction_following(
    model: HamiltonianVortexNetwork,
    examples: list[dict],
    tokenizer: CharTokenizer,
    device: torch.device,
    max_new_tokens: int = 50,
    temperature: float = 0.3,
    top_k: int = 5,
) -> dict:
    """Evaluate instruction-following accuracy.

    For each example, generates a response from the model given the instruction
    (without the expected output), then checks if the generated response
    matches the expected output (exact match or substring match).

    Returns
    -------
    dict with keys:
        ``exact_match`` : float — fraction of exact matches
        ``substring_match`` : float — fraction of substring matches
        ``results`` : list[dict] — per-example results
    """
    model.eval()
    results = []
    exact_matches = 0
    substring_matches = 0

    for ex in examples:
        # Build prompt (instruction + input, without response)
        instruction = ex["instruction"]
        inp = ex.get("input", "")
        expected = ex["output"]

        if inp:
            prompt = (
                f"Below is an instruction that describes a task. Write a response.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Input:\n{inp}\n\n"
                f"### Response:\n"
            )
        else:
            prompt = (
                f"Below is an instruction that describes a task. Write a response.\n\n"
                f"### Instruction:\n{instruction}\n\n"
                f"### Response:\n"
            )

        # Generate response
        ids = tokenizer.encode(prompt)
        if not ids:
            ids = [0]

        generated = model.generate_incremental(
            prompt_ids=ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            device=device,
        )
        response_ids = generated[len(ids):]
        response = tokenizer.decode(response_ids).strip()

        # Check matches
        exact = response.strip() == expected.strip()
        substring = expected.strip().lower() in response.lower()

        if exact:
            exact_matches += 1
        if substring:
            substring_matches += 1

        results.append({
            "instruction": instruction,
            "input": inp,
            "expected": expected,
            "generated": response,
            "exact_match": exact,
            "substring_match": substring,
        })

    n = len(examples)
    return {
        "exact_match": exact_matches / n if n > 0 else 0.0,
        "substring_match": substring_matches / n if n > 0 else 0.0,
        "results": results,
    }


# ---------------------------------------------------------------------------
# Main Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Evaluate instruction-tuned HVN model")
    parser.add_argument("--checkpoint", type=str, default="hvn_checkpoint.pt",
                        help="Path to model checkpoint")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to tokenizer JSON (default: auto-detect from checkpoint)")
    parser.add_argument("--lora_checkpoint", type=str, default=None,
                        help="Path to LoRA adapter checkpoint (optional)")
    parser.add_argument("--num_examples", type=int, default=20,
                        help="Number of instruction examples for evaluation")
    parser.add_argument("--seq_len", type=int, default=128,
                        help="Sequence length for PPL evaluation")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for PPL evaluation")
    parser.add_argument("--max_new_tokens", type=int, default=50,
                        help="Max tokens to generate per instruction")
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Generation temperature (low = more deterministic)")
    parser.add_argument("--top_k", type=int, default=5,
                        help="Top-k sampling")
    parser.add_argument("--seed", type=int, default=123,
                        help="Random seed for evaluation examples")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # --- Load tokenizer ---
    tok_path = args.tokenizer or args.checkpoint.replace(".pt", "_tokenizer.json")
    tokenizer = CharTokenizer.load(tok_path)
    print(f"Tokenizer loaded: {tokenizer.vocab_size} tokens")

    # --- Load model ---
    ckpt = torch.load(args.checkpoint, weights_only=False, map_location=device)
    model = HamiltonianVortexNetwork(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=ckpt["num_layers"],
        n_heads=ckpt.get("n_heads", 4),
        bond_dimension=ckpt.get("bond_dimension", ckpt.get("bond_dim", 32)),
        chunk_size=ckpt.get("chunk_size", 64),
        n_threads=ckpt.get("n_threads", 1),
    )

    # Check if the checkpoint has LoRA layers (keys contain "lora.")
    state_dict = ckpt["model_state_dict"]
    has_lora = any("lora.lora_A" in k for k in state_dict.keys())

    if has_lora:
        # Apply LoRA first, then load the full state dict
        model = apply_lora(model, rank=8)
        model.load_state_dict(state_dict)
        param_info = count_lora_params(model)
        print(f"Model loaded with LoRA: {param_info['trainable']:,} trainable params")
    else:
        model.load_state_dict(state_dict)

    model.to(device)
    print(f"Model loaded: d_model={ckpt['d_model']}, layers={ckpt['num_layers']}")

    # --- Load LoRA adapter if provided ---
    if args.lora_checkpoint:
        model = apply_lora(model, rank=8)  # rank will be overridden by checkpoint
        model.load_lora(args.lora_checkpoint)
        param_info = count_lora_params(model)
        print(f"LoRA loaded: {param_info['trainable']:,} trainable params")

    # --- Generate evaluation examples ---
    print(f"\nGenerating {args.num_examples} instruction examples for evaluation...")
    eval_examples = generate_instruction_training_data(
        num_examples=args.num_examples,
        seed=args.seed,
    )

    # --- Evaluate perplexity ---
    print("\n" + "=" * 60)
    print("  Instruction Perplexity Evaluation")
    print("=" * 60)
    eval_dataset = InstructionDataset(eval_examples, tokenizer, seq_len=args.seq_len)
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)
    ppl_results = evaluate_instruction_ppl(model, eval_dataloader, device)
    print(f"  Perplexity: {ppl_results['ppl']:.2f}")
    print(f"  Loss: {ppl_results['loss']:.4f}")
    print(f"  Samples: {ppl_results['n_samples']}")

    # --- Evaluate instruction following ---
    print("\n" + "=" * 60)
    print("  Instruction Following Evaluation")
    print("=" * 60)
    follow_results = evaluate_instruction_following(
        model, eval_examples, tokenizer, device,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )
    print(f"  Exact match: {follow_results['exact_match']:.1%}")
    print(f"  Substring match: {follow_results['substring_match']:.1%}")

    # --- Show sample results ---
    print("\n  Sample results:")
    for i, r in enumerate(follow_results["results"][:5]):
        print(f"\n  [{i+1}] Instruction: {r['instruction']}")
        if r['input']:
            print(f"      Input: {r['input']}")
        print(f"      Expected: {r['expected']}")
        print(f"      Generated: {r['generated'][:100]}")
        print(f"      Match: {'✓' if r['substring_match'] else '✗'}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()