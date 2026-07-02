"""Long-context evaluation harness for HVN.

Tests HVN's O(1) scaling advantage at long sequence lengths by measuring:
  - Forward pass speed at N = {256, 512, 1024, 2048, 4096} tokens
  - Peak memory usage
  - A needle-in-haystack retrieval task to test long-range associative recall

Usage:
    python -m eval.long_context
    python -m eval.long_context --max_seq_len 2048 --d_model 128
    python -m eval.long_context --output-json results/long_context.json
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from hvn.core import HamiltonianVortexNetwork


def measure_forward_speed(
    model: nn.Module,
    seq_len: int,
    vocab_size: int,
    device: torch.device,
    warmup: int = 2,
) -> dict:
    """Measure forward pass speed for a given sequence length."""
    input_ids = torch.randint(0, vocab_size, (1, seq_len), device=device)

    # Warmup
    for _ in range(min(warmup, 2)):
        with torch.no_grad():
            _ = model(input_ids)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

    t0 = time.time()
    with torch.no_grad():
        logits = model(input_ids)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - t0

    peak_mem = 0
    if device.type == "cuda":
        peak_mem = torch.cuda.max_memory_allocated(device) / (1024 * 1024)

    tokens_per_sec = seq_len / elapsed if elapsed > 0 else 0
    ms_per_token = (elapsed / seq_len) * 1000 if seq_len > 0 else 0

    return {
        "seq_len": seq_len,
        "elapsed_s": elapsed,
        "tokens_per_sec": tokens_per_sec,
        "ms_per_token": ms_per_token,
        "peak_memory_mb": peak_mem,
        "logits_shape": list(logits.shape),
    }


def needle_in_haystack(
    model: nn.Module,
    vocab_size: int,
    seq_len: int,
    device: torch.device,
) -> dict:
    """Needle-in-haystack retrieval test.

    Places a distinctive "needle" token at a known position in a random sequence,
    then checks whether the model's output at the final position assigns high
    probability to the needle token (testing long-range associative recall).

    Returns a dict with the needle position, the rank of the needle token in
    the model's output distribution, and whether it was in the top-k.
    """
    if seq_len < 4:
        return {"error": "seq_len too short for needle test"}

    # Use a special needle token (last token in vocab) placed at position seq_len//2
    needle_token = vocab_size - 1
    needle_pos = seq_len // 2

    # Fill the rest with random tokens (excluding the needle)
    input_ids = torch.randint(0, needle_token, (1, seq_len), device=device)
    input_ids[0, needle_pos] = needle_token

    with torch.no_grad():
        logits = model(input_ids)  # [1, seq_len, vocab_size]

    # Check the output at the last position — does it "remember" the needle?
    last_logits = logits[0, -1, :]  # [vocab_size]
    probs = torch.softmax(last_logits, dim=-1)
    needle_prob = probs[needle_token].item()

    # Rank of the needle token in the output distribution
    sorted_indices = torch.argsort(last_logits, descending=True)
    needle_rank = (sorted_indices == needle_token).nonzero(as_tuple=True)[0].item()
    top_k_hit = needle_rank < 10  # is the needle in the top-10 predictions?

    return {
        "seq_len": seq_len,
        "needle_pos": needle_pos,
        "needle_token": needle_token,
        "needle_prob": needle_prob,
        "needle_rank": needle_rank,
        "top_10_hit": top_k_hit,
    }


def run_long_context_eval(
    d_model: int = 128,
    n_layers: int = 2,
    max_seq_len: int = 2048,
    vocab_size: int = 1000,
    output_json: str | None = None,
) -> dict:
    """Run the full long-context evaluation suite."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    model = HamiltonianVortexNetwork(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=n_layers,
        n_heads=4,
        bond_dimension=d_model // 4,
        chunk_size=64,
    )
    model.to(device)
    model.eval()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {param_count:,}")
    print(f"d_model={d_model}, layers={n_layers}, bond_dim={d_model // 4}")

    # --- Speed scaling test ---
    print("\n" + "=" * 70)
    print("FORWARD PASS SPEED SCALING")
    print("=" * 70)
    print(f"{'Seq Len':>8s} | {'Time (s)':>10s} | {'Tok/s':>10s} | {'ms/tok':>10s} | {'Peak Mem (MB)':>14s}")
    print("-" * 60)

    seq_lengths = [256, 512, 1024, 2048, 4096, 8192]
    seq_lengths = [s for s in seq_lengths if s <= max_seq_len]

    speed_results = []
    for seq_len in seq_lengths:
        try:
            result = measure_forward_speed(model, seq_len, vocab_size, device)
            speed_results.append(result)
            print(
                f"{result['seq_len']:>8d} | {result['elapsed_s']:>10.4f} | "
                f"{result['tokens_per_sec']:>10.1f} | {result['ms_per_token']:>10.4f} | "
                f"{result['peak_memory_mb']:>14.1f}"
            )
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"{seq_len:>8d} | {'OOM':>10s} | {'—':>10s} | {'—':>10s} | {'—':>14s}")
                break
            raise

    # --- Needle-in-haystack test ---
    print("\n" + "=" * 70)
    print("NEEDLE-IN-HAYSTACK RETRIEVAL TEST")
    print("=" * 70)
    print(f"{'Seq Len':>8s} | {'Needle Pos':>12s} | {'Needle Prob':>12s} | {'Rank':>8s} | {'Top-10 Hit':>12s}")
    print("-" * 60)

    needle_results = []
    for seq_len in seq_lengths:
        result = needle_in_haystack(model, vocab_size, seq_len, device)
        needle_results.append(result)
        print(
            f"{result['seq_len']:>8d} | {result['needle_pos']:>12d} | "
            f"{result['needle_prob']:>12.6f} | {result['needle_rank']:>8d} | "
            f"{'✓' if result['top_10_hit'] else '✗':>12s}"
        )

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    if speed_results:
        shortest = speed_results[0]
        longest = speed_results[-1]
        print(f"  Shortest seq ({shortest['seq_len']}): {shortest['ms_per_token']:.4f} ms/tok")
        print(f"  Longest seq  ({longest['seq_len']}): {longest['ms_per_token']:.4f} ms/tok")
        ratio = longest["ms_per_token"] / shortest["ms_per_token"] if shortest["ms_per_token"] > 0 else 0
        print(f"  Scaling ratio: {ratio:.2f}x (O(1) means ≈1.0)")

    hits = sum(1 for r in needle_results if r["top_10_hit"])
    print(f"  Needle-in-haystack: {hits}/{len(needle_results)} sequences had needle in top-10")

    full_results = {
        "model_config": {
            "d_model": d_model,
            "n_layers": n_layers,
            "vocab_size": vocab_size,
            "params": param_count,
        },
        "speed_scaling": speed_results,
        "needle_in_haystack": needle_results,
    }

    if output_json:
        output_path = os.path.abspath(output_json)
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(full_results, handle, indent=2)
        print(f"\n✓ Results saved to {output_path}")

    return full_results


def main():
    parser = argparse.ArgumentParser(
        description="Long-context evaluation harness for HVN"
    )
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--max_seq_len", type=int, default=2048, help="Maximum sequence length to test")
    parser.add_argument("--vocab_size", type=int, default=1000, help="Vocabulary size")
    parser.add_argument("--output-json", type=str, default=None, help="Save results as JSON")
    args = parser.parse_args()

    run_long_context_eval(
        d_model=args.d_model,
        n_layers=args.n_layers,
        max_seq_len=args.max_seq_len,
        vocab_size=args.vocab_size,
        output_json=args.output_json,
    )


if __name__ == "__main__":
    main()