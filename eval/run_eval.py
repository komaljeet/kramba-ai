"""Real-dataset evaluation harness for HVN vs Transformer vs SSM.

Trains all three architectures on a real dataset (WikiText-2, Tiny Shakespeare,
or a Python code corpus) and reports perplexity, training time, and parameter
efficiency.  Results are printed to stdout and optionally saved as JSON.

Usage:
    # WikiText-2 (auto-download)
    python -m eval.run_eval --dataset wikitext --epochs 30

    # Tiny Shakespeare (auto-download)
    python -m eval.run_eval --dataset shakespeare --epochs 50

    # Python code corpus (auto-download)
    python -m eval.run_eval --dataset code --epochs 40

    # Use a local dataset file
    python -m eval.run_eval --dataset shakespeare --local-path /path/to/input.txt

    # Custom output
    python -m eval.run_eval --dataset wikitext --output-json results/wikitext_results.json
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure the project root is on sys.path so we can import sibling modules
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmark import TransformerLM, SimpleSSMLM
from hvn.core import HamiltonianVortexNetwork
from quality_benchmark import train_and_evaluate
from train import CharTokenizer, TextDataset

from .datasets import DATASET_REGISTRY


def run_evaluation(
    dataset_name: str,
    local_path: str | None = None,
    epochs: int = 30,
    d_model: int = 128,
    n_layers: int = 2,
    batch_size: int = 16,
    seq_len: int = 64,
    lr: float = 3e-4,
    output_json: str | None = None,
    compile: bool = False,
    models: list[str] | None = None,
) -> dict:
    """Run a full evaluation on the named dataset and return results dict.

    Parameters
    ----------
    dataset_name : str
        One of: ``wikitext``, ``shakespeare``, ``code``.
    local_path : str, optional
        Path to a local copy of the dataset file (overrides download).
    epochs : int
        Training epochs for each model.
    d_model, n_layers, batch_size, seq_len, lr :
        Shared hyperparameters passed to every model.
    output_json : str, optional
        If provided, save the results dictionary as JSON to this path.

    Returns
    -------
    results : dict
        Nested dict with per-model metrics and dataset metadata.
    """
    if dataset_name not in DATASET_REGISTRY:
        raise ValueError(
            f"Unknown dataset '{dataset_name}'. "
            f"Available: {list(DATASET_REGISTRY.keys())}"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # --- Load dataset ---
    print(f"\nLoading dataset: {dataset_name}")
    loader = DATASET_REGISTRY[dataset_name]
    kwargs = {}
    if local_path:
        kwargs["local_path"] = local_path
        # wikitext uses local_dir, not local_path
        if dataset_name == "wikitext":
            kwargs = {"local_dir": local_path}
    train_text, eval_text = loader(**kwargs)

    train_chars = len(train_text)
    eval_chars = len(eval_text)
    print(f"  Train text: {train_chars:,} characters")
    print(f"  Eval text:  {eval_chars:,} characters")

    if train_chars < 100:
        raise ValueError(
            f"Training text too short ({train_chars} chars). "
            "Check the dataset source or provide --local-path."
        )

    # --- Tokenize ---
    tokenizer = CharTokenizer(train_text + eval_text)
    vocab_size = tokenizer.vocab_size
    token_ids = tokenizer.encode(train_text)
    eval_token_ids = tokenizer.encode(eval_text)

    if not token_ids:
        raise ValueError("Training text produced no tokens after encoding.")
    if not eval_token_ids:
        raise ValueError("Eval text produced no tokens after encoding.")

    print(f"  Vocabulary: {vocab_size} chars")
    print(f"  Train tokens: {len(token_ids):,}")
    print(f"  Eval tokens:  {len(eval_token_ids):,}")

    # --- Prepare data loaders ---
    train_dataset = TextDataset(token_ids, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Build a single eval batch from the middle of the eval text
    eval_start = min(
        len(eval_token_ids) - seq_len - 1,
        max(0, len(eval_token_ids) // 2),
    )
    eval_x = torch.tensor(
        [eval_token_ids[eval_start : eval_start + seq_len]], dtype=torch.long
    )
    eval_y = torch.tensor(
        [eval_token_ids[eval_start + 1 : eval_start + seq_len + 1]], dtype=torch.long
    )

    print(f"  Train samples: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"  Eval batch: {seq_len} tokens")

    # --- Build models ---
    all_models = {
        "HVN": HamiltonianVortexNetwork(
            vocab_size=vocab_size,
            d_model=d_model,
            num_layers=n_layers,
            bond_dimension=d_model // 4,
            chunk_size=64,
        ),
        "Transformer": TransformerLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=4,
        ),
        "SSM": SimpleSSMLM(
            vocab_size=vocab_size,
            d_model=d_model,
            n_layers=n_layers,
        ),
    }

    # Filter to requested models only
    if models is None:
        models = list(all_models.keys())
    selected_models = {name: all_models[name] for name in models if name in all_models}
    if not selected_models:
        raise ValueError(f"No valid models selected. Available: {list(all_models.keys())}")
    print(f"\n  Models to evaluate: {list(selected_models.keys())}")

    # --- Train and evaluate each ---
    print("\n" + "=" * 70)
    print(f"TRAINING & EVALUATION — Dataset: {dataset_name}")
    print("=" * 70)

    results = {}
    for name, model in selected_models.items():
        result = train_and_evaluate(
            model,
            name,
            train_loader,
            (eval_x, eval_y),
            epochs=epochs,
            lr=lr,
            device=device,
            compile=compile,
        )
        results[name] = result

    # --- Comparison table ---
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)
    header = (
        f"{'Model':15s} | {'Params':>10s} | {'Eval Loss':>10s} | "
        f"{'Eval PPL':>10s} | {'Train Time':>12s} | {'PPL/Param':>12s}"
    )
    print(header)
    print("-" * 80)

    for name in results:
        r = results[name]
        ppl_per_param = r["final_eval_ppl"] / r["params"] * 1000
        print(
            f"{name:15s} | {r['params']:>10,} | {r['final_eval_loss']:>10.4f} | "
            f"{r['final_eval_ppl']:>10.2f} | {r['total_time_s']:>10.1f}s | "
            f"{ppl_per_param:>10.4f}"
        )

    # --- Analysis ---
    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    best_ppl = min(results, key=lambda k: results[k]["final_eval_ppl"])
    fastest = min(results, key=lambda k: results[k]["total_time_s"])
    most_efficient = min(
        results, key=lambda k: results[k]["final_eval_ppl"] / results[k]["params"]
    )

    print(f"  Best perplexity:       {best_ppl} (PPL={results[best_ppl]['final_eval_ppl']:.2f})")
    print(f"  Fastest training:      {fastest} ({results[fastest]['total_time_s']:.1f}s)")
    print(
        f"  Most param-efficient:  {most_efficient} "
        f"(PPL/param={results[most_efficient]['final_eval_ppl'] / results[most_efficient]['params'] * 1000:.4f}/K)"
    )

    # --- Assemble full results dict ---
    full_results = {
        "dataset": dataset_name,
        "dataset_stats": {
            "train_chars": train_chars,
            "eval_chars": eval_chars,
            "vocab_size": vocab_size,
            "train_tokens": len(token_ids),
            "eval_tokens": len(eval_token_ids),
        },
        "hyperparameters": {
            "epochs": epochs,
            "d_model": d_model,
            "n_layers": n_layers,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "lr": lr,
        },
        "models": results,
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
        description="Real-dataset evaluation harness for HVN vs Transformer vs SSM"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to evaluate on: wikitext, shakespeare, or code",
    )
    parser.add_argument(
        "--local-path",
        type=str,
        default=None,
        help="Path to a local copy of the dataset (overrides download)",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--output-json",
        type=str,
        default=None,
        help="Optional path to save results as JSON",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Use torch.compile for faster training (GPU + Triton required)",
    )
    parser.add_argument(
        "--models",
        type=str,
        nargs="+",
        default=None,
        choices=["HVN", "Transformer", "SSM"],
        help="Which models to evaluate (default: all). E.g. --models HVN Transformer",
    )
    args = parser.parse_args()

    run_evaluation(
        dataset_name=args.dataset,
        local_path=args.local_path,
        epochs=args.epochs,
        d_model=args.d_model,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        output_json=args.output_json,
        compile=args.compile,
        models=args.models,
    )


if __name__ == "__main__":
    main()