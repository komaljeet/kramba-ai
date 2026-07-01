"""
Quality Benchmark: HVN vs Transformer vs SSM
=============================================

Trains all three architectures on the same text corpus and compares:
  - Training loss convergence
  - Validation perplexity
  - Parameter efficiency (perplexity per parameter)

Usage:
    python quality_benchmark.py
    python quality_benchmark.py --epochs 50 --d_model 128
"""

import argparse
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hvn.core import HamiltonianVortexNetwork
from benchmark import TransformerLM, SimpleSSMLM
from train import CharTokenizer, TextDataset, SAMPLE_TEXT


def train_and_evaluate(
    model: nn.Module,
    name: str,
    train_loader: DataLoader,
    eval_data: tuple[torch.Tensor, torch.Tensor],
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict:
    """Train a model and return training history + final eval metrics."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    eval_x, eval_y = eval_data
    eval_x = eval_x.to(device)
    eval_y = eval_y.to(device)

    history = []
    param_count = sum(p.numel() for p in model.parameters())

    print(f"\n  Training {name} ({param_count:,} params)...")

    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        n_batches = 0
        t0 = time.time()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            logits = model(batch_x)
            loss = criterion(
                logits.reshape(-1, model.vocab_size),
                batch_y.reshape(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate
        model.eval()
        with torch.no_grad():
            eval_logits = model(eval_x)
            eval_loss = criterion(
                eval_logits.reshape(-1, model.vocab_size),
                eval_y.reshape(-1),
            ).item()
        model.train()

        eval_ppl = math.exp(min(eval_loss, 20))
        elapsed = time.time() - t0

        history.append({
            "epoch": epoch,
            "train_loss": avg_loss,
            "eval_loss": eval_loss,
            "eval_ppl": eval_ppl,
            "time_s": elapsed,
        })

        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            print(f"    Epoch {epoch:3d}/{epochs}  |  train_loss={avg_loss:.4f}  |  "
                  f"eval_ppl={eval_ppl:.2f}  |  {elapsed:.1f}s")

    return {
        "name": name,
        "params": param_count,
        "history": history,
        "final_eval_loss": history[-1]["eval_loss"],
        "final_eval_ppl": history[-1]["eval_ppl"],
        "total_time_s": sum(h["time_s"] for h in history),
    }


def main():
    parser = argparse.ArgumentParser(description="Quality benchmark: HVN vs Transformer vs SSM")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--n_layers", type=int, default=2, help="Number of layers")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--seq_len", type=int, default=64, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: d_model={args.d_model}, layers={args.n_layers}, epochs={args.epochs}, seq_len={args.seq_len}")

    # --- Prepare data ---
    text = SAMPLE_TEXT * 50
    tokenizer = CharTokenizer(text)
    vocab_size = tokenizer.vocab_size
    token_ids = tokenizer.encode(text)

    # Split: 90% train, 10% eval
    split = int(len(token_ids) * 0.9)
    train_dataset = TextDataset(token_ids[:split], seq_len=args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    # Eval data: one batch
    eval_x = torch.tensor([token_ids[split:split + args.seq_len]], dtype=torch.long)
    eval_y = torch.tensor([token_ids[split + 1:split + args.seq_len + 1]], dtype=torch.long)

    print(f"Vocabulary: {vocab_size} chars")
    print(f"Train samples: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"Eval samples: 1 batch ({args.seq_len} tokens)")

    # --- Build models ---
    models = {
        "HVN": HamiltonianVortexNetwork(
            vocab_size=vocab_size, d_model=args.d_model,
            num_layers=args.n_layers, bond_dimension=args.d_model // 4,
        ),
        "Transformer": TransformerLM(
            vocab_size=vocab_size, d_model=args.d_model,
            n_layers=args.n_layers, n_heads=4,
        ),
        "SSM": SimpleSSMLM(
            vocab_size=vocab_size, d_model=args.d_model,
            n_layers=args.n_layers,
        ),
    }

    # --- Train and evaluate each ---
    print("\n" + "=" * 70)
    print("TRAINING & EVALUATION")
    print("=" * 70)

    results = {}
    for name, model in models.items():
        result = train_and_evaluate(
            model, name, train_loader, (eval_x, eval_y),
            epochs=args.epochs, lr=args.lr, device=device,
        )
        results[name] = result

    # --- Comparison table ---
    print("\n" + "=" * 70)
    print("FINAL RESULTS COMPARISON")
    print("=" * 70)
    print(f"{'Model':15s} | {'Params':>10s} | {'Eval Loss':>10s} | {'Eval PPL':>10s} | {'Train Time':>12s} | {'PPL/Param':>12s}")
    print("-" * 80)

    for name in ["HVN", "Transformer", "SSM"]:
        r = results[name]
        ppl_per_param = r["final_eval_ppl"] / r["params"] * 1000  # per 1K params
        print(f"{name:15s} | {r['params']:>10,} | {r['final_eval_loss']:>10.4f} | "
              f"{r['final_eval_ppl']:>10.2f} | {r['total_time_s']:>10.1f}s | {ppl_per_param:>10.4f}")

    print()

    # --- Analysis ---
    print("=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    best_ppl = min(results, key=lambda k: results[k]["final_eval_ppl"])
    fastest = min(results, key=lambda k: results[k]["total_time_s"])
    most_efficient = min(results, key=lambda k: results[k]["final_eval_ppl"] / results[k]["params"])

    print(f"  Best perplexity:       {best_ppl} (PPL={results[best_ppl]['final_eval_ppl']:.2f})")
    print(f"  Fastest training:      {fastest} ({results[fastest]['total_time_s']:.1f}s)")
    print(f"  Most param-efficient:  {most_efficient} (PPL/param={results[most_efficient]['final_eval_ppl']/results[most_efficient]['params']*1000:.4f}/K)")

    # --- Loss curves (text-based) ---
    print("\n" + "=" * 70)
    print("TRAINING LOSS CURVES (eval perplexity)")
    print("=" * 70)

    epochs = args.epochs
    n_points = min(10, epochs)
    step = max(1, epochs // n_points)

    for name in ["HVN", "Transformer", "SSM"]:
        hist = results[name]["history"]
        print(f"\n  {name}:")
        for i in range(0, epochs, step):
            if i < len(hist):
                h = hist[i]
                bar_len = int(h["eval_ppl"] / 2)
                bar = "█" * min(bar_len, 50)
                print(f"    Epoch {h['epoch']:3d}: PPL={h['eval_ppl']:6.2f} |{bar}")

    print()


if __name__ == "__main__":
    main()