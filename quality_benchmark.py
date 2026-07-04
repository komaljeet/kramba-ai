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
import json
import math
import os
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from hvn.core import HamiltonianVortexNetwork
from benchmark import TransformerLM, SimpleSSMLM
from train import CharTokenizer, TextDataset, SAMPLE_TEXT


def load_corpus_text(text_path: str | None, repeat: int = 50) -> str:
    """Load text from a file when provided, otherwise fall back to the bundled sample."""
    if text_path:
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Text corpus not found: {text_path}")
        with open(text_path, "r", encoding="utf-8") as handle:
            text = handle.read()
        if not text.strip():
            raise ValueError(f"Text corpus is empty: {text_path}")
        return text
    return SAMPLE_TEXT * repeat


def train_and_evaluate(
    model: nn.Module,
    name: str,
    train_loader: DataLoader,
    eval_data: tuple[torch.Tensor, torch.Tensor],
    epochs: int,
    lr: float,
    device: torch.device,
    compile: bool = False,
    precision: str = "fp32",
    grad_accum_steps: int = 1,
) -> dict:
    """Train a model and return training history + final eval metrics.

    Parameters
    ----------
    precision : str
        Mixed-precision mode: "fp32" (default), "fp16", or "bf16".
    grad_accum_steps : int
        Number of micro-batches to accumulate gradients over before each
        optimizer step.  Effective batch size = batch_size × grad_accum_steps.
    """

    if compile and device.type == "cuda":
        print(f"  Compiling {name} with torch.compile...")
        model = torch.compile(model, mode="reduce-overhead")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    # --- Mixed-precision setup ---
    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and precision == "fp16") else None

    if use_amp:
        print(f"  Mixed-precision: {precision.upper()}")
    if grad_accum_steps > 1:
        print(f"  Gradient accumulation: {grad_accum_steps} steps "
              f"(effective batch size = {train_loader.batch_size * grad_accum_steps})")

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
        optimizer.zero_grad()

        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(batch_x)
                loss = criterion(
                    logits.reshape(-1, model.vocab_size),
                    batch_y.reshape(-1),
                )
                loss = loss / grad_accum_steps

            if scaler is not None:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            total_loss += loss.item() * grad_accum_steps
            n_batches += 1

            if (step + 1) % grad_accum_steps == 0:
                if scaler is not None:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()

        # Handle remaining accumulated gradients at epoch end
        if n_batches % grad_accum_steps != 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        scheduler.step()
        avg_loss = total_loss / max(n_batches, 1)

        # Evaluate
        model.eval()
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
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
    parser.add_argument("--text", type=str, default=None, help="Optional path to a training corpus text file")
    parser.add_argument("--eval-text", type=str, default=None, help="Optional path to a separate evaluation corpus text file")
    parser.add_argument("--train-split", type=float, default=0.9, help="Fraction of the corpus used for training")
    parser.add_argument("--output-json", type=str, default=None, help="Optional path to save benchmark results as JSON")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for faster training (GPU + Triton required)")
    parser.add_argument("--precision", type=str, default="fp32", choices=["fp32", "fp16", "bf16"], help="Mixed-precision training mode (fp16/bf16 require CUDA)")
    parser.add_argument("--grad_accum", type=int, default=1, help="Gradient accumulation steps (effective batch size = batch_size × grad_accum)")
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
    parser.add_argument("--use_deltanet", action="store_true", help="Use Gated-DeltaNet blocks (Issue #34): fixes readout/update dynamics to close the PPL gap. Preserves O(1) inference.")
    parser.add_argument("--deltanet_conv_kernel", type=int, default=4, help="Kernel size of the short causal depthwise conv in DeltaNet blocks (Issue #34).")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: d_model={args.d_model}, layers={args.n_layers}, epochs={args.epochs}, seq_len={args.seq_len}")

    # --- Prepare data ---
    try:
        train_text = load_corpus_text(args.text, repeat=50)
        eval_text = load_corpus_text(args.eval_text or args.text, repeat=1)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc))

    tokenizer = CharTokenizer(train_text + eval_text)
    vocab_size = tokenizer.vocab_size
    token_ids = tokenizer.encode(train_text)
    eval_token_ids = tokenizer.encode(eval_text)

    if not token_ids:
        raise SystemExit("The training corpus did not contain any tokenizable characters")
    if not eval_token_ids:
        raise SystemExit("The evaluation corpus did not contain any tokenizable characters")

    split = max(1, min(len(token_ids) - 2, int(len(token_ids) * args.train_split)))
    train_dataset = TextDataset(token_ids[:split], seq_len=args.seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    eval_start = min(len(eval_token_ids) - args.seq_len - 1, max(0, len(eval_token_ids) // 2))
    eval_x = torch.tensor([eval_token_ids[eval_start:eval_start + args.seq_len]], dtype=torch.long)
    eval_y = torch.tensor([eval_token_ids[eval_start + 1:eval_start + args.seq_len + 1]], dtype=torch.long)

    print(f"Vocabulary: {vocab_size} chars")
    print(f"Train samples: {len(train_dataset)}, Batches: {len(train_loader)}")
    print(f"Eval samples: 1 batch ({args.seq_len} tokens)")

    # --- Build models ---
    models = {
        "HVN": HamiltonianVortexNetwork(
            vocab_size=vocab_size, d_model=args.d_model,
            num_layers=args.n_layers, bond_dimension=args.d_model // 4,
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
            compile=args.compile, precision=args.precision,
            grad_accum_steps=args.grad_accum,
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

    if args.output_json:
        output_path = os.path.abspath(args.output_json)
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(results, handle, indent=2)
        print(f"\nSaved benchmark summary to {output_path}")

    print()


if __name__ == "__main__":
    main()