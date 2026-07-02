#!/usr/bin/env python
"""
Scaling Laws Evaluation for the Hamiltonian Vortex Network (HVN).

Trains HVN and a Transformer baseline at multiple model sizes (d_model =
{64, 128, 256, 512}) and plots scaling curves:

  1. PPL vs. parameter count (log-log)
  2. PPL vs. wall-clock training time
  3. Wall-clock time vs. parameter count (crossover-point analysis)

Usage
-----
    python -m eval.scaling_laws --dataset shakespeare --epochs 20 --output-json results/scaling_laws.json
    python -m eval.scaling_laws --dataset shakespeare --epochs 20 --d_models 64,128,256 --output-json results/scaling_laws.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # non-interactive backend — safe for headless GPU servers
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup — allow running as `python -m eval.scaling_laws` or `python eval/scaling_laws.py`
# ---------------------------------------------------------------------------
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from benchmark import TransformerLM, count_parameters  # noqa: E402
from hvn.core import HamiltonianVortexNetwork  # noqa: E402
from quality_benchmark import train_and_evaluate  # noqa: E402
from train import CharTokenizer, TextDataset  # noqa: E402
from .datasets import DATASET_REGISTRY  # noqa: E402


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_scaling_laws(
    dataset_name: str,
    local_path: str | None = None,
    d_models: list[int] | None = None,
    epochs: int = 20,
    n_layers: int = 2,
    batch_size: int = 16,
    seq_len: int = 64,
    lr: float = 3e-4,
    output_json: str | None = None,
    output_plot: str | None = None,
    compile_model: bool = False,
    models: list[str] | None = None,
    precision: str = "fp32",
    grad_accum_steps: int = 1,
    rank_k: int = 1,
    use_nonlinear_update: bool = False,
    use_causal_reinjection: bool = False,
    use_cross_head_mixing: bool = False,
    use_swiglu: bool = False,
    use_per_token_eta: bool = False,
    n_threads: int = 1,
) -> dict:
    """Train HVN and Transformer at multiple d_model sizes and collect scaling data.

    Returns a dict with per-size, per-model results plus a summary analysis.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — running on CPU. This will be very slow.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    if d_models is None:
        d_models = [64, 128, 256, 512]

    if models is None:
        models = ["HVN", "Transformer"]

    # --- Load dataset -------------------------------------------------------
    print(f"\n{'='*70}")
    print(f"Loading dataset: {dataset_name}")
    print(f"{'='*70}")

    loader_fn = DATASET_REGISTRY[dataset_name]
    train_text, eval_text = loader_fn(local_path)
    tokenizer = CharTokenizer(train_text + eval_text)
    vocab_size = tokenizer.vocab_size

    train_ids = tokenizer.encode(train_text)
    eval_ids = tokenizer.encode(eval_text)

    train_dataset = TextDataset(train_ids, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # Single eval batch from the middle of eval text
    mid = len(eval_ids) // 2
    eval_x = torch.tensor([eval_ids[mid : mid + seq_len]], dtype=torch.long, device=device)
    eval_y = torch.tensor([eval_ids[mid + 1 : mid + 1 + seq_len]], dtype=torch.long, device=device)

    print(f"  Vocab size:      {vocab_size}")
    print(f"  Train chars:      {len(train_text):,}")
    print(f"  Eval chars:       {len(eval_text):,}")
    print(f"  Train samples:    {len(train_dataset)}")
    print(f"  d_models:         {d_models}")
    print(f"  Models:           {models}")
    print(f"  Epochs per run:   {epochs}")

    # --- Run scaling sweep --------------------------------------------------
    all_results: list[dict] = []

    for d_model in d_models:
        print(f"\n{'='*70}")
        print(f"d_model = {d_model}")
        print(f"{'='*70}")

        bond_dim = max(d_model // 4, 8)

        model_specs: dict[str, nn.Module] = {}
        if "HVN" in models:
            model_specs["HVN"] = HamiltonianVortexNetwork(
                vocab_size=vocab_size,
                d_model=d_model,
                num_layers=n_layers,
                bond_dimension=bond_dim,
                chunk_size=64,
                dropout=0.1,
                rank_k=rank_k,
                use_nonlinear_update=use_nonlinear_update,
                use_causal_reinjection=use_causal_reinjection,
                use_cross_head_mixing=use_cross_head_mixing,
                use_swiglu=use_swiglu,
                use_per_token_eta=use_per_token_eta,
                n_threads=n_threads,
            )
        if "Transformer" in models:
            model_specs["Transformer"] = TransformerLM(
                vocab_size=vocab_size,
                d_model=d_model,
                n_layers=n_layers,
                n_heads=4,
            )

        for name, model in model_specs.items():
            param_count = count_parameters(model)
            print(f"\n  {name} (d_model={d_model}, params={param_count:,})")

            result = train_and_evaluate(
                model=model,
                name=name,
                train_loader=train_loader,
                eval_data=(eval_x, eval_y),
                epochs=epochs,
                lr=lr,
                device=device,
                compile=compile_model,
                precision=precision,
                grad_accum_steps=grad_accum_steps,
            )

            # Enrich with scaling metadata
            result["d_model"] = d_model
            result["n_layers"] = n_layers
            all_results.append(result)

            print(f"    Final PPL: {result['final_eval_ppl']:.2f}  |  Time: {result['total_time_s']:.1f}s")

            # Free memory between runs
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Summary table ------------------------------------------------------
    print(f"\n{'='*70}")
    print("Scaling Laws Summary")
    print(f"{'='*70}")
    print(f"{'Model':<14} {'d_model':>8} {'Params':>10} {'Eval PPL':>10} {'Train Time':>12} {'PPL/1K params':>14}")
    print("-" * 70)

    for r in all_results:
        ppl_per_1k = r["final_eval_ppl"] / max(r["params"] / 1000, 1)
        print(
            f"{r['name']:<14} {r['d_model']:>8} {r['params']:>10,} "
            f"{r['final_eval_ppl']:>10.2f} {r['total_time_s']:>10.1f}s "
            f"{ppl_per_1k:>14.4f}"
        )

    # --- Crossover analysis -------------------------------------------------
    analysis = _compute_crossover(all_results)
    print(f"\n{'='*70}")
    print("Crossover Analysis")
    print(f"{'='*70}")
    for line in analysis["summary_lines"]:
        print(f"  {line}")

    # --- Assemble full results ----------------------------------------------
    full_results = {
        "dataset": dataset_name,
        "dataset_stats": {
            "vocab_size": vocab_size,
            "train_chars": len(train_text),
            "eval_chars": len(eval_text),
        },
        "hyperparameters": {
            "d_models": d_models,
            "n_layers": n_layers,
            "epochs": epochs,
            "batch_size": batch_size,
            "seq_len": seq_len,
            "lr": lr,
            "compile": compile_model,
        },
        "models": all_results,
        "analysis": analysis,
    }

    # --- JSON output --------------------------------------------------------
    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nResults saved to {output_json}")

    # --- Plots --------------------------------------------------------------
    if output_plot:
        os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
        _plot_scaling_curves(all_results, analysis, output_plot, dataset_name)
        print(f"Plots saved to {output_plot}")

    return full_results


# ---------------------------------------------------------------------------
# Crossover analysis
# ---------------------------------------------------------------------------

def _compute_crossover(results: list[dict]) -> dict:
    """Determine the crossover point where HVN beats Transformer on wall-clock time.

    Also computes scaling exponents (power-law fits) for PPL vs params.
    """
    hvn = sorted([r for r in results if r["name"] == "HVN"], key=lambda r: r["params"])
    tr  = sorted([r for r in results if r["name"] == "Transformer"], key=lambda r: r["params"])

    summary_lines: list[str] = []

    # --- Power-law fit: PPL ~ a * params^b  →  log(PPL) = log(a) + b*log(params)
    def _power_law_fit(data: list[dict]) -> tuple[float, float] | None:
        """Return (a, b) for PPL = a * params^b, or None if insufficient data."""
        if len(data) < 2:
            return None
        import numpy as np
        log_p = np.array([math.log(r["final_eval_ppl"]) for r in data])
        log_n = np.array([math.log(r["params"]) for r in data])
        # Linear regression: log_p = log_a + b * log_n
        n = len(data)
        slope = (n * np.sum(log_p * log_n) - np.sum(log_p) * np.sum(log_n)) / (
            n * np.sum(log_n**2) - np.sum(log_n) ** 2
        )
        intercept = (np.sum(log_p) - slope * np.sum(log_n)) / n
        return (math.exp(intercept), slope)

    hvn_fit = _power_law_fit(hvn)
    tr_fit = _power_law_fit(tr)

    if hvn_fit:
        a, b = hvn_fit
        summary_lines.append(f"HVN scaling:      PPL ≈ {a:.4f} × params^{b:.4f}")
    if tr_fit:
        a, b = tr_fit
        summary_lines.append(f"Transformer scaling: PPL ≈ {a:.4f} × params^{b:.4f}")

    if hvn_fit and tr_fit:
        b_hvn, b_tr = hvn_fit[1], tr_fit[1]
        if b_hvn < b_tr:
            summary_lines.append(
                f"HVN scales BETTER (exponent {b_hvn:.4f} < {b_tr:.4f}) — "
                f"improves faster with size"
            )
        elif b_hvn > b_tr:
            summary_lines.append(
                f"HVN scales WORSE (exponent {b_hvn:.4f} > {b_tr:.4f}) — "
                f"Transformer improves faster with size"
            )
        else:
            summary_lines.append(
                f"Both scale at the same rate (exponent {b_hvn:.4f})"
            )

    # --- Wall-clock crossover ------------------------------------------------
    # Find the d_model where HVN is faster than Transformer (if any)
    crossover_d = None
    crossover_found = False
    for h, t in zip(hvn, tr):
        if h["d_model"] == t["d_model"]:
            if h["total_time_s"] < t["total_time_s"]:
                crossover_d = h["d_model"]
                crossover_found = True
                summary_lines.append(
                    f"Wall-clock crossover at d_model={crossover_d}: "
                    f"HVN {h['total_time_s']:.1f}s < Transformer {t['total_time_s']:.1f}s"
                )
                break

    if not crossover_found:
        # Check if HVN is always slower
        hvn_always_slower = all(
            h["total_time_s"] > t["total_time_s"]
            for h, t in zip(hvn, tr)
            if h["d_model"] == t["d_model"]
        )
        if hvn_always_slower:
            summary_lines.append(
                "No wall-clock crossover found within tested sizes — "
                "HVN is slower at all tested d_model values."
            )
        else:
            summary_lines.append("No direct crossover comparison possible.")

    # --- PPL comparison at each size ----------------------------------------
    summary_lines.append("")
    summary_lines.append("Per-size PPL comparison:")
    for h, t in zip(hvn, tr):
        if h["d_model"] == t["d_model"]:
            ratio = h["final_eval_ppl"] / t["final_eval_ppl"] if t["final_eval_ppl"] > 0 else float("inf")
            summary_lines.append(
                f"  d_model={h['d_model']:>4}: HVN PPL={h['final_eval_ppl']:.2f}  "
                f"Transformer PPL={t['final_eval_ppl']:.2f}  ratio={ratio:.2f}x"
            )

    return {
        "summary_lines": summary_lines,
        "hvn_power_law": list(hvn_fit) if hvn_fit else None,
        "transformer_power_law": list(tr_fit) if tr_fit else None,
        "crossover_d_model": crossover_d,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_scaling_curves(
    results: list[dict],
    analysis: dict,
    output_path: str,
    dataset_name: str,
) -> None:
    """Generate a 3-panel figure: PPL vs params, PPL vs time, time vs params."""
    import numpy as np

    hvn = sorted([r for r in results if r["name"] == "HVN"], key=lambda r: r["params"])
    tr  = sorted([r for r in results if r["name"] == "Transformer"], key=lambda r: r["params"])

    hvn_params = [r["params"] for r in hvn]
    hvn_ppl   = [r["final_eval_ppl"] for r in hvn]
    hvn_time  = [r["total_time_s"] for r in hvn]

    tr_params = [r["params"] for r in tr]
    tr_ppl    = [r["final_eval_ppl"] for r in tr]
    tr_time   = [r["total_time_s"] for r in tr]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"HVN Scaling Laws — {dataset_name}", fontsize=14, fontweight="bold")

    # --- Panel 1: PPL vs Parameters (log-log) --------------------------------
    ax = axes[0]
    if hvn_params:
        ax.loglog(hvn_params, hvn_ppl, "o-", color="tab:blue", label="HVN", linewidth=2, markersize=8)
    if tr_params:
        ax.loglog(tr_params, tr_ppl, "s-", color="tab:orange", label="Transformer", linewidth=2, markersize=8)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Eval Perplexity")
    ax.set_title("PPL vs Parameters (log-log)")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)

    # Annotate power-law exponents
    if analysis.get("hvn_power_law"):
        a, b = analysis["hvn_power_law"]
        ax.annotate(f"HVN: PPL ∝ N^{b:.3f}", xy=(0.05, 0.05), xycoords="axes fraction",
                    fontsize=9, color="tab:blue",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    if analysis.get("transformer_power_law"):
        a, b = analysis["transformer_power_law"]
        ax.annotate(f"Transformer: PPL ∝ N^{b:.3f}", xy=(0.05, 0.15), xycoords="axes fraction",
                    fontsize=9, color="tab:orange",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    # --- Panel 2: PPL vs Wall-clock Time ------------------------------------
    ax = axes[1]
    if hvn_time:
        ax.plot(hvn_time, hvn_ppl, "o-", color="tab:blue", label="HVN", linewidth=2, markersize=8)
    if tr_time:
        ax.plot(tr_time, tr_ppl, "s-", color="tab:orange", label="Transformer", linewidth=2, markersize=8)
    ax.set_xlabel("Wall-clock Training Time (s)")
    ax.set_ylabel("Eval Perplexity")
    ax.set_title("PPL vs Training Time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.invert_yaxis()  # Lower PPL is better — invert so "up" = better

    # --- Panel 3: Wall-clock Time vs Parameters ------------------------------
    ax = axes[2]
    if hvn_params:
        ax.plot(hvn_params, hvn_time, "o-", color="tab:blue", label="HVN", linewidth=2, markersize=8)
    if tr_params:
        ax.plot(tr_params, tr_time, "s-", color="tab:orange", label="Transformer", linewidth=2, markersize=8)
    ax.set_xlabel("Parameters")
    ax.set_ylabel("Wall-clock Training Time (s)")
    ax.set_title("Time vs Parameters (crossover analysis)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Mark crossover if found
    crossover_d = analysis.get("crossover_d_model")
    if crossover_d:
        ax.axvline(x=crossover_d, color="green", linestyle="--", alpha=0.5)
        ax.annotate(f"crossover\nd_model={crossover_d}", xy=(crossover_d, ax.get_ylim()[0]),
                    fontsize=9, color="green", ha="center",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="HVN Scaling Laws Evaluation — train at multiple model sizes and plot scaling curves.",
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        choices=list(DATASET_REGISTRY.keys()),
        help="Dataset to train on.",
    )
    parser.add_argument("--local-path", type=str, default=None,
                        help="Local path to dataset file (if not using download).")
    parser.add_argument(
        "--d_models", type=str, default="64,128,256,512",
        help="Comma-separated list of d_model values to test.",
    )
    parser.add_argument("--epochs", type=int, default=20,
                        help="Training epochs per model size.")
    parser.add_argument("--n_layers", type=int, default=2,
                        help="Number of layers for all models.")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="Training batch size.")
    parser.add_argument("--seq_len", type=int, default=64,
                        help="Sequence length for training samples.")
    parser.add_argument("--lr", type=float, default=3e-4,
                        help="Learning rate.")
    parser.add_argument("--output-json", type=str, default=None,
                        help="Path to save JSON results.")
    parser.add_argument("--output-plot", type=str, default=None,
                        help="Path to save scaling-curve plot (PNG).")
    parser.add_argument("--compile", action="store_true",
                        help="Use torch.compile for faster training.")
    parser.add_argument(
        "--models", type=str, nargs="+", default=None,
        choices=["HVN", "Transformer"],
        help="Which models to compare (default: both).",
    )
    parser.add_argument(
        "--precision", type=str, default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Mixed-precision training mode (fp16/bf16 require CUDA).",
    )
    parser.add_argument(
        "--grad_accum", type=int, default=1,
        help="Gradient accumulation steps (effective batch size = batch_size × grad_accum).",
    )
    # --- Issue #21 quality strategies ---
    parser.add_argument("--rank_k", type=int, default=1, help="Rank-k entanglement update (Strategy 1). 1=rank-1 (original), 4 or 8 recommended.")
    parser.add_argument("--nonlinear_update", action="store_true", help="Enable nonlinear gated Hamiltonian update (Strategy 4)")
    parser.add_argument("--causal_reinjection", action="store_true", help="Enable multi-chunk causal re-injection (Strategy 2)")
    parser.add_argument("--cross_head_mixing", action="store_true", help="Enable cross-head Hamiltonian mixing (Strategy 3)")
    parser.add_argument("--swiglu", action="store_true", help="Use SwiGLU FFN instead of GELU FFN (Strategy 6)")
    parser.add_argument("--per_token_eta", action="store_true", help="Use per-token entanglement rate instead of chunk-averaged (Strategy 5)")
    # --- Issue #22: Multi-thread Hamiltonian memory ---
    parser.add_argument("--n_threads", type=int, default=1, help="Number of independent Hamiltonian threads per head (Issue #22). 1=single Hamiltonian (backward-compatible), 4 recommended for better quality.")
    args = parser.parse_args()

    d_models = [int(x.strip()) for x in args.d_models.split(",")]

    output_json = args.output_json
    output_plot = args.output_plot

    # Default output paths if neither is specified
    if output_json is None and output_plot is None:
        output_json = "results/scaling_laws.json"
        output_plot = "results/scaling_laws.png"

    run_scaling_laws(
        dataset_name=args.dataset,
        local_path=args.local_path,
        d_models=d_models,
        epochs=args.epochs,
        n_layers=args.n_layers,
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        lr=args.lr,
        output_json=output_json,
        output_plot=output_plot,
        compile_model=args.compile,
        models=args.models,
        precision=args.precision,
        grad_accum_steps=args.grad_accum,
        rank_k=args.rank_k,
        use_nonlinear_update=args.nonlinear_update,
        use_causal_reinjection=args.causal_reinjection,
        use_cross_head_mixing=args.cross_head_mixing,
        use_swiglu=args.swiglu,
        use_per_token_eta=args.per_token_eta,
        n_threads=args.n_threads,
    )


if __name__ == "__main__":
    main()