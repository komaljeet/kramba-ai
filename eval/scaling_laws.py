#!/usr/bin/env python
"""
Scaling Laws Evaluation for the Hamiltonian Vortex Network (HVN).

Trains HVN, a Transformer baseline, and an SSM baseline at multiple model sizes
(d_model = {128, 256, 512, 768}) and plots scaling curves:

  1. PPL vs. parameter count (log-log) + fitted power-law exponents
  2. PPL vs. wall-clock training time
  3. Wall-clock time vs. parameter count (crossover-point analysis)

Issue #35 — Real-scale scaling
------------------------------
The original flat exponent (PPL ∝ params^-0.007) was fit on ~34K characters of
Tiny-Shakespeare, deep in the overfit/saturation regime. To measure architectural
scaling rather than regularization noise, this harness supports:

  * ``--dataset wikitext103`` — a ~103M-token corpus (streamed, char-capped).
  * ``--tokenizer bpe`` — subword tokenization (the regime real LMs are fit in).
  * ``--token_budget`` — **compute-matched** training: every model size sees the
    same number of tokens, rather than the same number of epochs. This is the
    correct way to compare sizes (a big model must not be starved relative to a
    small one).
  * Full held-out eval PPL (averaged over the whole valid split in ``seq_len``
    chunks), not a single mid-text batch.
  * A per-model eval-PPL trajectory vs. tokens-seen, from which we report the
    **saturation token budget** (where each model stops improving).

Usage
-----
    # Real-scale run (recommended): download+unzip WikiText-103, then:
    python -m eval.scaling_laws --dataset wikitext103 --local-path /path/to/wikitext-103 \
        --tokenizer bpe --bpe_vocab 8000 --token_budget 8_000_000 \
        --d_models 128,256,512,768 --precision bf16 --compile \
        --output-json results/scaling_wt103.json --output-plot results/scaling_wt103.png

    # Fast smoke test on the bundled corpus:
    python -m eval.scaling_laws --dataset shakespeare --token_budget 500_000 \
        --d_models 128,256 --models HVN Transformer SSM
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from itertools import cycle

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

from benchmark import TransformerLM, SimpleSSMLM, count_parameters  # noqa: E402
from hvn.core import HamiltonianVortexNetwork  # noqa: E402
from hvn.core.moe import collect_moe_aux_loss  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model  # noqa: E402
from train import CharTokenizer, TextDataset  # noqa: E402
from .datasets import DATASET_REGISTRY  # noqa: E402

_MODEL_COLORS = {
    "HVN": "tab:blue", "HVN-DeltaNet": "tab:red",
    "HVN-DeltaNet-Weightless": "tab:purple",
    "Transformer": "tab:orange", "SSM": "tab:green",
}
_MODEL_MARKERS = {
    "HVN": "o", "HVN-DeltaNet": "D", "HVN-DeltaNet-Weightless": "v",
    "Transformer": "s", "SSM": "^",
}

# Models that are HVN variants (used by the analysis to compare against the baselines).
_HVN_VARIANTS = ("HVN", "HVN-DeltaNet", "HVN-DeltaNet-Weightless")


# ---------------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------------

def _build_tokenizer(kind: str, train_text: str, eval_text: str, bpe_vocab: int):
    """Return a tokenizer exposing ``.encode`` / ``.vocab_size``."""
    if kind == "char":
        return CharTokenizer(train_text + eval_text, add_tool_tokens=False)
    if kind == "bpe":
        from hvn.tokenizer import BPETokenizer
        print(f"  Training BPE tokenizer (vocab={bpe_vocab}) on the train split ...")
        # Train BPE on train text only (avoid leaking eval statistics into the vocab).
        return BPETokenizer.train(train_text, vocab_size=bpe_vocab)
    raise ValueError(f"Unknown tokenizer: {kind!r}")


# ---------------------------------------------------------------------------
# Compute-matched training (token budget, not epochs)
# ---------------------------------------------------------------------------

def _evaluate_ppl(
    model: nn.Module,
    eval_ids: list[int],
    seq_len: int,
    batch_size: int,
    device: torch.device,
    use_amp: bool,
    amp_dtype: torch.dtype,
    max_eval_batches: int | None = None,
) -> float:
    """Token-averaged held-out perplexity over the whole eval split in seq_len chunks."""
    criterion = nn.CrossEntropyLoss(reduction="sum")
    n_chunks = max(1, (len(eval_ids) - 1) // seq_len)

    was_training = model.training
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    batch_x: list[list[int]] = []
    batch_y: list[list[int]] = []
    n_batches = 0

    def _flush() -> tuple[float, int]:
        nonlocal batch_x, batch_y
        if not batch_x:
            return 0.0, 0
        x = torch.tensor(batch_x, dtype=torch.long, device=device)
        y = torch.tensor(batch_y, dtype=torch.long, device=device)
        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                logits = model(x)
                loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        batch_x, batch_y = [], []
        return loss.item(), y.numel()

    with torch.no_grad():
        for c in range(n_chunks):
            start = c * seq_len
            chunk = eval_ids[start:start + seq_len + 1]
            if len(chunk) < seq_len + 1:
                break
            batch_x.append(chunk[:-1])
            batch_y.append(chunk[1:])
            if len(batch_x) == batch_size:
                loss_sum, ntok = _flush()
                total_loss += loss_sum
                total_tokens += ntok
                n_batches += 1
                if max_eval_batches is not None and n_batches >= max_eval_batches:
                    break
        loss_sum, ntok = _flush()
        total_loss += loss_sum
        total_tokens += ntok

    if was_training:
        model.train()

    mean_loss = total_loss / max(total_tokens, 1)
    return math.exp(min(mean_loss, 20))


def train_compute_matched(
    model: nn.Module,
    name: str,
    train_ids: list[int],
    eval_ids: list[int],
    token_budget: int,
    seq_len: int,
    batch_size: int,
    lr: float,
    device: torch.device,
    eval_interval_tokens: int,
    precision: str = "fp32",
    compile_model: bool = False,
    max_eval_batches: int | None = None,
) -> dict:
    """Train *model* on exactly ``token_budget`` target tokens, tracking eval PPL.

    Returns a result dict with the final PPL, wall-clock time, and the
    (tokens_seen, eval_ppl) trajectory used for saturation analysis.
    """
    if compile_model and device.type == "cuda":
        print(f"  Compiling {name} with torch.compile ...")
        model = torch.compile(model, mode="reduce-overhead")

    model.to(device)
    model.train()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    use_amp = precision in ("fp16", "bf16") and device.type == "cuda"
    amp_dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    scaler = torch.amp.GradScaler("cuda") if (use_amp and precision == "fp16") else None

    tokens_per_batch = batch_size * seq_len
    total_steps = max(1, token_budget // tokens_per_batch)
    # Cosine schedule over the compute-matched horizon (in optimizer steps).
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

    train_dataset = TextDataset(train_ids, seq_len=seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"\n  Training {name} ({param_count:,} params) — "
          f"budget {token_budget:,} tok = {total_steps:,} steps of {tokens_per_batch:,} tok")

    trajectory: list[dict] = []
    tokens_seen = 0
    next_eval_at = eval_interval_tokens
    t0 = time.time()
    # Loss-spike guard threshold (Issue #44): grad clipping is to 1.0, so a
    # healthy pre-clip norm is O(1–10); a spike is orders of magnitude larger.
    spike_grad_norm = 1e3
    skipped_steps = 0

    batch_iter = cycle(train_loader)
    for step in range(total_steps):
        batch_x, batch_y = next(batch_iter)
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
            logits = model(batch_x)
            loss = criterion(logits.reshape(-1, model.vocab_size), batch_y.reshape(-1))
            # Issue #123: MoE needs its load-balancing term in the objective or
            # routing collapses onto a handful of experts and the remaining
            # capacity is dead weight. Exactly zero for non-MoE models.
            loss = loss + collect_moe_aux_loss(model)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # Loss-spike guard (Issue #44): under bf16 at large width the
            # Gated-DeltaNet block can throw an occasional loss/grad spike; a
            # single non-finite step otherwise poisons the weights *and* the
            # AdamW moments, turning a transient spike into permanent NaN.  Skip
            # the step when the loss is non-finite or the grad norm explodes
            # (clip_grad_norm_ returns the pre-clip total norm), leaving the
            # optimizer state untouched so training recovers.
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if torch.isfinite(loss) and torch.isfinite(grad_norm) and grad_norm < spike_grad_norm:
                optimizer.step()
            else:
                skipped_steps += 1
                optimizer.zero_grad(set_to_none=True)
        scheduler.step()

        tokens_seen += tokens_per_batch
        if tokens_seen >= next_eval_at or step == total_steps - 1:
            ppl = _evaluate_ppl(model, eval_ids, seq_len, batch_size, device,
                                use_amp, amp_dtype, max_eval_batches)
            elapsed = time.time() - t0
            trajectory.append({
                "tokens_seen": tokens_seen,
                "eval_ppl": ppl,
                "train_loss": loss.item(),
                "time_s": elapsed,
            })
            skip_note = f"  |  skipped={skipped_steps}" if skipped_steps else ""
            print(f"    {tokens_seen:>12,} tok  |  train_loss={loss.item():.4f}  |  "
                  f"eval_ppl={ppl:.3f}  |  {elapsed:.1f}s{skip_note}")
            next_eval_at += eval_interval_tokens

    total_time = time.time() - t0
    final_ppl = trajectory[-1]["eval_ppl"] if trajectory else float("inf")
    saturation = _saturation_tokens(trajectory)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "name": name,
        "params": param_count,
        "final_eval_ppl": final_ppl,
        "total_time_s": total_time,
        "tokens_seen": tokens_seen,
        "trajectory": trajectory,
        "saturation_tokens": saturation,
        "skipped_steps": skipped_steps,
    }


def _saturation_tokens(trajectory: list[dict], rel_threshold: float = 0.01) -> int | None:
    """Token budget beyond which eval PPL improves by < ``rel_threshold`` of the
    total improvement — i.e. where the model stops meaningfully learning.

    Returns the tokens_seen at the earliest trajectory point whose PPL is within
    ``rel_threshold`` (relative) of the best (final-or-better) PPL, or None if the
    curve is still improving at the end.
    """
    if len(trajectory) < 2:
        return None
    best = min(t["eval_ppl"] for t in trajectory)
    for t in trajectory:
        if t["eval_ppl"] <= best * (1.0 + rel_threshold):
            # Only call it "saturated" if this happens before the final point.
            if t["tokens_seen"] < trajectory[-1]["tokens_seen"]:
                return t["tokens_seen"]
            return None
    return None


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def run_scaling_laws(
    dataset_name: str,
    local_path: str | None = None,
    d_models: list[int] | None = None,
    token_budget: int = 8_000_000,
    eval_interval_tokens: int | None = None,
    n_layers: int = 2,
    batch_size: int = 16,
    seq_len: int = 128,
    lr: float = 3e-4,
    output_json: str | None = None,
    output_plot: str | None = None,
    compile_model: bool = False,
    models: list[str] | None = None,
    precision: str = "fp32",
    tokenizer_kind: str = "char",
    bpe_vocab: int = 8000,
    max_train_chars: int | None = None,
    max_eval_chars: int | None = None,
    max_eval_batches: int | None = None,
    use_deltanet: bool = False,
    grad_checkpoint: bool = False,
) -> dict:
    """Train HVN / Transformer / SSM at multiple d_model sizes under a fixed
    per-model token budget (compute-matched) and collect scaling data."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available — running on CPU. This will be very slow.")
    else:
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")

    if d_models is None:
        d_models = [128, 256, 512, 768]
    if models is None:
        models = ["HVN", "Transformer", "SSM"]
    if eval_interval_tokens is None:
        eval_interval_tokens = max(token_budget // 10, 1)

    # --- Load dataset -------------------------------------------------------
    print(f"\n{'='*70}\nLoading dataset: {dataset_name}\n{'='*70}")
    loader_fn = DATASET_REGISTRY[dataset_name]
    if dataset_name == "wikitext103":
        train_text, eval_text = loader_fn(local_path, max_train_chars, max_eval_chars)
    else:
        train_text, eval_text = loader_fn(local_path)
        if max_train_chars is not None:
            train_text = train_text[:max_train_chars]
        if max_eval_chars is not None:
            eval_text = eval_text[:max_eval_chars]

    tokenizer = _build_tokenizer(tokenizer_kind, train_text, eval_text, bpe_vocab)
    vocab_size = tokenizer.vocab_size

    train_ids = tokenizer.encode(train_text)
    eval_ids = tokenizer.encode(eval_text)

    print(f"  Tokenizer:        {tokenizer_kind} (vocab {vocab_size:,})")
    print(f"  Train chars:      {len(train_text):,}  ->  {len(train_ids):,} tokens")
    print(f"  Eval chars:       {len(eval_text):,}  ->  {len(eval_ids):,} tokens")
    print(f"  d_models:         {d_models}")
    print(f"  Models:           {models}")
    print(f"  Token budget:     {token_budget:,} / model  (compute-matched)")
    print(f"  Eval interval:    {eval_interval_tokens:,} tokens")
    if len(train_ids) < token_budget:
        passes = token_budget / max(len(train_ids), 1)
        print(f"  NOTE: budget > corpus ({passes:.1f} passes over train tokens).")

    # --- Run scaling sweep --------------------------------------------------
    all_results: list[dict] = []

    for d_model in d_models:
        print(f"\n{'='*70}\nd_model = {d_model}\n{'='*70}")
        bond_dim = max(d_model // 4, 8)

        model_specs: dict[str, nn.Module] = {}
        if "HVN" in models:
            model_specs["HVN"] = HamiltonianVortexNetwork(
                vocab_size=vocab_size, d_model=d_model, num_layers=n_layers,
                bond_dimension=bond_dim, chunk_size=64, dropout=0.1,
                use_deltanet=use_deltanet, grad_checkpoint=grad_checkpoint,
            )
        if "HVN-DeltaNet" in models:
            # Issue #39: sweep the Gated-DeltaNet block (Issue #34) alongside the
            # default HVN block so the two curves are directly comparable.
            model_specs["HVN-DeltaNet"] = HamiltonianVortexNetwork(
                vocab_size=vocab_size, d_model=d_model, num_layers=n_layers,
                bond_dimension=bond_dim, chunk_size=64, dropout=0.1,
                use_deltanet=True, grad_checkpoint=grad_checkpoint,
            )
        if "HVN-DeltaNet-Weightless" in models:
            # Issue #40 / #45: the hybrid weightless DeltaNet — freeze the four
            # d×d q/k/v/out projections as seed-regenerated random features (one
            # int on disk each) while keeping the learned gates/conv. Included so
            # the final pass quantifies its real-scale PPL cost vs dense DeltaNet
            # (previously only measured at d=128 on tiny-Shakespeare, §8.8).
            model_specs["HVN-DeltaNet-Weightless"] = HamiltonianVortexNetwork(
                vocab_size=vocab_size, d_model=d_model, num_layers=n_layers,
                bond_dimension=bond_dim, chunk_size=64, dropout=0.1,
                use_deltanet=True, weightless=True, grad_checkpoint=grad_checkpoint,
            )
        # --- Issue #123 2x2: {dense, shared} x {dense FFN, MoE FFN} ---------
        # Sharing cuts stored params (~3.67x at d=512/12L); MoE buys capacity at
        # sub-linear compute (3.79x params at 1.40x FLOPs). Neither has been run
        # with the other. All four arms are EdgeHVN+DeltaNet so the only
        # variables are the two levers.
        _edge_arms = {
            "EdgeHVN-DeltaNet":            dict(shared_weights=False, moe_experts=1),
            "EdgeHVN-DeltaNet-Shared":     dict(shared_weights=True,  moe_experts=1),
            "EdgeHVN-DeltaNet-MoE":        dict(shared_weights=False, moe_experts=8),
            "EdgeHVN-DeltaNet-Shared-MoE": dict(shared_weights=True,  moe_experts=8),
        }
        for _arm, _cfg in _edge_arms.items():
            if _arm in models:
                model_specs[_arm] = EdgeHVN(
                    vocab_size=vocab_size, d_model=d_model, num_layers=n_layers,
                    n_heads=4, dropout=0.1, use_deltanet=True,
                    grad_checkpoint=grad_checkpoint, moe_top_k=2, **_cfg,
                )

        # --- Ternary (BitNet b1.58) arms, incl. composed with sharing ---
        # Sharing and ternary are ORTHOGONAL levers: sharing cuts the NUMBER of
        # weights (~3.67x at d=512/12L), ternary cuts BYTES PER weight (fp32 ->
        # 1.58 bits, ~20x). Their storage benefits multiply. Never previously
        # combined anywhere in this repo -- these arms test whether their QUALITY
        # costs also compose cleanly, or compound worse than the product.
        #
        # Reason to suspect worse: a shared block's weights are reused at every
        # one of the 12 virtual layers, so quantization error in that single block
        # is applied 12 times rather than once. That is a real mechanism for
        # super-additive damage, and it is the point of measuring rather than
        # assuming.
        #
        # NOTE on param counts: TernaryLinear keeps a full-precision shadow weight
        # for the STE gradient during training, so `param_count` is UNCHANGED for
        # these arms -- the storage win only materialises after `.freeze()`. Read
        # the ternary arms' size benefit as arithmetic (1.58/32 bits), and read
        # these runs for QUALITY cost only.
        _ternary_arms = {
            "EdgeHVN-DeltaNet-Ternary": False,
            "EdgeHVN-DeltaNet-Shared-Ternary": True,
        }
        for _arm, _shared in _ternary_arms.items():
            if _arm in models:
                _m = EdgeHVN(
                    vocab_size=vocab_size, d_model=d_model, num_layers=n_layers,
                    n_heads=4, dropout=0.1, use_deltanet=True, shared_weights=_shared,
                    grad_checkpoint=grad_checkpoint,
                )
                # Same exclusion set the trained checkpoints use
                # (train_edge_conversational.load_checkpoint_for_resume): the tiny
                # gate projections and the output decoder stay full precision.
                _excl = ("beta_proj", "alpha_proj", "decoder")
                _targets = [n for n, mod in _m.named_modules()
                            if isinstance(mod, nn.Linear) and not any(p in n for p in _excl)]
                apply_ternary_to_model(_m, target_names=_targets)
                model_specs[_arm] = _m

        if "Transformer" in models:
            model_specs["Transformer"] = TransformerLM(
                vocab_size=vocab_size, d_model=d_model, n_layers=n_layers, n_heads=4,
            )
        if "SSM" in models:
            model_specs["SSM"] = SimpleSSMLM(
                vocab_size=vocab_size, d_model=d_model, n_layers=n_layers,
            )

        for name, model in model_specs.items():
            result = train_compute_matched(
                model=model, name=name, train_ids=train_ids, eval_ids=eval_ids,
                token_budget=token_budget, seq_len=seq_len, batch_size=batch_size,
                lr=lr, device=device, eval_interval_tokens=eval_interval_tokens,
                precision=precision, compile_model=compile_model,
                max_eval_batches=max_eval_batches,
            )
            result["d_model"] = d_model
            result["n_layers"] = n_layers
            all_results.append(result)
            sat = result["saturation_tokens"]
            sat_str = f"{sat:,} tok" if sat else "not saturated"
            print(f"    -> {name}: PPL {result['final_eval_ppl']:.3f}  |  "
                  f"{result['total_time_s']:.1f}s  |  saturates: {sat_str}")

    # --- Summary table ------------------------------------------------------
    print(f"\n{'='*70}\nScaling Laws Summary\n{'='*70}")
    print(f"{'Model':<14}{'d_model':>8}{'Params':>12}{'Eval PPL':>11}{'Time':>10}{'Saturates':>16}")
    print("-" * 71)
    for r in all_results:
        sat = r["saturation_tokens"]
        sat_str = f"{sat:,}" if sat else "—"
        print(f"{r['name']:<14}{r['d_model']:>8}{r['params']:>12,}"
              f"{r['final_eval_ppl']:>11.3f}{r['total_time_s']:>9.1f}s{sat_str:>16}")

    # --- Scaling / crossover analysis ---------------------------------------
    analysis = _compute_analysis(all_results)
    print(f"\n{'='*70}\nScaling Analysis\n{'='*70}")
    for line in analysis["summary_lines"]:
        print(f"  {line}")

    # --- Assemble + persist -------------------------------------------------
    full_results = {
        "dataset": dataset_name,
        "dataset_stats": {
            "tokenizer": tokenizer_kind,
            "vocab_size": vocab_size,
            "train_chars": len(train_text),
            "eval_chars": len(eval_text),
            "train_tokens": len(train_ids),
            "eval_tokens": len(eval_ids),
        },
        "hyperparameters": {
            "d_models": d_models, "n_layers": n_layers, "token_budget": token_budget,
            "eval_interval_tokens": eval_interval_tokens, "batch_size": batch_size,
            "seq_len": seq_len, "lr": lr, "precision": precision,
            "tokenizer": tokenizer_kind, "bpe_vocab": bpe_vocab,
            "compile": compile_model, "use_deltanet": use_deltanet,
        },
        "models": all_results,
        "analysis": analysis,
    }

    if output_json:
        os.makedirs(os.path.dirname(os.path.abspath(output_json)), exist_ok=True)
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(full_results, f, indent=2)
        print(f"\nResults saved to {output_json}")

    if output_plot:
        os.makedirs(os.path.dirname(os.path.abspath(output_plot)), exist_ok=True)
        _plot_scaling_curves(all_results, analysis, output_plot, dataset_name)
        print(f"Plots saved to {output_plot}")

    return full_results


# ---------------------------------------------------------------------------
# Analysis: power-law fits (per model) + wall-clock crossover
# ---------------------------------------------------------------------------

def _power_law_fit(data: list[dict]) -> tuple[float, float] | None:
    """Return (a, b) for PPL = a * params^b via log-log least squares, or None."""
    if len(data) < 2:
        return None
    import numpy as np
    log_p = np.array([math.log(r["final_eval_ppl"]) for r in data])
    log_n = np.array([math.log(r["params"]) for r in data])
    n = len(data)
    slope = (n * np.sum(log_p * log_n) - np.sum(log_p) * np.sum(log_n)) / (
        n * np.sum(log_n**2) - np.sum(log_n) ** 2
    )
    intercept = (np.sum(log_p) - slope * np.sum(log_n)) / n
    return (math.exp(intercept), slope)


def _compute_analysis(results: list[dict]) -> dict:
    """Fit PPL~params^b for every model, plus HVN-vs-Transformer wall-clock crossover."""
    model_names = sorted({r["name"] for r in results})
    by_model = {
        name: sorted([r for r in results if r["name"] == name], key=lambda r: r["params"])
        for name in model_names
    }

    summary_lines: list[str] = []
    power_laws: dict[str, list[float] | None] = {}
    for name in model_names:
        fit = _power_law_fit(by_model[name])
        power_laws[name] = list(fit) if fit else None
        if fit:
            a, b = fit
            summary_lines.append(f"{name:<12} scaling:  PPL ≈ {a:.4f} × params^{b:.4f}")

    # Compare each HVN variant's exponent to Transformer/SSM if present.
    for hvn_name in _HVN_VARIANTS:
        hvn_fit = power_laws.get(hvn_name)
        if not hvn_fit:
            continue
        b_hvn = hvn_fit[1]
        for other in ("Transformer", "SSM"):
            of = power_laws.get(other)
            if of:
                verdict = "steeper (better)" if b_hvn < of[1] else "flatter (worse)"
                summary_lines.append(
                    f"{hvn_name} exponent {b_hvn:.4f} vs {other} {of[1]:.4f} — "
                    f"{hvn_name} is {verdict}."
                )

    # Head-to-head: HVN-DeltaNet vs default HVN block (Issue #39 decision input).
    hvn_fit, dn_fit = power_laws.get("HVN"), power_laws.get("HVN-DeltaNet")
    if hvn_fit and dn_fit:
        verdict = "steeper (better)" if dn_fit[1] < hvn_fit[1] else "flatter (worse)"
        summary_lines.append(
            f"HVN-DeltaNet exponent {dn_fit[1]:.4f} vs default HVN {hvn_fit[1]:.4f} — "
            f"DeltaNet is {verdict}."
        )
        by_d = {}
        for r in results:
            if r["name"] in ("HVN", "HVN-DeltaNet"):
                by_d.setdefault(r["d_model"], {})[r["name"]] = r["final_eval_ppl"]
        for d in sorted(by_d):
            pair = by_d[d]
            if "HVN" in pair and "HVN-DeltaNet" in pair:
                base, dn = pair["HVN"], pair["HVN-DeltaNet"]
                pct = 100.0 * (base - dn) / base
                summary_lines.append(
                    f"  d={d:<4}: HVN {base:.2f} -> DeltaNet {dn:.2f} ({pct:+.1f}% PPL)"
                )

    # Saturation summary.
    summary_lines.append("")
    summary_lines.append("Saturation (token budget where eval PPL stops improving):")
    for name in model_names:
        for r in by_model[name]:
            sat = r["saturation_tokens"]
            sat_str = f"{sat:,} tok" if sat else f"still improving at {r['tokens_seen']:,} tok"
            summary_lines.append(f"  {name:<12} d={r['d_model']:<4}: {sat_str}")

    # Wall-clock crossover (HVN vs Transformer).
    crossover_d = None
    hvn, tr = by_model.get("HVN", []), by_model.get("Transformer", [])
    for h, t in zip(hvn, tr):
        if h["d_model"] == t["d_model"] and h["total_time_s"] < t["total_time_s"]:
            crossover_d = h["d_model"]
            summary_lines.append(
                f"Wall-clock crossover at d_model={crossover_d}: "
                f"HVN {h['total_time_s']:.1f}s < Transformer {t['total_time_s']:.1f}s"
            )
            break
    if crossover_d is None and hvn and tr:
        summary_lines.append("No wall-clock crossover within tested sizes.")

    return {
        "summary_lines": summary_lines,
        "power_laws": power_laws,
        "crossover_d_model": crossover_d,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_scaling_curves(results, analysis, output_path, dataset_name) -> None:
    """3-panel figure: PPL vs params (log-log), PPL vs time, time vs params."""
    model_names = sorted({r["name"] for r in results})
    by_model = {
        name: sorted([r for r in results if r["name"] == name], key=lambda r: r["params"])
        for name in model_names
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f"HVN Scaling Laws — {dataset_name}", fontsize=14, fontweight="bold")

    ax = axes[0]
    for i, name in enumerate(model_names):
        rs = by_model[name]
        color = _MODEL_COLORS.get(name, f"C{i}")
        marker = _MODEL_MARKERS.get(name, "o")
        ax.loglog([r["params"] for r in rs], [r["final_eval_ppl"] for r in rs],
                  marker + "-", color=color, label=name, linewidth=2, markersize=8)
    ax.set_xlabel("Parameters"); ax.set_ylabel("Eval Perplexity")
    ax.set_title("PPL vs Parameters (log-log)"); ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    y = 0.05
    for name in model_names:
        pl = analysis["power_laws"].get(name)
        if pl:
            ax.annotate(f"{name}: PPL ∝ N^{pl[1]:.3f}", xy=(0.05, y), xycoords="axes fraction",
                        fontsize=9, color=_MODEL_COLORS.get(name, "black"),
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
            y += 0.09

    ax = axes[1]
    for i, name in enumerate(model_names):
        rs = by_model[name]
        ax.plot([r["total_time_s"] for r in rs], [r["final_eval_ppl"] for r in rs],
                _MODEL_MARKERS.get(name, "o") + "-", color=_MODEL_COLORS.get(name, f"C{i}"),
                label=name, linewidth=2, markersize=8)
    ax.set_xlabel("Wall-clock Training Time (s)"); ax.set_ylabel("Eval Perplexity")
    ax.set_title("PPL vs Training Time"); ax.legend(); ax.grid(True, alpha=0.3)
    ax.invert_yaxis()

    ax = axes[2]
    for i, name in enumerate(model_names):
        rs = by_model[name]
        ax.plot([r["params"] for r in rs], [r["total_time_s"] for r in rs],
                _MODEL_MARKERS.get(name, "o") + "-", color=_MODEL_COLORS.get(name, f"C{i}"),
                label=name, linewidth=2, markersize=8)
    ax.set_xlabel("Parameters"); ax.set_ylabel("Wall-clock Training Time (s)")
    ax.set_title("Time vs Parameters (crossover analysis)"); ax.legend(); ax.grid(True, alpha=0.3)
    crossover_d = analysis.get("crossover_d_model")
    if crossover_d:
        ax.axvline(x=crossover_d, color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int(s: str) -> int:
    """Parse an int allowing underscores/commas (e.g. 8_000_000, 8,000,000)."""
    return int(s.replace("_", "").replace(",", ""))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="HVN Scaling Laws — compute-matched training across model sizes.",
    )
    parser.add_argument("--dataset", type=str, required=True,
                        choices=list(DATASET_REGISTRY.keys()), help="Dataset to train on.")
    parser.add_argument("--local-path", type=str, default=None,
                        help="Local path to dataset file/dir (WikiText-103: the extracted dir).")
    parser.add_argument("--d_models", type=str, default="128,256,512,768",
                        help="Comma-separated d_model values to test.")
    parser.add_argument("--token_budget", type=_parse_int, default=8_000_000,
                        help="Per-model training token budget (compute-matched).")
    parser.add_argument("--eval_interval_tokens", type=_parse_int, default=None,
                        help="Eval PPL every N tokens (default: budget/10).")
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seq_len", type=int, default=128)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--output-json", type=str, default=None)
    parser.add_argument("--output-plot", type=str, default=None)
    parser.add_argument("--compile", action="store_true", help="Use torch.compile.")
    parser.add_argument("--models", type=str, nargs="+", default=None,
                        choices=["HVN", "HVN-DeltaNet", "HVN-DeltaNet-Weightless",
                                 "EdgeHVN-DeltaNet", "EdgeHVN-DeltaNet-Shared",
                                 "EdgeHVN-DeltaNet-MoE", "EdgeHVN-DeltaNet-Shared-MoE",
                                 "EdgeHVN-DeltaNet-Ternary",
                                 "EdgeHVN-DeltaNet-Shared-Ternary",
                                 "Transformer", "SSM"],
                        help="Which models to compare (default: HVN, Transformer, SSM). "
                             "Add 'HVN-DeltaNet' to sweep the Issue #34 block (Issue #39); "
                             "'HVN-DeltaNet-Weightless' for the Issue #40 hybrid; "
                             "'EdgeHVN-DeltaNet' + 'EdgeHVN-DeltaNet-Shared' for the "
                             "ALBERT-style weight-sharing A/B (same class, sharing is the "
                             "only variable -- the fewer-params-same-quality test).")
    parser.add_argument("--precision", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"])
    parser.add_argument("--tokenizer", type=str, default="char", choices=["char", "bpe"],
                        help="Tokenizer: char (default) or bpe (recommended for real-scale).")
    parser.add_argument("--bpe_vocab", type=int, default=8000,
                        help="BPE vocabulary size (when --tokenizer bpe).")
    parser.add_argument("--max_train_chars", type=_parse_int, default=None,
                        help="Cap train-split characters read (streaming / memory bound).")
    parser.add_argument("--max_eval_chars", type=_parse_int, default=None,
                        help="Cap eval-split characters read.")
    parser.add_argument("--max_eval_batches", type=_parse_int, default=None,
                        help="Cap number of eval batches per PPL measurement (speed).")
    parser.add_argument("--use_deltanet", action="store_true",
                        help="Use Gated-DeltaNet HVN blocks (Issue #34) in the sweep.")
    parser.add_argument("--grad_checkpoint", action="store_true",
                        help="Gradient (activation) checkpointing (Issue #49): exact, "
                             "training-only activation-memory saving for the HVN variants so a "
                             "larger d_model fits in 6 GB (~20-30%% slower/step).")
    args = parser.parse_args()

    d_models = [int(x.strip()) for x in args.d_models.split(",")]

    output_json = args.output_json
    output_plot = args.output_plot
    if output_json is None and output_plot is None:
        output_json = "results/scaling_laws.json"
        output_plot = "results/scaling_laws.png"

    run_scaling_laws(
        dataset_name=args.dataset, local_path=args.local_path, d_models=d_models,
        token_budget=args.token_budget, eval_interval_tokens=args.eval_interval_tokens,
        n_layers=args.n_layers, batch_size=args.batch_size, seq_len=args.seq_len,
        lr=args.lr, output_json=output_json, output_plot=output_plot,
        compile_model=args.compile, models=args.models, precision=args.precision,
        tokenizer_kind=args.tokenizer, bpe_vocab=args.bpe_vocab,
        max_train_chars=args.max_train_chars, max_eval_chars=args.max_eval_chars,
        max_eval_batches=args.max_eval_batches, use_deltanet=args.use_deltanet,
        grad_checkpoint=args.grad_checkpoint,
    )


if __name__ == "__main__":
    main()
