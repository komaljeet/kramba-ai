"""
Auto-train orchestrator: staged curriculum scheduler (Issue #54)
================================================================

Milestone 2, Issue D of 5. A single-command scheduler that trains the coding
curriculum **in order**

    HTML → CSS → JS → PHP → Python → DevOps        (the #53 stage order)

continuing from the #51 base checkpoint, evaluating and checkpointing every
stage, and **resuming** cleanly after an interruption (kill / cron / ``/loop``).

What it wires together (no new training loop — it reuses existing pieces)
------------------------------------------------------------------------
* ``eval.curriculum.load_curriculum_stage`` (#53) — per-stage license-filtered,
  deduped, budget-capped ``(train, eval)`` streams (offline-safe fallback).
* ``train.train_model`` (#50 ``--full`` Category-A exact speed levers: fused
  AdamW, per-epoch loss sync, pinned/async H2D) — carries the model **weights**
  forward between stages (a fresh optimizer per stage, standard for a staged
  curriculum).
* A per-stage held-out **bits-per-char / PPL + non-degenerate completion** eval
  (``evaluate_stage``). This is the seam the #55 coding-eval harness plugs into
  — swap/augment ``evaluate_stage`` and the logged ``metrics`` grows richer; the
  orchestration (carry-forward, checkpoint, resume) is unchanged.

Each completed stage writes ``checkpoint_stage_{name}.pt`` (+ its tokenizer) and
appends a metrics row to ``auto_train_metrics.jsonl``. A small JSON state file
(``auto_train_state.json``) records which stages are done and which checkpoint
to carry forward, so a re-run **resumes from the last completed stage** rather
than restarting — safe to drive unattended under cron or ``/loop``.

O(1) inference is preserved: this only orchestrates training; the served model
is exactly what ``train_model`` produces (no KV cache, fixed-size Hamiltonian).

Usage
-----
    # Full run over the whole curriculum, continuing from the #51 checkpoint:
    python auto_train.py --from hvn_chat_checkpoint.pt \
        --stages html,css,js,php,python,devops

    # Resume after a kill (default): re-run the exact same command — completed
    # stages are skipped and the last stage checkpoint is carried forward.
    python auto_train.py --from hvn_chat_checkpoint.pt

    # Start over, ignoring any prior state:
    python auto_train.py --from hvn_chat_checkpoint.pt --restart

    # Fast offline loop-validation (tiny budget/epochs — CI / 6 GB box):
    python auto_train.py --from tiny_base.pt --smoke
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

# Force UTF-8 stdout/stderr so the ✓/→ characters printed here (and by the
# reused train.py loop) don't crash on a legacy cp1252 Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass

import torch
from torch.utils.data import DataLoader

from hvn.core import HamiltonianVortexNetwork
from train import CharTokenizer, TextDataset, train_model
from eval.curriculum import CURRICULUM_STAGES, load_curriculum_stage
# Per-stage scoring lives in the #55 coding-eval harness (eval.coding), imported
# lazily inside evaluate_stage to avoid a circular import.

STATE_VERSION = 1


# ---------------------------------------------------------------------------
# Checkpoint / tokenizer (re)loading — mirrors web/hvn_worker/worker.py so the
# orchestrator rebuilds exactly the block the checkpoint was trained with.
# ---------------------------------------------------------------------------

def build_model_from_ckpt(ckpt: dict) -> HamiltonianVortexNetwork:
    """Rebuild an HVN with the SAME hyperparameters the checkpoint stored.

    In particular ``use_deltanet``/``deltanet_conv_kernel`` (#34) and the
    ``weightless`` projections (#40): the #51 chat checkpoint is a Gated-DeltaNet
    model, so omitting these builds the wrong block and ``load_state_dict`` fails.
    """
    return HamiltonianVortexNetwork(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=ckpt["num_layers"],
        n_heads=ckpt.get("n_heads", 4),
        chunk_size=ckpt.get("chunk_size", 64),
        use_deltanet=ckpt.get("use_deltanet", False),
        deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        weightless=ckpt.get("weightless", False),
        weightless_seed=ckpt.get("weightless_seed", 1234),
    )


def load_checkpoint(path: str, device: torch.device):
    """Load ``(model, tokenizer, ckpt_dict)`` from a checkpoint + its tokenizer.

    Tokenizer auto-detect (same convention as generate.py / the worker): try BPE
    first, fall back to CharTokenizer, both from ``<stem>_tokenizer.json``.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_model_from_ckpt(ckpt)
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    tok_path = path.replace(".pt", "_tokenizer.json")
    tokenizer = _load_tokenizer(tok_path)
    return model, tokenizer, ckpt


def _load_tokenizer(tok_path: str):
    try:
        from hvn.tokenizer import BPETokenizer

        return BPETokenizer.load(tok_path)
    except Exception:
        return CharTokenizer.load(tok_path)


def _stage_checkpoint_path(out_dir: str, stage: str) -> str:
    return os.path.join(out_dir, f"checkpoint_stage_{stage}.pt")


# ---------------------------------------------------------------------------
# Per-stage evaluation (the #55 coding-eval seam)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_stage(
    model: HamiltonianVortexNetwork,
    tokenizer,
    stage: str,
    eval_text: str,
    seq_len: int,
    device: torch.device,
    allow_exec: bool = False,
) -> dict:
    """Score one trained stage on its held-out split (the #55 coding-eval seam).

    Delegates to :func:`eval.coding.score_stage`, which returns tokenization-
    invariant **bits-per-char** + PPL, the **structural completion-validity rate**
    (the per-language #55 ``coding_score``), and — for a language with a runtime
    when ``allow_exec`` is set — a HumanEval-style exec ``pass@k``. ``allow_exec``
    defaults to **False** so unattended curriculum runs stay fast and side-effect-
    free; the exec pass@k is an explicit ``python -m eval.coding`` pass.

    The returned dict is a superset of the original contract (``ppl`` /
    ``bits_per_char`` / ``non_degenerate`` / ``completion_sample``), so the
    per-stage metrics log grows richer without touching the scheduler.
    """
    # Imported here (not at module load) to avoid a circular import: eval.coding
    # is a leaf that reuses build_chat_checkpoint; auto_train depends on it.
    from eval.coding import score_stage

    return score_stage(
        model, tokenizer, stage, eval_text, seq_len, device, allow_exec=allow_exec,
    )


# ---------------------------------------------------------------------------
# Resumable state
# ---------------------------------------------------------------------------

def _load_state(state_path: str) -> dict | None:
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _atomic_write_json(path: str, obj: dict) -> None:
    """Write JSON atomically (temp file + os.replace) so a kill mid-write can't
    corrupt the resume state."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)


def _append_metrics(metrics_path: str, row: dict) -> None:
    with open(metrics_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row) + "\n")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def run_curriculum(
    base_checkpoint: str,
    stages: list[str],
    out_dir: str = ".",
    state_path: str | None = None,
    metrics_path: str | None = None,
    epochs_per_stage: int = 2,
    seq_len: int = 128,
    batch_size: int = 16,
    lr: float = 1e-4,
    precision: str = "bf16",
    max_train_chars: int | None = None,
    max_eval_chars: int | None = None,
    local_dir: str | None = None,
    allow_download: bool = False,
    resume: bool = True,
    device: torch.device | None = None,
    seed: int = 42,
    progress_cb=None,
) -> dict:
    """Train ``stages`` in order from ``base_checkpoint``, one checkpoint each.

    Returns the final state dict. Resumable: on a re-run with ``resume=True`` any
    stage already recorded ``completed`` in the state file is skipped and the last
    stage checkpoint is carried forward instead of the base.

    ``progress_cb``, if given, is called with a small event dict at the run/stage
    boundaries (``{"event": "run_start"|"stage_start"|"stage_complete"|
    "run_complete", ...}``). It is the seam the Issue #57 in-app training task
    uses to stream progress to the UI and to abort cooperatively: a callback that
    raises (e.g. on a user cancel) propagates out and stops the run cleanly after
    the current stage's checkpoint/state have been committed. The CLI passes no
    callback, so this is a no-op for ``python auto_train.py``.
    """
    def _emit(event: dict) -> None:
        if progress_cb is not None:
            progress_cb(event)
    os.makedirs(out_dir, exist_ok=True)
    if state_path is None:
        state_path = os.path.join(out_dir, "auto_train_state.json")
    if metrics_path is None:
        metrics_path = os.path.join(out_dir, "auto_train_metrics.jsonl")
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"

    for stage in stages:
        if stage not in CURRICULUM_STAGES:
            raise ValueError(f"unknown stage {stage!r}; expected {CURRICULUM_STAGES}")

    # --- Resume bookkeeping ------------------------------------------------
    prior = _load_state(state_path) if resume else None
    completed: list[dict] = []
    carry_checkpoint = base_checkpoint
    if prior and prior.get("base") == base_checkpoint:
        completed = prior.get("completed", [])
        if completed:
            carry_checkpoint = prior.get("current_checkpoint") or base_checkpoint
            done = [c["stage"] for c in completed]
            print(f"[resume] {len(done)} stage(s) already done: {', '.join(done)}")
            print(f"[resume] carrying forward: {carry_checkpoint}")
    elif prior and not resume:
        pass

    done_stages = {c["stage"] for c in completed}

    state = {
        "version": STATE_VERSION,
        "base": base_checkpoint,
        "stages": stages,
        "config": {
            "epochs_per_stage": epochs_per_stage,
            "seq_len": seq_len,
            "batch_size": batch_size,
            "lr": lr,
            "precision": precision,
            "local_dir": local_dir,
        },
        "completed": completed,
        "current_checkpoint": carry_checkpoint,
    }
    _atomic_write_json(state_path, state)

    todo = [s for s in stages if s not in done_stages]
    _emit({"event": "run_start", "stages": stages, "done": sorted(done_stages),
           "todo": todo, "base": base_checkpoint})

    for stage in stages:
        if stage in done_stages:
            print(f"\n=== stage '{stage}' already complete — skipping ===")
            continue

        # Cancel/stream seam: emitted BEFORE the expensive stage starts so a
        # cooperative cancel (callback raises) aborts without burning GPU on a
        # stage nobody's waiting for.
        _emit({"event": "stage_start", "stage": stage,
               "index": len(completed), "total": len(stages),
               "from": carry_checkpoint})

        print(f"\n{'=' * 64}\n=== stage '{stage}'  (from {carry_checkpoint}) ===\n{'=' * 64}")
        t0 = time.time()

        # 1. Load the carried-forward model + tokenizer.
        model, tokenizer, _ckpt = load_checkpoint(carry_checkpoint, device)

        # 2. Load this stage's (train, eval) streams (#53 loader).
        train_text, eval_text = load_curriculum_stage(
            stage, local_dir=local_dir, allow_download=allow_download,
            max_train_chars=max_train_chars, max_eval_chars=max_eval_chars, seed=seed,
        )

        # 3. Tokenize with the CARRIED tokenizer (vocab is fixed by the base
        #    checkpoint — BPE handles new code subwords via its byte fallback).
        train_ids = tokenizer.encode(train_text)
        ds = TextDataset(train_ids, seq_len=seq_len)
        loader = DataLoader(
            ds, batch_size=batch_size, shuffle=True, drop_last=False,
            pin_memory=use_cuda,
        )
        print(f"  train tokens: {len(train_ids):,} | batches/epoch: {len(loader)}")

        # 4. Train this stage, carrying the model weights forward. Saves the
        #    stage checkpoint (train_model writes save_path itself).
        stage_ckpt = _stage_checkpoint_path(out_dir, stage)
        train_model(
            model, loader, epochs=epochs_per_stage, lr=lr, device=device,
            save_path=stage_ckpt, precision=precision,
            fused_optimizer=use_cuda, defer_loss_sync=True, non_blocking=use_cuda,
        )
        # Save the tokenizer alongside so each stage checkpoint is independently
        # servable (worker/generate auto-detect <stem>_tokenizer.json).
        tokenizer.save(stage_ckpt.replace(".pt", "_tokenizer.json"))

        # 5. Evaluate the trained stage (the #55 seam).
        metrics = evaluate_stage(model, tokenizer, stage, eval_text, seq_len, device)
        elapsed = time.time() - t0

        row = {
            "stage": stage,
            "checkpoint": stage_ckpt,
            "from": carry_checkpoint,
            "epochs": epochs_per_stage,
            "train_chars": len(train_text),
            "eval_chars": len(eval_text),
            "train_tokens": len(train_ids),
            "seconds": round(elapsed, 1),
            "metrics": metrics,
        }
        _append_metrics(metrics_path, row)
        print(f"  ✓ stage '{stage}' done in {elapsed:.1f}s | "
              f"PPL {metrics['ppl']} | {metrics['bits_per_char']} bits/char | "
              f"coding_score={metrics.get('coding_score')} "
              f"({metrics.get('metric_kind')}) | "
              f"non_degenerate={metrics['non_degenerate']}")
        print(f"  ✓ checkpoint: {stage_ckpt}")

        # 6. Commit progress to the resume state (atomic) and carry forward.
        completed.append(row)
        carry_checkpoint = stage_ckpt
        state["completed"] = completed
        state["current_checkpoint"] = carry_checkpoint
        _atomic_write_json(state_path, state)

        _emit({"event": "stage_complete", "stage": stage,
               "index": len(completed) - 1, "total": len(stages),
               "checkpoint": stage_ckpt, "metrics": metrics,
               "seconds": round(elapsed, 1)})

    _emit({"event": "run_complete", "completed": len(completed),
           "total": len(stages), "final_checkpoint": state["current_checkpoint"]})

    print(f"\n{'=' * 64}")
    print(f"Curriculum complete: {len(completed)}/{len(stages)} stage(s).")
    print(f"  state:   {state_path}")
    print(f"  metrics: {metrics_path}")
    print(f"  final checkpoint: {state['current_checkpoint']}")
    print(f"{'=' * 64}")
    return state


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Staged curriculum auto-train orchestrator (Issue #54)")
    p.add_argument("--from", dest="base", type=str, default="hvn_chat_checkpoint.pt",
                   help="Base checkpoint to continue from (the #51 pipeline output)")
    p.add_argument("--stages", type=str, default=",".join(CURRICULUM_STAGES),
                   help="Comma-separated curriculum stages, in order "
                        f"(default: {','.join(CURRICULUM_STAGES)})")
    p.add_argument("--out-dir", type=str, default=".",
                   help="Directory for checkpoint_stage_*.pt, state and metrics")
    p.add_argument("--state", type=str, default=None,
                   help="Resume-state JSON path (default <out-dir>/auto_train_state.json)")
    p.add_argument("--metrics", type=str, default=None,
                   help="Metrics JSONL path (default <out-dir>/auto_train_metrics.jsonl)")
    p.add_argument("--epochs-per-stage", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--max-train-chars", type=int, default=None,
                   help="Override the per-stage train char budget (#53 CURRICULUM_BUDGETS)")
    p.add_argument("--max-eval-chars", type=int, default=None,
                   help="Override the per-stage eval char budget")
    p.add_argument("--local-dir", type=str, default=None,
                   help="Directory of pre-fetched permissive code for the stages (#53)")
    p.add_argument("--download", action="store_true",
                   help="Allow best-effort curated permissive downloads (#53)")
    p.add_argument("--restart", action="store_true",
                   help="Ignore any prior state and start the curriculum from scratch")
    p.add_argument("--smoke", action="store_true",
                   help="Fast offline loop-validation: tiny budget/epochs (CI / 6 GB box)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    stages = [s.strip() for s in args.stages.split(",") if s.strip()]

    max_train_chars = args.max_train_chars
    max_eval_chars = args.max_eval_chars
    epochs = args.epochs_per_stage
    seq_len = args.seq_len
    allow_download = args.download

    if args.smoke:
        # Tiny everything — just exercise the full staged loop end-to-end offline.
        max_train_chars = max_train_chars or 60_000
        max_eval_chars = max_eval_chars or 6_000
        epochs = 1
        seq_len = min(seq_len, 64)
        allow_download = False
        print("[smoke] fast offline loop-validation (not a quality run)")

    if not os.path.exists(args.base):
        print(f"✗ base checkpoint not found: {args.base}\n"
              f"  Build one first with build_chat_checkpoint.py (Issue #51).")
        return 2

    torch.manual_seed(args.seed)
    run_curriculum(
        base_checkpoint=args.base,
        stages=stages,
        out_dir=args.out_dir,
        state_path=args.state,
        metrics_path=args.metrics,
        epochs_per_stage=epochs,
        seq_len=seq_len,
        batch_size=args.batch_size,
        lr=args.lr,
        precision=args.precision,
        max_train_chars=max_train_chars,
        max_eval_chars=max_eval_chars,
        local_dir=args.local_dir,
        allow_download=allow_download,
        resume=not args.restart,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
