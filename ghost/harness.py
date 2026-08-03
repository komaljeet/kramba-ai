"""GHOST Phase 2 — GPU harness: dense checkpoint series + prediction scoring.

Written by: Rambarun Komaljeet

``ghost/extrapolator.py`` is pure tensor arithmetic. This module supplies the
two things it needs from the real system, and nothing else:

1. ``--make-series`` — continue training an existing EdgeHVN checkpoint and
   save the weights every ``--save-every`` steps. The repo's own checkpoints
   are spaced 500 steps apart, which is far coarser than the N=10..500 jumps
   #126 scores; without a dense series there is no "real trained weights at the
   same step" to compare a prediction against at N=10 at all.

2. ``--score`` — for each N, evaluate three sets of weights on ONE fixed
   held-out probe batch set:

       real        the actually-trained checkpoint at step S+N   (the target)
       predicted   extrapolator output from checkpoints at S-2g, S-g, S
       frozen      the weights at S, unchanged                   (NULL CONTROL)

   The null control is binding, per #126. Over a short horizon the frozen
   weights will also land within 5% of real, so "predicted is within 5%" alone
   proves nothing — the prediction has to beat doing nothing.

Optimizer-state policy
----------------------
This codebase's ``_save_edge_checkpoint`` stores weights and architecture only,
never AdamW moments, so resuming from any checkpoint here already restarts the
optimizer state from zero. GHOST inherits that: a jump carries no moment state
and none is extrapolated. Recorded explicitly because #126 asks for the policy
to be stated, and because it is a real (if pre-existing) cost of every jump —
the first steps after one run with cold moments.

Usage
-----
    python -m ghost.harness --make-series --base <ckpt.pt> --steps 600 \
        --save-every 10 --out-dir results/ghost/series
    python -m ghost.harness --score --series-dir results/ghost/series \
        --out results/ghost/extrapolation_scores.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghost.extrapolator import ExtrapolationConfig, build_plan, extrapolate  # noqa: E402

GPT2_VOCAB_SIZE = 50257


# --------------------------------------------------------------------------
# Shared plumbing
# --------------------------------------------------------------------------


def require_cuda() -> torch.device:
    """This project trains and evaluates on GPU, never CPU (docs/ai_rules.md)."""
    if not torch.cuda.is_available():
        raise SystemExit(
            "CUDA is not available. This project requires GPU for all training, "
            "benchmarking and evaluation — fix the torch install rather than "
            "falling back to CPU (docs/ai_rules.md)."
        )
    return torch.device("cuda")


def build_tokenizer():
    from transformers import GPT2TokenizerFast
    return GPT2TokenizerFast.from_pretrained("gpt2")


def load_probe_batches(
    tokenizer,
    n_batches: int,
    batch_size: int,
    seq_len: int,
    data_dir: str,
    seed: int = 1234,
) -> List[torch.Tensor]:
    """A FIXED held-out probe set, identical for every arm and every N.

    Single-batch loss has batch-to-batch variance comparable to the effect
    being measured, so every comparison in this phase runs on the same
    ``n_batches`` windows, drawn once with a fixed seed.
    """
    import train_edge_conversational as tec

    torch.manual_seed(seed)
    loader, _val, _stats = tec.build_dataset_pipeline(
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        data_dir=data_dir,
        val_fraction=0.0,
        num_workers=0,
    )
    batches: List[torch.Tensor] = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batches.append(batch[0] if isinstance(batch, (tuple, list)) else batch)
    if not batches:
        raise RuntimeError("probe loader yielded no batches")
    return batches


@torch.no_grad()
def evaluate_loss(model, batches: Sequence[torch.Tensor], device: torch.device) -> float:
    """Mean next-token cross-entropy over the fixed probe batches."""
    model.eval()
    total, count = 0.0, 0
    for xb in batches:
        xb = xb.to(device)
        x, y = xb[:, :-1], xb[:, 1:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, GPT2_VOCAB_SIZE).float(), y.reshape(-1)
            )
        total += float(loss)
        count += 1
    return total / max(count, 1)


# --------------------------------------------------------------------------
# --make-series
# --------------------------------------------------------------------------


def _wanted_steps(args, base_step: int) -> set:
    """Exactly which steps need a snapshot — nothing more.

    A full every-K series is enormous (80 x 332 MB = 26 GB for an 83M model)
    and almost all of it is never read. Only these are:

      * the three extrapolator inputs at each consistency gap g:
        anchor-2g, anchor-g, anchor   (for g in --consistency-gaps)
      * one real target per N: anchor+N

    ``--save-every`` remains available for a genuinely dense series, but the
    default derives the minimal set from the analysis that will consume it.
    """
    if args.save_every:
        return {base_step + i for i in range(0, args.steps + 1, args.save_every)}
    anchor = base_step + args.warmup_steps
    wanted = {base_step, anchor}
    for g in args.consistency_gaps:
        wanted.update({anchor - 2 * g, anchor - g})
    for n in args.n_values:
        wanted.add(anchor + n)
    return {s for s in wanted if base_step <= s <= base_step + args.steps}


def make_series(args) -> int:
    """Continue training from ``--base`` and snapshot the weights we need."""
    import train_edge_conversational as tec

    device = require_cuda()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[ghost] loading base checkpoint {args.base}")
    model, base_step, _ckpt = tec.load_checkpoint_for_resume(args.base, device)
    print(f"[ghost] resumed at global_step={base_step}")

    tokenizer = build_tokenizer()
    loader, _val, stats = tec.build_dataset_pipeline(
        tokenizer=tokenizer,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
        data_dir=args.data_dir,
        val_fraction=0.0,
        num_workers=0,
    )
    print(f"[ghost] data: {stats}")

    # Match the original run's optimizer (AdamW, lr 3e-4, wd 0.01). No LR
    # schedule: a changing LR is itself a source of weight-delta inconsistency
    # and would confound the trajectory measurement this series exists for.
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01, fused=True)

    wanted = _wanted_steps(args, base_step)
    print(f"[ghost] {len(wanted)} snapshots planned: {sorted(wanted)}")

    def snapshot(step: int) -> None:
        """Atomic, retrying snapshot write.

        Written tmp-then-replace (the #54 discipline used elsewhere in this
        repo) and retried: a mid-run ``torch.save`` on this machine died once
        with a Windows ERROR_PATH_NOT_FOUND while writing the 6th of 80
        checkpoints, consistent with the Avast real-time-scanning interference
        already documented for this box in docs/training.md (#120). A snapshot
        is pure output — losing the run to a transient AV file lock is not
        acceptable when a retry costs nothing.
        """
        sd = {k: v.detach().to("cpu", copy=True)
              for k, v in (getattr(model, "_orig_mod", model)).state_dict().items()}
        path = os.path.join(args.out_dir, f"step{step}.pt")
        tmp = path + ".tmp"
        last: Optional[Exception] = None
        for attempt in range(4):
            try:
                torch.save({"model_state_dict": sd, "global_step": step}, tmp)
                os.replace(tmp, path)
                return
            except (RuntimeError, OSError) as exc:  # transient FS/AV failure
                last = exc
                print(f"[ghost] snapshot {step} attempt {attempt + 1} failed: {exc}",
                      flush=True)
                try:
                    if os.path.exists(tmp):
                        os.remove(tmp)
                except OSError:
                    pass
                time.sleep(2.0 * (attempt + 1))
        raise RuntimeError(f"snapshot at step {step} failed after 4 attempts") from last

    meta = {
        "base_checkpoint": args.base.replace("\\", "/"),
        "base_step": base_step,
        "steps": args.steps,
        "save_every": args.save_every,
        "warmup_steps": args.warmup_steps,
        "snapshot_steps": sorted(wanted),
        "lr": args.lr,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "optimizer": "AdamW(fused, wd=0.01), constant LR",
        "optimizer_state_saved": False,
        "note": "constant LR by design — an LR schedule is itself a source of "
                "weight-delta inconsistency and would confound the measurement",
    }
    with open(os.path.join(args.out_dir, "series_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)

    model.train()
    snapshot(base_step)
    step = base_step
    done = 0
    t0 = time.time()
    print(f"[ghost] training {args.steps} steps, snapshot every {args.save_every}")
    while done < args.steps:
        for batch in loader:
            xb = (batch[0] if isinstance(batch, (tuple, list)) else batch).to(device)
            x, y = xb[:, :-1], xb[:, 1:]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, GPT2_VOCAB_SIZE).float(), y.reshape(-1)
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            done += 1
            if step in wanted:
                snapshot(step)
            if done % 50 == 0:
                rate = done / max(time.time() - t0, 1e-6)
                print(f"[ghost] step {step} ({done}/{args.steps})  loss {float(loss):.4f}  "
                      f"{rate:.2f} steps/s", flush=True)
            if done >= args.steps:
                break
    print(f"[ghost] series complete in {(time.time() - t0) / 60:.1f} min -> {args.out_dir}")
    return 0


# --------------------------------------------------------------------------
# --score
# --------------------------------------------------------------------------


def _series_steps(series_dir: str) -> List[int]:
    steps = []
    for name in os.listdir(series_dir):
        if name.startswith("step") and name.endswith(".pt"):
            try:
                steps.append(int(name[4:-3]))
            except ValueError:
                pass
    return sorted(steps)


def _load(series_dir: str, step: int) -> dict:
    return torch.load(os.path.join(series_dir, f"step{step}.pt"),
                      map_location="cpu", weights_only=False)


def score(args) -> int:
    """Score predicted vs real vs frozen at each N, on fixed probe batches."""
    import train_edge_conversational as tec

    device = require_cuda()
    steps = _series_steps(args.series_dir)
    if len(steps) < 4:
        raise SystemExit(f"need at least 4 snapshots in {args.series_dir}, found {len(steps)}")
    meta_path = os.path.join(args.series_dir, "series_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    print(f"[ghost] {len(steps)} snapshots available: {steps}")

    gap = args.gap
    anchor = args.anchor_step
    if anchor is None:
        anchor = int(meta.get("base_step", steps[0])) + int(
            meta.get("warmup_steps", args.warmup_steps))
    for need in (anchor, anchor - gap, anchor - 2 * gap):
        if need not in steps:
            raise SystemExit(f"missing snapshot at step {need} (have {steps})")
    w1, w2, w3 = (_load(args.series_dir, anchor - 2 * gap + i * gap) for i in range(3))
    print(f"[ghost] anchor step {anchor}; inputs "
          f"{anchor - 2 * gap}, {anchor - gap}, {anchor} (gap {gap})")

    tokenizer = build_tokenizer()
    probes = load_probe_batches(tokenizer, args.probe_batches, args.batch_size,
                                args.seq_len, args.data_dir)
    print(f"[ghost] {len(probes)} fixed probe batches "
          f"({args.batch_size}x{args.seq_len})")

    model, _s, _c = tec.load_checkpoint_for_resume(args.base, device)

    def loss_of(state: Dict[str, torch.Tensor]) -> float:
        missing, unexpected = model.load_state_dict(
            {k: v.to(device) for k, v in state.items()}, strict=False)
        if missing or unexpected:
            print(f"  [warn] missing={len(missing)} unexpected={len(unexpected)}")
        return evaluate_loss(model, probes, device)

    cfg = ExtrapolationConfig(cos_threshold=args.cos_threshold,
                              ratio_tol=args.ratio_tol,
                              allow_poly=not args.no_poly)
    plan = build_plan(w1["model_state_dict"], w2["model_state_dict"],
                      w3["model_state_dict"], cfg)
    print(f"[ghost] plan: {plan.n_extrapolated}/{len(plan.tensors)} tensors, "
          f"{plan.param_fraction_extrapolated:.4%} of parameters extrapolated")

    frozen_loss = loss_of(w3["model_state_dict"])
    print(f"[ghost] frozen (W3, null control) loss {frozen_loss:.5f}")

    rows = []
    for n in args.n_values:
        target = anchor + n
        if target not in steps:
            print(f"[ghost] N={n}: no real checkpoint at step {target}, skipping")
            continue
        real_loss = loss_of(_load(args.series_dir, target)["model_state_dict"])
        pred, _ = extrapolate(w1["model_state_dict"], w2["model_state_dict"],
                              w3["model_state_dict"], n, gap, cfg, plan)
        pred_loss = loss_of(pred)

        rel = abs(pred_loss - real_loss) / max(abs(real_loss), 1e-9)
        rel_frozen = abs(frozen_loss - real_loss) / max(abs(real_loss), 1e-9)
        row = {
            "N": n,
            "target_step": target,
            "real_loss": real_loss,
            "predicted_loss": pred_loss,
            "frozen_loss": frozen_loss,
            "rel_err_predicted": rel,
            "rel_err_frozen": rel_frozen,
            "within_5pct": rel <= 0.05,
            "beats_frozen": pred_loss < frozen_loss,
            "beats_frozen_margin": frozen_loss - pred_loss,
        }
        rows.append(row)
        print(f"[ghost] N={n:<4} real {real_loss:.5f}  pred {pred_loss:.5f} "
              f"({rel:+.2%})  frozen {frozen_loss:.5f} ({rel_frozen:+.2%})  "
              f"{'BEATS frozen' if row['beats_frozen'] else 'loses to frozen'}")

    # Delta-consistency vs measurement scale — the question the repo's own
    # 500-step-spaced checkpoints cannot answer.
    consistency = []
    for k in args.consistency_gaps:
        if (anchor - 2 * k) not in steps or (anchor - k) not in steps:
            print(f"[ghost] gap {k}: missing inputs, skipping")
            continue
        a, b, c = (_load(args.series_dir, anchor - 2 * k + i * k) for i in range(3))
        p = build_plan(a["model_state_dict"], b["model_state_dict"],
                       c["model_state_dict"], cfg)
        cosines = sorted(t.cos for t in p.tensors)
        consistency.append({
            "gap_steps": k,
            "median_cos": cosines[len(cosines) // 2],
            "n_extrapolated": p.n_extrapolated,
            "n_tensors": len(p.tensors),
            "param_fraction_extrapolated": p.param_fraction_extrapolated,
        })
        print(f"[ghost] gap {k:<4} median cos {consistency[-1]['median_cos']:+.4f}  "
              f"params extrapolated {p.param_fraction_extrapolated:.2%}")

    out = {
        "series_dir": args.series_dir.replace("\\", "/"),
        "series_meta": meta,
        "gap": gap,
        "anchor_step": anchor,
        "probe": {"batches": len(probes), "batch_size": args.batch_size,
                  "seq_len": args.seq_len, "seed": 1234},
        "extrapolation_config": asdict(cfg),
        "plan_summary": {k: v for k, v in plan.summary().items() if k != "tensors"},
        "optimizer_state_policy": "not carried — this codebase never saves AdamW "
                                  "moments, so every jump starts with cold moments",
        "gate_N": args.gate_n,
        "results": rows,
        "delta_consistency_vs_gap": consistency,
    }
    gate = next((r for r in rows if r["N"] == args.gate_n), None)
    if gate:
        out["gate"] = {
            "N": args.gate_n,
            "within_5pct": gate["within_5pct"],
            "beats_frozen": gate["beats_frozen"],
            "passed": bool(gate["within_5pct"] and gate["beats_frozen"]),
        }
        print(f"\n[ghost] GATE N={args.gate_n}: within 5% = {gate['within_5pct']}, "
              f"beats frozen null control = {gate['beats_frozen']} -> "
              f"{'PASS' if out['gate']['passed'] else 'FAIL'}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[ghost] wrote {args.out}")
    return 0


# --------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description="GHOST Phase 2 GPU harness")
    mode = p.add_mutually_exclusive_group(required=True)
    mode.add_argument("--make-series", action="store_true")
    mode.add_argument("--score", action="store_true")

    p.add_argument("--base", default="checkpoints/edge_chinchilla20k/"
                                     "edge_conversational_step10000.pt")
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)

    p.add_argument("--steps", type=int, default=800)
    p.add_argument("--save-every", type=int, default=None,
                   help="dense series every K steps; default saves only the "
                        "steps the analysis actually consumes")
    p.add_argument("--warmup-steps", type=int, default=200,
                   help="steps of continuation before the anchor, so the "
                        "transient from any data-mix shift has settled")
    p.add_argument("--out-dir", default="results/ghost/series")

    p.add_argument("--series-dir", default="results/ghost/series")
    p.add_argument("--anchor-step", type=int, default=None)
    p.add_argument("--gap", type=int, default=10,
                   help="step spacing between the three extrapolator inputs")
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--n-values", type=int, nargs="+", default=[10, 50, 100, 500])
    p.add_argument("--gate-n", type=int, default=10)
    p.add_argument("--consistency-gaps", type=int, nargs="+", default=[10, 20, 50, 100])
    p.add_argument("--cos-threshold", type=float, default=0.30)
    p.add_argument("--ratio-tol", type=float, default=2.0)
    p.add_argument("--no-poly", action="store_true")
    p.add_argument("--out", default="results/ghost/extrapolation_scores.json")

    args = p.parse_args(argv)
    return make_series(args) if args.make_series else score(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
