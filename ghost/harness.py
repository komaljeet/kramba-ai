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
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghost.extrapolator import ExtrapolationConfig, build_plan, extrapolate  # noqa: E402

GPT2_VOCAB_SIZE = 50257

Tensors = Mapping[str, torch.Tensor]


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


def oracle_weights(
    w2: Tensors, w3: Tensors, w_true: Tensors, n_steps: int, gap: int
) -> Tuple[Dict[str, torch.Tensor], dict]:
    """The best GHOST that could possibly exist, allowed to cheat.

    For each tensor, pick the scalar ``alpha`` that minimises the distance
    between the prediction ``W3 + alpha * f * D2`` and the ACTUAL future
    weights — using the future weights to choose it. That is

        alpha* = <dW_true, f*D2> / |f*D2|^2

    the least-squares optimum, which no real method can beat within this
    family. It is an upper bound on every adaptive, fitted or learned variant
    that predicts from the last observed delta: a per-tensor learned scaling,
    a schedule of alphas, a meta-learned shrinkage — all of them are choosing
    an alpha without being allowed to see the answer, so all of them do at
    best this well.

    If the oracle cannot beat the frozen baseline, no adaptive GHOST can.
    """
    f = n_steps / float(gap)
    out: Dict[str, torch.Tensor] = {}
    alphas: List[float] = []
    for name, t3 in w3.items():
        if name not in w2 or name not in w_true or not t3.is_floating_point():
            out[name] = t3.clone()
            continue
        step = (f * (t3 - w2[name]).to(torch.float64))
        target = (w_true[name] - t3).to(torch.float64)
        denom = float(torch.dot(step.flatten(), step.flatten()))
        if denom <= 1e-30:
            out[name] = t3.clone()
            continue
        alpha = float(torch.dot(target.flatten(), step.flatten())) / denom
        alphas.append(alpha)
        out[name] = (t3.to(torch.float64) + alpha * step).to(t3.dtype)
    alphas_sorted = sorted(alphas)
    stats = {
        "n_tensors_fitted": len(alphas),
        "alpha_median": alphas_sorted[len(alphas_sorted) // 2] if alphas_sorted else 0.0,
        "alpha_min": alphas_sorted[0] if alphas_sorted else 0.0,
        "alpha_max": alphas_sorted[-1] if alphas_sorted else 0.0,
    }
    return out, stats


def average_state_dicts(sds: Sequence[Tensors]) -> Dict[str, torch.Tensor]:
    """Plain mean of float tensors; non-float entries taken from the newest."""
    newest = sds[-1]
    out: Dict[str, torch.Tensor] = {}
    for name, t in newest.items():
        if not t.is_floating_point() or any(name not in sd for sd in sds):
            out[name] = t.clone()
            continue
        acc = torch.zeros_like(t, dtype=torch.float64)
        for sd in sds:
            acc += sd[name].to(torch.float64)
        out[name] = (acc / len(sds)).to(t.dtype)
    return out


def average(args) -> int:
    """Latest-weight averaging (LAWA/SWA-style) vs the latest checkpoint alone.

    The opposite move to extrapolation, and the reason it is worth testing after
    #126's negative result: extrapolation AMPLIFIES the noise in a weight delta
    by ``f``, whereas averaging CANCELS it while leaving the shared drift
    intact. Same observation about trajectory coherence, opposite arithmetic.

    Unlike GHOST this does not skip any training step, so it cannot violate the
    rule the extrapolation result established (the information in a step comes
    from gradients on unseen data). It asks a narrower question: are the weights
    you have ALREADY paid for being used as well as they could be?

    The comparison that matters is not "is the average better than the latest"
    but "how many real training steps is the average worth" — so the real
    checkpoints ahead of the anchor are evaluated on the same probe set and the
    average is placed on that scale.
    """
    import train_edge_conversational as tec

    device = require_cuda()
    ckpt_dir, prefix = os.path.dirname(args.base), os.path.basename(args.base)
    stem = prefix.rsplit("step", 1)[0]

    def path_for(step: int) -> str:
        return os.path.join(ckpt_dir, f"{stem}step{step}.pt")

    def load_sd(step: int) -> Dict[str, torch.Tensor]:
        return torch.load(path_for(step), map_location="cpu",
                          weights_only=False)["model_state_dict"]

    anchor = args.anchor_step
    spacing = args.avg_spacing
    available = [s for s in ([anchor - i * spacing for i in range(max(args.avg_k) - 1, -1, -1)]
                             + args.future_steps) if os.path.exists(path_for(s))]
    if anchor not in available:
        raise SystemExit(f"anchor checkpoint {path_for(anchor)} not found")
    print(f"[ghost] anchor {anchor}, spacing {spacing}, available {available}")

    tokenizer = build_tokenizer()
    probes = load_probe_batches(tokenizer, args.probe_batches, args.batch_size,
                                args.seq_len, args.data_dir)
    model, _s, _c = tec.load_checkpoint_for_resume(args.base, device)

    def loss_of(sd: Dict[str, torch.Tensor]) -> float:
        model.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=False)
        return evaluate_loss(model, probes, device)

    rows: List[dict] = []
    base_loss = loss_of(load_sd(anchor))
    print(f"[ghost] latest checkpoint (step {anchor}) loss {base_loss:.5f}")

    future = {}
    for s in args.future_steps:
        if os.path.exists(path_for(s)):
            future[s] = loss_of(load_sd(s))
            print(f"[ghost] real training to step {s:<6} loss {future[s]:.5f} "
                  f"({future[s] - base_loss:+.5f})")

    for k in args.avg_k:
        steps = [anchor - i * spacing for i in range(k - 1, -1, -1)]
        if not all(os.path.exists(path_for(s)) for s in steps):
            print(f"[ghost] k={k}: missing checkpoints, skipping")
            continue
        avg_loss = loss_of(average_state_dicts([load_sd(s) for s in steps]))
        gain = base_loss - avg_loss
        # Translate the gain into "steps of real training bought", by finding
        # the furthest real checkpoint the average is at least as good as.
        equiv = 0
        for s in sorted(future):
            if avg_loss <= future[s]:
                equiv = s - anchor
        rows.append({"k": k, "steps_averaged": steps, "loss": avg_loss,
                     "gain_vs_latest": gain, "steps_of_training_equivalent": equiv})
        print(f"[ghost] avg k={k:<3} ({steps[0]}..{steps[-1]})  loss {avg_loss:.5f}  "
              f"{'BETTER' if gain > 0 else 'worse'} than latest by {abs(gain):.5f}  "
              f"~= {equiv} steps of real training")

    out = {
        "anchor_step": anchor,
        "spacing": spacing,
        "probe": {"batches": len(probes), "batch_size": args.batch_size,
                  "seq_len": args.seq_len, "seed": 1234},
        "latest_loss": base_loss,
        "real_future_losses": {str(k): v for k, v in future.items()},
        "averages": rows,
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[ghost] wrote {args.out}")
    return 0


def ab_continue(args) -> int:
    """Does the averaging advantage SURVIVE continued training, or wash out?

    This is the experiment that separates "free better checkpoint" (already
    measured, certain) from "free training steps" (the claim worth wiring into
    the training loop). Three arms continue training from the same anchor:

      latest      the anchor checkpoint, unchanged                (baseline)
      averaged    mean of the last k checkpoints
      perturbed   anchor + Gaussian noise, per-tensor norm matched to the
                  averaging displacement                          (CONTROL)

    The perturbed arm is not decoration. Averaging MOVES the weights; without a
    matched-norm control a gain could be nothing more than perturbation acting
    as noise injection, which is a known (if weak) regulariser. If perturbed
    tracks averaged, the direction averaging picks is irrelevant and the result
    is not about averaging at all.

    Every arm replays an IDENTICAL, pre-materialised batch sequence and starts
    from a freshly constructed optimizer, so the only difference between arms is
    the starting weights.
    """
    import train_edge_conversational as tec

    device = require_cuda()
    ckpt_dir = os.path.dirname(args.base)
    stem = os.path.basename(args.base).rsplit("step", 1)[0]

    def load_sd(step: int) -> Dict[str, torch.Tensor]:
        return torch.load(os.path.join(ckpt_dir, f"{stem}step{step}.pt"),
                          map_location="cpu", weights_only=False)["model_state_dict"]

    anchor = args.anchor_step
    avg_steps = [anchor - i * args.avg_spacing for i in range(args.avg_k[0] - 1, -1, -1)]
    print(f"[ghost] anchor {anchor}; averaging {avg_steps}")

    latest_sd = load_sd(anchor)
    avg_sd = average_state_dicts([load_sd(s) for s in avg_steps])

    gen = torch.Generator().manual_seed(4242)
    pert_sd: Dict[str, torch.Tensor] = {}
    for name, t in latest_sd.items():
        if not t.is_floating_point():
            pert_sd[name] = t.clone()
            continue
        disp = (avg_sd[name].to(torch.float64) - t.to(torch.float64))
        dnorm = float(torch.linalg.vector_norm(disp))
        noise = torch.randn(t.shape, generator=gen, dtype=torch.float64)
        nnorm = float(torch.linalg.vector_norm(noise))
        if nnorm > 0 and dnorm > 0:
            noise = noise * (dnorm / nnorm)
        pert_sd[name] = (t.to(torch.float64) + noise).to(t.dtype)

    tokenizer = build_tokenizer()
    probes = load_probe_batches(tokenizer, args.probe_batches, args.batch_size,
                                args.seq_len, args.data_dir)
    torch.manual_seed(args.seed)
    loader, _v, _s = tec.build_dataset_pipeline(
        tokenizer=tokenizer, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)

    # Materialise once so every arm sees byte-identical batches in the same
    # order. ~16 MB for 1,000 batches of 4x512 int64 — cheap insurance against
    # data order becoming a confound between arms.
    batches: List[torch.Tensor] = []
    for batch in loader:
        batches.append((batch[0] if isinstance(batch, (tuple, list)) else batch).clone())
        if len(batches) >= args.ab_steps:
            break
    print(f"[ghost] materialised {len(batches)} identical batches for all arms")

    results: Dict[str, List[dict]] = {}
    for arm, sd in (("latest", latest_sd), ("averaged", avg_sd), ("perturbed", pert_sd)):
        # Re-seed per arm so dropout draws the same mask sequence in every arm
        # (see repeat_average for the measurement that exposed this).
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model, _st, _ck = tec.load_checkpoint_for_resume(args.base, device)
        model.load_state_dict({k: v.to(device) for k, v in sd.items()}, strict=False)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.01, fused=True)
        traj = [{"step": 0, "loss": evaluate_loss(model, probes, device)}]
        print(f"[ghost] --- arm '{arm}' start loss {traj[0]['loss']:.5f}")
        model.train()
        t0 = time.time()
        for i, xb in enumerate(batches, start=1):
            xb = xb.to(device)
            x, y = xb[:, :-1], xb[:, 1:]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, GPT2_VOCAB_SIZE).float(), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if i % args.ab_eval_every == 0:
                l = evaluate_loss(model, probes, device)
                traj.append({"step": i, "loss": l})
                print(f"[ghost]     {arm:<10} step {i:<5} probe loss {l:.5f} "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
                model.train()
        results[arm] = traj
        del model, opt
        torch.cuda.empty_cache()

    print("\n[ghost] === A/B: does the averaging advantage survive training? ===")
    print(f"[ghost] {'step':>6}  {'latest':>9}  {'averaged':>9}  {'perturbed':>9}  "
          f"{'avg-latest':>11}")
    verdict_rows = []
    for idx in range(len(results["latest"])):
        st = results["latest"][idx]["step"]
        a = results["latest"][idx]["loss"]
        b = results["averaged"][idx]["loss"]
        c = results["perturbed"][idx]["loss"]
        verdict_rows.append({"step": st, "latest": a, "averaged": b,
                             "perturbed": c, "advantage": a - b})
        print(f"[ghost] {st:>6}  {a:>9.5f}  {b:>9.5f}  {c:>9.5f}  {a - b:>+11.5f}")

    first, last = verdict_rows[0]["advantage"], verdict_rows[-1]["advantage"]
    survives = last > 0
    retained = (last / first) if abs(first) > 1e-9 else 0.0
    print(f"\n[ghost] advantage at start {first:+.5f} -> at step "
          f"{verdict_rows[-1]['step']} {last:+.5f} ({retained:.0%} retained)")
    print(f"[ghost] VERDICT: averaging advantage {'SURVIVES' if survives else 'WASHES OUT'} "
          f"continued training")
    pert_helps = verdict_rows[-1]["perturbed"] < verdict_rows[-1]["latest"]
    print(f"[ghost] control: perturbed {'also beats' if pert_helps else 'loses to'} latest "
          f"-> gain is {'NOT clearly' if pert_helps else 'specifically'} from averaging")

    out = {"anchor_step": anchor, "averaged_steps": avg_steps, "ab_steps": args.ab_steps,
           "lr": args.lr, "batch_size": args.batch_size, "seq_len": args.seq_len,
           "probe_batches": len(probes), "trajectories": results,
           "comparison": verdict_rows,
           "verdict": {"advantage_start": first, "advantage_end": last,
                       "fraction_retained": retained, "survives": survives,
                       "perturbed_control_also_beats_latest": pert_helps}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[ghost] wrote {args.out}")
    return 0


def repeat_average(args) -> int:
    """Does averaging COMPOUND when applied repeatedly during training?

    #128 established that averaging the last k checkpoints is both a better
    model and a real (decaying) head start. The remaining question is whether
    applying it over and over inside the loop stacks, or whether each
    application merely re-buys the same fading gain and the run lands in the
    same place. That is what decides inside-the-loop vs end-of-run.

    Two arms, identical pre-materialised batch sequence, identical fresh
    optimizer:

      baseline   plain training; snapshots taken but never fed back
      periodic   every ``--ra-interval`` steps, the live weights are REPLACED
                 by the mean of the last ``--ra-k`` snapshots, then training
                 continues from there

    A third number costs nothing and is the reference that matters: the mean of
    the BASELINE arm's final k snapshots — i.e. the already-proven end-of-run
    application. Periodic averaging only earns a place in the training loop if
    it beats that, not merely if it beats the raw baseline.

    Optimizer-state policy: AdamW moments are CARRIED across an injection
    rather than reset. The moments describe a trajectory the injection
    perturbs, so this is a real (small) inconsistency — but resetting them
    every interval would throw away far more, and the arms differ only in the
    injection, so the comparison stays clean either way.
    """
    import train_edge_conversational as tec

    device = require_cuda()
    tokenizer = build_tokenizer()
    probes = load_probe_batches(tokenizer, args.probe_batches, args.batch_size,
                                args.seq_len, args.data_dir)
    torch.manual_seed(args.seed)
    loader, _v, _s = tec.build_dataset_pipeline(
        tokenizer=tokenizer, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)

    batches: List[torch.Tensor] = []
    for batch in loader:
        batches.append((batch[0] if isinstance(batch, (tuple, list)) else batch).clone())
        if len(batches) >= args.ra_steps:
            break
    print(f"[ghost] materialised {len(batches)} identical batches for both arms")

    results: Dict[str, dict] = {}
    for arm in ("baseline", "periodic"):
        # Re-seed per arm. Without this the arms are NOT identical before the
        # first injection: the model has dropout, so the second arm draws a
        # different mask sequence and drifts from the first on identical data.
        # Measured on the run that exposed this: 0.0036 apart at step 250 and
        # 0.0041 at step 500, with no injection yet applied. Small next to the
        # ~0.05 effect, but it is avoidable noise sitting directly on top of
        # the quantity being compared.
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model, _st, _ck = tec.load_checkpoint_for_resume(args.base, device)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=0.01, fused=True)
        buf: List[Dict[str, torch.Tensor]] = []
        traj = [{"step": 0, "loss": evaluate_loss(model, probes, device)}]
        injections: List[dict] = []
        print(f"[ghost] --- arm '{arm}' start loss {traj[0]['loss']:.5f}")
        model.train()
        t0 = time.time()
        for i, xb in enumerate(batches, start=1):
            xb = xb.to(device)
            x, y = xb[:, :-1], xb[:, 1:]
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, GPT2_VOCAB_SIZE).float(), y.reshape(-1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)

            if i % args.ra_interval == 0:
                core = getattr(model, "_orig_mod", model)
                buf.append({k: v.detach().to("cpu", copy=True)
                            for k, v in core.state_dict().items()})
                buf[:] = buf[-args.ra_k:]
                before = evaluate_loss(model, probes, device)
                after = before
                if arm == "periodic" and len(buf) == args.ra_k:
                    avg = average_state_dicts(buf)
                    model.load_state_dict({k: v.to(device) for k, v in avg.items()},
                                          strict=False)
                    after = evaluate_loss(model, probes, device)
                    # The injected weights become the buffer's newest entry,
                    # otherwise the next average re-averages the pre-injection
                    # trajectory and the intervention partially undoes itself.
                    buf[-1] = {k: v.clone() for k, v in avg.items()}
                    injections.append({"step": i, "before": before, "after": after,
                                       "gain": before - after})
                    print(f"[ghost]     {arm} step {i:<5} inject: {before:.5f} -> "
                          f"{after:.5f} ({before - after:+.5f})", flush=True)
                else:
                    print(f"[ghost]     {arm:<9} step {i:<5} probe loss {before:.5f} "
                          f"({(time.time() - t0) / 60:.1f} min)", flush=True)
                traj.append({"step": i, "loss": after})
                model.train()

        final = {"trajectory": traj, "injections": injections,
                 "final_loss": traj[-1]["loss"]}
        if arm == "baseline" and len(buf) >= args.ra_k:
            tail = average_state_dicts(buf[-args.ra_k:])
            model.load_state_dict({k: v.to(device) for k, v in tail.items()},
                                  strict=False)
            final["tail_average_loss"] = evaluate_loss(model, probes, device)
            print(f"[ghost] baseline + END-OF-RUN average of last {args.ra_k}: "
                  f"{final['tail_average_loss']:.5f}")
        results[arm] = final
        del model, opt, buf
        torch.cuda.empty_cache()

    base_final = results["baseline"]["final_loss"]
    tail = results["baseline"].get("tail_average_loss")
    per_final = results["periodic"]["final_loss"]
    print("\n[ghost] === does periodic averaging compound? ===")
    print(f"[ghost] baseline, no averaging          {base_final:.5f}")
    if tail is not None:
        print(f"[ghost] baseline + end-of-run average   {tail:.5f} "
              f"({tail - base_final:+.5f} vs baseline)")
    print(f"[ghost] periodic averaging in-loop      {per_final:.5f} "
          f"({per_final - base_final:+.5f} vs baseline)")
    compounds = tail is not None and per_final < tail
    print(f"[ghost] periodic vs end-of-run: {per_final - tail:+.5f}" if tail else "")
    print(f"[ghost] VERDICT: periodic averaging {'COMPOUNDS' if compounds else 'does NOT compound'} "
          f"-- {'belongs in the training loop' if compounds else 'keep it as an end-of-run step'}")
    if results["periodic"]["injections"]:
        gains = [x["gain"] for x in results["periodic"]["injections"]]
        print(f"[ghost] per-injection gains: "
              f"{', '.join(f'{g:+.4f}' for g in gains)}")

    out = {"base": args.base.replace("\\", "/"), "ra_steps": args.ra_steps,
           "ra_interval": args.ra_interval, "ra_k": args.ra_k, "lr": args.lr,
           "optimizer_state_on_injection": "carried, not reset",
           "results": results,
           "verdict": {"baseline_final": base_final, "tail_average": tail,
                       "periodic_final": per_final, "compounds": compounds}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[ghost] wrote {args.out}")
    return 0


def parallel_ensemble(args) -> int:
    """3 parallel networks + weight averaging vs ONE network at 3x batch (#130).

    The maintainer's proposal: train K networks side by side, periodically
    average their weights, and hand the average back to each one perturbed --
    ``avg+delta``, ``avg``, ``avg-delta`` -- so they stay diverse instead of
    collapsing into the same model.

    The honest comparison holds TOTAL TOKENS constant. Three networks at batch 4
    obviously beat one network at batch 4; that is just more data. So the
    baseline is one network at batch 12, and every arm sees the *same tokens in
    the same order*: ``3S`` batches of 4 are materialised once, the ensemble
    arms take one each per round, and the single arm concatenates each group of
    three into a batch of 12.

    Arms:
      single       1 network,  batch 3B          -- the baseline
      avg3         3 networks, batch B, averaged every --pe-interval rounds
      avg3_delta   same, but redistributed as avg+d, avg, avg-d, where d is the
                   per-tensor spread across the three nets before averaging
                   (i.e. "keep them as diverse as they just were")

    Wall-clock is reported per arm and is part of the verdict: the ensemble arms
    run 3x as many forward/backward passes at a third the batch, which is a
    worse fit for the GPU. On a single card a token-win that loses on wall-clock
    is not a win.
    """
    import train_edge_conversational as tec

    device = require_cuda()
    tokenizer = build_tokenizer()
    probes = load_probe_batches(tokenizer, args.probe_batches, args.batch_size,
                                args.seq_len, args.data_dir)
    torch.manual_seed(args.seed)
    loader, _v, _s = tec.build_dataset_pipeline(
        tokenizer=tokenizer, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)

    K, rounds = args.pe_nets, args.pe_rounds
    need = K * rounds
    batches: List[torch.Tensor] = []
    for batch in loader:
        batches.append((batch[0] if isinstance(batch, (tuple, list)) else batch).clone())
        if len(batches) >= need:
            break
    if len(batches) < need:
        raise SystemExit(f"need {need} batches, loader gave {len(batches)}")
    print(f"[ghost] {len(batches)} batches of {args.batch_size} materialised "
          f"-> {rounds} rounds x {K} nets, identical tokens for every arm")

    def fresh():
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        m, _st, _ck = tec.load_checkpoint_for_resume(args.base, device)
        o = torch.optim.AdamW(m.parameters(), lr=args.lr, weight_decay=0.01, fused=True)
        return m, o

    def train_on(model, opt, xb):
        xb = xb.to(device)
        x, y = xb[:, :-1], xb[:, 1:]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, GPT2_VOCAB_SIZE).float(), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return float(loss)

    results: Dict[str, dict] = {}

    # ---- arm 1: one network, batch K*B ----
    model, opt = fresh()
    traj = [{"round": 0, "loss": evaluate_loss(model, probes, device)}]
    print(f"[ghost] --- arm 'single' (1 net, batch {K * args.batch_size}) "
          f"start {traj[0]['loss']:.5f}")
    model.train()
    t0 = time.time()
    for r in range(rounds):
        big = torch.cat(batches[r * K:(r + 1) * K], dim=0)
        train_on(model, opt, big)
        if (r + 1) % args.pe_eval_every == 0:
            l = evaluate_loss(model, probes, device)
            traj.append({"round": r + 1, "loss": l})
            print(f"[ghost]     single     round {r + 1:<5} {l:.5f} "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            model.train()
    results["single"] = {"trajectory": traj, "final": traj[-1]["loss"],
                         "minutes": (time.time() - t0) / 60}
    del model, opt
    torch.cuda.empty_cache()

    # ---- arms 2 & 3: K networks + averaging ----
    for arm, use_delta in (("avg3", False), ("avg3_delta", True)):
        nets = []
        for _ in range(K):
            nets.append(fresh())
        traj = [{"round": 0, "loss": evaluate_loss(nets[0][0], probes, device)}]
        print(f"[ghost] --- arm '{arm}' ({K} nets, batch {args.batch_size}, "
              f"average every {args.pe_interval}) start {traj[0]['loss']:.5f}")
        for m, _o in nets:
            m.train()
        t0 = time.time()
        for r in range(rounds):
            for i, (m, o) in enumerate(nets):
                train_on(m, o, batches[r * K + i])
            if (r + 1) % args.pe_interval == 0:
                sds = [{k: v.detach().to("cpu", copy=True)
                        for k, v in getattr(m, "_orig_mod", m).state_dict().items()}
                       for m, _o in nets]
                avg = average_state_dicts(sds)
                spread = {}
                if use_delta:
                    for name, t in avg.items():
                        if not t.is_floating_point():
                            continue
                        stack = torch.stack([sd[name].to(torch.float64) for sd in sds])
                        spread[name] = stack.std(dim=0)
                for i, (m, _o) in enumerate(nets):
                    if use_delta and i != K // 2:
                        sign = 1.0 if i < K // 2 else -1.0
                        new = {k: (v.to(torch.float64) + sign * spread[k]).to(v.dtype)
                               if k in spread else v.clone() for k, v in avg.items()}
                    else:
                        new = {k: v.clone() for k, v in avg.items()}
                    m.load_state_dict({k: v.to(device) for k, v in new.items()},
                                      strict=False)
                    m.train()
            if (r + 1) % args.pe_eval_every == 0:
                sds = [{k: v.detach().to("cpu", copy=True)
                        for k, v in getattr(m, "_orig_mod", m).state_dict().items()}
                       for m, _o in nets]
                probe_model = nets[0][0]
                saved = {k: v.detach().clone() for k, v in
                         getattr(probe_model, "_orig_mod", probe_model).state_dict().items()}
                avg_eval = average_state_dicts(sds)
                probe_model.load_state_dict(
                    {k: v.to(device) for k, v in avg_eval.items()}, strict=False)
                l = evaluate_loss(probe_model, probes, device)
                probe_model.load_state_dict(saved, strict=False)
                traj.append({"round": r + 1, "loss": l})
                print(f"[ghost]     {arm:<10} round {r + 1:<5} {l:.5f} (averaged) "
                      f"({(time.time() - t0) / 60:.1f} min)", flush=True)
                for m, _o in nets:
                    m.train()
        results[arm] = {"trajectory": traj, "final": traj[-1]["loss"],
                        "minutes": (time.time() - t0) / 60}
        del nets
        torch.cuda.empty_cache()

    print("\n[ghost] === equal tokens: 3 nets + averaging vs 1 net at 3x batch ===")
    print(f"[ghost] {'arm':<12} {'final loss':>11} {'minutes':>9}")
    for arm in ("single", "avg3", "avg3_delta"):
        r = results[arm]
        print(f"[ghost] {arm:<12} {r['final']:>11.5f} {r['minutes']:>9.1f}")
    best = min(results, key=lambda a: results[a]["final"])
    faster = min(results, key=lambda a: results[a]["minutes"])
    print(f"[ghost] best quality: {best}   fastest: {faster}")
    print(f"[ghost] VERDICT: {'ensemble wins' if best != 'single' else 'SINGLE network wins'} "
          f"on quality at equal tokens; "
          f"{'ensemble' if faster != 'single' else 'single'} wins on wall-clock")

    out = {"base": args.base.replace("\\", "/"), "nets": K, "rounds": rounds,
           "batch_per_net": args.batch_size, "single_batch": K * args.batch_size,
           "average_interval": args.pe_interval, "seq_len": args.seq_len,
           "tokens_per_arm": rounds * K * args.batch_size * args.seq_len,
           "results": results,
           "verdict": {"best_quality": best, "fastest": faster}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[ghost] wrote {args.out}")
    return 0


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
        true_sd = _load(args.series_dir, target)["model_state_dict"]
        real_loss = loss_of(true_sd)
        pred, _ = extrapolate(w1["model_state_dict"], w2["model_state_dict"],
                              w3["model_state_dict"], n, gap, cfg, plan)
        pred_loss = loss_of(pred)

        oracle_loss = None
        oracle_stats = None
        if args.oracle:
            ow, oracle_stats = oracle_weights(w2["model_state_dict"],
                                              w3["model_state_dict"],
                                              true_sd, n, gap)
            oracle_loss = loss_of(ow)

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
        if oracle_loss is not None:
            row["oracle_loss"] = oracle_loss
            row["rel_err_oracle"] = abs(oracle_loss - real_loss) / max(abs(real_loss), 1e-9)
            row["oracle_beats_frozen"] = oracle_loss < frozen_loss
            row["oracle_alpha"] = oracle_stats
        rows.append(row)
        print(f"[ghost] N={n:<4} real {real_loss:.5f}  pred {pred_loss:.5f} "
              f"({rel:+.2%})  frozen {frozen_loss:.5f} ({rel_frozen:+.2%})  "
              f"{'BEATS frozen' if row['beats_frozen'] else 'loses to frozen'}")
        if oracle_loss is not None:
            print(f"         ORACLE (cheats, upper bound) {oracle_loss:.5f} "
                  f"({row['rel_err_oracle']:+.2%})  median alpha "
                  f"{oracle_stats['alpha_median']:+.3f}  "
                  f"{'BEATS frozen' if row['oracle_beats_frozen'] else 'loses to frozen'}")

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
    mode.add_argument("--parallel-ensemble", action="store_true",
                      help="#130: K nets + averaging vs 1 net at Kx batch")
    mode.add_argument("--repeat-average", action="store_true",
                      help="does periodic in-loop averaging compound?")
    mode.add_argument("--ab-continue", action="store_true",
                      help="A/B: continue training from latest vs averaged vs perturbed")
    mode.add_argument("--average", action="store_true",
                      help="latest-weight-averaging test (LAWA/SWA-style)")

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
    p.add_argument("--oracle", action="store_true",
                   help="also evaluate the cheating least-squares upper bound")
    p.add_argument("--avg-k", type=int, nargs="+", default=[2, 3, 5, 10])
    p.add_argument("--ab-steps", type=int, default=1000)
    p.add_argument("--ab-eval-every", type=int, default=100)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--ra-steps", type=int, default=2000)
    p.add_argument("--ra-interval", type=int, default=250)
    p.add_argument("--ra-k", type=int, default=3)
    p.add_argument("--pe-nets", type=int, default=3)
    p.add_argument("--pe-rounds", type=int, default=300)
    p.add_argument("--pe-interval", type=int, default=10)
    p.add_argument("--pe-eval-every", type=int, default=25)
    p.add_argument("--avg-spacing", type=int, default=500)
    p.add_argument("--future-steps", type=int, nargs="+",
                   default=[10500, 11000, 11500, 12000, 13000])
    p.add_argument("--out", default="results/ghost/extrapolation_scores.json")

    args = p.parse_args(argv)
    if args.make_series:
        return make_series(args)
    if args.average:
        return average(args)
    if args.ab_continue:
        return ab_continue(args)
    if args.repeat_average:
        return repeat_average(args)
    if args.parallel_ensemble:
        return parallel_ensemble(args)
    return score(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
