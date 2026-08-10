"""A/B: fixed cosine vs the phase-aware dynamic LR multiplier (Issue #133).

Written by: Rambarun Komaljeet

Every arm continues training from the SAME checkpoint over a byte-identical,
pre-materialised batch sequence, with per-arm re-seeding so dropout draws the
same mask sequence, the same Muon optimizer and the same warmup+cosine base
schedule. The only difference between arms is the global LR multiplier.

Arms
----
``cosine``    fixed warmup+cosine                              (baseline)
``dynamic``   the same schedule x DynamicLRScheduler's live multiplier
``fixed<M>``  the same schedule x a CONSTANT M                 (control)

The third arm is the one that makes the result mean anything, and it is
requested explicitly with ``--fixed-mult`` after the dynamic arm has run. If
``dynamic`` wins, the immediate question is whether the win came from *phase
awareness* or merely from having spent the run at a higher average LR. A
constant multiplier equal to the dynamic arm's own measured mean answers that
directly. This repo has been caught by exactly this class of missing control
before (#128's matched-norm perturbation arm) and by grading arms on the wrong
quantity (#132's train-loss retraction), which is why:

**Arms are graded on held-out probe loss, not training loss.** Training loss
measures how hard an arm fitted the data it saw; with the arms running at
different effective learning rates that is precisely the quantity that misleads.
The probe set is a fixed set of held-out windows, identical for every arm and
every evaluation.

"Steps to converge" is defined before the run, not after: the first evaluated
step at which an arm's held-out loss reaches the *baseline arm's final*
held-out loss. An arm that never reaches it has no value, which is reported as
such rather than as a large number.

Usage
-----
    python scripts/ab_dynamic_lr.py --steps 5000
    python scripts/ab_dynamic_lr.py --steps 5000 --fixed-mult 1.8 --arms fixed
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghost.dynamic_lr import DynamicLRScheduler  # noqa: E402
from ghost.harness import (  # noqa: E402
    GPT2_VOCAB_SIZE,
    build_tokenizer,
    require_cuda,
)

DEFAULT_BASE = "checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt"


def as_pair(batch):
    """The loader already yields (x, y, src) with x and y both seq_len wide.

    Taking ``x`` alone and re-slicing it into ``x[:, :-1], x[:, 1:]`` -- the
    convention ghost/harness.py uses -- drops a token AND leaves an odd width,
    which the megakernel scan's backward rejects outright ("S=511 must be
    divisible by block B=4"). Use the pair the pipeline already built.
    """
    return batch[0], batch[1]


@torch.no_grad()
def evaluate_pairs(model, pairs, device) -> float:
    """Mean next-token cross-entropy over the fixed held-out probe pairs."""
    model.eval()
    total = 0.0
    for x, y in pairs:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        total += float(loss)
    return total / max(len(pairs), 1)


def enable_scan_flags(model) -> None:
    """Triton + megakernel scan on every DeltaNet block, in EVERY arm.

    Mirrors ``continue_ft_hybrid.py``, which has had both unconditionally on
    since 2026-08-01. Identical in both arms, so it cannot confound the
    comparison -- and leaving them off would run DeltaNet's eager scan, which
    #122 measured at ~42x slower.
    """
    n_t = n_m = 0
    for blk in getattr(model, "blocks", []):
        inner = getattr(blk, "block", blk)
        if hasattr(inner, "use_triton_scan"):
            inner.use_triton_scan = True
            n_t += 1
        if hasattr(inner, "use_megakernel_scan"):
            inner.use_megakernel_scan = True
            n_m += 1
    print(f"[ab] scan flags: triton on {n_t} blocks, megakernel on {n_m}")


def build_arm_optimizer(model, args):
    """The training script's own optimizer, not a stand-in."""
    from hvn.optimizers import build_optimizer

    opt, desc = build_optimizer(
        model, args.optimizer, lr=args.lr, weight_decay=args.weight_decay,
        fused=(args.optimizer == "adamw"), muon_lr=args.muon_lr,
    )
    return opt, desc


def lr_scale_for_warmup(step: int, warmup: int) -> float:
    return (step / max(1, warmup)) if step < warmup else 1.0


def run_arm(
    arm: str,
    args,
    batches: Sequence[torch.Tensor],
    probes: Sequence[torch.Tensor],
    device: torch.device,
    fixed_mult: Optional[float] = None,
) -> dict:
    import train_edge_conversational as tec

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, base_step, _ck = tec.load_checkpoint_for_resume(args.base, device)
    enable_scan_flags(model)
    opt, desc = build_arm_optimizer(model, args)
    base_lrs = [pg["lr"] for pg in opt.param_groups]
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    dyn = None
    if arm == "dynamic":
        dyn = DynamicLRScheduler(
            log_path=os.path.join(args.out_dir, "lr_log.json"), run_name="dynamic")

    print(f"\n[ab] === arm '{arm}' === base step {base_step}  optimizer {desc}"
          + (f"  fixed multiplier {fixed_mult}" if fixed_mult else ""))

    traj = [{"step": 0, "heldout_loss": evaluate_pairs(model, probes, device)}]
    print(f"[ab]   step {0:>5}  held-out {traj[0]['heldout_loss']:.5f}")

    train_curve: List[dict] = []
    model.train()
    t0 = time.time()
    warmup_done = args.warmup_steps <= 0

    for i, (bx, by) in enumerate(batches, start=1):
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        # Warmup scales each group from its own base LR, so Muon's matrix-group
        # / aux-group ratio survives it (the training script does the same).
        if not warmup_done:
            s = lr_scale_for_warmup(i - 1, args.warmup_steps)
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = b * s
            if i >= args.warmup_steps:
                warmup_done = True

        step_loss = float(loss.detach())
        mult = 1.0
        if dyn is not None:
            mult = dyn.observe(i, step_loss)
        elif fixed_mult is not None:
            mult = fixed_mult

        sched_lr = opt.param_groups[0]["lr"]
        if mult != 1.0:
            saved = [pg["lr"] for pg in opt.param_groups]
            for pg, lr in zip(opt.param_groups, saved):
                pg["lr"] = lr * mult
            opt.step()
            for pg, lr in zip(opt.param_groups, saved):
                pg["lr"] = lr          # restore BEFORE cos.step(): recursive cosine
        else:
            opt.step()
        opt.zero_grad(set_to_none=True)

        if warmup_done:
            cos.step()

        train_curve.append({"step": i, "loss": step_loss,
                            "sched_lr": sched_lr, "mult": mult,
                            "eff_lr": sched_lr * mult})

        if i % args.eval_every == 0:
            hl = evaluate_pairs(model, probes, device)
            traj.append({"step": i, "heldout_loss": hl})
            recent = [c["loss"] for c in train_curve[-args.eval_every:]]
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  train {sum(recent)/len(recent):.5f}"
                  f"  xLR {mult:.2f}  eff_lr {sched_lr * mult:.3e}"
                  f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)
            model.train()

    wall = time.time() - t0
    if dyn is not None:
        dyn.close()

    mults = [c["mult"] for c in train_curve]
    out = {
        "arm": arm,
        "wall_clock_sec": wall,
        "steps": len(train_curve),
        "final_heldout_loss": traj[-1]["heldout_loss"],
        "final_train_loss_last200": sum(c["loss"] for c in train_curve[-200:])
                                    / max(1, len(train_curve[-200:])),
        "mean_multiplier": sum(mults) / max(1, len(mults)),
        "max_multiplier": max(mults),
        "min_multiplier": min(mults),
        "mean_eff_lr": sum(c["eff_lr"] for c in train_curve) / max(1, len(train_curve)),
        "trajectory": traj,
        "train_curve": train_curve,
        "phase_counts": dict(dyn._phase_counts) if dyn is not None else {},
        "n_adjustments": len(dyn.adjustments) if dyn is not None else 0,
    }
    del model, opt
    torch.cuda.empty_cache()
    return out


def steps_to_reach(traj: List[dict], target: float) -> Optional[int]:
    """First evaluated step whose held-out loss is at or below ``target``."""
    for p in traj:
        if p["heldout_loss"] <= target:
            return p["step"]
    return None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--arms", nargs="+", default=["cosine", "dynamic"],
                   choices=["cosine", "dynamic", "fixed"])
    p.add_argument("--fixed-mult", type=float, default=None,
                   help="constant multiplier for the 'fixed' control arm")
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--optimizer", default="muon")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out-dir", default="results/dynamic_lr")
    p.add_argument("--out", default=None)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True,
                   help="Make the arms bitwise comparable. Costs ~9 pct wall-clock "
                        "and is ON by default here because without it this A/B "
                        "cannot resolve its own target effect — see below.")
    args = p.parse_args(argv)

    if "fixed" in args.arms and args.fixed_mult is None:
        p.error("--arms fixed requires --fixed-mult")

    if args.deterministic:
        # Two identical-code arms land 0.0046 nats apart (sd over 8 runs), with
        # the sign of the gap flipping run to run. Root cause: this torch build
        # has no flash-attention, so the 2 TransformerBlocks' SDPA falls to the
        # mem-efficient CUTLASS kernel, whose backward accumulates grad_q with
        # atomics — nondeterministic by PyTorch's own warning. Muon then
        # amplifies it: Newton-Schulz normalises the update, so a ~2e-3 relative
        # gradient perturbation becomes a full-size step in a different
        # direction. The target effect for this experiment is 0.01-0.05 nats, so
        # the floor sits inside the band being measured and an A/B run without
        # this is not capable of detecting what it claims to.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        print("[ab] deterministic algorithms ON (~9 pct slower, arms bitwise "
              "comparable)")

    device = require_cuda()
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = args.out or os.path.join(args.out_dir, "ab_dynamic_lr.json")

    import train_edge_conversational as tec

    tokenizer = build_tokenizer()
    torch.manual_seed(args.seed)
    loader, _v, stats = tec.build_dataset_pipeline(
        tokenizer=tokenizer, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    print(f"[ab] data: {stats}")

    # A fixed held-out probe set drawn FIRST, then the training batches — so no
    # window scored in the probe is ever trained on by any arm.
    pairs: List[tuple] = []
    for batch in loader:
        x, y = as_pair(batch)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= args.probe_batches + args.steps:
            break
    if len(pairs) < args.probe_batches + args.steps:
        raise SystemExit(f"loader yielded only {len(pairs)} of "
                         f"{args.probe_batches + args.steps} batches")
    probes = pairs[:args.probe_batches]
    batches = pairs[args.probe_batches:]
    print(f"[ab] {len(probes)} held-out probe batches + {len(batches)} identical "
          f"training batches for every arm")

    results: Dict[str, dict] = {}
    if os.path.exists(out_path):
        # Arm C is normally launched later, after the dynamic arm's mean
        # multiplier is known. Keep the earlier arms rather than overwrite them.
        with open(out_path, encoding="utf-8") as fh:
            results = json.load(fh).get("arms", {})
        print(f"[ab] carrying forward {sorted(results)} from {out_path}")

    for arm in args.arms:
        name = f"fixed{args.fixed_mult:g}" if arm == "fixed" else arm
        results[name] = run_arm(arm, args, batches, probes, device,
                                fixed_mult=args.fixed_mult if arm == "fixed" else None)
        _write(out_path, args, results)

    _report(results)
    _write(out_path, args, results)
    print(f"\n[ab] wrote {out_path}")
    return 0


def _write(path: str, args, results: Dict[str, dict]) -> None:
    payload = {
        "base_checkpoint": args.base.replace("\\", "/"),
        "steps": args.steps,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "optimizer": args.optimizer,
        "lr": args.lr,
        "muon_lr": args.muon_lr,
        "warmup_steps": args.warmup_steps,
        "seed": args.seed,
        "probe_batches": args.probe_batches,
        "graded_on": "held-out probe loss (NOT train loss — see #132)",
        "converge_definition": "first evaluated step at or below the cosine "
                               "baseline's FINAL held-out loss",
        "arms": results,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


def _report(results: Dict[str, dict]) -> None:
    base = results.get("cosine")
    print("\n[ab] ================= RESULT =================")
    hdr = f"{'arm':<12} {'final held-out':>15} {'final train':>12} {'steps->base':>12} {'wall (min)':>11} {'mean xLR':>9}"
    print("[ab] " + hdr)
    for name, r in results.items():
        conv = "—"
        if base is not None:
            s = steps_to_reach(r["trajectory"], base["final_heldout_loss"])
            conv = str(s) if s is not None else "never"
        print(f"[ab] {name:<12} {r['final_heldout_loss']:>15.5f} "
              f"{r['final_train_loss_last200']:>12.5f} {conv:>12} "
              f"{r['wall_clock_sec'] / 60:>11.1f} {r['mean_multiplier']:>9.3f}")
    if base is not None and "dynamic" in results:
        d = results["dynamic"]
        gap = base["final_heldout_loss"] - d["final_heldout_loss"]
        print(f"\n[ab] dynamic - cosine held-out gap: {-gap:+.5f} "
              f"({'dynamic WINS' if gap > 0 else 'cosine WINS'})")
        print(f"[ab] dynamic phases: {d['phase_counts']} "
              f"over {d['n_adjustments']} adjustments")
        print(f"[ab] dynamic mean multiplier {d['mean_multiplier']:.3f} "
              f"(min {d['min_multiplier']:.2f}, max {d['max_multiplier']:.2f})")
        if gap > 0:
            print(f"[ab] NEXT: run the control arm — --arms fixed --fixed-mult "
                  f"{d['mean_multiplier']:.3f} — to separate phase awareness "
                  f"from simply training at a higher average LR.")


if __name__ == "__main__":
    raise SystemExit(main())
