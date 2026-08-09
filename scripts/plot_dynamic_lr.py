"""Plot the LR multiplier against the loss curve it reacted to (Issue #133).

Written by: Rambarun Komaljeet

The point of this figure is not decoration. The #133 spec asks to "visually
confirm the scheduler is reacting to real loss events not noise", which needs
the multiplier and the loss on a shared x-axis with the *noise band* drawn in.
A multiplier step that lines up with a wiggle inside the band is the scheduler
chasing noise; one that lines up with an excursion outside it is a reaction to
a real event.

Three panels, shared x:
  1. training loss (raw, faint) + its 100-step rolling mean + a +/-1 sigma band
  2. the LR multiplier, with each adjustment marked and coloured by phase
  3. the effective LR actually used (schedule x multiplier) vs the schedule alone

Held-out probe loss for every arm is overlaid on panel 1 as markers, since that
-- not the training loss -- is what the arms are graded on.

Usage
-----
    python scripts/plot_dynamic_lr.py \
        --ab results/dynamic_lr/ab_dynamic_lr.json \
        --out results/dynamic_lr/curves.png
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

PHASE_COLORS = {
    "spike": "#d62728",
    "plateau": "#ff7f0e",
    "transition": "#2ca02c",
    "post_transition": "#9467bd",
    "dropping_fast": "#1f77b4",
    "normal": "#7f7f7f",
    "cooldown": "#bcbd22",
}
ARM_COLORS = {"cosine": "#1f77b4", "dynamic": "#d62728"}


def rolling(x: np.ndarray, w: int) -> np.ndarray:
    if len(x) < w:
        return np.full_like(x, np.nan, dtype=float)
    c = np.convolve(x, np.ones(w) / w, mode="valid")
    return np.concatenate([np.full(w - 1, np.nan), c])


def rolling_std(x: np.ndarray, w: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    for i in range(w - 1, len(x)):
        out[i] = x[i - w + 1:i + 1].std(ddof=1)
    return out


def arm_color(name: str) -> str:
    return ARM_COLORS.get(name, "#8c564b")


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--ab", default="results/dynamic_lr/ab_dynamic_lr.json")
    p.add_argument("--lr-log", default="results/dynamic_lr/lr_log.json")
    p.add_argument("--out", default="results/dynamic_lr/curves.png")
    p.add_argument("--window", type=int, default=100)
    args = p.parse_args(argv)

    with open(args.ab, encoding="utf-8") as fh:
        ab = json.load(fh)
    arms: Dict[str, dict] = ab["arms"]
    if "dynamic" not in arms:
        raise SystemExit("no 'dynamic' arm in the A/B file — nothing to plot")
    dyn = arms["dynamic"]

    adjustments: List[dict] = []
    if os.path.exists(args.lr_log):
        with open(args.lr_log, encoding="utf-8") as fh:
            adjustments = json.load(fh).get("adjustments", [])

    curve = dyn["train_curve"]
    steps = np.array([c["step"] for c in curve], dtype=float)
    loss = np.array([c["loss"] for c in curve], dtype=float)
    mult = np.array([c["mult"] for c in curve], dtype=float)
    eff = np.array([c["eff_lr"] for c in curve], dtype=float)
    sched = np.array([c["sched_lr"] for c in curve], dtype=float)

    mean = rolling(loss, args.window)
    sd = rolling_std(loss, args.window)

    fig, (ax0, ax1, ax2) = plt.subplots(
        3, 1, figsize=(13, 11), sharex=True,
        gridspec_kw={"height_ratios": [3, 2, 2], "hspace": 0.12})

    # --- panel 1: the loss the scheduler was reacting to -------------------
    ax0.plot(steps, loss, color="#cccccc", lw=0.6, zorder=1,
             label="train loss (raw, per step)")
    ax0.plot(steps, mean, color="#333333", lw=1.6, zorder=3,
             label=f"rolling mean ({args.window} steps)")
    ax0.fill_between(steps, mean - sd, mean + sd, color="#999999", alpha=0.25,
                     zorder=2, label="+/-1 sigma noise band")
    for name, r in arms.items():
        t = r["trajectory"]
        ax0.plot([q["step"] for q in t], [q["heldout_loss"] for q in t],
                 "o-", ms=4, lw=1.8, color=arm_color(name), zorder=4,
                 label=f"{name}: held-out probe loss (GRADED ON THIS)")
    ax0.set_ylabel("loss (nats)")
    ax0.legend(loc="upper right", fontsize=8, ncol=2)
    ax0.set_title(
        f"Dynamic phase-aware LR vs fixed cosine — EdgeHVN 83M, "
        f"{ab['steps']:,} steps from {os.path.basename(ab['base_checkpoint'])}\n"
        f"multiplier reacts to the black rolling mean; a reaction inside the "
        f"grey band is noise-chasing, outside it is a real event",
        fontsize=11)
    ax0.grid(alpha=0.25)

    # --- panel 2: the multiplier and every decision behind it --------------
    ax1.plot(steps, mult, color="#d62728", lw=1.8, drawstyle="steps-post",
             label="LR multiplier")
    ax1.axhline(1.0, color="#666666", ls=":", lw=1)
    ax1.axhline(3.0, color="#aaaaaa", ls="--", lw=0.8)
    ax1.axhline(0.1, color="#aaaaaa", ls="--", lw=0.8)
    seen = set()
    for a in adjustments:
        if a["multiplier_after"] == a["multiplier_before"]:
            continue                     # a check that changed nothing
        ph = a["phase"]
        ax1.axvline(a["step"], color=PHASE_COLORS.get(ph, "#000000"),
                    alpha=0.55, lw=1.0,
                    label=(ph if ph not in seen else None))
        seen.add(ph)
    ax1.set_ylabel("multiplier (x base LR)")
    ax1.set_ylim(0, 3.2)
    if seen:
        ax1.legend(loc="upper left", fontsize=8, ncol=4, title="adjustment fired")
    else:
        ax1.legend(loc="upper left", fontsize=8)
    ax1.grid(alpha=0.25)

    # --- panel 3: what the optimizer actually saw -------------------------
    ax2.plot(steps, sched, color="#1f77b4", lw=1.6, label="cosine schedule LR")
    ax2.plot(steps, eff, color="#d62728", lw=1.6,
             label="effective LR (schedule x multiplier)")
    ax2.set_yscale("log")
    ax2.set_ylabel("learning rate (log)")
    ax2.set_xlabel("step")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.grid(alpha=0.25, which="both")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=140, bbox_inches="tight")
    print(f"wrote {args.out}")

    # A number for the "is it noise?" question, not only a picture: how far
    # outside the local sigma band the curve was at each adjustment.
    if adjustments:
        by_step = {int(s): i for i, s in enumerate(steps)}
        zs = []
        for a in adjustments:
            i = by_step.get(int(a["step"]))
            if i is None or np.isnan(sd[i]) or sd[i] <= 0:
                continue
            zs.append(abs(loss[i] - mean[i]) / sd[i])
        if zs:
            print(f"adjustments: {len(adjustments)}, "
                  f"median |loss - rolling mean| / sigma at the moment of "
                  f"adjustment = {np.median(zs):.2f} "
                  f"({sum(z > 1 for z in zs)}/{len(zs)} outside 1 sigma)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
