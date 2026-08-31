"""Are there dead units in a trained checkpoint, and how many?

Author: Rambarun Komaljeet

Why
---
Dormant-neuron recycling (ReDo, Sokar et al. 2023) reinitialises units that have
stopped contributing and zeroes their outgoing weights so they rejoin gradually.
It is a real technique, but it is worth exactly nothing if this model has no dead
units -- so measure the population before building the fix.

Two distinct failure modes, and only the second is specific to this stack:

1. **Dormant activations.** A unit whose output is negligible against its layer's
   average carries no signal and receives almost no gradient. The classic cause
   is ReLU saturating at zero; this model's FFN defaults to GELU
   (`deltanet.py:145`) and its gates are SiLU/sigmoid, none of which have a hard
   zero, so the classic mechanism should be WEAK here. Measured anyway, because
   "should be" is not a measurement.

2. **Ternary collapse.** BitNet b1.58 rounds every weight to {-1, 0, +1} x scale,
   so any weight under half the scale becomes exactly zero. A whole output row
   under that threshold is a unit that contributes nothing and gets no signal
   back through it -- dead by a different route, with no activation function
   involved. **This is documented here**: #109 found fresh attention layers
   "ternary-quantized at init, collapsing them to {-1,0,1} before they had any
   chance to learn a pattern", and traced a real recall regression to it.

Falsifiability, asked before running
--------------------------------------
*"If there were no dead units, would this report that?"* Yes -- zero is a valid
and expected outcome for mode 1, and the run is still informative.

**The detector needs its own known-answer cell**, because a detector that finds
nothing is indistinguishable from a detector that is broken. Before touching the
checkpoint, `selftest_detector()` plants a known number of dead units into a
synthetic tensor and asserts the detector finds exactly those and no others. If
that fails, the run is void and the checkpoint numbers mean nothing.

Thresholds
----------
ReDo's dormancy criterion: a unit is dormant when its mean absolute activation,
divided by the layer's mean over all units, falls below `tau`. `tau=0.025` is the
paper's value and is what `--tau` defaults to. It is reported alongside a sweep,
because a single threshold invites reading a number that was chosen rather than
measured.

Usage::

    python scripts/probe_dead_units.py
    python scripts/probe_dead_units.py --checkpoint checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import pandas as _pd  # noqa: F401  (pyarrow before CUDA -- see train_edge_conversational)
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_CKPT = "checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt"


def dormant_mask(acts: torch.Tensor, tau: float) -> torch.Tensor:
    """ReDo: unit is dormant when mean|a_i| / mean_j(mean|a_j|) < tau.

    `acts` is [..., units]; all leading dims are averaged over.
    """
    per_unit = acts.abs().float().reshape(-1, acts.shape[-1]).mean(0)
    denom = per_unit.mean().clamp(min=1e-12)
    return (per_unit / denom) < tau


def selftest_detector() -> None:
    """Plant known dead units; assert the detector finds exactly those.

    A detector that reports 0% is useless unless it has been shown capable of
    reporting non-zero. This is that demonstration, and it costs milliseconds.
    """
    torch.manual_seed(0)
    n_units, n_dead = 64, 7
    a = torch.rand(512, n_units) + 0.5          # all healthy, mean ~1
    a[:, :n_dead] = 0.0                          # plant exactly n_dead dead ones
    m = dormant_mask(a, tau=0.025)
    assert int(m.sum()) == n_dead, f"planted {n_dead} dead, detector found {int(m.sum())}"
    assert m[:n_dead].all() and not m[n_dead:].any(), "detector flagged the wrong units"

    # and it must report ZERO on an all-healthy tensor
    assert int(dormant_mask(torch.rand(512, n_units) + 0.5, 0.025).sum()) == 0, \
        "detector reports dead units on healthy data"

    # ternary-collapse detector: plant fully-zero rows
    w = torch.randn(32, 16); w[3] = 0.0; w[11] = 0.0
    zero_rows = int((w.abs().sum(dim=1) == 0).sum())
    assert zero_rows == 2, f"row detector found {zero_rows}, expected 2"
    print("[dead] selftest PASS -- detector finds planted dead units and only those")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint", default=DEFAULT_CKPT)
    p.add_argument("--tau", type=float, default=0.025, help="ReDo dormancy threshold")
    p.add_argument("--batches", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--tokenizer", default="data_cache/tokenizer_32k")
    p.add_argument("--out", default="results/dead_units/probe.json")
    args = p.parse_args(argv)

    selftest_detector()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md)")
    device = torch.device("cuda")

    import train_edge_conversational as tec
    model, step, _ = tec.load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    n_params = sum(q.numel() for q in model.parameters())
    print(f"[dead] {args.checkpoint} @ step {step}, {n_params:,} params")

    # ---- 1. ternary collapse: measured from the weights alone ---------------
    from hvn.edge.ternary import TernaryLinear, ternary_quantize_ste
    tern_rows, tern_dead_rows, tern_zeros, tern_total = 0, 0, 0, 0
    per_layer = []
    for name, mod in model.named_modules():
        if not isinstance(mod, TernaryLinear):
            continue
        with torch.no_grad():
            q = ternary_quantize_ste(mod.weight.detach())
        rows = q.shape[0]
        dead = int((q.abs().sum(dim=1) == 0).sum())
        zeros = int((q == 0).sum())
        tern_rows += rows
        tern_dead_rows += dead
        tern_zeros += zeros
        tern_total += q.numel()
        if dead:
            per_layer.append({"layer": name, "rows": rows, "dead_rows": dead})

    print(f"\n[dead] ternary weights ({tern_rows:,} output rows across "
          f"{sum(1 for _ in model.named_modules() if isinstance(_[1], TernaryLinear)) if False else 'n'} layers)")
    if tern_total:
        print(f"[dead]   weights rounded to exactly 0 : "
              f"{tern_zeros:,} / {tern_total:,}  ({100*tern_zeros/tern_total:.1f}%)")
        print(f"[dead]   FULLY DEAD output rows       : "
              f"{tern_dead_rows:,} / {tern_rows:,}  ({100*tern_dead_rows/max(tern_rows,1):.2f}%)")
        for r in per_layer[:8]:
            print(f"[dead]     {r['layer']}: {r['dead_rows']}/{r['rows']}")
    else:
        print("[dead]   no TernaryLinear modules found in this checkpoint")

    # ---- 2. dormant activations on real data -------------------------------
    tok = tec.load_tokenizer(args.tokenizer)
    loader, _v, _s = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=2, data_dir="data_cache",
        val_fraction=0.0, num_workers=0)

    acts = {}

    def hook(name):
        def fn(_m, _i, out):
            o = out[0] if isinstance(out, tuple) else out
            if torch.is_tensor(o) and o.dim() >= 2:
                acts.setdefault(name, []).append(o.detach().float().cpu())
        return fn

    handles = []
    for name, mod in model.named_modules():
        # the pointwise nonlinearities and gates -- where a unit can go silent
        if isinstance(mod, (nn.GELU, nn.SiLU, nn.ReLU)):
            handles.append(mod.register_forward_hook(hook(name)))
    print(f"\n[dead] hooked {len(handles)} activation sites")

    seen = 0
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            model(x)
            seen += 1
            if seen >= args.batches:
                break
    for h in handles:
        h.remove()

    rows = []
    tot_units = tot_dorm = 0
    for name, chunks in acts.items():
        a = torch.cat(chunks, dim=0)
        m = dormant_mask(a, args.tau)
        rows.append({"site": name, "units": int(m.numel()),
                     "dormant": int(m.sum()),
                     "pct": 100 * float(m.sum()) / max(int(m.numel()), 1)})
        tot_units += int(m.numel())
        tot_dorm += int(m.sum())

    rows.sort(key=lambda r: -r["pct"])
    print(f"[dead] dormant units at tau={args.tau} (ReDo default): "
          f"{tot_dorm:,} / {tot_units:,}  ({100*tot_dorm/max(tot_units,1):.2f}%)")
    for r in rows[:6]:
        print(f"[dead]     {r['site'][:52]:<52} {r['dormant']:>5}/{r['units']:<6} "
              f"{r['pct']:>5.1f}%")

    print(f"\n[dead] threshold sweep (a single tau invites reading a chosen number)")
    sweep = {}
    for t in (0.0, 0.001, 0.01, 0.025, 0.05, 0.1):
        d = sum(int(dormant_mask(torch.cat(c, 0), t).sum()) for c in acts.values())
        sweep[t] = d
        print(f"[dead]   tau={t:<6} {d:>6,} / {tot_units:,}  "
              f"({100*d/max(tot_units,1):.2f}%)")

    # ---- reading -----------------------------------------------------------
    print(f"\n[dead] ================== READING ==================")
    tern_pct = 100 * tern_dead_rows / max(tern_rows, 1)
    dorm_pct = 100 * tot_dorm / max(tot_units, 1)
    if tern_pct < 1 and dorm_pct < 1:
        print(f"[dead] Both populations are under 1% ({tern_pct:.2f}% dead ternary "
              f"rows, {dorm_pct:.2f}% dormant units). There is nothing here to "
              f"recycle -- ReDo would be reinitialising almost nothing. DROP the "
              f"idea rather than building it.")
    else:
        worst = "ternary collapse" if tern_pct >= dorm_pct else "dormant activations"
        print(f"[dead] Non-trivial dead population, dominated by {worst}: "
              f"{tern_pct:.2f}% of ternary output rows are entirely zero, "
              f"{dorm_pct:.2f}% of activation units are dormant.")
        print(f"[dead] That is capacity paid for and not used. Recycling is worth "
              f"building -- but note it is a QUALITY lever, not a speed one: it "
              f"does not change step time, only what the parameters buy.")
    print(f"[dead] tau=0.0 gives {sweep[0.0]:,} units that are EXACTLY zero -- "
          f"those are unambiguous, no threshold argument needed.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"checkpoint": args.checkpoint, "step": step,
                   "ternary": {"rows": tern_rows, "dead_rows": tern_dead_rows,
                               "pct_dead_rows": tern_pct,
                               "pct_weights_zero": 100 * tern_zeros / max(tern_total, 1),
                               "per_layer": per_layer},
                   "dormant": {"units": tot_units, "dormant": tot_dorm,
                               "pct": dorm_pct, "tau": args.tau,
                               "sweep": {str(k): v for k, v in sweep.items()},
                               "sites": rows}}, fh, indent=2)
    print(f"[dead] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
