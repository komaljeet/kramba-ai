"""#205 step 2: fit laws of motion to each recorded BCD turn.

Quantities per turn (t = steps into the turn):
  d_end(t)   = ||w_t - w_end||    (active block, every 5 steps)
  d_start(t) = ||w_t - w_0||
  heldout(t) = held-out loss (16 fixed batches), every 10 steps
  val(t)     = val loss (8 other fixed batches), every 10 steps

Forms, all with a free offset c:  y = c + A * g(t)
  linear      g = t                         (null, 2 params)
  exp         g = exp(-t/tau)               (3)
  power       g = (t + 1)^p, p of either sign  (3)
  stretched   g = exp(-(t/tau)^beta)        (4)

Scores:
  interp  fit on even-indexed points, RMSE on the odd-indexed (hidden) points
  forecast fit on t <= 50, RMSE on t > 50 (what an early-switch rule needs)
Output: results/weight_law_205/fits.json and fits_per_turn.csv
"""
import csv
import json
import math
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")

FORMS = {
    "linear": (lambda t, c, A: c + A * t, 2),
    "exp": (lambda t, c, A, tau: c + A * np.exp(-t / tau), 3),
    "power": (lambda t, c, A, p: c + A * np.power(t + 1.0, p), 3),
    "stretched": (lambda t, c, A, tau, b: c + A * np.exp(-np.power(t / tau, b)), 4),
}
# Every form is linear in (c, A) once its shape parameters are fixed, so the fit is a
# dense grid over the shape parameters with closed-form least squares for (c, A).
# (scipy is not installed in this venv; a grid also cannot get stuck in a local minimum.)
TAUS = np.geomspace(0.5, 1e4, 240)
PS = np.concatenate([-np.geomspace(1e-3, 8.0, 120), np.geomspace(1e-3, 8.0, 120)])
BETAS = np.geomspace(0.1, 5.0, 40)
GRIDS = {"linear": [()], "exp": [(x,) for x in TAUS], "power": [(x,) for x in PS],
         "stretched": [(x, b) for x in TAUS[::3] for b in BETAS]}
GFUN = {"linear": lambda t: t, "exp": lambda t, tau: np.exp(-t / tau),
        "power": lambda t, p: np.power(t + 1.0, p),
        "stretched": lambda t, tau, b: np.exp(-np.power(t / tau, b))}


def fit(name, t, y):
    f, k = FORMS[name]
    if len(t) <= k:
        return None
    best = None
    for sp in GRIDS[name]:
        g = GFUN[name](t, *sp)
        X = np.stack([np.ones_like(t), g], 1)
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = float(np.sqrt(np.mean((X @ coef - y) ** 2)))
        if best is None or r < best[1]:
            best = (np.array([coef[0], coef[1], *sp]), r)
    return best


def score(t, y):
    t, y = np.asarray(t, float), np.asarray(y, float)
    out = {}
    ev, od = np.arange(0, len(t), 2), np.arange(1, len(t), 2)
    early = t <= 50
    for name, (f, k) in FORMS.items():
        r = {}
        b = fit(name, t[ev], y[ev])
        r["interp_rmse"] = (float(np.sqrt(np.mean((f(t[od], *b[0]) - y[od]) ** 2)))
                            if b is not None else None)
        b = fit(name, t[early], y[early])
        r["forecast_rmse"] = (float(np.sqrt(np.mean((f(t[~early], *b[0]) - y[~early]) ** 2)))
                              if b is not None else None)
        b = fit(name, t, y)
        r["params"] = list(map(float, b[0])) if b is not None else None
        r["full_rmse"] = b[1] if b is not None else None
        out[name] = r
    return out


def main(run_dir):
    turns = [json.loads(l) for l in open(os.path.join(run_dir, "turns.jsonl"))]
    turns = [tr for tr in turns if tr["length"] == 100]
    rows, per = [], []
    for tr in turns:
        rec = {"turn": tr["turn"], "block": tr["block"], "block_idx": tr["block_idx"],
               "rotation": tr["rotation"], "start_step": tr["start_step"],
               "lr_start": tr["lr_start"], "lr_mean": float(np.mean(tr["lr"]))}
        series = {"d_end": (tr["snap_t"], tr["d_end"]),
                  "d_start": (tr["snap_t"], tr["d_start"]),
                  "heldout": (tr["heldout_t"], tr["heldout"]),
                  "val": (tr["val_t"], tr["val"])}
        rec["fits"] = {k: score(*v) for k, v in series.items()}
        h = dict(zip(tr["heldout_t"], tr["heldout"]))
        rec["heldout_drop_total"] = h[0] - h[100]
        rec["heldout_drop_0_30"] = h[0] - h[30]
        rec["heldout_drop_70_100"] = h[70] - h[100]
        rec["heldout_drop_50_100"] = h[50] - h[100]
        rec["heldout_min_t"] = min(h, key=h.get)
        # straightness: displacement / path length (1 = straight line)
        path = sum(tr["upd_norm"])
        rec["straightness"] = tr["d_start"][-1] / path if path else None
        rec["cos5_mean"] = float(np.mean(tr["cos5"]))
        rec["upd_norm_first10"] = float(np.mean(tr["upd_norm"][:10]))
        rec["upd_norm_last10"] = float(np.mean(tr["upd_norm"][-10:]))
        rec["train_loss_first10"] = float(np.mean(tr["train_loss"][:10]))
        rec["train_loss_last10"] = float(np.mean(tr["train_loss"][-10:]))
        per.append(rec)
        for q, fs in rec["fits"].items():
            for name, r in fs.items():
                rows.append({"turn": tr["turn"], "block": tr["block"], "rotation": tr["rotation"],
                             "quantity": q, "form": name, "interp_rmse": r["interp_rmse"],
                             "forecast_rmse": r["forecast_rmse"], "full_rmse": r["full_rmse"],
                             "params": json.dumps(r["params"])})

    # summary: per quantity, median RMSE by form, wins, ratio to linear
    summ = {}
    for q in ("d_end", "d_start", "heldout", "val"):
        s = {}
        for mode in ("interp_rmse", "forecast_rmse"):
            med = {}
            wins = {n: 0 for n in FORMS}
            ratio = {}
            for name in FORMS:
                v = [r["fits"][q][name][mode] for r in per if r["fits"][q][name][mode] is not None]
                med[name] = float(np.median(v)) if v else None
                ratio[name] = float(np.median([
                    r["fits"][q][name][mode] / r["fits"][q]["linear"][mode] for r in per
                    if r["fits"][q][name][mode] is not None
                    and r["fits"][q]["linear"][mode]]))
            for r in per:
                cand = {n: r["fits"][q][n][mode] for n in FORMS
                        if r["fits"][q][n][mode] is not None}
                wins[min(cand, key=cand.get)] += 1
            s[mode] = {"median_rmse": med, "wins": wins, "median_ratio_to_linear": ratio}
        summ[q] = s
    # rate (exp tau of heldout and d_start) vs block, rotation, lr
    taus = []
    for r in per:
        taus.append({"turn": r["turn"], "block_idx": r["block_idx"], "rotation": r["rotation"],
                     "lr_mean": r["lr_mean"],
                     "tau_heldout": (r["fits"]["heldout"]["exp"]["params"] or [None] * 3)[2],
                     "tau_d_start": (r["fits"]["d_start"]["exp"]["params"] or [None] * 3)[2],
                     "beta_d_start": (r["fits"]["d_start"]["stretched"]["params"] or [None] * 4)[3],
                     "p_d_start_power_growth": None,
                     "drop_total": r["heldout_drop_total"], "drop_70_100": r["heldout_drop_70_100"],
                     "drop_0_30": r["heldout_drop_0_30"], "min_t": r["heldout_min_t"],
                     "straightness": r["straightness"], "cos5_mean": r["cos5_mean"]})
    json.dump({"summary": summ, "per_turn": per, "rates": taus},
              open(os.path.join(run_dir, "fits.json"), "w"), indent=1)
    with open(os.path.join(run_dir, "fits_per_turn.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    with open(os.path.join(run_dir, "turn_rates.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(taus[0]))
        w.writeheader(); w.writerows(taus)

    print(f"{len(per)} complete turns")
    for q, s in summ.items():
        for mode, v in s.items():
            print(f"{q:8s} {mode:14s} median RMSE " +
                  " ".join(f"{n}={v['median_rmse'][n]:.4g}" for n in FORMS) +
                  " | wins " + " ".join(f"{n}={v['wins'][n]}" for n in FORMS) +
                  " | ratio-to-linear " + " ".join(
                      f"{n}={v['median_ratio_to_linear'][n]:.2f}" for n in FORMS))
    print("\nturn blk rot  lr_mean  drop0-100 drop0-30 drop70-100 min_t tau_ho  tau_dst beta  straight cos5")
    for x in taus:
        print(f"{x['turn']:>4} {x['block_idx']:>3} {x['rotation']:>3} {x['lr_mean']:.2e} "
              f"{x['drop_total']:>9.4f} {x['drop_0_30']:>8.4f} {x['drop_70_100']:>9.4f} "
              f"{x['min_t']:>5} {x['tau_heldout'] or float('nan'):>7.1f} "
              f"{x['tau_d_start'] or float('nan'):>7.1f} {x['beta_d_start'] or float('nan'):>5.2f} "
              f"{x['straightness']:>8.3f} {x['cos5_mean']:>5.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
