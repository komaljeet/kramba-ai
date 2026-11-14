"""#205 step 4 table: held-out at equal steps, turns, and steps to reach arm A's final loss."""
import json
import os
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "results/weight_law_205"


def load(n):
    p = os.path.join(R, n, "summary.json")
    return json.load(open(p)) if os.path.exists(p) else None


def first_step_below(traj, target):
    prev = None
    for x in traj:
        if x["heldout"] <= target:
            if prev is None:
                return x["step"]
            f = (prev["heldout"] - target) / max(prev["heldout"] - x["heldout"], 1e-12)
            return prev["step"] + f * (x["step"] - prev["step"])
        prev = x
    return None


rows = []
hashes = set()
for seed in (7, 8, 9):
    A = load(f"A_s{seed}")
    for arm in ("A", "B", "C"):
        s = load(f"{arm}_s{seed}")
        if s is None:
            continue
        hashes.add((s["hashes"]["heldout"], s["hashes"]["val"], s["hashes"]["train_window"]))
        h = {x["step"]: x["heldout"] for x in s["heldout_traj"]}
        reach = first_step_below(s["heldout_traj"], A["final_heldout"]) if A else None
        rows.append({"arm": arm, "seed": seed, "final_heldout": s["final_heldout"],
                     "vs_A_same_seed": s["final_heldout"] - A["final_heldout"] if A else None,
                     "heldout_1400": h.get(1400), "heldout_2100": h.get(2100),
                     "turns": s["n_turns"], "mean_turn_len": s["mean_turn_len_completed"],
                     "step_reaching_A_final": reach, "wall_min": s["wall_min"]})
rec = load("record_s7")
out = {"rows": rows, "same_sets_all_arms": len(hashes) == 1,
       "record_s7_final": rec["final_heldout"] if rec else None}
if rec and load("A_s7"):
    out["same_seed_rerun_diff_A_s7_vs_record"] = load("A_s7")["final_heldout"] - rec["final_heldout"]
if load("A_s7") and load("A_s8"):
    out["seed_diff_A"] = load("A_s8")["final_heldout"] - load("A_s7")["final_heldout"]
json.dump(out, open(os.path.join(R, "ab_table.json"), "w"), indent=1)
print(f"same held-out/val/train sha in all arms: {out['same_sets_all_arms']}")
print("arm seed  final    vs A    @1400   @2100  turns mean_len reachA wall")
for r in rows:
    print(f"{r['arm']:>3} {r['seed']:>4} {r['final_heldout']:.4f} {r['vs_A_same_seed']:+.4f} "
          f"{r['heldout_1400']:.4f} {r['heldout_2100']:.4f} {r['turns']:>5} "
          f"{r['mean_turn_len']:>7.1f} {r['step_reaching_A_final'] or float('nan'):>6.0f} "
          f"{r['wall_min']:.1f}")
for k in ("record_s7_final", "same_seed_rerun_diff_A_s7_vs_record", "seed_diff_A"):
    if k in out:
        print(k, f"{out[k]:+.4f}" if "diff" in k else f"{out[k]:.4f}")
