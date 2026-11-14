"""#206: summarize results/scan_study/ step-level runs (profile_step_202.py jsonl) and
the d512 loss A/B (ab_local_loss.py json).

Step time per block position = mean of turn steps 2.. (step 1 carries the BCD
switch), averaged over the 7 positions, as in #202/#204.
"""
import json, os, sys, glob
from collections import OrderedDict

D = sys.argv[1] if len(sys.argv) > 1 else "results/scan_study"


def load(tag):
    p = os.path.join(D, tag + ".jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def per_pos(rows):
    by = OrderedDict()
    for r in rows:
        if r["pos_in_turn"] >= 2:
            by.setdefault(r["block"], []).append(r["sec"])
    return [sum(v) / len(v) for v in by.values()]


print("== step time (s): 7 block positions, mean of turn steps 2-3")
means = {}
for r_ in (1, 2, 3):
    for arm in ("base", "v1", "t"):
        tag = f"time_{arm}{r_}"
        rows = load(tag)
        if not rows:
            continue
        pp = per_pos(rows)
        m = sum(pp) / len(pp)
        means.setdefault(arm, []).append(m)
        print(f"{tag:12s} " + " ".join(f"{x:6.2f}" for x in pp) + f" | mean {m:.3f}"
              f" | peak {max(r['peak_res'] / r['total_mem'] for r in rows):.1%}")
for arm, v in means.items():
    print(f"{arm:5s} mean of repeats {sum(v) / len(v):.3f}  spread {max(v) - min(v):.3f}  ({len(v)} runs)")

print("\n== identity (sha256 of per-token loss and active-block grads per step)")
base = load("hash_base")
for tag in ("hash_v1", "hash_v1_falsify", "hash_t"):
    o = load(tag)
    if not (base and o):
        continue
    n = min(len(base), len(o))
    eq_g = sum(base[i]["grad_sha"] == o[i]["grad_sha"] for i in range(n))
    eq_l = sum(base[i]["loss_sha"] == o[i]["loss_sha"] for i in range(n))
    print(f"base vs {tag}: grad equal {eq_g}/{n}, loss equal {eq_l}/{n}")
ref204 = os.path.join(os.path.dirname(D), "frozen_codes_204", "hash_codes.jsonl")
if base and os.path.exists(ref204):
    r = [json.loads(l) for l in open(ref204)]
    n = min(len(base), len(r))
    print(f"base vs #204 hash_codes: grad equal {sum(base[i]['grad_sha'] == r[i]['grad_sha'] for i in range(n))}/{n}, "
          f"loss equal {sum(base[i]['loss_sha'] == r[i]['loss_sha'] for i in range(n))}/{n}")

print("\n== VRAM with held-out eval after every step (peak reserved, share of card)")
ev = {}
for arm in ("base", "v1", "t"):
    rows = load("evalv_" + arm)
    if not rows:
        continue
    ev[arm] = rows
    by = OrderedDict()
    for r in rows:
        t, e = by.get(r["block"], (0, 0))
        by[r["block"]] = (max(t, r["peak_res"] / r["total_mem"]),
                          max(e, r.get("heldout_peak_res", 0) / r["total_mem"]))
    print(f"evalv_{arm:5s} train " + " ".join(f"{t:.1%}" for t, _ in by.values())
          + " | eval " + " ".join(f"{e:.1%}" for _, e in by.values()))
if "base" in ev:
    for arm in ("v1", "t"):
        if arm in ev:
            a, b = ev["base"], ev[arm]
            n = min(len(a), len(b))
            eq = sum(a[i]["heldout"] == b[i]["heldout"] for i in range(n))
            dmax = max(abs(a[i]["heldout"] - b[i]["heldout"]) for i in range(n))
            print(f"held-out loss base vs {arm}: equal {eq}/{n}, max |diff| {dmax:.2e}")

print("\n== d512 loss A/B (ab_local_loss, final held-out, 16 fixed probe batches)")
rows = {}
for p in sorted(glob.glob(os.path.join(D, "lossab_*.json"))):
    tag = os.path.basename(p)[:-5]
    j = json.load(open(p))
    for arm, r in j["arms"].items():
        rows[(tag, arm)] = r
        traj = " ".join(f"{t['heldout_loss']:.4f}" for t in r["trajectory"][1:])
        med = sorted(r["median_step_sec_by_block"].values())
        print(f"{tag:22s} {arm:7s} final {r['final_heldout_loss']:.5f} | median step "
              f"{med[len(med) // 2] * 1e3:.0f} ms | train {r['train_sec']:.0f} s | curve {traj}")
for seed in ("s7", "s8"):
    for arm in ("bcd_k2", "full"):
        a, b = rows.get((f"lossab_{seed}_v1", arm)), rows.get((f"lossab_{seed}_t", arm))
        if a and b:
            dd = [y["heldout_loss"] - x["heldout_loss"] for x, y in zip(a["trajectory"], b["trajectory"])]
            print(f"t - v1, {seed} {arm:7s}: final {b['final_heldout_loss'] - a['final_heldout_loss']:+.5f} "
                  f"| max |diff| over curve {max(abs(x) for x in dd):.5f}")
    a, b = rows.get((f"lossab_{seed}_v1", "bcd_k2")), rows.get((f"lossab_{seed}_v1_repeat", "bcd_k2"))
for arm in ("bcd_k2", "full"):
    a, b = rows.get(("lossab_s7_v1", arm)), rows.get(("lossab_s7_v1_repeat", arm))
    if a and b:
        print(f"v1 same-seed repeat, s7 {arm:7s}: final diff {b['final_heldout_loss'] - a['final_heldout_loss']:+.5f}")
    a, b = rows.get(("lossab_s7_v1", arm)), rows.get(("lossab_s8_v1", arm))
    if a and b:
        print(f"v1 seed 7 vs seed 8, {arm:7s}: final diff {b['final_heldout_loss'] - a['final_heldout_loss']:+.5f}")
