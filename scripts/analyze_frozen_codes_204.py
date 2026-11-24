"""#204: summarize results/frozen_codes_204/*.jsonl (written by profile_step_202.py).

Step time per block position = mean of turn steps 2.. (step 1 carries the BCD
switch, and with #204 the codes refresh), averaged over the 7 positions, as
in #202. Also: identity (hash) comparison, peak VRAM, host RAM, refresh cost.
"""
import json, os, sys
from collections import OrderedDict

D = sys.argv[1] if len(sys.argv) > 1 else "results/frozen_codes_204"


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
for tag in ["time_base1", "time_codes1", "time_base2", "time_codes2", "time_base3", "time_codes3"]:
    rows = load(tag)
    if not rows:
        continue
    pp = per_pos(rows)
    first = [r["sec"] for r in rows if r["pos_in_turn"] == 1]
    later = [r["sec"] for r in rows if r["pos_in_turn"] >= 2]
    sw = [r["codes"]["refresh_sec"] for r in rows if r["pos_in_turn"] == 1]
    print(f"{tag:12s} " + " ".join(f"{x:6.2f}" for x in pp) + f" | mean {sum(pp) / len(pp):.3f}"
          f" | turn-step-1 mean {sum(first) / len(first):.2f} (refresh {sum(sw[1:]) / max(1, len(sw) - 1):.2f}s,"
          f" first {sw[0]:.2f}s)"
          f" | peak {max(r['peak_res'] / r['total_mem'] for r in rows):.1%}"
          f" | ws max {max(r['host']['ws'] for r in rows) / 2**30:.2f}G private max "
          f"{max(r['host']['private'] for r in rows) / 2**30:.2f}G sys avail min "
          f"{min(r['host']['sys_avail'] for r in rows) / 2**30:.2f}G codes home "
          f"{rows[-1]['codes_home_bytes'] / 2**30:.2f}G")

print("\n== identity")
a, b, f = load("hash_base"), load("hash_codes"), load("hash_codes_falsify")
if a and b:
    n = min(len(a), len(b))
    eq_g = sum(a[i]["grad_sha"] == b[i]["grad_sha"] for i in range(n))
    eq_l = sum(a[i]["loss_sha"] == b[i]["loss_sha"] for i in range(n))
    lr0 = sum(1 for r in a[:n] if any(x > 0 for x in r["lr"]))
    print(f"base vs codes: grad equal {eq_g}/{n}, loss equal {eq_l}/{n}; steps with lr>0: {lr0}; "
          f"blocks {sorted(set(r['block'][:12] for r in a[:n]))}")
    print(f"codes hits in codes run: {b[-1]['codes_hits']}, in base run: {a[-1]['codes_hits']}")
if a and f:
    n = min(len(a), len(f))
    print(f"falsifier vs base: grad differ {sum(a[i]['grad_sha'] != f[i]['grad_sha'] for i in range(n))}/{n}, "
          f"loss differ {sum(a[i]['loss_sha'] != f[i]['loss_sha'] for i in range(n))}/{n}")

print("\n== VRAM with held-out eval after every step (peak reserved, share of card)")
for tag in ["evalv_base", "evalv_codes"]:
    rows = load(tag)
    if not rows:
        continue
    by = OrderedDict()
    for r in rows:
        t, e = by.get(r["block"], (0, 0))
        by[r["block"]] = (max(t, r["peak_res"] / r["total_mem"]), max(e, r.get("heldout_peak_res", 0) / r["total_mem"]))
    print(f"{tag:12s} train " + " ".join(f"{t:.1%}" for t, _ in by.values())
          + " | eval " + " ".join(f"{e:.1%}" for _, e in by.values())
          + f" | ws max {max(r['host_after_eval']['ws'] for r in rows if 'host_after_eval' in r) / 2**30:.2f}G")
ea, eb = load("evalv_base"), load("evalv_codes")
if ea and eb:
    n = min(len(ea), len(eb))
    print(f"held-out loss equal {sum(ea[i]['heldout'] == eb[i]['heldout'] for i in range(n))}/{n}")
