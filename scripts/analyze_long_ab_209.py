"""#209: summarize results/long_ab_209/ab (long held-out A/B, --scan-kernel t vs default).

  python scripts/analyze_long_ab_209.py [results/long_ab_209/ab]

Decision rule (fixed before the runs): t is recommended as a default only if BOTH
  |mean(t) - mean(def)| at the final step, and
  |mean(t) - mean(def)| of the mean over the last 5 turn-end evals
are within the default arm's own seed spread |def_s0 - def_s1| (same quantity).
"""
import glob, json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "long_ab_209", "ab")


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


ev, st = {}, {}
for p in sorted(glob.glob(os.path.join(D, "eval_*.jsonl"))):
    tag = os.path.basename(p)[5:-6]
    ev[tag] = {r["step"]: r for r in jl(p)}
    st[tag] = jl(os.path.join(D, f"steps_{tag}.jsonl"))
tags = [t for t in ("def_s0", "t_s0", "def_s1", "t_s1", "tbroken_s0") if t in ev] + \
       [t for t in ev if t not in ("def_s0", "t_s0", "def_s1", "t_s1", "tbroken_s0")]

steps = sorted({s for e in ev.values() for s in e})
print("held-out NLL (64 fixed windows) at each eval step")
print("step   " + " ".join(f"{t:>10s}" for t in tags))
for s in steps:
    print(f"{s:5d}  " + " ".join(f"{ev[t][s]['heldout_nll']:10.5f}" if s in ev[t] else " " * 10
                                for t in tags))
shas = {e[s]["target_sha"] for e in ev.values() for s in e}
print(f"distinct held-out target sha: {len(shas)} (must be 1)")

print("\nper run: steps done, mean s/step (all steps but each process's first), "
      "peak VRAM train / eval (% of card), resumes")
tot = None
for t in tags:
    rows = st[t]
    if not rows:
        continue
    tot = rows[0]["total_mem"]
    res = jl(os.path.join(D, f"resumes_{t}.jsonl"))
    first = {1} | {r["step"] + 1 for r in res}
    secs = [r["sec"] for r in rows if r["step"] not in first]
    tp = max(r["peak_res"] for r in rows) / tot
    epk = max((r["eval_peak_res"] for r in ev[t].values()), default=0) / tot
    print(f"  {t:12s} steps {rows[-1]['step']:5d}  {sum(secs) / max(1, len(secs)):6.3f} s/step  "
          f"train peak {tp:6.1%}  eval peak {epk:6.1%}  resumes {[r['step'] for r in res]}")
    print(f"    scan kernel(s) seen: {sorted({str(r['scan_kernel']) for r in rows})}  "
          f"last t_stats {rows[-1]['t_stats']}  wall {(rows[-1]['wall'] - rows[0]['wall']) / 3600:.2f} h")
    turns = {}
    for r in rows:
        turns.setdefault((r["step"] - 1) // 100, []).append(r)
    print("    per-turn peak %: " + " ".join(f"{max(x['peak_res'] for x in v) / tot * 100:.1f}"
                                         for _, v in sorted(turns.items())))
    print("    per-turn s/step: " + " ".join(
        f"{sum(x['sec'] for x in v if x['step'] not in first) / max(1, sum(1 for x in v if x['step'] not in first)):.2f}"
        for _, v in sorted(turns.items())))


def metric(t, upto=None):
    e = ev.get(t, {})
    ends = [s for s in sorted(e) if s % 100 == 0 and (upto is None or s <= upto)]
    if not ends:
        return None, None, None
    last5 = ends[-5:]
    return ends[-1], e[ends[-1]]["heldout_nll"], sum(e[s]["heldout_nll"] for s in last5) / len(last5)


print()
M = {t: metric(t) for t in tags}
for t in tags:
    print(f"  {t:12s} final step {M[t][0]}  final {M[t][1]}  mean last 5 turn-ends {M[t][2]}")

if all(t in M and M[t][0] is not None for t in ("def_s0", "def_s1", "t_s0", "t_s1")):
    fin = {t: M[t][0] for t in ("def_s0", "def_s1", "t_s0", "t_s1")}
    if len(set(fin.values())) != 1:
        print(f"!! runs ended at different steps {fin}; comparing at the common step")
        common = min(fin.values())
        M.update({t: metric(t, common) for t in fin})
    for i, name in ((1, "final"), (2, "last-5 mean")):
        d0, d1, t0, t1 = (M[t][i] for t in ("def_s0", "def_s1", "t_s0", "t_s1"))
        spread, tspread = abs(d0 - d1), abs(t0 - t1)
        diff = (t0 + t1) / 2 - (d0 + d1) / 2
        print(f"\n{name}: def mean {(d0 + d1) / 2:.5f} spread {spread:.5f} | t mean {(t0 + t1) / 2:.5f} "
              f"spread {tspread:.5f} | t - def {diff:+.5f} (paired s0 {t0 - d0:+.5f}, s1 {t1 - d1:+.5f})"
              f" -> {'WITHIN' if abs(diff) <= spread else 'OUTSIDE'} the default seed spread")
    ok = all(abs((M['t_s0'][i] + M['t_s1'][i]) / 2 - (M['def_s0'][i] + M['def_s1'][i]) / 2)
             <= abs(M['def_s0'][i] - M['def_s1'][i]) for i in (1, 2))
    print(f"VERDICT (fixed rule): {'PASS -- t within the default seed spread on both' if ok else 'FAIL'}")
    for s in ("0", "1"):
        a, b = ev[f"t_s{s}"], ev[f"def_s{s}"]
        last = M[f"t_s{s}"][0]
        d = [x - y for x, y in zip(a[last]["per_window"], b[last]["per_window"])]
        m = sum(d) / len(d)
        sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
        print(f"  per-window t_s{s} - def_s{s} at step {last}: mean {m:+.5f}, sem {sd / len(d) ** 0.5:.5f}")

if "tbroken_s0" in ev:
    print("\nfalsifier (scan backward returns zero gradients), same 1800-step schedule:")
    for s in sorted(ev["tbroken_s0"]):
        line = f"  step {s:5d} tbroken {ev['tbroken_s0'][s]['heldout_nll']:.5f}"
        for ref in ("def_s0", "t_s0"):
            if s in ev.get(ref, {}):
                line += f" | - {ref} {ev['tbroken_s0'][s]['heldout_nll'] - ev[ref][s]['heldout_nll']:+.5f}"
        if s in ev.get("def_s0", {}) and s in ev.get("def_s1", {}):
            line += f" | def seed spread {abs(ev['def_s0'][s]['heldout_nll'] - ev['def_s1'][s]['heldout_nll']):.5f}"
        print(line)
    print(f"  broken backward calls at the last step: {st['tbroken_s0'][-1]['broken_bwd_calls'] if st['tbroken_s0'] else None}")
