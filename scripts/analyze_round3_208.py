"""#208: summarize results/round3_208/.

  python scripts/analyze_round3_208.py time time_def_ time_t_     # step time per arm (profile_step_202 runs)
  python scripts/analyze_round3_208.py kernels prof_def            # kernel ranking per profiled step
  python scripts/analyze_round3_208.py ops prof_ops                # GPU time by autograd node / aten op+shape
  python scripts/analyze_round3_208.py ab                          # held-out loss A/B (ab/ eval_*.jsonl)

Step time = mean of turn steps 2.. per block position, averaged over the 7 positions
(the #202/#204/#206/#207 metric). Reuses scripts/analyze_round2_207.py on this directory.
"""
import glob, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import analyze_round2_207 as A

D = os.path.join(os.path.dirname(HERE), "results", "round3_208")
A.D = D


def ops(tag, top=40):
    rows = [r for r in A.load(tag) if "ops" in r]
    for key in ("nodes", "aten"):
        tot = {}
        for r in rows:
            for k, v in r["ops"][key].items():
                tot[k] = tot.get(k, 0.0) + v / len(rows)
        print(f"== {tag} {key}: mean GPU s per profiled step over {len(rows)} steps")
        for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:top]:
            print(f"{v:8.4f}  {k[:170]}")


def ab():
    ev = {}
    for p in sorted(glob.glob(os.path.join(D, "ab", "eval_*.jsonl"))):
        tag = os.path.basename(p)[5:-6]
        e = {r["step"]: r for r in map(json.loads, open(p))}
        if e:
            ev[tag] = e
    if not ev:
        print("no ab/eval_*.jsonl"); return
    steps = sorted({s for e in ev.values() for s in e})
    print("held-out NLL (64 fixed windows) per eval step")
    print(f"{'arm':22s} " + " ".join(f"{s:>9d}" for s in steps) + "   eval peak   train peak")
    for tag, e in ev.items():
        st = [json.loads(l) for l in open(os.path.join(D, "ab", f"steps_{tag}.jsonl"))]
        tp = max(r["peak_res"] / r["total_mem"] for r in st)
        epk = max((r.get("eval_peak_res", 0) for r in e.values()), default=0) / st[0]["total_mem"]
        print(f"{tag:22s} " + " ".join(f"{e[s]['heldout_nll']:9.5f}" if s in e else " " * 9 for s in steps)
              + f"   {epk:8.1%}   {tp:8.1%}")
    shas = {e[s]["target_sha"] for e in ev.values() for s in e}
    print(f"distinct held-out target sha: {len(shas)} (must be 1)")
    last = steps[-1]

    def nll(t):
        return ev[t][last]["heldout_nll"] if t in ev and last in ev[t] else None
    arms = {}
    for t in ev:
        if "_s" in t:
            arm, seed = t.rsplit("_s", 1)
            arms.setdefault(arm, {})[seed] = nll(t)
    print(f"\nfinal step {last}:")
    for arm, d in arms.items():
        v = [x for x in d.values() if x is not None]
        print(f"  {arm:14s} seeds {sorted(d)} mean {sum(v) / len(v):.5f} spread {max(v) - min(v):.5f}")
    if "def" in arms:
        dv = [x for x in arms["def"].values() if x is not None]
        spread = max(dv) - min(dv)
        mdef = sum(dv) / len(dv)
        for arm, d in arms.items():
            if arm == "def":
                continue
            v = [x for x in d.values() if x is not None]
            m = sum(v) / len(v)
            paired = [d[s] - arms["def"][s] for s in d if s in arms["def"]]
            print(f"  {arm} - def: mean diff {m - mdef:+.5f} | paired per seed "
                  + " ".join(f"{x:+.5f}" for x in paired)
                  + f" | default seed spread {spread:.5f} -> "
                  + ("WITHIN spread" if abs(m - mdef) <= spread else "OUTSIDE spread"))
    # per-window paired view for each same-seed pair
    for t in ev:
        if t.startswith("def_s"):
            continue
        seed = t.rsplit("_s", 1)[-1]
        ref = f"def_s{seed}"
        if ref in ev and last in ev[t] and last in ev[ref]:
            a_, b_ = ev[t][last]["per_window"], ev[ref][last]["per_window"]
            d = [x - y for x, y in zip(a_, b_)]
            m = sum(d) / len(d)
            sd = (sum((x - m) ** 2 for x in d) / (len(d) - 1)) ** 0.5
            print(f"  per-window {t} - {ref}: mean {m:+.5f}, sem {sd / len(d) ** 0.5:.5f}")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"kernels": lambda: A.kernels(args[0]), "time": lambda: A.time_(args),
     "ops": lambda: ops(args[0]), "ab": ab, "vram": lambda: A.vram(args)}[cmd]()
