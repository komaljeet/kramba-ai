"""#200 (A): summarise results/bcd_stall_200/stall_steps.jsonl and switch.jsonl.

Author: Rambarun Komaljeet

Per mode (normal / sync / slow) and active block: unprofiled step time (2nd step of
the turn), profiled step time, and from the profiled steps: compute-stream busy,
stall (= HtoD on the compute stream + compute idle waiting on a side-stream copy),
idle not explained by copies, GB moved and achieved GB/s. Then the positive-control
check: the stall must rise by about the step-time rise when copies are forced onto
the critical path. Switch cost: wall time of ``_activate`` per switch.
"""
import argparse, json, os, statistics as S


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/bcd_stall_200")
    x = ap.parse_args()
    import glob
    rows, sw = [], []
    for f in sorted(glob.glob(os.path.join(x.dir, "stall_steps_*.jsonl"))):
        rows += [json.loads(l) for l in open(f) if l.strip()]
    for f in sorted(glob.glob(os.path.join(x.dir, "switch_*.jsonl"))):
        sw += [json.loads(l) for l in open(f) if l.strip()]
    blocks = list(dict.fromkeys(r["block"] for r in rows))
    modes = list(dict.fromkeys(r["mode"] for r in rows))
    agg = {}
    print(f"{'mode':<7} {'block':<16} {'unprof':>7} {'prof':>7} {'busy':>7} {'stall':>6} "
          f"{'stall%':>6} {'wait':>6} {'wait_lo':>7} {'h2d_cmp':>7} {'idle_o':>6} "
          f"{'GB/step':>7} {'GB/s':>5} {'host_pf':>7}")
    for m in modes:
        for b in blocks:
            R = [r for r in rows if r["mode"] == m and r["block"] == b]
            P = [r for r in R if r.get("trace")]
            U = [r["sec"] for r in R if r["pos_in_turn"] == 2]
            if not P:
                continue
            f = lambda k: S.mean(r["trace"][k] for r in P)
            sec = S.mean(r["sec"] for r in P)
            gb = f("side_h2d_gb") + f("h2d_on_compute_gb")
            dur = f("side_h2d_dur") + f("h2d_on_compute")
            a = {"unprof": S.mean(U) if U else float("nan"), "prof": sec, "busy": f("busy"),
                 "stall": f("stall"), "wait": f("copy_wait"), "wait_lo": f("copy_wait_loose"),
                 "h2d": f("h2d_on_compute"), "idle_o": f("idle_other"), "gb": gb,
                 "gbps": gb / dur if dur else 0, "host_pf": S.mean(r["host"]["prefetch"] for r in P)}
            agg[(m, b)] = a
            print(f"{m:<7} {b[:16]:<16} {a['unprof']:>7.2f} {a['prof']:>7.2f} {a['busy']:>7.2f} "
                  f"{a['stall']:>6.3f} {100 * a['stall'] / sec:>5.2f}% {a['wait']:>6.3f} "
                  f"{a['wait_lo']:>7.3f} {a['h2d']:>7.3f} {a['idle_o']:>6.3f} {gb:>7.1f} "
                  f"{a['gbps']:>5.1f} {a['host_pf']:>7.2f}")
    for m in modes:
        A = [agg[(m, b)] for b in blocks if (m, b) in agg]
        if A:
            st, pr = S.mean(a["stall"] for a in A), S.mean(a["prof"] for a in A)
            print(f"  {m}: mean over blocks: step {pr:.2f} s, stall {st:.3f} s ({100 * st / pr:.2f}%), "
                  f"idle other {S.mean(a['idle_o'] for a in A):.3f} s")
    if "normal" in modes:
        print("\npositive controls (per block: step-time rise vs stall rise, vs normal)")
        for m in [m for m in modes if m != "normal"]:
            for b in blocks:
                if (m, b) in agg and ("normal", b) in agg:
                    n, c = agg[("normal", b)], agg[(m, b)]
                    ds, dt = c["stall"] - n["stall"], c["prof"] - n["prof"]
                    print(f"  {m:<5} {b[:16]:<16} step +{dt:6.3f} s, stall +{ds:6.3f} s "
                          f"({100 * ds / dt if dt > 0.05 else float('nan'):.0f}% of the rise)")
    print("\nswitch cost (_activate wall time, synchronized)")
    for m in dict.fromkeys(s["mode"] for s in sw):
        v = [s["activate_sec"] for s in sw if s["mode"] == m]
        print(f"  {m}: n={len(v)} median {S.median(v):.3f} s, max {max(v):.3f} s")
    first = {r["block"] + r["mode"]: r["sec"] for r in rows if r["pos_in_turn"] == 1 and r["step"] > 1}
    ex = []
    for r in rows:
        if r["pos_in_turn"] == 2 and r["mode"] == "normal" and r["block"] + r["mode"] in first:
            ex.append(first[r["block"] + r["mode"]] - r["sec"])
    if ex:
        print(f"  first step of a turn minus second (normal mode, incl. _activate): "
              f"median {S.median(ex):.3f} s, range {min(ex):.3f}..{max(ex):.3f}")


if __name__ == "__main__":
    main()
