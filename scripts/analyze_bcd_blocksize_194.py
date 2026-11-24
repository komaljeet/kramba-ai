"""#194: compare the k2 and k4 arms of scripts/exp_bcd_blocksize_194.py.

Author: Rambarun Komaljeet

Usage: python scripts/analyze_bcd_blocksize_194.py [--dir results/bcd_blocksize_194]
       [--pairs k2:k4,k4:k5,k2:k5]   (gap = a - b per pair)

Prints, from the logs only (works on partial runs):
  1. sanity: same held-out tokens in both arms (target sha, also vs #197), same
     step-1 loss (both arms start from the same init);
  2. held-out loss at every step both arms evaluated: a - b (positive = b better),
     exp(gap), and the paired per-window standard error of the gap (eval sampling
     error only -- NOT the run-to-run training noise, which #197 put at ~0.006 nats
     after 180 steps with one sample and is not re-measured here);
  3. equal training time: b's held-out loss linearly interpolated at a's cumulative
     training seconds (eval time excluded, checkpoint saves included);
  4. timing per arm (mean step, per-block median, wall clock) and peak VRAM per turn.
"""
import argparse, json, math, os, statistics as S

SHA_197 = "6307af1580bd2fcd656c2b47ab4317b0b44a2fe6b410e0041fe091b02d00a0aa"
STEP1_197 = 11.23556


def load(path):
    if not os.path.exists(path):
        return []
    out = []
    for line in open(path):
        line = line.strip()
        if line:
            out.append(json.loads(line))
    return out


def dedup(rows):
    # A resumed run re-logs steps after the checkpoint; keep the LAST copy.
    d = {}
    for r in rows:
        d[r["step"]] = r
    return [d[k] for k in sorted(d)]


def cum_seconds(steps):
    t, out = 0.0, {}
    for r in steps:
        t += r["sec"]
        out[r["step"]] = t
    return out


def interp(xs, ys, x):
    if not xs or x < xs[0] or x > xs[-1]:
        return None
    for i in range(1, len(xs)):
        if xs[i] >= x:
            f = (x - xs[i - 1]) / (xs[i] - xs[i - 1]) if xs[i] > xs[i - 1] else 0.0
            return ys[i - 1] + f * (ys[i] - ys[i - 1])
    return ys[-1]


def rotation(ly):
    # optimizer steps in one full BCD rotation (k2 1300, k4 700, k5 600)
    return ly["n_blocks"] * ly["steps_per_block"] if ly else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/bcd_blocksize_194")
    ap.add_argument("--pairs", default="k2:k4,k4:k5,k2:k5",
                    help="comma list of a:b; gap = a - b, positive = b better")
    x = ap.parse_args()
    pairs = [tuple(p.split(":")) for p in x.pairs.split(",") if p]
    tags = sorted({t for p in pairs for t in p})
    E = {t: dedup(load(os.path.join(x.dir, f"eval_{t}.jsonl"))) for t in tags}
    ST = {t: dedup(load(os.path.join(x.dir, f"steps_{t}.jsonl"))) for t in tags}
    TU = {t: load(os.path.join(x.dir, f"turns_{t}.jsonl")) for t in tags}
    LY = {}
    for t in tags:
        p = os.path.join(x.dir, f"layout_{t}.json")
        LY[t] = json.load(open(p)) if os.path.exists(p) else None

    print("== sanity ==")
    for t in tags:
        ly = LY[t]
        print(f"  {t}: layout {ly['n_blocks'] if ly else '?'} blocks "
              f"(rotation {rotation(ly)} steps), {len(ST[t])} steps logged, {len(E[t])} evals")
    shas = {r["target_sha"] for t in tags for r in E[t]}
    print(f"  held-out target sha: {'same in all evals' if len(shas) == 1 else 'DIFFERENT: ' + str(shas)}"
          + (", matches #197" if shas == {SHA_197} else ", DOES NOT match #197" if shas else ""))
    for t in tags:
        s1 = [r for r in E[t] if r["step"] == 1]
        if s1:
            print(f"  {t} step-1 held-out {s1[0]['heldout_nll']:.5f} (#197: {STEP1_197})")
    for a, b in pairs:
        compare(a, b, E, ST, LY)
    timing(tags, E, ST, TU)


def compare(a, b, E, ST, LY):
    class x:  # keep the original a/b names below
        pass
    x.a, x.b = a, b
    ra_rot, rb_rot = rotation(LY[a]), rotation(LY[b])
    print(f"\n== held-out loss at equal steps (gap = {x.a} - {x.b}; positive = {x.b} better) ==")
    ea = {r["step"]: r for r in E[x.a]}
    eb = {r["step"]: r for r in E[x.b]}
    print(f"  {'step':>5} {x.a:>9} {x.b:>9} {'gap':>8} {'exp(gap)':>8} {'pairedSE':>8}  note")
    for s in sorted(set(ea) | set(eb)):
        ra, rb = ea.get(s), eb.get(s)
        la = f"{ra['heldout_nll']:.5f}" if ra else "-"
        lb = f"{rb['heldout_nll']:.5f}" if rb else "-"
        note = []
        if ra_rot and s % ra_rot == 0:
            note.append(f"{x.a} full rotations: {s // ra_rot}")
        if rb_rot and s % rb_rot == 0:
            note.append(f"{x.b} full rotations: {s // rb_rot}")
        if ra and rb:
            d = [p - q for p, q in zip(ra["per_window"], rb["per_window"])]
            g = sum(d) / len(d)
            se = S.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            print(f"  {s:>5} {la:>9} {lb:>9} {g:>+8.4f} {math.exp(g):>8.4f} {se:>8.4f}  {'; '.join(note)}")
        else:
            print(f"  {s:>5} {la:>9} {lb:>9} {'':>8} {'':>8} {'':>8}  {'; '.join(note)}")
    fa = max((s for s in ea if ra_rot and s % ra_rot == 0), default=None)
    fb = max((s for s in eb if rb_rot and s % rb_rot == 0), default=None)
    if fa and fb and fa != fb:
        g = ea[fa]["heldout_nll"] - eb[fb]["heldout_nll"]
        print(f"  full rotations only: {x.a}@{fa} ({fa // ra_rot} rotations) - {x.b}@{fb} "
              f"({fb // rb_rot} rotations) = {g:+.4f} (exp {math.exp(g):.4f}); "
              f"note the LR is not at the same point")

    print(f"\n== equal training time ({x.b} interpolated at {x.a}'s elapsed training seconds) ==")
    ca, cb = cum_seconds(ST[x.a]), cum_seconds(ST[x.b])
    # Step 1 is left out: interpolating between the init loss and step 100 is
    # meaningless on a curve this steep.
    bx = [cb[s] for s in sorted(eb) if s in cb and s > 1]
    by = [eb[s]["heldout_nll"] for s in sorted(eb) if s in cb and s > 1]
    print(f"  {'step '+x.a:>9} {'hours':>6} {x.a:>9} {x.b+'@same':>9} {'gap':>8}")
    for s in sorted(ea):
        if s not in ca or s == 1:
            continue
        v = interp(bx, by, ca[s])
        if v is None:
            continue
        g = ea[s]["heldout_nll"] - v
        print(f"  {s:>9} {ca[s] / 3600:>6.2f} {ea[s]['heldout_nll']:>9.5f} {v:>9.5f} {g:>+8.4f}")
    if ST[x.a] and ST[x.b]:
        n = min(ST[x.a][-1]["step"], ST[x.b][-1]["step"])
        ma = S.mean(r["sec"] for r in ST[x.a] if r["step"] <= n)
        mb = S.mean(r["sec"] for r in ST[x.b] if r["step"] <= n)
        print(f"  mean step over steps 1-{n}: {x.a} {ma:.3f} s, {x.b} {mb:.3f} s, "
              f"{x.b}/{x.a} {mb / ma:.3f}")


def timing(tags, E, ST, TU):
    print("\n== timing and VRAM ==")
    for t in tags:
        st = ST[t]
        if not st:
            continue
        secs = [r["sec"] for r in st]
        by_block = {}
        for r in st:
            by_block.setdefault(r["block"], []).append(r["sec"])
        meds = [S.median(v) for v in by_block.values()]
        evs = [r["eval_sec"] for r in E[t]]
        wall = st[-1]["wall"] - st[0]["wall"] + st[0]["sec"]
        print(f"  {t}: mean step {S.mean(secs):.3f} s, mean of per-block medians "
              f"{S.mean(meds):.3f} s over {len(by_block)} blocks, train time "
              f"{sum(secs) / 3600:.2f} h, wall (incl. eval/saves/resumes) {wall / 3600:.2f} h, "
              f"eval {S.mean(evs) if evs else float('nan'):.0f} s each")
        tu = TU[t]
        if tu:
            worst = max(tu, key=lambda r: r["peak_res"])
            n_over = sum(r["over_85"] for r in tu)
            print(f"      peak reserved per turn: max {worst['peak_res'] / 2**20:.0f} MiB "
                  f"({worst['peak_res_pct']:.1f}%) at {worst['block']}; turns over 85%: {n_over}")
        ev_peak = [r.get("eval_peak_res", 0) for r in E[t]]
        if ev_peak:
            print(f"      eval peak reserved: max {max(ev_peak) / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
