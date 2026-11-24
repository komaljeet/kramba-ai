"""#200: compare k4 with 50/25/10 steps per block turn against the #194 k4@100 run.

Author: Rambarun Komaljeet

Usage: python scripts/analyze_bcd_spb_200.py [--dir results/bcd_spb_200]
       [--ref-dir results/bcd_blocksize_194 --ref-tag k4] [--arms k4s50,k4s25,k4s10]

Works on partial runs. Prints:
  1. sanity per arm: layout (7 blocks), turn lengths in the step log equal to the arm's
     steps-per-block, block order, lr > 0 before the last step, held-out sha equal to the
     reference and #197, step-1 held-out equal to the reference (same init);
  2. held-out loss at equal steps, gap = arm - k4@100 (positive = shorter turns WORSE),
     exp(gap), paired per-window SE (eval sampling only);
  3. equal training time: k4@100 interpolated at the arm's cumulative training seconds;
  4. timing: mean step, switch overhead (first step of a turn minus the median of the
     rest of that turn), peak VRAM per turn and in eval;
  5. the verdict by the rule set before any result (see RULE).
"""
import argparse, json, math, os, statistics as S, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from analyze_bcd_blocksize_194 import SHA_197, load, dedup, cum_seconds, interp  # noqa: E402

RULE = ("an arm is WORSE than k4@100 if arm - ref > 0.012 nats at step 2800 and > 0 at 2100 "
        "and 2600; BETTER if ref - arm > 0.012 at 2800 and > 0 at 2100 and 2600; otherwise "
        "cannot tell (0.012 = 2x the #197 floor, which is one sample at 180 steps)")


def runs(steps):
    out, cur, n = [], None, 0
    for r in steps:
        if r["block"] != cur:
            if cur is not None:
                out.append((cur, n, r["step"] - n))
            cur, n = r["block"], 0
        n += 1
    if cur is not None:
        out.append((cur, n, steps[-1]["step"] - n + 1))
    return out   # (block, length, first step)


def arm_data(d, tag):
    ly = os.path.join(d, f"layout_{tag}.json")
    return {"E": dedup(load(os.path.join(d, f"eval_{tag}.jsonl"))),
            "ST": dedup(load(os.path.join(d, f"steps_{tag}.jsonl"))),
            "TU": load(os.path.join(d, f"turns_{tag}.jsonl")),
            "LY": json.load(open(ly)) if os.path.exists(ly) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/bcd_spb_200")
    ap.add_argument("--ref-dir", default="results/bcd_blocksize_194")
    ap.add_argument("--ref-tag", default="k4")
    ap.add_argument("--arms", default="k4s50,k4s25,k4s10")
    x = ap.parse_args()
    ref = arm_data(x.ref_dir, x.ref_tag)
    arms = {t: arm_data(x.dir, t) for t in x.arms.split(",") if t}
    allr = {"k4@100": ref, **arms}
    ref1 = {r["step"]: r for r in ref["E"]}.get(1)
    ref_sha = {r["target_sha"] for r in ref["E"]}

    print("== sanity ==")
    for t, D in allr.items():
        ly, st = D["LY"], D["ST"]
        if not st:
            print(f"  {t}: no steps logged yet"); continue
        spb = ly["steps_per_block"] if ly else None
        rr = runs(st)
        full = rr[:-1]   # last turn may be cut by the log end
        bad = [r for r in full if r[1] != spb]
        names = ly["blocks"] if ly else []
        order_ok = all(names[(i + names.index(rr[0][0])) % len(names)] == r[0]
                       for i, r in enumerate(rr)) if names else False
        # step 1 runs at lr 0 by design (warmup starts at 0; measured in #194)
        lr_min = min((min(r["lr"]) for r in st if 1 < r["step"] < 2800), default=float("nan"))
        shas = {r["target_sha"] for r in D["E"]}
        s1 = {r["step"]: r for r in D["E"]}.get(1)
        print(f"  {t}: {ly['n_blocks'] if ly else '?'} blocks, steps/block {spb}, "
              f"{len(st)} steps (last {st[-1]['step']}), {len(D['E'])} evals; "
              f"turns {len(rr)}, wrong-length turns {len(bad)}"
              + (f" {bad[:3]}" if bad else "") + f", block order {'ok' if order_ok else 'WRONG'}, "
              f"min lr steps 2-2799 {lr_min:.3g}"
              + (", sha " + ("same as ref and #197" if shas == ref_sha == {SHA_197} else
                             f"DIFFERENT {shas}") if shas else "")
              + (f", step-1 held-out {s1['heldout_nll']:.5f}"
                 + (" (= ref)" if ref1 and s1["heldout_nll"] == ref1["heldout_nll"] else
                    f" (ref {ref1['heldout_nll']:.5f})" if ref1 else "") if s1 else ""))

    verdicts = {}
    for t, D in arms.items():
        print(f"\n== {t} vs k4@100, equal steps (gap = {t} - k4@100; positive = {t} worse) ==")
        ea = {r["step"]: r for r in D["E"]}
        eb = {r["step"]: r for r in ref["E"]}
        print(f"  {'step':>5} {t:>9} {'k4@100':>9} {'gap':>8} {'exp(gap)':>8} {'pairedSE':>8}")
        gaps = {}
        for s in sorted(set(ea) & set(eb)):
            d = [p - q for p, q in zip(ea[s]["per_window"], eb[s]["per_window"])]
            g = sum(d) / len(d)
            se = S.stdev(d) / math.sqrt(len(d)) if len(d) > 1 else float("nan")
            gaps[s] = g
            if s == 1 or s % 100 == 0:
                print(f"  {s:>5} {ea[s]['heldout_nll']:>9.5f} {eb[s]['heldout_nll']:>9.5f} "
                      f"{g:>+8.4f} {math.exp(g):>8.4f} {se:>8.4f}")
        need = (2100, 2600, 2800)
        if all(s in gaps for s in need):
            if gaps[2800] > 0.012 and gaps[2100] > 0 and gaps[2600] > 0:
                v = "WORSE than k4@100"
            elif -gaps[2800] > 0.012 and gaps[2100] < 0 and gaps[2600] < 0:
                v = "BETTER than k4@100"
            else:
                v = "cannot tell"
        else:
            v = "not finished (needs held-out at 2100, 2600, 2800)"
        verdicts[t] = v

        print(f"\n  equal training time (k4@100 interpolated at {t}'s elapsed training seconds)")
        ca, cb = cum_seconds(D["ST"]), cum_seconds(ref["ST"])
        bx = [cb[s] for s in sorted(eb) if s in cb and s > 1]
        by = [eb[s]["heldout_nll"] for s in sorted(eb) if s in cb and s > 1]
        print(f"  {'step':>9} {'hours':>6} {t:>9} {'k4@100':>9} {'gap':>8}")
        for s in sorted(ea):
            if s not in ca or s == 1 or s % 700 and s != max(ea):
                continue
            vv = interp(bx, by, ca[s])
            if vv is None:
                continue
            print(f"  {s:>9} {ca[s] / 3600:>6.2f} {ea[s]['heldout_nll']:>9.5f} {vv:>9.5f} "
                  f"{ea[s]['heldout_nll'] - vv:>+8.4f}")

    print("\n== timing and VRAM (switch overhead = first step of a turn minus the median of the rest) ==")
    for t, D in allr.items():
        st = D["ST"]
        if not st:
            continue
        by = {r["step"]: r for r in st}
        ex = []
        for blk, n, first in runs(st)[1:-1]:
            rest = [by[s]["sec"] for s in range(first + 1, first + n) if s in by]
            if len(rest) >= 3 and first in by:
                ex.append(by[first]["sec"] - S.median(rest))
        n = st[-1]["step"]
        mean = S.mean(r["sec"] for r in st)
        rmean = S.mean(r["sec"] for r in ref["ST"] if r["step"] <= n)
        tu = D["TU"]
        worst = max(tu, key=lambda r: r["peak_res"]) if tu else None
        evp = [r.get("eval_peak_res", 0) for r in D["E"]]
        print(f"  {t}: mean step {mean:.3f} s over steps 1-{n} (k4@100 same steps {rmean:.3f} s, "
              f"ratio {mean / rmean:.3f}); switch overhead median "
              f"{S.median(ex) if ex else float('nan'):.3f} s over {len(ex)} switches, "
              f"sum {sum(ex) / 60:.1f} min; train time {sum(r['sec'] for r in st) / 3600:.2f} h")
        if worst:
            print(f"      peak reserved per turn: max {worst['peak_res'] / 2**20:.0f} MiB "
                  f"({worst['peak_res_pct']:.1f}%) at {worst['block']}; turns over 85%: "
                  f"{sum(r['over_85'] for r in tu)}; eval peak max "
                  f"{max(evp) / 2**20 if evp else 0:.0f} MiB "
                  f"({100 * max(evp) / (6143 * 2**20) if evp else 0:.1f}%)")

    print(f"\n== verdict (rule set before results: {RULE}) ==")
    for t, v in verdicts.items():
        print(f"  {t}: {v}")


if __name__ == "__main__":
    main()
