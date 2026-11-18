"""#214: summarize the BCD-vs-full held-out A/B (eval_<tag>.jsonl, steps_<tag>.jsonl).

Usage: python scripts/summarize_bcd_penalty_214.py <ab dir> --full full_s0 full_s1
           --bcd bcd100_s0 bcd100_s1 [--other bcd50_s0 bcd200_s0] [--tokens-per-step 32768]

Gap = BCD held-out minus full held-out at the same optimizer step (same tokens, same
windows). "% worse" = gap / full loss. "ppl x" = exp(gap). Steps-equivalent: the step at
which the FULL curve (seed mean, linear between eval points) first reaches the BCD loss;
BCD needed step / that step times as many steps for that loss.
"""
import os, sys, json, math, argparse
from statistics import mean, stdev


def load(d, tag):
    ev = {}
    p = os.path.join(d, f"eval_{tag}.jsonl")
    if not os.path.exists(p):
        return None, None
    for l in open(p):
        if l.strip():
            r = json.loads(l); ev[r["step"]] = r
    st = [json.loads(l) for l in open(os.path.join(d, f"steps_{tag}.jsonl")) if l.strip()]
    return ev, st


def first_reach(curve, target):
    """curve: sorted [(step, loss)], loss mostly falling. First step where loss <= target."""
    for (s0, l0), (s1, l1) in zip(curve, curve[1:]):
        if l1 <= target:
            if l0 <= target:
                return s0
            return s0 + (s1 - s0) * (l0 - target) / (l0 - l1)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("--full", nargs="+", required=True)
    ap.add_argument("--bcd", nargs="+", required=True)
    ap.add_argument("--other", nargs="*", default=[])
    ap.add_argument("--tokens-per-step", type=int, default=8 * 4096)
    a = ap.parse_args()
    tags = a.full + a.bcd + a.other
    E, S = {}, {}
    for t in tags:
        e, s = load(a.dir, t)
        if e is not None:
            E[t], S[t] = e, s
    full = [t for t in a.full if t in E]; bcd = [t for t in a.bcd if t in E]
    other = [t for t in a.other if t in E]
    steps = sorted(set.intersection(*[set(E[t]) for t in full + bcd]))
    out = []
    P = out.append
    shas = {E[t][s]["target_sha"] for t in E for s in E[t]}
    P(f"distinct held-out target sha over all evals: {len(shas)} (must be 1)")
    P("")
    P("held-out NLL (64 fixed windows x 4096 tokens), same windows for every arm")
    hdr = f"{'step':>6} {'Mtok':>7} " + " ".join(f"{t:>11}" for t in full + bcd + other)
    hdr += f" {'gap nats':>9} {'% worse':>8} {'ppl x':>6} {'full spread':>11} {'bcd spread':>10}"
    P(hdr)
    fm_curve = []
    rows = []
    for s in sorted(set().union(*[set(E[t]) for t in E])):
        vals = {t: E[t][s]["heldout_nll"] for t in E if s in E[t]}
        line = f"{s:>6} {s * a.tokens_per_step / 1e6:>7.1f} " + " ".join(
            f"{vals[t]:>11.5f}" if t in vals else f"{'':>11}" for t in full + bcd + other)
        if all(t in vals for t in full + bcd):
            fm = mean(vals[t] for t in full); bm = mean(vals[t] for t in bcd)
            g = bm - fm
            fs = (max(vals[t] for t in full) - min(vals[t] for t in full)) if len(full) > 1 else float("nan")
            bs = (max(vals[t] for t in bcd) - min(vals[t] for t in bcd)) if len(bcd) > 1 else float("nan")
            line += f" {g:>+9.4f} {100 * g / fm:>7.2f}% {math.exp(g):>6.3f} {fs:>11.5f} {bs:>10.5f}"
            fm_curve.append((s, fm)); rows.append((s, fm, bm, g))
        P(line)
    P("")
    P("steps-equivalent: step where the full curve (seed mean) first reaches the BCD loss")
    P(f"{'step':>6} {'bcd mean':>9} {'full reaches it at':>18} {'bcd needs x steps':>17} {'extra steps':>11}")
    for s, fm, bm, g in rows:
        if s < 500:
            continue
        r = first_reach(fm_curve, bm)
        if r is None:
            P(f"{s:>6} {bm:>9.5f} {'not reached':>18}")
        else:
            P(f"{s:>6} {bm:>9.5f} {r:>18.0f} {s / r:>17.2f} {s - r:>11.0f}")
    P("")
    P("paired per-window gap at the last common step (bcd seed k - full seed k)")
    last = rows[-1][0]
    for fb in zip(full, bcd):
        f, b = fb
        d = [x - y for x, y in zip(E[b][last]["per_window"], E[f][last]["per_window"])]
        P(f"  {b} - {f} @ {last}: mean {mean(d):+.5f}  sem {stdev(d) / math.sqrt(len(d)):.5f}  "
          f"windows worse {sum(x > 0 for x in d)}/{len(d)}")
    for o in other:
        if last in E[o]:
            for f in full:
                d = [x - y for x, y in zip(E[o][last]["per_window"], E[f][last]["per_window"])]
                P(f"  {o} - {f} @ {last}: mean {mean(d):+.5f}  sem {stdev(d) / math.sqrt(len(d)):.5f}")
    P("")
    P("per run: steps, mean s/step (steps 2..N, evals excluded), peak reserved % of card, "
      "grad elements per step (min..max), resumes")
    for t in E:
        st = S[t]
        secs = [r["sec"] for r in st[1:]]
        pk = max(r["peak_res"] / r["total_mem"] for r in st[1:]) if len(st) > 1 else float("nan")
        gn = [r["grad_numel"] for r in st]
        wall = (st[-1]["wall"] - st[0]["wall"]) / 3600 if len(st) > 1 else 0
        res = os.path.join(a.dir, f"resumes_{t}.jsonl")
        nres = sum(1 for _ in open(res)) if os.path.exists(res) else 0
        P(f"  {t:<12} steps {len(st):>5}  {mean(secs) if secs else float('nan'):6.3f} s/step  "
          f"peak {100 * pk:5.1f}%  grad elems {min(gn):,}..{max(gn):,}  wall {wall:.2f} h  resumes {nres}")
        if t not in full:
            per = {}
            for r in st[1:]:
                per.setdefault(r["block"], []).append(r["sec"])
            P("      per block s/step: " + "  ".join(f"{k}:{mean(v):.2f}" for k, v in per.items()))
    print("\n".join(out))


if __name__ == "__main__":
    main()
