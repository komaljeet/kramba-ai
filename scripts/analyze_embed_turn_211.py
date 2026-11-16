"""#211: summarize profile_step_202 runs per block turn, with phase groups.

  python scripts/analyze_embed_turn_211.py DIR tag1 tag2 ...

Per turn: mean step time over turn steps 2.. (the #202 metric), GPU time grouped
into forward layers, loss, backward recompute+backward layers, optimizer, rest,
peak reserved / allocated, host available RAM. Prints each tag's per-turn delta
vs the FIRST tag so the phase that grew is visible.
"""
import json, os, statistics as st, sys


def load(d, tag):
    return [json.loads(l) for l in open(os.path.join(d, tag + ".jsonl")) if l.strip()]


def group(ph):
    g = {"fwd": 0.0, "loss": 0.0, "bwd": 0.0, "opt": 0.0, "other": 0.0}
    for k, v in ph.items():
        if k.startswith("fwd."):
            g["fwd"] += v
        elif k.startswith("loss.") or k == "bwd.loss_recompute" or k == "bwd.head":
            g["loss"] += v
        elif k.startswith("bwd."):
            g["bwd"] += v
        elif k in ("optim", "clip", "pre_opt", "post_opt"):
            g["opt"] += v
        else:
            g["other"] += v
    return g


def turns(rows):
    out, cur = [], None
    for r in rows:
        if cur is None or r["block"] != cur[0]:
            cur = (r["block"], []); out.append(cur)
        cur[1].append(r)
    return out


def summary(rows):
    res = []
    for blk, rs in turns(rows):
        body = rs[1:] if len(rs) > 1 else rs
        gs = [group(r["phases"]) for r in body]
        res.append({
            "block": blk.split("+")[0], "n": len(rs),
            "sec": st.mean(r["sec"] for r in body),
            "g": {k: st.mean(g[k] for g in gs) for k in gs[0]},
            "res": max(r["peak_res"] for r in rs) / 2**20,
            "alloc": max(r["peak_alloc"] for r in rs) / 2**20,
            "avail": min(r["host"]["sys_avail"] for r in rs) / 2**20,
        })
    return res


def main():
    d, tags = sys.argv[1], sys.argv[2:]
    base = summary(load(d, tags[0]))
    for tag in tags:
        s = summary(load(d, tag))
        print(f"== {tag}  (mean over 7 positions: {st.mean(t['sec'] for t in s):.3f} s)")
        for t, b in zip(s, base):
            dg = " ".join(f"{k} {t['g'][k] - b['g'][k]:+.2f}" for k in t["g"])
            print(f"  {t['block'][:16]:16s} {t['sec']:6.2f} s  (d {t['sec'] - b['sec']:+.2f}: {dg})"
                  f"  res {t['res']:.0f} alloc {t['alloc']:.0f} MiB  host avail {t['avail']:.0f} MiB")


if __name__ == "__main__":
    main()
