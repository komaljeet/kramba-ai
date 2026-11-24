"""#211 E3: per-turn step time, with the trainer's WDDM shared (host-RAM) GPU memory.

  python scripts/analyze_recovery_211.py DIR MONITOR_CSV tag1 tag2 ...

Per turn: mean step time over turn steps 2.. (falls back to all steps), the turn's
wall-clock window, and the median trainer "Shared Usage" (MiB) in the monitor CSV
over that window (the trainer = the process with the most dedicated memory, ballast
excluded by size). Ballast end time read from TAG.ballast.log's mtime if present.
"""
import json, os, statistics as st, sys


def mon_rows(path):
    out = []
    for line in list(open(path))[1:]:
        p = line.strip().split(",")
        procs = {}
        for tok in p[6].split():
            pid, v = tok.split(":")
            d, s = v.split("/")
            procs[pid] = (float(d), float(s))
        out.append((float(p[0]), procs))
    return out


def main():
    d, mon, tags = sys.argv[1], sys.argv[2], sys.argv[3:]
    M = mon_rows(mon)
    for tag in tags:
        rows = [json.loads(l) for l in open(os.path.join(d, tag + ".jsonl")) if l.strip()]
        bl = os.path.join(d, tag + ".ballast.log")
        print(f"== {tag}" + (f"  (ballast log present)" if os.path.exists(bl) else ""))
        turns, cur = [], None
        for r in rows:
            if cur is None or r["block"] != cur[0]:
                cur = (r["block"], []); turns.append(cur)
            cur[1].append(r)
        prev_end = rows[0]["wall"] - rows[0]["sec"]
        for i, (blk, rs) in enumerate(turns):
            body = rs[1:] if len(rs) > 1 else rs
            t0, t1 = prev_end, rs[-1]["wall"]
            prev_end = t1
            sh = [max(p.values(), key=lambda v: v[0])[1] for w, p in M if t0 <= w <= t1 and p]
            ded = [max(p.values(), key=lambda v: v[0])[0] for w, p in M if t0 <= w <= t1 and p]
            print(f"  turn {i + 1:2d} {blk.split('+')[0][:16]:16s} {st.mean(r['sec'] for r in body):6.2f} s"
                  f"  trainer ded {st.median(ded) if ded else float('nan'):6.0f} shared {st.median(sh) if sh else float('nan'):5.0f} MiB"
                  f"  drv_free {min(r.get('drv_free', 0) for r in rs) / 2**20:5.0f}")


if __name__ == "__main__":
    main()
