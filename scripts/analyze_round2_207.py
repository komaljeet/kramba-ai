"""#207: summarize results/round2_207/ (profile_step_202.py jsonl runs).

  python scripts/analyze_round2_207.py kernels prof_main     # kernel ranking, mean per profiled step
  python scripts/analyze_round2_207.py time A B [C ...]      # step time per arm (tags without repeat digit)
  python scripts/analyze_round2_207.py hash REF OTHER ...    # sha equality vs REF
  python scripts/analyze_round2_207.py vram TAG ...          # peak reserved per turn, training and eval
  python scripts/analyze_round2_207.py smi CSV               # nvidia-smi clocks / power / throttle summary

Step time per block position = mean of turn steps 2.. (step 1 carries the BCD switch),
averaged over the 7 positions, as in #202/#204/#206.
"""
import csv, glob, json, os, sys
from collections import OrderedDict

D = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "round2_207")


def load(tag):
    p = os.path.join(D, tag + ".jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else None


def per_pos(rows):
    by = OrderedDict()
    for r in rows:
        if r["pos_in_turn"] >= 2:
            by.setdefault(r["block"], []).append(r["sec"])
    return [sum(v) / len(v) for v in by.values()]


def kernels(tag, top=45):
    rows = [r for r in load(tag) if "kernels" in r]
    tot = {}
    for r in rows:
        for k, v in r["kernels"].items():
            tot[k] = tot.get(k, 0.0) + v / len(rows)
    print(f"{tag}: {len(rows)} profiled steps, mean step {sum(r['sec'] for r in rows) / len(rows):.3f} s, "
          f"kernel sum {sum(tot.values()):.3f} s/step (top 150 per step only)")
    for k, v in sorted(tot.items(), key=lambda kv: -kv[1])[:top]:
        print(f"{v:8.4f}  {k[:150]}")


def time_(tags):
    for arm in tags:
        ms = []
        paths = sorted(glob.glob(os.path.join(D, f"{arm}[0-9].jsonl")))
        if not paths and os.path.exists(os.path.join(D, f"{arm}.jsonl")):
            paths = [os.path.join(D, f"{arm}.jsonl")]
        for p in paths:
            rows = [json.loads(l) for l in open(p)]
            pp = per_pos(rows)
            if len(pp) != 7:
                print(f"{os.path.basename(p)}: only {len(pp)} positions"); continue
            m = sum(pp) / 7
            ms.append(m)
            print(f"{os.path.basename(p):26s} " + " ".join(f"{x:6.2f}" for x in pp) + f" | mean {m:.3f}"
                  f" | peak {max(r['peak_res'] / r['total_mem'] for r in rows):.1%}")
        if ms:
            print(f"  {arm}: mean of {len(ms)} = {sum(ms) / len(ms):.3f}, spread {max(ms) - min(ms):.3f}")


def hash_(ref, others):
    b = load(ref)
    for t in others:
        o = load(t)
        n = min(len(b), len(o))
        g = sum(b[i]["grad_sha"] == o[i]["grad_sha"] for i in range(n))
        l = sum(b[i]["loss_sha"] == o[i]["loss_sha"] for i in range(n))
        lr0 = sum(1 for i in range(n) if max(b[i]["lr"]) > 0)
        print(f"{ref} vs {t}: grad equal {g}/{n}, loss equal {l}/{n} (steps with lr>0: {lr0})")


def vram(tags):
    for t in tags:
        rows = load(t)
        tr, ev = OrderedDict(), OrderedDict()
        for r in rows:
            tr[r["block"]] = max(tr.get(r["block"], 0), r["peak_res"] / r["total_mem"])
            if "heldout_peak_res" in r:
                ev[r["block"]] = max(ev.get(r["block"], 0), r["heldout_peak_res"] / r["total_mem"])
        print(f"{t:22s} train " + " ".join(f"{v:.1%}" for v in tr.values())
              + (" | eval " + " ".join(f"{v:.1%}" for v in ev.values()) if ev else ""))
        if any("heldout" in r for r in rows):
            print("   heldout: " + " ".join(f"{r['heldout']:.9f}" for r in rows if "heldout" in r))


def smi(path):
    rows = list(csv.reader(open(path)))
    hdr = [h.strip() for h in rows[0]]
    data = [dict(zip(hdr, [c.strip() for c in r])) for r in rows[1:] if len(r) == len(hdr)]

    def num(r, k):
        try:
            return float(r[k].split()[0])
        except Exception:
            return None
    busy = [r for r in data if (num(r, "utilization.gpu [%]") or 0) >= 50]
    print(f"{len(data)} samples, {len(busy)} with utilization >= 50%")
    for k in ("clocks.current.sm [MHz]", "clocks.current.memory [MHz]", "power.draw [W]",
              "temperature.gpu", "utilization.gpu [%]"):
        v = sorted(x for x in (num(r, k) for r in busy) if x is not None)
        if v:
            print(f"  {k:30s} min {v[0]:8.1f}  p10 {v[len(v) // 10]:8.1f}  median {v[len(v) // 2]:8.1f}"
                  f"  p90 {v[9 * len(v) // 10]:8.1f}  max {v[-1]:8.1f}")
    print(f"  max SM clock (clocks.max.sm): {busy[0].get('clocks.max.sm [MHz]') if busy else '?'}; "
          f"power limit enforced: {busy[0].get('enforced.power.limit [W]') if busy else '?'}")
    capk = [k for k in hdr if k.endswith("sw_power_cap")]
    if capk:
        for state in ("Active", "Not Active"):
            sel = [r for r in busy if r[capk[0]] == state]
            c = sorted(num(r, "clocks.current.sm [MHz]") for r in sel)
            p = sorted(num(r, "power.draw [W]") for r in sel)
            if c:
                print(f"  power cap {state:10s}: {len(sel)} samples, SM clock median {c[len(c) // 2]:.0f} "
                      f"(min {c[0]:.0f}), power median {p[len(p) // 2]:.1f} W, max {p[-1]:.1f} W")
    for k in hdr:
        if k.startswith("clocks_event_reasons") or k.startswith("clocks_throttle_reasons"):
            act = sum(1 for r in busy if r[k] == "Active" or (k.endswith(".active") and r[k] not in ("0x0000000000000000",)))
            if k.endswith(".active"):
                from collections import Counter
                print(f"  {k}: " + ", ".join(f"{c}x {v}" for v, c in Counter(r[k] for r in busy).most_common()))
            else:
                print(f"  {k:55s} active in {act}/{len(busy)} busy samples ({act / max(1, len(busy)):.0%})")


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    {"kernels": lambda: kernels(args[0]), "time": lambda: time_(args),
     "hash": lambda: hash_(args[0], args[1:]), "vram": lambda: vram(args),
     "smi": lambda: smi(os.path.join(D, args[0]))}[cmd]()
