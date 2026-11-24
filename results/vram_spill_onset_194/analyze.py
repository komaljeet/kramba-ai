"""#194 VRAM spill onset: apply the verdict rule in run_point.sh to every finished point.

Run from the worktree root: python results/vram_spill_onset_194/analyze.py
"""
import json, os, statistics as S

O = "results/vram_spill_onset_194"
TOTAL = 6441926656          # torch total_memory of the RTX 4050 Laptop (6143.4 MiB)
MiB = 2 ** 20
BLOCKS = ["blocks.5", "blocks.0", "blocks.10"]   # turn order; label by first member


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


def win_max(host, t0, t1, kind):
    # training process's counter, max over host samples inside [t0, t1]
    v = [x for h in host if t0 <= h["wall"] <= t1
         for k, x in h.items() if k.startswith("process(") and k.endswith(kind)]
    return max(v) if v else None


def point(tag):
    st, tu, ev = rows(f"{O}/steps_{tag}.jsonl"), rows(f"{O}/turns_{tag}.jsonl"), rows(f"{O}/eval_{tag}.jsonl")
    mem, host = rows(f"{O}/mem_{tag}.jsonl"), rows(f"{O}/host_{tag}.jsonl")
    if not st:
        return None
    ball = mem[0]["ballast_mib"] if mem else None
    out = {"tag": tag, "ballast": ball, "blocks": {}, "n_steps": len(st)}
    for b in BLOCKS:
        r = [x for x in st if x["block"].split("+")[0] == b]
        t = [x for x in tu if x["block"].split("+")[0] == b]
        if len(r) < 4 or not t:
            continue
        out["blocks"][b] = {"med": S.median([x["sec"] for x in r[1:4]]), "steps": [x["sec"] for x in r],
                            "peak_res": t[0]["peak_res"], "peak_alloc": t[0]["peak_alloc"],
                            "shared": win_max(host, r[0]["wall"] - r[0]["sec"], r[-1]["wall"], "shared"),
                            "ded": win_max(host, r[0]["wall"] - r[0]["sec"], r[-1]["wall"], "dedicated")}
    if ev:
        out["eval"] = {"sec": ev[0]["eval_sec"], "peak_res": ev[0]["eval_peak_res"],
                       "shared": win_max(host, ev[0]["wall"] - ev[0]["eval_sec"], ev[0]["wall"], "shared"),
                       "ded": win_max(host, ev[0]["wall"] - ev[0]["eval_sec"], ev[0]["wall"], "dedicated")}
    if mem:
        out["retries"] = mem[-1]["num_alloc_retries"]; out["ooms"] = mem[-1]["num_ooms"]
        out["min_drv_free"] = min(m["drv_free"] for m in mem)
        out["max_res_now"] = max(m["reserved"] for m in mem)
    pid = mem[0]["pid"] if mem else None
    sh = [v for h in host for k, v in h.items() if k.startswith("process(") and k.endswith("shared")]
    de = [v for h in host for k, v in h.items() if k.startswith("process(") and k.endswith("dedicated")]
    out["proc_shared_max"] = max(sh) if sh else None
    out["proc_ded_max"] = max(de) if de else None
    out["smi_max"] = max(h["smi_used_mib"] for h in host) if host else None
    ash = [v for h in host for k, v in h.items() if k.startswith("adapter(") and "176eb" in k and k.endswith("shared")]
    out["adapter_shared_max"] = max(ash) if ash else None
    return out


tags = []
for l in open(f"{O}/status.txt"):
    if l.strip().endswith("DONE"):
        tags.append(l.split()[0])
P = {t: point(t) for t in tags}
P = {t: p for t, p in P.items() if p}
base = [P[t] for t in ("base_a", "base_b") if t in P]
print("points:", list(P))
if not base:
    raise SystemExit("no baseline yet")
bmed = {b: S.mean(p["blocks"][b]["med"] for p in base) for b in BLOCKS if all(b in p["blocks"] for p in base)}
bev = S.mean(p["eval"]["sec"] for p in base if "eval" in p)
bsh = max((p["proc_shared_max"] or 0) for p in base)
if len(base) == 2:
    for b in bmed:
        a, c = (p["blocks"][b]["med"] for p in base)
        print(f"baseline spread {b:10s} {a:6.2f} vs {c:6.2f} s  {100 * abs(a - c) / ((a + c) / 2):.2f}%")
    a, c = (p["eval"]["sec"] for p in base)
    print(f"baseline spread eval       {a:6.2f} vs {c:6.2f} s  {100 * abs(a - c) / ((a + c) / 2):.2f}%")
print(f"baseline process shared-usage peak: {bsh / MiB:.0f} MiB")

print(f"\n{'tag':8s} {'ballast':>7s} | {'what':10s} {'pressure':>8s} {'MiB':>6s} {'sec':>7s} {'ratio':>6s} spill")
for t, p in P.items():
    shr = (p["proc_shared_max"] or 0) - bsh
    for b in BLOCKS:
        if b not in p["blocks"]:
            continue
        x = p["blocks"][b]
        pct = 100 * x["peak_res"] / TOTAL
        r = x["med"] / bmed[b]
        sh = (x["shared"] or 0) - bsh
        sp = r > 1.05 or sh > 64 * MiB
        print(f"{t:8s} {p['ballast']:>7} | {b:10s} {pct:7.1f}% {x['peak_res'] / MiB:6.0f} {x['med']:7.2f} {r:6.3f} "
              f"{'SPILL' if sp else 'ok'}  shared +{sh / MiB:.0f} ded {(x['ded'] or 0) / MiB:.0f} alloc {x['peak_alloc'] / MiB:.0f}  steps {[round(s, 1) for s in x['steps']]}")
    if "eval" in p:
        e = p["eval"]; pct = 100 * e["peak_res"] / TOTAL; r = e["sec"] / bev
        print(f"{t:8s} {p['ballast']:>7} | {'eval':10s} {pct:7.1f}% {e['peak_res'] / MiB:6.0f} {e['sec']:7.2f} {r:6.3f} "
              f"{'SPILL' if r > 1.05 or (e['shared'] or 0) - bsh > 64 * MiB else 'ok'}  shared +{((e['shared'] or 0) - bsh) / MiB:.0f} ded {(e['ded'] or 0) / MiB:.0f}")
    print(f"{'':17s}  retries {p.get('retries')} ooms {p.get('ooms')}  proc shared max "
          f"{(p['proc_shared_max'] or 0) / MiB:.0f} MiB (+{shr / MiB:.0f})  proc dedicated max "
          f"{(p['proc_ded_max'] or 0) / MiB:.0f} MiB  adapter shared max {(p['adapter_shared_max'] or 0) / MiB:.0f} MiB  "
          f"smi max {p['smi_max']} MiB  max reserved(now) {p.get('max_res_now', 0) / MiB:.0f} MiB  "
          f"min driver free {p.get('min_drv_free', 0) / MiB:.0f} MiB")
