"""Summarize #199 arms -> results/bcd_prefix/summary.json + printed tables."""
import json
import os
import statistics as st

D = os.path.dirname(os.path.abspath(__file__))
ARMS = ["base", "forced", "nograd", "base2"]


def load(tag):
    p = os.path.join(D, f"{tag}.jsonl")
    return [json.loads(l) for l in open(p)] if os.path.exists(p) else []


def block_index(name):
    return int(name.split(".")[1])


out = {}
for tag in ARMS:
    rows = load(tag)
    if not rows:
        continue
    per = {}
    for r in rows:
        per.setdefault(r["block"], []).append(r)
    arm = {}
    for blk, rs in per.items():
        members = rs[0]["members"]
        first = (-1 if members[0] == "token_embedding"
                 else block_index(members[0]) if members[0].startswith("blocks.") else 99)
        last = max([block_index(m) for m in members if m.startswith("blocks.")] or [first])
        mbs = sum(r["micro_batches"] for r in rs)
        # call counts per micro-batch, summed over all steps of this block
        calls = {}
        for r in rs:
            for k, v in r["calls"].items():
                c = calls.setdefault(k, [0, 0])
                c[0] += v["fwd"]
                c[1] += v["bwd"]
        layer_calls = {k: v for k, v in calls.items() if k.startswith("blocks.")}
        prefix = [k for k in layer_calls if block_index(k) < first]
        suffix = [k for k in layer_calls if block_index(k) > last]
        active = [k for k in layer_calls if first <= block_index(k) <= last]

        def per_mb(keys, i):
            vals = sorted({layer_calls[k][i] / mbs for k in keys})
            return vals
        timed = [r for r in rs if r["step_in_block"] > 0 and r["step_sec"] is not None]
        # median over timed steps of per-block bwd ms (prefix blocks have none unless forced)
        def sum_bwd(r, keys):
            return sum(r["bwd_ms"].get(k, 0.0) for k in keys)
        arm[blk] = {
            "members": members, "micro_batches": mbs,
            "prefix_blocks": len(prefix), "suffix_blocks": len(suffix),
            "prefix_fwd_calls_per_mb": per_mb(prefix, 0) if prefix else None,
            "prefix_bwd_calls_per_mb": per_mb(prefix, 1) if prefix else None,
            "active_fwd_bwd_per_mb": [per_mb(active, 0), per_mb(active, 1)] if active else None,
            "suffix_bwd_calls_per_mb": per_mb(suffix, 1) if suffix else None,
            "embedding_calls_per_mb": [c / mbs for c in calls.get("token_embedding", [0, 0])],
            "prefix_in_requires_grad": sorted({rs[-1]["req"][k]["in_req"] for k in prefix}) if prefix else None,
            "prefix_out_requires_grad": sorted({rs[-1]["req"][k]["out_req"] for k in prefix}) if prefix else None,
            "step_sec_median": round(st.median(r["step_sec"] for r in timed), 3) if timed else None,
            "step_secs": [round(r["step_sec"], 3) for r in timed],
            "prefix_fwd_ms_median": round(st.median(r["prefix_fwd_ms"] for r in timed if r["prefix_fwd_ms"] is not None), 1) if timed else None,
            "prefix_bwd_ms_median": round(st.median(sum_bwd(r, prefix) for r in timed), 1) if timed and prefix else 0.0,
            "fwd_end_ms_median": round(st.median(r["fwd_end_ms"] for r in timed), 1) if timed else None,
            "bwd_end_ms_median": round(st.median(r["bwd_end_ms"] for r in timed), 1) if timed else None,
            "peak_reserved_mib": round(max(r["peak_reserved"] for r in rs) / 2**20),
            "peak_frac": round(max(r["peak_reserved"] for r in rs) / rs[0]["total_mem"], 3),
        }
    out[tag] = arm

json.dump(out, open(os.path.join(D, "summary.json"), "w"), indent=1)
for tag, arm in out.items():
    print(f"== {tag}")
    for blk, a in arm.items():
        print(f"  {blk:24s} pre={a['prefix_blocks']:2d} suf={a['suffix_blocks']:2d} "
              f"prefix fwd/bwd per mb={a['prefix_fwd_calls_per_mb']}/{a['prefix_bwd_calls_per_mb']} "
              f"active={a['active_fwd_bwd_per_mb']} suffix bwd={a['suffix_bwd_calls_per_mb']} "
              f"emb={a['embedding_calls_per_mb']} in_req={a['prefix_in_requires_grad']} "
              f"step={a['step_sec_median']}s {a['step_secs']} prefix_fwd={a['prefix_fwd_ms_median']}ms "
              f"prefix_bwd={a['prefix_bwd_ms_median']}ms fwd_end={a['fwd_end_ms_median']} "
              f"bwd_end={a['bwd_end_ms_median']} peak={a['peak_reserved_mib']}MiB ({a['peak_frac']})")
