"""#202: rank where the 1B k4 step goes, from ``profile_step_202.py`` output.

Author: Rambarun Komaljeet

Usage: analyze_step_202.py RUN.jsonl [RUN2.jsonl ...] [--pos 2,3]

Groups each step's CUDA-event phases into components (layer forward split into
upstream-frozen / active / downstream-frozen, recompute, layer backward, loss,
optimizer...) and prints seconds and % of step per component, per block
position and as the mean over the 7 positions (each position weighted equally,
as in a full rotation). Also prints the kernel-category view of profiled steps.
"""
import json, re, sys, argparse
from collections import defaultdict

ORDER = ["token_embedding", "blocks 0-3", "blocks 4-7", "blocks 8-11", "blocks 12-15",
         "blocks 16-19", "blocks 20-23+tail"]


def active_layers(block_name):
    if block_name == "token_embedding":
        return set()
    return {int(x) for x in re.findall(r"blocks\.(\d+)", block_name)}


def pos_label(block_name):
    L = sorted(active_layers(block_name))
    if not L:
        return "token_embedding"
    return f"blocks {L[0]}-{L[-1]}" + ("+tail" if "final_norm" in block_name or "decoder" in block_name else "")


def components(row):
    act = active_layers(row["block"])
    lo = min(act) if act else -1
    c = defaultdict(float)
    for k, v in row["phases"].items():
        m = re.match(r"(fwd\.L|bwd\.recomp\.L|bwd\.grad\.L)(\d+)", k)
        if m:
            i = int(m.group(2))
            where = "active" if i in act else ("upstream" if i < lo or lo < 0 and False else "downstream")
            if lo >= 0 and i < lo:
                where = "upstream"
            kind = {"fwd.L": "layer fwd", "bwd.recomp.L": "layer recompute",
                    "bwd.grad.L": "layer backward"}[m.group(1)]
            c[f"{kind} ({where})"] += v
        else:
            c[k] += v
    return c


def kcat(name):
    n = name.lower()
    if "memcpy" in n or "memset" in n:
        return "memcpy/memset"
    if "_amax_kernel" in n or "_cast_kernel" in n:
        return "fp8 quantize"
    if "ternary" in n:
        return "ternary quantize"
    if "e4m3" in n or "e5m2" in n or "f8" in n or "fp8" in n:
        return "fp8 GEMM"
    if "gemm" in n or "cutlass" in n or "sm80" in n or "sm89" in n or "xmma" in n or "ampere" in n:
        return "bf16/fp32 GEMM"
    if "flash" in n or "fmha" in n or "attention" in n:
        return "attention"
    if "scan" in n or "delta" in n or "chunk" in n or "_fwd_kernel" in n or "_bwd_kernel" in n:
        return "scan (triton)"
    if "softmax" in n or "nll" in n or "cross_entropy" in n:
        return "loss softmax/nll"
    if "reduce" in n:
        return "reductions"
    if "elementwise" in n or "vectorized" in n or "unrolled" in n:
        return "elementwise"
    if "triton" in n:
        return "other triton"
    return "other"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--pos", default="2,3", help="positions in turn to use (1 = switch step)")
    ap.add_argument("--kernels", action="store_true")
    ap.add_argument("--per-run", action="store_true",
                    help="one row per run: wall step time and peak reserved per position")
    a = ap.parse_args()
    pos = {int(x) for x in a.pos.split(",")}
    if a.per_run:
        print("| run | " + " | ".join(ORDER) + " | mean s | max peak reserved |")
        print("|---|" + "---:|" * (len(ORDER) + 2))
        for r in a.runs:
            rows = [json.loads(l) for l in open(r)]
            by = defaultdict(list)
            for x in rows:
                if x["pos_in_turn"] in pos:
                    by[pos_label(x["block"])].append(x)
            secs = [sum(x["sec"] for x in by[b]) / len(by[b]) if by[b] else float("nan") for b in ORDER]
            peak = max(x["peak_res"] / x["total_mem"] for x in rows)
            print(f"| {r.split('/')[-1].split(chr(92))[-1]} | " + " | ".join(f"{v:.3f}" for v in secs)
                  + f" | {sum(secs) / len(secs):.3f} | {peak:.1%} |")
        return
    rows = [json.loads(l) for r in a.runs for l in open(r)]
    by = defaultdict(list)
    for r in rows:
        if r["pos_in_turn"] in pos:
            by[pos_label(r["block"])].append(r)
    blocks = [b for b in ORDER if b in by] + [b for b in by if b not in ORDER]
    comp = {}
    for b in blocks:
        acc = defaultdict(float)
        for r in by[b]:
            for k, v in components(r).items():
                acc[k] += v / len(by[b])
        acc["__sec"] = sum(r["sec"] for r in by[b]) / len(by[b])
        acc["__peak"] = max(r["peak_res"] / r["total_mem"] for r in by[b])
        comp[b] = acc
    keys = sorted({k for c in comp.values() for k in c if not k.startswith("__")},
                  key=lambda k: -sum(c.get(k, 0) for c in comp.values()))
    mean_sec = sum(comp[b]["__sec"] for b in blocks) / len(blocks)
    print(f"steps per position: {[len(by[b]) for b in blocks]}")
    print("| component | " + " | ".join(blocks) + " | mean s | % of step |")
    print("|---|" + "---:|" * (len(blocks) + 2))
    for k in keys:
        vals = [comp[b].get(k, 0.0) for b in blocks]
        m = sum(vals) / len(vals)
        print(f"| {k} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {m:.3f} | {m / mean_sec:.1%} |")
    print("| **step (wall)** | " + " | ".join(f"{comp[b]['__sec']:.3f}" for b in blocks)
          + f" | {mean_sec:.3f} | |")
    print("| peak reserved | " + " | ".join(f"{comp[b]['__peak']:.1%}" for b in blocks) + " | | |")
    if a.kernels:
        prof = [r for r in rows if "kernels" in r]
        cat = defaultdict(lambda: defaultdict(float))
        tops = defaultdict(float)
        for r in prof:
            b = pos_label(r["block"])
            for n, t in r["kernels"].items():
                cat[b][kcat(n)] += t
                tops[n] += t / len(prof)
        cats = sorted({k for c in cat.values() for k in c}, key=lambda k: -sum(c.get(k, 0) for c in cat.values()))
        pb = [b for b in blocks if b in cat]
        print("\n| kernel category | " + " | ".join(pb) + " | mean s |")
        print("|---|" + "---:|" * (len(pb) + 1))
        for k in cats:
            vals = [cat[b].get(k, 0.0) for b in pb]
            print(f"| {k} | " + " | ".join(f"{v:.3f}" for v in vals) + f" | {sum(vals) / len(vals):.3f} |")
        print("\ntop kernels (mean s per profiled step):")
        for n, t in sorted(tops.items(), key=lambda kv: -kv[1])[:40]:
            print(f"  {t:7.3f}  [{kcat(n)}] {n[:150]}")


if __name__ == "__main__":
    main()
