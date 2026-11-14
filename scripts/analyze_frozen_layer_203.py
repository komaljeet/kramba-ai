"""#203: cut a frozen_layer_203.py kernel trace into phases and class every kernel.

Author: Rambarun Komaljeet

Phases are delimited by the ``_phase_marker_203`` kernels on the compute stream; the
k-th marker carries the k-th label the harness recorded. Per phase:
  wall    first marker start -> next marker start (GPU clock)
  busy    sum of compute-stream kernel/memcpy/memset durations (one stream: serial)
  idle    wall - busy: the compute stream had nothing to run (host behind, or a
          stream wait on the weight copy)
  side    side-stream (weight prefetch) memcpy time overlapping the phase window
  host    CUDA runtime calls on the host overlapping the window, by name (to see
          whether an idle gap is the host blocked in a sync)

Falsifier for the cut itself: the sum of phase walls must equal the marker span and
each phase wall must match the CUDA-event phase time of the same micro-batch within
a few percent; otherwise the markers were mis-ordered and the table is void.
"""
import collections, gzip, json, os, sys

CLASSES = [
    ("marker", ("_phase_marker_203",)),
    ("scan_fwd", ("_gated_delta_fwd_kernel",)),
    ("scan_bwd", ("_mk_ckpt_kernel", "_mk_bwd_kernel")),
    ("quant_weight", ("_ternary_fp8_kernel", "_quantize_kernel", "_absmean")),
    ("quant_act", ("_amax_kernel", "_cast_kernel")),
    ("memcpy", ("memcpy",)),
    ("memset", ("memset",)),
]


def kclass(name):
    low = name.lower()
    for c, keys in CLASSES:
        if any(k.lower() in low for k in keys):
            return c
    if any(k in low for k in ("gemm", "xmma", "cutlass", "s16816", "s1688", "matmul")):
        return "gemm_fp8" if any(k in low for k in ("e4m3", "e5m2", "fp8", "f8")) else "gemm_bf16"
    if any(k in low for k in ("flash", "fmha", "attention", "_attn")):
        return "attention"
    return "elementwise_norm_other"


def main(path, phase_filter=None):
    meta = json.load(open(path))
    tp = meta["trace"]
    tr = json.load(open(tp) if os.path.exists(tp) else gzip.open(tp + ".gz", "rt"))
    labels = meta["trace_labels"]
    evs = tr["traceEvents"] if isinstance(tr, dict) else tr
    gpu = [e for e in evs if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")]
    host = [e for e in evs if e.get("ph") == "X" and e.get("cat") == "cuda_runtime"]
    cpu_ops = [e for e in evs if e.get("ph") == "X" and e.get("cat") == "cpu_op"]
    marks = sorted((e for e in gpu if "_phase_marker_203" in e["name"]), key=lambda e: e["ts"])
    assert len(marks) == len(labels), (len(marks), len(labels))
    cs = marks[0]["args"]["stream"]
    comp = sorted((e for e in gpu if e["args"].get("stream") == cs), key=lambda e: e["ts"])
    side = [e for e in gpu if e["args"].get("stream") != cs]
    res = collections.OrderedDict()
    for k in range(len(marks) - 1):
        t0, t1 = marks[k]["ts"], marks[k + 1]["ts"]
        lab = labels[k]
        r = res.setdefault(lab, {"n": 0, "wall": 0.0, "busy": 0.0, "cls": collections.Counter(),
                                 "kern": collections.Counter(), "nk": 0, "side": 0.0,
                                 "host": collections.Counter(), "cpu": collections.Counter()})
        r["n"] += 1
        r["wall"] += t1 - t0
        for e in comp:
            if t0 <= e["ts"] < t1:
                c = kclass(e["name"])
                r["busy"] += e["dur"]; r["cls"][c] += e["dur"]; r["nk"] += 1
                r["kern"][(c, e["name"][:110])] += e["dur"]
        for e in side:
            a, b = max(t0, e["ts"]), min(t1, e["ts"] + e["dur"])
            if b > a:
                r["side"] += b - a
        for e in host:
            a, b = max(t0, e["ts"]), min(t1, e["ts"] + e["dur"])
            if b > a:
                r["host"][e["name"]] += b - a
        for e in cpu_ops:
            if e["name"] in ("aten::copy_",) and e.get("dur", 0) > 1000:
                a, b = max(t0, e["ts"]), min(t1, e["ts"] + e["dur"])
                if b > a:
                    r["cpu"]["aten::copy_ >1ms (host)"] += b - a
    return res


if __name__ == "__main__":
    res = main(sys.argv[1])
    want = sys.argv[2].split(",") if len(sys.argv) > 2 else list(res)
    for lab in want:
        if lab not in res:
            continue
        r = res[lab]
        ms = lambda us: us / 1e3
        print(f"\n== {lab}: wall {ms(r['wall']):.2f} ms  busy {ms(r['busy']):.2f}  "
              f"idle {ms(r['wall']-r['busy']):.2f}  kernels {r['nk']}  side-copy {ms(r['side']):.2f}")
        for c, v in r["cls"].most_common():
            print(f"   {c:<24}{ms(v):8.2f} ms")
        print("   top kernels:")
        for (c, n), v in r["kern"].most_common(14):
            print(f"     {ms(v):7.3f} [{c}] {n}")
        print("   host runtime:", {k: round(ms(v), 2) for k, v in r["host"].most_common(5)},
              {k: round(ms(v), 2) for k, v in r["cpu"].items()})
