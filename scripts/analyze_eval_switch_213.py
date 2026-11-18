"""#213: eval right after each BCD switch (--val-every = --spb) vs eval off, 16 steps.

Author: Rambarun Komaljeet
"""
import json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else "results/eval_offload_213"


def rows(n):
    return [json.loads(l) for l in open(os.path.join(D, n)) if l.strip()]


off, on, ev = rows("steps_off16.jsonl"), rows("steps_onsw.jsonl"), rows("evals_onsw.jsonl")
K = ("loss_sha", "grad_sha", "rng_sha")
same = [all(a[k] == b[k] for k in K) for a, b in zip(off, on)]
print(f"identical steps: {sum(same)}/{len(same)}")
evs = {e["step"]: e for e in ev}
worst = 0.0
for a, b in zip(off, on):
    t = a["total_mem"]
    e = evs.get(b["step"])
    d = (b["peak_res"] - a["peak_res"]) / t
    worst = max(worst, b["peak_res"] / t)
    print(f"step {a['step']:>2} {a['block'][:24]:<24} off {a['peak_res'] / t:.1%} on {b['peak_res'] / t:.1%} "
          f"({d:+.1%}) sec {a['sec']:.2f}/{b['sec']:.2f}"
          + (f" | eval after: {e['eval_sec']:.1f}s peak {e['eval_peak_res'] / t:.1%} "
             f"active {e['block'][:20]}" if e else ""))
print(f"max on-step peak {worst:.1%}; max eval peak "
      f"{max(e['eval_peak_res'] / e['total_mem'] for e in ev):.1%}")
