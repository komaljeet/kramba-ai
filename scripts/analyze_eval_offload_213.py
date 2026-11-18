"""#213: summarise scripts/eval_offload_213.py output (off / on / falsify arms).

Author: Rambarun Komaljeet
"""
import json, os, sys

D = sys.argv[1] if len(sys.argv) > 1 else "results/eval_offload_213"


def rows(name):
    p = os.path.join(D, name)
    return [json.loads(l) for l in open(p) if l.strip()] if os.path.exists(p) else []


off, on, fal = rows("steps_off.jsonl"), rows("steps_on.jsonl"), rows("steps_falsify.jsonl")
ev = rows("evals_on.jsonl")
K = ("loss_sha", "grad_sha", "rng_sha")

print("== identity: eval every step vs eval off ==")
for a, b in zip(off, on):
    same = all(a[k] == b[k] for k in K)
    print(f"step {a['step']:>2} {a['block'][:28]:<28} lr {a['lr'][0]:.4g} nz {a['grad_nz']}/{a['grad_n']}"
          f"  {'IDENTICAL' if same else 'DIFF ' + str([k for k in K if a[k] != b[k]])}")
print(f"identical steps: {sum(all(a[k] == b[k] for k in K) for a, b in zip(off, on))}/{min(len(off), len(on))}")

print("\n== falsifier vs eval off ==")
for a, b in zip(off, fal):
    print(f"step {a['step']} " + " ".join(f"{k}:{'same' if a[k] == b[k] else 'DIFF'}" for k in K))

print("\n== evals (64 windows) ==")
shas = {e["target_sha"] for e in ev}
print(f"target sha distinct values: {len(shas)}")
for e in ev:
    t = e["total_mem"]
    print(f"after step {e['step']:>2} active {e['block'][:28]:<28} {e['eval_sec']:.1f}s "
          f"peak {e['eval_peak_res'] / t:.1%} loss {e['overall']:.5f}")
    if "harness_mean" in e:
        print(f"   harness {e['harness_mean']:.6f} vs trainer {e['overall']:.6f}  "
              f"bit-equal windows {e['bit_equal_windows']}/{e['windows']} "
              f"max|d| {e['max_abs_diff']:.3g}  harness {e['harness_sec']:.1f}s")

print("\n== step time and peak: off vs on (step after an eval) ==")
for a, b in zip(off, on):
    t = a["total_mem"]
    print(f"step {a['step']:>2} off {a['sec']:.2f}s peak {a['peak_res'] / t:.1%} | "
          f"on {b['sec']:.2f}s peak {b['peak_res'] / t:.1%} start-res {b['res_at_start'] / t:.1%}")
if ev and off and on:
    import statistics as st
    es = [e["eval_sec"] for e in ev]
    # step 1 is warm-up in both arms; compare steps 2.. (each on-step follows an eval)
    d = [b["sec"] - a["sec"] for a, b in zip(off[1:], on[1:])]
    print(f"\neval sec: mean {st.mean(es):.2f} min {min(es):.2f} max {max(es):.2f}")
    print(f"post-eval step slowdown: mean {st.mean(d):+.3f}s (n={len(d)})")
    for iv in (100, 500):
        print(f"per-step cost at --val-every {iv}: {(st.mean(es) + st.mean(d)) / iv:.3f} s/step "
              f"(= {100 * (st.mean(es) + st.mean(d)) / iv / st.mean(a['sec'] for a in off[1:]):.2f}% of a step)")
    print(f"max eval peak {max(e['eval_peak_res'] / e['total_mem'] for e in ev):.1%}, "
          f"max on-step peak {max(b['peak_res'] / b['total_mem'] for b in on):.1%}, "
          f"max off-step peak {max(a['peak_res'] / a['total_mem'] for a in off):.1%}")
