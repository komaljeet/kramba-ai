"""Summarize scripts/ab_offload_losshead_1b.py arms into one JSON + a printout.

Speed: cycle-mean wall over the measured (non-warmup) steps -- one step per BCD
block. Correctness: per-step loss compared BIT FOR BIT (hex) between arms that
must agree exactly:
  main  vs D (branch, fast head off)   -> lever 1 alone is pure scheduling
  B     vs C (branch, old blanket wait) -> new prefetch ordering vs old, same values
  main  vs B, step 0                   -> both levers, initial weights
  main  vs E (main, loss chunk 256)    -> size of a bf16 drift main already accepts
VRAM: per-step peak reserved, max over the cycle, and the trainer's #163 guard
view (cumulative peak through step 1, 85% of the card).
"""
import json
import os
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "offload_losshead_1b")
arms = {}
for f in sorted(os.listdir(D)):
    if f.endswith(".json") and not f.startswith("summary"):
        arms[f[:-5]] = json.load(open(os.path.join(D, f)))


def meas(a):
    return [s for s in a["steps"] if not s["warmup"]]


summ = {"arms": {}, "pairs": {}}
G = 2 ** 30
for n, a in arms.items():
    m = meas(a)
    card = a["card_bytes"]
    if not m:
        summ["arms"][n] = {"aborted": a.get("aborted"), "n": 0}
        continue
    guard_view = max([a.get("build_peak_reserved", 0)] + [s["peak_reserved"] for s in a["steps"][:1]])
    summ["arms"][n] = {
        "n": len(m), "aborted": a.get("aborted"),
        "cycle_mean_wall": sum(s["wall"] for s in m) / len(m),
        "cycle_mean_fwd_gpu": sum(s["fwd_gpu"] for s in m) / len(m),
        "blocks": sorted({s["active"] for s in m}),
        "peak_reserved_gib": max(s["peak_reserved"] for s in a["steps"]) / G,
        "peak_reserved_pct": 100 * max(s["peak_reserved"] for s in a["steps"]) / card,
        "peak_alloc_gib": max(s["peak_alloc"] for s in a["steps"]) / G,
        "peak_block": max(a["steps"], key=lambda s: s["peak_reserved"])["active"],
        "guard_view_step1_pct": 100 * guard_view / card,
        "min_driver_free_gib": min(s["driver_free_after"] for s in a["steps"]) / G,
        "min_ram_free_gib": min(s["ram_free_gib"] for s in a["steps"]),
        "step0_loss_hex": a["steps"][0]["loss_hex"],
        "per_block_wall": {s["active"]: s["wall"] for s in m},
    }


def cmp(x, y):
    if x not in arms or y not in arms:
        return None
    sx, sy = arms[x]["steps"], arms[y]["steps"]
    k = min(len(sx), len(sy))
    same = [sx[i]["loss_hex"] == sy[i]["loss_hex"] for i in range(k)]
    return {"steps_compared": k, "bit_identical_steps": sum(same),
            "first_mismatch": next((i for i, v in enumerate(same) if not v), None),
            "max_abs_dloss": max(abs(sx[i]["loss"] - sy[i]["loss"]) for i in range(k)),
            "step0_identical": same[0] if k else None}


for x in [n for n in arms if n.startswith("A")]:
    for y in [n for n in arms if n.startswith(("B", "D"))]:
        summ["pairs"][f"{x}~{y}"] = cmp(x, y)
for x in [n for n in arms if n.startswith("B")]:
    summ["pairs"][f"{x}~C_branch_blanket"] = cmp(x, "C_branch_blanket")
for x in [n for n in arms if n.startswith("A")]:
    # control: a bf16 reduction-order change main already accepts (chunk 512 -> 256)
    summ["pairs"][f"{x}~E_main_chunk256"] = cmp(x, "E_main_chunk256")
for x, y in (("A1_main", "A2_main"), ("B1_branch", "B2_branch")):
    summ["pairs"][f"{x}~{y}"] = cmp(x, y)

print("%-16s %4s %9s %8s %9s %7s %8s %-22s" % ("arm", "n", "wall/step", "fwd gpu", "peak rsv", "% card",
                                               "guard1%", "peak block"))
for n, s in summ["arms"].items():
    if not s.get("n"):
        print(n, "ABORTED/empty:", s.get("aborted"))
        continue
    print("%-16s %4d %9.3f %8.3f %8.2fG %6.1f%% %7.1f%% %-22s" % (
        n, s["n"], s["cycle_mean_wall"], s["cycle_mean_fwd_gpu"], s["peak_reserved_gib"],
        s["peak_reserved_pct"], s["guard_view_step1_pct"], s["peak_block"]))
A = [summ["arms"][n]["cycle_mean_wall"] for n in summ["arms"] if n.startswith("A") and summ["arms"][n].get("n")]
B = [summ["arms"][n]["cycle_mean_wall"] for n in summ["arms"] if n.startswith("B") and summ["arms"][n].get("n")]
if A and B:
    a, b = sum(A) / len(A), sum(B) / len(B)
    summ["combined"] = {"main_mean": a, "branch_mean": b, "saving_s": a - b, "saving_pct": 100 * (a - b) / a}
    print("\ncombined: main %.3f -> branch %.3f s/step = -%.3f s (%.1f%%)" % (a, b, a - b, 100 * (a - b) / a))
for k, v in summ["pairs"].items():
    print("%-30s %s" % (k, v))
json.dump(summ, open(os.path.join(D, "summary.json"), "w"), indent=1)
