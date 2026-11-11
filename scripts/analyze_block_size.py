"""Does a bigger BCD block shorten the full-quality pretrain estimate? (#192)

Joins two measurements:
  quality  results/local_loss/ab_d1024_L16_blocksize.json (+ the #188 baseline arms):
           held-out loss at equal steps for 1/2/3/4 layers per block, d1024 x 16 layers
  cost     results/bcd_memory/ga8_k{2,3,4,6}.jsonl: real 1B step time per block, at the
           production grad_accum=8, cycle average = mean of per-block medians

Mapping (an ASSUMPTION, stated here so it can be attacked): BCD's cost is
under-training, each layer's block trains 1/N of the steps (#151), so a 1B layout with N
blocks per rotation is matched to the small-scale quality at the same N, interpolated
linearly in 1/N.

Days follow the project's own convention, 211 = 132 x 1.6: the full-quality estimate is
the BCD-accepted estimate times BCD's perplexity penalty, exp(gap to full training). That
convention is crude; the RELATIVE change between block sizes is what this reports.

Usage: python scripts/analyze_block_size.py
"""
import json
import math
import os
import statistics as S

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SMALL_ARMS = ["bcd_k1", "bcd_k2", "bcd_k3", "bcd_balanced"]
ONE_B = [2, 3, 4, 6]                                                  # layers per block
TAIL = "final_norm+decoder"
# A block that is ONLY final_norm+decoder trains just a LayerNorm (the tied decoder is the
# embedding table), so its turn is wasted. Some arms/configs have one because the budget
# packing left the tail alone; production merges it into the last layer group. It is
# excluded from the block count and the step-time average, as if merged.
BASE_DAYS_ACCEPTED = 132.0


def cycle_avg(path):
    rows = [json.loads(l) for l in open(path, encoding="utf-8")]
    per = {}
    for r in rows:
        if r["step_sec"] and r["step_in_block"] >= 1 and r["block"] != TAIL:
            per.setdefault(r["block"], []).append(r["step_sec"])
    return S.mean(S.median(v) for v in per.values()), len(per)


def interp(points, x):
    pts = sorted(points)
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 + (x - x0) / (x1 - x0) * (y1 - y0)
    raise ValueError(f"{x} outside {pts[0][0]}..{pts[-1][0]}")


def main():
    d = json.load(open(os.path.join(REPO, "results/local_loss/ab_d1024_L16_blocksize.json"),
                       encoding="utf-8"))
    arms = d["arms"]
    full = arms["full"]["final_heldout_loss"]
    SMALL = {}
    for a in SMALL_ARMS:
        SMALL[sum(1 for b in arms[a]["blocks"] if b[0] != TAIL)] = a   # blocks that train
    gap = {n: arms[a]["final_heldout_loss"] - full for n, a in SMALL.items()}
    print("small-scale (d1024 x 16, 4000 steps): training blocks -> held-out, gap to full")
    for n, a in SMALL.items():
        print(f"  {n:>2} blocks ({a:<13}) {arms[a]['final_heldout_loss']:.4f}  gap {gap[n]:.4f}"
              f"  ppl x{math.exp(gap[n]):.2f}  train {arms[a]['train_sec']/60:.1f} min")
    pts = [(1.0 / n, g) for n, g in gap.items()]

    print("\n1B projection (production grad_accum=8 step times; today = 2 layers/block)")
    hdr = f"{'layers/blk':>10}{'blocks':>7}{'step s':>8}{'vs 2':>7}{'gap':>7}{'ppl x':>7}" \
          f"{'rel days':>10}{'days@211':>10}"
    print(hdr)
    out = {}
    for k in ONE_B:
        path = os.path.join(REPO, f"results/bcd_memory/ga8_k{k}.jsonl")
        if not os.path.exists(path):
            print(f"{k:>10}  (no ga8 data)")
            continue
        t, n = cycle_avg(path)
        out[k] = dict(blocks=n, step=t, gap=interp(pts, 1.0 / n))
    if 2 in out:
        base = out[2]["step"] * math.exp(out[2]["gap"])
        for k, r in out.items():
            rel = r["step"] * math.exp(r["gap"]) / base
            print(f"{k:>10}{r['blocks']:>7}{r['step']:>8.3f}{r['step']/out[2]['step']:>7.3f}"
                  f"{r['gap']:>7.3f}{math.exp(r['gap']):>7.2f}{rel:>10.3f}{211*rel:>10.0f}")
        print(f"\nunder the project's convention today's full-quality figure would be "
              f"{BASE_DAYS_ACCEPTED * math.exp(out[2]['gap']):.0f} days "
              f"(132 x exp({out[2]['gap']:.3f})), not 211 (132 x 1.6): the 1.6 came from a "
              f"layout with half the layers per block.")


if __name__ == "__main__":
    main()
