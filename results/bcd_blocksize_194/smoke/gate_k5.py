"""#194 k5 gate A + smoke verdict, from the smoke logs (rule in run_smoke_k5.sh).

Run from the worktree root: python results/bcd_blocksize_194/smoke/gate_k5.py
"""
import json, os, statistics as S

D = "results/bcd_blocksize_194"
O = D + "/smoke"
TOTAL = 6441926656  # bytes, RTX card total_memory (6143.4 MiB); overwritten below if torch is present
try:
    import torch
    TOTAL = torch.cuda.get_device_properties(0).total_memory
except Exception:
    pass


def rows(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def layer_span(block):
    ls = [int(m.split(".")[1]) for m in block.split("+") if m.startswith("blocks.")]
    return (min(ls), len(ls)) if ls else None


# depth predictor: least squares T = c0 + c1*start + c2*n_layers over the k2/k4
# long-run per-block medians (layer blocks only; the embedding turn is its own row)
X, Y, emb = [], [], []
for t in ("k2", "k4"):
    by = {}
    for r in rows(f"{D}/steps_{t}.jsonl"):
        by.setdefault(r["block"], []).append(r["sec"])
    for b, v in by.items():
        sp = layer_span(b)
        if sp:
            X.append((1.0, sp[0], sp[1])); Y.append(S.median(v))
        else:
            emb.append(S.median(v))
import numpy as np
coef, *_ = np.linalg.lstsq(np.array(X), np.array(Y), rcond=None)
resid = max(abs(float(np.dot(x, coef)) - y) for x, y in zip(X, Y))
print(f"depth fit: T = {coef[0]:.3f} {coef[1]:+.4f}*start {coef[2]:+.4f}*layers "
      f"(max residual on k2/k4 {resid:.3f} s); embedding turn {S.mean(emb):.3f} s")

ok = True
lay = json.load(open(f"{O}/layout_fp_k5.json"))
print(f"\nlayout: {lay['n_blocks']} blocks, layers per group {lay.get('layers_per_group')}")
for n, p in zip(lay["blocks"], lay["params"]):
    print(f"  {p:>11,}  {n}")

st = rows(f"{O}/steps_fp_k5.jsonl")
by = {}
for r in st:
    by.setdefault(r["block"], []).append(r["sec"])
print("\n(b) per-block step time (median of 3; first step of a turn carries fingerprint + load)")
meds = []
for b, v in by.items():
    m = S.median(v)
    meds.append(m)
    sp = layer_span(b)
    pred = float(np.dot((1.0, sp[0], sp[1]), coef)) if sp else S.mean(emb)
    bad = m > 1.5 * pred
    ok &= not bad
    print(f"  {m:7.3f} s  predicted {pred:7.3f}  ratio {m / pred:5.3f}  {'FAIL' if bad else 'ok'}  "
          f"steps {[round(s, 2) for s in v]}  {b}")
rot = S.mean(meds)
print(f"(c) mean step over a rotation (mean of per-block medians): {rot:.3f} s "
      f"(limit 13.9) {'ok' if rot <= 13.9 else 'FAIL'}")
ok &= rot <= 13.9

print("\n(a) peak reserved")
for r in rows(f"{O}/turns_fp_k5.jsonl"):
    pct = 100.0 * r["peak_res"] / TOTAL
    ok &= pct <= 85.0
    print(f"  turn  {r['block'][:40]:40s} @{r['step']:>3}: {r['peak_res'] / 2**20:6.0f} MiB {pct:5.1f}% "
          f"(alloc {r['peak_alloc'] / 2**20:.0f} MiB)  {'ok' if pct <= 85 else 'FAIL'}")
for r in rows(f"{O}/eval_fp_k5.jsonl"):
    pct = 100.0 * r["eval_peak_res"] / TOTAL
    ok &= pct <= 85.0
    print(f"  eval  @{r['step']:>3} active {r['block'][:40]:40s}: {r['eval_peak_res'] / 2**20:6.0f} MiB "
          f"{pct:5.1f}%  {r['eval_sec']:.0f} s  held-out {r['heldout_nll']:.5f}  {'ok' if pct <= 85 else 'FAIL'}")

print("\nsmoke")
fp = rows(f"{O}/fingerprint_fp_k5.jsonl")
ill = sum(len(r["illegal_changed"]) for r in fp)
print(f"  full-LR rotation: {len(fp)} turns, illegal changes {ill}, never changed at stored "
      f"precision after the rotation: {fp[-1]['never_changed_full_so_far']}, "
      f"active unchanged (stored) per turn {[len(r['active_unchanged_full']) for r in fp]}")
ok &= ill == 0 and fp[-1]["never_changed_full_so_far"] == 0 and len(fp) == lay["n_blocks"]
lrs = [min(r["lr"]) for r in st]
print(f"  lr at warmup 0: min {min(lrs):.3g} over {len(lrs)} steps")
ok &= min(lrs) > 0
if os.path.exists(f"{O}/fingerprint_neg_k5.jsonl"):
    neg = rows(f"{O}/fingerprint_neg_k5.jsonl")
    nch = sum(r["n_changed_full"] + r["n_changed"] for r in neg)
    print(f"  negative control (lr 0): {len(neg)} turns, params changed {nch} (must be 0), "
          f"blocks {[r['block'][:12] for r in neg]}")
    ok &= nch == 0 and len(neg) == 6
else:
    print("  negative control: MISSING"); ok = False
print("\nGATE", "PASS" if ok else "FAIL")
