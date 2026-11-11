"""#197 Stage 1 comparator: step-0 gradients of two harness runs (exp_fp8_grad_197.py).

usage: python exp_fp8_grad_197_cmp.py <dir> <tagX> <tagRef> [--json out.json]

Refuses to compare unless both runs saw the same active-block weights and the
same 8 micro-batches (sha256 recorded by the harness). Reports, vs the reference:
relative error ||X-R||/||R|| and cosine for the whole block gradient and split by
tensor type, plus the per-tensor median / max relative error.
"""
import json, os, sys
import torch

ATTN_LAYERS = {6, 18}   # --attn-layers 6,18 in PROD_ARGV


def kind(name):
    if name.startswith("token_embedding") or name.startswith("decoder"):
        return "embedding"
    parts = name.split(".")
    if parts[0] != "blocks":
        return "other"
    if ".ffn." in name:
        return "ffn"
    if name.endswith("norm.weight") and parts[2] == "norm":
        return "norm"
    return "attention" if int(parts[1]) in ATTN_LAYERS else "deltanet"


def compare(d, tx, tr):
    mx = json.load(open(os.path.join(d, f"m_{tx}.json")))
    mr = json.load(open(os.path.join(d, f"m_{tr}.json")))
    assert mx["param_sha"] == mr["param_sha"], f"{tx}/{tr}: different weights"
    assert mx["target_sha"] == mr["target_sha"], f"{tx}/{tr}: different data"
    gx = torch.load(os.path.join(d, f"g_{tx}.pt"))
    gr = torch.load(os.path.join(d, f"g_{tr}.pt"))
    assert gx.keys() == gr.keys(), "different trainable sets"
    acc = {}
    per = []
    for n in gr:
        x, r = gx[n].double(), gr[n].double()
        k = kind(n)
        a = acc.setdefault(k, [0.0, 0.0, 0.0, 0.0, 0, 0])
        a[0] += float(((x - r) ** 2).sum()); a[1] += float((r ** 2).sum())
        a[2] += float((x * r).sum()); a[3] += float((x ** 2).sum())
        a[4] += x.numel(); a[5] += int((x != r).sum())
        rn = float(r.norm())
        per.append((n, float((x - r).norm()) / rn if rn > 0 else float("inf")))
    def summ(a):
        return {"rel_err": (a[0] / a[1]) ** 0.5 if a[1] > 0 else float("nan"),
                "cos": a[2] / (a[1] * a[3]) ** 0.5 if a[1] * a[3] > 0 else float("nan"),
                "numel": a[4], "n_differ": a[5]}
    tot = [sum(v[i] for v in acc.values()) for i in range(6)]
    rs = sorted(v for _, v in per)
    out = {"x": tx, "ref": tr, "block": mr["block"], "active": mr["active"],
           "all": summ(tot), "by_kind": {k: summ(v) for k, v in sorted(acc.items())},
           "per_tensor_median": rs[len(rs) // 2], "per_tensor_max": rs[-1],
           "worst_tensor": max(per, key=lambda t: t[1])[0],
           "peak_res_x": mx["peak_res"], "peak_res_ref": mr["peak_res"],
           "total_mem": mr["total_mem"]}
    return out


if __name__ == "__main__":
    d, tx, tr = sys.argv[1:4]
    o = compare(d, tx, tr)
    print(json.dumps(o, indent=1))
    if "--json" in sys.argv:
        json.dump(o, open(sys.argv[sys.argv.index("--json") + 1], "w"), indent=1)
