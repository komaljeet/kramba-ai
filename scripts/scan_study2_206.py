"""#206 step 2 (b)(c)(d): transposed-layout scan kernels at the 1B shape.

Author: Rambarun Komaljeet

Times the study kernels in scripts/scan_kernels_206.py against production at N = 64,
S = 4096, D = 32 (inputs as #203), and checks numbers two ways:
  vs production  same bf16/fp32 inputs, max |diff| and bit-equality
  vs fp64        N = 8, fp32 inputs (so gradients are stored fp32), a float64 torch
                 reference of the recurrence and its adjoint (autograd in fp64):
                 relative error ||x - ref|| / ||ref|| and max |x - ref| for
                 production and candidate, so we know which is closer.

Falsifier for the "closer to fp64" comparison: the same metric on production run
twice must give exactly the same number, and a deliberately wrong candidate (beta
scaled by 1 + 2^-10) must be clearly farther from fp64 than either kernel.
"""
import json, sys, os
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import hvn.core.deltanet_triton as DT
from scan_study_206 import Prob, ev_time, S, D, dev
import scan_kernels_206 as K


def fwdT(p, BV, nw, write_o=True, write_ckpt=False, o=None, ckpt=None):
    o = p.o if o is None else o
    ckpt = p.ckpt if ckpt is None else ckpt
    return K.fwdT_kernel[(p.N, D // BV)](
        p.q, p.k, p.v, p.a, p.beta3, o, ckpt, p.nblk, p.H, p.sB, p.sS, p.sH,
        S=p.S, D=D, BV=BV, B=p.B, WRITE_O=write_o, WRITE_CKPT=write_ckpt, num_warps=nw)


def bwdT(p, BV, nw, parts=None):
    split = D // BV
    if split == 1:
        dq, dk, da, db = p.dq, p.dk, p.da, p.db
    else:
        if parts is None or parts[0].shape[0] != split:
            parts = (torch.empty((split,) + tuple(p.q.shape), device=dev),
                     torch.empty((split,) + tuple(p.q.shape), device=dev),
                     torch.empty((split,) + tuple(p.q.shape), device=dev),
                     torch.empty((split,) + tuple(p.beta3.shape), device=dev))
        dq, dk, da, db = parts
    K.bwdT_kernel[(p.N, split)](
        p.q, p.k, p.v, p.a, p.beta3, p.ckpt, p.do, dq, dk, p.dv, da, db, p.nblk,
        p.H, p.sB, p.sS, p.sH, S=p.S, D=D, B=p.B, BV=BV, SPLIT=split, num_warps=nw)
    if split > 1:
        p.dq.copy_(dq.sum(0)); p.dk.copy_(dk.sum(0)); p.da.copy_(da.sum(0)); p.db.copy_(db.sum(0))
    return parts


def bits(x):
    return x.view(torch.int16 if x.dtype == torch.bfloat16 else torch.int32)


def cmp(a, b):
    return {"bitequal": bool(torch.equal(bits(a), bits(b))),
            "maxdiff": (a.float() - b.float()).abs().max().item(),
            "frac_diff": (bits(a) != bits(b)).float().mean().item()}


# --------------------------------------------------------------------------- fp64
def ref64(p):
    """fp64 forward + adjoint of the recurrence via autograd, [N,S,D] layout."""
    N, Sx = p.N, p.S
    toN = lambda t: t.permute(0, 2, 1, 3).reshape(N, Sx, -1).double().detach().requires_grad_(True)
    q, k, v, a = (toN(t) for t in (p.q, p.k, p.v, p.a))
    b = p.beta3.permute(0, 2, 1).reshape(N, Sx).double().detach().requires_grad_(True)
    st = torch.zeros(N, D, D, dtype=torch.float64, device=dev)   # [dv, dk]
    outs = []
    for t in range(Sx):
        st = st * a[:, t, None, :]
        Sk = (st * k[:, t, None, :]).sum(-1)
        u = v[:, t] - Sk
        st = st + b[:, t, None, None] * (u[:, :, None] * k[:, t, None, :])
        outs.append((st * q[:, t, None, :]).sum(-1))
    o = torch.stack(outs, 1)
    o.backward(p.do.double())
    back = lambda g: g.reshape(p.Bt, p.H, Sx, -1).permute(0, 2, 1, 3)
    return {"o": o.detach(), "dq": back(q.grad), "dk": back(k.grad), "dv": back(v.grad),
            "da": back(a.grad), "db": b.grad.reshape(p.Bt, p.H, Sx).permute(0, 2, 1)}


def errs(x, r):
    x = x.double(); r = r.double()
    return {"rel": ((x - r).norm() / r.norm()).item(), "max": (x - r).abs().max().item()}


def accuracy(out):
    """N = 8, S = 4096, fp32 inputs. Production vs candidates vs fp64."""
    p = Prob(1, 8)
    p.q, p.k, p.v = p.q.float(), p.k.float(), p.v.float()
    p.dq, p.dk, p.dv = (torch.empty_like(p.q) for _ in range(3))
    p.sB, p.sS, p.sH, _ = p.q.stride()
    p.S = S
    r = ref64(p)

    def grab():
        return {"o": p.o.clone(), "dq": p.dq.clone(), "dk": p.dk.clone(), "dv": p.dv.clone(),
                "da": p.da.clone(), "db": p.db.clone()}

    def score(g):
        return {kk: errs(g[kk], r[kk].reshape(g[kk].shape)) for kk in g}

    res = {}
    # production twice (determinism of the metric)
    for tag in ("prod", "prod_again"):
        p.fwd(); p.ck(); p.bw(); torch.cuda.synchronize()
        res[tag] = score(grab())
    # candidate: transposed fwd (BV 32 nw 1) + fused ckpt + transposed bwd (BV 32 nw 1)
    for tag, fbv, fnw, bbv, bnw in (("T_shipped_fwd8w2_bwd32w1", 8, 2, 32, 1),
                                     ("T_fwd32w1_bwd32w1", 32, 1, 32, 1),
                                     ("prod_bwd_nw2", None, None, None, 2)):
        if fbv is None:
            p.fwd(); p.ck(); p.bw(num_warps=bnw)
        else:
            fwdT(p, fbv, fnw, write_o=True, write_ckpt=True)
            bwdT(p, bbv, bnw)
        torch.cuda.synchronize()
        res[tag] = score(grab())
    # falsifier: beta scaled by 1 + 2^-10 in production
    keep = p.beta3.clone(); p.beta3.mul_(1 + 2 ** -10)
    p.fwd(); p.ck(); p.bw(); torch.cuda.synchronize()
    res["falsifier_beta_1e-3"] = score(grab())
    p.beta3.copy_(keep)
    out["accuracy_N8_fp32in"] = res
    for kk, vv in res.items():
        print(kk, {a: f"{b['rel']:.2e}" for a, b in vv.items()}, flush=True)


def timing(out):
    p = Prob(1, 64); p.S = S
    p.fwd(); p.ck(); p.bw(); torch.cuda.synchronize()
    ref = p.outs()   # o, ckpt, dq, dk, dv, da, db
    res = {"prod": {"fwd": ev_time(p.fwd), "ckpt": ev_time(p.ck), "bwd": ev_time(p.bw),
                    "fwd_nw1": ev_time(lambda: p.fwd(num_warps=1)),
                    "ckpt_nw8": ev_time(lambda: p.ck(num_warps=8)),
                    "bwd_nw2": ev_time(lambda: p.bw(num_warps=2))}}
    print(res, flush=True)
    rows = []
    for BV in ((8, 16, 32) if not os.environ.get("SKIP_FWD") else ()):
        for nw in (1, 2, 4):
            for mode, wo, wc in (("fwd", True, False), ("ckpt", False, True), ("fused", True, True)):
                try:
                    ck = fwdT(p, BV, nw, wo, wc); torch.cuda.synchronize()
                    row = {"kernel": "fwdT", "mode": mode, "BV": BV, "nw": nw,
                           "ms": ev_time(lambda: fwdT(p, BV, nw, wo, wc)),
                           "regs": ck.n_regs, "spills": ck.n_spills}
                    if wo:
                        row["o_vs_prod"] = cmp(p.o, ref[0])
                    if wc:
                        row["ckpt_vs_prod"] = cmp(p.ckpt, ref[1])
                except Exception as e:
                    row = {"kernel": "fwdT", "mode": mode, "BV": BV, "nw": nw, "error": repr(e)[:300]}
                rows.append(row); print(row, flush=True)
    p.ck(); torch.cuda.synchronize()   # production checkpoint for every bwd
    for BV in (32, 16, 8):
        for nw in (1, 2, 4):
            try:
                parts = bwdT(p, BV, nw); torch.cuda.synchronize()
                row = {"kernel": "bwdT", "BV": BV, "split": D // BV, "nw": nw,
                       "ms": ev_time(lambda: bwdT(p, BV, nw, parts))}
                for i, nm in zip(range(2, 7), ("dq", "dk", "dv", "da", "db")):
                    row[nm + "_vs_prod"] = cmp(p.outs()[i], ref[i])
            except Exception as e:
                row = {"kernel": "bwdT", "BV": BV, "nw": nw, "error": repr(e)[:300]}
            rows.append(row); print(row, flush=True)
    res["rows"] = rows
    out["timing_N64" + ("_bwd" if os.environ.get("SKIP_FWD") else "")] = res


if __name__ == "__main__":
    path = sys.argv[1]
    which = sys.argv[2].split(",") if len(sys.argv) > 2 else ["timing", "accuracy"]
    out = json.load(open(path)) if os.path.exists(path) else {}
    for w in which:
        {"timing": timing, "accuracy": accuracy}[w](out)
        json.dump(out, open(path, "w"), indent=1)
