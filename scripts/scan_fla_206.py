"""#206 step 2 (a): flash-linear-attention's chunked KDA kernels as the chunk-parallel reference.

Author: Rambarun Komaljeet

KDA's recurrence (fla.ops.kda) is S_t = (I - b k k^T) diag(exp g_t) S_{t-1} + b k v^T with
S in [K, V], o_t = scale * S_t^T q_t. Transposing ours (S [d_v, d_k],
S_t = S_{t-1} diag(alpha)(I - b k k^T) + b v k^T, o = S q) gives exactly that with
g = log(alpha), scale = 1. fla is installed in the venv (0.5.1) but is NOT a repo
dependency; nothing here adds it.

Times chunk_kda forward and forward+backward at the 1B shape (B 1, T 4096, H 64, K = V
= 32), bf16 q/k/v, fp32 g and beta, and checks it against the fp64 reference of
scan_study2_206 (N = 8 heads) next to production. Falsifier for the mapping: if g or
the transpose were wrong the rel error vs fp64 would be O(1), not O(1e-3).
"""
import json, sys, os
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
from scan_study_206 import Prob, ev_time, S, D, dev
from scan_study2_206 import ref64, errs
from fla.ops.kda import chunk_kda


def kda(p, dtype=torch.bfloat16):
    g = torch.log(p.a)                              # [B,T,H,K] fp32
    beta = p.beta3                                  # [B,T,H]
    o, _ = chunk_kda(p.q.to(dtype), p.k.to(dtype), p.v.to(dtype), g, beta, scale=1.0)
    return o


def main(path):
    out = {}
    p = Prob(1, 64)
    leaf = [t.detach().clone().requires_grad_(True) for t in (p.q, p.k, p.v, p.a, p.beta3)]

    def fwd():
        with torch.no_grad():
            g = torch.log(p.a)
            return chunk_kda(p.q, p.k, p.v, g, p.beta3, scale=1.0)[0]

    do = p.do.reshape(1, 64, S, D).permute(0, 2, 1, 3).contiguous().bfloat16()

    def fb():
        q, k, v, a, b = leaf
        o, _ = chunk_kda(q, k, v, torch.log(a), b, scale=1.0)
        o.backward(do)

    out["kda_fwd_ms"] = ev_time(fwd, 10)
    out["kda_fwd_bwd_ms"] = ev_time(fb, 10)
    out["prod"] = {"fwd": ev_time(p.fwd), "ckpt": ev_time(p.ck), "bwd": ev_time(p.bw)}
    print(out, flush=True)

    # accuracy vs fp64, N = 8, fp32 inputs for production; KDA in bf16 and fp32 inputs
    p8 = Prob(1, 8); p8.S = S
    p8.q, p8.k, p8.v = p8.q.float(), p8.k.float(), p8.v.float()
    p8.dq, p8.dk, p8.dv = (torch.empty_like(p8.q) for _ in range(3))
    r = ref64(p8)
    acc = {}
    p8.fwd(); p8.ck(); p8.bw(); torch.cuda.synchronize()
    acc["prod_fp32in"] = {"o": errs(p8.o, r["o"]), "dq": errs(p8.dq, r["dq"]), "da": errs(p8.da, r["da"]),
                          "db": errs(p8.db, r["db"])}
    do8 = p8.do.reshape(1, 8, S, D).permute(0, 2, 1, 3).contiguous()
    for dt in (torch.bfloat16, torch.float32):
        try:
            q, k, v = (t.to(dt).detach().requires_grad_(True) for t in (p8.q, p8.k, p8.v))
            a = p8.a.detach().clone().requires_grad_(True); b = p8.beta3.detach().clone().requires_grad_(True)
            o, _ = chunk_kda(q, k, v, torch.log(a), b, scale=1.0)
            o.backward(do8.to(dt))
            oN = o.float().permute(0, 2, 1, 3).reshape(8, S, D)
            acc[f"kda_{str(dt)[6:]}"] = {"o": errs(oN, r["o"]), "dq": errs(q.grad.float(), r["dq"]),
                                         "da": errs(a.grad, r["da"]), "db": errs(b.grad, r["db"])}
        except Exception as e:
            acc[f"kda_{str(dt)[6:]}"] = {"error": repr(e)[:300]}
    out["accuracy_N8"] = acc
    for kk, vv in acc.items():
        print(kk, vv, flush=True)
    json.dump(out, open(path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
