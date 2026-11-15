"""#207 step 5: is batching Muon's Newton-Schulz over same-shape matrices bit-identical?

The active block at 1B k4 (4 DeltaNet layers) holds, per layer, six 2048x2048 Muon
matrices (q, k, v, alpha, gate, out), one 4096x2048 and one 2048x4096 (FFN), and the
64x2048 beta projection. ``zeropower_via_newtonschulz5`` runs once per matrix.
Batched variant: stack same-shape matrices, per-matrix Frobenius norm, ``@`` on the
3-D stack (cuBLAS strided-batched GEMM). Same element-wise ops in the same order.

Pre-registered (before running): "identical" needs torch.equal on every matrix's
update for every group; "worth it" needs the batched time per step to be >= 0.05 s
lower than the loop at these shapes. Either failing = not adopted.
Falsifier: the loop on an input with one element moved by one bf16 ulp must break
torch.equal (a first try, eps 1e-7 -> 1.0000001e-7, rounded away and did not).
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from hvn.optimizers import zeropower_via_newtonschulz5 as ns_loop


def ns_batched(G, steps=5, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    nrm = torch.stack([x.norm() for x in X])          # same reduction as X.norm() per matrix
    X = X / (nrm.view(-1, 1, 1) + eps)
    transposed = G.size(-2) > G.size(-1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


def timed(fn, n=5):
    fn(); torch.cuda.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter(); fn(); torch.cuda.synchronize(); ts.append(time.perf_counter() - t0)
    return sorted(ts)[len(ts) // 2]


def main():
    torch.manual_seed(0)
    groups = {(2048, 2048): 24, (4096, 2048): 4, (2048, 4096): 4, (64, 2048): 4}
    res = {}
    tot_loop = tot_b = 0.0
    for shape, n in groups.items():
        G = torch.randn(n, *shape, device="cuda") * 0.01
        ref = [ns_loop(G[i]) for i in range(n)]
        out = ns_batched(G)
        eq = sum(torch.equal(ref[i], out[i]) for i in range(n))
        maxd = max(float((ref[i].float() - out[i].float()).abs().max()) for i in range(n))
        G0 = G[0].clone()
        G0[0, 0] += G0[0, 0].abs() * 2 ** -7      # one bf16 ulp on one input element
        fals = ns_loop(G0)
        tl_ = timed(lambda: [ns_loop(G[i]) for i in range(n)])
        tb = timed(lambda: ns_batched(G))
        tot_loop += tl_; tot_b += tb
        res[str(shape)] = {"n": n, "equal": eq, "max_abs_diff": maxd,
                           "falsifier_equal": bool(torch.equal(ref[0], fals)),
                           "loop_s": tl_, "batched_s": tb}
        print(shape, res[str(shape)], flush=True)
    res["total"] = {"loop_s": tot_loop, "batched_s": tot_b}
    print("total", res["total"])
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "round2_207", "muon_batch.json")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
