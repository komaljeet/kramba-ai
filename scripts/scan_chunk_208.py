"""#208 candidate 2: would a chunked scan (chunk 16 / 32) beat the current kernels at 1B?

Author: Rambarun Komaljeet

Measure first, build only if the saving clears 0.05 s per step. At the real per-layer
shape (batch 1, 64 heads, d_head 32, seq 4096 -> N = 64 sequences) this times:

  current   the production autograd path (`megakernel_scan_bshd`), v1 and t kernels:
              F  forward under no_grad (a frozen-before layer)
              C  checkpointed forward + backward (recompute + adjoint), as every layer
                 that needs a gradient runs it (torch.utils.checkpoint, non-reentrant)
  chunked   the repo's chunk-parallel forward megakernel (#96 Design A,
            `_mk_fwd_kernel`) at CHUNK 16 and 32, BV 8/16/32. Two variants:
              real     Neumann solve with CHUNK-1 terms (the exact chunked forward)
              nosolve  0 Neumann terms: NOT a correct scan, a LOWER BOUND on any
                       chunked forward of this structure (Grams + carries + output
                       only; a faster triangular solve cannot go below it)
            Inputs are handed over as contiguous fp32 [N,S,D], prepared outside the
            timed region -- favourable to the chunked kernel.

Projection to the 1B k4 step (per optimizer step, 8 micro-batches, 22 DeltaNet layers,
mean over the 7 block positions): 14.14 layers per micro-batch run C, 7.86 run F only.
A chunked scan needs at least 2 forwards (first pass + recompute) plus a backward
that is no cheaper than a forward, so C_chunk >= 3 * F_chunk. The bound used:
  saving <= 8 * (14.14 * (C_cur - 3 F_chunk_lb) + 7.86 * (F_cur - F_chunk_lb))
with F_chunk_lb the nosolve time. If that is under 50 ms the candidate is dropped
without building a backward.

Falsifier (would the bench see a real chunked win?): the real chunked forward's output
is checked against an fp64 reference (it must be a correct scan, else its time means
nothing), and the nosolve variant must be WRONG (rel error >> 1e-6) -- proving the
solve really ran in the "real" variant and was really skipped in the bound.
"""
import json, os, sys
import torch
import torch.utils.checkpoint

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
import hvn.core.deltanet_triton as DT
from scan_study_206 import inputs, ev_time, S, D, dev

# Grad-chain / forward-only DeltaNet layers per micro-batch at each 1B k4 block position
# (attention at 6 and 18; layers before the active block are frozen, forward only).
GRAD_LAYERS = [22, 22, 18, 15, 11, 7, 4]
FWD_LAYERS = [0, 0, 4, 7, 11, 15, 18]
G_MEAN = sum(GRAD_LAYERS) / 7
F_MEAN = sum(FWD_LAYERS) / 7
MB = 8


def current(kernel, q, k, v, a, b, do):
    DT.SCAN_KERNEL[0] = kernel
    with torch.no_grad():
        f = ev_time(lambda: DT.megakernel_scan_bshd(q, k, v, a, b))
    qs = [t.detach().clone().requires_grad_(True) for t in (q, k, v, a, b)]

    def chain():
        o = torch.utils.checkpoint.checkpoint(DT.megakernel_scan_bshd, *qs, use_reentrant=False)
        o.backward(do)
    st0 = dict(DT.SCAN_T_STATS)
    c = ev_time(chain)
    st1 = dict(DT.SCAN_T_STATS)
    DT.SCAN_KERNEL[0] = "v1"
    return {"F_ms": f, "C_ms": c, "t_stats_delta": {kk: st1[kk] - st0[kk] for kk in st1}}


def chunk_fwd(qn, kn, vn, an, bn, chunk, bv, bd, nneum):
    N = qn.shape[0]
    o = torch.empty_like(qn)
    s_t = torch.tensor([S], dtype=torch.int32, device=dev)
    n_t = torch.tensor([nneum], dtype=torch.int32, device=dev)
    DT._mk_fwd_kernel[(N, D // bv)](qn, kn, vn, an, bn, o, s_t, n_t,
                                   D=D, CHUNK=chunk, BV=bv, BD=bd, num_warps=4)
    return o


def ref64_o(qn, kn, vn, an, bn):
    q, k, v, a, b = (t.double() for t in (qn, kn, vn, an, bn))
    N = q.shape[0]
    st = torch.zeros(N, D, D, dtype=torch.float64, device=dev)   # [dv, dk]
    outs = []
    for t in range(S):
        st = st * a[:, t, None, :]
        u = v[:, t] - (st * k[:, t, None, :]).sum(-1)
        st = st + b[:, t, None, None] * (u[:, :, None] * k[:, t, None, :])
        outs.append((st * q[:, t, None, :]).sum(-1))
    return torch.stack(outs, 1)


def rel(x, r):
    return ((x.double() - r).norm() / r.norm()).item()


def main(out_path):
    res = {"shape": {"Bt": 1, "H": 64, "S": S, "D": D}, "grad_layers_mean": G_MEAN,
           "fwd_layers_mean": F_MEAN}
    q, k, v, a, b = inputs(1, 64)
    g = torch.Generator(device=dev).manual_seed(5)
    do = torch.randn(64, S, D, device=dev, generator=g)
    for kern in ("v1", "t"):
        res[kern] = current(kern, q, k, v, a, b, do)
        print(kern, res[kern], flush=True)
    toN = lambda t: t.permute(0, 2, 1, 3).reshape(64, S, -1).float().contiguous()
    qn, kn, vn, an = (toN(t) for t in (q, k, v, a))
    bn = b.reshape(1, S, 64).permute(0, 2, 1).reshape(64, S).float().contiguous()
    rows = []
    for chunk in (16, 32):
        for bv in (8, 16, 32):
            for variant, nn in (("real", chunk - 1), ("nosolve", 0)):
                row = {"chunk": chunk, "bv": bv, "variant": variant}
                try:
                    chunk_fwd(qn, kn, vn, an, bn, chunk, bv, 8, nn); torch.cuda.synchronize()
                    row["F_ms"] = ev_time(lambda: chunk_fwd(qn, kn, vn, an, bn, chunk, bv, 8, nn))
                except Exception as e:
                    row["error"] = repr(e)[:300]
                rows.append(row); print(row, flush=True)
    res["chunked"] = rows
    # accuracy at N = 8 (fp64 reference is a python loop; 8 sequences keep it short)
    r = ref64_o(qn[:8], kn[:8], vn[:8], an[:8], bn[:8])
    acc = {}
    for chunk in (16, 32):
        for variant, nn in (("real", chunk - 1), ("nosolve", 0)):
            o = chunk_fwd(qn[:8].contiguous(), kn[:8].contiguous(), vn[:8].contiguous(),
                          an[:8].contiguous(), bn[:8].contiguous(), chunk, 16, 8, nn)
            acc[f"c{chunk}_{variant}"] = rel(o, r)
    for kern in ("v1", "t"):
        DT.SCAN_KERNEL[0] = kern
        with torch.no_grad():
            o = DT.megakernel_scan_bshd(q[:, :, :8].float().contiguous(), k[:, :, :8].float().contiguous(),
                                        v[:, :, :8].float().contiguous(), a[:, :, :8].contiguous(),
                                        b[:, :, :8].contiguous())
        acc[kern] = rel(o, r)
    DT.SCAN_KERNEL[0] = "v1"
    res["fwd_rel_err_vs_fp64_N8"] = acc
    print("accuracy", acc, flush=True)
    # projection
    ok = [rw for rw in rows if "F_ms" in rw]
    lb = min(rw["F_ms"] for rw in ok if rw["variant"] == "nosolve")
    best_real = min(rw["F_ms"] for rw in ok if rw["variant"] == "real")
    proj = {}
    for kern in ("v1", "t"):
        F, C = res[kern]["F_ms"], res[kern]["C_ms"]
        proj[kern] = {
            "bound_saving_s_per_step_nosolve": MB * (G_MEAN * (C - 3 * lb) + F_MEAN * (F - lb)) / 1e3,
            "bound_saving_s_per_step_real": MB * (G_MEAN * (C - 3 * best_real) + F_MEAN * (F - best_real)) / 1e3,
            "scan_s_per_step_now": MB * (G_MEAN * C + F_MEAN * F) / 1e3}
    res["F_chunk_lb_ms"] = lb
    res["F_chunk_best_real_ms"] = best_real
    res["projection"] = proj
    print("projection", proj, flush=True)
    json.dump(res, open(out_path, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
