"""#208 candidate 3: time the residual-gradient add folded into the RMSNorm backward.

Author: Rambarun Komaljeet

At the 1B shape (x fp32 [4096, 2048], weight fp32 [2048]) compares, per block backward:
  default  ATen ``_fused_rms_norm_backward`` (via autograd of F.rms_norm) + the engine's
           ``dx += g_res`` add
  fused    hvn/core/fused_norm_residual.py: one kernel writes dx = norm_bwd + g_res
for the weight-trainable case (active layer) and the frozen case (no dw).
Projection: per 1B k4 step, 8 micro-batches x 14.14 DeltaNet block backwards (mean over
the 7 block positions; frozen-before layers have no backward), of which 4 per micro-batch
(the active block) need dw.

Falsifier: the fused run's dx is compared with the default's (must agree to fp32
rounding), so a fast-but-wrong kernel cannot pass as a saving.
"""
import json, os, sys
import torch
import torch.nn.functional as F

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import hvn.core.fused_norm_residual as FNR


def ev(fn, n=50):
    fn(); torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(n):
        fn()
    b.record(); torch.cuda.synchronize()
    return a.elapsed_time(b) / n


def main(out):
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(0)
    x = torch.randn(1, 4096, 2048, device=dev, generator=g)
    w = 1 + 0.1 * torch.randn(2048, device=dev, generator=g)
    gy = torch.randn(1, 4096, 2048, device=dev, generator=g)
    gr = torch.randn(1, 4096, 2048, device=dev, generator=g)
    res = {}
    for need_w in (True, False):
        xa = x.clone().requires_grad_()
        wa = w.clone().requires_grad_(need_w)
        ya = F.rms_norm(xa, (2048,), wa, 1e-6)

        def default():
            # what autograd runs: norm backward, then dx_norm + g_res
            dx, *dw = torch.autograd.grad(ya, [xa] + ([wa] if need_w else []), gy, retain_graph=True)
            return dx.add_(gr)

        xf = x.clone().requires_grad_()
        wf = w.clone().requires_grad_(need_w)
        yf, link = FNR.norm_residual(xf, wf, 1e-6)

        def fused():
            return torch.autograd.grad([yf, link], [xf] + ([wf] if need_w else []), [gy, gr],
                                       retain_graph=True)[0]
        d0, d1 = default(), fused()
        rel = ((d0 - d1).norm() / d0.norm()).item()
        tag = "active" if need_w else "frozen"
        res[tag] = {"default_ms": ev(default), "fused_ms": ev(fused), "dx_rel_diff": rel}
        res[tag]["saving_ms"] = res[tag]["default_ms"] - res[tag]["fused_ms"]
        print(tag, res[tag], flush=True)
    per_mb = 4 * res["active"]["saving_ms"] + (14.14 - 4) * res["frozen"]["saving_ms"]
    res["projected_saving_s_per_step"] = 8 * per_mb / 1e3
    print("projected saving s/step", res["projected_saving_s_per_step"], flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
