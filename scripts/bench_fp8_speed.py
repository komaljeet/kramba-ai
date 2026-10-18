"""FP8 speed signal for Phase 1 (exp/fp8-pretrain-feasibility). Not production.

Three numbers at the real 896-wide shape (B=1, S=4096 tokens per micro-batch):
  1. raw GEMM ceiling: _scaled_mm on pre-quantized fp8 vs bf16 matmul
  2. one Linear fwd+bwd incl. the (unfused, eager) dynamic quantization
  3. end-to-end train step of a small-depth 896-wide model, bf16 vs fp8_mixed
"""
from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import exp_fp8_feasibility as E  # noqa: E402  (imports pyarrow before CUDA)
import argparse  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402


def timeit(fn, iters=30, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters * 1e3


def main():
    torch.backends.cuda.matmul.allow_tf32 = True
    N = 4096
    out = {"gemm": [], "linear": [], "e2e": {}}
    for K, O in [(896, 896), (896, 1792), (1792, 896), (896, 50272)]:
        a = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(O, K, device="cuda", dtype=torch.bfloat16) * 0.05
        aq, as_ = E._q(a, E.E4M3, E.E4M3_MAX, True)
        wq, ws = E._q(w, E.E4M3, E.E4M3_MAX, True)
        tb = timeit(lambda: a @ w.t())
        tf = timeit(lambda: torch._scaled_mm(aq, wq.t(), scale_a=as_, scale_b=ws,
                                             out_dtype=torch.bfloat16))
        out["gemm"].append({"K": K, "O": O, "bf16_ms": tb, "fp8_ms": tf, "speedup": tb / tf})
        a32 = a.float().requires_grad_()
        w32 = w.float().requires_grad_()
        g = torch.randn(N, O, device="cuda", dtype=torch.bfloat16)

        def lin_bf16():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                y = F.linear(a32, w32)
            y.backward(g)

        def lin_fp8():
            y = E.fp8_linear(a32, w32, None)
            y.backward(g)
        tb2, tf2 = timeit(lin_bf16), timeit(lin_fp8)
        out["linear"].append({"K": K, "O": O, "bf16_ms": tb2, "fp8_ms": tf2, "speedup": tb2 / tf2})
        print(out["gemm"][-1], out["linear"][-1], flush=True)
        del a, w, aq, wq, a32, w32, g
        torch.cuda.empty_cache()

    # End-to-end: 896 wide, 4 layers (attention at 2), real ternary + fast stack.
    args = argparse.Namespace(attn_layers="2", layers=4, d_model=896, n_heads=14)
    for arm in ("bf16", "fp8_mixed", "bf16"):
        model, _ = E.build(args, 0)
        if arm == "fp8_mixed":
            E.install_fp8(model, exempt=("alpha_proj", "beta_proj", "decoder"), scaled=True)
        x = torch.randint(0, 50257, (1, 4096), device="cuda")

        def step():
            loss = E.loss_fn(model, x, x)
            loss.backward()
            model.zero_grad(set_to_none=True)
        torch.cuda.reset_peak_memory_stats()
        ms = timeit(step, iters=20, warm=5)
        key = arm if arm not in out["e2e"] else arm + "_repeat"
        out["e2e"][key] = {"ms_per_microbatch": ms,
                           "peak_mib": torch.cuda.max_memory_allocated() / 2**20}
        print(key, out["e2e"][key], flush=True)
        del model
        torch.cuda.empty_cache()
    path = os.path.join(ROOT, "results", "fp8_feasibility", "speed.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
