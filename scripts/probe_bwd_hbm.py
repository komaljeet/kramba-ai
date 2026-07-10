"""
Design B go/no-go probe: is the current sequential backward HBM-bound?

Design B (fused blockwise-recompute backward) ONLY wins if the current 2-kernel
backward is HBM-bound: it replaces the 268 MB [N,S,D,D] states write+read with a
67 MB block-boundary checkpoint (B=4), same compute. So its entire upside is HBM
traffic reduction. If the backward is latency/occupancy-bound (32 programs, low
occupancy), cutting HBM does nothing -> Design B is doomed like Design A, and we
stop before writing a kernel.

Test: measure the backward's achieved HBM bandwidth at the real shape and compare
to the 4050's peak. The backward moves (fp32):
  - states  : written by _gd_fwd_states_kernel, read by _gd_bwd_kernel
              [N,S,D,D] x4B = 32*512*64*64*4 = 268.4 MB, x2 (write+read) = 536.9 MB
  - inputs  : q,k,v,alpha [N,S,D] x4B x4 = 33.6 MB read x2 (fwd-states + bwd) ~67 MB
              beta [N,S] x4B = 65 KB (negligible)
  - do      : [N,S,D] x4B = 4.19 MB read
  - grads   : dq,dk,dv,dalpha [N,S,D] x4 + dbeta [N,S] = ~16.8 MB write
So dominant traffic = the states roundtrip (~537 MB) + inputs/grads (~88 MB) ~ 625 MB.

If achieved_bw / peak_bw is high (>= ~55-60%), the backward IS HBM-bound ->
Design B (4x less states HBM) has a real shot. If it's low (<= ~25%), the backward
is latency/occupancy/compute-bound -> Design B won't help -> stop.

    python scripts/probe_bwd_hbm.py
"""
from __future__ import annotations

import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hvn.core.deltanet_triton import gated_delta_scan

N, S, D = 32, 512, 64
DEV = torch.device("cuda")


def _inputs(decay, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    v = torch.randn(N, S, D, device=DEV, generator=g)
    if decay == "trained":
        a = torch.sigmoid(3.0 + torch.randn(N, S, D, device=DEV, generator=g))
    else:
        a = torch.empty(N, S, D, device=DEV).uniform_(0.1, 0.2, generator=g)
    b = torch.sigmoid(torch.randn(N, S, 1, device=DEV, generator=g))
    for t in (q, k, v, a, b):
        t.requires_grad_(True)
    return q, k, v, a, b


def _bench_bwd(args, warmup=5, iters=30):
    """Time the BACKWARD only (fwd done once, graph retained)."""
    q, k, v, a, b = args
    out = gated_delta_scan(q, k, v, a, b)
    do = torch.ones_like(out)
    # warmup backward (retain_graph so we can repeat without re-running fwd)
    for _ in range(warmup):
        out.backward(do, retain_graph=True)
        for t in (q, k, v, a, b):
            t.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        out.backward(do, retain_graph=True)
        for t in (q, k, v, a, b):
            t.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _bench_fwbwd(args, warmup=5, iters=30):
    q, k, v, a, b = args
    do = torch.ones(N, S, D, device=DEV)

    def step():
        out = gated_delta_scan(q, k, v, a, b)
        out.backward(do, retain_graph=False)
        for t in (q, k, v, a, b):
            t.grad = None

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def _peak_bw():
    """Measure peak achievable HBM bandwidth on this GPU (large D2D copy)."""
    n = 256 * 1024 * 1024  # 256 MB
    src = torch.empty(n, dtype=torch.float32, device=DEV)
    dst = torch.empty_like(src)
    for _ in range(5):
        dst.copy_(src)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(50):
        dst.copy_(src)
    torch.cuda.synchronize()
    secs = (time.perf_counter() - t0) / 50
    # read src + write dst = 2 * n * 4 bytes
    return (2 * n * 4) / secs / 1e9  # GB/s


def main():
    assert torch.cuda.is_available()
    peak = _peak_bw()
    bytes_states = N * S * D * D * 4 * 2          # write + read
    bytes_inputs = N * S * D * 4 * 4 * 2          # q,k,v,alpha read x2 (fwd-states+bwd)
    bytes_io = (N * S * D * 4 * 4 + N * S * 4)    # do read + grads write
    bytes_total = bytes_states + bytes_inputs + bytes_io
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"  peak HBM BW (D2D copy): {peak:6.1f} GB/s")
    print(f"  backward traffic: states {bytes_states/1e6:6.1f} MB  "
          f"inputs {bytes_inputs/1e6:6.1f} MB  io {bytes_io/1e6:6.1f} MB  "
          f"total {bytes_total/1e6:6.1f} MB\n")
    print(f"  {'decay':7s} {'fwd+bwd ms':>11s} {'bwd-only ms':>12s} "
          f"{'achieved GB/s':>14s} {'%peak':>7s}  verdict")
    go = True
    for decay in ("trained", "hard"):
        args = _inputs(decay)
        fwbwd = _bench_fwbwd(args)
        bwd = _bench_bwd(args)
        ach = (bytes_total / 1e9) / (bwd / 1000)
        pct = 100 * ach / peak
        if pct >= 55:
            verdict = "HBM-bound -> B has a shot"
        elif pct >= 25:
            verdict = "MIXED -> B maybe"
        else:
            verdict = "latency-bound -> B doomed"
            go = False
        print(f"  {decay:7s} {fwbwd:11.3f} {bwd:12.3f} {ach:14.1f} {pct:6.1f}%  {verdict}")
    print()
    print("GO / NO-GO for Design B:", "GO (backward is HBM-bound)" if go
          else "NO-GO (backward is latency/occupancy-bound; HBM cut won't help)")


if __name__ == "__main__":
    main()