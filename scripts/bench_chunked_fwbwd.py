"""
Decisive #95 measurement: chunk-parallel scan FWD+BWD vs the sequential Triton
kernel, at the real training shape (B4 x H8 = N32, S512, d64). fwd-only had the
hybrid losing to the sequential kernel; but the sequential kernel's BACKWARD
recomputes a [N,S,D,D] state tensor + a second pass, so the training regime
(fwd+bwd) is where the chunked path might actually compete or win.

    python scripts/bench_chunked_fwbwd.py
"""
from __future__ import annotations

import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hvn.core.deltanet_triton import gated_delta_scan, chunked_scan_forward

N, S, D = 32, 512, 64
DEV = torch.device("cuda")


def _inputs(decay, seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1).requires_grad_(True)
    k = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1).requires_grad_(True)
    v = torch.randn(N, S, D, device=DEV, generator=g).requires_grad_(True)
    if decay == "trained":
        a = torch.sigmoid(3.0 + torch.randn(N, S, D, device=DEV, generator=g)).requires_grad_(True)
    else:
        a = torch.empty(N, S, D, device=DEV).uniform_(0.1, 0.2, generator=g).requires_grad_(True)
    b = torch.sigmoid(torch.randn(N, S, 1, device=DEV, generator=g)).requires_grad_(True)
    return q, k, v, a, b


def _bench(fwbwd, args, warmup=5, iters=20):
    grads = [t for t in args if isinstance(t, torch.Tensor) and t.requires_grad]

    def step():
        out = fwbwd(*args)
        out.sum().backward()
    for _ in range(warmup):
        step()
        for t in grads:
            t.grad = None
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
        for t in grads:
            t.grad = None
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} S={S} d={D}  fwd+bwd\n")
    for decay in ("trained", "hard"):
        args = _inputs(decay)
        seq_ms = _bench(gated_delta_scan, args)
        print(f"  [{decay:7s}] sequential Triton : {seq_ms:7.2f} ms")
        for c in (64, 128, 256):
            ch_ms = _bench(chunked_scan_forward, (*args, c))
            print(f"  [{decay:7s}] chunked C={c:<3} fwd+bwd: {ch_ms:7.2f} ms   ({seq_ms/ch_ms:4.2f}x vs seq)")
        print()

    # correctness at bench shape
    from hvn.core.deltanet import DeltaNetBlock
    q, k, v, a, b = _inputs("trained")
    with torch.no_grad():
        seq = DeltaNetBlock._scan_sequential(q, k, v, a, b)
        ch = chunked_scan_forward(q, k, v, a, b, 128)
    print(f"correctness (trained, C=128) fwd rel: {((ch-seq).norm()/seq.norm()).item():.2e}")


if __name__ == "__main__":
    main()