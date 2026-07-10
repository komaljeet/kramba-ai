"""
Benchmark the chunk-parallel scan FORWARD vs the sequential Triton scan, at the
real training shape (B4 x H8 = N32, S512, d64), fwd-only / no-grad. This is the
#95 proof: does the chunk-parallel path beat the sequential kernel that is ~60%
of the HVN step at --fast?

    python scripts/bench_chunked_fwd.py
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


def _inputs(decay="trained", seed=0):
    g = torch.Generator(device=DEV).manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(N, S, D, device=DEV, generator=g), dim=-1)
    v = torch.randn(N, S, D, device=DEV, generator=g)
    if decay == "trained":      # ~0.95 (real init)
        a = torch.sigmoid(3.0 + torch.randn(N, S, D, device=DEV, generator=g))
    else:                       # hard decay stress
        a = torch.empty(N, S, D, device=DEV).uniform_(0.1, 0.2, generator=g)
    b = torch.sigmoid(torch.randn(N, S, 1, device=DEV, generator=g))
    return q, k, v, a, b


def _bench(fn, args, warmup=5, iters=30):
    for _ in range(warmup):
        fn(*args)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn(*args)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000   # ms


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} S={S} d={D}  fwd-only\n")
    for decay in ("trained", "hard"):
        q, k, v, a, b = _inputs(decay)
        with torch.no_grad():
            seq_ms = _bench(gated_delta_scan, (q, k, v, a, b))
            print(f"  [{decay:7s}] sequential Triton : {seq_ms:7.2f} ms")
            for c in (64, 128, 256):
                ch_ms = _bench(chunked_scan_forward, (q, k, v, a, b, c))
                print(f"  [{decay:7s}] chunked fwd C={c:<3}: {ch_ms:7.2f} ms   ({seq_ms/ch_ms:4.2f}x vs seq)")
            print()

    # correctness sanity at the bench shape (no_grad)
    q, k, v, a, b = _inputs("trained")
    from hvn.core.deltanet import DeltaNetBlock
    with torch.no_grad():
        seq = DeltaNetBlock._scan_sequential(q, k, v, a, b)
        ch = chunked_scan_forward(q, k, v, a, b, 128)
    rel = ((ch - seq).norm() / seq.norm()).item()
    print(f"correctness (trained, C=128) vs sequential: rel {rel:.2e}  {'OK' if rel < 1e-4 else 'FAIL'}")


if __name__ == "__main__":
    main()