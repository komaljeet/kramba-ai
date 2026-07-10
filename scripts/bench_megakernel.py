"""
#96 kill-switch gate: fused megakernel FWD-ONLY vs the sequential Triton kernel,
at the real training shape (B4 x H8 = N32, S512, d64). Forward-only is the EASY
case for the chunk-parallel path (the sequential kernel's backward recomputes a
[N,S,D,D] state tensor + a second pass — that's where sequential is heaviest).
So if the fused megakernel can't beat sequential forward-only, it won't beat it
fwd+bwd either: STOP, don't write the backward (the plan's kill-switch).

GATE: seq_ms / mega_ms > 1.0 at d=64 in BOTH decay regimes.

    python scripts/bench_megakernel.py
"""
from __future__ import annotations

import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hvn.core.deltanet_triton import gated_delta_scan, megakernel_forward

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
    return q, k, v, a, b


def _bench(fwd, args, warmup=5, iters=20):
    def step():
        fwd(*args)
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        step()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1000


def main():
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} S={S} d={D}  FWD-ONLY (no_grad)\n")
    all_win = True
    for decay in ("trained", "hard"):
        args = _inputs(decay)
        with torch.no_grad():
            seq_ms = _bench(gated_delta_scan, args)
        print(f"  [{decay:7s}] sequential Triton fwd : {seq_ms:7.3f} ms")
        for c in (16, 32, 64):
            with torch.no_grad():
                mega_ms = _bench(lambda *a: megakernel_forward(*a, chunk=c), args)
            win = seq_ms / mega_ms
            tag = "WIN" if win > 1.0 else "lose"
            all_win &= win > 1.0
            print(f"  [{decay:7s}] mega C={c:<3} fwd      : {mega_ms:7.3f} ms   ({win:4.2f}x vs seq)  {tag}")
        print()
    print("=" * 60)
    print("GATE (seq_ms/mega_ms > 1.0 in BOTH regimes):",
          "PASS — proceed to backward" if all_win else "FAIL — kill-switch, do NOT write backward")


if __name__ == "__main__":
    main()