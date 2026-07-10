"""
#96 Design B speed gate: fused blockwise-recompute backward vs the current
sequential 2-kernel backward, at the real training shape (B4 x H8 = N32, S512,
d64). Design B keeps the optimal sequential fwd; only the backward changes
(blockwise SRAM recompute + 4x-less states HBM). The backward was measured
HBM-bound at ~80% of peak (scripts/probe_bwd_hbm.py), so the 4x states-HBM cut
should translate to a backward speedup.

GATE: seq_ms / mega_ms > 1.0 at d=64 in BOTH decay regimes (fwd+bwd).

    python scripts/bench_megakernel_bwd.py
"""
from __future__ import annotations

import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from hvn.core.deltanet_triton import gated_delta_scan, megakernel_scan

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
    do = torch.ones(N, S, D, device=DEV)

    def step():
        q, k, v, a, b = (t.detach().clone().requires_grad_(True) for t in args)
        out = fwd(q, k, v, a, b)
        out.backward(do, retain_graph=False)

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
    print(f"GPU: {torch.cuda.get_device_name(0)}  N={N} S={S} d={D}  FWD+BWD\n")
    best = {}
    for decay in ("trained", "hard"):
        args = _inputs(decay)
        seq_ms = _bench(gated_delta_scan, args)
        print(f"  [{decay:7s}] sequential  fwd+bwd : {seq_ms:7.3f} ms")
        bw = 0.0
        for bb in (2, 4, 8):
            for buf in (True, False):
                tag_b = "buf" if buf else "Ob2"
                mega_ms = _bench(lambda *a: megakernel_scan(*a, block=bb, buffered=buf), args)
                win = seq_ms / mega_ms
                mark = "WIN" if win > 1.0 else "lose"
                bw = max(bw, win)
                print(f"  [{decay:7s}] design-B B={bb:<3} {tag_b:4s} : {mega_ms:7.3f} ms   ({win:4.2f}x vs seq)  {mark}")
        best[decay] = bw
        print()
    print("=" * 60)
    all_win = all(w > 1.0 for w in best.values())
    print(f"best win: trained {best['trained']:.2f}x   hard {best['hard']:.2f}x")
    print("GATE (best-B seq_ms/mega_ms > 1.0 in BOTH regimes):",
          "PASS — Design B wins; wire it in" if all_win else "FAIL — Design B does not beat sequential bwd")


if __name__ == "__main__":
    main()