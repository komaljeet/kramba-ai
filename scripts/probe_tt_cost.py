"""What TT actually buys and costs, at the sizes this repo trains (Issue #161).

Reports two things the decision depends on, and deliberately does NOT report a
quality number -- that needs a real A/B, not a probe:

1. **Parameter compression** of a DeltaNet block at real configs.
2. **Measured forward+backward wall time** against the dense block.

The FLOP arithmetic in ``hvn/edge/tt_linear.py`` predicts TT is *slower* below
d=4096. This measures whether that prediction holds in the real block, because a
number derived on paper is not a measurement of this code (the repo's own rule).

Run with nothing else on the GPU -- a contended timing is an artefact, not a
result (see docs/state.md, the code-vocab A/B).

Usage: python scripts/probe_tt_cost.py
"""

import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hvn.core.deltanet import DeltaNetBlock  # noqa: E402
from hvn.edge.tt_linear import balanced_factors, full_tt_rank  # noqa: E402

CONFIGS = [
    # (d_model, n_heads, label)
    (384, 6, "ab_looped_small / Phase 0"),
    (896, 14, "#136 recommended pretrain"),
    (2048, 32, "1.07B ceiling config"),
]
RANKS = [16, 32, 64]
TARGETS = ("qkv", "out", "ffn")


def n_params(m: torch.nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def time_block(block: torch.nn.Module, x: torch.Tensor, iters: int = 20) -> float:
    """Median seconds per forward+backward."""
    times = []
    for i in range(iters + 5):
        if x.grad is not None:
            x.grad = None
        block.zero_grad(set_to_none=True)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        block(x).pow(2).sum().backward()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i >= 5:  # discard warmup
            times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


def main() -> None:
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if dev == "cpu":
        print("WARNING: no CUDA. Timings below are CPU and not comparable to "
              "anything in docs/. Parameter counts are still valid.\n")
    print(f"device: {dev}\n")

    print("=" * 78)
    print("1. PARAMETER COMPRESSION of a DeltaNet block (targets: qkv+out+ffn)")
    print("=" * 78)
    print(f"{'d_model':>8} {'rank':>5} {'dense params':>14} {'TT params':>13} "
          f"{'smaller':>9}  {'max useful rank':>15}")
    for d, h, label in CONFIGS:
        dense = DeltaNetBlock(d_model=d, n_heads=h)
        d_n = n_params(dense)
        rf, cf = balanced_factors(d), balanced_factors(d)
        r_max = full_tt_rank(rf, cf)
        del dense
        for r in RANKS:
            try:
                tt = DeltaNetBlock(d_model=d, n_heads=h, tt_targets=TARGETS, tt_rank=r)
            except ValueError as e:
                print(f"{d:>8} {r:>5} {'--':>14} {'REFUSED':>13} {'':>9}  {str(e)[:40]}")
                continue
            t_n = n_params(tt)
            print(f"{d:>8} {r:>5} {d_n:>14,} {t_n:>13,} {d_n / t_n:>8.2f}x "
                  f"{r_max:>15}")
            del tt
        print(f"{'':>8} ({label})")

    print("\n" + "=" * 78)
    print("2. MEASURED forward+backward, dense vs TT (batch 4, seq 512)")
    print("=" * 78)
    print(f"{'d_model':>8} {'rank':>5} {'dense s/it':>12} {'TT s/it':>11} "
          f"{'TT vs dense':>13}")
    B, S = 4, 512
    for d, h, label in CONFIGS:
        if d >= 2048 and dev == "cuda":
            print(f"{d:>8} {'--':>5} {'skipped (VRAM headroom on a 6 GB card)':>40}")
            continue
        x = torch.randn(B, S, d, device=dev, requires_grad=True)
        dense = DeltaNetBlock(d_model=d, n_heads=h).to(dev)
        t_dense = time_block(dense, x)
        del dense
        if dev == "cuda":
            torch.cuda.empty_cache()
        for r in RANKS:
            try:
                tt = DeltaNetBlock(d_model=d, n_heads=h, tt_targets=TARGETS,
                                   tt_rank=r).to(dev)
            except ValueError:
                continue
            t_tt = time_block(tt, x)
            ratio = t_tt / t_dense
            verdict = "SLOWER" if ratio > 1.05 else ("faster" if ratio < 0.95 else "~same")
            print(f"{d:>8} {r:>5} {t_dense:>12.4f} {t_tt:>11.4f} "
                  f"{ratio:>11.2f}x  {verdict}")
            del tt
            if dev == "cuda":
                torch.cuda.empty_cache()
        del x

    print("\nNote: this probe measures cost and size only. Whether TT-compressed")
    print("weights LEARN as well is a separate A/B on held-out loss at matched")
    print("tokens, with a full-length twin arm for the noise floor (#122).")


if __name__ == "__main__":
    main()
