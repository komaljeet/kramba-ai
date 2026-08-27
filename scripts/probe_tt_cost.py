"""What TT actually buys and costs, at the sizes this repo trains (Issue #161).

Reports two things the decision depends on, and deliberately does NOT report a
quality number -- that needs a real A/B, not a probe:

1. **Parameter compression** of a DeltaNet block at real configs.
2. **Measured forward+backward wall time** against the dense block.

This probe is why the layer has two contraction orders. The FLOP arithmetic said
the batch-streaming ("direct") order costs 2.1x dense at d=896; measured, it cost
2.27x AND was relatively worse at d=896 than at d=384, which the FLOP ratio says
should not happen. That contradiction is what exposed the direct order as
bandwidth-bound on an oversized intermediate. The default "fused" order (fold the
cores first, then one GEMM) measures at 1.03x. A number derived on paper is not a
measurement of this code -- the repo's own rule, and it paid here.

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
    print("   Both contraction orders, because the choice between them was")
    print("   worth 2.2x and the FLOP arithmetic pointed the wrong way.")
    print("=" * 78)
    print(f"{'d_model':>8} {'rank':>5} {'dense s/it':>12} "
          f"{'fused':>9} {'vs dense':>9} {'direct':>9} {'vs dense':>9}")
    B, S = 4, 512
    for d, h, label in CONFIGS:
        if d >= 2048 and dev == "cuda":
            print(f"{d:>8}  skipped (VRAM headroom on a 6 GB card)")
            continue
        x = torch.randn(B, S, d, device=dev, requires_grad=True)
        dense = DeltaNetBlock(d_model=d, n_heads=h).to(dev)
        t_dense = time_block(dense, x)
        del dense
        if dev == "cuda":
            torch.cuda.empty_cache()
        for r in RANKS:
            row = {}
            for order in ("fused", "direct"):
                try:
                    tt = DeltaNetBlock(d_model=d, n_heads=h, tt_targets=TARGETS,
                                       tt_rank=r, tt_contract=order).to(dev)
                except ValueError:
                    row[order] = None
                    continue
                row[order] = time_block(tt, x)
                del tt
                if dev == "cuda":
                    torch.cuda.empty_cache()
            if row.get("fused") is None:
                continue
            tf, td = row["fused"], row["direct"]
            print(f"{d:>8} {r:>5} {t_dense:>12.4f} "
                  f"{tf:>9.4f} {tf / t_dense:>8.2f}x "
                  f"{td:>9.4f} {td / t_dense:>8.2f}x")
        del x

    print("\nNote: this probe measures cost and size only. Whether TT-compressed")
    print("weights LEARN as well is a separate A/B on held-out loss at matched")
    print("tokens, with a full-length twin arm for the noise floor (#122).")


if __name__ == "__main__":
    main()
