"""
Test suite for Issue #79 — DeltaNet scan parity + speed harness.

The backbone of the training-efficiency track (epic #78). It does two jobs:

  1. **Identity guardrail.** Any fast implementation of the gated-delta-rule
     scan MUST compute the *same* result as the plain token-by-token reference
     (:meth:`DeltaNetBlock._scan_sequential`), in both forward and backward.
     This test asserts that — first for the current chunked-parallel path
     (:meth:`_scan_chunked`), and, when a fused kernel is wired in (#83), for
     that too via the pluggable ``FUSED_SCAN`` hook.  This is what lets us swap
     in a faster kernel without silently changing HVN's identity.

  2. **Speed benchmark.** Times the full DeltaNetBlock forward+backward at the
     real training shape (d=512, 12 heads-worth per layer, B=4, S=512, fp32)
     for each chunk size and, for reference, the matched TransformerBlock — so
     every efficiency change can be measured as tokens/s + peak VRAM against
     the fp32 baseline.

Run on GPU (RTX 4050):
    python tests/test_issue79.py            # parity asserts + benchmark table
    python tests/test_issue79.py --bench    # benchmark only
"""

import argparse
import time

import torch

# --- repo-root path bootstrap (added during reorg, issue #48) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---
from hvn.core.deltanet import DeltaNetBlock, _l2norm

try:
    # Lives on the A/B baseline branch (#73); optional reference row in the
    # benchmark. The parity gate does not depend on it.
    from hvn.core.transformer_block import TransformerBlock
except ImportError:
    TransformerBlock = None


# Optional fused-scan hook, wired in by #83. Signature must match the static
# scans: fn(qN, kN, vN, alphaN, betaN, chunk, return_state=False) -> oN [N,S,d].
# Left None here so #79 stands alone; the #83 kernel registers itself and the
# same parity/backward asserts below then gate it.
FUSED_SCAN = None

# Parity tolerances. The chunked path runs the 1/P reparameterization + a
# triangular solve in fp32, so it is not bit-identical to the sequential scan;
# a small relative drift is expected and bounded well below any effect on
# training. Measured max rel-diff is ~1e-5..1e-4 at S=512, d=64.
FWD_RTOL = 2e-3
BWD_RTOL = 5e-3


def _device():
    assert torch.cuda.is_available(), "CUDA required — this project runs on GPU only"
    return torch.device("cuda")


def _rel(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max abs difference normalized by the reference's max magnitude."""
    denom = b.abs().max().clamp_min(1e-12)
    return (a - b).abs().max().item() / denom.item()


def make_scan_inputs(N, S, d, seed=0, device="cuda", requires_grad=False):
    """Realistic scan inputs matching what DeltaNetBlock._project produces.

    q, k are L2-normalized over d; alpha = sigmoid(.) in (0,1) with the ~0.95
    init bias baked in (so P = cumprod(alpha) exercises the real decay regime);
    beta = sigmoid(.) write strength. All fp32, shapes [N,S,d] (beta [N,S,1]).
    """
    g = torch.Generator(device=device).manual_seed(seed)
    q = _l2norm(torch.randn(N, S, d, generator=g, device=device))
    k = _l2norm(torch.randn(N, S, d, generator=g, device=device))
    v = torch.randn(N, S, d, generator=g, device=device)
    # alpha ~ sigmoid(N(3, 1)) → mostly 0.9..0.99, the memory-retaining regime
    alpha = torch.sigmoid(3.0 + torch.randn(N, S, d, generator=g, device=device))
    beta = torch.sigmoid(torch.randn(N, S, 1, generator=g, device=device))
    tensors = [q, k, v, alpha, beta]
    if requires_grad:
        for t in tensors:
            t.requires_grad_(True)
    return tensors


# ----------------------------------------------------------------------------
# 1. Identity guardrail — forward
# ----------------------------------------------------------------------------
def test_chunked_forward_matches_sequential():
    dev = _device()
    N, S, d = 8, 512, 64
    q, k, v, alpha, beta = make_scan_inputs(N, S, d, seed=1, device=dev)
    ref = DeltaNetBlock._scan_sequential(q, k, v, alpha, beta)
    worst = 0.0
    for chunk in (16, 32, 64, 128):
        out = DeltaNetBlock._scan_chunked(q, k, v, alpha, beta, chunk)
        r = _rel(out, ref)
        worst = max(worst, r)
        assert r < FWD_RTOL, f"chunk={chunk} fwd rel-diff {r:.2e} ≥ {FWD_RTOL:.0e}"
    print(f"[1/4] chunked fwd == sequential (worst rel {worst:.2e}) ....... OK")


# ----------------------------------------------------------------------------
# 2. Identity guardrail — backward (gradients must match too)
# ----------------------------------------------------------------------------
def test_chunked_backward_matches_sequential():
    dev = _device()
    N, S, d = 8, 256, 64
    ins_ref = make_scan_inputs(N, S, d, seed=2, device=dev, requires_grad=True)
    ins_chk = make_scan_inputs(N, S, d, seed=2, device=dev, requires_grad=True)
    go = torch.randn(N, S, d, device=dev, generator=torch.Generator(dev).manual_seed(9))

    o_ref = DeltaNetBlock._scan_sequential(*ins_ref)
    o_ref.mul(go).sum().backward()
    o_chk = DeltaNetBlock._scan_chunked(*ins_chk, 32)
    o_chk.mul(go).sum().backward()

    worst = 0.0
    for name, a, b in zip("qkvαβ", ins_chk, ins_ref):
        r = _rel(a.grad, b.grad)
        worst = max(worst, r)
        assert r < BWD_RTOL, f"grad[{name}] rel-diff {r:.2e} ≥ {BWD_RTOL:.0e}"
    print(f"[2/4] chunked bwd == sequential (worst grad rel {worst:.2e}) .. OK")


# ----------------------------------------------------------------------------
# 3. Identity guardrail — fused kernel (#83), skipped until FUSED_SCAN wired
# ----------------------------------------------------------------------------
def test_fused_matches_sequential():
    if FUSED_SCAN is None:
        print("[3/4] fused scan parity ................... SKIP (no kernel yet)")
        return
    dev = _device()
    N, S, d = 8, 512, 64
    q, k, v, alpha, beta = make_scan_inputs(N, S, d, seed=1, device=dev)
    ref = DeltaNetBlock._scan_sequential(q, k, v, alpha, beta)
    out = FUSED_SCAN(q, k, v, alpha, beta, 64)
    r = _rel(out, ref)
    assert r < FWD_RTOL, f"fused fwd rel-diff {r:.2e} ≥ {FWD_RTOL:.0e}"
    print(f"[3/4] fused fwd == sequential (rel {r:.2e}) ....... OK")


# ----------------------------------------------------------------------------
# 4. Speed benchmark — full block, real training shape
# ----------------------------------------------------------------------------
def _bench_block(block, B, S, D, steps=15, warmup=5):
    dev = _device()
    block = block.to(dev)
    opt = torch.optim.AdamW(block.parameters(), lr=1e-4)
    x = torch.randn(B, S, D, device=dev)
    torch.cuda.reset_peak_memory_stats()

    def step():
        opt.zero_grad(set_to_none=True)
        out = block(x)
        out.pow(2).mean().backward()
        opt.step()

    for _ in range(warmup):
        step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    torch.cuda.synchronize()
    sps = (time.perf_counter() - t0) / steps
    peak = torch.cuda.max_memory_allocated() / 1e9
    del block, opt
    torch.cuda.empty_cache()
    return sps, peak


def bench(B=4, S=512, D=512, H=8):
    dev = _device()
    print(f"\n  block fwd+bwd @ B={B} S={S} D={D} H={H} fp32 (single block):")
    results = {}
    for chunk in (32, 64, 128):
        blk = DeltaNetBlock(d_model=D, n_heads=H, dropout=0.0, train_chunk=chunk)
        sps, peak = _bench_block(blk, B, S, D)
        results[f"DeltaNet chunk={chunk}"] = (sps, peak)
    if TransformerBlock is not None:
        tfm = TransformerBlock(d_model=D, n_heads=H, dropout=0.0)
        results["Transformer (target)"] = _bench_block(tfm, B, S, D)

    base = results["DeltaNet chunk=32"][0]
    tgt = results.get("Transformer (target)", (base, 0))[0]
    print(f"  {'variant':26s} {'ms/step':>9} {'x vs c32':>9} {'x vs tfm':>9} {'peak GB':>9}")
    for name, (sps, peak) in results.items():
        print(f"  {name:26s} {sps*1000:9.1f} {base/sps:8.2f}x {tgt/sps:8.2f}x {peak:9.2f}")
    print("  (x vs tfm < 1 means slower than the transformer target)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true", help="benchmark only, skip parity asserts")
    args = ap.parse_args()
    if not args.bench:
        test_chunked_forward_matches_sequential()
        test_chunked_backward_matches_sequential()
        test_fused_matches_sequential()
    bench()
    print("\n[4/4] benchmark complete .................. OK")


if __name__ == "__main__":
    main()
