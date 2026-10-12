"""Fused read RMSNorm (hvn/core/fused_read_norm.py) vs DeltaNetBlock's reference.

Guards against vacuity:
  * B=2 and H>1 with a random per-head weight: a wrong (b, h) row mapping or a
    weight indexed by the wrong head gives a large error, not a small one.
  * S not a multiple of the 64-row tile and d not a power of two: masking bugs
    show up as wrong edge rows.
  * `test_dw_reduction_is_exercised` checks the weight gradient is far from a
    plausible-but-wrong one (missing the rms factor), so the dw bound means
    something.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

EPS = 1e-6
CASES = [(2, 4, 100, 32), (1, 64, 4096, 32), (3, 5, 37, 48), (1, 28, 130, 32)]


def _ref(oN, w, B, H):
    N, S, d = oN.shape
    o = oN.view(B, H, S, d).permute(0, 2, 1, 3)
    of = o.float()
    rms = of.pow(2).mean(dim=-1, keepdim=True).add(EPS).rsqrt()
    out = (of * rms * w.float()).to(o.dtype)
    return out.reshape(B, S, H * d)


def _fused(oN, w, B, H):
    from hvn.core.fused_read_norm import fused_read_norm
    return fused_read_norm(oN, w, B, H, EPS)


def _data(B, H, S, d, dtype=torch.float32, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    oN = (torch.randn(B * H, S, d, device="cuda", generator=g) * 3).to(dtype)
    w = torch.rand(H, d, device="cuda", generator=g) + 0.5
    gy = torch.randn(B, S, H * d, device="cuda", generator=g).to(dtype)
    return oN, w, gy


def _run(fn, oN, w, gy, B, H):
    o = oN.clone().requires_grad_(True)
    ww = w.clone().requires_grad_(True)
    y = fn(o, ww, B, H)
    y.backward(gy)
    return y.detach(), o.grad.detach(), ww.grad.detach()


def _rel(a, b):
    return ((a.float() - b.float()).abs().max() / b.float().abs().max()).item()


@pytest.mark.parametrize("case", CASES)
def test_fp32_matches_reference(case):
    B, H, S, d = case
    oN, w, gy = _data(B, H, S, d)
    y0, do0, dw0 = _run(_ref, oN, w, gy, B, H)
    y1, do1, dw1 = _run(_fused, oN, w, gy, B, H)
    assert y1.shape == y0.shape and y1.is_contiguous()
    assert _rel(y1, y0) < 1e-5, f"y {_rel(y1, y0):.2e}"
    assert _rel(do1, do0) < 1e-5, f"do {_rel(do1, do0):.2e}"
    assert _rel(dw1, dw0) < 1e-4, f"dw {_rel(dw1, dw0):.2e}"


def test_bf16_input_matches_reference():
    B, H, S, d = CASES[0]
    oN, w, gy = _data(B, H, S, d, dtype=torch.bfloat16)
    y0, do0, dw0 = _run(_ref, oN, w, gy, B, H)
    y1, do1, dw1 = _run(_fused, oN, w, gy, B, H)
    assert y1.dtype == y0.dtype == torch.bfloat16
    assert _rel(y1, y0) < 1e-2 and _rel(do1, do0) < 1e-2 and _rel(dw1, dw0) < 1e-3


def test_dw_reduction_is_exercised():
    B, H, S, d = CASES[0]
    oN, w, gy = _data(B, H, S, d)
    _, _, dw0 = _run(_ref, oN, w, gy, B, H)
    o4 = oN.view(B, H, S, d).permute(0, 2, 1, 3)
    wrong = (gy.view(B, S, H, d) * o4).sum(dim=(0, 1))          # forgot the rms factor
    assert _rel(wrong, dw0) > 1e-1
    swapped = dw0.flip(0)                                        # heads mixed up
    assert _rel(swapped, dw0) > 1e-2


def test_deltanet_block_uses_fused_read_norm_and_matches():
    import hvn.core.fused_read_norm as fr
    from hvn.core.deltanet import DeltaNetBlock

    torch.manual_seed(0)
    blk = DeltaNetBlock(256, n_heads=8, dropout=0.0, use_triton_scan=True,
                        use_megakernel_scan=True).cuda()
    with torch.no_grad():
        blk.read_norm_weight.uniform_(0.5, 1.5)
    x = torch.randn(2, 128, 256, device="cuda")

    def run(fused, autocast=True):
        blk.use_fused_read_norm = fused
        for p in blk.parameters():
            p.grad = None
        xr = x.clone().requires_grad_(True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
            y = blk(xr)
        y.float().pow(2).mean().backward()
        return (y.float().detach(), xr.grad.clone(),
                blk.read_norm_weight.grad.float().clone())

    calls = {"n": 0}
    orig = fr._FusedReadNorm.apply

    def counting(*a):
        calls["n"] += 1
        return orig(*a)

    truth = run(False, autocast=False)          # fp32 everywhere: the accuracy yardstick
    fr._FusedReadNorm.apply = counting
    try:
        ref = run(False)
        assert calls["n"] == 0
        got = run(True)
        assert calls["n"] == 1, f"fused read norm not used ({calls['n']} calls)"
    finally:
        fr._FusedReadNorm.apply = orig
    for t, a, b, what in zip(truth, ref, got, ("out", "dx", "dw_read")):
        e_ref = (a - t).abs().mean().item()
        e_fus = (b - t).abs().mean().item()
        assert e_fus <= 1.1 * e_ref + 1e-12, f"{what}: fused err {e_fus:.3e} vs reference {e_ref:.3e}"
        assert e_ref > 0, f"{what}: reference equals fp32 truth -- yardstick is vacuous"
