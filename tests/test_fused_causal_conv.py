"""Fused causal depthwise conv (hvn/core/fused_causal_conv.py) vs the eager conv.

The claim is BIT-identity (torch.equal), forward and every gradient, so every
comparison here is exact, never allclose.

What would make these vacuous, and the guard:
  * A fast path that silently falls back to eager would compare eager to eager
    and pass. Every test asserts the STATS counters moved.
  * Inputs that cannot tell a missing bf16 cast or a dropped tap from the right
    answer would let a wrong kernel pass. `test_equality_has_resolving_power`
    shows that two near-miss variants are NOT equal to eager on the same inputs.
  * A full-block test that only checks the output would miss a wrong gradient
    reaching earlier layers; the block test compares every parameter grad and
    the input grad over a sequence of optimizer steps, so a wrong grad shows up
    as diverging weights in later steps too.
"""
from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.core.deltanet import DeltaNetBlock  # noqa: E402
import hvn.core.fused_causal_conv as fcc  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _block(D, K=4, seed=0, H=None):
    torch.manual_seed(seed)
    blk = DeltaNetBlock(D, n_heads=H or max(1, D // 32), conv_kernel=K).cuda()
    with torch.no_grad():  # non-trivial bias (init may be near zero)
        blk.conv.bias.normal_(0, 0.5)
    return blk


def _conv_run(blk, x, gy, fused, cdt, w_grad=True):
    blk.use_fused_conv = fused
    blk.conv.weight.requires_grad_(w_grad)
    blk.conv.bias.requires_grad_(w_grad)
    blk.conv.weight.grad = blk.conv.bias.grad = None
    xr = x.clone().requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=cdt, enabled=cdt is not None):
        y = blk._causal_conv(xr)
    y.backward(gy)
    gw = blk.conv.weight.grad
    gb = blk.conv.bias.grad
    return (y.detach(), xr.grad, None if gw is None else gw.clone(),
            None if gb is None else gb.clone())


@pytest.mark.parametrize("B,S,D", [(1, 4096, 2048), (1, 4096, 256), (2, 129, 96), (3, 5, 70), (1, 2, 64)])
@pytest.mark.parametrize("xdt,cdt", [(torch.float32, torch.bfloat16),
                                     (torch.bfloat16, torch.bfloat16)])
@pytest.mark.parametrize("w_grad", [True, False])
def test_bit_identical_to_eager(B, S, D, xdt, cdt, w_grad):
    blk = _block(D)
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(B, S, D, device="cuda", generator=g).to(xdt)
    gy = torch.randn(B, S, D, device="cuda", generator=g).to(cdt)
    before = dict(fcc.STATS)
    ref = _conv_run(blk, x, gy, False, cdt, w_grad)
    assert fcc.STATS == before, "eager run touched the fused path"
    out = _conv_run(blk, x, gy, True, cdt, w_grad)
    assert fcc.STATS["fwd"] == before["fwd"] + 1
    assert fcc.STATS["bwd"] == before["bwd"] + 1
    assert fcc.STATS["bwd_param"] == before["bwd_param"] + int(w_grad)
    names = ("y", "dx", "dw", "db")
    for n, a, b in zip(names, ref, out):
        if a is None:
            assert b is None, n
            continue
        assert a.dtype == b.dtype and a.shape == b.shape, n
        assert a.stride() == b.stride() if n == "y" else True, n
        assert torch.equal(a, b), f"{n}: max|diff| {(a.float()-b.float()).abs().max()}"


def test_strided_grad_and_zero_signs():
    """A non-contiguous incoming grad (the real case: it comes back through the
    q/k/v GEMMs) and inputs full of +/-0.0 and a -0.0 bias, where skipping vs
    adding a zero tap would flip a sign bit."""
    blk = _block(128)
    with torch.no_grad():
        blk.conv.bias[:8] = -0.0
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(1, 300, 128, device="cuda", generator=g)
    x[:, :, :16] = 0.0
    x[:, :, 16:24] = -0.0
    gbig = torch.randn(1, 128, 300, device="cuda", generator=g).to(torch.bfloat16)
    gbig[:, :8] = -0.0
    gy = gbig.transpose(1, 2)               # strided [B, S, D]
    ref = _conv_run(blk, x, gy, False, torch.bfloat16)
    out = _conv_run(blk, x, gy, True, torch.bfloat16)
    for a, b in zip(ref, out):
        # torch.equal treats -0.0 == 0.0; compare the bit patterns too.
        assert torch.equal(a, b)
        assert torch.equal(a.view(torch.int16) if a.dtype == torch.bfloat16
                           else a.view(torch.int32),
                           b.view(torch.int16) if b.dtype == torch.bfloat16
                           else b.view(torch.int32))


def test_equality_has_resolving_power():
    """Two near-miss kernels must NOT be torch.equal to eager on these inputs,
    or the tests above could not catch them."""
    blk = _block(256)
    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(1, 512, 256, device="cuda", generator=g)
    blk.use_fused_conv = True
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ref = blk._causal_conv(x.clone())
    w = blk.conv.weight.detach().view(256, 1, 4)
    # (a) no bf16 rounding of x before the multiply
    xt = F.pad(x.transpose(1, 2), (3, 0))
    no_cast = F.conv1d(xt, w.float(), blk.conv.bias.float(), groups=256).to(torch.bfloat16)
    assert not torch.equal(ref, no_cast.transpose(1, 2))
    # (b) one tap dropped
    w2 = w.clone()
    w2[..., 0] = 0
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        drop = F.conv1d(xt, w2, blk.conv.bias, groups=256)
    assert not torch.equal(ref, drop.transpose(1, 2))


def test_fallbacks_are_untouched():
    """CPU, fp32 compute (no autocast), fp16 autocast (cuDNN, different
    arithmetic) and the packed doc_ids path stay eager."""
    blk = _block(64)
    before = dict(fcc.STATS)
    blk.use_fused_conv = True
    x = torch.randn(1, 33, 64, device="cuda")
    blk._causal_conv(x)                                        # fp32, no autocast
    with torch.amp.autocast("cuda", dtype=torch.float16):
        blk._causal_conv(x)                                    # fp16
    doc = torch.zeros(1, 33, dtype=torch.long, device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        blk._causal_conv(x, doc_ids=doc)                       # packed path
    blk_cpu = _block(64).cpu()
    blk_cpu.use_fused_conv = True
    blk_cpu._causal_conv(torch.randn(1, 33, 64))
    assert fcc.STATS == before


def test_full_block_sequence_bit_identical():
    """Whole DeltaNetBlock (Triton scan on, as in training) under bf16 autocast:
    loss, input grad and EVERY parameter grad equal at each of 4 SGD steps, so
    a wrong gradient anywhere would also show up as drift in later steps."""
    D, S = 256, 384
    blocks = [_block(D, seed=5, H=8) for _ in range(2)]
    blocks[1].load_state_dict(blocks[0].state_dict())
    for blk, fused in zip(blocks, (False, True)):
        blk.use_triton_scan = True
        blk.use_fused_conv = fused
    opts = [torch.optim.SGD(b.parameters(), lr=0.5) for b in blocks]
    g = torch.Generator(device="cuda").manual_seed(4)
    before = fcc.STATS["fwd"]
    for step in range(4):
        x = torch.randn(2, S, D, device="cuda", generator=g)
        res = []
        for blk, opt in zip(blocks, opts):
            opt.zero_grad(set_to_none=True)
            xr = x.clone().requires_grad_(True)
            torch.manual_seed(100 + step)          # same dropout masks in both arms
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = blk(xr).float().pow(2).mean()
            loss.backward()
            res.append((loss.detach(), xr.grad.clone(),
                        {n: p.grad.clone() for n, p in blk.named_parameters()
                         if p.grad is not None}))
            opt.step()
        (l0, dx0, g0), (l1, dx1, g1) = res
        assert torch.equal(l0, l1), f"step {step} loss"
        assert torch.equal(dx0, dx1), f"step {step} dx"
        assert g0.keys() == g1.keys() and "conv.weight" in g0
        for n in g0:
            assert torch.equal(g0[n], g1[n]), f"step {step} grad {n}"
    assert fcc.STATS["fwd"] == before + 4
