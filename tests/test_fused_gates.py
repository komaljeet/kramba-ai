"""2026-10-17 DeltaNet fusions vs the eager code they replace. BIT-identity only.

Covered (all torch.equal / bit-pattern equality, never allclose):
  * fused_sigmoid_f32  == torch.sigmoid(h.float())       -- forward over ALL
    65,536 bf16 (and fp16) bit patterns; backward == aten sigmoid_backward + cast.
  * fused_residual_add3 == x + d + f                       -- forward, all grads.
  * fused_read_norm_gate == bf16(fused_read_norm(oN) * gate) -- forward, all grads.
  * whole DeltaNetBlock with ternary layers, bf16 autocast, fast flags on vs
    off, over a SEQUENCE of optimizer steps (a wrong grad anywhere shows up as
    drifting weights later, not just at step 0).

Vacuity guards:
  * STATS counters must move: a fast path that silently fell back would
    compare eager with eager and pass.
  * `test_equality_has_resolving_power`: near-miss variants (approximate exp,
    approximate divide, wrong add order, bf16 instead of fp32 product) are
    shown NOT equal on the same inputs, so the equality checks can fail.
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

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

import hvn.core.fused_gates as fg  # noqa: E402
import hvn.core.fused_read_norm as frn  # noqa: E402


def _all_bits(dtype):
    bits = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16)
    return bits.view(dtype).cuda()


def _same_bits(a, b):
    """Bit equality, NaNs compared by NaN-ness (payloads are not meaningful)."""
    if a.shape != b.shape or a.dtype != b.dtype:
        return False
    na, nb = torch.isnan(a), torch.isnan(b)
    if not torch.equal(na, nb):
        return False
    iv = {torch.float32: torch.int32, torch.bfloat16: torch.int16, torch.float16: torch.int16}[a.dtype]
    return torch.equal(a[~na].view(iv), b[~nb].view(iv))


# ---------------------------------------------------------------- sigmoid
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sigmoid_forward_exhaustive(dtype):
    h = _all_bits(dtype)
    before = fg.STATS["sigmoid"]
    got = fg.fused_sigmoid_f32(h)
    assert fg.STATS["sigmoid"] == before + 1
    ref = torch.sigmoid(h.float())
    assert got.dtype == torch.float32
    assert _same_bits(got, ref), (got != ref).sum().item()


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_sigmoid_backward_bit_identical(dtype):
    g = torch.Generator(device="cuda").manual_seed(0)
    # every representable input, x 8 random upstream grads each (incl. huge/tiny)
    h = _all_bits(dtype).repeat(8)
    h = torch.where(torch.isfinite(h), h, torch.zeros_like(h))
    gy = torch.randn(h.shape, device="cuda", generator=g) * torch.exp(
        torch.randn(h.shape, device="cuda", generator=g) * 4)
    res = []
    for fused in (False, True):
        hr = h.clone().requires_grad_(True)
        y = fg.fused_sigmoid_f32(hr) if fused else torch.sigmoid(hr.float())
        y.backward(gy)
        res.append((y.detach(), hr.grad))
    assert _same_bits(res[1][0], res[0][0])
    assert res[1][1].dtype == dtype
    assert _same_bits(res[1][1], res[0][1])


def test_sigmoid_strided_grad_real_shape():
    """Real 1B alpha shape; the grad arrives through a .view like in _project."""
    g = torch.Generator(device="cuda").manual_seed(1)
    h = (torch.randn(1, 4096, 2048, device="cuda", generator=g) * 3).bfloat16()
    gy = torch.randn(1, 4096, 32, 64, device="cuda", generator=g).transpose(2, 3)  # non-contig
    assert not gy.is_contiguous()
    res = []
    for fused in (False, True):
        hr = h.clone().requires_grad_(True)
        y = (fg.fused_sigmoid_f32(hr) if fused else torch.sigmoid(hr.float())).view(1, 4096, 64, 32)
        (y * gy).sum().backward()
        res.append((y.detach(), hr.grad))
    assert torch.equal(res[0][0], res[1][0]) and torch.equal(res[0][1], res[1][1])


# ---------------------------------------------------------------- add3
@pytest.mark.parametrize("shape", [(1, 4096, 2048), (2, 37, 96), (1, 1, 5)])
@pytest.mark.parametrize("dt", [torch.bfloat16, torch.float16])
def test_add3_bit_identical(shape, dt):
    g = torch.Generator(device="cuda").manual_seed(2)
    x = torch.randn(shape, device="cuda", generator=g) * 100
    d = (torch.randn(shape, device="cuda", generator=g) * 3).to(dt)
    f = (torch.randn(shape, device="cuda", generator=g) * 0.01).to(dt)
    gy = torch.randn(shape, device="cuda", generator=g) * 1e3
    res = []
    before = fg.STATS["add3"]
    for fused in (False, True):
        xs = [t.clone().requires_grad_(True) for t in (x, d, f)]
        out = fg.fused_residual_add3(*xs) if fused else xs[0] + xs[1] + xs[2]
        out.backward(gy)
        res.append((out.detach(), *[t.grad for t in xs]))
    assert fg.STATS["add3"] == before + 1
    for a, b in zip(*res):
        assert a.dtype == b.dtype and torch.equal(a, b)


# ---------------------------------------------------------------- read norm + gate
EPS = 1e-6


def _rg_data(B, H, S, d, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    oN = torch.randn(B * H, S, d, device="cuda", generator=g) * 3
    w = torch.rand(H, d, device="cuda", generator=g) + 0.5
    gate = F.silu(torch.randn(B, S, H * d, device="cuda", generator=g) * 2).bfloat16()
    lin = torch.randn(H * d, H * d, device="cuda", generator=g) / (H * d) ** 0.5
    return oN, w, gate, lin


def _rg_run(fused, oN, w, gate, lin, B, H):
    o = oN.clone().requires_grad_(True)
    ww = w.clone().requires_grad_(True)
    gt = gate.clone().requires_grad_(True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        if fused:
            y = frn.fused_read_norm_gate(o, ww, gt, B, H, EPS)
        else:
            y = frn.fused_read_norm(o, ww, B, H, EPS) * gt
        out = F.linear(y, lin)                  # autocast casts y here (eager arm)
    out.float().pow(2).mean().backward()
    return out.detach(), o.grad, ww.grad, gt.grad


@pytest.mark.parametrize("case", [(1, 64, 4096, 32), (2, 4, 100, 32), (3, 5, 37, 48)])
def test_read_gate_bit_identical(case):
    B, H, S, d = case
    data = _rg_data(B, H, S, d)
    before = frn.STATS["read_gate"]
    ref = _rg_run(False, *data, B, H)
    got = _rg_run(True, *data, B, H)
    assert frn.STATS["read_gate"] == before + 1
    for name, a, b in zip(("out", "d_oN", "d_w", "d_gate"), ref, got):
        assert a.dtype == b.dtype, name
        assert torch.equal(a, b), f"{name}: max|diff| {(a.float() - b.float()).abs().max():.3e}"


def test_read_gate_available_only_where_exact():
    oN, w, gate, _ = _rg_data(1, 4, 16, 32)
    assert not frn.fused_read_norm_gate_available(oN, w, gate)   # no autocast
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        assert frn.fused_read_norm_gate_available(oN, w, gate)
        assert not frn.fused_read_norm_gate_available(oN, w, None)
        assert not frn.fused_read_norm_gate_available(oN.bfloat16(), w, gate)
        assert not frn.fused_read_norm_gate_available(oN, w, gate.half())


# ---------------------------------------------------------------- resolving power
def test_equality_has_resolving_power():
    """Near-misses must FAIL the same equality checks, or they prove nothing."""
    import triton
    import triton.language as tl

    @triton.jit
    def approx_sig(x_ptr, y_ptr, n, BLOCK: tl.constexpr):
        offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        x = tl.load(x_ptr + offs, mask=m, other=0.0).to(tl.float32)
        tl.store(y_ptr + offs, 1.0 / (1.0 + tl.exp(-x)), mask=m)

    h = _all_bits(torch.bfloat16)
    h = h[torch.isfinite(h)].contiguous()
    y = torch.empty(h.shape, device="cuda")
    approx_sig[(triton.cdiv(h.numel(), 1024),)](h, y, h.numel(), BLOCK=1024)
    assert not torch.equal(y, torch.sigmoid(h.float())), "approx exp/div indistinguishable"

    g = torch.Generator(device="cuda").manual_seed(3)
    x = torch.randn(1 << 16, device="cuda", generator=g) * 100
    d = (torch.randn(1 << 16, device="cuda", generator=g) * 3).bfloat16()
    f = (torch.randn(1 << 16, device="cuda", generator=g) * 0.01).bfloat16()
    assert not torch.equal(x + d + f, x + (d.float() + f.float())), "add order undetectable"

    oN, w, gate, _ = _rg_data(1, 4, 256, 32)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        good = frn.fused_read_norm_gate(oN, w, gate, 1, 4, EPS)
        y = frn.fused_read_norm(oN, w, 1, 4, EPS)
    assert not torch.equal(good, y.bfloat16() * gate), "bf16 product undetectable"


# ---------------------------------------------------------------- whole block, sequence
def test_full_ternary_block_sequence_bit_identical():
    """DeltaNetBlock with TernaryLinear projections (as in the real model), Triton
    + megakernel scan on, bf16 autocast. Fast flags off vs on: loss, input grad
    and EVERY parameter grad equal at each of 4 optimizer steps."""
    from hvn.core.deltanet import DeltaNetBlock
    from hvn.edge.ternary import apply_ternary_to_model

    D, S = 256, 384
    blocks = []
    for _ in range(2):
        torch.manual_seed(5)
        blk = DeltaNetBlock(D, n_heads=8, dropout=0.0, use_triton_scan=True,
                            use_megakernel_scan=True)
        # the trainer's own exclusion (train_edge_conversational.py build)
        names = [n for n, m in blk.named_modules() if isinstance(m, torch.nn.Linear)
                 and not any(p in n for p in ("beta_proj", "alpha_proj", "decoder"))]
        apply_ternary_to_model(blk, names)
        blocks.append(blk.cuda())
    blocks[1].load_state_dict(blocks[0].state_dict())
    for blk, on in zip(blocks, (False, True)):
        blk.use_fused_read_gate = on
        blk.use_fused_sigmoid = on
        blk.use_fused_residual = on
        for m in blk.modules():                 # + ternary dequantized straight to bf16
            if type(m).__name__ == "TernaryLinear":
                m.use_autocast_dtype = on
                m.use_fused = True              # the --fast quantizer
    n_tern = sum(type(m).__name__ == "TernaryLinear" for m in blocks[1].modules())
    assert n_tern >= 4, "ternary layers not applied -- not the real block"
    opts = [torch.optim.AdamW(b.parameters(), lr=1e-2) for b in blocks]
    g = torch.Generator(device="cuda").manual_seed(4)
    s0 = (frn.STATS["read_gate"], fg.STATS["sigmoid"], fg.STATS["add3"])
    for step in range(4):
        x = torch.randn(2, S, D, device="cuda", generator=g)
        res = []
        for blk, opt in zip(blocks, opts):
            opt.zero_grad(set_to_none=True)
            xr = x.clone().requires_grad_(True)
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
        assert g0.keys() == g1.keys() and "alpha_proj.weight" in g0
        for n in g0:
            assert torch.equal(g0[n], g1[n]), f"step {step} grad {n}"
    s1 = (frn.STATS["read_gate"], fg.STATS["sigmoid"], fg.STATS["add3"])
    # per step: 1 read-gate, 2 sigmoids (alpha, beta), 1 residual add
    assert s1 == (s0[0] + 4, s0[1] + 8, s0[2] + 4), (s0, s1)
    for a, b in zip(blocks[0].parameters(), blocks[1].parameters()):
        assert torch.equal(a, b)


# ---------------------------------------------------------------- ternary to bf16
@pytest.mark.parametrize("fused_q", [False, True])
@pytest.mark.parametrize("step_cache", [False, True])
@pytest.mark.parametrize("frozen_bf16", [False, True])
def test_ternary_autocast_dtype_bit_identical(fused_q, step_cache, frozen_bf16):
    """TernaryLinear fed an fp32 activation under bf16 autocast (FFN w1/w3,
    out_proj): dequantizing in bf16 vs fp32 (then autocast's cast) must give
    identical outputs, input grads, weight grads and weights over 3 steps.
    ``frozen_bf16``: the BCD-parked case (bf16 weight, no grad)."""
    from hvn.edge.ternary import TernaryLinear, enable_ternary_step_cache

    torch.manual_seed(0)
    layers = []
    for on in (False, True):
        torch.manual_seed(0)
        m = TernaryLinear(256, 384).cuda()
        if frozen_bf16:
            m.weight.data = m.weight.data.bfloat16()
            m.weight.requires_grad_(False)
        m.use_fused = fused_q
        m.use_autocast_dtype = on
        enable_ternary_step_cache(m, step_cache)
        layers.append(m)
    seen = []
    orig = F.linear

    def spy(x, w, b=None):
        seen.append(w.dtype)
        return orig(x, w, b)

    g = torch.Generator(device="cuda").manual_seed(1)
    opts = [None if frozen_bf16 else torch.optim.SGD(m.parameters(), lr=0.5) for m in layers]
    for step in range(3):
        x = torch.randn(2, 64, 256, device="cuda", generator=g) * 2
        res = []
        for m, opt in zip(layers, opts):
            xr = x.clone().requires_grad_(True)
            seen.clear()
            F.linear = spy
            try:
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    y = m(xr)
            finally:
                F.linear = orig
            res.append(seen[0])
            y.float().pow(2).mean().backward()
            res.append((y.detach(), xr.grad, None if frozen_bf16 else m.weight.grad.clone()))
            if opt is not None:
                opt.step()
                opt.zero_grad(set_to_none=True)
        dt_off, (y0, dx0, gw0), dt_on, (y1, dx1, gw1) = res
        assert dt_on == torch.bfloat16, "fast path did not dequantize to bf16 (vacuous)"
        assert dt_off == torch.float32, "reference arm is not the old fp32 path"
        assert torch.equal(y0, y1) and torch.equal(dx0, dx1), f"step {step}"
        if not frozen_bf16:
            assert torch.equal(gw0, gw1), f"step {step} weight grad"
    assert torch.equal(layers[0].weight, layers[1].weight)
