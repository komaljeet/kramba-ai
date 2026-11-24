"""#207: bias add after the fp8 GEMM, alone and fused into its consumer.

Author: Rambarun Komaljeet

Claims, each with a must-fail partner:

1. Kernel identity, forward AND backward, with ``torch.equal``: ``bias_add`` ==
   ``y + b``; ``bias_silu`` == ``F.silu(y + b)``; ``bias_gelu`` ==
   ``F.gelu(y + b)``; ``add3_bias`` == ``fused_residual_add3(x, d, y + b)`` (and
   ``x + d + (y + b)``). Every one of the 65,536 bf16 values goes through each
   activation (b = 0), plus a random bias, at a real 1B shape too.
   Partner: the same kernels without rounding ``y + b`` to bf16 first must
   DIFFER from ATen, so the comparison sees a sub-ulp change in the add.
2. The DeltaNet block with fp8 Linears gives the same output and the same
   gradients (inputs and every parameter) in all three ``--fp8-bias`` modes,
   AND the fused path really ran (call counters), so (2) cannot pass by
   comparing eager to eager.
3. A BCD + offload + checkpoint trajectory (frozen blocks' recompute skips the
   unread FFN GEMM, #195) is bit-identical across modes over several turns.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import hvn.edge.fused_bias as FB  # noqa: E402
import hvn.edge.fp8_fused as F8  # noqa: E402
from hvn.core.fused_gates import fused_residual_add3  # noqa: E402

fp8_only = pytest.mark.skipif(not (torch.cuda.is_available() and F8.fp8_supported()),
                              reason="needs CUDA with fp8 tensor cores (sm_89+)")
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")


def _all_bf16():
    v = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    return v[torch.isfinite(v.float())]


def _eager(y, b, act):
    t = y + b
    return {"none": lambda: t, "silu": lambda: F.silu(t), "gelu": lambda: F.gelu(t)}[act]()


_FUSED = {"none": FB.bias_add, "silu": FB.bias_silu, "gelu": FB.bias_gelu}


def _check_act(y, b, act, round_t=True):
    """(fwd equal, dy equal, db equal) of fused vs eager, same upstream grad."""
    g = torch.Generator(device="cuda").manual_seed(3)
    dy = torch.randn(y.shape, device="cuda", generator=g).to(torch.bfloat16)
    y0, b0 = y.clone().requires_grad_(), b.clone().requires_grad_()
    ref = _eager(y0, b0, act)
    ref.backward(dy)
    y1, b1 = y.clone().requires_grad_(), b.clone().requires_grad_()
    FB._ROUND_T[0] = round_t
    try:
        out = _FUSED[act](y1, b1)
        out.backward(dy)
    finally:
        FB._ROUND_T[0] = True
    return torch.equal(out, ref), torch.equal(y1.grad, y0.grad), torch.equal(b1.grad, b0.grad)


@cuda_only
@pytest.mark.parametrize("act", ["none", "silu", "gelu"])
def test_every_bf16_value(act):
    v = _all_bf16().cuda()
    n = (v.numel() // 256) * 256
    y = v[:n].view(-1, 256).contiguous()
    zero = torch.zeros(256, device="cuda", dtype=torch.bfloat16)
    assert _check_act(y, zero, act) == (True, True, True)
    b = (torch.randn(256, device="cuda", generator=torch.Generator(device="cuda").manual_seed(1))
         * 0.5).to(torch.bfloat16)
    assert _check_act(y, b, act) == (True, True, True)


@cuda_only
@pytest.mark.parametrize("act,shape", [("none", (4096, 2048)), ("silu", (4096, 2048)),
                                       ("gelu", (4096, 4096)), ("gelu", (48, 80))])
def test_real_shapes(act, shape):
    g = torch.Generator(device="cuda").manual_seed(2)
    y = (torch.randn(shape, device="cuda", generator=g) * 2).to(torch.bfloat16)
    b = torch.randn(shape[1], device="cuda", generator=g).to(torch.bfloat16)
    assert _check_act(y, b, act) == (True, True, True)


def _partner_case():
    g = torch.Generator(device="cuda").manual_seed(2)
    y = (torch.randn(512, 1024, device="cuda", generator=g) * 2).to(torch.bfloat16)
    b = (torch.randn(1024, device="cuda", generator=g) * 0.37).to(torch.bfloat16)
    return y, b


@cuda_only
@pytest.mark.parametrize("act", ["none", "silu", "gelu"])
def test_must_fail_add_rounded_toward_zero(act):
    """Partner of the two tests above: rounding ``y + b`` to bf16 toward zero
    instead of to nearest (a sub-ulp change) must be caught, forward and, where
    the backward reads it, backward."""
    fwd, dy, _ = _check_act(*_partner_case(), act, round_t=False)
    assert not fwd
    if act != "none":
        assert not dy


@cuda_only
def test_must_fail_gelu_tables_off_by_one_ulp():
    """GELU goes through ATen-made tables. Partner: the same tables with every
    entry one ulp off must fail forward and backward (the bias gradient, a bf16
    sum over rows, can round the difference away, so it is not asserted)."""
    y, b = _partner_case()
    key = str(y.device)
    fwd, h = FB.gelu_tables(y.device)
    try:
        FB._LUT[key] = ((fwd.view(torch.int16) ^ 1).view(torch.bfloat16),
                        (h.view(torch.int32) ^ 1).view(torch.float32))
        ok_fwd, ok_dy, ok_db = _check_act(y, b, "gelu")
    finally:
        FB._LUT[key] = (fwd, h)
    assert not ok_fwd and not ok_dy
    assert _check_act(y, b, "gelu") == (True, True, True)


def _add3_case(shape=(1, 4096, 2048), seed=4):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn(shape, device="cuda", generator=g)
    d = torch.randn(shape, device="cuda", generator=g).to(torch.bfloat16)
    y = torch.randn((shape[0] * shape[1], shape[2]), device="cuda", generator=g).to(torch.bfloat16)
    b = torch.randn(shape[2], device="cuda", generator=g).to(torch.bfloat16)
    up = torch.randn(shape, device="cuda", generator=g)
    return x, d, y, b, up


def _add3_grads(fn, x, d, y, b, up):
    xs = [t.clone().requires_grad_() for t in (x, d, y, b)]
    out = fn(*xs)
    out.backward(up)
    return out, [t.grad for t in xs]


@cuda_only
@pytest.mark.parametrize("shape", [(1, 4096, 2048), (2, 24, 80)])
def test_add3_bias(shape):
    case = _add3_case(shape)
    o0, g0 = _add3_grads(lambda x, d, y, b: fused_residual_add3(x, d, (y + b).view(x.shape)), *case)
    o1, g1 = _add3_grads(lambda x, d, y, b: x + d + (y + b).view(x.shape), *case)
    o2, g2 = _add3_grads(FB.add3_bias, *case)
    assert torch.equal(o0, o2) and torch.equal(o1, o2)
    for a, c, e in zip(g0, g1, g2):
        assert torch.equal(a, e) and torch.equal(c, e)


@cuda_only
def test_add3_bias_must_fail_rounded_toward_zero():
    case = _add3_case((1, 256, 512))
    o0, _ = _add3_grads(lambda x, d, y, b: fused_residual_add3(x, d, (y + b).view(x.shape)), *case)
    FB._ROUND_T[0] = False
    try:
        o2, _ = _add3_grads(FB.add3_bias, *case)
    finally:
        FB._ROUND_T[0] = True
    assert not torch.equal(o0, o2)


# ---------------------------------------------------------------------------
# amax hints: the producer's amax replaces fp8_quantize's own pass.
# ---------------------------------------------------------------------------
@pytest.fixture
def hints_on():
    """amax hints are opt-in (--fused-bias-amax); these tests turn them on."""
    prev = FB.AMAX_HINTS[0]
    FB.AMAX_HINTS[0] = True
    yield
    FB.AMAX_HINTS[0] = prev


def _quant_both(t, dtype):
    """fp8_quantize(t) with the hint (if any) and with it removed: (hinted, plain)."""
    hinted = F8.fp8_quantize(t, dtype, transpose=True)     # a taken hint is removed
    hits = FB.STATS["amax_hits"]
    plain = F8.fp8_quantize(t, dtype, transpose=True)
    assert FB.STATS["amax_hits"] == hits, "a hint was used twice"
    return hinted, plain


def _same_q(a, b):
    return (torch.equal(a[0].view(torch.uint8), b[0].view(torch.uint8))
            and torch.equal(a[1], b[1])
            and torch.equal(a[2].view(torch.uint8), b[2].view(torch.uint8)))


def _amax_producers():
    """(name, tensor written by a fused kernel with a hint, fp8 dtype it feeds)."""
    FB.clear_amax_hints()
    g = torch.Generator(device="cuda").manual_seed(6)
    y = (torch.randn(4096, 4096, device="cuda", generator=g) * 3).bfloat16()
    b = torch.randn(4096, device="cuda", generator=g).bfloat16()
    yield "gelu fwd", FB.bias_gelu(y, b), F8.E4M3
    for act, fn in (("silu", FB.bias_silu), ("gelu", FB.bias_gelu)):
        yr = y[:, :2048].contiguous().requires_grad_()
        out = fn(yr, b[:2048])
        gt, = torch.autograd.grad(out, yr, torch.randn(out.shape, device="cuda",
                                                       generator=g).bfloat16())
        yield f"{act} bwd", gt, F8.E5M2
    x, d, y3, b3, up = _add3_case((1, 4096, 2048))
    d.requires_grad_(); y3.requires_grad_()
    out = FB.add3_bias(x, d, y3, b3)
    gd, gy = torch.autograd.grad(out, (d, y3), up)
    yield "add3 bwd gy", gy, F8.E5M2
    yield "add3 bwd gd", gd.view(-1, gd.shape[-1]), F8.E5M2


@cuda_only
def test_amax_hints_give_the_same_codes_and_are_used(hints_on):
    for name, t, dt in _amax_producers():
        hits = FB.STATS["amax_hits"]
        hinted, plain = _quant_both(t, dt)
        assert FB.STATS["amax_hits"] == hits + 1, f"{name}: hint not taken"
        assert _same_q(hinted, plain), name


@cuda_only
def test_amax_hint_must_fail_partners(hints_on):
    """A hint is refused after an in-place write (version bump), and a wrong
    amax would be visible in the codes, so the equality above is not vacuous."""
    FB.clear_amax_hints()
    g = torch.Generator(device="cuda").manual_seed(7)
    y = torch.randn(512, 1024, device="cuda", generator=g).bfloat16()
    b = torch.randn(1024, device="cuda", generator=g).bfloat16()
    h = FB.bias_gelu(y, b)
    h.mul_(2)                                     # in-place: the hint is stale now
    hits = FB.STATS["amax_hits"]
    stale, plain = _quant_both(h, F8.E4M3)
    assert FB.STATS["amax_hits"] == hits, "stale hint was taken"
    assert _same_q(stale, plain)
    h2 = FB.bias_gelu(y, b)
    FB._HINTS[-1] = FB._HINTS[-1][:5] + (FB._HINTS[-1][5] * 2,)   # corrupt the hinted amax
    wrong, plain2 = _quant_both(h2, F8.E4M3)
    assert not _same_q(wrong, plain2), "a doubled amax went unnoticed"


# ---------------------------------------------------------------------------
# Block level and BCD trajectory.
# ---------------------------------------------------------------------------
def _fp8_block(seed=0, d=64, h=4):
    from hvn.core.deltanet import DeltaNetBlock
    from hvn.edge.fp8_fused import FP8Linear
    torch.manual_seed(seed)
    blk = DeltaNetBlock(d, n_heads=h, dropout=0.0, use_triton_scan=True,
                        use_megakernel_scan=True).cuda()
    for m in blk.modules():
        if type(m) is torch.nn.Linear and m.out_features % 16 == 0 and m.in_features % 16 == 0:
            m.__class__ = FP8Linear
            m.use_fp8 = True
    for n in ("alpha_proj", "beta_proj"):
        getattr(blk, n).use_fp8 = False
    with torch.no_grad():
        for p in blk.parameters():
            p.add_(torch.randn_like(p) * 0.02)
    return blk


def _block_run(mode, seed=0):
    FB.set_mode(mode)
    try:
        blk = _fp8_block(seed)
        g = torch.Generator(device="cuda").manual_seed(5)
        x = torch.randn(1, 64, 64, device="cuda", generator=g).requires_grad_()
        up = torch.randn(1, 64, 64, device="cuda", generator=g)
        before = dict(FB.STATS)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = blk(x)
        out.backward(up)
        moved = {k: FB.STATS[k] - before[k] for k in FB.STATS}
        return out.detach(), x.grad, [p.grad for p in blk.parameters()], moved
    finally:
        FB.set_mode("fused")


@fp8_only
def test_block_modes_bit_identical_and_paths_ran():
    o0, x0, p0, m0 = _block_run("off")
    assert sum(m0.values()) == 0, m0
    prev = FB.AMAX_HINTS[0]
    for mode in ("add", "fused", "fused-nohint"):
        FB.AMAX_HINTS[0] = mode != "fused-nohint"
        try:
            o, xg, pg, moved = _block_run(mode.split("-")[0])
        finally:
            FB.AMAX_HINTS[0] = prev
        if mode == "add":
            assert moved["bias"] == 3 and moved["silu"] == moved["gelu"] == moved["add3"] == 0, moved
        else:
            assert moved["silu"] == moved["gelu"] == moved["add3"] == 1 and moved["bias"] == 0, moved
            # hints: GELU output (fwd); grads into gate_proj, ffn[0], ffn[3], out_proj (bwd)
            want = 5 if mode == "fused" else 0
            assert moved["amax_hits"] == want, moved
        assert torch.equal(o0, o), mode
        assert torch.equal(x0, xg), mode
        assert len(p0) == len(pg)
        for i, (a, b) in enumerate(zip(p0, pg)):
            assert (a is None) == (b is None)
            assert a is None or torch.equal(a, b), f"{mode}: param {i}"


def _trajectory(mode, steps=6):
    from hvn.core.block_coordinate import BlockCoordinateScheduler
    from hvn.edge.model import EdgeHVN
    from hvn.edge.fp8_fused import FP8Linear
    FB.set_mode(mode)
    try:
        V, GA = 64, 2
        torch.manual_seed(0)
        m = EdgeHVN(vocab_size=V, d_model=64, num_layers=4, n_heads=4, use_deltanet=True,
                    chunk_size=8, dropout=0.0, grad_checkpoint=True).cuda()
        for blk in m.blocks:
            blk.use_triton_scan = blk.use_megakernel_scan = True
            for name, mod in blk.named_modules():
                if (type(mod) is torch.nn.Linear and not name.endswith(("alpha_proj", "beta_proj"))):
                    mod.__class__ = FP8Linear
                    mod.use_fp8 = True
        opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
        sched = BlockCoordinateScheduler(m, opt, steps_per_block=2, offload_inactive_weights=True,
                                         device=torch.device("cuda"), offload_prefetch=True,
                                         partition="balanced", layers_per_block=2)
        m._active_no_ckpt = 1
        g = torch.Generator().manual_seed(1)
        ids = torch.randint(0, V, (steps, GA, 1, 32), generator=g)
        before = dict(FB.STATS)
        losses, grads = [], []
        for s in range(steps):
            for mb in range(GA):
                x = ids[s, mb].cuda()
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = m(x)
                loss = F.cross_entropy(logits.float().reshape(-1, V), x.reshape(-1)) / GA
                loss.backward()
                losses.append(loss.detach().clone())
            grads.append([p.grad.detach().clone() for p in m.parameters()
                          if p.requires_grad and p.grad is not None])
            opt.step()
            opt.zero_grad()
            sched.step()
        moved = {k: FB.STATS[k] - before[k] for k in FB.STATS}
        return losses, grads, moved, F8.FP8_STATS["skipped"]
    finally:
        FB.set_mode("fused")


@fp8_only
def test_bcd_offload_trajectory_bit_identical():
    l0, g0, m0, _ = _trajectory("off")
    prev = FB.AMAX_HINTS[0]
    for mode in ("add", "fused", "fused+hints"):
        FB.AMAX_HINTS[0] = mode == "fused+hints"
        try:
            l1, g1, m1, skipped = _trajectory(mode.split("+")[0])
        finally:
            FB.AMAX_HINTS[0] = prev
        assert sum(m1.values()) > 0, f"{mode}: fused path never ran"
        if mode != "add":
            assert m1["gelu"] > 0 and m1["add3"] > 0 and m1["silu"] > 0, m1
            assert (m1["amax_hits"] > 0) == (mode == "fused+hints"), m1
        assert len(l0) == len(l1) and all(torch.equal(a, b) for a, b in zip(l0, l1)), mode
        for s, (a, b) in enumerate(zip(g0, g1)):
            assert len(a) == len(b) > 0, f"step {s}: no active gradients"
            for i, (x, y) in enumerate(zip(a, b)):
                assert torch.equal(x, y), f"{mode} step {s} grad {i}"
    assert skipped > 0, "no frozen recompute skipped a GEMM: the skip path is untested"
