"""Strided-IO megakernel scan (`megakernel_scan_bshd`) == the copy path, bit for bit.

Author: Rambarun Komaljeet

`megakernel_scan_bshd` reads q/k/v/alpha/beta straight from [B, S, H, d] in
their own dtype and writes dq/dk/dv in that dtype, instead of the caller's
permute + fp32-cast copies and autograd's fp32 -> bf16 cast on the way back.
The claim is bit-identity, so every comparison here is `torch.equal`, not a
tolerance. Two ways it could be silently not bit-identical, each with its own
guard:

* addressing -- a wrong stride/head mapping reads the wrong row. Guarded by
  oddly strided inputs (sliced last dim, head-major permute, strided S) whose
  `.contiguous()` would have papered over a bug, and by a check that those
  cases really took the stride path, not the copy fallback;
* rounding -- the old path rounds dq fp32 -> bf16 with torch (round-to-nearest-
  even); the new path rounds inside Triton. Guarded by an exhaustive-style
  rounding test whose tie cases are proven to separate RTNE from truncation.

Unwritten outputs are guarded by NaN-poisoning `torch.empty`: a gradient row
the kernel never stores stays NaN and fails `torch.equal`, instead of
inheriting a freed block that happens to hold the old path's identical bits.
"""

import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@pytest.fixture
def poison_empty(monkeypatch):
    """Make torch.empty / empty_like return NaN-filled float tensors."""
    e, el = torch.empty, torch.empty_like

    def _p(t):
        if t.is_floating_point():
            t.fill_(float("nan"))
        return t

    monkeypatch.setattr(torch, "empty", lambda *a, **kw: _p(e(*a, **kw)))
    monkeypatch.setattr(torch, "empty_like", lambda *a, **kw: _p(el(*a, **kw)))
    yield


def _inputs(B, S, H, d, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)  # noqa: E731
    q = torch.nn.functional.normalize(r(B, S, H, d), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(r(B, S, H, d), dim=-1).to(dtype)
    v = r(B, S, H, d).to(dtype)
    a = torch.sigmoid(r(B, S, H, d) + 1.0)          # fp32, as the block makes it
    b = torch.sigmoid(r(B, S, H, 1))                 # fp32
    do = r(B * H, S, d)                              # fp32 [N,S,d], as read-norm bwd gives
    return [q, k, v, a, b], do


def _old_path(ts, do):
    """The pre-change block code: permute to [N,S,d], megakernel_scan (which
    casts to fp32 inside), autograd casts the grads back."""
    from hvn.core.deltanet_triton import megakernel_scan
    ts = [t.detach().clone().requires_grad_(True) for t in ts]
    B, S, H, _ = ts[0].shape
    to_N = lambda t: t.permute(0, 2, 1, 3).reshape(B * H, S, -1)  # noqa: E731
    o = megakernel_scan(*[to_N(t) for t in ts])
    o.backward(do)
    return o.detach(), [t.grad for t in ts]


def _new_path(ts, do):
    from hvn.core.deltanet_triton import megakernel_scan_bshd
    ts = [t.detach().requires_grad_(True) for t in ts]   # keep the strides
    o = megakernel_scan_bshd(*ts)
    o.backward(do)
    return o.detach(), [t.grad for t in ts]


def _assert_bitwise(old, new, tag):
    (o1, g1), (o2, g2) = old, new
    assert o2.dtype == torch.float32 and o2.shape == o1.shape
    assert torch.isfinite(o1).all(), f"{tag}: reference o not finite"
    assert torch.equal(o1, o2), f"{tag}: o differs, max {(o1 - o2).abs().max():.3e}"
    for nm, a, b in zip(("q", "k", "v", "alpha", "beta"), g1, g2):
        assert a.dtype == b.dtype, f"{tag}: d{nm} dtype {b.dtype} != {a.dtype}"
        assert torch.isfinite(a.float()).all(), f"{tag}: reference d{nm} not finite"
        assert a.float().abs().sum() > 0, f"{tag}: reference d{nm} is all zero (vacuous)"
        assert torch.equal(a, b), (f"{tag}: d{nm} differs, max "
                                   f"{(a.float() - b.float()).abs().max():.3e}")


# ---------------------------------------------------------------- rounding

@cuda_only
def test_triton_bf16_store_rounds_like_torch():
    """The new path rounds dq fp32 -> bf16 in Triton; the old one in torch.
    They must agree on every bit pattern, ties included."""
    import triton
    import triton.language as tl

    @triton.jit
    def _cast(x_ptr, y_ptr, n, RZ: tl.constexpr, BLK: tl.constexpr):
        i = tl.program_id(0) * BLK + tl.arange(0, BLK)
        m = i < n
        x = tl.load(x_ptr + i, mask=m)
        if RZ:
            y = x.to(y_ptr.dtype.element_ty, fp_downcast_rounding="rtz")
        else:
            y = x.to(y_ptr.dtype.element_ty)      # exactly what the kernels do
        tl.store(y_ptr + i, y, mask=m)

    g = torch.Generator(device="cuda").manual_seed(0)
    rnd = torch.randint(-2 ** 31, 2 ** 31 - 1, (1 << 22,), device="cuda",
                        dtype=torch.int64, generator=g).to(torch.int32)
    # exact ties (low 16 bits = 0x8000) with even and odd kept LSB, plus
    # just-below / just-above ties, across the whole exponent range
    hi = torch.randint(0, 1 << 16, (1 << 18,), device="cuda", generator=g).to(torch.int32)
    ties = torch.cat([(hi << 16) | 0x8000, (hi << 16) | 0x7FFF, (hi << 16) | 0x8001])
    special = torch.tensor([0, 1, 0x7FFFFF, 0x800000, 0x7F7FFFFF, 0x7F800000,
                            0x7FC00000, 0x3F808000, 0x3F818000], device="cuda",
                           dtype=torch.int64).to(torch.int32)
    bits = torch.cat([rnd, ties, special, special | (-2 ** 31)])
    x = bits.view(torch.float32)
    ref = x.to(torch.bfloat16)

    def run(rz):
        y = torch.empty(x.numel(), device="cuda", dtype=torch.bfloat16)
        _cast[(triton.cdiv(x.numel(), 1024),)](x, y, x.numel(), RZ=rz, BLK=1024)
        return y

    y = run(False)
    nan = torch.isnan(ref)
    assert torch.equal(torch.isnan(y), nan)
    bad = (y.view(torch.int16) != ref.view(torch.int16)) & ~nan
    assert not bad.any(), (f"{int(bad.sum())} patterns round differently, e.g. "
                           f"{hex(int(bits[bad.nonzero()[0]].item()) & 0xFFFFFFFF)}")
    # anti-vacuity: the input set must separate RTNE from truncation, or this
    # test would pass for a kernel that truncates
    yz = run(True)
    assert ((yz.view(torch.int16) != ref.view(torch.int16)) & ~nan).sum() > 1000


# ------------------------------------------------------------ bit-identity

CASES = [  # B, S, H, d
    (1, 64, 4, 32),     # the 1B shape family (B=1, d_head 32)
    (2, 128, 3, 16),
    (3, 96, 2, 64),     # B>1, S not a power of 2
    (1, 256, 64, 32),   # many heads, as at 1B (H=64)
    (2, 32, 1, 8),
]


@cuda_only
@pytest.mark.parametrize("B,S,H,d", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_bitwise_matches_copy_path(poison_empty, B, S, H, d, dtype):
    ts, do = _inputs(B, S, H, d, seed=B * 100 + S + H + d, dtype=dtype)
    _assert_bitwise(_old_path(ts, do), _new_path(ts, do), f"{B}x{S}x{H}x{d} {dtype}")


def _strided_variants(B, S, H, d, seed):
    """Same values as _inputs, but laid out so .contiguous() would be a copy."""
    ts, do = _inputs(B, S, H, d, seed)
    q, k, v, a, b = ts
    out = {}
    # 1. last-dim slice of a wider buffer: sH = W, sS = H*W, not d / H*d
    W = d + 5
    def sliced(t):
        big = torch.full((B, S, H, W), float("nan"), device="cuda", dtype=t.dtype)
        big[..., 3:3 + d] = t
        return big[..., 3:3 + d]
    out["sliced"] = [sliced(t) for t in (q, k, v, a)] + [b]
    # 2. head-major storage viewed as [B,S,H,d]: sS = d, sH = S*d
    hm = lambda t: t.permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)  # noqa: E731
    out["head_major"] = [hm(t) for t in (q, k, v, a)] + [hm(b)]
    # 3. every other time step of a 2S buffer: sS doubled
    def every2(t):
        big = torch.full((B, 2 * S) + tuple(t.shape[2:]), float("nan"),
                         device="cuda", dtype=t.dtype)
        big[:, ::2] = t
        return big[:, ::2]
    out["strided_S"] = [every2(t) for t in (q, k, v, a)] + [every2(b)]
    return out, ts, do


@cuda_only
@pytest.mark.parametrize("B,S,H,d", [(2, 64, 3, 32), (1, 128, 8, 16), (3, 32, 2, 64)])
def test_oddly_strided_inputs_take_the_stride_path(poison_empty, B, S, H, d):
    """The case `.contiguous()` used to paper over. NaN padding in the
    surrounding buffers means a wrong address reads NaN, not a plausible value."""
    from hvn.core import deltanet_triton as dt
    variants, ts, do = _strided_variants(B, S, H, d, seed=7)
    ref = _old_path(ts, do)
    for name, vts in variants.items():
        assert not vts[0].is_contiguous(), f"{name}: test input is contiguous (vacuous)"
        assert all(torch.equal(x, y) for x, y in zip(vts, ts)), name
        before = dt.SCAN_IO_STATS["fallback_copies"]
        new = _new_path(vts, do)
        assert dt.SCAN_IO_STATS["fallback_copies"] == before, (
            f"{name}: took the contiguous-copy fallback, stride path not tested")
        _assert_bitwise(ref, new, name)


@cuda_only
def test_mixed_layouts_fall_back_and_still_match(poison_empty):
    from hvn.core import deltanet_triton as dt
    ts, do = _inputs(2, 64, 4, 32, seed=3)
    mixed = [ts[0].permute(0, 2, 1, 3).contiguous().permute(0, 2, 1, 3)] + ts[1:]
    before = dt.SCAN_IO_STATS["fallback_copies"]
    new = _new_path(mixed, do)
    assert dt.SCAN_IO_STATS["fallback_copies"] == before + 1
    _assert_bitwise(_old_path(ts, do), new, "mixed")


@cuda_only
def test_a_wrong_head_mapping_would_be_caught():
    """Anti-vacuity for addressing: a kernel that read head 1's rows for head
    0 (simulated by swapping the inputs) must give a different o and dq, so
    the bitwise tests above can see a head mix-up."""
    ts, do = _inputs(1, 64, 4, 32, seed=11)
    sw = [t[:, :, [1, 0, 2, 3]].contiguous() for t in ts]
    o1, g1 = _new_path(ts, do)
    o2, g2 = _new_path(sw, do)
    assert not torch.equal(o1, o2)
    assert not torch.equal(g1[0], g2[0])


@cuda_only
def test_strided_backward_does_not_sync():
    from hvn.core.deltanet_triton import megakernel_scan_bshd
    ts, do = _inputs(1, 64, 4, 32, seed=5)
    ts = [t.requires_grad_(True) for t in ts]
    megakernel_scan_bshd(*ts).backward(do)        # compile/warm
    o = megakernel_scan_bshd(*ts)
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode("error")
    try:
        o.backward(do)
    finally:
        torch.cuda.set_sync_debug_mode("default")


# ------------------------------------------------------------- block level

@cuda_only
@pytest.mark.parametrize("packed", [False, True])
def test_deltanet_block_bitwise_and_uses_stride_path(monkeypatch, packed):
    """Whole DeltaNetBlock under bf16 autocast: use_fused_scan_io on vs off gives
    the same output, x.grad and every parameter grad, bit for bit -- and the on
    run really called the strided scan without falling back."""
    import hvn.core.deltanet as dn
    from hvn.core import deltanet_triton as dt
    calls = {"bshd": 0, "old": 0}
    real_new, real_old = dn._triton_megakernel_scan_bshd, dn._triton_megakernel_scan

    def spy_new(*a):
        calls["bshd"] += 1
        return real_new(*a)

    def spy_old(*a):
        calls["old"] += 1
        return real_old(*a)

    monkeypatch.setattr(dn, "_triton_megakernel_scan_bshd", spy_new)
    monkeypatch.setattr(dn, "_triton_megakernel_scan", spy_old)

    torch.manual_seed(0)
    B, S, D, H = 2, 128, 256, 8
    blk = dn.DeltaNetBlock(d_model=D, n_heads=H, dropout=0.0, use_triton_scan=True,
                           use_megakernel_scan=True).cuda()
    x0 = torch.randn(B, S, D, device="cuda")
    go = torch.randn(B, S, D, device="cuda")
    doc = None
    if packed:
        doc = torch.zeros(B, S, dtype=torch.long, device="cuda")
        doc[:, 50:] = 1
        doc[1, 90:] = 2

    def run(io):
        blk.use_fused_scan_io = io
        blk.zero_grad(set_to_none=True)
        x = x0.clone().requires_grad_(True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            o = blk(x, doc_ids=doc)
        (o.float() * go).sum().backward()
        return o.detach(), x.grad, {n: p.grad.clone() for n, p in blk.named_parameters()
                                    if p.grad is not None}

    fb = dt.SCAN_IO_STATS["fallback_copies"]
    o_off, xg_off, g_off = run(False)
    assert calls == {"bshd": 0, "old": 1}
    o_on, xg_on, g_on = run(None)                 # the default
    assert calls == {"bshd": 1, "old": 1}, calls
    assert dt.SCAN_IO_STATS["fallback_copies"] == fb, "block hit the copy fallback"

    assert torch.equal(o_off, o_on)
    assert xg_off.abs().sum() > 0 and torch.equal(xg_off, xg_on)
    assert g_off.keys() == g_on.keys() and len(g_on) > 5
    for n in g_off:
        assert torch.equal(g_off[n], g_on[n]), f"param grad {n} differs"
