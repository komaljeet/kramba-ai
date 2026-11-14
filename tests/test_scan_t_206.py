"""#206: the "t" (transposed-layout) scan kernels and the v1 launch-width change.

Author: Rambarun Komaljeet

Claims and the test for each:
* v1 tuned warps (fwd 1, ckpt 8 at d_head 32) give the same bits as Triton's default
  4 warps -> torch.equal on o and all grads. Anti-vacuity: the backward at 2 warps
  (a launch change known to move bits) must be caught by the same comparison.
* "t" is the same recurrence -> close to an fp64 reference, about as close as v1.
  Anti-vacuity: production with beta scaled by (1 + 2^-10) must fail the same bound.
* "t" under torch.utils.checkpoint writes the backward's checkpoints in the recompute
  (counter) and gives the same bits as the un-checkpointed "t" path, whose backward
  runs the separate checkpoint pass. The recompute reproduces o bit for bit.
"""
import pytest
import torch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


def _inputs(B, S, H, d, seed, dtype=torch.bfloat16):
    g = torch.Generator(device="cuda").manual_seed(seed)
    r = lambda *s: torch.randn(*s, device="cuda", generator=g)  # noqa: E731
    q = torch.nn.functional.normalize(r(B, S, H, d), dim=-1).to(dtype)
    k = torch.nn.functional.normalize(r(B, S, H, d), dim=-1).to(dtype)
    v = r(B, S, H, d).to(dtype)
    a = torch.sigmoid(r(B, S, H, d) + 2.0)
    b = torch.sigmoid(r(B, S, H, 1))
    do = r(B * H, S, d)
    return [q, k, v, a, b], do


def _run(ts, do, kernel, ckpt=False):
    from hvn.core import deltanet_triton as dt
    old = dt.SCAN_KERNEL[0]
    dt.SCAN_KERNEL[0] = kernel
    try:
        xs = [t.detach().clone().requires_grad_(True) for t in ts]
        if ckpt:
            o = torch.utils.checkpoint.checkpoint(dt.megakernel_scan_bshd, *xs, use_reentrant=False)
        else:
            o = dt.megakernel_scan_bshd(*xs)
        o.backward(do)
        return o.detach(), [x.grad for x in xs]
    finally:
        dt.SCAN_KERNEL[0] = old


def _ref64(ts, do):
    q, k, v, a, b = [t.detach().double().requires_grad_(True) for t in ts]
    B, S, H, d = q.shape
    st = torch.zeros(B, H, d, d, dtype=torch.float64, device="cuda")   # [dv, dk]
    outs = []
    for t in range(S):
        st = st * a[:, t, :, None, :]
        u = v[:, t] - (st * k[:, t, :, None, :]).sum(-1)
        st = st + b[:, t, :, :, None] * (u[..., None] * k[:, t, :, None, :])
        outs.append((st * q[:, t, :, None, :]).sum(-1))       # [B,H,dv]
    o = torch.stack(outs, 2).reshape(B * H, S, d)
    o.backward(do.double())
    return o.detach(), [x.grad for x in (q, k, v, a, b)]


def _rel(x, r):
    return ((x.double() - r).norm() / r.norm()).item()


@cuda_only
@pytest.mark.parametrize("B,S,H,d", [(1, 64, 4, 32), (1, 256, 64, 32), (2, 128, 3, 32)])
def test_v1_tuned_warps_bit_identical(B, S, H, d):
    from hvn.core import deltanet_triton as dt
    ts, do = _inputs(B, S, H, d, seed=S + H)
    saved = dict(dt.SCAN_V1_WARPS)
    try:
        dt.SCAN_V1_WARPS.update(fwd=None, ckpt=None)
        o0, g0 = _run(ts, do, "v1")
        dt.SCAN_V1_WARPS.update(saved)
        assert dt.SCAN_V1_WARPS["fwd"] == 1 and dt.SCAN_V1_WARPS["ckpt"] == 8
        o1, g1 = _run(ts, do, "v1")
    finally:
        dt.SCAN_V1_WARPS.clear(); dt.SCAN_V1_WARPS.update(saved)
    assert torch.equal(o0, o1)
    for a, b in zip(g0, g1):
        assert a.float().abs().sum() > 0
        assert torch.equal(a, b)


@cuda_only
def test_warp_change_that_moves_bits_is_detected():
    """Anti-vacuity: the v1 backward at 2 warps changes bits (#206 tune sweep);
    the comparison used above must see it."""
    from hvn.core import deltanet_triton as dt
    ts, do = _inputs(1, 256, 64, 32, seed=1)
    q, k, v, a, b = ts
    Bt, S, H, D = q.shape
    beta3 = b.reshape(Bt, S, H).contiguous()
    nblk = S // dt._MegaScan.block
    ckpt = torch.empty(Bt * H, nblk, D, D, device="cuda")
    sB, sS, sH, _ = q.stride()
    dt._mk_ckpt_kernel_s[(Bt * H,)](k, v, a, beta3, ckpt, nblk, H, sB, sS, sH, S=S, D=D,
                                    B=dt._MegaScan.block)
    outs = []
    for nw in (4, 2):
        g = [torch.empty_like(t) for t in (q, k, v, a)] + [torch.empty_like(beta3)]
        dt._mk_bwd_kernel_s[(Bt * H,)](q, k, v, a, beta3, ckpt, do, g[0], g[1], g[2], g[3], g[4],
                                       nblk, H, sB, sS, sH, S=S, D=D, B=dt._MegaScan.block,
                                       num_warps=nw)
        outs.append(g)
    assert any(not torch.equal(x, y) for x, y in zip(*outs))


@cuda_only
@pytest.mark.parametrize("B,S,H,d", [(1, 512, 8, 32), (2, 128, 3, 16), (1, 128, 2, 64)])
def test_t_matches_fp64_like_v1(B, S, H, d):
    ts, do = _inputs(B, S, H, d, seed=7, dtype=torch.float32)
    ro, rg = _ref64(ts, do)
    o1, g1 = _run(ts, do, "v1")
    ot, gt = _run(ts, do, "t")
    e1 = [_rel(o1, ro)] + [_rel(x, r) for x, r in zip(g1, rg)]
    et = [_rel(ot, ro)] + [_rel(x, r) for x, r in zip(gt, rg)]
    for a, b in zip(e1, et):
        assert b < max(4 * a, 1e-6), (e1, et)
    # anti-vacuity: a 2^-10 change in beta is far outside that bound
    bad = [t.clone() for t in ts]
    bad[4] = bad[4] * (1 + 2 ** -10)
    ob, _ = _run(bad, do, "v1")
    assert _rel(ob, ro) > 10 * max(e1[0], 1e-6)


@cuda_only
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_t_checkpoint_recompute_fuses_and_matches(dtype):
    from hvn.core import deltanet_triton as dt
    ts, do = _inputs(1, 256, 64, 32, seed=3, dtype=dtype)
    f0, s0 = dt.SCAN_T_STATS["fused_ckpt"], dt.SCAN_T_STATS["separate_ckpt"]
    o_plain, g_plain = _run(ts, do, "t", ckpt=False)
    assert dt.SCAN_T_STATS["separate_ckpt"] == s0 + 1
    assert dt.SCAN_T_STATS["fused_ckpt"] == f0
    o_ck, g_ck = _run(ts, do, "t", ckpt=True)
    assert dt.SCAN_T_STATS["fused_ckpt"] == f0 + 1, "recompute did not write the checkpoints"
    assert dt.SCAN_T_STATS["separate_ckpt"] == s0 + 1, "checkpointed run also ran the pass"
    assert torch.equal(o_plain, o_ck)
    for a, b in zip(g_plain, g_ck):
        assert a.float().abs().sum() > 0
        assert torch.equal(a, b)
