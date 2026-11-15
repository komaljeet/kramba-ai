"""#208: residual gradient folded into the RMSNorm backward (``--fused-norm-residual``).

Author: Rambarun Komaljeet

Claims, each with a must-fail partner:

1. Forward equals ``F.rms_norm`` bit for bit; ``dx`` (with the residual gradient added)
   and ``dw`` are as close to an fp64 reference as ATen's own fp32 backward + add.
   Partners: the same kernel with the residual dropped, or with the weight one part in
   2^10 off, must FAIL the same tolerance -- so the check can see a small error.
2. In a DeltaNet block, gradients (input and every parameter) match the default path to
   fp32 rounding, AND the fused backward really ran with the residual gradient
   attached (call counters: ``bwd`` up, ``bwd_no_res`` zero).
   Partner: the residual sum fed a detached ``x`` instead of ``x_link`` (the residual
   gradient is lost) must fail both the gradient check and the counter check.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Before any CUDA work: pandas/pyarrow after CUDA init crashes this venv (tests/conftest.py).
import train_edge_conversational as T  # noqa: E402
import hvn.core.fused_norm_residual as FNR  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="Triton kernels need CUDA")

M, N, EPS = 512, 2048, 1e-6


def _inputs():
    g = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(M, N, device="cuda", generator=g) * 3
    w = 1 + 0.1 * torch.randn(N, device="cuda", generator=g)
    gy = torch.randn(M, N, device="cuda", generator=g)
    gr = torch.randn(M, N, device="cuda", generator=g)
    return x, w, gy, gr


def _ref64(x, w, gy, gr):
    x64, w64 = x.double().requires_grad_(), w.double().requires_grad_()
    y = x64 * torch.rsqrt(x64.pow(2).mean(-1, keepdim=True) + EPS) * w64
    y.backward(gy.double())
    return x64.grad + gr.double(), w64.grad


def _rel(a, r):
    return ((a.double() - r).norm() / r.norm()).item()


def _fused(x, w, gy, gr, w_bwd=None):
    """dx, dw from the fused kernel; ``w_bwd`` swaps the weight the backward reads."""
    xx, ww = x.clone().requires_grad_(), w.clone().requires_grad_()
    y, link = FNR.norm_residual(xx, ww, EPS)
    if w_bwd is not None:
        ww.data.copy_(w_bwd)
    terms = (y * gy).sum() + ((link * gr).sum() if gr is not None else 0)
    terms.backward()
    return y, xx.grad, ww.grad


def _aten(x, w, gy, gr):
    xx, ww = x.clone().requires_grad_(), w.clone().requires_grad_()
    y = F.rms_norm(xx, (N,), ww, EPS)
    ((y * gy).sum() + (xx * gr).sum()).backward()
    return y, xx.grad, ww.grad


TOL = 4.0   # fused error may be at most TOL x ATen's (both are fp32-rounding sized)


def _within(fused_err, aten_err):
    return fused_err <= TOL * aten_err + 1e-9


@cuda_only
def test_forward_identical_and_backward_as_close_to_fp64_as_aten():
    x, w, gy, gr = _inputs()
    rdx, rdw = _ref64(x, w, gy, gr)
    ya, dxa, dwa = _aten(x, w, gy, gr)
    yf, dxf, dwf = _fused(x, w, gy, gr)
    assert torch.equal(ya, yf)
    ea, ef = _rel(dxa, rdx), _rel(dxf, rdx)
    wa, wf = _rel(dwa, rdw), _rel(dwf, rdw)
    print(f"dx rel err aten {ea:.2e} fused {ef:.2e}; dw aten {wa:.2e} fused {wf:.2e}")
    assert ea < 1e-5 and wa < 1e-5      # ATen itself is fp32-rounding close
    assert _within(ef, ea) and _within(wf, wa)


@cuda_only
def test_must_fail_residual_dropped_or_weight_off():
    x, w, gy, gr = _inputs()
    rdx, rdw = _ref64(x, w, gy, gr)
    _, dxa, dwa = _aten(x, w, gy, gr)
    ea, wa = _rel(dxa, rdx), _rel(dwa, rdw)
    _, dx_nores, _ = _fused(x, w, gy, None)          # residual gradient never attached
    assert not _within(_rel(dx_nores, rdx), ea)
    _, dx_w, dw_w = _fused(x, w, gy, gr, w_bwd=w * (1 + 2 ** -10))
    assert not _within(_rel(dx_w, rdx), ea)


def _block_grads(enabled, unlink=False, ckpt=False):
    from hvn.core.deltanet import DeltaNetBlock
    torch.manual_seed(0)
    blk = DeltaNetBlock(d_model=64, n_heads=4, dropout=0.0).cuda()
    blk.norm.use_fused = True
    g = torch.Generator(device="cuda").manual_seed(1)
    x = torch.randn(2, 32, 64, device="cuda", generator=g).requires_grad_()
    dy = torch.randn(2, 32, 64, device="cuda", generator=g)
    FNR.ENABLED[0] = enabled
    orig = FNR.norm_residual
    if unlink:
        FNR.norm_residual = lambda x_, w_, e_: (orig(x_, w_, e_)[0], x_.detach())
    before = dict(FNR.STATS)
    try:
        if ckpt:
            import torch.utils.checkpoint as C
            out = C.checkpoint(blk, x, use_reentrant=False)
        else:
            out = blk(x)
        out.backward(dy)
    finally:
        FNR.ENABLED[0] = False
        FNR.norm_residual = orig
    moved = {k: FNR.STATS[k] - before[k] for k in FNR.STATS}
    return [x.grad] + [p.grad for p in blk.parameters()], moved


def _max_rel(ga, gb):
    worst = 0.0
    for a, b in zip(ga, gb):
        if a is None and b is None:
            continue
        worst = max(worst, ((a - b).norm() / b.norm().clamp_min(1e-30)).item())
    return worst


@cuda_only
def test_block_grads_match_and_fused_path_ran():
    g0, m0 = _block_grads(False)
    assert m0["fwd"] == 0 and m0["bwd"] == 0
    g1, m1 = _block_grads(True)
    assert m1["fwd"] == 1 and m1["bwd"] == 1 and m1["bwd_no_res"] == 0, m1
    assert _max_rel(g1, g0) < 1e-5


@cuda_only
def test_block_under_checkpoint():
    """Non-reentrant checkpoint (the production recompute): same grads, fused path ran
    in the recompute (forward twice, backward once)."""
    g0, _ = _block_grads(False, ckpt=True)
    g1, m1 = _block_grads(True, ckpt=True)
    assert m1["fwd"] == 2 and m1["bwd"] == 1 and m1["bwd_no_res"] == 0, m1
    assert _max_rel(g1, g0) < 1e-5


@cuda_only
def test_block_must_fail_residual_not_linked():
    g0, _ = _block_grads(False)
    g2, m2 = _block_grads(True, unlink=True)
    assert m2["bwd_no_res"] == 1      # the counter sees the missing link
    assert _max_rel(g2, g0) > 1e-2    # and so do the gradients


def _trajectory(enabled, steps=6):
    """BCD + offload + checkpoint (+ no-ckpt 1) over several turns, as production."""
    from hvn.core.block_coordinate import BlockCoordinateScheduler
    from hvn.core.hvn_block import RMSNorm
    from hvn.edge.model import EdgeHVN
    V, GA = 64, 2
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=V, d_model=32, num_layers=4, n_heads=4, use_deltanet=True,
                chunk_size=8, dropout=0.0, grad_checkpoint=True).cuda()
    for mod in m.modules():
        if type(mod) is RMSNorm:
            mod.use_fused = True
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    sched = BlockCoordinateScheduler(m, opt, steps_per_block=2, offload_inactive_weights=True,
                                     device=torch.device("cuda"), offload_prefetch=True,
                                     partition="balanced", layers_per_block=2)
    m._active_no_ckpt = 1
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, V, (steps, GA, 1, 32), generator=g)
    FNR.ENABLED[0] = enabled
    before = dict(FNR.STATS)
    losses = []
    try:
        for s in range(steps):
            for mb in range(GA):
                x = ids[s, mb].cuda()
                loss = F.cross_entropy(m(x).float().reshape(-1, V), x.reshape(-1)) / GA
                loss.backward()
                losses.append(loss.item())
            opt.step()
            opt.zero_grad()
            sched.step()
    finally:
        FNR.ENABLED[0] = False
    return losses, {k: FNR.STATS[k] - before[k] for k in FNR.STATS}


@cuda_only
def test_bcd_offload_trajectory_close_and_path_ran():
    l0, m0 = _trajectory(False)
    l1, m1 = _trajectory(True)
    assert m0["bwd"] == 0
    assert m1["bwd"] > 0 and m1["bwd_no_res"] == 0, m1
    assert l0[0] == l1[0]                       # forward unchanged before any update
    worst = max(abs(a - b) / abs(a) for a, b in zip(l0, l1))
    assert worst < 1e-4, worst                  # rounding-level drift only


def test_flag_default_off(monkeypatch):
    """The trainer flag parses to False by default and True when given; the two must
    differ, or this test could not see a flipped default."""
    import argparse
    orig = argparse.ArgumentParser.parse_args
    got = {}

    def stop(self, *a, **k):
        got["ns"] = orig(self, *a, **k)
        raise SystemExit(0)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
    for argv, want in (([], False), (["--fused-norm-residual"], True)):
        monkeypatch.setattr(sys, "argv", ["train_edge_conversational.py"] + argv)
        with pytest.raises(SystemExit):
            T.main()
        assert got["ns"].fused_norm_residual is want
