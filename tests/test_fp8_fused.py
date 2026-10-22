"""Gates for ``--fp8-linear`` (hvn/edge/fp8_fused.py).

Author: Rambarun Komaljeet

Two different claims, tested separately:

1. **Faithful.** The fused kernels + weight cache compute EXACTLY what the
   validated eager recipe (exp/fp8-pretrain-feasibility ``_q`` / ``_FP8MM``,
   kept verbatim as ``fp8_quantize_reference`` / ``_FP8MMReference``) computes.
   Gate: fp8 bytes, scale bits, y/dX/dW and a multi-step training trajectory
   are bit-identical. FP8 itself being lossy is not an excuse here: both sides
   run the same fp8 arithmetic.
2. **Sane vs bf16.** FP8 is lossy, so vs bf16 the bar is "close, and not
   identical" (identical would mean the fp8 path never ran). Quality evidence is
   the 3000-step feasibility run, not this test.

Each faithfulness test has a must-fail partner showing the comparison can see
the bug it guards against (approximate division, a stale weight cache, the
switch being ignored).
"""

import copy

import pytest
import torch
import torch.nn as nn

# Before any CUDA work: pandas/pyarrow after CUDA init crashes this venv
# (see tests/conftest.py).
from train_edge_conversational import apply_fast_stack

import hvn.edge.fp8_fused as F8
from hvn.edge.fp8_fused import (E4M3, E5M2, fp8_quantize, fp8_quantize_reference,
                                fp8_supported)

fp8_only = pytest.mark.skipif(not (torch.cuda.is_available() and fp8_supported()),
                              reason="needs CUDA FP8 tensor cores (sm_89+) + Triton")


def _bits(t):
    return t.contiguous().view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


def _assert_quant_equal(t, dtype, ctx=""):
    q1, s1 = fp8_quantize_reference(t, dtype)
    q2, s2, qt = fp8_quantize(t, dtype, transpose=True)
    nbad = (_bits(q1) != _bits(q2)).sum().item()
    assert nbad == 0, f"{ctx}: {nbad} of {q1.numel()} fp8 codes differ"
    assert torch.equal(_bits(s1), _bits(s2)), f"{ctx}: scale {s1.item()!r} vs {s2.item()!r}"
    assert torch.equal(_bits(q1.t().contiguous()), _bits(qt)), f"{ctx}: transposed copy differs"


# ---------------------------------------------------------------------------
# 1a. The quantizer in isolation.
# ---------------------------------------------------------------------------
@fp8_only
def test_quantizer_bit_identical_on_real_shapes():
    """The four shapes the debugger measured, generated in the same order.

    1792x896 is the one where the ``/`` prototype differed in 1 of 3.67M codes.
    """
    torch.manual_seed(0)
    for K, O in [(896, 896), (896, 1792), (1792, 896), (896, 50272)]:
        a = torch.randn(4096, K, device="cuda", dtype=torch.bfloat16)
        w = torch.randn(O, K, device="cuda", dtype=torch.bfloat16) * 0.05
        g = torch.randn(4096, O, device="cuda", dtype=torch.bfloat16)
        _assert_quant_equal(a, E4M3, f"x {K}x{O}")
        _assert_quant_equal(w, E4M3, f"w {K}x{O}")
        _assert_quant_equal(g, E5M2, f"g {K}x{O}")
        # fp32 input: the kernel rounds to bf16 itself (autocast's cast).
        _assert_quant_equal(a.float() * 1.37 + 1e-3, E4M3, f"x fp32 {K}x{O}")
        del a, w, g


@fp8_only
def test_quantizer_odd_shapes_and_specials():
    torch.manual_seed(1)
    for shape in [(1, 16), (37, 53), (65, 129), (3, 4097)]:
        for dt in (E4M3, E5M2):
            _assert_quant_equal(torch.randn(shape, device="cuda"), dt, f"{shape} {dt}")
    _assert_quant_equal(torch.zeros(64, 32, device="cuda"), E4M3, "all zero (1e-12 floor)")
    t = torch.randn(100, 48, device="cuda"); t[3, 5] = float("inf")
    _assert_quant_equal(t, E4M3, "inf")
    t = torch.randn(100, 48, device="cuda"); t[99, 47] = float("nan")
    q1, s1 = fp8_quantize_reference(t, E4M3)
    q2, s2, _ = fp8_quantize(t, E4M3)
    # A NaN must poison the whole tensor exactly as eager does, not be dropped
    # by Triton's max -- a silently dropped NaN would hide a divergence.
    assert torch.isnan(s1) and torch.isnan(s2)
    assert torch.equal(_bits(q1), _bits(q2))


@fp8_only
@pytest.mark.parametrize("dtype", [E4M3, E5M2])
def test_cast_exhaustive_over_bf16(dtype):
    """Every finite bf16 value inside the fp8 range, with amax pinned to MAX.

    Covers every rounding tie, the fp8 subnormals and the clamp edge -- the
    cases random normals barely touch.
    """
    maxv = F8._FP8_MAX[dtype]
    allv = torch.arange(-32768, 32768, dtype=torch.int32).to(torch.int16).view(torch.bfloat16)
    v = allv[torch.isfinite(allv) & (allv.float().abs() <= maxv)]
    v = torch.cat([v, torch.tensor([maxv], dtype=torch.bfloat16)])
    pad = (-v.numel()) % 64
    v = torch.cat([v, torch.zeros(pad, dtype=torch.bfloat16)]).view(-1, 64).cuda()
    _assert_quant_equal(v, dtype, f"exhaustive {dtype}")


@fp8_only
def test_scale_sweep_catches_approximate_division():
    """2,000 amax values across 40 binades: the scale bits must match eager.

    This is the test the ``tl.div_rn`` line exists for. Must-fail check, done
    by hand on 2026-10-22: replacing ``tl.div_rn(one, amax)`` with
    ``one / amax`` in ``_cast_kernel`` makes 546 of 8000 comparisons differ.
    """
    g = torch.Generator().manual_seed(7)
    amaxes = torch.exp2(torch.rand(2000, generator=g) * 40 - 20)
    base = torch.rand(2000, 16, 16, generator=g) * 2 - 1
    bad = 0
    for i in range(2000):
        t = (base[i] * amaxes[i]).cuda()
        t[0, 0] = amaxes[i]
        for dt in (E4M3, E5M2):
            q1, s1 = fp8_quantize_reference(t, dt)
            q2, s2, _ = fp8_quantize(t, dt)
            bad += int(not torch.equal(_bits(s1), _bits(s2)))
            bad += int(not torch.equal(_bits(q1), _bits(q2)))
    assert bad == 0, f"{bad} of 8000 scale/code comparisons differ"


# ---------------------------------------------------------------------------
# 1b. One Linear, forward + backward.
# ---------------------------------------------------------------------------
@fp8_only
@pytest.mark.parametrize("x_dtype", [torch.bfloat16, torch.float32])
def test_linear_fwd_bwd_bit_identical(x_dtype):
    torch.manual_seed(2)
    for N, K, O in [(4096, 896, 896), (4096, 1792, 896), (4096, 896, 1792), (208, 128, 384)]:
        x0 = torch.randn(N, K, device="cuda", dtype=x_dtype)
        w0 = torch.randn(O, K, device="cuda") * 0.05
        gy = torch.randn(N, O, device="cuda", dtype=torch.bfloat16)

        xr, wr = x0.clone().requires_grad_(), w0.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yr = F8.fp8_linear_reference(xr, wr.to(torch.bfloat16), None)
        yr.backward(gy)

        lin = nn.Linear(K, O, bias=False).cuda()
        with torch.no_grad():
            lin.weight.copy_(w0)
        F8.enable_fp8_linear(lin, exempt=(), weight_cache=False)
        xf = x0.clone().requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            yf = lin(xf)
        yf.backward(gy)

        ctx = f"N{N} K{K} O{O} x{x_dtype}"
        assert yr.dtype == yf.dtype == torch.bfloat16
        assert torch.equal(yr, yf), f"y differs {ctx}"
        assert xf.grad.dtype == x_dtype and torch.equal(xr.grad, xf.grad), f"dX differs {ctx}"
        assert torch.equal(wr.grad, lin.weight.grad), f"dW differs {ctx}"


# ---------------------------------------------------------------------------
# 1c. Wired into a real model: a multi-step training trajectory.
# ---------------------------------------------------------------------------
V, D, L, H, S = 512, 128, 4, 2, 64


def _model(ternary=True):
    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model
    torch.manual_seed(0)
    plan = ["transformer" if i == 2 else "deltanet" for i in range(L)]
    m = EdgeHVN(vocab_size=V, d_model=D, num_layers=L, n_heads=H, block_plan=plan,
                attn_window=S, tie_embeddings=True, grad_checkpoint=True,
                dropout=0.0).cuda()
    if ternary:
        lin = [n for n, x in m.named_modules() if isinstance(x, nn.Linear)]
        ex = ("beta_proj", "alpha_proj", "decoder")
        apply_ternary_to_model(m, target_names=[n for n in lin if not any(p in n for p in ex)])
    return m


def _train(m, steps=5, accum=2, fixed=False):
    from hvn.optimizers import build_optimizer
    opt, _ = build_optimizer(m, "muon", lr=3e-4, weight_decay=0.01, muon_lr=0.02)
    g = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(steps):
        tot = 0.0
        for _ in range(accum):
            if fixed:      # one repeated batch: learnable, so loss must fall
                g.manual_seed(1)
            x = torch.randint(0, V, (2, S), generator=g).cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = m(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, V).float(), torch.roll(x, -1, 1).reshape(-1)) / accum
            loss.backward()
            tot += loss.item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(tot)
    return losses


def _fast(m, fp8, cache=True):
    return apply_fast_stack(m, triton_scan=True, megakernel_scan=True, tf32=True,
                            compile_model=False, verbose=False, fused_ternary=True,
                            ternary_step_cache=cache, fused_rmsnorm=True, fp8_linear=fp8)


@pytest.fixture
def reference_mode():
    def on():
        F8._REFERENCE_MODE[0] = True
    yield on
    F8._REFERENCE_MODE[0] = False


def _run(m, reference_mode=None, **kw):
    if reference_mode is not None:
        reference_mode()
    try:
        return _train(m, **kw)
    finally:
        F8._REFERENCE_MODE[0] = False


@fp8_only
@pytest.mark.parametrize("ternary", [True, False])
def test_trajectory_bit_identical_to_eager_reference(ternary, reference_mode):
    """fused + per-step weight cache == eager feasibility recipe, every step."""
    base = _model(ternary)
    ref = _fast(copy.deepcopy(base), fp8=True)
    fus = _fast(copy.deepcopy(base), fp8=True)
    for k in F8.FP8_STATS:
        F8.FP8_STATS[k] = 0
    lr = _run(ref, reference_mode)
    ref_fwd = F8.FP8_STATS["fwd"]
    for k in F8.FP8_STATS:
        F8.FP8_STATS[k] = 0
    lf = _run(fus)
    assert lr == lf, f"loss trajectory moved: reference {lr} vs fused {lf}"
    for (n, pr), (_, pf) in zip(ref.named_parameters(), fus.named_parameters()):
        assert torch.equal(pr, pf), f"{n} diverged after {len(lr)} steps"
    # Non-vacuity: both ran fp8, the cache was hit AND refreshed every step,
    # and training moved the weights.
    n_fp8 = sum(1 for m in fus.modules() if getattr(m, "use_fp8", False))
    assert n_fp8 >= 6 * L - 2, n_fp8
    assert ref_fwd > 0 and F8.FP8_STATS["fwd"] == ref_fwd and F8.FP8_STATS["bwd"] > 0
    assert F8.FP8_STATS["w_hits"] > 0, "weight cache never hit"
    assert F8.FP8_STATS["w_misses"] >= n_fp8 * len(lf), "weight cache not refreshed every step"
    assert any(not torch.equal(p0, p1) for p0, p1 in zip(base.parameters(), fus.parameters()))


@fp8_only
def test_stale_weight_cache_is_detectable(monkeypatch, reference_mode):
    """Must-fail partner: a cache that never refreshes MUST move the trajectory.

    Otherwise the bit-identity above would also pass with a broken cache (#117).
    """
    base = _model()
    ref = _fast(copy.deepcopy(base), fp8=True)
    lr = _run(ref, reference_mode)
    stale = {}
    real = F8._fp8_weight

    def never_refresh(mod):
        if id(mod) not in stale:
            stale[id(mod)] = real(mod)
        return stale[id(mod)]

    monkeypatch.setattr(F8, "_fp8_weight", never_refresh)
    bad = _fast(copy.deepcopy(base), fp8=True)
    lb = _run(bad)
    assert lr[0] == lb[0], "step 1 should match (weights not yet updated)"
    assert lr != lb, "a never-refreshed fp8 weight went unnoticed"


@fp8_only
def test_fp8_vs_bf16_close_but_not_identical():
    """FP8 is lossy: same ballpark as bf16, and NOT bit-equal (else it never ran)."""
    base = _model()
    lb = _train(_fast(copy.deepcopy(base), fp8=False), steps=10, fixed=True)
    l8 = _train(_fast(copy.deepcopy(base), fp8=True), steps=10, fixed=True)
    print("bf16", lb)
    print("fp8 ", l8)
    assert lb != l8, "fp8 trajectory equals bf16 -- the fp8 path did not run"
    rel = max(abs(a - b) / abs(a) for a, b in zip(lb, l8))
    assert rel < 0.02, f"fp8 loss {l8} vs bf16 {lb}: max rel diff {rel:.3%}"
    assert l8[-1] < 0.9 * l8[0], f"fp8 run did not learn: {l8}"


@fp8_only
def test_eval_stays_bf16_and_flag_off_is_untouched():
    m = _fast(_model(), fp8=True)
    x = torch.randint(0, V, (1, S), device="cuda")
    F8.FP8_STATS["fwd"] = 0
    m.eval()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        m(x)
    assert F8.FP8_STATS["fwd"] == 0, "eval forward took the fp8 path"
    off = _fast(_model(), fp8=False)
    assert not any(getattr(mod, "use_fp8", False) for mod in off.modules())
    assert not any(isinstance(mod, F8.FP8Linear) for mod in off.modules())


@fp8_only
def test_token_count_not_multiple_of_16_falls_back_to_bf16():
    lin = nn.Linear(128, 384).cuda()
    F8.enable_fp8_linear(lin, exempt=(), weight_cache=False)
    x = torch.randn(200, 128, device="cuda", requires_grad=True)
    F8.FP8_STATS["fallback_rows"] = 0
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = lin(x)
        ref = nn.functional.linear(x, lin.weight, lin.bias)
    y.sum().backward()
    assert F8.FP8_STATS["fallback_rows"] == 1
    assert torch.equal(y, ref)


# ---------------------------------------------------------------------------
# The 1B composition: BCD balanced partition + CPU weight offload (bf16 park,
# prefetch) + Muon, over a full block cycle + 2 steps, so every block is active
# once (cached fp8 weight) and frozen the rest of the time (streamed weight,
# quantized per use, no dW GEMM). Same harness as test_ternary_step_cache.py.
# ---------------------------------------------------------------------------
def _bcd_run(reference):
    import torch.nn.functional as Fn
    from hvn.core.block_coordinate import BlockCoordinateScheduler
    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model
    from hvn.optimizers import build_optimizer

    Dm, Lb, Sb, Vb, ga = 256, 4, 512, 2048, 2
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=Vb, d_model=Dm, num_layers=Lb, n_heads=8,
                block_plan=["deltanet", "transformer", "deltanet", "deltanet"],
                grad_checkpoint=True, tie_embeddings=True, dropout=0.0)
    names = [n for n, mod in m.named_modules() if isinstance(mod, nn.Linear)
             and not any(p in n for p in ("beta_proj", "alpha_proj", "decoder"))]
    apply_ternary_to_model(m, names)
    _fast(m, fp8=True)
    m.train()
    opt, _ = build_optimizer(m, "muon", lr=3e-3, weight_decay=0.01, fused=False,
                             muon_lr=0.05)
    sched = BlockCoordinateScheduler(
        m, opt, steps_per_block=1, device="cuda", offload_inactive_weights=True,
        offload_park_dtype=torch.bfloat16, offload_prefetch=True,
        offload_cache_blocks=1, partition="balanced")
    n_steps = sched.num_blocks + 2
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, Vb, (n_steps, ga, 1, Sb), generator=g)
    for k in F8.FP8_STATS:
        F8.FP8_STATS[k] = 0
    F8._REFERENCE_MODE[0] = reference
    losses = []
    try:
        for s in range(n_steps):
            for mb in range(ga):
                x = ids[s, mb].cuda()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = m(x)
                    loss = Fn.cross_entropy(out.float().reshape(-1, Vb), x.reshape(-1)) / ga
                loss.backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad()
            sched.step()
    finally:
        F8._REFERENCE_MODE[0] = False
    params = [p.detach().float().cpu().clone() for p in m.parameters()]
    return losses, params, dict(F8.FP8_STATS)


@fp8_only
def test_bcd_offload_trajectory_bit_identical_to_reference():
    lr, pr, sr = _bcd_run(reference=True)
    lf, pf, sf = _bcd_run(reference=False)
    assert lr == lf, f"BCD+offload loss trajectory moved:\n ref {lr}\n fus {lf}"
    for i, (a, b) in enumerate(zip(pr, pf)):
        assert torch.equal(a, b), f"param {i} diverged"
    assert sr["fwd"] == sf["fwd"] > 0 and sf["bwd"] > 0
    assert sf["w_hits"] > 0 and sf["w_misses"] > 0, sf
