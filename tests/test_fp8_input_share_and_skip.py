"""Gates for the two #195 savings in ``--fp8-linear`` (hvn/edge/fp8_fused.py).

Author: Rambarun Komaljeet

1. **Shared input quantization.** q/k/v/gate_proj read the same x (a
   transposed view from the fused causal conv). ``fp8_quantize`` now quantizes
   such a view without a ``.contiguous()`` copy, and ``_quantize_input``
   reuses one quantization across the four Linears. The hazard is a STALE hit:
   the cache serving codes for bytes that are no longer x's. Tested: in-place
   write (version counter), a different tensor at a reused address, and
   same-address views with another shape / stride. Each staleness test checks
   the result against a fresh quantization, so a cache that wrongly hits fails.
2. **Frozen-recompute GEMM skip.** In a frozen offloaded block's checkpoint
   recompute, out_proj / ffn[3] outputs are never read. Tested: the premise
   (no backward reads those outputs) on CPU, and active-block gradients
   bit-identical with vs without the skip on GPU, with a must-fail partner.
"""

import contextlib

import pytest
import torch
import torch.nn as nn

# Before any CUDA work: pandas/pyarrow after CUDA init crashes this venv
# (see tests/conftest.py).
from train_edge_conversational import apply_fast_stack

import hvn.edge.fp8_fused as F8
from hvn.edge.fp8_fused import E4M3, E5M2, fp8_quantize, fp8_quantize_reference

gpu_fp8 = pytest.mark.skipif(not (torch.cuda.is_available() and F8.fp8_supported()),
                             reason="needs CUDA FP8 tensor cores (sm_89+) + Triton")
DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


def _bits(t):
    return t.contiguous().view(torch.uint8) if t.element_size() == 1 else t.view(torch.int32)


def _same(a, b):
    """Two (q, scale, q_t) triples hold the same bytes."""
    return (torch.equal(_bits(a[0]), _bits(b[0])) and torch.equal(_bits(a[1]), _bits(b[1]))
            and ((a[2] is None and b[2] is None)
                 or (a[2] is not None and b[2] is not None
                     and torch.equal(_bits(a[2]), _bits(b[2])))))


def _fresh(x, transpose=True):
    """Quantization of x's current bytes, bypassing every cache."""
    return fp8_quantize(x.detach().clone(), E4M3, transpose=transpose)


@pytest.fixture(autouse=True)
def _clean_cache():
    F8._X_CACHE[0] = None
    yield
    F8._X_CACHE[0] = None


# ---------------------------------------------------------------------------
# 1. Input-quantization cache: hits only when the bytes are provably the same.
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("dev", DEVICES)
def test_cache_hits_on_same_tensor(dev):
    x = torch.randn(64, 32, device=dev)
    h0 = F8.FP8_STATS["x_hits"]
    a = F8._quantize_input(x, True)
    b = F8._quantize_input(x.reshape(64, 32), True)    # new view object, same bytes
    assert F8.FP8_STATS["x_hits"] == h0 + 1, "same tensor did not hit: sharing is dead"
    assert b is a and _same(a, _fresh(x))


@pytest.mark.parametrize("dev", DEVICES)
def test_cache_misses_after_inplace_write(dev):
    x = torch.randn(64, 32, device=dev)
    old = F8._quantize_input(x, True)
    x.mul_(3.0)                                        # same address, bumps _version
    new = F8._quantize_input(x, True)
    assert _same(new, _fresh(x)), "served the pre-write codes"
    assert not _same(new, old), "write did not change the quantization: test is vacuous"


@pytest.mark.parametrize("dev", DEVICES)
def test_cache_misses_after_write_through_another_view(dev):
    base = torch.randn(64, 32, device=dev)
    x = base[:]                                        # view, shares the version counter
    F8._quantize_input(x, True)
    base[0, 0] = 1000.0
    assert _same(F8._quantize_input(x, True), _fresh(x))


@pytest.mark.parametrize("dev", DEVICES)
def test_cache_misses_on_same_address_other_shape_or_stride(dev):
    a = torch.randn(32, 32, device=dev)
    a[0, 1] = 50.0                                     # make a and a.t() differ
    qa = F8._quantize_input(a, True)
    # Same data_ptr and version; only the stride differs.
    at = a.t()
    assert at.data_ptr() == a.data_ptr() and at.shape == a.shape
    qat = F8._quantize_input(at, True)
    assert _same(qat, _fresh(at)) and not _same(qat, qa)
    # Same data_ptr and version; only the shape differs.
    v = a.view(16, 64)
    assert _same(F8._quantize_input(v, True), _fresh(v))


@pytest.mark.parametrize("dev", DEVICES)
def test_cache_upgrades_when_transpose_is_needed(dev):
    x = torch.randn(64, 32, device=dev)
    q, s, qt = F8._quantize_input(x, False)            # frozen weight: no x^T
    assert qt is None
    full = F8._quantize_input(x, True)                 # trainable weight needs x^T
    assert full[2] is not None and _same(full, _fresh(x))


def test_inference_tensor_is_not_cached():
    with torch.inference_mode():
        x = torch.randn(64, 32)
    assert _same(F8._quantize_input(x, True), _fresh(x))
    assert F8._X_CACHE[0] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs the CUDA caching allocator")
def test_cache_holds_address_so_reuse_cannot_serve_stale():
    """A freed address IS reused by the next same-size tensor -- that is the hazard.

    Part 1 (must-fail): drop the cache's strong reference, free x, and the next
    same-size tensor lands on x's address and is served x's codes -- proving the
    sequence can expose the bug. Part 2: the real cache keeps x's address
    alive, so the same sequence quantizes fresh.
    """
    shape = (1024, 512)               # 2 MiB fp32: large pool, no small temporaries next to it

    def rand(scale):                  # one allocation, no temporary: predictable address
        return torch.empty(shape, device="cuda").normal_().mul_(scale)

    x = rand(1); px = x.data_ptr()
    stale = F8._quantize_input(x, True)
    key, val = F8._X_CACHE[0][0], F8._X_CACHE[0][2]   # (not unpacking x_ref: a name would hold it)
    F8._X_CACHE[0] = (key, None, val)                  # the bug: no strong reference
    del x
    y = rand(7)
    assert y.data_ptr() == px, "allocator did not reuse the address; test cannot see the bug"
    got = F8._quantize_input(y, True)
    assert got is stale and not _same(got, _fresh(y)),         "without the strong ref a reused address must serve stale codes"
    del y, got, stale, val
    F8._X_CACHE[0] = None

    x = rand(1); px = x.data_ptr()
    F8._quantize_input(x, True); del x
    y = rand(7)
    assert y.data_ptr() != px, "the cache did not keep x's address alive"
    assert _same(F8._quantize_input(y, True), _fresh(y)), "stale codes for a new tensor"


# ---------------------------------------------------------------------------
# 1b. Transposed-view quantization: same bytes as quantizing a contiguous copy.
# ---------------------------------------------------------------------------
@gpu_fp8
@pytest.mark.parametrize("dtype", [E4M3, E5M2])
def test_transposed_view_bytes_equal_contiguous_path(dtype, monkeypatch):
    calls = []
    real = F8.fp8_quantize

    def spy(t, dt, transpose=False):
        calls.append(t.is_contiguous())
        return real(t, dt, transpose)
    monkeypatch.setattr(F8, "fp8_quantize", spy)

    torch.manual_seed(3)
    for M, N in [(4096, 2048), (37, 53), (65, 129), (1, 16), (3, 4097)]:
        base = torch.randn(1, N, M, device="cuda", dtype=torch.bfloat16)
        base[0, N // 2, M // 3] = 30.0
        x = base.transpose(1, 2).reshape(M, N)          # what the fused conv hands q_proj
        if M > 1:
            assert not x.is_contiguous() and x.t().is_contiguous()
        for tr in (True, False):
            calls.clear()
            got = F8.fp8_quantize(x, dtype, transpose=tr)
            want = real(x.contiguous(), dtype, transpose=tr)
            assert _same(got, want), f"{M}x{N} {dtype} transpose={tr}"
            q1, s1 = fp8_quantize_reference(x, dtype)
            assert torch.equal(_bits(q1), _bits(got[0])) and torch.equal(_bits(s1), _bits(got[1]))
            if M > 1:   # non-vacuity: the view path recursed on the contiguous t.t()
                assert calls == [False, True], calls


# ---------------------------------------------------------------------------
# 2a. The premise of the skip, on CPU: no backward reads those outputs.
# ---------------------------------------------------------------------------
def _block_grads(block, x, gy, poison=()):
    """Grads after replacing each ``poison`` module's output VALUE with NaN.

    ``out + (nan - out).detach()`` keeps the autograd edge (grad flows exactly
    as before) but every element is NaN: any backward that reads the value,
    directly or through a later op, produces different grads.
    """
    hs = [m.register_forward_hook(
        lambda m, i, o: o + (torch.full_like(o, float("nan")) - o).detach()) for m in poison]
    try:
        block.zero_grad(set_to_none=True)
        xi = x.clone().requires_grad_()
        block(xi).backward(gy)
        return [xi.grad] + [p.grad for p in block.parameters()]
    finally:
        for h in hs:
            h.remove()


def test_unread_output_premise_holds():
    from hvn.core.deltanet import DeltaNetBlock
    torch.manual_seed(0)
    blk = DeltaNetBlock(d_model=64, n_heads=4, dropout=0.0)
    x = torch.randn(2, 16, 64)
    gy = torch.randn(2, 16, 64)
    lin = blk.fp8_unread_output_linears()
    assert len(lin) == 2 and lin[0] is blk.out_proj and lin[1] is blk.ffn[3]
    ref = _block_grads(blk, x, gy)
    got = _block_grads(blk, x, gy, poison=lin)
    for i, (a, b) in enumerate(zip(ref, got)):
        assert torch.equal(a, b), f"grad {i} read a skipped output"
    # Must-fail: ffn[0] feeds GELU, whose backward reads its input.
    bad = _block_grads(blk, x, gy, poison=[blk.ffn[0]])
    assert any(not torch.equal(a, b) for a, b in zip(ref, bad)), "poison check is blind"


# ---------------------------------------------------------------------------
# 2b. The real skip, on GPU: active-block grads identical with vs without it.
# ---------------------------------------------------------------------------
def _grads_bcd(monkeypatch, mode):
    """Active = blocks.0's group; 2 optimizer steps x 3 micro-batches; returns
    every trainable grad at each step + the fp8 counters."""
    import torch.nn.functional as Fn
    from hvn.core.block_coordinate import BlockCoordinateScheduler
    from hvn.edge.model import EdgeHVN
    from hvn.optimizers import build_optimizer

    if mode == "noskip":
        monkeypatch.setattr(F8, "skip_unread_outputs", lambda block: contextlib.nullcontext())
    elif mode == "perturb":     # must-fail: the frozen dx uses a 1-ulp-wrong weight scale
        real = F8._FP8LinearOutputUnread.forward

        def fwd(ctx, x, w, wq, ws, wq_t):
            return real(ctx, x, w, wq, torch.nextafter(ws, torch.full_like(ws, 1e30)), wq_t)
        monkeypatch.setattr(F8._FP8LinearOutputUnread, "forward", staticmethod(fwd))

    Dm, Sb, Vb, ga = 256, 512, 2048, 3
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=Vb, d_model=Dm, num_layers=4, n_heads=8,
                block_plan=["deltanet", "transformer", "deltanet", "deltanet"],
                grad_checkpoint=True, tie_embeddings=True, dropout=0.0)
    apply_fast_stack(m, triton_scan=True, megakernel_scan=True, tf32=True,
                     compile_model=False, verbose=False, fp8_linear=True)
    m.train()
    opt, _ = build_optimizer(m, "muon", lr=3e-3, weight_decay=0.01, fused=False, muon_lr=0.05)
    probe = BlockCoordinateScheduler(m, opt, partition="balanced")
    idx = next(i for i, b in enumerate(probe.blocks) if "blocks.0" in b[0].split("+"))
    del probe
    sched = BlockCoordinateScheduler(
        m, opt, steps_per_block=100, device="cuda", offload_inactive_weights=True,
        offload_park_dtype=torch.bfloat16, offload_prefetch=True,
        offload_cache_blocks=1, partition="balanced", initial_active_idx=idx)
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, Vb, (2, ga, 1, Sb), generator=g)
    for k in F8.FP8_STATS:
        F8.FP8_STATS[k] = 0
    out = []
    for s in range(2):
        for mb in range(ga):
            x = ids[s, mb].cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                lo = Fn.cross_entropy(m(x).float().reshape(-1, Vb), x.reshape(-1)) / ga
            lo.backward()
        out.append([p.grad.detach().clone() for p in m.parameters() if p.grad is not None])
        opt.step()
        opt.zero_grad()
    return out, dict(F8.FP8_STATS)


@gpu_fp8
def test_frozen_recompute_skip_keeps_active_grads_bit_identical(monkeypatch):
    ref, s_ref = _grads_bcd(monkeypatch, "noskip")
    monkeypatch.undo()
    got, s_got = _grads_bcd(monkeypatch, "skip")
    assert s_ref["skipped"] == 0 and s_got["skipped"] > 0, (s_ref, s_got)
    assert s_got["x_hits"] > 0, "input sharing never hit"
    assert len(ref[0]) > 0 and [len(r) for r in ref] == [len(r) for r in got]
    for step, (a, b) in enumerate(zip(ref, got)):
        for i, (ga, gb) in enumerate(zip(a, b)):
            assert torch.equal(ga, gb), f"step {step} grad {i} changed by the skip"
    monkeypatch.undo()
    bad, _ = _grads_bcd(monkeypatch, "perturb")
    assert any(not torch.equal(ga, gb) for a, b in zip(ref, bad) for ga, gb in zip(a, b)), \
        "a 1-ulp error in the skipped path's saved scale went unseen: comparison is blind"
