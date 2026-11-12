"""#196: deterministic causal attention backward (GPU only).

The mem-efficient attention backward adds into dq with atomics, so gradients
differ run to run. ``set_deterministic_attention(True)`` routes the causal path
through ``_DetCausalSDPA``, whose backward selects the kernel's ordered dq path.

Falsifiability: ``test_default_backward_differs_run_to_run`` FAILS (does not
skip) if the plain path is repeatable at this shape -- then the "bit-equal
across runs" test would prove nothing. Production shape 1x64x4096x32 bf16.

Run:  python -m pytest tests/test_deterministic_attention_196.py -q
"""
from __future__ import annotations

import os
import sys
import types

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.core import transformer_block as tb

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

SHAPE = (1, 64, 4096, 32)  # production attention layer: B, H, S, d_head


@pytest.fixture(autouse=True)
def _clean_flags():
    tb.set_deterministic_attention(False)
    torch.use_deterministic_algorithms(False)
    yield
    tb.set_deterministic_attention(False)
    torch.use_deterministic_algorithms(False)


def _inputs(seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed)
    q, k, v, go = (torch.randn(SHAPE, device="cuda", dtype=torch.bfloat16, generator=g)
                   for _ in range(4))
    return q, k, v, go


def _run(fn, q, k, v, go):
    t = [x.clone().requires_grad_() for x in (q, k, v)]
    o = fn(*t)
    o.backward(go)
    torch.cuda.synchronize()
    return o.detach(), [x.grad for x in t]


def _sdpa(q, k, v):
    return F.scaled_dot_product_attention(q, k, v, is_causal=True)


def _det(q, k, v):
    tb.set_deterministic_attention(True)
    try:
        return tb._causal_sdpa(q, k, v)
    finally:
        tb.set_deterministic_attention(False)


def test_default_backward_differs_run_to_run():
    """Falsifier: without this, 'bit-equal across runs' below is vacuous."""
    q, k, v, go = _inputs()
    ref = _run(_sdpa, q, k, v, go)[1][0]
    differs = any(not torch.equal(ref, _run(_sdpa, q, k, v, go)[1][0]) for _ in range(5))
    assert differs, "plain F.sdpa dq was identical over 6 runs: the determinism test proves nothing"


def test_output_bit_equal_and_grads_bit_equal_across_runs():
    q, k, v, go = _inputs()
    ref_out = _sdpa(q, k, v)
    # Reference grads: F.sdpa under the global deterministic flag.
    torch.use_deterministic_algorithms(True)
    try:
        _, ref_g = _run(_sdpa, q, k, v, go)
    finally:
        torch.use_deterministic_algorithms(False)
    for i in range(3):
        out, g = _run(_det, q, k, v, go)
        assert torch.equal(out, ref_out), f"run {i}: output != F.sdpa"
        for name, a, b in zip("qkv", g, ref_g):
            assert torch.equal(a, b), f"run {i}: d{name} not bit-equal"
        assert not torch.are_deterministic_algorithms_enabled(), "global flag leaked"
    # Not all-zero grads (a dead backward would also be 'repeatable').
    assert all(x.abs().sum() > 0 for x in g)


def test_switch_routes_through_det_function_only_when_on():
    q, k, v, _ = _inputs()
    t = [x.clone().requires_grad_() for x in (q, k, v)]
    assert "DetCausalSDPA" not in type(tb._causal_sdpa(*t).grad_fn).__name__
    tb.set_deterministic_attention(True)
    assert "DetCausalSDPA" in type(tb._causal_sdpa(*t).grad_fn).__name__
    with torch.no_grad():  # inference keeps the plain kernel
        assert tb._causal_sdpa(*t).grad_fn is None


def test_block_uses_switch():
    blk = tb.TransformerBlock(d_model=64, n_heads=2).cuda()
    x = torch.randn(1, 16, 64, device="cuda", requires_grad=True)
    tb.set_deterministic_attention(True)
    seen = []
    h = blk.out_proj.register_forward_pre_hook(lambda m, a: seen.append(a[0].grad_fn))
    blk(x).sum().backward()
    h.remove()
    # out_proj input is reshape(transpose(o)); walk back to o's grad_fn.
    fn, names = seen[0], []
    while fn is not None and len(names) < 6:
        names.append(type(fn).__name__)
        fn = fn.next_functions[0][0] if fn.next_functions else None
    assert any("DetCausalSDPA" in n for n in names), names


def test_autocast_matches_sdpa():
    q, k, v, _ = (x.float() for x in _inputs())
    with torch.autocast("cuda", dtype=torch.bfloat16):
        ref = _sdpa(q, k, v)
        tb.set_deterministic_attention(True)
        out = tb._causal_sdpa(*(x.requires_grad_() for x in (q, k, v)))
    assert out.dtype == ref.dtype == torch.bfloat16
    assert torch.equal(out.detach(), ref)


@pytest.mark.parametrize("prev", [(False, False), (True, True)])
def test_global_flag_restored_after_exception(prev):
    q, k, v, go = _inputs()
    out, lse, seed, off = torch.ops.aten._scaled_dot_product_efficient_attention(
        q, k, v, None, True, 0.0, True)
    ctx = types.SimpleNamespace(saved_tensors=(q, k, v, out, lse, seed, off))
    torch.use_deterministic_algorithms(prev[0], warn_only=prev[1])
    # Good call restores too (the e2e path).
    tb._DetCausalSDPA.backward(ctx, go)
    assert (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled()) == prev
    with pytest.raises(RuntimeError):
        tb._DetCausalSDPA.backward(ctx, go[..., :16])  # wrong shape -> kernel raises
    assert (torch.are_deterministic_algorithms_enabled(),
            torch.is_deterministic_algorithms_warn_only_enabled()) == prev
