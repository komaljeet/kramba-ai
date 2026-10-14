"""Frozen-weight scale cache for TernaryLinear (2026-10-14).

A BCD-inactive block's ternary weights are re-quantized on every forward,
recompute and micro-batch, although they cannot change until the next optimizer
step or block switch. The cache keeps only the absmean SCALE (one scalar per
layer) for the step epoch. Failure mode, as for every weight cache here, is
silent staleness (#117), so:

* bit-exact against the uncached run over a full BCD cycle + 2 steps on the
  real composition (EdgeHVN hybrid, fused ternary, offload with bf16 park,
  Muon) -- every block is frozen, activated, updated and re-parked, so a scale
  that outlived its weight would change a loss or gradient;
* the cache must actually be hit (stats), else the equality is vacuous;
* anti-vacuity: a scale cache that ignores the epoch must FAIL the same
  comparison;
* the scale passed in must be the same bits the quantizer would compute.
"""

import pytest
import torch

import hvn.edge.ternary as T
from hvn.edge.ternary import TernaryLinear
from tests.test_ternary_step_cache import _bcd_run, _first_mismatch

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")


@cuda_only
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_precomputed_scale_is_bit_identical(dtype):
    from hvn.edge.ternary_fused import (ternary_absmean_scale,
                                        ternary_quantize_ste_fused)
    g = torch.Generator(device="cuda").manual_seed(0)
    w = (torch.randn(512, 384, device="cuda", generator=g) * 0.02).to(dtype)
    a = ternary_quantize_ste_fused(w)
    b = ternary_quantize_ste_fused(w, scale=ternary_absmean_scale(w))
    assert torch.equal(a, b)
    # and the scale actually matters (a wrong one is visible, even in bf16)
    c = ternary_quantize_ste_fused(w, scale=ternary_absmean_scale(w) * 1.02)
    assert not torch.equal(a, c)


def _bcd_with_frozen_stats(cached):
    T.FROZEN_SCALE_STATS.update(hits=0, misses=0)
    out = _bcd_run(cached)
    return out, dict(T.FROZEN_SCALE_STATS)


@cuda_only
def test_bcd_offload_frozen_scale_bit_exact_and_hit():
    (ref, _, _, _), f_ref = _bcd_with_frozen_stats(False)
    (got, _, n_steps, ga), f_got = _bcd_with_frozen_stats(True)
    assert f_ref == {"hits": 0, "misses": 0}
    # frozen layers are quantized 2*ga times per step (fwd + recompute) minus
    # those not recomputed; at least most calls must be hits.
    assert f_got["misses"] > 0 and f_got["hits"] > f_got["misses"], f_got
    assert _first_mismatch(ref, got) is None, _first_mismatch(ref, got)


def test_bcd_switch_alone_invalidates():
    """A block switch with NO optimizer step in between must still expire the
    cache: switching re-parks weights, which can change their bits."""
    import torch.nn as nn
    from hvn.core.block_coordinate import BlockCoordinateScheduler
    net = nn.Sequential(TernaryLinear(8, 8), TernaryLinear(8, 8))
    T.enable_ternary_step_cache(net, True)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    sched = BlockCoordinateScheduler(net, opt, steps_per_block=1)
    e0 = T._STEP_EPOCH[0]
    net[0]._fscale = ("k", torch.zeros(()))
    T._STEP_LIVE.add(net[0])
    sched.step()                      # switch, no opt.step()
    assert T._STEP_EPOCH[0] > e0
    assert net[0]._fscale is None
    T.enable_ternary_step_cache(net, False)


@cuda_only
def test_bcd_offload_anti_vacuity_epochless_scale_is_caught(monkeypatch):
    """A frozen scale that never expires must fail the same comparison."""
    (ref, _, _, _), _ = _bcd_with_frozen_stats(False)
    from hvn.edge.ternary_fused import ternary_absmean_scale
    forever = {}

    def stale(self, w):
        if id(self) not in forever:
            forever[id(self)] = ternary_absmean_scale(w)
        return forever[id(self)]

    monkeypatch.setattr(TernaryLinear, "_frozen_step_scale", stale)
    (got, _, _, _), _ = _bcd_with_frozen_stats(True)
    assert _first_mismatch(ref, got) is not None
