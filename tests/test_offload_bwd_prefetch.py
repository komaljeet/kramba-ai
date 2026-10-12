"""Backward recompute prefetch for CPU weight offload (2026-10-12).

Under offload, every inactive block downstream of the active one is recomputed
in backward (grad checkpointing), and with the default `cache_blocks=1` its
weights were always re-fetched by a synchronous PAGEABLE copy that also blocked
the host. Measured at 2048x24: 144-192 such misses and 1.2-1.7 s of main-stream
copy time per step. Now each recompute prefetches the next one on the side
stream (`GPUBufferPool.prefetch(..., behind=True)`).

What each test would catch:
  * `test_bwd_prefetch_removes_recompute_misses` -- the feature silently not
    running (misses stay > 0). Its OFF arm proves the miss counter can see a
    miss at all, so a 0 cannot come from a counter that never fires.
  * `test_bwd_prefetch_gradients_match_resident_model` -- wrong weights during
    backward (e.g. block i evicted while its backward still reads it, or a set
    handed to two blocks): gradients are compared by VALUE with a model that
    never offloads, over every active block.
  * `test_parity_bound_catches_wrong_weights_in_backward` -- proves the parity
    bound is loose enough for the reference's own nondeterminism but tight
    enough to see one block's weights served for another's.
  * `test_bwd_prefetch_pattern_survives_lagging_main_stream` + its anti-vacuity
    twin -- the side-stream copy overwriting a set the main stream has not
    finished reading. Pool-level, two sets only, main stream held behind a GPU
    sleep; the twin skips the release wait and must FAIL. Also asserts the
    backward pattern needs no third buffer set (the VRAM claim).
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

from hvn.core.block_coordinate import BlockCoordinateScheduler, GPUBufferPool  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

D, L, S, V = 512, 8, 1024, 4096
PLAN = ["deltanet"] * 3 + ["transformer"] + ["deltanet"] * 4
# Gradient parity bound. The never-offloaded reference itself is intermittently
# nondeterministic at 1.8e-3 / 4.05e-3 relative (measured 2026-10-12: 2 of 8
# fresh processes, the SAME value in every offload arm of the process, so the
# reference run was the odd one out; seen with backward prefetch OFF too).
# Wrong weights give O(1) errors (#164: 1.1-2.6), and
# test_parity_bound_catches_wrong_weights_in_backward proves this bound sees them.
TOL = 2e-2


def _mk():
    torch.manual_seed(0)
    return EdgeHVN(vocab_size=V, d_model=D, num_layers=L, n_heads=16, block_plan=PLAN,
                   grad_checkpoint=True, tie_embeddings=True, dropout=0.0).to("cuda")


def _ids():
    torch.manual_seed(1)
    return torch.randint(0, V, (1, S), device="cuda")


def _step(model, ids):
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids)
        loss = F.cross_entropy(out.float().reshape(-1, V), ids.reshape(-1))
    loss.backward()
    return loss


def _offloaded(bwd_prefetch=True):
    model = _mk()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    sched = BlockCoordinateScheduler(model, opt, steps_per_block=1, device="cuda",
                                     offload_inactive_weights=True)
    model._offload_bwd_prefetch = bwd_prefetch
    return model, opt, sched


def _count_bwd_misses(model, sched, ids):
    """Backward load() misses for the first block (token_embedding active, so
    every layer is recomputed)."""
    pool = sched.buffer_pool
    state = {"bwd": False, "miss": 0}
    orig = pool.load

    def load(key, params):
        if state["bwd"] and key not in pool._resident and pool._offloadable(params):
            state["miss"] += 1
        return orig(key, params)

    pool.load = load
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        out = model(ids)
        loss = F.cross_entropy(out.float().reshape(-1, V), ids.reshape(-1))
    state["bwd"] = True
    loss.backward()
    state["bwd"] = False
    torch.cuda.synchronize()
    pool.load = orig
    return state["miss"]


def test_bwd_prefetch_removes_recompute_misses():
    ids = _ids()
    counts = {}
    for on in (False, True):
        model, opt, sched = _offloaded(bwd_prefetch=on)
        assert sched.active_block_name == "token_embedding"
        counts[on] = _count_bwd_misses(model, sched, ids)
        del model, opt, sched
    # OFF: every layer downstream of token_embedding is recomputed from a miss.
    assert counts[False] == L, f"counter broken or setup changed: {counts[False]} misses with prefetch OFF"
    assert counts[True] == 0, f"backward prefetch is not covering recomputes: {counts[True]} misses"


def _grads_by_active(model, opt, sched, ids, sleep_cycles=0):
    """{active block: {param name: grad}} over one full BCD cycle (lr=0)."""
    hooks = []
    if sleep_cycles:
        # Hold the main stream far behind the host during backward: every
        # block's backward is queued behind a long GPU sleep.
        def fwd_hook(mod, inp, out):
            if out.requires_grad:
                out.register_hook(lambda g: torch.cuda._sleep(sleep_cycles))
        hooks = [b.register_forward_hook(fwd_hook) for b in model.blocks]
    res = {}
    try:
        for _ in range(sched.num_blocks):
            name = sched.active_block_name
            _step(model, ids)
            res[name] = {n: p.grad.detach().float().cpu().clone()
                         for n, p in model.named_parameters() if p.grad is not None}
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
    finally:
        for h in hooks:
            h.remove()
    torch.cuda.synchronize()
    return res


def _reference_grads(ids):
    """Same cycle on a never-offloaded model (BCD without offload)."""
    model = _mk()
    opt = torch.optim.SGD(model.parameters(), lr=0.0)
    sched = BlockCoordinateScheduler(model, opt, steps_per_block=1)
    res = _grads_by_active(model, opt, sched, ids)
    del model, opt, sched
    return res


def _worst(ref, got):
    worst, n = 0.0, 0
    assert ref.keys() == got.keys()
    for blk in ref:
        assert ref[blk].keys() == got[blk].keys(), f"{blk}: different params got grads"
        for name, g in ref[blk].items():
            scale = g.abs().max().item() or 1.0
            worst = max(worst, (got[blk][name] - g).abs().max().item() / scale)
            n += 1
    assert n > 0
    return worst


def test_bwd_prefetch_gradients_match_resident_model():
    ids = _ids()
    ref = _reference_grads(ids)
    model, opt, sched = _offloaded(bwd_prefetch=True)
    got = _grads_by_active(model, opt, sched, ids)
    worst = _worst(ref, got)
    assert worst < TOL, f"offload + bwd prefetch grads differ from resident: rel {worst:.3e}"


def test_parity_bound_catches_wrong_weights_in_backward():
    """Anti-vacuity for TOL: serve block i-2's weights in place of block i-1's
    for the backward recompute (a plausible mix-up between two same-shape
    blocks) and require the parity check to FAIL."""
    ids = _ids()
    ref = _reference_grads(ids)
    model, opt, sched = _offloaded(bwd_prefetch=True)
    pool = sched.buffer_pool
    orig = pool.prefetch
    homes = {}

    def wrong_prefetch(key, params, behind=False):
        if behind and key in homes:
            swap = homes[key]
            saved = [pool._cpu_home[id(p)] for p in params]
            for p, q in zip(params, swap):
                pool._cpu_home[id(p)] = pool._cpu_home[id(q)]
            try:
                return orig(key, params, behind=True)
            finally:
                for p, h in zip(params, saved):
                    pool._cpu_home[id(p)] = h
        return orig(key, params, behind=behind)

    names = [f"blocks.{i}" for i in range(L)]
    for i in range(1, L):
        if PLAN[i] == PLAN[i - 1]:
            homes[names[i]] = list(model.blocks[i - 1].parameters())
    pool.prefetch = wrong_prefetch
    got = _grads_by_active(model, opt, sched, ids)
    worst = _worst(ref, got)
    assert worst > 10 * TOL, f"parity check cannot see wrong backward weights ({worst:.3e})"




# ---------------------------------------------------------------------------
# Pool-level race detector for the backward pattern. A model-level version
# (GPU sleeps in every block's backward) was tried first and was BLIND: with
# the behind-wait removed it still passed, because the free set it reused had
# been released long before. Here the pool holds exactly two sets of one
# shape, so every behind-prefetch MUST reuse the set released one step
# earlier, whose readers are still queued behind a GPU sleep.
# ---------------------------------------------------------------------------

def _bwd_pattern_stress(*, broken: bool, cache_blocks: int, chains: int = 6,
                        length: int = 6, sleep_cycles: int = 5_000_000):
    import torch.nn as nn
    dev = torch.device("cuda")
    pool = GPUBufferPool(dev, cache_blocks=cache_blocks)
    blocks = [[nn.Parameter(torch.full((256, 256), float(k + 1)), requires_grad=False)
               for _ in range(3)] for k in range(length)]
    for params in blocks:
        pool.register_cpu_home(params)
    reused = [0]
    orig = pool._order_prefetch_after

    def _order(ready):
        if ready is not pool.FRESH:
            reused[0] += 1
            if broken:
                return
        orig(ready)
    pool._order_prefetch_after = _order

    reads = []
    for _ in range(chains):
        # backward visits blocks in reverse; block i's recompute starts with
        # load(i) then prefetch(i-1, behind=True); its reads (recompute +
        # backward) run on the main stream; the hook then evicts i.
        pool.load(f"k{length - 1}", blocks[length - 1])
        for i in range(length - 1, -1, -1):
            pool.load(f"k{i}", blocks[i])
            if i > 0:
                pool.prefetch(f"k{i - 1}", blocks[i - 1], behind=True)
            torch.cuda._sleep(sleep_cycles)
            reads.append(torch.stack([(p.data == float(i + 1)).all() for p in blocks[i]]).all())
            torch.cuda._sleep(sleep_cycles)
            reads.append(torch.stack([(p.data == float(i + 1)).all() for p in blocks[i]]).all())
            pool.evict(f"k{i}")
    pool.evict_all()
    torch.cuda.synchronize()
    bad = sum(1 for r in reads if not bool(r))
    sets = sum(len(v) for v in pool._free_buffers.values())
    return bad, len(reads), reused[0], sets


def test_bwd_prefetch_pattern_survives_lagging_main_stream():
    for cache_blocks in (1, 2):
        bad, n, reused, sets = _bwd_pattern_stress(broken=False, cache_blocks=cache_blocks)
        assert reused >= 20, f"cache={cache_blocks}: only {reused} reused sets -- not exercising the wait"
        assert bad == 0, f"cache={cache_blocks}: {bad}/{n} reads saw another block's weights"
        assert sets == 2, f"cache={cache_blocks}: backward prefetch grew the pool to {sets} sets (expected 2)"


def test_bwd_prefetch_stress_catches_a_missing_wait():
    """Anti-vacuity: skipping the release wait must corrupt reads here."""
    bad, n, reused, _ = _bwd_pattern_stress(broken=True, cache_blocks=1)
    assert reused >= 20
    assert bad > 0, f"skipping the wait corrupted 0/{n} reads -- the stress is blind"
