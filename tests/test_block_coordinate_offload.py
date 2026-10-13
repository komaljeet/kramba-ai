"""Tests for CPU weight offload on BCD-inactive blocks (Issue #101, extends
#99, corrects a first, reverted attempt).

The critical gate: does offloading change what the model computes? It must
not -- the forward math is identical, only *where* (CPU vs GPU) each block's
weight tensor physically sits at any instant changes. All GPU-gated (the
mechanism only bites under CUDA + a live autograd graph; correctness is
meaningless to test on CPU where nothing ever moves).

Covers:
  1. offload_inactive_weights requires device AND model.grad_checkpoint=True
     (fails fast otherwise -- the whole mechanism is meaningless without
     checkpointing's recompute window to time eviction against).
  2. Default (offload_inactive_weights=False) is unaffected -- no forward
     path changes, regression guard for #99's existing 7 tests.
  3. THE gate: gradient parity. Two identical models (same seed, same BCD
     active block, same grad_checkpoint), differing ONLY in
     offload_inactive_weights. The active block's gradients (the only ones
     that actually accumulate, matching #99's requires_grad scheme) must
     match closely between the two -- offload changes tensor placement,
     not values.
  4. Inactive blocks actually end up back on CPU after a full forward+
     backward step (the memory-saving mechanism, verified directly).
  5. A full training loop with offload enabled still converges (loss
     decreases over multiple BCD cycles).

Run:  python -m pytest tests/test_block_coordinate_offload.py -q
  or:  python tests/test_block_coordinate_offload.py
"""
from __future__ import annotations


def _pinned_ok(sched, p, name):
    """#157: is `p` a deliberately-resident parameter, and is it where it must be?

    A parameter reachable from two modules -- the tied embedding/decoder table --
    is never evicted, because forward() uses it at both ends and no single
    block's activation window covers both uses. Returns True if the caller
    should skip its usual device/dtype expectation, and ASSERTS the parameter is
    actually resident, so the exemption cannot quietly hide a leak.
    """
    if id(p) not in getattr(sched, "_pinned", ()):
        return False
    assert p.device.type == "cuda", (
        f"pinned parameter in {name!r} was evicted to {p.device}; "
        f"forward() will fail at its far use")
    return True



import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.core.block_coordinate import BlockCoordinateScheduler
from hvn.edge.model import EdgeHVN

_HAS_CUDA = torch.cuda.is_available()


def _tiny_model(num_layers=3, grad_checkpoint=False):
    return EdgeHVN(
        vocab_size=64, d_model=32, num_layers=num_layers, n_heads=4,
        use_deltanet=True, chunk_size=8, dropout=0.0,
        grad_checkpoint=grad_checkpoint,
    )


def test_offload_requires_device_and_checkpoint():
    model = _tiny_model(grad_checkpoint=False)  # checkpoint OFF on purpose
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    try:
        BlockCoordinateScheduler(model, opt, steps_per_block=2,
                                 offload_inactive_weights=True, device="cpu")
        raise AssertionError("expected ValueError: grad_checkpoint not set")
    except ValueError as e:
        assert "grad_checkpoint" in str(e)

    model2 = _tiny_model(grad_checkpoint=True)
    opt2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    try:
        BlockCoordinateScheduler(model2, opt2, steps_per_block=2,
                                 offload_inactive_weights=True)
        raise AssertionError("expected ValueError: device missing")
    except ValueError as e:
        assert "device is required" in str(e)


def test_default_offload_off_forward_unaffected():
    """Regression guard: offload_inactive_weights=False must not touch the
    model's _offload_active_names -- forward() takes the pre-#101 path."""
    model = _tiny_model(grad_checkpoint=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    BlockCoordinateScheduler(model, opt, steps_per_block=1)
    assert model._offload_active_names is None


if not _HAS_CUDA:
    print("  [skip] CUDA not available -- offload correctness tests skipped")
else:
    def _make_pair(num_layers=3):
        """Two identically-initialized models + BCD schedulers, same active
        block, differing ONLY in offload_inactive_weights."""
        device = torch.device("cuda")
        torch.manual_seed(0)
        m_off = _tiny_model(num_layers, grad_checkpoint=True).to(device)
        torch.manual_seed(0)
        m_on = _tiny_model(num_layers, grad_checkpoint=True).to(device)
        for (n1, p1), (n2, p2) in zip(m_off.named_parameters(), m_on.named_parameters()):
            assert n1 == n2
            assert torch.equal(p1, p2), f"init mismatch at {n1}"

        opt_off = torch.optim.AdamW(m_off.parameters(), lr=1e-2)
        opt_on = torch.optim.AdamW(m_on.parameters(), lr=1e-2)
        sched_off = BlockCoordinateScheduler(m_off, opt_off, steps_per_block=2)
        sched_on = BlockCoordinateScheduler(
            m_on, opt_on, steps_per_block=2,
            offload_inactive_weights=True, device=device,
        )
        assert sched_off.active_block_name == sched_on.active_block_name
        return device, m_off, opt_off, sched_off, m_on, opt_on, sched_on

    def test_gradient_parity_offload_vs_no_offload():
        device, m_off, opt_off, sched_off, m_on, opt_on, sched_on = _make_pair()
        torch.manual_seed(42)
        ids = torch.randint(0, 64, (2, 16), device=device)

        logits_off = m_off(ids)
        loss_off = F.cross_entropy(logits_off.reshape(-1, 64), ids.reshape(-1))
        opt_off.zero_grad()
        loss_off.backward()

        logits_on = m_on(ids)
        loss_on = F.cross_entropy(logits_on.reshape(-1, 64), ids.reshape(-1))
        opt_on.zero_grad()
        loss_on.backward()

        assert torch.allclose(logits_off, logits_on, atol=1e-5, rtol=1e-4), (
            f"logits diverged: max|diff|={((logits_off-logits_on).abs().max()).item():.2e}"
        )
        assert torch.allclose(loss_off, loss_on, atol=1e-6), (
            f"loss diverged: {loss_off.item():.6f} vs {loss_on.item():.6f}"
        )

        active_name = sched_off.active_block_name
        active_params_off = dict(m_off.named_parameters())
        active_params_on = dict(m_on.named_parameters())
        _, active_list_off = sched_off.blocks[sched_off._active_idx]
        checked = 0
        for n, p_off in m_off.named_parameters():
            if p_off.requires_grad:
                p_on = active_params_on[n]
                assert p_off.grad is not None and p_on.grad is not None, f"missing grad at {n}"
                assert torch.allclose(p_off.grad, p_on.grad, atol=1e-5, rtol=1e-4), (
                    f"gradient mismatch at {n} (active block {active_name!r}): "
                    f"max|diff|={((p_off.grad - p_on.grad).abs().max()).item():.2e}"
                )
                checked += 1
        assert checked > 0, "no active-block gradients were actually checked"

    def test_inactive_blocks_actually_evicted_to_cpu():
        device, _, _, _, m_on, opt_on, sched_on = _make_pair()
        ids = torch.randint(0, 64, (2, 16), device=device)
        logits = m_on(ids)
        loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
        opt_on.zero_grad()
        loss.backward()

        active_name = sched_on.active_block_name
        for name, blk_params in [
            ("token_embedding", list(m_on.token_embedding.parameters())),
            *[(f"blocks.{i}", list(b.parameters())) for i, b in enumerate(m_on.blocks)],
            ("final_norm+decoder",
             list(m_on.final_norm.parameters()) + list(m_on.decoder.parameters())),
        ]:
            for p in blk_params:
                if id(p) in sched_on._pinned:
                    # #157: a parameter reachable from two modules (the tied
                    # embedding/decoder table) is deliberately never evicted --
                    # forward() uses it at both ends, so no single block's
                    # activation window covers both uses. Assert that instead
                    # of skipping, or this exemption could hide a real leak.
                    assert p.device.type == "cuda", (
                        f"pinned parameter in {name!r} was evicted to "
                        f"{p.device}; forward() will fail at the far use")
                    continue
                expected = "cuda" if name == active_name else "cpu"
                assert p.device.type == expected, (
                    f"block {name!r} (active={active_name!r}) on {p.device}, "
                    f"expected {expected} after a full forward+backward step"
                )

    def test_no_pileup_during_forward_when_active_is_early():
        """Regression test for the pileup bug found by scripts/
        bench_offload_vram.py (not a unit test at the time -- a real-config
        peak-memory measurement exceeded this card's 6141 MiB where the
        design should have kept ~1 block resident). Root cause: blocks
        downstream of the active one were kept GPU-resident for the ENTIRE
        forward pass, only evicted via the backward hook -- so when the
        active block is early, most of the model piles up before anything
        frees. Fix: checkpointed blocks are evicted immediately after their
        OWN (call #1, discarded) forward, regardless of position relative
        to active; only call #2 (the recompute, only for downstream blocks)
        needs deferred eviction. This test checks memory state DURING
        forward (before backward runs at all) with `blocks.0` forced
        active, so several real downstream blocks exist to pile up if the
        bug were still present.
        """
        device = torch.device("cuda")
        num_layers = 5
        torch.manual_seed(0)
        model = _tiny_model(num_layers, grad_checkpoint=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=1,
            offload_inactive_weights=True, device=device,
        )
        # force "blocks.0" active (index 1 in the partition: 0=embedding,
        # 1=blocks.0, ..., num_layers=blocks.{num_layers-1}, last=tail)
        target_idx = next(i for i, (n, _) in enumerate(sched.blocks) if n == "blocks.0")
        sched._active_idx = target_idx
        sched._activate(target_idx)
        assert sched.active_block_name == "blocks.0"

        ids = torch.randint(0, 64, (2, 16), device=device)
        logits = model(ids)  # forward only -- no backward yet

        # Every CHECKPOINTED layer block except blocks.0 (active) must
        # already be back on CPU -- blocks.1..blocks.{N-1}, downstream of
        # blocks.0, are exactly the case that piled up before this fix.
        # "final_norm+decoder" is excluded: it is NOT checkpointed, so it
        # correctly stays resident until backward whenever it's downstream
        # of active (i.e. essentially always, since it's positionally last)
        # -- that is by design (see _offload_wrap_plain), not a regression,
        # and is a small, fixed-size exception that doesn't scale with depth.
        # Backward recompute prefetch (2026-10-12): the FIRST block backward
        # will recompute is fetched at the end of forward on purpose -- one
        # fixed block, not a pileup. Exactly that one is expected on the GPU.
        order = model._offload_recompute_order
        assert order, "downstream blocks exist, so a recompute list must too"
        ahead = order[-1][0]
        assert ahead == f"blocks.{num_layers - 1}"
        for name, params in sched.blocks:
            if name == "final_norm+decoder":
                continue
            expected = "cuda" if name in ("blocks.0", ahead) else "cpu"
            for p in params:
                if _pinned_ok(sched, p, name):
                    continue
                assert p.device.type == expected, (
                    f"block {name!r} on {p.device} immediately after forward "
                    f"(before backward) -- expected {expected}. Downstream "
                    f"pileup regression."
                )

        # sanity: finish the step so nothing is left half-configured
        loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    def test_lru_cache_hit_avoids_reload_and_preserves_parity():
        """Issue #103: with offload_cache_blocks > 1, a downstream block's
        call #2 (checkpoint recompute) should find its buffer still
        resident (cache hit -- GPUBufferPool.load is a no-op re-copy) instead
        of a guaranteed miss. Verified two ways: (a) directly, by patching
        GPUBufferPool.load to count hits vs misses per key, since nothing
        else observable distinguishes a hit from a redundant-but-harmless
        re-copy; (b) gradient parity still holds against the cache_blocks=1
        (old-behavior) config -- caching must not change any value, only
        which buffer object a param temporarily points at.
        """
        from hvn.core.block_coordinate import GPUBufferPool

        device = torch.device("cuda")
        torch.manual_seed(0)
        model = _tiny_model(num_layers=4, grad_checkpoint=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=2,
            offload_inactive_weights=True, device=device,
            offload_cache_blocks=4,  # enough room for every non-active block
        )

        pool = sched.buffer_pool
        hits = {"n": 0}
        misses = {"n": 0}
        orig_load = GPUBufferPool.load

        def _counting_load(self, key, params):
            was_resident = key in self._resident
            orig_load(self, key, params)
            (hits if was_resident else misses)["n"] += 1

        pool.load = _counting_load.__get__(pool, GPUBufferPool)

        torch.manual_seed(42)
        ids = torch.randint(0, 64, (2, 16), device=device)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
        opt.zero_grad()
        loss.backward()

        assert hits["n"] > 0, (
            "expected at least one cache hit (a downstream block's call #2 "
            "reusing call #1's still-resident buffer) with offload_cache_blocks=4"
        )

        # Gradient parity against the old cache_blocks=1 behavior.
        torch.manual_seed(0)
        model_base = _tiny_model(num_layers=4, grad_checkpoint=True).to(device)
        opt_base = torch.optim.AdamW(model_base.parameters(), lr=1e-2)
        sched_base = BlockCoordinateScheduler(
            model_base, opt_base, steps_per_block=2,
            offload_inactive_weights=True, device=device,
        )
        assert sched_base.active_block_name == sched.active_block_name
        torch.manual_seed(42)
        ids2 = torch.randint(0, 64, (2, 16), device=device)
        logits_base = model_base(ids2)
        loss_base = F.cross_entropy(logits_base.reshape(-1, 64), ids2.reshape(-1))
        opt_base.zero_grad()
        loss_base.backward()

        assert torch.allclose(loss, loss_base, atol=1e-6), (
            f"loss diverged between cache_blocks=4 and cache_blocks=1: "
            f"{loss.item():.6f} vs {loss_base.item():.6f}"
        )

    def test_prefetch_is_invoked_and_preserves_gradient_parity():
        """Issue #115: offload_prefetch=True (the default) overlaps the
        NEXT offloadable segment's CPU->GPU transfer with the CURRENT
        segment's compute, via a side CUDA stream + event. Verified two
        ways, mirroring #103's LRU-cache test pattern: (a) directly, by
        patching GPUBufferPool.prefetch to count real invocations (proving
        the forward-loop wiring actually calls it, not just that nothing
        crashes); (b) gradient parity against offload_prefetch=False --
        this must be a pure scheduling/timing change, not a numerical one.
        """
        from hvn.core.block_coordinate import GPUBufferPool

        device = torch.device("cuda")
        torch.manual_seed(0)
        model = _tiny_model(num_layers=4, grad_checkpoint=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=2,
            offload_inactive_weights=True, device=device,
            # offload_prefetch defaults to True -- explicit here for clarity.
            offload_prefetch=True,
        )

        pool = sched.buffer_pool
        calls = {"n": 0}
        orig_prefetch = GPUBufferPool.prefetch

        def _counting_prefetch(self, key, params, behind=False):
            calls["n"] += 1
            orig_prefetch(self, key, params, behind=behind)

        pool.prefetch = _counting_prefetch.__get__(pool, GPUBufferPool)

        torch.manual_seed(42)
        ids = torch.randint(0, 64, (2, 16), device=device)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
        opt.zero_grad()
        loss.backward()

        assert calls["n"] > 0, (
            "offload_prefetch=True but GPUBufferPool.prefetch was never "
            "called during a real forward pass -- the forward-loop wiring "
            "is not actually invoking it"
        )

        # Gradient parity against offload_prefetch=False.
        torch.manual_seed(0)
        model_np = _tiny_model(num_layers=4, grad_checkpoint=True).to(device)
        opt_np = torch.optim.AdamW(model_np.parameters(), lr=1e-2)
        sched_np = BlockCoordinateScheduler(
            model_np, opt_np, steps_per_block=2,
            offload_inactive_weights=True, device=device,
            offload_prefetch=False,
        )
        assert sched_np.active_block_name == sched.active_block_name
        torch.manual_seed(42)
        ids2 = torch.randint(0, 64, (2, 16), device=device)
        logits_np = model_np(ids2)
        loss_np = F.cross_entropy(logits_np.reshape(-1, 64), ids2.reshape(-1))
        opt_np.zero_grad()
        loss_np.backward()

        assert torch.allclose(loss, loss_np, atol=1e-6), (
            f"loss diverged between offload_prefetch=True and False: "
            f"{loss.item():.6f} vs {loss_np.item():.6f} -- prefetch must be "
            f"a pure scheduling change"
        )

    def test_prefetch_pinned_memory_bounded_at_depth():
        """Issue #118: GPUBufferPool.prefetch() used to pin each block's
        ACTUAL CPU home tensor directly and leave it pinned until that
        block's next BCD activate/deactivate cycle -- since one forward
        pass prefetches EVERY block (not just the BCD-active one), pinned
        allocations grew with total LAYER COUNT, OOMing the OS/driver's
        pinned-memory ceiling at real depth (confirmed: ~8 GiB pinned
        partway through one forward pass at 95 layers / ~4B params -- see
        the issue for the full repro). This test targets the actual root
        cause directly and fast: many (30) TINY layers, so the bug (pinned
        allocation count scaling with layer count) is exercised without
        needing gigabytes -- a real depth stress test would be too slow
        for routine test runs, this isolates the same mechanism cheaply.

        Asserts the number of DISTINCT pinned buffers ever allocated stays
        bounded by (num_distinct_shapes x max_per_shape), not by layer
        count -- the old bug would have shown one pinned allocation per
        layer (30), scaling linearly with depth.
        """
        from hvn.core.block_coordinate import _PinnedStagingPool

        device = torch.device("cuda")
        torch.manual_seed(0)
        model = _tiny_model(num_layers=30, grad_checkpoint=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=2,
            offload_inactive_weights=True, device=device,
        )

        pool = sched.buffer_pool
        real_allocs = {"n": 0}
        orig_pin_acquire = _PinnedStagingPool.acquire

        def _counting_acquire(self, shape, dtype):
            sig = (tuple(shape), dtype)
            free = self._free.get(sig)
            if not free:
                in_use = self._in_use.get(sig, [])
                if len(in_use) < self.max_per_shape:
                    real_allocs["n"] += 1  # about to hit the torch.empty(...).pin_memory() branch
            return orig_pin_acquire(self, shape, dtype)

        pool._pinned_pool.acquire = _counting_acquire.__get__(pool._pinned_pool, _PinnedStagingPool)

        ids = torch.randint(0, 64, (2, 16), device=device)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        num_shapes = len({sig for _, params in sched.blocks for sig in [pool._signature(params)]})
        # A block partitions into several params of a FEW distinct shapes
        # shared across all 30 layers -- bound is small and flat, not 30x.
        max_expected = num_shapes * pool._pinned_pool.max_per_shape * 4  # generous slack, still << 30
        assert real_allocs["n"] > 0, "no pinned buffers were ever allocated -- prefetch may not be firing"
        assert real_allocs["n"] < 30, (
            f"pinned allocation count ({real_allocs['n']}) scales with layer count (30) -- "
            f"this is the #118 bug: pinning isn't bounded, it's growing with depth"
        )
        assert real_allocs["n"] <= max_expected, (
            f"pinned allocation count ({real_allocs['n']}) exceeds the expected bounded "
            f"ceiling ({max_expected}) for {num_shapes} distinct shapes"
        )

    def test_prefetch_training_converges():
        """Issue #115: a full multi-cycle training loop with the default
        offload_prefetch=True must still converge, and must still retain
        cross-cycle training correctly (the #117 gate), not just avoid
        crashing."""
        device = torch.device("cuda")
        torch.manual_seed(0)
        model = _tiny_model(num_layers=3, grad_checkpoint=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=3,
            offload_inactive_weights=True, device=device,
        )

        torch.manual_seed(1)
        ids = torch.randint(0, 64, (2, 8), device=device)

        losses = []
        n_steps = sched.num_blocks * 3 * 3
        for _ in range(n_steps):
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())

        early_mean = sum(losses[:10]) / 10
        late_mean = sum(losses[-10:]) / 10
        assert late_mean < early_mean, (
            f"loss did not decrease with prefetch enabled (default): "
            f"early={early_mean:.4f} late={late_mean:.4f}"
        )

    def test_cache_reconciled_on_block_switch():
        """Issue #103: BlockCoordinateScheduler._activate's evict_all() call
        must flush the LRU cache before its own per-switch CPU-move loop --
        otherwise a still-cached block's buffer gets silently orphaned (see
        GPUBufferPool.evict_all's docstring). Checks device residency is
        exactly right (only the newly active block on GPU) after a real
        switch, with caching enabled."""
        device = torch.device("cuda")
        torch.manual_seed(0)
        model = _tiny_model(num_layers=4, grad_checkpoint=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=1,
            offload_inactive_weights=True, device=device,
            offload_cache_blocks=4,
        )

        ids = torch.randint(0, 64, (2, 16), device=device)
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()  # triggers a real block switch

        active_name = sched.active_block_name
        for name, blk_params in [
            ("token_embedding", list(model.token_embedding.parameters())),
            *[(f"blocks.{i}", list(b.parameters())) for i, b in enumerate(model.blocks)],
            ("final_norm+decoder",
             list(model.final_norm.parameters()) + list(model.decoder.parameters())),
        ]:
            for p in blk_params:
                if _pinned_ok(sched, p, name):
                    continue
                expected = "cuda" if name == active_name else "cpu"
                assert p.device.type == expected, (
                    f"block {name!r} (active={active_name!r}) on {p.device} "
                    f"after a block switch with caching enabled, expected {expected}"
                )

    def test_deactivated_block_retains_training_across_cycles():
        """Issue #117 (P0): a block's trained values must survive a
        deactivate -> reactivate round trip, not reset to their pristine
        pre-training value. `test_training_converges_with_offload` (below)
        only checks aggregate early-vs-late loss over a run, which still
        trends down even under the bug this test targets (the *currently*
        active block always shows real local descent within its own window,
        every cycle, restarting from the same discarded point) -- so this
        test checks a SPECIFIC block's parameter value directly across two
        full cycles instead of relying on the aggregate loss trend.

        Root cause this guards against: `GPUBufferPool.register_cpu_home`'s
        register-if-absent dedup meant a block's `_cpu_home` entry was
        captured once ever (its untrained init value) and never refreshed
        on later deactivations, so every subsequent `load()` (used to fetch
        an inactive block for a forward pass) fetched stale, untrained
        weights -- silently discarding all training. Fixed via
        `register_cpu_home(params, force=True)` in the deactivation loop.
        """
        device = torch.device("cuda")
        torch.manual_seed(0)
        model = _tiny_model(num_layers=3, grad_checkpoint=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=3,
            offload_inactive_weights=True, device=device,
        )

        torch.manual_seed(1)
        ids = torch.randint(0, 64, (2, 8), device=device)

        target_name, target_params = sched.blocks[1]  # "blocks.0"
        assert target_name == "blocks.0"
        p0 = target_params[4]  # a real (d, d) weight matrix, not a norm gain
        init_value = p0.data.detach().clone().cpu()

        def value_diff() -> float:
            return (p0.data.detach().cpu() - init_value).abs().max().item()

        diffs_after_each_cycle = []
        n_steps = sched.num_blocks * 3 * 2  # 2 full cycles
        for step in range(n_steps):
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            # Sample right after blocks.0's own active window ends each cycle
            # (steps 5 and 5 + num_blocks*3, i.e. the step right after its
            # 3rd active step, once it has gone back to being inactive).
            if step in (5, 5 + sched.num_blocks * 3):
                diffs_after_each_cycle.append(value_diff())

        assert len(diffs_after_each_cycle) == 2
        cycle1_diff, cycle2_diff = diffs_after_each_cycle
        assert cycle1_diff > 1e-6, (
            "blocks.0 shows no change from its initial value after its own "
            "active window -- training isn't happening at all, a different "
            "bug than #117."
        )
        assert cycle2_diff > cycle1_diff - 1e-9, (
            f"blocks.0's value did not accumulate further training in cycle "
            f"2 (cycle1_diff={cycle1_diff:.6f}, cycle2_diff={cycle2_diff:.6f}) "
            f"-- this is the #117 bug: the block reset to its pre-training "
            f"value on deactivation instead of retaining cycle 1's training."
        )

    def test_bf16_parking_shrinks_inactive_blocks_and_restores_native_dtype_when_active():
        """Issue #113: offload_park_dtype=torch.bfloat16 stores INACTIVE
        blocks' weights in bf16 on CPU (halving CPU RAM / PCIe transfer vs
        fp32), while the ACTIVE block is always restored to its native
        dtype (fp32 here) before training resumes on it -- the optimizer's
        precision must be unaffected, only frozen weights get coarser."""
        device = torch.device("cuda")
        torch.manual_seed(0)
        model = _tiny_model(num_layers=3, grad_checkpoint=True).to(device)
        assert all(p.dtype == torch.float32 for p in model.parameters()), (
            "test assumes fp32-native params (matches docs/training.md's "
            "measured EdgeHVN default) -- if this ever changes, this test's "
            "premise (bf16 parking actually shrinks storage) needs revisiting"
        )
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=2,
            offload_inactive_weights=True, device=device,
            offload_park_dtype=torch.bfloat16,
        )

        def _expected(name, p):
            # 1-D params (biases, norm gains) always park at native dtype --
            # see test_bf16_parking_keeps_1d_params_trainable.
            if name == sched.active_block_name or p.dim() <= 1:
                return torch.float32
            return torch.bfloat16

        for name, params in sched.blocks:
            for p in params:
                expected = _expected(name, p)
                assert p.dtype == expected, (
                    f"block {name!r} (active={sched.active_block_name!r}) is "
                    f"{p.dtype}, expected {expected}"
                )

        # Cycle through every block once; each must come back to fp32 the
        # instant it's active, and go back to bf16 the instant it parks.
        for _ in range(sched.num_blocks):
            sched.step()
            for name, params in sched.blocks:
                for p in params:
                    expected = _expected(name, p)
                    if id(p) in getattr(sched, "_pinned", ()):
                        # #157: a pinned (two-module) parameter is never parked,
                        # so it keeps its native dtype. Assert that rather than
                        # skipping -- a pinned param silently demoted to bf16
                        # would change the tied table's precision for the whole
                        # run with nothing else failing.
                        assert p.dtype == torch.float32, (
                            f"pinned parameter in {name!r} was parked to "
                            f"{p.dtype}; it is never evicted and must keep its "
                            f"native dtype")
                        continue
                    assert p.dtype == expected, (
                        f"after a switch, block {name!r} "
                        f"(active={sched.active_block_name!r}) is {p.dtype}, "
                        f"expected {expected}"
                    )

    def test_bf16_parking_requires_autocast():
        """Issue #113: forwarding through a bf16-parked block alongside an
        fp32-active block mixes dtypes -- PyTorch ops like Conv1d reject
        this outside autocast with a cryptic low-level error ('Input type
        and bias type should be the same'). Confirmed directly: the same
        forward call that raises this RuntimeError's fail-fast message
        below, with the guard removed, instead crashes several frames deep
        inside torch.nn.modules.conv. Must fail fast with a clear message
        instead, before backward, not mid-conv."""
        device = torch.device("cuda")
        torch.manual_seed(0)
        model = _tiny_model(num_layers=3, grad_checkpoint=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
        BlockCoordinateScheduler(
            model, opt, steps_per_block=2,
            offload_inactive_weights=True, device=device,
            offload_park_dtype=torch.bfloat16,
        )
        ids = torch.randint(0, 64, (2, 16), device=device)
        try:
            model(ids)
            raise AssertionError("expected RuntimeError: missing autocast")
        except RuntimeError as e:
            assert "Issue #113" in str(e), f"wrong/unclear error raised: {e}"

        # With autocast, the same call must succeed.
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(ids)
        assert logits.shape == (2, 16, 64)

    def test_training_converges_with_bf16_parking():
        """Issue #113: a full multi-cycle training loop with bf16 parking
        enabled (under autocast, as required) must still converge -- not
        just avoid crashing. Companion to
        test_deactivated_block_retains_training_across_cycles (which
        proves cross-cycle retention at the default park_dtype=None) --
        this confirms retention ALSO holds when parked weights are
        actually downcast, not just device-moved."""
        device = torch.device("cuda")
        torch.manual_seed(0)
        model = _tiny_model(num_layers=3, grad_checkpoint=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-2)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=3,
            offload_inactive_weights=True, device=device,
            offload_park_dtype=torch.bfloat16,
        )

        torch.manual_seed(1)
        ids = torch.randint(0, 64, (2, 8), device=device)

        losses = []
        n_steps = sched.num_blocks * 3 * 3
        for _ in range(n_steps):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(ids)
                loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())

        early_mean = sum(losses[:10]) / 10
        late_mean = sum(losses[-10:]) / 10
        assert late_mean < early_mean, (
            f"loss did not decrease with bf16 parking enabled: "
            f"early={early_mean:.4f} late={late_mean:.4f}"
        )

    def test_bf16_parking_keeps_1d_params_trainable():
        """Each park rounds the block to the park dtype with no fp32 master,
        so a tenure whose total update is under half a bf16 ulp is erased.
        DeltaNet's alpha_proj.bias starts at 3.0 (ulp 2^-6): with bf16-parked
        1-D params it stayed at exactly 3.0 in every block across a whole run.
        1-D params must park at native dtype and keep their training."""
        device = torch.device("cuda")
        torch.manual_seed(0)
        model = _tiny_model(num_layers=3, grad_checkpoint=True).to(device)
        lr, spb = 1e-4, 3
        # Non-vacuity: one tenure's largest possible AdamW move (~lr per step)
        # must be below half a bf16 ulp at 3.0, or bf16 parking would let the
        # bias move too and this test could not fail.
        assert lr * spb < 2.0 ** -7
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=spb,
            offload_inactive_weights=True, device=device,
            offload_park_dtype=torch.bfloat16,
        )
        biases = {n: p for n, p in model.named_parameters() if n.endswith("alpha_proj.bias")}
        assert len(biases) == 3
        init = {n: p.detach().float().cpu().clone() for n, p in biases.items()}

        torch.manual_seed(1)
        ids = torch.randint(0, 64, (2, 16), device=device)
        for _ in range(sched.num_blocks * spb * 3):  # 3 full cycles
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = F.cross_entropy(model(ids).reshape(-1, 64), ids.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            # Parking must still shrink matrices -- the fix is 1-D only.
            for name, params in sched.blocks:
                if name == sched.active_block_name:
                    continue
                for p in params:
                    if id(p) not in sched._pinned and p.dim() >= 2:
                        assert p.dtype == torch.bfloat16, f"{name} matrix parked as {p.dtype}"

        for n, p in biases.items():
            moved = (p.detach().float().cpu() - init[n]).abs().max().item()
            assert moved > 1e-4, (
                f"{n} moved only {moved:.2e} after 3 cycles -- its training is "
                f"being rounded away on every bf16 park")

    def test_training_converges_with_offload():
        torch.manual_seed(0)
        device = torch.device("cuda")
        model = _tiny_model(num_layers=3, grad_checkpoint=True).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=5e-3)
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=5,
            offload_inactive_weights=True, device=device,
        )

        torch.manual_seed(1)
        ids = torch.randint(0, 64, (4, 16), device=device)

        losses = []
        n_steps = sched.num_blocks * 5 * 3  # 3 full cycles
        for _ in range(n_steps):
            logits = model(ids)
            loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())

        early_mean = sum(losses[:10]) / 10
        late_mean = sum(losses[-10:]) / 10
        assert late_mean < early_mean, (
            f"loss did not decrease with offload enabled: early={early_mean:.4f} "
            f"late={late_mean:.4f}"
        )


# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed"
          f"{' (CUDA unavailable, correctness tests skipped)' if not _HAS_CUDA else ''}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())


def test_prefetch_does_not_corrupt_activations_at_depth():
    """The prefetch side stream must not write into memory the main stream is
    still using (2026-08-27).

    `GPUBufferPool.prefetch` ordered the MAIN stream after the prefetch event
    but never ordered the PREFETCH stream after the main one. Its buffers come
    from `torch.empty` on the main stream, and the caching allocator recycles
    blocks freed by activation tensors whose kernels are still in flight -- so
    the H2D weight copy overwrote live activations and the forward went NaN
    from about the third block onward, silently, while the step still ran.

    WHY THE EXISTING PREFETCH TEST CANNOT CATCH THIS
    ------------------------------------------------
    `test_prefetch_is_invoked_and_preserves_gradient_parity` uses a 4-layer
    model at 2x16 tokens, fully GPU-resident. Its kernels retire before the
    host can run ahead, so the race window never opens and it passes against
    the broken code. This test exists specifically to fail there: enough depth
    and sequence length that the host outruns the GPU, plus an explicit
    finiteness assertion rather than a parity one.

    Deliberately asserts `torch.isfinite`, NOT closeness to a reference: the
    failure mode is NaN, and a NaN-vs-NaN comparison silently passes
    (`abs(nan - nan) > tol` is False). That vacuous-check trap is exactly how
    the bug survived a probe that was written to detect it.

    THIS TEST IS PROBABILISTIC, AND THAT IS STATED RATHER THAN HIDDEN.
    Measured against the broken code on 2026-08-27: a single fresh construction
    catches it roughly 1 time in 3 (observed 1/4 and 1/6 across runs, and only
    ever on the FIRST forward -- afterwards the buffers are resident and the
    prefetch path returns early). Seeding the allocator with freed blocks first
    did not make it reliable. It is a real race, so it cannot be made
    deterministic from the outside.

    Consequence, and do not lose this: **a single PASS is not proof the bug is
    absent.** Three fresh attempts below put the catch rate around 70%. If this
    test ever needs to prove the fix, remove the `wait_stream` line in
    `GPUBufferPool.prefetch` and run it several times -- it should fail most of
    them. The fix itself rests on the controlled evidence in the commit
    message, not on this test alone.
    """
    if not torch.cuda.is_available():
        return

    import torch.nn as nn
    from hvn.edge.model import EdgeHVN
    from hvn.core.block_coordinate import BlockCoordinateScheduler

    # THE CONFIG IS THE TEST. Measured 2026-08-27 against the broken code:
    #   16 layers, d=256,  seq 1024 -> 0 non-finite   (passes: does NOT catch it)
    #   24 layers, d=512,  seq 2048 -> 0 non-finite   (passes: does NOT catch it)
    #   24 layers, d=1216, seq 4096 -> 1,867,776 of 4,980,736 non-finite  CATCHES IT
    # The race needs enough kernel duration for the host to run ahead of the
    # GPU, so a cheap small-model proxy cannot reach it -- which is exactly why
    # the existing prefetch test passed against the bug for weeks. Do not shrink
    # this config to make the suite faster; it would silently stop testing.
    torch.manual_seed(0)
    n_layers, seq = 24, 4096
    model = EdgeHVN(
        vocab_size=512, d_model=1216, num_layers=n_layers, n_heads=19,
        block_plan=["deltanet"] * n_layers, grad_checkpoint=True,
        tie_embeddings=False,
    ).to("cpu")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    BlockCoordinateScheduler(
        model, opt, steps_per_block=50, device="cuda",
        offload_inactive_weights=True, offload_prefetch=True,
    )

    x = torch.randint(0, 512, (1, seq), device="cuda")
    y = torch.randint(0, 512, (1, seq), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        hidden = model(x, return_hidden=True)
        hidden = hidden[0] if isinstance(hidden, tuple) else hidden

    n_bad = int((~torch.isfinite(hidden)).sum())
    assert n_bad == 0, (
        f"{n_bad} of {hidden.numel()} hidden values are non-finite -- the "
        f"prefetch stream is writing over memory the main stream still holds")

    logits = model.decoder(hidden)
    loss = nn.functional.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
    assert torch.isfinite(loss), f"loss is {loss.item()}, not finite"
    loss.backward()
    assert all(torch.isfinite(p.grad).all()
               for p in model.parameters() if p.grad is not None), \
        "non-finite gradients after a prefetched offload forward"


def test_pinned_staging_overflow_never_double_frees():
    """#164 (found while verifying it): `_PinnedStagingPool.acquire`'s overflow
    branch hands out a buffer an earlier acquire still holds -- within one
    prefetch, when a block has more same-shaped weights than the cap. Every
    holder used to release it, so it entered the free list more than once, and
    the next prefetch got the SAME pinned buffer for two weights. Deterministic
    check of the bookkeeping; the end-to-end value check is the next test."""
    if not torch.cuda.is_available():
        return
    from hvn.core.block_coordinate import _PinnedStagingPool
    pool = _PinnedStagingPool(max_per_shape=2)
    held = [pool.acquire(torch.Size([8, 8]), torch.float32) for _ in range(6)]
    assert len({id(b) for b in held}) == 2, "overflow branch not exercised"
    for b in held:
        pool.release(b)
    free = pool._free[((8, 8), torch.float32)]
    assert len(free) == len({id(b) for b in free}) == 2, (
        f"free list holds {len(free)} entries for "
        f"{len({id(b) for b in free})} buffers -- a buffer was freed twice")
    again = [pool.acquire(torch.Size([8, 8]), torch.float32) for _ in range(2)]
    assert again[0] is not again[1], "same pinned buffer handed out twice"


def _offload_logit_drift(reserve_enabled: bool) -> float:
    """Max |logits(offload) - logits(no offload)| over 4 forward+backward
    passes of an identical model. Weights never change (lr=0), so any
    difference is the offload path computing with wrong weights."""
    from hvn.core.block_coordinate import _PinnedStagingPool
    d, L, S, V = 512, 8, 1024, 4096

    def mk():
        torch.manual_seed(0)
        return EdgeHVN(vocab_size=V, d_model=d, num_layers=L, n_heads=16,
                       block_plan=["deltanet"] * L, grad_checkpoint=True,
                       tie_embeddings=False, dropout=0.0).to("cuda")

    torch.manual_seed(1)
    ids = torch.randint(0, V, (1, S), device="cuda")
    ref = mk()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ref_logits = ref(ids).float()
    del ref
    model = mk()
    opt = torch.optim.AdamW(model.parameters(), lr=0.0)
    sched = BlockCoordinateScheduler(model, opt, steps_per_block=1000,
                                     offload_inactive_weights=True, device="cuda")
    if not reserve_enabled:
        # Force every prefetch through acquire()'s overflow branch, so that
        # branch's own correctness is tested, not just avoided.
        sched.buffer_pool._pinned_pool.reserve = lambda *a: None
    worst = 0.0
    for _ in range(4):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            out = model(ids)
            loss = F.cross_entropy(out.float().reshape(-1, V), ids.reshape(-1))
        loss.backward()
        opt.zero_grad(set_to_none=True)
        worst = max(worst, (out.float() - ref_logits).abs().max().item())
    return worst


def test_prefetched_weights_match_no_offload_at_scale():
    """End-to-end value check for the double-free above. Measured on the
    broken code: max|dlogits| 1.1-2.6 on every one of 6 forwards; 0.0 fixed.
    Unlike test_prefetch_does_not_corrupt_activations_at_depth this compares
    VALUES, not finiteness -- swapped weights are perfectly finite. Needs a
    model whose kernels outlast the host (the prefetch copy queues behind the
    main stream); the 32-wide test models catch it only ~5% of steps."""
    if not torch.cuda.is_available():
        return
    for reserve_enabled in (True, False):
        drift = _offload_logit_drift(reserve_enabled)
        assert drift < 1e-3, (
            f"offloaded forward differs from the resident one by {drift:.3f} "
            f"(reserve={'on' if reserve_enabled else 'off'}) -- prefetched "
            f"weights are not the weights")


# ---------------------------------------------------------------------------
# Per-buffer release events for the forward prefetch (2026-10-12).
#
# The prefetch used to make the side stream wait for the WHOLE main stream, so
# the next block's weight copy waited for the current block's compute to
# finish (0% overlap, measured at 2048x24). It now waits only for the release
# event of the specific buffer set it reuses (fresh allocations keep the
# blanket wait -- the 2026-08-27 allocator race). These tests cover both sides:
# the copy must not overwrite a set something still reads, and it must not be
# serialized behind unrelated main-stream work.
# ---------------------------------------------------------------------------

def _stress_blocks():
    import torch.nn as nn
    blocks = {}
    for k in range(6):
        shape = (256, 256) if k % 3 else (128, 512)      # two shape signatures
        blocks[f"b{k}"] = ([nn.Parameter(torch.full(shape, float(k + 1)), requires_grad=False)
                            for _ in range(3)], float(k + 1))
    return blocks


def _pool_stress(*, broken: bool, cache_blocks: int, seed: int, iters: int = 160,
                 sleep_cycles: int = 3_000_000):
    """Drive GPUBufferPool through randomized block switches while the main
    stream is held far behind the host by `torch.cuda._sleep`, then check every
    read saw its own block's weights.

    Each iteration: (maybe) prefetch, load, sleep on the main stream, READ the
    weights on the main stream, then release by one of the real patterns --
    immediate evict (forward call #1), or left resident for LRU capacity
    eviction (deferred, like the backward hook). Because the read is queued
    behind a sleep, a prefetch that does not wait for the right release event
    overwrites the set before the read runs. That makes the race deterministic
    instead of the ~1-in-3 the model-level depth test gets.

    Returns (n_bad_reads, n_reads, n_reused_prefetches)."""
    import random
    from hvn.core.block_coordinate import GPUBufferPool
    dev = torch.device("cuda")
    pool = GPUBufferPool(dev, cache_blocks=cache_blocks)
    blocks = _stress_blocks()
    for params, _ in blocks.values():
        pool.register_cpu_home(params)
    reused = [0]
    orig = pool._order_prefetch_after

    def _order(ready):
        if ready is not pool.FRESH:
            reused[0] += 1
            if broken:
                return               # the bug under test: skip the per-set wait
        orig(ready)
    pool._order_prefetch_after = _order

    rng = random.Random(seed)
    keys = list(blocks)
    reads, prev = [], None
    for _ in range(iters):
        key = rng.choice([k for k in keys if k != prev])
        prev = key
        params, val = blocks[key]
        if rng.random() < 0.75:
            pool.prefetch(key, params)
        pool.load(key, params)
        torch.cuda._sleep(sleep_cycles)
        reads.append(torch.stack([(p.data == val).all() for p in params]).all())
        if rng.random() < 0.5:
            pool.evict(key)
    pool.evict_all()
    torch.cuda.synchronize()
    bad = sum(1 for r in reads if not bool(r))
    return bad, len(reads), reused[0]


def test_prefetch_release_events_survive_randomized_stress():
    """No read may ever see another block's weights, across cache sizes and
    seeds, with the main stream deliberately lagging."""
    if not torch.cuda.is_available():
        return
    for cache_blocks in (1, 2, 3):
        for seed in (0, 1):
            bad, n, reused = _pool_stress(broken=False, cache_blocks=cache_blocks, seed=seed)
            assert reused > 20, (
                f"cache={cache_blocks} seed={seed}: only {reused} prefetches reused a "
                f"set -- the stress is not exercising the release-event path")
            assert bad == 0, (
                f"cache={cache_blocks} seed={seed}: {bad}/{n} reads saw the wrong "
                f"weights -- a prefetch overwrote a buffer still being read")


def test_prefetch_stress_catches_a_missing_release_wait():
    """Anti-vacuity: the stress above must FAIL when the per-set wait is
    skipped. Without this, it could pass because the race window never opened."""
    if not torch.cuda.is_available():
        return
    bad, n, reused = _pool_stress(broken=True, cache_blocks=1, seed=0)
    assert reused > 20
    assert bad > 0, (
        f"skipping the release-event wait corrupted 0/{n} reads -- the stress "
        f"test cannot see the race it exists to catch")


def test_prefetch_free_sets_are_fifo_and_bounded():
    """Prefetch takes the OLDEST free set, and only when two are free;
    otherwise it allocates. LIFO handed out the set the current block had just
    released, whose event is after that block's compute -- serializing again."""
    if not torch.cuda.is_available():
        return
    import torch.nn as nn
    from hvn.core.block_coordinate import GPUBufferPool
    pool = GPUBufferPool(torch.device("cuda"), cache_blocks=1)

    def mk(v):
        return [nn.Parameter(torch.full((64, 64), v), requires_grad=False)]
    a, b, c = mk(1.0), mk(2.0), mk(3.0)
    for ps in (a, b, c):
        pool.register_cpu_home(ps)
    pool.load("a", a)
    set_a = a[0].data.data_ptr()
    pool.evict("a")
    # one free set -> prefetch must allocate, not reuse it
    pool.prefetch("b", b)
    assert b[0].data.data_ptr() != set_a, "prefetch reused the only free set"
    pool.load("b", b)
    pool.evict("b")
    sig = next(iter(pool._free_buffers))
    assert len(pool._free_buffers[sig]) == 2
    # two free -> FIFO: the older one (a's), not b's just-released set
    pool.prefetch("c", c)
    assert c[0].data.data_ptr() == set_a, "prefetch did not take the oldest free set"
    pool.load("c", c)
    torch.cuda.synchronize()
    assert torch.equal(c[0].data.cpu(), torch.full((64, 64), 3.0))
    pool.evict_all()
    total = len(pool._free_buffers[sig])
    assert total == 2, f"{total} sets for one signature -- pool grew past the 2-free bound"


def test_prefetch_copy_overlaps_main_stream_work():
    """The point of the change: a reused set's copy must run WHILE unrelated
    main-stream work is still going. Under the old blanket `wait_stream` this
    fails -- the copy cannot start until the main stream's queued sleep ends."""
    if not torch.cuda.is_available():
        return
    import torch.nn as nn
    from hvn.core.block_coordinate import GPUBufferPool
    pool = GPUBufferPool(torch.device("cuda"), cache_blocks=2)

    def mk(v):
        return [nn.Parameter(torch.full((1024, 1024), v), requires_grad=False)]
    a, b, c = mk(1.0), mk(2.0), mk(3.0)
    for ps in (a, b, c):
        pool.register_cpu_home(ps)
    pool.load("a", a)
    pool.load("b", b)
    pool.evict("a")
    pool.evict("b")                                # two free sets, both released
    torch.cuda.synchronize()

    torch.cuda._sleep(600_000_000)                 # a few hundred ms of main-stream work
    main_done = torch.cuda.Event()
    main_done.record()
    pool.prefetch("c", c)
    pool._pending_events["c"].synchronize()        # host waits for the copy only
    overlapped = not main_done.query()
    torch.cuda.synchronize()
    pool.load("c", c)
    torch.cuda.synchronize()
    assert torch.equal(c[0].data.cpu(), torch.full((1024, 1024), 3.0)), "prefetched wrong values"
    assert overlapped, ("the prefetch copy finished only after the main stream's "
                        "unrelated work -- it is serialized behind compute again")


def test_prefetch_many_block_switches_match_resident_model():
    """End to end, many rapid BCD switches: steps_per_block=1 over two full
    cycles, so every block is active once per cycle and the free-set FIFO is
    driven through a different reuse pattern on each step. lr=0, so every
    step's logits must equal the never-offloaded model's -- values, not
    finiteness (swapped weights are finite). Cache sizes 1 and 2, hybrid plan
    (two shape signatures), tied embeddings (the 1B config).

    WHAT THIS TEST CANNOT DO, measured 2026-10-12: it does not detect the race
    the per-set release wait prevents. With that wait removed it still PASSED
    (4/4 runs), even with a 25 ms `torch.cuda._sleep` queued before every block.
    Instrumented: the host never gets more than ~1 block ahead of the GPU in
    this forward, and FIFO hands the prefetch a set released >= 2 blocks
    earlier, so its release event had already completed in 138/138 reuses --
    there was nothing left to race. The deterministic detector is
    `test_prefetch_release_events_survive_randomized_stress` (and its
    anti-vacuity twin, which proves it fails without the wait). This test
    guards VALUES across block switches and reuse patterns."""
    if not torch.cuda.is_available():
        return
    d, L, S, V = 512, 8, 1024, 4096

    def mk():
        torch.manual_seed(0)
        return EdgeHVN(vocab_size=V, d_model=d, num_layers=L, n_heads=16,
                       block_plan=["deltanet"] * 3 + ["transformer"] + ["deltanet"] * 4,
                       grad_checkpoint=True, tie_embeddings=True, dropout=0.0).to("cuda")

    torch.manual_seed(1)
    ids = torch.randint(0, V, (1, S), device="cuda")
    ref = mk()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        ref_logits = ref(ids).float()
    del ref
    for cache_blocks in (1, 2):
        model = mk()
        opt = torch.optim.SGD(model.parameters(), lr=0.0)
        sched = BlockCoordinateScheduler(model, opt, steps_per_block=1, device="cuda",
                                         offload_inactive_weights=True,
                                         offload_cache_blocks=cache_blocks)
        worst, actives = 0.0, set()
        for _ in range(2 * sched.num_blocks):
            actives.add(sched.active_block_name)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = model(ids)
                loss = F.cross_entropy(out.float().reshape(-1, V), ids.reshape(-1))
            loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            sched.step()
            worst = max(worst, (out.float() - ref_logits).abs().max().item())
        assert len(actives) == sched.num_blocks, "not every block was active"
        assert worst < 1e-3, (
            f"cache={cache_blocks}: offloaded logits differ from resident by "
            f"{worst:.3f} across block switches -- prefetched weights are not the weights")
        del model, opt, sched
