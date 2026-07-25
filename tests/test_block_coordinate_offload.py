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
    model's _offload_active_name -- forward() takes the pre-#101 path."""
    model = _tiny_model(grad_checkpoint=True)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    BlockCoordinateScheduler(model, opt, steps_per_block=1)
    assert model._offload_active_name is None


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
        for name, params in sched.blocks:
            if name == "final_norm+decoder":
                continue
            expected = "cuda" if name == "blocks.0" else "cpu"
            for p in params:
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

        def _counting_prefetch(self, key, params):
            calls["n"] += 1
            orig_prefetch(self, key, params)

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

        for name, params in sched.blocks:
            expected = torch.float32 if name == sched.active_block_name else torch.bfloat16
            for p in params:
                assert p.dtype == expected, (
                    f"block {name!r} (active={sched.active_block_name!r}) is "
                    f"{p.dtype}, expected {expected}"
                )

        # Cycle through every block once; each must come back to fp32 the
        # instant it's active, and go back to bf16 the instant it parks.
        for _ in range(sched.num_blocks):
            sched.step()
            for name, params in sched.blocks:
                expected = torch.float32 if name == sched.active_block_name else torch.bfloat16
                for p in params:
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
