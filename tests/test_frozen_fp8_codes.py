"""#204: fp8 codes of BCD-frozen weights made once per turn, streamed instead of
the bf16 weight (hvn/edge/fp8_fused.py ``refresh_frozen_codes``).

Author: Rambarun Komaljeet

The claim is "bit-identical to quantizing on every use". A cache can only break
that by being STALE, so every equality test here has a must-fail partner that
makes the cache stale on purpose and checks the comparison sees it:

- trajectory over two full BCD rotations (blocks are trained, then frozen with
  new values, so their codes MUST be remade): equal to the per-use route;
  partner: codes made once and never remade -> trajectory differs.
- a frozen weight mutated between turns through ``p.data`` (no version bump,
  the worst case): equal; partner: the per-turn expiry disabled -> differs.
- a frozen weight mutated within a turn through a tracked in-place op: equal;
  partner: the version counter dropped from the key -> differs.
- the active block never reads codes (its weight moves every step).
- the offload-aware held-out eval (bf16, shares the pool buffers) is unchanged.
"""

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as Fn

# Before any CUDA work: pandas/pyarrow after CUDA init crashes this venv
# (see tests/conftest.py).
from train_edge_conversational import apply_fast_stack

import hvn.edge.fp8_fused as F8
from hvn.edge.fp8_fused import fp8_supported

fp8_only = pytest.mark.skipif(not (torch.cuda.is_available() and fp8_supported()),
                              reason="needs CUDA FP8 tensor cores (sm_89+) + Triton")

Dm, Lb, Sb, Vb, GA = 256, 4, 512, 2048, 2


def _build():
    from hvn.core.block_coordinate import BlockCoordinateScheduler
    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model
    from hvn.optimizers import build_optimizer
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=Vb, d_model=Dm, num_layers=Lb, n_heads=8,
                block_plan=["deltanet", "transformer", "deltanet", "deltanet"],
                grad_checkpoint=True, tie_embeddings=True, dropout=0.0)
    names = [n for n, mod in m.named_modules() if isinstance(mod, nn.Linear)
             and not any(p in n for p in ("beta_proj", "alpha_proj", "decoder"))]
    apply_ternary_to_model(m, names)
    apply_fast_stack(m, triton_scan=True, megakernel_scan=True, tf32=True,
                     compile_model=False, verbose=False, fused_ternary=True,
                     ternary_step_cache=True, fused_rmsnorm=True, fp8_linear=True)
    m.train()
    opt, _ = build_optimizer(m, "muon", lr=3e-3, weight_decay=0.01, fused=False,
                             muon_lr=0.05)
    sched = BlockCoordinateScheduler(
        m, opt, steps_per_block=2, device="cuda", offload_inactive_weights=True,
        offload_park_dtype=torch.bfloat16, offload_prefetch=True,
        offload_cache_blocks=1, partition="balanced")
    return m, opt, sched


def _frozen_weight(m):
    """A fp8 ternary weight of a frozen, CPU-parked layer block."""
    from hvn.edge.ternary import TernaryLinear
    for i, blk in enumerate(m.blocks):
        if f"blocks.{i}" in m._offload_active_names:
            continue
        for mod in blk.modules():
            if isinstance(mod, TernaryLinear) and mod.use_fp8 and mod.weight.device.type == "cpu":
                return mod.weight
    raise AssertionError("no frozen parked fp8 weight")


@torch.no_grad()
def _heldout(m, x):
    """scripts/profile_step_202.py heldout(): offload-aware bf16 eval."""
    pool, act = m._offload_pool, m._offload_active_names
    m.eval()

    def seg(name, mods, fn):
        params = [p for q in mods for p in q.parameters()]
        if name in act or not pool._offloadable(params):
            return fn()
        pool.load(name, params)
        out = fn()
        pool.evict(name)
        return out
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        z = seg("token_embedding", [m.token_embedding], lambda: m.token_embedding(x))
        for i, blk in enumerate(m.blocks):
            z = seg(f"blocks.{i}", [blk], lambda: blk(z, pos))
        out = seg("final_norm+decoder", [m.final_norm, m.decoder],
                  lambda: m.decoder(m.final_norm(z))).float().cpu()
    m.train()
    return out


def _run(codes, mutate=None, eval_every=0):
    """Train two full rotations + 2 steps. ``mutate(m, step, switched)`` runs
    after every optimizer step (both arms, same edits)."""
    F8.FROZEN_CODES[0] = codes
    for k in F8.FP8_STATS:
        F8.FP8_STATS[k] = 0
    F8.CODES_STATS.update(refreshed=0, refresh_calls=0, refresh_sec=0.0, home_bytes=0)
    try:
        m, opt, sched = _build()
        n_steps = 2 * 2 * sched.num_blocks + 2
        g = torch.Generator().manual_seed(1)
        ids = torch.randint(0, Vb, (n_steps, GA, 1, Sb), generator=g)
        losses, evals = [], []
        for s in range(n_steps):
            for mb in range(GA):
                x = ids[s, mb].cuda()
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    out = m(x)
                    loss = Fn.cross_entropy(out.float().reshape(-1, Vb), x.reshape(-1)) / GA
                loss.backward()
                losses.append(loss.item())
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            switched = sched.step()
            if mutate is not None:
                mutate(m, s, switched)
            if eval_every and s % eval_every == 0:
                evals.append(_heldout(m, ids[s, 0].cuda()))
        params = [p.detach().float().cpu().clone() for p in m.parameters()]
        return losses, params, evals, dict(F8.FP8_STATS), dict(F8.CODES_STATS)
    finally:
        F8.FROZEN_CODES[0] = True


def _same(a, b):
    la, pa, ea = a[:3]
    lb, pb, eb = b[:3]
    return (la == lb and all(torch.equal(x, y) for x, y in zip(pa, pb))
            and len(ea) == len(eb) and all(torch.equal(x, y) for x, y in zip(ea, eb)))


@pytest.fixture(scope="module")
def per_use():
    return _run(codes=False, eval_every=3)


@fp8_only
def test_codes_trajectory_bit_identical(per_use):
    on = _run(codes=True, eval_every=3)
    lr, pr, er, sr, _ = per_use
    lf, pf, ef, sf, cf = on
    assert lr == lf, f"loss trajectory moved:\n per-use {lr}\n codes   {lf}"
    for i, (a, b) in enumerate(zip(pr, pf)):
        assert torch.equal(a, b), f"param {i} diverged"
    for i, (a, b) in enumerate(zip(er, ef)):
        assert torch.equal(a, b), f"held-out logits {i} differ"
    # Not vacuous: the codes route ran, and the per-use quantize mostly did not.
    assert sf["codes_hits"] > 0 and sr["codes_hits"] == 0, (sf, sr)
    assert cf["refreshed"] > 0 and cf["home_bytes"] > 0, cf
    assert sf["frozen_fused"] == cf["refreshed"], (sf, cf)   # quantized only at refresh
    assert sr["frozen_fused"] > 4 * sf["frozen_fused"], (sr, sf)
    # Remade once per turn (plus the first), not once per micro-batch.
    assert cf["refresh_calls"] <= 2 * 2 * 7 + 2, cf


@fp8_only
def test_codes_never_remade_is_caught(per_use, monkeypatch):
    """Must-fail partner: codes made once and never remade. Blocks are trained
    and then frozen with new values, so this cache serves old codes."""
    monkeypatch.setattr(F8, "_codes_key", lambda mod: ("never-stale",))
    stale = _run(codes=True, eval_every=3)
    assert not _same(per_use, stale), "a never-refreshed cache matched: the test is blind"


def _mutate_between_turns(m, s, switched):
    if switched and s in (3, 9):
        _frozen_weight(m).data.mul_(-1)      # through .data: no version bump


@fp8_only
def test_mutation_between_turns_is_seen():
    a = _run(codes=False, mutate=_mutate_between_turns)
    b = _run(codes=True, mutate=_mutate_between_turns)
    assert _same(a, b)


@fp8_only
def test_mutation_between_turns_needs_the_turn_expiry(monkeypatch):
    """Must-fail partner: without the per-turn expiry, a ``p.data`` write leaves
    pointer and version unchanged, so the codes go stale."""
    a = _run(codes=False, mutate=_mutate_between_turns)
    monkeypatch.setattr(F8, "_bump_codes_turn", lambda *x, **k: None)
    import hvn.core.block_coordinate as bc
    monkeypatch.setattr(bc, "_ACTIVATE_CALLBACKS",
                        [cb for cb in bc._ACTIVATE_CALLBACKS
                         if getattr(cb, "__name__", "") != "_bump_codes_turn"])
    b = _run(codes=True, mutate=_mutate_between_turns)
    assert not _same(a, b), "stale codes after a between-turn write were not seen"


def _mutate_within_turn(m, s, switched):
    if not switched and s in (2, 8):
        with torch.no_grad():
            _frozen_weight(m).mul_(-1)       # tracked: bumps the version counter


@fp8_only
def test_mutation_within_turn_is_seen():
    a = _run(codes=False, mutate=_mutate_within_turn)
    b = _run(codes=True, mutate=_mutate_within_turn)
    assert _same(a, b)


@fp8_only
def test_mutation_within_turn_needs_the_version_key(monkeypatch):
    """Must-fail partner: the key without the version counter."""
    a = _run(codes=False, mutate=_mutate_within_turn)
    orig = F8._codes_key
    monkeypatch.setattr(F8, "_codes_key", lambda mod: orig(mod)[:2])
    b = _run(codes=True, mutate=_mutate_within_turn)
    assert not _same(a, b), "a within-turn write was not seen"


@fp8_only
def test_active_block_never_reads_codes(monkeypatch):
    orig = F8._streamed_codes
    seen = []

    def checked(mod):
        assert not mod.weight.requires_grad, "active (trainable) weight read frozen codes"
        assert mod.weight.device.type == "cpu", "codes streamed while the weight was too"
        seen.append(1)
        return orig(mod)
    monkeypatch.setattr(F8, "_streamed_codes", checked)
    _run(codes=True)
    assert seen
