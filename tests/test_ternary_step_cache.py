"""Within-step quantized-weight cache for trainable TernaryLinear (2026-10-14).

Author: Rambarun Komaljeet

The cache reuses one quantized weight for every forward AND checkpoint
recompute inside an optimizer step. Its failure mode is silent staleness
(#117: a cache served untrained weights for a whole epic while aggregate loss
still fell), so the gate here is a SEQUENCE of optimizer steps -- the state
that would go stale carries forward -- and the comparison is bit-exact:

* per-micro-batch losses, every parameter gradient, and the weights after
  every step must be ``torch.equal`` to an uncached run of the same code;
* the cache must actually be exercised: exactly one quantization per trainable
  layer per step, all other calls hits;
* anti-vacuity: a deliberately stale cache (never invalidated) run through
  the SAME comparison must fail it -- otherwise the comparison could not see
  the bug it exists for;
* invalidation must not depend on the version counter: an optimizer that
  writes through ``p.data`` (no version bump) is covered by the step hooks.
"""

import copy

import pytest
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

import hvn.edge.ternary as T
from hvn.edge.ternary import TernaryLinear, enable_ternary_step_cache

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

D, GA, STEPS = 256, 3, 4


class _Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.a = TernaryLinear(D, 2 * D)
        self.b = TernaryLinear(2 * D, D)

    def forward(self, x):
        return x + self.b(torch.nn.functional.gelu(self.a(x)))


class _Net(nn.Module):
    def __init__(self, n=3):
        super().__init__()
        self.blocks = nn.ModuleList(_Block() for _ in range(n))

    def forward(self, x):
        for blk in self.blocks:
            x = checkpoint(blk, x, use_reentrant=False)
        return x


class _DataSGD(torch.optim.Optimizer):
    """Writes through ``p.data`` -- the version counter does NOT move."""

    def __init__(self, params, lr):
        super().__init__(params, {"lr": lr})

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    p.data.add_(p.grad, alpha=-g["lr"])


def _run(net, cached, fused, opt_kind, frozen_block=None):
    torch.manual_seed(0)
    net = copy.deepcopy(net).cuda()
    for m in net.modules():
        if isinstance(m, TernaryLinear):
            m.use_fused = fused
    enable_ternary_step_cache(net, cached)
    if frozen_block is not None:
        for p in net.blocks[frozen_block].parameters():
            p.requires_grad_(False)
    params = [p for p in net.parameters() if p.requires_grad]
    opt = (torch.optim.SGD(params, lr=0.5, momentum=0.9) if opt_kind == "sgd"
           else _DataSGD(params, lr=0.5))
    g = torch.Generator(device="cuda").manual_seed(1)
    xs = torch.randn(STEPS, GA, 2, 64, D, device="cuda", generator=g)
    T.STEP_CACHE_STATS.update(hits=0, misses=0)
    trace = []
    for s in range(STEPS):
        losses = []
        for mb in range(GA):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                y = net(xs[s, mb].requires_grad_())
                loss = (y.float() ** 2).mean() * (1 + s)
            loss.backward()
            losses.append(loss.detach().clone())
        grads = [p.grad.clone() for p in params]
        opt.step()
        opt.zero_grad(set_to_none=True)
        trace.append((losses, grads, [p.detach().clone() for p in params]))
    stats = dict(T.STEP_CACHE_STATS)
    enable_ternary_step_cache(net, False)
    return trace, stats


def _first_mismatch(ref, got):
    for s, ((l0, g0, w0), (l1, g1, w1)) in enumerate(zip(ref, got)):
        for i, (a, b) in enumerate(zip(l0, l1)):
            if not torch.equal(a, b):
                return f"step {s} micro-batch {i} loss {a.item()} vs {b.item()}"
        for i, (a, b) in enumerate(zip(g0, g1)):
            if not torch.equal(a, b):
                return f"step {s} grad {i} max|d|={(a - b).abs().max().item():.3e}"
        for i, (a, b) in enumerate(zip(w0, w1)):
            if not torch.equal(a, b):
                return f"step {s} weight {i}"
    return None


def _n_trainable_ternary(net, frozen_block=None):
    n = 0
    for bi, blk in enumerate(net.blocks):
        if bi != frozen_block:
            n += sum(isinstance(m, TernaryLinear) for m in blk.modules())
    return n


@cuda_only
@pytest.mark.parametrize("fused", [True, False])
@pytest.mark.parametrize("opt_kind", ["sgd", "data"])
def test_cached_matches_uncached_bit_exact_over_steps(fused, opt_kind):
    torch.manual_seed(0)
    net = _Net()
    ref, s_ref = _run(net, False, fused, opt_kind)
    got, s_got = _run(net, True, fused, opt_kind)
    assert s_ref == {"hits": 0, "misses": 0}
    n = _n_trainable_ternary(net)
    # one quantization per trainable layer per step; every other call hits:
    # GA micro-batches x (forward + checkpoint recompute) = 2*GA calls per step
    assert s_got["misses"] == n * STEPS, s_got
    assert s_got["hits"] == n * STEPS * (2 * GA - 1), s_got
    assert _first_mismatch(ref, got) is None, _first_mismatch(ref, got)
    # the weights really moved, so a stale cache WOULD have been visible
    assert not torch.equal(ref[0][2][0], ref[-1][2][0])


@cuda_only
def test_frozen_layers_are_not_cached():
    torch.manual_seed(0)
    net = _Net()
    ref, _ = _run(net, False, True, "sgd", frozen_block=1)
    got, s_got = _run(net, True, True, "sgd", frozen_block=1)
    assert s_got["misses"] == _n_trainable_ternary(net, 1) * STEPS
    assert _first_mismatch(ref, got) is None


@cuda_only
@pytest.mark.parametrize("opt_kind", ["sgd", "data"])
def test_anti_vacuity_stale_cache_is_caught(monkeypatch, opt_kind):
    """A cache that is never invalidated must FAIL the same comparison."""
    torch.manual_seed(0)
    net = _Net()
    ref, _ = _run(net, False, True, opt_kind)

    def stale(self, dtype):
        w = self.weight
        if self.use_step_cache and w.requires_grad and torch.is_grad_enabled():
            if self._scache is None:
                with torch.no_grad():
                    self._scache = (None, self._quantize_raw(w.detach(), dtype))
            return T._CachedSTE.apply(w, self._scache[1])
        return self._quantize_raw(w, dtype)

    monkeypatch.setattr(TernaryLinear, "_dequantized", stale)
    monkeypatch.setattr(T, "_STEP_LIVE", set())   # hooks can no longer reach entries
    got, _ = _run(net, True, True, opt_kind)
    msg = _first_mismatch(ref, got)
    assert msg is not None and not msg.startswith("step 0"), msg


@cuda_only
def test_cache_freed_at_optimizer_step():
    """Entries must not outlive the step (VRAM): the pre-step hook drops them."""
    torch.manual_seed(0)
    net = _Net().cuda()
    enable_ternary_step_cache(net, True)
    opt = torch.optim.SGD(net.parameters(), lr=0.1)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        net(torch.randn(2, 8, D, device="cuda", requires_grad=True)).float().sum().backward()
    live = [m for m in net.modules() if isinstance(m, TernaryLinear) and m._scache is not None]
    assert len(live) == _n_trainable_ternary(net)
    opt.step()
    assert all(m._scache is None for m in net.modules() if isinstance(m, TernaryLinear))
    enable_ternary_step_cache(net, False)


# ---------------------------------------------------------------------------
# The real composition: EdgeHVN hybrid + fused ternary + Triton scan/megakernel
# + BCD balanced partition + CPU weight offload (bf16 park, prefetch, reused
# buffer sets) + Muon, over a FULL block cycle plus two steps, so every block is
# active once and its cache is filled, used, invalidated and refilled.
# Measured deterministic here (uncached vs uncached bit-identical), so the gate
# is exact equality, not a tolerance.
# ---------------------------------------------------------------------------
def _bcd_run(cached):
    import torch.nn.functional as F
    from hvn.core.block_coordinate import BlockCoordinateScheduler
    from hvn.edge.model import EdgeHVN
    from hvn.optimizers import build_optimizer

    Dm, L, S, V, ga = 256, 4, 512, 2048, 2
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=V, d_model=Dm, num_layers=L, n_heads=8,
                block_plan=["deltanet", "transformer", "deltanet", "deltanet"],
                grad_checkpoint=True, tie_embeddings=True, dropout=0.0)
    names = [n for n, mod in m.named_modules() if isinstance(mod, nn.Linear)
             and not any(p in n for p in ("beta_proj", "alpha_proj", "decoder"))]
    T.apply_ternary_to_model(m, names)
    for mod in m.modules():
        if isinstance(mod, TernaryLinear):
            mod.use_fused = True
        if hasattr(mod, "use_triton_scan"):
            mod.use_triton_scan = True
        if hasattr(mod, "use_megakernel_scan"):
            mod.use_megakernel_scan = True
    enable_ternary_step_cache(m, cached)
    m.train()
    opt, _ = build_optimizer(m, "muon", lr=3e-3, weight_decay=0.01, fused=False,
                             muon_lr=0.05)
    sched = BlockCoordinateScheduler(
        m, opt, steps_per_block=1, device="cuda", offload_inactive_weights=True,
        offload_park_dtype=torch.bfloat16, offload_prefetch=True,
        offload_cache_blocks=1, partition="balanced")
    n_steps = sched.num_blocks + 2
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, V, (n_steps, ga, 1, S), generator=g)
    T.STEP_CACHE_STATS.update(hits=0, misses=0)
    trace = []
    for s in range(n_steps):
        losses = []
        for mb in range(ga):
            x = ids[s, mb].cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                out = m(x)
                loss = F.cross_entropy(out.float().reshape(-1, V), x.reshape(-1)) / ga
            loss.backward()
            losses.append(loss.detach().clone())
        grads = [p.grad.detach().cpu().clone() for p in m.parameters() if p.grad is not None]
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad()
        sched.step()
        trace.append((losses, grads, []))
    trace[-1] = (trace[-1][0], trace[-1][1],
                 [p.detach().float().cpu().clone() for p in m.parameters()])
    stats = dict(T.STEP_CACHE_STATS)
    enable_ternary_step_cache(m, False)
    return trace, stats, n_steps, ga


@cuda_only
def test_bcd_offload_real_model_bit_exact():
    ref, s_ref, n_steps, ga = _bcd_run(False)
    got, s_got, _, _ = _bcd_run(True)
    assert s_ref == {"hits": 0, "misses": 0}
    assert s_got["misses"] > 0 and s_got["hits"] == s_got["misses"] * (2 * ga - 1), s_got
    assert _first_mismatch(ref, got) is None, _first_mismatch(ref, got)


@cuda_only
def test_bcd_offload_anti_vacuity_stale_cache_is_caught(monkeypatch):
    ref, _, _, _ = _bcd_run(False)

    def stale(self, dtype):
        w = self.weight
        if self.use_step_cache and w.requires_grad and torch.is_grad_enabled():
            if self._scache is None:
                with torch.no_grad():
                    self._scache = (None, self._quantize_raw(w.detach(), dtype))
            return T._CachedSTE.apply(w, self._scache[1])
        return self._quantize_raw(w, dtype)

    monkeypatch.setattr(TernaryLinear, "_dequantized", stale)
    monkeypatch.setattr(T, "_STEP_LIVE", set())
    got, _, _, _ = _bcd_run(True)
    assert _first_mismatch(ref, got) is not None
