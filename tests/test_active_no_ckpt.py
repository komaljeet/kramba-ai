"""#202: the active BCD block's last N layers run without checkpointing.

Author: Rambarun Komaljeet

Two claims, tested separately:

1. **Same numbers.** Keeping a layer's activations instead of recomputing them
   in backward runs the same ops in the same order, so losses and every
   gradient must be bit-identical (``torch.equal``) to the all-checkpointed run,
   over several optimizer steps and BCD turns.
2. **The recompute is really gone.** Each kept layer's forward runs once per
   micro-batch, not twice (forward + recompute). Without this, (1) would pass
   just as well if the flag did nothing.

Plus a must-fail partner showing the equality check sees a one-ulp change.
CUDA only: offload needs a device.
"""
import os
import sys

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# Before any CUDA work: pandas/pyarrow after CUDA init crashes this venv
# (see tests/conftest.py).
from train_edge_conversational import _parse_active_no_ckpt  # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="offload needs CUDA")
L, V, GA = 6, 64, 2


def _run(n_keep, steps=8):
    """n_keep: int, or a list with one N per layer block (resolved like the trainer)."""
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=V, d_model=32, num_layers=L, n_heads=4, use_deltanet=True,
                chunk_size=8, dropout=0.0, grad_checkpoint=True).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    sched = BlockCoordinateScheduler(m, opt, steps_per_block=2, offload_inactive_weights=True,
                                     device=torch.device("cuda"), offload_prefetch=True,
                                     partition="balanced", layers_per_block=3)
    if isinstance(n_keep, list):
        m._active_no_ckpt = _parse_active_no_ckpt(",".join(map(str, n_keep)), sched._block_members)
    else:
        m._active_no_ckpt = n_keep
    calls = {i: 0 for i in range(L)}
    for i, blk in enumerate(m.blocks):
        blk.register_forward_pre_hook(lambda mod, inp, i=i: calls.__setitem__(i, calls[i] + 1))
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, V, (steps, GA, 2, 16), generator=g)
    losses, grads, per_step_calls = [], [], []
    for s in range(steps):
        for k in calls:
            calls[k] = 0
        active = set(sched.active_block_members)
        for mb in range(GA):
            x = ids[s, mb].cuda()
            loss = F.cross_entropy(m(x).reshape(-1, V), x.reshape(-1)) / GA
            loss.backward()
            losses.append(loss.detach().clone())
        grads.append([p.grad.detach().clone() for p in m.parameters()
                      if p.requires_grad and p.grad is not None])
        per_step_calls.append((active, dict(calls)))
        opt.step()
        opt.zero_grad()
        sched.step()
    return losses, grads, per_step_calls


@cuda_only
@pytest.mark.parametrize("n_keep", [1, 3, [2, 1]])
def test_no_ckpt_bit_identical_and_recompute_gone(n_keep):
    l0, g0, c0 = _run(0)
    l1, g1, c1 = _run(n_keep)
    per_block = n_keep if isinstance(n_keep, list) else None
    assert len(l0) == len(l1) and all(torch.equal(a, b) for a, b in zip(l0, l1)), "loss moved"
    for s, (a, b) in enumerate(zip(g0, g1)):
        assert len(a) == len(b) > 0, f"step {s}: no active gradients to compare"
        for i, (x, y) in enumerate(zip(a, b)):
            assert torch.equal(x, y), f"step {s} grad {i}: max |d| {(x - y).abs().max().item():.3e}"
    layer_turns = 0
    for (active, calls0), (_, calls1) in zip(c0, c1):
        act = sorted(int(n.split(".")[1]) for n in active if n.startswith("blocks."))
        if not act:
            continue
        layer_turns += 1
        k = per_block[act[0] // 3] if per_block else n_keep     # 3 layers per block
        kept = act[-k:]
        for i in act:
            assert calls0[i] == 2 * GA, f"checkpointed active layer {i}: {calls0[i]} calls"
            want = GA if i in kept else 2 * GA
            assert calls1[i] == want, f"layer {i} (kept={i in kept}): {calls1[i]} calls, want {want}"
    assert layer_turns > 0, "no layer block was ever active: the call check is vacuous"


@cuda_only
def test_equality_check_sees_one_ulp():
    """Must-fail partner: the gradient comparison above is sensitive at the
    last bit, so 'bit-identical' is a real claim, not a tolerance."""
    _, g0, _ = _run(0, steps=1)
    x = g0[0][0].clone()
    flat = x.view(-1)
    flat[0] = torch.nextafter(flat[0], flat[0] + 1)
    assert not torch.equal(g0[0][0], x)


def test_parse_active_no_ckpt():
    members = [("token_embedding",), ("blocks.0", "blocks.1"),
               ("blocks.2", "blocks.3", "final_norm", "decoder")]
    assert _parse_active_no_ckpt("0", members) == 0
    assert _parse_active_no_ckpt("2", members) == 2
    assert _parse_active_no_ckpt(None, members) == 0
    d = _parse_active_no_ckpt("1,3", members)
    assert d == {frozenset(members[1]): 1, frozenset(members[2]): 3}
    with pytest.raises(ValueError):
        _parse_active_no_ckpt("1,2,3", members)
