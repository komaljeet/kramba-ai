"""BCD optimizer-state retention (Issue #152).

Author: Rambarun Komaljeet

#112 measured block-coordinate descent at ~3.9x worse perplexity and named "the
outgoing block's optimizer state deleted on every switch" as a candidate cause,
never tested. `retain_optimizer_state=True` parks that state on CPU and restores
it when the block returns.

The restore is deliberately **undecayed**, and that is the whole design. Momentum
over k steps of zero gradient decays to `beta^k * m`; Muon runs beta=0.95 and a
block is away ~1,300 steps, so `0.95^1300 ~= 1e-29` -- a *correctly decayed*
restore is arithmetically identical to deletion, i.e. to what BCD already does.
Only a "paused, not decaying" restore has any content to test.

The load-bearing test here is the last one: retention must not leak state into
the WRONG block. Silent cross-block contamination would look like a quality win
and be nearly impossible to spot in a loss curve -- the #117 failure shape.
"""

import pytest
import torch
import torch.nn as nn

from hvn.core.block_coordinate import BlockCoordinateScheduler
from hvn.edge.model import EdgeHVN


def _setup(retain: bool, steps_per_block: int = 2):
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=64, d_model=32, num_layers=3, n_heads=4,
                use_deltanet=True, dropout=0.0, tie_embeddings=False)
    opt = torch.optim.SGD(m.parameters(), lr=0.1, momentum=0.9)
    sched = BlockCoordinateScheduler(m, opt, steps_per_block=steps_per_block,
                                     retain_optimizer_state=retain)
    return m, opt, sched


def _train_one_step(m, opt):
    x = torch.randint(0, 64, (2, 8))
    loss = nn.functional.cross_entropy(
        m(x).reshape(-1, 64), torch.roll(x, -1, 1).reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad(set_to_none=True)


def test_default_is_deletion():
    """#112's measurements were all taken with deletion; it stays the default."""
    _m, _o, s = _setup(retain=False)
    assert s.retain_optimizer_state is False


def test_deletion_still_clears_state_when_off():
    """The documented BCD behaviour must be untouched by this feature."""
    m, opt, s = _setup(retain=False)
    first = s.blocks[0][1]
    _train_one_step(m, opt)
    assert any(p in opt.state for p in first)
    for _ in range(s.steps_per_block):
        s.step()
    assert not any(p in opt.state for p in first), "state survived with retain=False"
    assert not s._parked_state, "nothing should be parked with retain=False"


def test_retention_parks_and_restores_the_same_values():
    """The point of the feature: a block resumes with the momentum it had."""
    m, opt, s = _setup(retain=True)
    first = s.blocks[0][1]
    _train_one_step(m, opt)
    saved = {p: opt.state[p]["momentum_buffer"].clone()
             for p in first if "momentum_buffer" in opt.state.get(p, {})}
    assert saved, "no momentum to test with"

    # Rotate all the way round, back to block 0.
    for _ in range(s.steps_per_block * len(s.blocks)):
        s.step()

    for p, before in saved.items():
        assert p in opt.state, "state was not restored"
        after = opt.state[p]["momentum_buffer"]
        assert torch.equal(after.cpu(), before.cpu()), \
            "restored momentum differs from what was parked (must be UNDECAYED)"


def test_parked_state_is_off_gpu_while_inactive():
    """Retention must not silently re-spend the VRAM BCD exists to save."""
    m, opt, s = _setup(retain=True)
    _train_one_step(m, opt)
    for _ in range(s.steps_per_block):
        s.step()
    assert s._parked_state, "nothing parked"
    for st in s._parked_state.values():
        for v in st.values():
            if torch.is_tensor(v):
                assert v.device.type == "cpu", "parked state is still on device"


def test_state_is_restored_to_the_block_it_came_from():
    """The dangerous failure: state leaking into the WRONG block.

    Cross-contamination would look like a quality effect and be invisible in a
    loss curve -- the #117 shape. Asserted directly rather than trusted.
    """
    m, opt, s = _setup(retain=True)
    _train_one_step(m, opt)
    owner = {}
    for name, params in s.blocks:
        for p in params:
            owner[id(p)] = name

    # `active_block_name` is a property on the scheduler. Resolved by reading
    # the class rather than guessed: an earlier draft used a `hasattr` fallback
    # on a name that does not exist, so `active` was always None and this
    # assertion never ran -- a test that passed while checking nothing.
    assert isinstance(s.active_block_name, str)

    checked = 0
    for _ in range(s.steps_per_block * len(s.blocks) * 2):
        s.step()
        active = s.active_block_name
        for p in list(opt.state):
            assert owner[id(p)] == active, (
                f"optimizer state present for a {owner[id(p)]!r} parameter "
                f"while {active!r} is active — state leaked across blocks")
            checked += 1
    assert checked > 0, "no optimizer state was ever inspected; test is vacuous"
