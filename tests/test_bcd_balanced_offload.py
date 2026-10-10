"""Balanced BCD partition + CPU weight offload together (Issue #164).

Before #164 the combination was refused: EdgeHVN's offload path asked "is this
segment the active block?" by comparing ONE name to the active block's name, and
a balanced group's name ("blocks.0+blocks.1") matches no segment. Every layer of
the active group would then be load()-ed from its CPU home and evicted back to
it -- after the optimizer had updated the GPU copy. That is #117's failure mode
(training silently discarded) in a new place, so these tests are built the way
#117 taught: they compare a SEQUENCE of steps across several group switches,
not one step, and they assert the specific quantities that would go stale.

  1. Parity over 2.5 rotations: balanced+offload vs balanced without offload,
     identical seed and batches. Every step's loss and every final parameter
     must match. Run tied and untied (untied makes token_embedding and the tail
     members of multi-layer groups -- the non-checkpointed wrapper path), and
     with the LRU cache on (cache_blocks=3).
  2. Residency, checked after the forward AND after the backward of every step:
     every member of the active group on the GPU, every other parameter on the
     CPU (except the pinned tied table, which must be on the GPU). This is the
     "a merged group moves as a whole, not partially" claim, checked directly.
  3. bf16 parking (what train_edge_conversational.py uses under --offload):
     each group comes back at fp32 holding the value it left with, not its
     init -- per group, across two tenures.
  4. The vacuity guard: the OLD single-name check is reinstated on purpose and
     the parity test MUST fail against it. Without this, (1) could pass by never
     exercising a merged group at all.

GPU only: nothing moves on CPU, so offload correctness is meaningless there.
Run:  python -m pytest tests/test_bcd_balanced_offload.py -q
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

from hvn.core.block_coordinate import (  # noqa: E402
    BlockCoordinateScheduler, partition_named_blocks,
)
from hvn.edge.model import EdgeHVN  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="offload correctness needs CUDA")

V = 512
SPB = 2          # steps per block
PLAN = ["deltanet", "deltanet", "transformer", "deltanet", "deltanet", "deltanet"]


def _model(tie: bool) -> EdgeHVN:
    return EdgeHVN(vocab_size=V, d_model=32, num_layers=len(PLAN), n_heads=4,
                   block_plan=PLAN, chunk_size=8, dropout=0.0,
                   grad_checkpoint=True, tie_embeddings=tie, attn_window=16)


def _budget(m: EdgeHVN) -> int:
    """Two layers plus the tail: forces multi-layer groups and puts the tail
    (and, untied, the embedding) inside one. Tied: [emb] [b0 b1] [b2 b3]
    [b4 b5 tail]. Untied: [emb b0 b1] [b2 b3 b4] [b5 tail]."""
    sizes = {n: sum(p.numel() for p in ps) for n, ps in partition_named_blocks(m)}
    layer = max(s for n, s in sizes.items() if n.startswith("blocks."))
    return 2 * layer + sizes["final_norm+decoder"]


def _segments(m: EdgeHVN) -> list[tuple[str, list]]:
    return ([("token_embedding", list(m.token_embedding.parameters()))]
            + [(f"blocks.{i}", list(b.parameters())) for i, b in enumerate(m.blocks)]
            + [("final_norm+decoder",
                list(m.final_norm.parameters()) + list(m.decoder.parameters()))])


def _check_residency(m, sched, when: str, strict_inactive: bool = True) -> int:
    """Every member of the active group on cuda, everything else on cpu.
    Returns how many parameters were checked as OFFLOADED (caller asserts > 0,
    so this cannot pass by offloading nothing)."""
    active = set(sched.active_block_members)
    n_cpu = 0
    for name, params in _segments(m):
        for p in params:
            if id(p) in sched._pinned:
                assert p.device.type == "cuda", f"pinned table in {name} evicted ({when})"
                continue
            if name in active:
                assert p.device.type == "cuda", (
                    f"{name} is a member of the ACTIVE group "
                    f"{sched.active_block_name!r} but sits on {p.device} ({when}) "
                    f"-- part of the group was evicted")
            elif strict_inactive and (name != "final_norm+decoder"
                                      or when != "after forward"):
                # The tail is not checkpointed: when downstream of the active
                # group it legitimately stays resident until backward. With an
                # LRU cache > 1 (strict_inactive=False after the forward),
                # downstream layers are kept warm for their recompute on
                # purpose (#103), so only the active side is checked then.
                assert p.device.type == "cpu", (
                    f"{name} is inactive (active={sched.active_block_name!r}) but "
                    f"sits on {p.device} ({when})")
                n_cpu += 1
    return n_cpu


def _run(offload: bool, tie: bool, n_steps: int, *, cache_blocks: int = 1,
         old_single_name_check: bool = False, check_residency: bool = True):
    device = torch.device("cuda")
    torch.manual_seed(0)
    m = _model(tie).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    sched = BlockCoordinateScheduler(
        m, opt, steps_per_block=SPB, partition="balanced", partition_budget=_budget(m),
        offload_inactive_weights=offload, device=device if offload else None,
        offload_cache_blocks=cache_blocks)

    if old_single_name_check:
        # Reinstate the pre-#164 behaviour exactly: the model is told the
        # group's joined name, which equals a segment name only for
        # single-member groups.
        orig = sched._activate

        def _old(idx):
            orig(idx)
            m._offload_active_names = frozenset({sched.blocks[idx][0]})
        sched._activate = _old
        m._offload_active_names = frozenset({sched.active_block_name})

    torch.manual_seed(1)
    batches = [torch.randint(0, V, (2, 16), device=device) for _ in range(n_steps)]
    losses, offloaded_checks = [], 0
    for ids in batches:
        logits = m(ids)
        if offload and check_residency:
            offloaded_checks += _check_residency(m, sched, "after forward",
                                                 strict_inactive=cache_blocks == 1)
        loss = F.cross_entropy(logits.reshape(-1, V), ids.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if offload and check_residency:
            offloaded_checks += _check_residency(m, sched, "after backward")
        opt.step()
        sched.step()
        losses.append(loss.item())
    params = {n: p.detach().float().cpu().clone() for n, p in m.named_parameters()}
    return sched, losses, params, offloaded_checks


def _steps(sched) -> int:
    # 2.5 rotations: every group is deactivated and REACTIVATED at least once,
    # which is when a stale CPU home would be served (#117).
    return int(sched.num_blocks * SPB * 2.5)


def _n_groups(tie):
    m = _model(tie)
    opt = torch.optim.AdamW(m.parameters())
    return BlockCoordinateScheduler(m, opt, partition="balanced",
                                    partition_budget=_budget(m))


@pytest.mark.parametrize("tie,cache_blocks", [(True, 1), (False, 1), (True, 3)])
def test_balanced_offload_matches_no_offload_over_rotations(tie, cache_blocks):
    probe = _n_groups(tie)
    members = probe._block_members
    # Guard the premise: real merged groups, one of them holding the tail, and
    # (untied) one holding the embedding. Otherwise this is the layer test again.
    assert sum(len(g) > 1 for g in members) >= 2, members
    assert any("final_norm+decoder" in g and len(g) > 1 for g in members), members
    if not tie:
        assert any("token_embedding" in g and len(g) > 1 for g in members), members
    n = _steps(probe)

    _, ref_losses, ref_params, _ = _run(False, tie, n)
    sched, losses, params, n_cpu = _run(True, tie, n, cache_blocks=cache_blocks)
    assert n_cpu > 0, "nothing was ever offloaded -- the test is vacuous"

    for i, (a, b) in enumerate(zip(ref_losses, losses)):
        assert abs(a - b) <= 1e-5 + 1e-4 * abs(a), (
            f"step {i}: loss {b:.6f} with offload vs {a:.6f} without")
    for name, ref in ref_params.items():
        diff = (params[name] - ref).abs().max().item()
        assert diff <= 1e-4, f"{name}: max|diff| {diff:.2e} after {n} steps"
    # The sequence must actually have trained (not a frozen model matching a
    # frozen model).
    assert ref_losses[-1] < ref_losses[0]


def test_old_single_name_check_is_caught():
    """Vacuity guard: with the pre-#164 single-name comparison put back, the
    parity test above must fail -- either the run breaks outright or the
    numbers diverge. If this ever passes cleanly, the parity test has stopped
    exercising merged groups and its PASS means nothing."""
    tie = True
    n = _steps(_n_groups(tie))
    _, ref_losses, ref_params, _ = _run(False, tie, n)
    try:
        _, losses, params, _ = _run(True, tie, n, old_single_name_check=True,
                                    check_residency=False)
    except RuntimeError:
        return  # broke loudly (e.g. optimizer stepping a CPU-homed param)
    worst = max((params[k] - v).abs().max().item() for k, v in ref_params.items())
    assert worst > 1e-3, (
        f"old single-name offload check matched the reference (max|diff| "
        f"{worst:.2e}) -- the parity test cannot see the #164 bug")


def test_old_single_name_check_fails_residency():
    """Same guard for the residency check alone: under the old comparison a
    merged active group is NOT fully GPU-resident after the step."""
    with pytest.raises((AssertionError, RuntimeError)):
        _run(True, True, _steps(_n_groups(True)), old_single_name_check=True)


def test_bf16_parked_groups_return_with_their_training():
    """--offload in train_edge_conversational.py parks inactive blocks in bf16.
    Each group must (a) be fp32 in full while active and bf16 in full while
    parked, and (b) come back holding the value it left with -- for EVERY
    member, including non-first ones, across two tenures."""
    device = torch.device("cuda")
    torch.manual_seed(0)
    m = _model(True).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    sched = BlockCoordinateScheduler(
        m, opt, steps_per_block=SPB, partition="balanced", partition_budget=_budget(m),
        offload_inactive_weights=True, device=device,
        offload_park_dtype=torch.bfloat16)
    init = {id(p): p.detach().float().cpu().clone() for p in m.parameters()}
    left_with: dict[int, torch.Tensor] = {}
    returns_checked = 0

    def _dtypes_ok():
        active = set(sched.active_block_members)
        for name, params in _segments(m):
            for p in params:
                if id(p) in sched._pinned:
                    assert p.dtype == torch.float32
                    continue
                want = torch.float32 if name in active else torch.bfloat16
                assert p.dtype == want, (
                    f"{name} is {p.dtype}, want {want} "
                    f"(active={sched.active_block_name!r})")

    torch.manual_seed(1)
    _dtypes_ok()
    for _ in range(_steps(sched)):
        idx = sched._active_idx
        ids = torch.randint(0, V, (2, 16), device=device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = m(ids)
            loss = F.cross_entropy(logits.reshape(-1, V).float(), ids.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        outgoing = [p for p in sched.blocks[idx][1] if id(p) not in sched._pinned]
        snap = {id(p): p.detach().float().cpu().clone() for p in outgoing}
        if sched.step():
            left_with.update(snap)
            _dtypes_ok()
            for p in sched.blocks[sched._active_idx][1]:
                if id(p) in left_with and p.dim() >= 2:
                    now = p.detach().float().cpu()
                    was = left_with[id(p)]
                    tol = was.abs().max().item() * 2 ** -7 + 1e-8
                    assert (now - was).abs().max().item() <= tol, (
                        "group came back with a different value than it left with")
                    assert (was - init[id(p)]).abs().max().item() > tol, (
                        "group left without having trained -- vacuous")
                    returns_checked += 1
    # Every non-pinned matrix in every group must have been checked on return.
    expected = sum(1 for _, ps in sched.blocks for p in ps
                   if id(p) not in sched._pinned and p.dim() >= 2)
    assert returns_checked >= expected, (returns_checked, expected)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
