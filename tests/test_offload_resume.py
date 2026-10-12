"""Resuming a --block-coordinate --offload run must not lose the optimizer.

Author: Rambarun Komaljeet

Found 2026-10-12 by the 1024x20 offload integration run through the real
trainer. train() restored the optimizer state BEFORE the BCD cycle position.
``optimizer.load_state_dict`` casts each saved moment to its param's current
device and dtype -- and at that moment the fresh scheduler had block 0 active
and every other block parked on CPU in bf16. Any resume whose saved active
block was not block 0 therefore put that block's momentum on the CPU in bf16,
and the first ``optimizer.step()`` died with "Expected all tensors to be on the
same device". With 100 steps per block and saves every 250, that is nearly
every real resume.

These tests drive ``_restore_optimizer_and_bcd`` -- the exact function the
trainer calls -- so reverting the order fails here, on a toy model, in seconds.
"""

import pytest
import torch

from hvn.core.block_coordinate import BlockCoordinateScheduler
from hvn.edge.model import EdgeHVN
from hvn.optimizers import build_optimizer
from train_edge_conversational import _restore_optimizer_and_bcd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(),
                                reason="offload needs a CUDA device")

DEV = torch.device("cuda")
STEPS_PER_BLOCK = 3


def _build():
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=256, d_model=64, num_layers=4, n_heads=4,
                use_deltanet=True, dropout=0.0, tie_embeddings=True,
                grad_checkpoint=True).to(DEV)
    opt, _ = build_optimizer(m, "muon", lr=3e-4, weight_decay=0.0, muon_lr=0.02)
    sched = BlockCoordinateScheduler(
        m, opt, steps_per_block=STEPS_PER_BLOCK, offload_inactive_weights=True,
        device=DEV, offload_park_dtype=torch.bfloat16, offload_prefetch=True,
        partition="layer")
    m.train()
    return m, opt, sched


def _step(m, opt, sched):
    x = torch.randint(0, 256, (1, 32), device=DEV)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        loss = torch.nn.functional.cross_entropy(
            m(x).float().reshape(-1, 256), x.reshape(-1))
    loss.backward()
    opt.step()
    opt.zero_grad()
    sched.step()


def _saved_state(n_steps):
    m, opt, sched = _build()
    for _ in range(n_steps):
        _step(m, opt, sched)
    return {"optimizer_state_dict": opt.state_dict(),
            "bcd_state_dict": sched.state_dict()}


def test_test_is_not_vacuous():
    """The resume test below only bites if the saved state has an active block
    other than block 0 AND that block carries optimizer state. Either missing
    and the old order passes too."""
    st = _saved_state(STEPS_PER_BLOCK + 1)        # 1 step into block 1
    assert st["bcd_state_dict"]["active_idx"] != 0
    assert st["bcd_state_dict"]["step_in_block"] != 0
    assert len(st["optimizer_state_dict"]["state"]) > 0


def test_resume_mid_block_past_block_zero_trains():
    st = _saved_state(STEPS_PER_BLOCK + 1)
    m, opt, sched = _build()                       # fresh: block 0 active
    _restore_optimizer_and_bcd(opt, sched, st)
    assert sched._active_idx == st["bcd_state_dict"]["active_idx"]
    # Every restored moment must sit with its param: on the GPU, at the
    # param's native (fp32) dtype -- not parked bf16 on the CPU.
    n = 0
    for p, s in opt.state.items():
        for v in s.values():
            if torch.is_tensor(v) and v.dim() > 0:
                assert v.device == p.device, (v.device, p.device)
                assert v.dtype == p.dtype == torch.float32, (v.dtype, p.dtype)
                n += 1
    assert n > 0, "no optimizer state restored -- test would be vacuous"
    for _ in range(2 * STEPS_PER_BLOCK):           # crosses a block switch too
        _step(m, opt, sched)


def test_offload_resume_load_stays_on_cpu(tmp_path):
    """An --offload resume must not put the whole model, or the checkpoint,
    on the GPU (#163's CPU-resident start). Before the fix both went to the
    card and the returned dict kept the checkpoint there for the whole run."""
    from train_edge_conversational import (_save_edge_checkpoint,
                                           load_checkpoint_for_resume)
    m, opt, sched = _build()
    _step(m, opt, sched)
    path = str(tmp_path / "edge_conversational_step1.pt")
    _save_edge_checkpoint(m, path, global_step=1)
    ref = {k: v.detach().float().cpu().clone() for k, v in m.state_dict().items()}

    loaded, step, ck = load_checkpoint_for_resume(path, DEV, offload=True)
    assert step == 1
    assert all(p.device.type == "cpu" for p in loaded.parameters())
    assert all(t.device.type == "cpu" for t in ck["model_state_dict"].values()
               if torch.is_tensor(t))
    # and the weights are the saved ones, not a fresh init
    got = loaded.state_dict()
    assert set(ref) == set(got)
    assert all(torch.equal(got[k].float(), ref[k]) for k in ref)

    # non-offload resume still lands on the device
    loaded2, _, _ = load_checkpoint_for_resume(path, DEV, offload=False)
    assert all(p.device.type == "cuda" for p in loaded2.parameters())
