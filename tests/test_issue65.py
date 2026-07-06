"""
Test suite for Issue #65 — distributed data-parallel training path (DDP).

The acceptance criterion (from the issue, mirroring the #46
``batched == single-stream`` and #50 ``sync-removed == original`` proofs):

    A 2-process gloo (CPU) run trains weights equal — within fp reduction
    tolerance — to a single-process run over the same data + seed.

Why parity holds by construction: every rank's micro-batch is the same size,
the loss is a mean over the rank's tokens, and DDP averages gradients across
ranks — so mean-of-means equals the mean over the equivalent single-process
batch. These tests prove the implementation actually delivers that.

Checks:
  1. 2-process DDP (gloo, CPU) == single-process: final weights match.
  2. Same with gradient accumulation (dist bs=1 × accum=2 vs single bs=4),
     exercising the no_sync window logic.
  3. The rank-0 checkpoint has the standard single-GPU layout (loads into a
     fresh model via load_state_dict; config keys intact) — the #67 contract.
  4. Default single-GPU path untouched: train_model without --distributed
     is the pre-#65 behavior (no process group required or created).
  5. DeltaNet (#34) block parity under DDP — the recommended frontier block
     (also proves it needs no find_unused_parameters).

Run:  python tests/test_issue65.py         (CPU-only; ~1-2 min for the spawns)
"""

# --- repo-root path bootstrap (added during reorg, issue #48) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

import os
import tempfile

# Force UTF-8 stdout/stderr so the ✓ printed by train.py doesn't crash on a
# legacy cp1252 Windows console (same guard as auto_train.py).
for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
# Spawned ranks inherit this so their prints don't hit the same cp1252 wall.
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# Windows: the machine hostname can resolve to an unroutable adapter (e.g. the
# WSL/Hyper-V vEthernet), which makes gloo's full-mesh connect hang forever.
# Pin the single-machine parity test to loopback (real multi-node runs set
# GLOO_SOCKET_IFNAME/NCCL_SOCKET_IFNAME to the fabric interface instead).
if _sys.platform == "win32":
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from hvn.core import HamiltonianVortexNetwork
from hvn import distributed as hvn_dist
from train import TextDataset, train_model

TOL = 1e-5
SEED = 0
VOCAB, D_MODEL, LAYERS, HEADS = 32, 32, 2, 2
SEQ_LEN, N_TOKENS, EPOCHS, LR = 16, 16 * 17 + 1, 2, 1e-3  # 17 samples -> 16 used


def _build_model(use_deltanet=False):
    torch.manual_seed(SEED)
    return HamiltonianVortexNetwork(
        vocab_size=VOCAB, d_model=D_MODEL, num_layers=LAYERS, n_heads=HEADS,
        dropout=0.0, use_deltanet=use_deltanet,
    )


def _dataset():
    g = torch.Generator().manual_seed(SEED)
    ids = torch.randint(0, VOCAB, (N_TOKENS,), generator=g).tolist()
    ds = TextDataset(ids, seq_len=SEQ_LEN)
    # Truncate to a multiple of (world=2 × the largest per-rank batch) so
    # DistributedSampler needs no padding and shards are exactly disjoint.
    return torch.utils.data.Subset(ds, list(range(16)))


def _weights_flat(state_dict):
    return torch.cat([v.reshape(-1).float() for v in state_dict.values()])


def _single_process_reference(batch_size, grad_accum, use_deltanet=False):
    """The single-process baseline the distributed run must reproduce."""
    ds = _dataset()
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, drop_last=False)
    model = _build_model(use_deltanet)
    save = os.path.join(tempfile.gettempdir(), "issue65_single.pt")
    train_model(model, dl, epochs=EPOCHS, lr=LR, device=torch.device("cpu"),
                save_path=save, grad_accum_steps=grad_accum)
    os.remove(save)
    return model.state_dict()


def _ddp_worker(rank, world_size, init_file, batch_size, grad_accum,
                use_deltanet, fsdp, ckpt_path):
    """One rank of the 2-process gloo run (module-level for Windows spawn)."""
    # FileStore rendezvous: no TCP store, so the init works even when the
    # hostname doesn't resolve to a reachable address (Windows laptops).
    store = dist.FileStore(init_file, world_size)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world_size)
    try:
        ds = _dataset()
        # shuffle=False -> deterministic interleaved shards whose per-step
        # union across ranks equals the single-process batch.
        sampler = DistributedSampler(ds, num_replicas=world_size, rank=rank,
                                     shuffle=False)
        dl = DataLoader(ds, batch_size=batch_size, sampler=sampler,
                        drop_last=False)
        model = _build_model(use_deltanet)
        train_model(model, dl, epochs=EPOCHS, lr=LR, device=torch.device("cpu"),
                    save_path=ckpt_path, grad_accum_steps=grad_accum,
                    distributed=True, fsdp=fsdp)
    finally:
        dist.destroy_process_group()


def _run_ddp(batch_size, grad_accum, use_deltanet=False, fsdp=False):
    """Spawn the 2-process gloo run; return the rank-0 checkpoint dict."""
    tmp = tempfile.mkdtemp(prefix="issue65_")
    init_file = os.path.join(tmp, "init_store")
    ckpt_path = os.path.join(tmp, "issue65_ddp.pt")
    mp.spawn(_ddp_worker, args=(2, init_file, batch_size, grad_accum,
                                use_deltanet, fsdp, ckpt_path),
             nprocs=2, join=True)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return ckpt


def test_ddp_parity():
    ref = _single_process_reference(batch_size=4, grad_accum=1)
    ckpt = _run_ddp(batch_size=2, grad_accum=1)
    diff = (_weights_flat(ref) - _weights_flat(ckpt["model_state_dict"])).abs().max().item()
    assert diff < TOL, f"DDP(2×bs2) != single(bs4): max|Δw| = {diff}"
    print(f"[1/5] 2-process gloo DDP == single-process (max|Δw| {diff:.2e}) ... OK")
    return ckpt


def test_grad_accum_parity():
    ref = _single_process_reference(batch_size=4, grad_accum=1)
    ckpt = _run_ddp(batch_size=1, grad_accum=2)
    diff = (_weights_flat(ref) - _weights_flat(ckpt["model_state_dict"])).abs().max().item()
    assert diff < TOL, f"DDP(2×bs1×accum2) != single(bs4): max|Δw| = {diff}"
    print(f"[2/5] grad-accum under DDP (no_sync window) == single (max|Δw| {diff:.2e}) ... OK")


def test_checkpoint_layout(ckpt):
    for key in ("model_state_dict", "vocab_size", "d_model", "num_layers",
                "n_heads", "use_deltanet", "weightless"):
        assert key in ckpt, f"distributed checkpoint missing config key {key!r}"
    fresh = _build_model()
    fresh.load_state_dict(ckpt["model_state_dict"])  # raises on layout drift
    print("[3/5] rank-0 checkpoint keeps the single-GPU layout (serving contract) ... OK")


def test_default_path_untouched():
    assert not (dist.is_available() and dist.is_initialized()), \
        "single-process runs must not create a process group"
    assert hvn_dist.get_world_size() == 1 and hvn_dist.is_main_process()
    print("[4/5] default single-process path untouched (no process group) ... OK")


def test_deltanet_parity():
    ref = _single_process_reference(batch_size=4, grad_accum=1, use_deltanet=True)
    ckpt = _run_ddp(batch_size=2, grad_accum=1, use_deltanet=True)
    assert ckpt["use_deltanet"] is True
    diff = (_weights_flat(ref) - _weights_flat(ckpt["model_state_dict"])).abs().max().item()
    assert diff < TOL, f"DeltaNet DDP != single: max|Δw| = {diff}"
    print(f"[5/5] DeltaNet (#34) block parity under DDP (max|Δw| {diff:.2e}) ... OK")


if __name__ == "__main__":
    print("Issue #65 — DDP data-parallel parity tests (2-process gloo, CPU)")
    print("-" * 68)
    ckpt = test_ddp_parity()
    test_grad_accum_parity()
    test_checkpoint_layout(ckpt)
    test_default_path_untouched()
    test_deltanet_parity()
    print("-" * 68)
    print("All 5 Issue #65 checks passed.")
