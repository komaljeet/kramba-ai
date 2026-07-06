"""
Distributed data-parallel training helpers (Issue #65)
======================================================

Part of the frontier-readiness epic (#64). This module is the single place
`train.py` / `auto_train.py` touch torch.distributed, so the default
single-GPU path stays byte-for-byte unchanged when distribution is off
(explicit-flag discipline, cf. the #50 ``--full`` design).

Design constraints (from the issue / §1.1 frontier checklist):

* **Provable parity.** A 2-process gloo (CPU) run must train weights equal
  (within fp reduction tolerance) to a single-process run over the same
  data + seed — mirroring the #46 ``batched == single-stream`` and #50
  ``sync-removed == original`` proofs. See ``tests/test_issue65.py``.
* **Per-epoch (not per-step) metric sync.** Loss all-reduce extends the #50
  ``defer_loss_sync`` lever — one collective per epoch, never per step.
* **Rank-0-only side effects.** Checkpoints, resume state, and metrics files
  are written by rank 0 only; every other rank waits at a barrier.
* **O(1) inference untouched.** This changes *training orchestration* only:
  the saved checkpoint has exactly the single-GPU ``state_dict`` layout
  (DDP/FSDP wrappers are unwrapped/consolidated before saving).

Launch (single node, K processes)::

    torchrun --nproc_per_node=K train.py --distributed [--fsdp] ...
    torchrun --nproc_per_node=K auto_train.py --distributed --from base.pt ...

The backend defaults to NCCL when CUDA is available and gloo otherwise, so
the same commands drive both the laptop parity tests (CPU/gloo) and a real
multi-GPU box (NCCL).
"""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch
import torch.distributed as dist


def distributed_env_present() -> bool:
    """True when launched under torchrun / a SLURM-style launcher."""
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def init_distributed(backend: str | None = None) -> tuple[int, int, int]:
    """Initialise the process group from the standard env vars.

    Returns ``(rank, world_size, local_rank)``. Safe to call when the group
    is already initialised. Backend: NCCL if CUDA is available, else gloo
    (the CPU parity-test path), unless overridden.
    """
    if is_distributed():
        return get_rank(), get_world_size(), int(os.environ.get("LOCAL_RANK", 0))
    if not distributed_env_present():
        raise RuntimeError(
            "--distributed requires a launcher: "
            "torchrun --nproc_per_node=K train.py --distributed ..."
        )
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    return dist.get_rank(), dist.get_world_size(), local_rank


def is_distributed() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    return dist.get_world_size() if is_distributed() else 1


def is_main_process() -> bool:
    return get_rank() == 0


def barrier() -> None:
    if is_distributed():
        dist.barrier()


def cleanup() -> None:
    if is_distributed():
        dist.destroy_process_group()


def all_reduce_mean_(t: torch.Tensor) -> torch.Tensor:
    """In-place mean all-reduce (one collective; used once per epoch)."""
    if is_distributed():
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t /= get_world_size()
    return t


def wrap_model(model: torch.nn.Module, device: torch.device, fsdp: bool = False):
    """Wrap an already-on-device model for data-parallel training.

    DDP replicates the model and averages gradients — because every rank's
    micro-batch is the same size, the averaged gradient equals the gradient
    of the equivalent single-process batch (the parity the #65 test proves).
    FSDP additionally shards params/optimizer state for models that outgrow
    one card.
    """
    if not is_distributed():
        return model
    if fsdp:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        return FSDP(model, device_id=device if device.type == "cuda" else None)
    from torch.nn.parallel import DistributedDataParallel as DDP

    # The default HVN block still carries legacy per-head modules that the
    # batched forward (#8/#46) bypasses, so their params never get gradients —
    # DDP must be told to skip them. The DeltaNet block (#34) has no unused
    # params, so the recommended frontier configuration avoids the
    # find_unused_parameters traversal overhead.
    find_unused = not getattr(model, "use_deltanet", False)
    device_ids = [device.index] if device.type == "cuda" else None
    return DDP(model, device_ids=device_ids, find_unused_parameters=find_unused)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Peel DDP/compile wrappers so checkpoints keep the single-GPU layout."""
    inner = getattr(model, "module", model)          # DDP
    return getattr(inner, "_orig_mod", inner)        # torch.compile


def consolidated_state_dict(model: torch.nn.Module) -> dict:
    """A full (unsharded) ``state_dict`` with single-GPU key layout.

    For DDP / plain models this is just the unwrapped ``state_dict``. For
    FSDP it gathers the shards to rank 0 (other ranks get an empty dict) —
    the resulting checkpoint round-trips through ``generate.load_model`` and
    ``web/hvn_worker`` unchanged (the #67 serving contract).
    """
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
    except ImportError:  # ancient torch without FSDP
        return unwrap_model(model).state_dict()

    if isinstance(model, FSDP):
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
            return model.state_dict()
    return unwrap_model(model).state_dict()


def grad_sync_context(model: torch.nn.Module, sync: bool):
    """``model.no_sync()`` for the non-boundary micro-batches of a gradient-
    accumulation window, so DDP performs one gradient all-reduce per optimizer
    step (not per micro-batch) and accumulation stays equivalent to the
    single-process schedule."""
    if not sync and hasattr(model, "no_sync"):
        return model.no_sync()
    return nullcontext()
