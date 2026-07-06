"""
Multi-node launch + sharded-checkpoint I/O (Issue #67)
======================================================

Part of the frontier-readiness epic (#64). The plumbing that turns the #65
data-parallel / #66 tensor-parallel paths into an actual multi-node job:

* :func:`resolve_launch_env` / :func:`init_from_launcher` — process-group
  bring-up from the standard ``torchrun`` env vars, with **SLURM translation**
  (``SLURM_PROCID``/``SLURM_NTASKS``/``SLURM_LOCALID``) and an NCCL→gloo
  fallback so the same code path drives the CPU parity tests and a GPU
  cluster. Exact launch commands: ``docs/DISTRIBUTED_LAUNCH.md``.
* :func:`save_consolidated_checkpoint` — collect a (possibly DDP-wrapped or
  FSDP-sharded) model into **the same single-file checkpoint layout** the
  serving stack loads today (``generate.load_model`` /
  ``web/hvn_worker/worker.py``): ``{model_state_dict, vocab_size, d_model,
  num_layers, n_heads, n_threads, chunk_size, use_deltanet,
  deltanet_conv_kernel, weightless, weightless_seed}``. Rank-0-only write,
  atomic (tmp + replace, matching the #54 resume discipline), followed by a
  barrier — so a distributed run always produces a checkpoint registrable as
  a ``ModelVersion(checkpoint_path=...)`` with **no serving-side changes**.

O(1) inference untouched: this module only moves training-side bytes; what
lands on disk is exactly the single-GPU layout.
"""

from __future__ import annotations

import os
from datetime import timedelta

import torch
import torch.distributed as dist


# ---------------------------------------------------------------------------
# Launcher environment
# ---------------------------------------------------------------------------

def resolve_launch_env() -> dict | None:
    """Normalise the launcher environment to torchrun's contract.

    Returns ``{"rank", "world_size", "local_rank", "master_addr",
    "master_port"}`` or ``None`` when not under any launcher.

    Precedence: torchrun env (``RANK``/``WORLD_SIZE``) wins; otherwise SLURM
    (``srun python ...``) is translated — ``SLURM_PROCID → RANK``,
    ``SLURM_NTASKS → WORLD_SIZE``, ``SLURM_LOCALID → LOCAL_RANK``. Under
    SLURM, ``MASTER_ADDR``/``MASTER_PORT`` must be exported by the batch
    script (the standard one-liner is in docs/DISTRIBUTED_LAUNCH.md) — we
    fail loudly rather than guess a head node.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        src = os.environ
        rank, world = int(src["RANK"]), int(src["WORLD_SIZE"])
        local = int(src.get("LOCAL_RANK", 0))
    elif "SLURM_PROCID" in os.environ:
        rank = int(os.environ["SLURM_PROCID"])
        world = int(os.environ["SLURM_NTASKS"])
        local = int(os.environ.get("SLURM_LOCALID", 0))
        if "MASTER_ADDR" not in os.environ:
            raise RuntimeError(
                "SLURM job without MASTER_ADDR/MASTER_PORT: export them in "
                "the sbatch script (see docs/DISTRIBUTED_LAUNCH.md)")
        # Propagate the translation so init_method='env://' just works.
        os.environ.setdefault("RANK", str(rank))
        os.environ.setdefault("WORLD_SIZE", str(world))
        os.environ.setdefault("LOCAL_RANK", str(local))
    else:
        return None
    return {
        "rank": rank, "world_size": world, "local_rank": local,
        "master_addr": os.environ.get("MASTER_ADDR", "127.0.0.1"),
        "master_port": os.environ.get("MASTER_PORT", "29500"),
    }


def init_from_launcher(backend: str | None = None,
                       timeout_s: int = 1800) -> tuple[int, int, int]:
    """Initialise the process group from the launcher env (torchrun/SLURM).

    Backend: **NCCL** when CUDA is available (multi-GPU/multi-node fabric),
    **gloo** otherwise — the same commands drive the laptop parity tests and
    a cluster. Returns ``(rank, world_size, local_rank)`` and pins the CUDA
    device to ``local_rank``.
    """
    env = resolve_launch_env()
    if env is None:
        raise RuntimeError(
            "no launcher detected — start with torchrun (or srun); see "
            "docs/DISTRIBUTED_LAUNCH.md for the exact commands")
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank(), dist.get_world_size(), env["local_rank"]
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend, init_method="env://",
                            timeout=timedelta(seconds=timeout_s))
    if torch.cuda.is_available():
        torch.cuda.set_device(env["local_rank"])
    return dist.get_rank(), dist.get_world_size(), env["local_rank"]


# ---------------------------------------------------------------------------
# Checkpoint I/O — consolidate shards to the single-file serving layout
# ---------------------------------------------------------------------------

def _unwrap(model: torch.nn.Module) -> torch.nn.Module:
    inner = getattr(model, "module", model)          # DDP
    return getattr(inner, "_orig_mod", inner)        # torch.compile


def _full_state_dict(model: torch.nn.Module) -> dict:
    """Full (unsharded) state dict with single-GPU key layout.

    * plain / DDP / compiled models → the unwrapped ``state_dict``;
    * FSDP models → rank-0-only full-state-dict gather (CPU-offloaded), the
      collective every rank must enter.
    """
    try:
        from torch.distributed.fsdp import (
            FullyShardedDataParallel as FSDP,
            FullStateDictConfig,
            StateDictType,
        )
        if isinstance(model, FSDP):
            cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
            with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT, cfg):
                return model.state_dict()
    except ImportError:
        pass
    return _unwrap(model).state_dict()


def checkpoint_config(model: torch.nn.Module) -> dict:
    """The non-weights config keys of the standard checkpoint layout."""
    m = _unwrap(model)
    return {
        "vocab_size": m.vocab_size,
        "d_model": m.d_model,
        "num_layers": m.num_layers,
        "n_heads": m.n_heads,
        "n_threads": m.n_threads,
        "chunk_size": m.blocks[0].chunk_size if hasattr(m.blocks[0], "chunk_size") else 64,
        "use_deltanet": getattr(m, "use_deltanet", False),
        "deltanet_conv_kernel": getattr(m.blocks[0], "conv_kernel", 4),
        "weightless": getattr(m, "weightless", False),
        "weightless_seed": getattr(m, "weightless_seed", 1234),
    }


def save_consolidated_checkpoint(model: torch.nn.Module, save_path: str) -> None:
    """Save a (possibly DDP/FSDP-wrapped) model as a standard single-file
    checkpoint that ``generate.load_model`` / the hvn_worker load unchanged.

    Every rank must call this (the FSDP gather is a collective); only rank 0
    writes, atomically (tmp + ``os.replace``, the #54 discipline — a killed
    job never leaves a truncated checkpoint), and everyone leaves together at
    a barrier so the file is guaranteed on disk for all ranks.
    """
    state_dict = _full_state_dict(model)
    is_dist = dist.is_available() and dist.is_initialized()
    if not is_dist or dist.get_rank() == 0:
        payload = {"model_state_dict": state_dict, **checkpoint_config(model)}
        tmp = f"{save_path}.tmp"
        torch.save(payload, tmp)
        os.replace(tmp, save_path)
    if is_dist:
        dist.barrier()
