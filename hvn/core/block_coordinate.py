"""Block-coordinate-descent training (BAdam, Luo et al. 2024, arXiv:2404.02827).

Trains one parameter block at a time (round-robin over layers), instead of the
whole model at once. Every block is trained over the course of a full cycle --
nothing is skipped -- but only ONE block's Adam moments are ever VRAM-resident,
because the outgoing block's optimizer state is explicitly deleted on switch
(PyTorch's lazy state allocation alone is not sufficient: an inactive block
never re-accumulates state, but a block that *was* active still has its old
state sitting in the optimizer until something removes it).

Reported memory: 2M + (16/D)*M GB for M billion params (D blocks), vs 18M GB
for vanilla full AdamW -- matches this project's own measured 16.8 bytes/param
(Issue #69) almost exactly. Reported real-hardware result: full-parameter
fine-tune of Llama-3-8B on a single RTX 3090 (24 GB).

Training-time only -- the forward pass and model identity are untouched. No
change to which computation happens, only to which parameters are allowed to
accumulate gradients and hold optimizer state at any given moment.
"""

from __future__ import annotations

import torch
import torch.nn as nn


def partition_named_blocks(model: nn.Module) -> list[tuple[str, list[nn.Parameter]]]:
    """Partition a model's parameters into named blocks for BAdam-style cycling.

    For an EdgeHVN model (``model.blocks`` is a per-layer ``nn.ModuleList``):
    one block for the token embedding, one block per layer (matches the
    hybrid ``block_plan`` -- DeltaNet and attention layers partition the same
    way), and one block for ``final_norm`` + ``decoder``. This is BAdam's own
    recommended default ("by transformer modules").

    Falls back to one block per named top-level child for other module
    shapes (e.g. ``shared_weights=True`` EdgeHVN, or a plain nn.Module).
    """
    blocks: list[tuple[str, list[nn.Parameter]]] = []

    if getattr(model, "blocks", None) is not None:
        blocks.append(("token_embedding", list(model.token_embedding.parameters())))
        for i, blk in enumerate(model.blocks):
            blocks.append((f"blocks.{i}", list(blk.parameters())))
        tail = list(model.final_norm.parameters()) + list(model.decoder.parameters())
        blocks.append(("final_norm+decoder", tail))
    else:
        for name, child in model.named_children():
            params = list(child.parameters())
            if params:
                blocks.append((name, params))

    _validate_partition(model, blocks)
    return blocks


def _validate_partition(model: nn.Module, blocks: list[tuple[str, list[nn.Parameter]]]) -> None:
    seen: set[int] = set()
    for name, params in blocks:
        for p in params:
            if id(p) in seen:
                raise ValueError(f"parameter appears in more than one block (in {name!r})")
            seen.add(id(p))
    all_params = {id(p) for p in model.parameters()}
    if seen != all_params:
        missing = len(all_params - seen)
        extra = len(seen - all_params)
        raise ValueError(
            f"block partition does not cover model.parameters() exactly: "
            f"{missing} missing, {extra} extra"
        )


class BlockCoordinateScheduler:
    """Round-robin block-coordinate-descent scheduler (BAdam algorithm).

    Usage::

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        sched = BlockCoordinateScheduler(model, optimizer, steps_per_block=100)
        for step in range(max_steps):
            loss = compute_loss(...)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()          # only the active block has grad -> only
                                       # it gets updated (PyTorch skips grad=None)
            sched.step()               # advances the cycle; switches block +
                                       # clears the outgoing block's optim state
                                       # every `steps_per_block` calls

    Only one block has ``requires_grad=True`` at a time. On every switch, the
    OUTGOING block's optimizer state is explicitly deleted and its `.grad` is
    cleared (stale `.grad` tensors are not freed by `requires_grad_(False)`
    alone). The INCOMING block starts with fresh (implicitly zeroed, via
    PyTorch's own lazy allocation) optimizer state, matching the paper's
    Algorithm 1 ("clear memory for grad and optim states ... starting with
    zero initial states for every new active block").
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps_per_block: int = 100,
        order: str | list[int] = "sequential",
    ):
        if steps_per_block < 1:
            raise ValueError("steps_per_block must be >= 1")
        self.blocks = partition_named_blocks(model)
        self.optimizer = optimizer
        self.steps_per_block = steps_per_block

        n = len(self.blocks)
        if order == "sequential":
            self.order = list(range(n))
        elif order == "reverse":
            self.order = list(range(n - 1, -1, -1))
        elif isinstance(order, list):
            if sorted(order) != list(range(n)):
                raise ValueError(f"order must be a permutation of range({n})")
            self.order = list(order)
        else:
            raise ValueError(f"unknown order {order!r}")

        self._cycle_pos = 0
        self._step_in_block = 0
        self._active_idx = self.order[0]
        self._activate(self._active_idx)

    @property
    def active_block_name(self) -> str:
        return self.blocks[self._active_idx][0]

    @property
    def num_blocks(self) -> int:
        return len(self.blocks)

    def _freeze_all(self) -> None:
        for _, params in self.blocks:
            for p in params:
                p.requires_grad_(False)

    def _activate(self, idx: int) -> None:
        self._freeze_all()
        _, active_params = self.blocks[idx]
        for p in active_params:
            p.requires_grad_(True)
        for i, (_, params) in enumerate(self.blocks):
            if i == idx:
                continue
            for p in params:
                if p in self.optimizer.state:
                    del self.optimizer.state[p]
                p.grad = None

    def step(self) -> bool:
        """Call once per optimizer step. Returns True iff a block switch happened."""
        self._step_in_block += 1
        if self._step_in_block >= self.steps_per_block:
            self._step_in_block = 0
            self._cycle_pos = (self._cycle_pos + 1) % len(self.order)
            self._active_idx = self.order[self._cycle_pos]
            self._activate(self._active_idx)
            return True
        return False

    def state_dict(self) -> dict:
        return {
            "cycle_pos": self._cycle_pos,
            "step_in_block": self._step_in_block,
            "active_idx": self._active_idx,
            "order": self.order,
            "steps_per_block": self.steps_per_block,
        }

    def load_state_dict(self, sd: dict) -> None:
        self._cycle_pos = sd["cycle_pos"]
        self._step_in_block = sd["step_in_block"]
        self.order = sd["order"]
        self.steps_per_block = sd["steps_per_block"]
        self._active_idx = sd["active_idx"]
        self._activate(self._active_idx)
