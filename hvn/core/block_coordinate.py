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

from collections import OrderedDict

import torch
import torch.nn as nn


class GPUBufferPool:
    """Pre-allocated, reusable GPU buffers for offloaded block parameters
    (Issue #101), with an LRU residency cache on top (Issue #103). Every
    parameter swap under naive offload (``p.data = p.data.to(device)``)
    allocates a FRESH CUDA tensor -- under BCD's constant load/evict churn
    (dozens of swaps per training step) this fragments PyTorch's CUDA
    caching allocator badly: measured 317 reserved memory segments and
    ~290 GB of cumulative allocate/free traffic over just 6 steps, leaving
    peak *reserved* memory around 6 GiB even though peak *allocated* (truly
    live) memory was only ~1 GiB -- `empty_cache()` (per-step or
    per-eviction) barely helped (~2.6%), and ``expandable_segments``
    (PyTorch's own fix for exactly this pattern) is unsupported on Windows.

    The pool sidesteps the churn instead of working around it: buffers are
    grouped by shape signature (every DeltaNet layer at a given `d_model`
    shares one) and pooled for reuse across DIFFERENT blocks of the same
    shape -- a buffer is only freshly allocated when no free one of that
    shape exists yet.

    LRU cache (Issue #103): a checkpointed block is invoked TWICE per
    training step -- once during forward (call #1, whose activations
    checkpoint discards by design) and once during backward's recompute
    (call #2). Under pure evict-immediately-after-use, call #2 always
    re-fetches from CPU even though nothing changed (offloaded blocks are
    always frozen -- BCD guarantees only the active block has
    `requires_grad=True` -- so their data provably cannot go stale between
    call #1 and call #2). `load(key, ...)` now keeps a block's buffers
    resident (keyed by its block name, e.g. `"blocks.5"`) until either (a)
    a DIFFERENT block needs a buffer of the same shape and the cache is at
    its `cache_blocks` capacity (oldest-used evicted to make room), or (b)
    the caller explicitly calls `evict(key)` (used only when a call site
    knows for certain there is no call #2 coming, so holding the slot
    would only waste capacity another block could use). Default
    `cache_blocks=1` reproduces the OLD evict-immediately behavior exactly
    (a second block's load always evicts the first) -- pass a larger value
    to actually get cross-call reuse, sized to whatever VRAM budget the
    caller wants to spend on it.

    Eviction needs no copy at all: it just points `p.data` back at the
    untouched CPU tensor captured the first time this parameter was ever
    offloaded.
    """

    def __init__(self, device: torch.device, cache_blocks: int = 1):
        self.device = device
        self.cache_blocks = max(1, cache_blocks)
        self._free_buffers: dict[tuple, list[list[torch.Tensor]]] = {}
        self._resident: OrderedDict[str, dict] = OrderedDict()  # key -> {buffers, params, sig}
        self._cpu_home: dict[int, torch.Tensor] = {}  # id(p) -> its stable CPU tensor

    @staticmethod
    def _signature(params: list[nn.Parameter]) -> tuple:
        return tuple((tuple(p.shape), p.dtype) for p in params)

    def register_cpu_home(self, params: list[nn.Parameter], force: bool = False) -> None:
        """Capture each param's current tensor as its stable CPU home.
        Defensively requires `p.data` to actually BE on CPU at call time --
        calling this on a param that is currently GPU-resident (e.g. right
        after the caller's own `model.to(device)`, before anything has been
        evicted yet) is a safe no-op, not a silent corruption. This matters:
        an earlier version of this function had no such guard, and a caller
        (`BlockCoordinateScheduler._activate`'s construction-time eviction
        loop) called it BEFORE moving `p.data` to CPU rather than after --
        capturing the ORIGINAL GPU tensor as the "home", so every later
        `evict()` correctly ran but re-pointed `p.data` at a stale GPU
        reference instead of CPU. Caught by
        tests/test_block_coordinate_offload.py's device-residency assertions
        (gradient-parity still passed, since values were untouched -- this
        was a memory-placement bug, not a numerical one).

        `force` (Issue #117): default `False` is register-if-absent -- correct
        for a block's first-ever CPU residency (its untrained initial value)
        and for `load()`'s own internal call below (an inactive/frozen block's
        value provably cannot have changed since it was last registered, so
        there is nothing to refresh). `BlockCoordinateScheduler._activate()`'s
        per-switch deactivation loop calls this with `force=True` instead:
        a block that just finished its active tenure needs its freshly
        retrained CPU tensor (from `p.data.to("cpu")`, called immediately
        before this) to become the new authoritative home. Without `force`,
        `_cpu_home` kept pointing at whatever tensor was captured the FIRST
        time this parameter was ever found on CPU (its untrained init value)
        forever after -- silently discarding every cycle's training the next
        time `load()` fetched this block for a forward pass. A block would
        reset to its pristine initial value on every deactivation and retrain
        the identical trajectory every cycle, with zero cross-cycle learning
        retained. Found via a targeted multi-cycle repro (the existing
        parity/convergence tests only check aggregate early-vs-late loss over
        a run, which still trends down under this bug since the *currently*
        active block always shows real local descent within its own window --
        see tests/test_block_coordinate_offload.py::
        test_deactivated_block_retains_training_across_cycles.
        """
        for p in params:
            if p.data.device.type != "cpu":
                continue
            if force or id(p) not in self._cpu_home:
                self._cpu_home[id(p)] = p.data

    def load(self, key: str, params: list[nn.Parameter]) -> None:
        """Move `params` onto `self.device`, keyed by `key` (the block's
        stable name). If `key` is already resident (a cache hit -- e.g.
        call #2 of the same block, with nothing evicting it in between),
        this is a cheap no-op re-touch of LRU order, not a re-copy. On a
        miss, acquires a buffer for the shape signature (reusing a freed
        one if available, else allocating), copies CPU data in, and evicts
        the least-recently-used OTHER resident entry if this pushes the
        cache past `cache_blocks`."""
        self.register_cpu_home(params)
        if key in self._resident:
            self._resident.move_to_end(key)
            return
        sig = self._signature(params)
        free = self._free_buffers.get(sig)
        buffers = free.pop() if free else [
            torch.empty(p.shape, dtype=p.dtype, device=self.device) for p in params
        ]
        for p, buf in zip(params, buffers):
            buf.copy_(self._cpu_home[id(p)], non_blocking=True)
            p.data = buf
        self._resident[key] = {"buffers": buffers, "params": params, "sig": sig}
        while len(self._resident) > self.cache_blocks:
            oldest_key, oldest = next(iter(self._resident.items()))
            if oldest_key == key:
                break
            self._resident.popitem(last=False)
            self._release(oldest)

    def evict(self, key: str) -> None:
        """Force-release one resident entry now (used only when a call
        site knows for certain this block will not be touched again this
        step, so holding its slot would only deny a genuine reuse
        candidate a buffer -- see class docstring). No-op if not
        resident."""
        entry = self._resident.pop(key, None)
        if entry is not None:
            self._release(entry)

    def evict_all(self) -> None:
        """Force-release every resident entry now. Called by
        `BlockCoordinateScheduler._activate` before its own per-switch
        CPU-move loop, which otherwise assumes every inactive block is
        already CPU-resident -- without this, a still-cached (GPU-resident)
        block would have its buffer silently orphaned by that loop's own
        `.to("cpu")` call (values stay correct, since `.to()` copies
        current data, but the pool's bookkeeping and the CPU-home dedup in
        `register_cpu_home` would then disagree about where the real data
        lives)."""
        for entry in self._resident.values():
            self._release(entry)
        self._resident.clear()

    def _release(self, entry: dict) -> None:
        for p in entry["params"]:
            home = self._cpu_home.get(id(p))
            if home is not None:
                p.data = home
        self._free_buffers.setdefault(entry["sig"], []).append(entry["buffers"])


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
        offload_inactive_weights: bool = False,
        device: torch.device | str | None = None,
        offload_cache_blocks: int = 1,
    ):
        """
        offload_inactive_weights : bool
            Issue #101, extends #99: additionally stream every INACTIVE
            block's weights to CPU RAM, evicting them from GPU entirely
            (not just their optimizer state, which #99 alone already
            handles). #99 shrinks the "(16/D)*M" term in BAdam's memory
            formula; this attacks the "2*M" term (every parameter held as a
            bf16 tensor regardless of active/frozen status) -- the actual
            floor that caps the local ceiling at ~1.3-1.4B on this hardware
            (see docs/training.md).

            The scheduler does NOT move tensors itself here (an earlier,
            reverted attempt tried that and it silently breaks correctness
            -- see the module docstring's warning). It only tells the model
            which block name is exempt from offload (the active one);
            `EdgeHVN.forward()` does the actual load-before-use / evict-
            via-backward-hook dance, timed against each block's checkpoint
            recompute window. Requires `model.grad_checkpoint = True` and
            `device`.
        device : torch.device | str | None
            Compute device to keep the active block (and, transiently, each
            block being used) on. Required when
            `offload_inactive_weights=True`.
        offload_cache_blocks : int
            Issue #103: how many distinct inactive blocks' weights the
            offload pool keeps GPU-resident at once (LRU eviction beyond
            that). Default 1 reproduces the original evict-immediately
            behavior. Raising it lets a checkpointed block's backward
            recompute (call #2) hit an already-warm buffer instead of
            re-fetching from CPU, at the cost of `(cache_blocks - 1)` extra
            blocks' worth of GPU memory held at once -- size against
            whatever VRAM budget is available beyond the active block.
        """
        if steps_per_block < 1:
            raise ValueError("steps_per_block must be >= 1")
        if offload_inactive_weights:
            if device is None:
                raise ValueError("device is required when offload_inactive_weights=True")
            if not getattr(model, "grad_checkpoint", False):
                raise ValueError(
                    "offload_inactive_weights=True requires model.grad_checkpoint=True"
                )
        self.model = model
        self.blocks = partition_named_blocks(model)
        self.optimizer = optimizer
        self.steps_per_block = steps_per_block
        self.offload_inactive_weights = offload_inactive_weights
        self.device = torch.device(device) if device is not None else None
        self.buffer_pool = (
            GPUBufferPool(self.device, cache_blocks=offload_cache_blocks)
            if offload_inactive_weights else None
        )
        if self.buffer_pool is not None:
            model._offload_pool = self.buffer_pool

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

        if self.offload_inactive_weights and self.device.type == "cuda":
            # Issue #102: the caller's own `model.to(device)` (which must
            # happen before this constructor runs) allocates ONE large
            # reserved CUDA segment sized for the WHOLE model. The
            # `_activate` call just above correctly evicts almost
            # everything (`memory_allocated()` drops to a few MB), but
            # PyTorch's caching allocator does not proactively return freed
            # segments to the driver -- `memory_reserved()` stays pinned at
            # that original ~full-model size for the rest of the process,
            # since it is still cached as "free for PyTorch's own reuse".
            # A single `empty_cache()` HERE, while that freed memory is
            # still one clean, fully-empty block from the original
            # allocation, releases nearly all of it (measured: 5894 MiB ->
            # 64 MiB). Calling this LATER (e.g. mid-training-loop, once the
            # buffer pool's own differently-sized allocations have mixed
            # into the same segments) does not work anywhere near as well
            # (measured: only ~2.6% recovered) -- partial occupancy blocks
            # segment release. One-time cost, not per-step or per-block.
            # Measured full effect on a 1.54B config: peak reserved memory
            # 6130 MiB -> 1182 MiB (an 81% reduction), and training got
            # FASTER too (4.54s/step -> 2.53s/step), not just smaller --
            # less fragmented reserved space makes the buffer pool's own
            # allocations cheaper to satisfy.
            torch.cuda.empty_cache()

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
        active_name, active_params = self.blocks[idx]
        for p in active_params:
            p.requires_grad_(True)
        for i, (_, params) in enumerate(self.blocks):
            if i == idx:
                continue
            for p in params:
                if p in self.optimizer.state:
                    del self.optimizer.state[p]
                p.grad = None

        if self.offload_inactive_weights:
            # Tell the model which block is exempt from offload; the model's
            # forward() does the actual per-forward-pass load/evict dance for
            # the blocks it touches (via self.buffer_pool, not raw .to()
            # calls -- see GPUBufferPool's docstring for why), timed against
            # each checkpointed block's recompute window (see
            # EdgeHVN._offload_wrap_checkpointed).
            self.model._offload_active_name = active_name
            self.model._offload_device = self.device
            # The now-active block's weights must be GPU-resident (it was
            # possibly offloaded while inactive under the previous cycle).
            # register_cpu_home BEFORE moving to device -- capturing while
            # still CPU-resident is what makes it a valid "home" to evict
            # back to later, once this block becomes inactive again. (No
            # effect on the very first activation, from __init__, where
            # these params are still GPU-resident from the caller's own
            # model.to(device) -- register_cpu_home's own device guard
            # correctly skips that case as a no-op; the *inactive* loop
            # below registers the true CPU home once these params are
            # themselves evicted in a later cycle.)
            # Issue #103: flush the LRU cache's bookkeeping BEFORE the loop
            # below moves every inactive block's tensor to CPU directly --
            # without this, a block still GPU-resident in the cache (kept
            # warm across call #1/#2 within the outgoing active block's
            # tenure) would have its buffer silently orphaned: `.to("cpu")`
            # still gives correct values, but the pool would keep thinking
            # that buffer is "resident" for a key nothing points at anymore.
            self.buffer_pool.evict_all()
            self.buffer_pool.register_cpu_home(active_params)
            for p in active_params:
                p.data = p.data.to(self.device)
            # Explicitly evict every OTHER block right now too. Without
            # this, a block only ever gets evicted the next time forward()
            # actually runs and touches it -- fine in steady state, but the
            # very first activation (called from __init__, right after the
            # caller's own `model.to(device)` already put every parameter
            # on the GPU) would otherwise leave the ENTIRE model resident
            # until the first forward() call gets around to evicting each
            # block one by one. Measured: this left peak memory at ~5.9 GiB
            # for a 1.54B model *before a single training step ran* --
            # caught by inspecting torch.cuda.memory_allocated() directly
            # around a single instrumented step, not by the unit tests
            # (which never checked memory *before* the first forward call).
            for i, (_, params) in enumerate(self.blocks):
                if i == idx:
                    continue
                # Move to CPU FIRST, then register -- register_cpu_home
                # must observe the tensor actually on CPU to capture it as
                # a valid home (see its docstring for the bug this order
                # fixes: registering before the move captured the stale
                # GPU tensor instead).
                for p in params:
                    p.data = p.data.to("cpu")
                # force=True (Issue #117): this block just finished its
                # active tenure (or was already inactive, in which case
                # `.to("cpu")` above was a no-op and this harmlessly
                # re-registers the same tensor) -- its current CPU tensor
                # must become the new authoritative home so future load()
                # calls see any training it just received, not the value
                # captured the first time this parameter was ever offloaded.
                self.buffer_pool.register_cpu_home(params, force=True)

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
