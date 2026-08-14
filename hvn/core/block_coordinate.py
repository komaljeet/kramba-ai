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


def _dtype_itemsize(dtype: torch.dtype) -> int:
    """Bytes per element for `dtype` (Issue #113: used to decide whether
    parking at `park_dtype` actually shrinks a given parameter, rather than
    hardcoding an assumption that native dtype is always fp32)."""
    return torch.empty((), dtype=dtype).element_size()


class _PinnedStagingPool:
    """Bounded pool of pinned (page-locked) CPU staging buffers, keyed by
    shape+dtype signature (Issue #118).

    `GPUBufferPool.prefetch()` originally pinned each block's ACTUAL CPU
    home tensor directly, and left it pinned indefinitely -- only replaced
    (and thus un-pinned) the next time that specific block completed a
    full BCD activate/deactivate cycle. But a single forward pass
    prefetches EVERY block in sequence (not just the BCD-active one), so
    nothing released a block's pinning between "prefetched" and
    "consumed" within one forward pass -- pinned memory grew with total
    LAYER COUNT, not with anything bounded. Measured: ~8 GiB pinned
    partway through a single forward pass at 95 layers (~4B params),
    crashing with a CUDA out-of-memory error from inside `pin_memory()`
    itself -- not GPU VRAM (peak reserved was 64 MiB at the point of
    failure) and not regular CPU RAM (the model's bf16 CPU footprint was
    ~8 GiB total, well under this machine's ~15.3 GiB RAM) -- specifically
    the OS/driver's pinned-memory ceiling, which is far tighter than
    either. Smaller test/benchmark configs (3-4 layers in unit tests,
    24-36 layers in earlier benchmarks) never pinned enough cumulative
    memory to hit this, so it went undetected until real depth.

    Fix: pin a small, bounded number of reusable STAGING buffers instead
    of the home tensors themselves. `prefetch()` copies a block's
    (unpinned) CPU home data into a staging buffer (a fast plain host
    memcpy -- not PCIe-bound, so this doesn't reintroduce a transfer-time
    cost) just before issuing the actual async H2D copy FROM that pinned
    staging buffer. Since this project's blocks are shape-homogeneous per
    layer type (all `blocks.i` at a given d_model share identical
    per-parameter shapes), the number of distinct shape signatures is
    small and independent of total layer count -- so pinned memory is
    bounded by `max_per_shape` buffers per shape, not by depth.
    """

    def __init__(self, max_per_shape: int = 2):
        self.max_per_shape = max(1, max_per_shape)
        self._free: dict[tuple, list[torch.Tensor]] = {}
        self._in_use: dict[tuple, list[torch.Tensor]] = {}

    def acquire(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        sig = (tuple(shape), dtype)
        free = self._free.get(sig)
        buf = free.pop() if free else None
        if buf is None:
            in_use = self._in_use.setdefault(sig, [])
            if len(in_use) >= self.max_per_shape:
                # Rare fallback (shouldn't happen in the normal sequential
                # prefetch-then-consume-next-iteration pattern this pool is
                # sized for): every buffer of this shape is still in use.
                # A full device sync is a blunt but always-correct way to
                # guarantee it's safe to reclaim one -- acceptable here
                # since this branch is not the common hot path.
                torch.cuda.synchronize()
                buf = in_use.pop(0)
            else:
                buf = torch.empty(shape, dtype=dtype).pin_memory()
        self._in_use.setdefault(sig, []).append(buf)
        return buf

    def release(self, buf: torch.Tensor) -> None:
        """Return `buf` to the free pool. Caller MUST have already
        host-synchronized on any event tracking a copy FROM this buffer
        (`event.synchronize()`, not just a stream-level `wait_event()` --
        see `GPUBufferPool.load`) before calling this, since the buffer's
        memory becomes safe to overwrite again the moment this is called."""
        sig = (tuple(buf.shape), buf.dtype)
        in_use = self._in_use.get(sig)
        if in_use:
            for i, t in enumerate(in_use):
                if t is buf:
                    in_use.pop(i)
                    break
        self._free.setdefault(sig, []).append(buf)


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

    def __init__(
        self,
        device: torch.device,
        cache_blocks: int = 1,
        park_dtype: torch.dtype | None = None,
    ):
        """
        park_dtype : torch.dtype | None
            Issue #113: dtype to store INACTIVE blocks' weights in on CPU
            (e.g. `torch.bfloat16`). `None` (default) preserves the
            original behavior exactly -- park at whatever dtype the
            parameter already is. When set, a parameter is only downcast
            if its native dtype (captured via `capture_native_dtype`,
            called once per param before any parking ever happens) is
            LOWER precision than `park_dtype`... no -- HIGHER precision,
            i.e. only actually shrinks storage (e.g. fp32 -> bf16); a
            param whose native dtype is already <= park_dtype in size is
            left alone (nothing to save, and downcasting further isn't
            this lever's job -- see Issue #114 for the fp8 spike).
            `BlockCoordinateScheduler._activate()`'s active-branch always
            restores a param to its NATIVE dtype before training resumes
            on it -- the optimizer's tiny per-step updates need the
            parameter's original (typically fp32) precision to be
            representable at all; only the PARKED/inactive copy is
            allowed to be coarser, matching the reasoning in Issue #113's
            writeup (bf16 is already this project's native working
            precision elsewhere, so parking already-frozen weights in it
            adds no precision loss beyond what training already tolerates).
        """
        self.device = device
        self.cache_blocks = max(1, cache_blocks)
        self.park_dtype = park_dtype
        self._free_buffers: dict[tuple, list[list[torch.Tensor]]] = {}
        self._resident: OrderedDict[str, dict] = OrderedDict()  # key -> {buffers, params, sig}
        self._cpu_home: dict[int, torch.Tensor] = {}  # id(p) -> its stable CPU tensor
        self._native_dtype: dict[int, torch.dtype] = {}  # id(p) -> its true training dtype
        # Issue #115: a side CUDA stream for prefetching the NEXT block's
        # CPU->GPU copy while compute for the CURRENT block runs on the
        # main/default stream. None on non-CUDA devices (prefetch() then
        # silently falls back to a plain synchronous load -- see there).
        self._prefetch_stream = torch.cuda.Stream(device=device) if device.type == "cuda" else None
        self._pending_events: dict[str, torch.cuda.Event] = {}  # key -> H2D-copy-done event
        # Issue #118: bounded pinned staging buffers for prefetch() -- see
        # _PinnedStagingPool's docstring for why pinning home tensors
        # directly (the original #115 design) OOMs at real depth.
        self._pinned_pool = _PinnedStagingPool()
        self._staging_in_use: dict[str, list[torch.Tensor]] = {}  # key -> its staging buffers

    def capture_native_dtype(self, params: list[nn.Parameter]) -> None:
        """Record each param's true (training-time) dtype, once, before any
        parking ever happens. Idempotent -- a param already captured is
        left alone, so calling this repeatedly (e.g. every `_activate()`)
        is safe and never overwrites with a since-downcast parked dtype."""
        for p in params:
            self._native_dtype.setdefault(id(p), p.dtype)

    def native_dtype_for(self, p: nn.Parameter) -> torch.dtype:
        """The dtype `p` was originally constructed/trained at -- what it
        must be restored to before becoming the BCD-active block again."""
        return self._native_dtype.get(id(p), p.dtype)

    def park_dtype_for(self, p: nn.Parameter) -> torch.dtype:
        """The dtype `p` should be stored at while parked (inactive)."""
        native = self.native_dtype_for(p)
        if self.park_dtype is not None and _dtype_itemsize(self.park_dtype) < _dtype_itemsize(native):
            return self.park_dtype
        return native

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

    def _acquire_buffers(self, key: str, params: list[nn.Parameter]) -> list[torch.Tensor]:
        """Get (reusing a freed one if available, else allocating) a set of
        GPU buffers for `params`' shape signature, and register `key` as
        resident -- evicting the least-recently-used OTHER entry if this
        pushes the cache past `cache_blocks`. Does NOT copy data or touch
        `p.data` -- shared by both `load()` and `prefetch()`, which differ
        only in how/when the actual H2D copy happens."""
        sig = self._signature(params)
        free = self._free_buffers.get(sig)
        buffers = free.pop() if free else [
            torch.empty(p.shape, dtype=p.dtype, device=self.device) for p in params
        ]
        self._resident[key] = {"buffers": buffers, "params": params, "sig": sig}
        while len(self._resident) > self.cache_blocks:
            oldest_key, oldest = next(iter(self._resident.items()))
            if oldest_key == key:
                break
            self._resident.popitem(last=False)
            # Issue #115: an evicted-for-capacity entry could have an
            # in-flight prefetch nobody ever consumed via load() -- must
            # synchronize before its buffer goes back to _free_buffers,
            # same reasoning as evict()/evict_all().
            self._wait_and_clear_pending(oldest_key)
            self._release(oldest)
        return buffers

    def _offloadable(self, params: list[nn.Parameter]) -> list[nn.Parameter]:
        """Params that actually have a CPU home, i.e. ones that get offloaded.

        A param with no registered home was never evicted and is already
        GPU-resident, so there is nothing to fetch. This is not a rare edge
        case: with tied embeddings (#106) the decoder's weight IS
        token_embedding.weight, which lives in the embedding block and stays
        resident while that block is active -- so the tail block's parameter
        list contains a tensor the pool has never owned. Before this,
        prefetching the tail raised KeyError and the two features could not be
        combined at all.
        """
        return [q for q in params if id(q) in self._cpu_home]

    def load(self, key: str, params: list[nn.Parameter]) -> None:
        """Move `params` onto `self.device`, keyed by `key` (the block's
        stable name). If `key` is already resident (a cache hit -- e.g.
        call #2 of the same block, with nothing evicting it in between, OR
        a block `prefetch()`-ed ahead of when it's actually needed -- Issue
        #115), this is a cheap re-touch of LRU order, not a re-copy --
        except if a prefetch's H2D copy is still in flight, in which case
        the CURRENT stream is made to wait on its completion event first
        (a GPU-side dependency, not a host-blocking sync: compute already
        queued on this stream simply won't start until the copy is done).
        On a miss, acquires a buffer for the shape signature and copies CPU
        data in synchronously on the current stream (the pre-#115 behavior,
        unchanged for any block that was never prefetched)."""
        self.register_cpu_home(params)
        if key in self._resident:
            self._resident.move_to_end(key)
            event = self._pending_events.pop(key, None)
            if event is not None:
                torch.cuda.current_stream(self.device).wait_event(event)
                # Issue #118: wait_event() above is a cheap GPU-side stream
                # dependency, NOT a host-blocking sync -- it doesn't by
                # itself guarantee the copy is done from the HOST's
                # perspective right now. A host-side event.synchronize()
                # is required before it's safe to hand this key's pinned
                # staging buffers back to the pool for reuse by a plain
                # (host-side, immediate) memcpy in a future prefetch() --
                # otherwise a new copy could start overwriting a staging
                # buffer's memory while the old async H2D copy reading
                # from it (via wait_event's dependency, not yet actually
                # complete) is still in flight. Cheap in practice: by the
                # time load() runs, the transfer has usually already
                # finished (it was kicked off during the PREVIOUS block's
                # compute), so this typically returns instantly rather
                # than actually blocking.
                event.synchronize()
                self._release_staging(key)
            return
        params = self._offloadable(params)
        if not params:
            return
        buffers = self._acquire_buffers(key, params)
        for p, buf in zip(params, buffers):
            buf.copy_(self._cpu_home[id(p)], non_blocking=True)
            p.data = buf

    def prefetch(self, key: str, params: list[nn.Parameter]) -> None:
        """Issue #115: kick off `key`'s CPU->GPU copy on a dedicated side
        stream, so it can run CONCURRENTLY with compute already queued on
        the main/default stream for a DIFFERENT (e.g. currently-active or
        currently-computing) block -- instead of `load()`'s synchronous
        copy, which the calling stream must wait through before any
        compute using this data can proceed.

        No-op (falls back to nothing -- caller's later `load()` call will
        do the synchronous copy as normal) if this device has no CUDA
        stream (non-CUDA device) or if `key` is already resident (nothing
        to prefetch -- either a genuine cache hit, or a previous prefetch
        for this key hasn't been consumed yet).

        Requires a PINNED (page-locked) source for the H2D copy to
        actually run asynchronously -- PyTorch silently falls back to a
        synchronous copy for `non_blocking=True` on regular (pageable) CPU
        memory, which would defeat the entire point of this method. Issue
        #118: rather than pinning the CPU home tensor itself (which OOMs
        the OS/driver's pinned-memory ceiling at real depth -- see
        `_PinnedStagingPool`'s docstring), each param's home data is first
        copied (a fast plain host memcpy, not PCIe-bound) into a small,
        bounded, reusable pinned STAGING buffer, and the actual async H2D
        copy reads from that instead.
        """
        self.register_cpu_home(params)
        if self._prefetch_stream is None or key in self._resident:
            return
        params = self._offloadable(params)
        if not params:
            return
        buffers = self._acquire_buffers(key, params)
        staging_bufs = []
        with torch.cuda.stream(self._prefetch_stream):
            for p, buf in zip(params, buffers):
                home = self._cpu_home[id(p)]
                staging = self._pinned_pool.acquire(home.shape, home.dtype)
                staging.copy_(home)
                buf.copy_(staging, non_blocking=True)
                p.data = buf
                staging_bufs.append(staging)
        event = torch.cuda.Event()
        event.record(self._prefetch_stream)
        self._pending_events[key] = event
        self._staging_in_use[key] = staging_bufs

    def _release_staging(self, key: str) -> None:
        """Return `key`'s staging buffers (if any) to the pinned pool.
        Caller must have already host-synchronized (`event.synchronize()`)
        on `key`'s pending event first -- see `_PinnedStagingPool.release`."""
        bufs = self._staging_in_use.pop(key, None)
        if bufs:
            for b in bufs:
                self._pinned_pool.release(b)

    def _wait_and_clear_pending(self, key: str) -> None:
        """Issue #115: if `key` has a prefetch H2D copy still in flight
        (never consumed by a `load()` call), synchronize on it before its
        buffer gets handed back to `_free_buffers` for reuse elsewhere --
        without this, a NEW copy could start writing into the same buffer
        memory while the OLD (still in-flight) prefetch copy is still
        writing to it, a real data race, not just a staleness concern."""
        event = self._pending_events.pop(key, None)
        if event is not None:
            event.synchronize()
            self._release_staging(key)

    def evict(self, key: str) -> None:
        """Force-release one resident entry now (used only when a call
        site knows for certain this block will not be touched again this
        step, so holding its slot would only deny a genuine reuse
        candidate a buffer -- see class docstring). No-op if not
        resident."""
        self._wait_and_clear_pending(key)
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
        for key in list(self._resident.keys()):
            self._wait_and_clear_pending(key)
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

    # Issue #151: with tie_embeddings the token-embedding table and the decoder
    # output projection are the SAME nn.Parameter under two names, so a naive
    # walk places it in two blocks and _validate_partition rejects the model.
    # That combination looks exotic but is the documented 355M config:
    # --tie-embeddings has defaulted ON since #139, and #112 records
    # --block-coordinate as MANDATORY at that size -- so a 355M BCD run raised
    # ValueError at scheduler construction. Each parameter is assigned to the
    # FIRST block that claims it; later duplicates are skipped.
    #
    # Consequence, deliberate and worth knowing: a tied table belongs to
    # `token_embedding`, so it trains only while that block is active and does
    # NOT also update during the tail block's turn. That is the honest reading of
    # block-coordinate descent on a shared weight (one block trains at a time).
    # Updating it in both would silently give the shared table twice the update
    # frequency of every other parameter.
    claimed: set[int] = set()

    def _take(params) -> list[nn.Parameter]:
        out = []
        for p in params:
            if id(p) not in claimed:
                claimed.add(id(p))
                out.append(p)
        return out

    if getattr(model, "blocks", None) is not None:
        blocks.append(("token_embedding", _take(model.token_embedding.parameters())))
        for i, blk in enumerate(model.blocks):
            blocks.append((f"blocks.{i}", _take(blk.parameters())))
        tail = _take(list(model.final_norm.parameters())
                     + list(model.decoder.parameters()))
        blocks.append(("final_norm+decoder", tail))
    else:
        for name, child in model.named_children():
            params = _take(child.parameters())
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
        retain_optimizer_state: bool = False,
        offload_inactive_weights: bool = False,
        device: torch.device | str | None = None,
        offload_cache_blocks: int = 1,
        offload_park_dtype: torch.dtype | None = None,
        offload_prefetch: bool = True,
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
        offload_park_dtype : torch.dtype | None
            Issue #113: dtype to store INACTIVE blocks' weights in on CPU
            while parked (e.g. `torch.bfloat16`), instead of their native
            training dtype (typically fp32 -- see docs/training.md's
            Issue #102 section, "EdgeHVN's default parameter dtype is
            fp32"). Halves CPU RAM and per-swap PCIe transfer for parked
            blocks; the ACTIVE block is always restored to its native
            dtype before training resumes on it, so optimizer precision
            is unaffected -- only frozen, already-not-being-updated
            weights get stored coarser. `None` (default) preserves
            existing behavior exactly (park at native dtype, no
            conversion).
        offload_prefetch : bool
            Issue #115: overlap the NEXT offloadable segment's CPU->GPU
            transfer (on a side CUDA stream) with the CURRENT segment's
            compute, instead of paying the transfer serially when the next
            segment's own turn comes. Pure scheduling/timing change --
            gradients are unaffected (the existing gradient-parity test
            exercises this path, since it's on by default, and still
            passes). Default `True`; set `False` to fall back to fully
            serial (pre-#115) transfers, e.g. for A/B measurement or if
            something about a given environment makes the extra CUDA
            stream undesirable.
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
            # Issue #151 rejected tied embeddings here rather than patch this
            # subsystem blind -- correctly: it is the one #117 found silently
            # discarding all training for an epic. #157 lifts that guard, having
            # found and fixed the actual cause rather than the symptom.
            #
            # The cause was in two halves. The partition half is already fixed:
            # `_take()` above assigns the shared table to `token_embedding` only.
            # The remaining half is that the model's forward still meets the same
            # tensor through `model.decoder` when loading/prefetching the tail
            # block, and looked it up in `_cpu_home` by id -- where it was never
            # registered, because it stays resident with the embedding block.
            # `GPUBufferPool._offloadable()` now filters to params that actually
            # have a home: no home means never evicted, therefore already on the
            # device, therefore nothing to fetch.
            #
            # What makes this safe to enable rather than merely non-crashing is
            # that the guard had ALSO silently disabled 13 of this subsystem's own
            # tests -- every one of them builds a model, and tying became the
            # default in #151, so the entire offload suite raised at construction
            # instead of running. Those tests (gradient parity across cycles,
            # device residency per block, LRU reconciliation, convergence) are the
            # evidence #151 asked for, and they now execute against the tied
            # configuration. tests/test_bcd_tied_embeddings.py adds the
            # tied-specific cases, including one that breaks a CPU home on purpose
            # and requires the suite to catch it -- without that, these tests
            # could pass simply by never offloading anything.
        self.model = model
        self.blocks = partition_named_blocks(model)
        # #157: parameters reachable from more than one module must never be
        # evicted. The tied table is the case that matters: `_take()` assigns it
        # to `token_embedding`, but forward() uses it AGAIN at the end through
        # `model.decoder`, so a step where the embedding block is inactive would
        # otherwise find it parked on CPU. That is not hypothetical -- it is a
        # hard `Expected all tensors to be on the same device` from inside the
        # output projection, which is how this was found.
        #
        # Filtering the buffer pool's fetches (GPUBufferPool._offloadable) is NOT
        # sufficient on its own and was the first attempt: it stops the KeyError
        # but leaves the tensor wherever eviction put it. The tensor has to stay
        # resident, because there is no single block whose activation window
        # covers both of its uses.
        #
        # Cost is bounded and known: one vocab x d_model table, 250 MB at 1B.
        self._pinned: set[int] = set()
        _counts: dict[int, int] = {}
        _mods = [getattr(model, "token_embedding", None),
                 getattr(model, "decoder", None),
                 getattr(model, "final_norm", None)]
        _mods += list(getattr(model, "blocks", None) or [])
        for _mod in _mods:
            if _mod is None:
                continue
            for _p in dict.fromkeys(_mod.parameters()):   # unique within a module
                _counts[id(_p)] = _counts.get(id(_p), 0) + 1
        self._pinned = {k for k, v in _counts.items() if v > 1}
        self.optimizer = optimizer
        self.steps_per_block = steps_per_block
        # Issue #152: park outgoing optimizer state on CPU and restore it when
        # the block returns, instead of discarding it. Default OFF -- BCD's
        # documented behaviour (and the paper's Algorithm 1) is deletion, and
        # every measurement on record (#112's ~3.9x quality cost) was taken with
        # it off. See _activate for why the restore is deliberately undecayed.
        self.retain_optimizer_state = bool(retain_optimizer_state)
        self._parked_state: dict = {}
        self.offload_inactive_weights = offload_inactive_weights
        self.device = torch.device(device) if device is not None else None
        self.buffer_pool = (
            GPUBufferPool(
                self.device,
                cache_blocks=offload_cache_blocks,
                park_dtype=offload_park_dtype,
            )
            if offload_inactive_weights else None
        )
        if self.buffer_pool is not None:
            model._offload_pool = self.buffer_pool
            model._offload_prefetch = offload_prefetch
            # Issue #113: capture every param's true training dtype BEFORE
            # any parking/activation happens below, so park_dtype_for() can
            # later tell "should this be downcast while parked" from "is
            # this already at (or below) park_dtype natively" -- must run
            # before the very first _activate() call just below.
            for _, params in self.blocks:
                self.buffer_pool.capture_native_dtype(params)

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

        # Issue #152: with retain_optimizer_state, an outgoing block's optimizer
        # state is PARKED on CPU instead of discarded, and restored when the
        # block comes back, so it resumes with its momentum instead of cold.
        #
        # Why "frozen" and not a decayed restore: heavy-ball momentum over k
        # steps of zero gradient decays to beta^k * m. Muon runs beta=0.95 and a
        # block is away ~1,300 steps in a 14-block rotation at 100 steps each --
        # 0.95^1300 ~= 1e-29. A *correctly decayed* restore is therefore
        # arithmetically identical to deletion, which is what BCD already does.
        # The only variant with any content treats parked time as PAUSED rather
        # than as zero-gradient time: the block saw no gradients, not bad ones.
        # That is a modelling choice, stated rather than hidden, and it is the
        # hypothesis under test -- #112 measured BCD at ~3.9x worse perplexity
        # with "the outgoing block's optimizer state deleted on every switch"
        # named as a candidate cause, never tested.
        for i, (_, params) in enumerate(self.blocks):
            if i == idx:
                continue
            for p in params:
                st = self.optimizer.state.get(p)
                if st is not None:
                    if self.retain_optimizer_state:
                        self._parked_state[p] = {
                            k: (v.detach().to("cpu", copy=True)
                                if torch.is_tensor(v) else v)
                            for k, v in st.items()
                        }
                    del self.optimizer.state[p]
                p.grad = None

        if self.retain_optimizer_state:
            dev = self.device or next(
                (p.device for p in active_params if p is not None), None)
            for p in active_params:
                parked = self._parked_state.pop(p, None)
                if parked is not None:
                    self.optimizer.state[p] = {
                        k: (v.to(dev if dev is not None else p.device,
                                 dtype=p.dtype if v.is_floating_point() else v.dtype)
                            if torch.is_tensor(v) else v)
                        for k, v in parked.items()
                    }

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
                # Issue #113: restore to this param's NATIVE dtype (e.g.
                # fp32), not whatever it currently is -- if it was parked
                # in bf16 during its previous inactive tenure, the
                # optimizer's tiny per-step updates need full native
                # precision to be representable at all once training
                # resumes on this block. A no-op cast if it was never
                # downcast (offload_park_dtype=None, the default).
                p.data = p.data.to(device=self.device, dtype=self.buffer_pool.native_dtype_for(p))
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
                params = [p for p in params if id(p) not in self._pinned]
                for p in params:
                    # Issue #113: park at this param's park_dtype (bf16 if
                    # offload_park_dtype=torch.bfloat16 and native dtype is
                    # coarser; otherwise a no-op, identical to pre-#113
                    # behavior). Combined device+dtype .to() so this is a
                    # single copy, not two.
                    p.data = p.data.to(device="cpu", dtype=self.buffer_pool.park_dtype_for(p))
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
