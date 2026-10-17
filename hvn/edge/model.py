"""
EdgeHVN — Edge-Deployment Architecture Variant (Issue #32)
=============================================================

Wraps the reference ``HamiltonianVortexNetwork`` with an opt-in
compression pipeline combining the issue #32 techniques that are (a)
end-to-end verified to preserve accuracy and (b) compose safely: ALBERT-
style weight sharing across layers (Phase 2.4, applied at construction
time) plus post-training GPTQ 4-bit (Phase 1.1) and/or ternary BitNet
(Phase 2.1) weight compression.

The remaining, more exotic Phase 3/4/5 techniques (wavelet/DCT,
holographic/superposition encoding, Kronecker/circulant structure, NVMe
streaming, succinct indexing, task-specific loading, simulated-annealing /
topological pruning, population coding, free-energy quantization) are
implemented as standalone, independently-tested utilities in
``hvn.edge.decomposition``, ``hvn.edge.transform``, ``hvn.edge.holographic``,
``hvn.edge.streaming`` and ``hvn.edge.recovery`` — they are
checkpoint-storage or inference-serving concerns better applied by hand to
specific tensors than force-fit into one automatic training-time pipeline,
and are demonstrated individually in ``test_edge_devices.py``.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..core.decoder import HVNDecoder
from ..core.hvn_block import HVNBlock
from ..core.deltanet import DeltaNetBlock
from ..core.transformer_block import TransformerBlock
from .shared import SharedWeightHVN
from .ternary import (TernaryLinear, _advance_cpu_rng_like_uniform,
                      apply_ternary_to_model)


class EdgeHVN(nn.Module):
    """Edge-deployment variant of ``HamiltonianVortexNetwork``.

    Parameters
    ----------
    vocab_size, d_model, num_layers, n_heads, bond_dimension, chunk_size, dropout :
        Passed through to the underlying ``HVNBlock``/``HVNDecoder``
        construction, matching ``HamiltonianVortexNetwork``'s signature.
    shared_weights : bool
        If True, all ``num_layers`` blocks share one set of weights
        (Phase 2.4, ALBERT-style) via :class:`~hvn.edge.shared.SharedWeightHVN`.
    use_deltanet : bool
        If True, use Gated-DeltaNet blocks (Issue #34) instead of ``HVNBlock``
        — fixes the readout/update dynamics that caused HVN's PPL gap while
        preserving O(1) inference.  Default False (original HVN path).
    deltanet_conv_kernel : int
        Kernel size of the DeltaNet short causal depthwise conv (Issue #34).
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        num_layers: int = 2,
        n_heads: int = 4,
        bond_dimension: Optional[int] = None,
        chunk_size: Optional[int] = 64,
        dropout: float = 0.1,
        shared_weights: bool = False,
        use_deltanet: bool = False,
        deltanet_conv_kernel: int = 4,
        block_type: Optional[str] = None,
        use_triton_scan: bool = False,
        use_megakernel_scan: bool = False,
        block_plan: Optional[list[str]] = None,
        attn_window: Optional[int] = None,
        grad_checkpoint: bool = False,
        # Issue #139 measured this rather than assuming it: 1,500 from-scratch
        # steps, identical batches/seed/--deterministic, only this flag differing
        # -> held-out 5.50681 tied vs 5.56450 untied, i.e. 0.058 nats BETTER with
        # 31% FEWER parameters (83.08M -> 57.35M), 12.5 sigma against #133's
        # noise floor. Strictly better AND smaller, not a trade. Costs ~7% per
        # step (the shared table takes gradients from both the lookup and the
        # head, serialising an accumulation).
        #
        # DEFAULT ON since #151. The training CLI has defaulted it on since #139;
        # this makes the library agree, so a model built directly (benchmarks,
        # A/B harnesses, eval) matches what training actually produces.
        #
        # Existing checkpoints are unaffected: every loader passes the flag the
        # checkpoint itself recorded (`ckpt.get("tie_embeddings", False)` in
        # load_checkpoint_for_resume and the #112 audit rebuild), so an untied
        # checkpoint still rebuilds untied regardless of this default.
        tie_embeddings: bool = True,
        eos_token_id: int = 50256,
        moe_experts: int = 1,
        moe_top_k: int = 2,
        ffn_activation: str = "gelu",
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.shared_weights = shared_weights
        # Recorded so a checkpoint can rebuild the same training recipe on
        # resume (a no-dropout run must not silently resume at p=0.1).
        self.dropout_p = float(dropout)
        self.use_deltanet = use_deltanet
        # Issue #100: exact activation checkpointing (mirrors the core
        # HamiltonianVortexNetwork's #49 implementation, which EdgeHVN never
        # inherited since it has its own forward()). Recomputes each block's
        # activations in the backward pass instead of storing them -- bit-
        # identical result, ~20-30% more compute, large activation-VRAM cut.
        # Composes with #99 block-coordinate-descent (that shrinks optimizer-
        # state memory; this shrinks activation memory -- independent axes).
        self.grad_checkpoint = grad_checkpoint
        # Issue #101: CPU weight offload for BCD-inactive blocks (extends
        # #99). Set externally by BlockCoordinateScheduler -- None disables
        # offload entirely (the default / back-compat path, unchanged).
        # #164: a SET of forward() segment names ("token_embedding",
        # "blocks.i", "final_norm+decoder") -- every member of the active BCD
        # block, which under the balanced partition spans several layers.
        self._offload_active_names: frozenset[str] | None = None
        self._offload_device: torch.device | None = None
        # Optional explicit block override (A/B baseline). None → existing
        # use_deltanet/HVNBlock behaviour, unchanged.
        self.block_type = block_type

        # Per-layer block plan (hybrid token-recall). If given, a list of block-type
        # strings of length num_layers ("deltanet" / "transformer" / "hvn");
        # _make_block(i) dispatches by plan[i]. If None, the legacy single-type path
        # broadcasts block_type/use_deltanet to all layers — back-compat with all
        # existing pure-HVN/DeltaNet/transformer checkpoints and the A/B baselines.
        if block_plan is not None:
            if len(block_plan) != num_layers:
                raise ValueError(
                    f"block_plan length {len(block_plan)} != num_layers {num_layers}"
                )
            plan = list(block_plan)
        else:
            _bt = (
                "transformer" if block_type == "transformer"
                else ("deltanet" if use_deltanet else "hvn")
            )
            plan = [_bt] * num_layers
        self.block_plan = plan
        # Issue #140: recorded so a checkpoint knows which activation it was
        # trained with -- loading relu2 weights into a gelu model would succeed
        # silently (identical shapes) and quietly measure the wrong thing.
        self.ffn_activation = ffn_activation
        # Sliding-window width for transformer layers (None = full causal). Only
        # bites at inference when context > window (see TransformerBlock._attention).
        self.attn_window = attn_window

        self.token_embedding = nn.Embedding(vocab_size, d_model)

        def _make_block(i: int):
            t = plan[i]
            if t == "transformer":
                # Matched transformer baseline (A/B #73) / hybrid attention layer
                # (token-recall). `window` caps each layer's KV cache at W keys.
                return TransformerBlock(
                    d_model=d_model, n_heads=n_heads, dropout=dropout,
                    window=attn_window,
                )
            if t == "deltanet":
                # Issue #123: moe_experts>1 routes this block's FFN through a
                # Mixture-of-Experts (capacity at sub-linear compute). Default
                # 1 = dense, byte-identical to the pre-#123 path.
                # Issue #140: ffn_activation selects the FFN's pointwise op.
                # "gelu" (default) is byte-identical to the pre-#140 block.
                return DeltaNetBlock(
                    d_model=d_model, n_heads=n_heads, dropout=dropout,
                    conv_kernel=deltanet_conv_kernel,
                    use_triton_scan=use_triton_scan,
                    moe_experts=moe_experts, moe_top_k=moe_top_k,
                    use_megakernel_scan=use_megakernel_scan,
                    ffn_activation=ffn_activation,
                )
            if t == "hvn":
                return HVNBlock(
                    d_model=d_model, n_heads=n_heads, bond_dimension=bond_dimension,
                    chunk_size=chunk_size, dropout=dropout,
                )
            raise ValueError(f"unknown block type {t!r} at layer {i}")

        # Does the plan mix block types? A hybrid stack does; a plain one does not.
        _plan_types = {plan[i] for i in range(num_layers)} if plan else set()

        if shared_weights and len(_plan_types) > 1:
            # #155: share PER TYPE so the hybrid plan survives. Collapsing a
            # hybrid to one repeated block (the branch below) would train a
            # different architecture than the flags describe -- silently, since
            # it loads and runs perfectly well. See hvn/edge/shared.py.
            from hvn.edge.shared import share_blocks_by_type
            self.blocks, self._shared_canonical = share_blocks_by_type(
                [_make_block(i) for i in range(num_layers)],
                [plan[i] for i in range(num_layers)],
            )
            self.shared_block = None
        elif shared_weights:
            # Uniform stack: the original ALBERT path, one block for every layer.
            self.shared_block = SharedWeightHVN(_make_block(0), num_layers)
            self.blocks = None
        else:
            self.blocks = nn.ModuleList([_make_block(i) for i in range(num_layers)])
            self.shared_block = None

        self.final_norm = nn.LayerNorm(d_model)
        # #168: when tied, the decoder's own [vocab, d_model] weight is thrown
        # away a few lines below -- at 1B (50257x2048) that was a 0.38 GiB fp32
        # allocation that only existed to be freed, and it set the constructor's
        # physical-RAM peak. Build the decoder on meta instead and advance the
        # CPU RNG exactly as its init would have, so every parameter (and the
        # generator state left behind) is bit-identical to the old path
        # (pinned by tests/test_edge_ctor_ram.py). CPU only: elsewhere the
        # discarded init would have drawn from a different generator.
        _lean_tied = (tie_embeddings
                      and self.token_embedding.weight.device.type == "cpu")
        if _lean_tied:
            with torch.device("meta"):
                self.decoder = HVNDecoder(d_model=d_model, vocab_size=vocab_size,
                                          dropout=dropout)
            # nn.Linear.reset_parameters: kaiming_uniform_ on the weight, then
            # uniform_(-1/sqrt(fan_in), 1/sqrt(fan_in)) on the bias.
            _advance_cpu_rng_like_uniform(vocab_size * d_model)
            _bias = torch.empty(vocab_size)
            _bound = 1 / math.sqrt(d_model) if d_model > 0 else 0
            nn.init.uniform_(_bias, -_bound, _bound)
            self.decoder.decoder.bias = nn.Parameter(_bias)
        else:
            self.decoder = HVNDecoder(d_model=d_model, vocab_size=vocab_size,
                                      dropout=dropout)

        # Issue #106: tie the input embedding and output projection weights
        # (classic GPT-2/ALBERT trick). Both are [vocab_size, d_model]
        # (nn.Embedding and nn.Linear store weight this way), so sharing the
        # same tensor is a direct assignment -- no transpose needed. Default
        # False: existing checkpoints/configs are completely unaffected;
        # opt in explicitly since this changes what's actually being trained,
        # not just memory bookkeeping. Halves the embedding+output-projection
        # param count and (with AdamW's fp32 exp_avg/exp_avg_sq) roughly
        # a 3x cut in optimizer-state VRAM for that chunk. HVNDecoder's
        # `use_mlp` default is False (EdgeHVN never sets it True), so
        # `self.decoder.decoder` is always a plain nn.Linear here.
        self.tie_embeddings = tie_embeddings
        # Issue #139: EOS is a property of the TOKENIZER, not a constant. GPT-2's
        # is 50256; the 32K vocab's is 0. A 32K model served with EOS_ID=50256
        # would never sample that id -- it is out of range -- so generation would
        # run to the token limit on every single turn, silently, and look like a
        # quality problem rather than a configuration one. Defaults to GPT-2's so
        # every existing checkpoint and caller is unchanged.
        self.eos_token_id = eos_token_id
        if tie_embeddings:
            # nn.Embedding's default init (std~1.0) is ~20x larger than
            # nn.Linear's (std~0.05 at this scale) -- tying naively (keeping
            # whichever init happened to run last) hands the decoder a
            # wildly oversized weight, producing huge logits and diverging
            # loss. Caught empirically: a CPU parity run (tied vs untied,
            # otherwise identical) showed loss climbing to 25+ instead of
            # settling ~9. Fix: re-init the shared tensor to GPT-2's
            # standard small std (0.02) AFTER tying, matching the
            # convention every tied-embedding implementation actually uses
            # (nanoGPT, GPT-2) -- this is not optional, it's part of what
            # "tying correctly" means, not a separate tunable.
            nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
            self.decoder.decoder.weight = self.token_embedding.weight

        self._compression_metadata: dict = {}

    def _offload_wrap_plain(self, name: str, modules: list[nn.Module], fn, input_tensor=None):
        """Offload wrapper for a NON-checkpointed call (`token_embedding`,
        `final_norm`+`decoder` -- neither is ever wrapped in
        `torch.utils.checkpoint`). Issue #101, extends #99.

        A plain (non-checkpointed) forward retains whatever standard
        autograd needs for backward -- it does NOT discard anything the way
        checkpoint's initial pass does. So eviction timing here really does
        depend on whether backward will ever revisit this call:

        - `input_tensor` (this block's input) does NOT require grad (this
          block sits upstream of the active BCD block, or has no meaningful
          input at all, e.g. token_embedding): nothing downstream of the
          active block needs to backprop through this segment -- PyTorch's
          autograd never builds a graph node for an all-requires_grad=False
          computation, so there is provably no backward reference to worry
          about. Safe to evict immediately after forward.
        - It DOES require grad (this block sits downstream of active): the
          gradient must flow through it via the chain rule using its real
          weight values (even though the weight's own `.grad` is never
          computed, being frozen). Defer eviction to a hook on
          `input_tensor` -- it fires when THIS segment's local backward has
          finished (the last thing that local backward computes is the
          gradient of ITS OWN input), not when it starts. Hooking the
          *output* instead fires too early, before the weights are even
          used, and previously produced
          `native_layer_norm_backward: weight is on cpu` -- caught by
          tests/test_block_coordinate_offload.py's gradient-parity test.
        """
        if name in self._offload_active_names:
            return fn()

        params = [p for m in modules for p in m.parameters()]
        # Pinned params (e.g. tied embeddings) have no CPU home and must stay
        # on GPU. Skip offload wrapper entirely for them.
        if not self._offload_pool._offloadable(params):
            return fn()

        self._offload_pool.load(name, params)
        out = fn()

        if input_tensor is not None and input_tensor.requires_grad:
            def _evict(grad, k=name):
                self._offload_pool.evict(k)
                return None  # hooks must return None or a replacement Tensor
            input_tensor.register_hook(_evict)
        else:
            self._offload_pool.evict(name)
        return out

    def _offload_wrap_checkpointed(self, name: str, block: nn.Module, x_in, position_ids,
                                   doc_ids=None):
        """Offload wrapper for a checkpointed LAYER block (Issue #101).

        Unlike `_offload_wrap_plain`, a checkpointed segment gets invoked
        TWICE per training step: once synchronously here, during
        `forward()` (call #1 -- its initial pass, whose activations/saved
        state checkpoint discards immediately, by design), and once later,
        invisibly to this method, during `backward()` (call #2 -- the
        recompute, immediately followed by this segment's own local
        backward). This asymmetry means the correct eviction policy is
        NARROWER than `_offload_wrap_plain`'s:

        - Call #1 is ALWAYS safe to evict immediately after, regardless of
          whether this block sits before or after the active BCD block --
          checkpoint's own discard-everything behavior on its initial pass
          means nothing retains a reference to these weights past this
          call, for THIS call. (An earlier version deferred eviction for
          every block downstream of active, for its ENTIRE call, not just
          call #2 -- correct but wasteful: it left every downstream block
          GPU-resident for the whole forward pass, defeating much of the
          memory saving. Measured: a config that should have fit given the
          "~1 block resident at a time" design instead peaked well past
          this card's 6141 MiB -- caught by
          scripts/bench_offload_vram.py, not a unit test, since it is a
          peak-memory regression, not a correctness one.)
        - Call #2 (the recompute) will only ever happen if `x_in` requires
          grad (i.e. this block is downstream of active) -- otherwise
          checkpoint's own machinery never revisits it, matching
          `_offload_wrap_plain`'s reasoning.

        Issue #103: when a call #2 IS coming (`x_in.requires_grad`), call
        #1's "always safe to evict immediately" no longer means "should" --
        the pool's LRU cache (`offload_cache_blocks > 1`) can instead keep
        this block warm, so call #2's `pool.load(name, ...)` inside
        `_self_loading_forward` is a cache HIT (no PCIe re-fetch) instead of
        a guaranteed miss. Actual eviction is deferred to a hook on `x_in`,
        timed against call #2's own local backward finishing -- same
        reasoning as `_offload_wrap_plain`'s hook (hooking the input, not
        the output, because the last thing a segment's local backward
        computes is the gradient of its OWN input). If no call #2 is
        coming, evict right away instead -- holding the slot would only
        deny it to a block that CAN actually reuse it.

        Loading is handled INSIDE the checkpointed closure itself (not just
        here), so call #2 -- invoked later by checkpoint's own machinery,
        entirely outside this method's synchronous scope -- also gets fresh
        (or cache-hit) weights; see the `_self_loading_forward` closure
        below.
        """
        # doc_ids (Issue #132) is a non-differentiable int tensor, so it is
        # captured by the closure rather than passed as a checkpoint arg --
        # checkpoint only needs to track tensors that carry grad. Kept out of
        # the kwargs entirely when unset, so the unpacked path is unchanged.
        _kw = {} if doc_ids is None else {"doc_ids": doc_ids}
        if name in self._offload_active_names:
            if doc_ids is None:
                return checkpoint(block, x_in, position_ids, use_reentrant=False)

            def _active_forward(x_arg, pos_arg):
                return block(x_arg, position_ids=pos_arg, **_kw)
            return checkpoint(_active_forward, x_in, position_ids, use_reentrant=False)

        pool = self._offload_pool
        params = list(block.parameters())

        # Backward recompute prefetch (2026-10-12). Backward recomputes the
        # offloaded blocks in REVERSE forward order, so when block i's
        # recompute starts, the next one needed is the previous entry of this
        # forward's recompute list. Prefetching it here, on the side stream,
        # overlaps its copy with block i's recompute + backward instead of
        # paying a synchronous pageable copy (which also drains the host's
        # launch queue) when its own turn comes.
        order = getattr(self, "_offload_recompute_order", None)
        recompute_coming = x_in.requires_grad and order is not None
        pos = len(order) if recompute_coming else 0
        if recompute_coming:
            order.append((name, params))
        calls = [0]

        def _self_loading_forward(x_arg, pos_arg):
            # Runs on EVERY invocation checkpoint makes (call #1 here, call
            # #2 later during backward) -- a cache hit on the second call
            # is a cheap no-op (Issue #103), a miss re-copies into a reused
            # buffer (Issue #101). Either way, safe to call unconditionally.
            pool.load(name, params)
            calls[0] += 1
            if calls[0] == 2 and recompute_coming and pos > 0:
                # Issued after this block's load() (so it is resident and
                # most-recently-used) and BEFORE its recompute kernels are
                # queued. `behind=True`: this block stays resident for its
                # own backward -- see GPUBufferPool.prefetch.
                prev_name, prev_params = order[pos - 1]
                pool.prefetch(prev_name, prev_params, behind=True)
            return block(x_arg, position_ids=pos_arg, **_kw)

        pool.load(name, params)  # ensure call #1 itself has weights present
        out = checkpoint(_self_loading_forward, x_in, position_ids, use_reentrant=False)

        if x_in.requires_grad:
            def _evict(grad, k=name):
                pool.evict(k)
                return None
            x_in.register_hook(_evict)
        else:
            pool.evict(name)  # no call #2 coming -- free the slot now
        return out

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None,
                doc_ids: Optional[torch.Tensor] = None,
                return_hidden: bool = False) -> torch.Tensor:
        """Forward the full stack.

        ``doc_ids`` ([B, S], Issue #132) marks which packed document each
        position belongs to, so blocks can cut their sequence-axis carries at
        document boundaries (DeltaNet: conv taps + recurrent state; attention:
        a block-diagonal mask). ``None`` (the default) is the unpacked path and
        is byte-identical to before.
        """
        B, S = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)
        if doc_ids is not None:
            if doc_ids.shape != input_ids.shape:
                raise ValueError(
                    f"doc_ids shape {tuple(doc_ids.shape)} must match input_ids "
                    f"{tuple(input_ids.shape)}"
                )
            # HVNBlock carries its own Hamiltonian state along the sequence and
            # has no boundary-reset mechanism, so a packed row would leak across
            # documents there. Only the deltanet+transformer hybrid (the FT
            # target) supports packing.
            unsupported = sorted({t for t in self.block_plan if t == "hvn"})
            if unsupported or self.shared_block is not None:
                raise NotImplementedError(
                    "Packed training (doc_ids) supports 'deltanet' and "
                    "'transformer' blocks only; this model's block_plan is "
                    f"{self.block_plan}"
                    + (" with shared_weights=True" if self.shared_block is not None else "")
                    + ". The hybrid FT configuration (deltanet + windowed "
                    "attention) is the supported path."
                )
        # Only forward doc_ids when packing is actually in use, so the unpacked
        # path stays byte-identical and block types that never take the kwarg
        # (HVNBlock, SharedWeightHVN) are unaffected.
        blk_kw = {} if doc_ids is None else {"doc_ids": doc_ids}

        offload = (getattr(self, "_offload_active_names", None) is not None
                   and self.training and torch.is_grad_enabled())
        if offload and not self.grad_checkpoint:
            raise RuntimeError(
                "weight offload (Issue #101) requires grad_checkpoint=True -- "
                "eviction timing is tied to each block's checkpoint recompute "
                "window, which does not exist without checkpointing."
            )
        if offload:
            park_dtype = getattr(getattr(self, "_offload_pool", None), "park_dtype", None)
            if park_dtype is not None and not torch.is_autocast_enabled():
                raise RuntimeError(
                    "offload_park_dtype (Issue #113) parks inactive blocks at a "
                    "coarser dtype (e.g. bf16) than the active block's native "
                    "dtype (typically fp32) -- forwarding through a frozen "
                    "parked block now mixes dtypes with the active block's "
                    "activations, which PyTorch ops like Conv1d reject outside "
                    "autocast ('Input type and bias type should be the same'). "
                    "Wrap the forward+loss computation in "
                    "`torch.amp.autocast('cuda', dtype=torch.bfloat16)` (already "
                    "this project's standard training pattern, e.g. "
                    "scripts/bench_offload_vram.py) -- autocast reconciles the "
                    "mixed dtypes automatically. Confirmed via a direct repro: "
                    "without autocast this raises deep inside conv1d; with it, "
                    "training runs and converges normally."
                )

        if offload:
            # Offloaded blocks whose backward recompute is coming, in forward
            # order (filled by _offload_wrap_checkpointed). None disables the
            # backward recompute prefetch (2026-10-12).
            self._offload_recompute_order = (
                [] if (getattr(self, "_offload_prefetch", True)
                       and getattr(self, "_offload_bwd_prefetch", True)) else None)
            x = self._offload_wrap_plain(
                "token_embedding", [self.token_embedding],
                lambda: self.token_embedding(input_ids),
            )
        else:
            x = self.token_embedding(input_ids)

        if self.shared_block is not None:
            # doc_ids is rejected above for the shared path, so blk_kw is empty here.
            x = self.shared_block(x, position_ids=position_ids)
        else:
            use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
            for i, block in enumerate(self.blocks):
                name = f"blocks.{i}"
                x_in = x
                if offload:
                    # offload requires grad_checkpoint=True (enforced above),
                    # so use_ckpt is guaranteed True here.
                    x = self._offload_wrap_checkpointed(name, block, x_in, position_ids,
                                                        doc_ids=doc_ids)
                    # Issue #115: kick off the NEXT offloadable segment's
                    # H2D transfer now, on a side stream -- issued AFTER
                    # this block's own load()/compute above (not before),
                    # so the LRU cache has already registered block i as
                    # just-used before block i+1 potentially competes for
                    # the same slot (cache_blocks=1 by default -- prefetching
                    # i+1 BEFORE touching block i's own residency would
                    # evict block i's buffer out from under its own
                    # in-flight compute). GPU-level overlap doesn't depend
                    # on this Python-side ordering -- both this block's
                    # compute (already enqueued on the main stream above)
                    # and the next block's copy (enqueued on the side
                    # stream below) run concurrently regardless of which
                    # line issues first, since neither call blocks the
                    # host thread. Covers both the next layer block and,
                    # on the last layer, the tail (final_norm+decoder) --
                    # both go through the same pool.load()/_offload_wrap_*
                    # path, so a prefetch here is transparently consumed
                    # there either way.
                    if getattr(self, "_offload_prefetch", True):
                        if i + 1 < len(self.blocks):
                            next_name = f"blocks.{i + 1}"
                            next_params = list(self.blocks[i + 1].parameters())
                        else:
                            next_name = "final_norm+decoder"
                            next_params = list(self.final_norm.parameters()) + list(self.decoder.parameters())
                        if next_name not in self._offload_active_names:
                            self._offload_pool.prefetch(next_name, next_params)
                elif use_ckpt:
                    if doc_ids is None:
                        x = checkpoint(block, x_in, position_ids, use_reentrant=False)
                    else:
                        # doc_ids is non-differentiable; capture it in a closure
                        # rather than adding a positional checkpoint arg.
                        def _ckpt_forward(x_arg, pos_arg, _b=block, _d=doc_ids):
                            return _b(x_arg, position_ids=pos_arg, doc_ids=_d)
                        x = checkpoint(_ckpt_forward, x_in, position_ids,
                                       use_reentrant=False)
                else:
                    x = block(x_in, position_ids=position_ids, **blk_kw)

        if offload:
            if return_hidden:
                out = self._offload_wrap_plain(
                    "final_norm+decoder", [self.final_norm],
                    lambda: self.final_norm(x), input_tensor=x,
                )
            else:
                out = self._offload_wrap_plain(
                    "final_norm+decoder", [self.final_norm, self.decoder],
                    lambda: self.decoder(self.final_norm(x)),
                    input_tensor=x,
                )
            # Backward recompute prefetch: the FIRST block backward recomputes
            # is the last entry of the list; nothing earlier can fetch it, so
            # start it now, alongside the loss and the tail's backward. Behind
            # the tail, which stays resident until its own backward is done.
            order = getattr(self, "_offload_recompute_order", None)
            if order:
                last_name, last_params = order[-1]
                self._offload_pool.prefetch(last_name, last_params, behind=True)
            return out

        x = self.final_norm(x)
        # #158: with return_hidden the caller does the vocabulary projection
        # itself, one slice of positions at a time. The full [B, S, vocab] logits
        # are ~500 MB at 4096x32003 and their gradient the same again -- three
        # times the size of the weights on this card. Returning [B, S, d_model]
        # instead (4 MB) lets hvn.edge.chunked_loss project and score in slices,
        # so only one slice's logits ever exist.
        if return_hidden:
            return x
        return self.decoder(x)

    # ------------------------------------------------------------------
    # Incremental decode (issue #97 Phase B)
    #
    # Every block type (HVNBlock, DeltaNetBlock, TransformerBlock) now
    # implements a uniform init_state(batch_size, device, dtype) /
    # forward_step(x_t, state, position_idx) contract, so dispatch here is
    # per-layer via block_plan exactly like `forward()` above -- mirroring
    # HamiltonianVortexNetwork.forward_step's own per-block loop
    # (hvn/core/model.py:387). init_conversation_state/chat_step match that
    # class's method names and conv_state dict keys exactly (verified
    # signatures) so hvn/chat.py's ChatSession/StateCheckpoint work against
    # EdgeHVN with ZERO changes to their own logic -- only a formatter swap
    # is needed at the call site (see hvn/chat.py's formatter param + the
    # new hvn/edge/chat.py:EdgeConversationFormatter).
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int, device=None, dtype=None) -> list:
        """Mixed per-layer state: each block's own init_state, by block_plan."""
        if self.shared_block is not None:
            raise NotImplementedError(
                "incremental decode is not implemented for shared_weights=True "
                "models -- no hybrid checkpoint uses SharedWeightHVN."
            )
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        return [block.init_state(batch_size, device=device, dtype=dtype)
                for block in self.blocks]

    def enable_graph_decode(self, max_positions: int | None = None) -> dict:
        """Prepare every block for CUDA-graph-captured decode (Issue #138).

        Currently: precompute RoPE tables on the attention blocks. Without them
        ``forward_step`` builds cos/sin from a Python int each token, which is a
        CPU->GPU copy that capture rejects, and which would bake one position
        into the captured graph even if the copy were removed.

        Opt-in and idempotent. Returns what it touched so a caller can assert
        rather than assume — a model with zero attention blocks would otherwise
        report success having done nothing.
        """
        from hvn.core.transformer_block import TransformerBlock

        touched = {"attention_blocks": 0, "max_positions": None}
        for block in self.blocks:
            inner = getattr(block, "block", block)
            if isinstance(inner, TransformerBlock):
                n = inner.enable_rope_table(max_positions=max_positions)
                touched["attention_blocks"] += 1
                touched["max_positions"] = n
        return touched

    def forward_step(self, input_ids: torch.Tensor, states: list,
                     position_idx: Optional[int] = None) -> tuple[torch.Tensor, list]:
        """Process one token incrementally. input_ids: [B, 1] -> (logits [B,1,V], new_states)."""
        x = self.token_embedding(input_ids)
        new_states = []
        for i, block in enumerate(self.blocks):
            x, new_state = block.forward_step(x, states[i], position_idx=position_idx)
            new_states.append(new_state)
        x = self.final_norm(x)
        logits = self.decoder(x)
        return logits, new_states

    def init_conversation_state(self, device=None, dtype=None) -> dict:
        """Conversation state for chat_step. Same dict keys as
        HamiltonianVortexNetwork.init_conversation_state (hvn/core/model.py:1079)
        -- "hamiltonian_states" is a naming carryover from that shared contract
        (ChatSession/StateCheckpoint reference it by name), not a claim that
        EdgeHVN has Hamiltonian dynamics; it holds whatever mixed per-layer
        state init_state() above produces.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        return {
            "hamiltonian_states": self.init_state(1, device=device, dtype=dtype),
            "position": 0,
            "history_ids": [],
        }

    def _filter_top_k(self, logits: torch.Tensor, top_k: int) -> torch.Tensor:
        """Top-k filtering on the last dim (identical logic to
        HamiltonianVortexNetwork._filter_top_k, hvn/core/model.py:734)."""
        if not top_k or top_k <= 0:
            return logits
        k = min(top_k, logits.size(-1))
        values, _ = torch.topk(logits, k, dim=-1)
        threshold = values[..., -1:].contiguous()
        return logits.masked_fill(logits < threshold, float("-inf"))

    @staticmethod
    def _apply_repetition_penalty_masked(logits: torch.Tensor,
                                         seen_mask: torch.Tensor,
                                         penalty: float) -> torch.Tensor:
        """Repetition penalty from a device-resident boolean mask (Issue #138).

        Identical arithmetic to :meth:`_apply_repetition_penalty`, but the set of
        already-emitted ids lives on the GPU instead of in a Python ``set``.
        Maintaining that set required ``int(next_id.item())`` every token — a
        device-to-host sync that stops the CPU running ahead of the GPU, which is
        exactly what a CUDA graph exists to enable. Two such syncs per token were
        the whole reason chat_step saw 1.94x where forward_step saw 4.50x.
        """
        if penalty == 1.0:
            return logits
        return torch.where(seen_mask,
                           torch.where(logits > 0, logits / penalty,
                                       logits * penalty),
                           logits)

    def _apply_repetition_penalty(self, logits: torch.Tensor, seen_ids: set[int],
                                  penalty: float) -> torch.Tensor:
        """Standard repetition penalty (Keskar et al., CTRL): divide the logit
        of every already-generated token by `penalty` if positive, multiply if
        negative -- pushes previously-used tokens down regardless of sign,
        unlike a flat subtraction. `penalty=1.0` is an exact no-op (the
        default -- zero behavior change unless a caller opts in).
        """
        if penalty == 1.0 or not seen_ids:
            return logits
        idx = torch.tensor(list(seen_ids), device=logits.device, dtype=torch.long)
        vals = logits[idx]
        logits = logits.clone()
        logits[idx] = torch.where(vals > 0, vals / penalty, vals * penalty)
        return logits

    @torch.no_grad()
    def chat_step(
        self,
        input_ids: list[int],
        conv_state: dict,
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 10,
        device: Optional[torch.device] = None,
        repetition_penalty: float = 1.0,
        use_cuda_graph: bool = False,
    ) -> tuple[list[int], dict]:
        """Process a conversation turn incrementally. Same overall shape as
        HamiltonianVortexNetwork.chat_step (hvn/core/model.py:1118): feed
        input tokens through forward_step, then sample a response.

        One deliberate difference: generation stops as soon as EOS_ID (GPT-2's
        50256) is sampled, instead of always running the full max_new_tokens --
        EdgeHVN's chat FT trains turns to end with an explicit EOS token
        (finetune_edge_conversational.py's InstructionDataset), unlike the core
        model's bracket-tag format, which has no single stop token and so
        can't early-stop this way.

        `repetition_penalty` (default 1.0 = off, matching prior behavior
        exactly): real degenerate looping was observed in this checkpoint's
        Docker verification (issue #97 Phase B writeup, docs/state.md) --
        pass e.g. 1.2-1.3 to discourage it. Penalizes tokens already emitted
        THIS turn (input_ids + generated-so-far), not the full conversation
        history -- matches the common convention (e.g. HF `generate`'s
        default window) and keeps the cost O(turn length) not O(history).
        """
        EOS_ID = getattr(self, "eos_token_id", 50256)
        if device is None:
            device = next(self.parameters()).device

        self.eval()
        states = conv_state["hamiltonian_states"]
        pos = conv_state["position"]
        # Device-resident "already emitted" mask, so the repetition penalty and
        # the EOS bookkeeping cost no host syncs (#138).
        seen_mask = torch.zeros(self.vocab_size, dtype=torch.bool, device=device)
        if input_ids:
            seen_mask[torch.tensor(input_ids, device=device, dtype=torch.long)] = True

        # zeros, not empty: this buffer is READ during graph capture below,
        # before the sampling loop ever writes it. torch.empty handed the
        # embedding a garbage id (observed: 4.6e18 against a vocab of 256), which
        # fires a device-side assert inside Indexing.cu and poisons the CUDA
        # context — surfacing later at an unrelated line.
        cur = torch.zeros(1, 1, dtype=torch.long, device=device)
        # Issue #138: position lives on the device so a captured graph indexes a
        # moving value rather than folding one position in. Harmless when
        # use_cuda_graph is False -- forward_step accepts either.
        pos_t = torch.zeros((), dtype=torch.long, device=device)

        # Prefill stays EAGER, always. Shapes vary with the prompt, and before the
        # KV ring wraps the mask changes every token; capture needs a fixed shape.
        if len(input_ids) > 0:
            input_row = torch.tensor([input_ids], dtype=torch.long, device=device)
            for t in range(input_row.size(1)):
                pos_t.fill_(pos)
                logits_step, states = self.forward_step(
                    input_row[:, t : t + 1], states, position_idx=pos_t
                )
                pos += 1
        else:
            logits_step = None

        graph = None
        graph_logits = None
        if use_cuda_graph and logits_step is not None and device.type == "cuda":
            graph, graph_logits = self._capture_decode_graph(cur, states, pos_t)

        next_logits = logits_step[0, -1, :] / temperature

        resp_id_tensors: list[torch.Tensor] = []
        # EOS is still checked EVERY token, deliberately. Batching the check
        # (sync once per 16) was tried and reverted: the loop then generates past
        # EOS, and those tokens are already written into the KV cache and the ring
        # counter by the time they are discarded — so the returned text is right
        # while the conversation STATE is polluted for the next turn. A sync per
        # token is the price of not corrupting multi-turn state.
        #
        # What IS removed is the SECOND sync per token: seen_ids was a Python set
        # needing int(next_id.item()) to maintain, and is now a device mask.
        for _ in range(max_new_tokens):
            logits = self._apply_repetition_penalty_masked(
                next_logits, seen_mask, repetition_penalty)
            logits = self._filter_top_k(logits, top_k)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            if int(next_id.item()) == EOS_ID:
                break
            resp_id_tensors.append(next_id)
            seen_mask.index_fill_(0, next_id.view(-1), True)

            cur.copy_(next_id.view(1, 1))
            pos_t.fill_(pos)
            if graph is not None:
                # Replay writes into the SAME buffers the capture used: cur, pos_t
                # and the state tensors. Nothing is returned because nothing moves.
                graph.replay()
                logits_step = graph_logits
            else:
                logits_step, states = self.forward_step(cur, states,
                                                        position_idx=pos_t)
            pos += 1
            next_logits = logits_step[0, -1, :] / temperature

        response_ids = torch.cat(resp_id_tensors).tolist() if resp_id_tensors else []


        conv_state["hamiltonian_states"] = states
        conv_state["position"] = pos
        conv_state["history_ids"].extend(input_ids)
        conv_state["history_ids"].extend(response_ids)

        return response_ids, conv_state

    def _capture_decode_graph(self, cur, states, pos_t):
        """Capture one decode step for replay (Issue #138).

        Returns ``(graph, static_logits)`` or ``(None, None)`` if capture is not
        possible — a failure here must degrade to eager decoding, never raise,
        because a serving path that dies on an optimisation is worse than a slow
        one.

        Requires ``enable_graph_decode()`` to have run: without it the attention
        blocks build RoPE from a Python int (a CPU->GPU copy capture rejects) and
        keep their ring counter as a Python int, which a graph cannot increment —
        producing a graph that replays one slot forever. That failure is silent,
        so this refuses to capture rather than risk it.
        """
        from hvn.core.transformer_block import TransformerBlock

        attn = [getattr(b, "block", b) for b in self.blocks]
        attn = [b for b in attn if isinstance(b, TransformerBlock)]
        if attn and not all(getattr(b, "graph_safe_decode", False) for b in attn):
            return None, None
        # Warm-up and capture each run a REAL forward_step, and decode state is
        # now mutated in place (#138), so those four steps advance the ring
        # counter, the KV slots and the DeltaNet state. Without snapshot/restore
        # the graphed run starts four tokens ahead of where `pos` says it is, and
        # diverges from eager on the very first sampled token.
        tensors = [v for st in states if isinstance(st, dict)
                   for k in sorted(st) if torch.is_tensor(v := st[k])]
        snapshot = [t.clone() for t in tensors]

        try:
            side = torch.cuda.Stream()
            side.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(side):
                for _ in range(3):      # settle allocator / kernel choices
                    self.forward_step(cur, states, position_idx=pos_t)
            torch.cuda.current_stream().wait_stream(side)
            torch.cuda.synchronize()

            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                static_logits, _ = self.forward_step(cur, states,
                                                     position_idx=pos_t)
        except torch.cuda.CudaError:    # pragma: no cover - hardware dependent
            # NOT swallowed. A device-side assert kills the CUDA context, so
            # "falling back to eager" would fail at the next kernel launch with a
            # traceback pointing somewhere innocent. Re-raise while the cause is
            # still legible.
            raise
        except Exception as exc:        # pragma: no cover - hardware dependent
            print(f"  [graph] capture unavailable ({type(exc).__name__}: {exc}); "
                  f"falling back to eager decode")
            for dst, src in zip(tensors, snapshot):
                dst.copy_(src)
            return None, None

        for dst, src in zip(tensors, snapshot):
            dst.copy_(src)
        return graph, static_logits

    # ------------------------------------------------------------------
    # Compression pipeline
    # ------------------------------------------------------------------

    @torch.no_grad()
    def compress_gptq(
        self,
        calibration_forward: Callable[[], None],
        bits: int = 4,
        group_size: int = 128,
        percdamp: float = 0.01,
    ) -> dict:
        """Apply Phase 1.1 GPTQ 4-bit quantization to every exercised ``nn.Linear`` layer."""
        from ..compress import compress_linear_layers_gptq
        metadata = compress_linear_layers_gptq(
            self, calibration_forward, bits=bits, group_size=group_size, percdamp=percdamp,
        )
        self._compression_metadata["gptq"] = metadata
        return metadata

    def compress_ternary(self, target_names: Optional[list[str]] = None) -> dict[str, TernaryLinear]:
        """Replace named ``nn.Linear`` layers with (trainable, STE) Phase 2.1 ternary weights.

        After calling this, continue training/fine-tuning briefly so the
        straight-through estimator can adapt the weights to their ternary
        constraint (see ``hvn.edge.recovery.recovery_finetune_step``).
        """
        replaced = apply_ternary_to_model(self, target_names)
        self._compression_metadata["ternary"] = {name: True for name in replaced}
        return replaced

    def memory_report(self) -> dict:
        """Summarize achieved compression vs. a dense fp32 baseline.

        Returns
        -------
        dict
            ``dense_fp32_bytes``, ``compressed_bytes``, ``compression_ratio``.
        """
        dense_bytes = sum(p.numel() * 4 for p in self.parameters())
        compressed_bytes = 0
        accounted: set[str] = set()

        for name, mod in self.named_modules():
            if isinstance(mod, TernaryLinear):
                compressed_bytes += mod.memory_bytes()
                accounted.add(f"{name}.weight")
                if mod.bias is not None:
                    accounted.add(f"{name}.bias")

        for name, info in self._compression_metadata.get("gptq", {}).items():
            compressed_bytes += info["quant_bytes"]
            accounted.add(f"{name}.weight")

        for name, p in self.named_parameters():
            if name not in accounted:
                compressed_bytes += p.numel() * 4

        return {
            "dense_fp32_bytes": dense_bytes,
            "compressed_bytes": compressed_bytes,
            "compression_ratio": dense_bytes / max(compressed_bytes, 1),
        }

    def num_params(self, effective: bool = False) -> int:
        """Total parameter count.

        Parameters
        ----------
        effective : bool
            If True and ``shared_weights=True``, counts the shared block's
            weights once instead of ``num_layers`` times (the actual
            storage footprint), rather than the nominal parameter count
            PyTorch reports for the unrolled forward pass.
        """
        if effective and self.shared_block is not None:
            non_block = sum(
                p.numel() for n, p in self.named_parameters() if not n.startswith("shared_block.block")
            )
            return self.shared_block.num_shared_params() + non_block
        return sum(p.numel() for p in self.parameters())
