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

from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from ..core.decoder import HVNDecoder
from ..core.hvn_block import HVNBlock
from ..core.deltanet import DeltaNetBlock
from ..core.transformer_block import TransformerBlock
from .shared import SharedWeightHVN
from .ternary import TernaryLinear, apply_ternary_to_model


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
        block_plan: Optional[list[str]] = None,
        attn_window: Optional[int] = None,
        grad_checkpoint: bool = False,
        tie_embeddings: bool = False,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.shared_weights = shared_weights
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
        self._offload_active_name: str | None = None
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
                return DeltaNetBlock(
                    d_model=d_model, n_heads=n_heads, dropout=dropout,
                    conv_kernel=deltanet_conv_kernel,
                    use_triton_scan=use_triton_scan,
                )
            if t == "hvn":
                return HVNBlock(
                    d_model=d_model, n_heads=n_heads, bond_dimension=bond_dimension,
                    chunk_size=chunk_size, dropout=dropout,
                )
            raise ValueError(f"unknown block type {t!r} at layer {i}")

        if shared_weights:
            # Degenerate for hybrid (shares one block type); hybrid uses
            # shared_weights=False. Uses plan[0] for the shared block, unchanged
            # for the existing shared-weight HVN path.
            self.shared_block = SharedWeightHVN(_make_block(0), num_layers)
            self.blocks = None
        else:
            self.blocks = nn.ModuleList([_make_block(i) for i in range(num_layers)])
            self.shared_block = None

        self.final_norm = nn.LayerNorm(d_model)
        self.decoder = HVNDecoder(d_model=d_model, vocab_size=vocab_size, dropout=dropout)

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
        if name == self._offload_active_name:
            return fn()

        params = [p for m in modules for p in m.parameters()]
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

    def _offload_wrap_checkpointed(self, name: str, block: nn.Module, x_in, position_ids):
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
        if name == self._offload_active_name:
            return checkpoint(block, x_in, position_ids, use_reentrant=False)

        pool = self._offload_pool
        params = list(block.parameters())

        def _self_loading_forward(x_arg, pos_arg):
            # Runs on EVERY invocation checkpoint makes (call #1 here, call
            # #2 later during backward) -- a cache hit on the second call
            # is a cheap no-op (Issue #103), a miss re-copies into a reused
            # buffer (Issue #101). Either way, safe to call unconditionally.
            pool.load(name, params)
            return block(x_arg, position_ids=pos_arg)

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

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        offload = (getattr(self, "_offload_active_name", None) is not None
                   and self.training and torch.is_grad_enabled())
        if offload and not self.grad_checkpoint:
            raise RuntimeError(
                "weight offload (Issue #101) requires grad_checkpoint=True -- "
                "eviction timing is tied to each block's checkpoint recompute "
                "window, which does not exist without checkpointing."
            )

        if offload:
            x = self._offload_wrap_plain(
                "token_embedding", [self.token_embedding],
                lambda: self.token_embedding(input_ids),
            )
        else:
            x = self.token_embedding(input_ids)

        if self.shared_block is not None:
            x = self.shared_block(x, position_ids=position_ids)
        else:
            use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
            for i, block in enumerate(self.blocks):
                name = f"blocks.{i}"
                x_in = x
                if offload:
                    # offload requires grad_checkpoint=True (enforced above),
                    # so use_ckpt is guaranteed True here.
                    x = self._offload_wrap_checkpointed(name, block, x_in, position_ids)
                elif use_ckpt:
                    x = checkpoint(block, x_in, position_ids, use_reentrant=False)
                else:
                    x = block(x_in, position_ids=position_ids)

        if offload:
            return self._offload_wrap_plain(
                "final_norm+decoder", [self.final_norm, self.decoder],
                lambda: self.decoder(self.final_norm(x)),
                input_tensor=x,
            )

        x = self.final_norm(x)
        return self.decoder(x)

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
