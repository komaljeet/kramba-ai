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

        self._compression_metadata: dict = {}

    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=input_ids.device).unsqueeze(0).expand(B, -1)

        x = self.token_embedding(input_ids)
        if self.shared_block is not None:
            x = self.shared_block(x, position_ids=position_ids)
        else:
            use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()
            for block in self.blocks:
                if use_ckpt:
                    x = checkpoint(block, x, position_ids, use_reentrant=False)
                else:
                    x = block(x, position_ids=position_ids)

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
