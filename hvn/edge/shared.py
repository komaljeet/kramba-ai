"""
ALBERT-Style Weight Sharing Across HVN Layers (Issue #32, Phase 2.4)
========================================================================

Shares one set of ``HVNBlock`` parameters across all layers (Lan et al.
2019, ALBERT), differentiating layers only by lightweight per-layer
modulation (RoPE frequency offset + entanglement-rate bias) rather than
fully independent weights.  For ``L`` layers this turns ``O(L)`` parameter
storage into ``O(1) + O(L)`` (the ``O(L)`` term being just a couple of
scalars per layer), the single largest lever for reducing a deep model's
footprint.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class SharedWeightHVN(nn.Module):
    """Wraps a single ``HVNBlock`` and applies it ``num_layers`` times.

    Because every layer reuses the same underlying weights, per-layer
    differentiation would otherwise collapse to a no-op (applying the same
    function repeatedly).  To preserve some of the per-depth specialization
    that separate layers would normally provide, each "virtual layer" gets
    a small learned scalar bias added to the RoPE position offset and the
    residual-mixing scale — enough to break the symmetry between layers
    without needing separate weight matrices.

    Parameters
    ----------
    block : nn.Module
        A single ``HVNBlock`` instance (or any module with the same
        ``forward(x, position_ids=None) -> x`` signature) whose parameters
        are shared across all virtual layers.
    num_layers : int
        Number of times to apply *block*.
    """

    def __init__(self, block: nn.Module, num_layers: int):
        super().__init__()
        self.block = block
        self.num_layers = num_layers
        self.layer_position_offset = nn.Parameter(torch.zeros(num_layers))
        self.layer_residual_scale = nn.Parameter(torch.ones(num_layers))

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        B, S, D = x.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        for layer_idx in range(self.num_layers):
            offset = self.layer_position_offset[layer_idx].round().long()
            layer_positions = (position_ids + offset).clamp(min=0)
            out = self.block(x, position_ids=layer_positions)
            scale = self.layer_residual_scale[layer_idx]
            x = x + scale * (out - x)
        return x

    def num_shared_params(self) -> int:
        """Parameters actually stored (shared block + tiny per-layer scalars)."""
        block_params = sum(p.numel() for p in self.block.parameters())
        return block_params + self.layer_position_offset.numel() + self.layer_residual_scale.numel()

    def num_unshared_equivalent_params(self) -> int:
        """Parameters an equivalent stack of ``num_layers`` independent blocks would use."""
        block_params = sum(p.numel() for p in self.block.parameters())
        return block_params * self.num_layers
