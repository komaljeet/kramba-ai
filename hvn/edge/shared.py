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


class SharedBlockRef(nn.Module):
    """One virtual layer that BORROWS another block's weights (#155).

    Why this exists alongside ``SharedWeightHVN``
    ---------------------------------------------
    ``SharedWeightHVN`` replaces the whole block stack with a single block
    applied ``num_layers`` times. That is correct for a uniform stack, but the
    hybrid architecture this repo now trains is *not* uniform: it interleaves
    sliding-window attention layers among DeltaNet layers, and the recall
    finding (#135) depends on which positions are which. Collapsing that plan to
    one repeated block would silently train a different architecture than the
    flags describe -- the substitution class of bug that #139/#164 both were.

    So sharing is done **per block type** instead. The model keeps its normal
    ``nn.ModuleList`` of ``num_layers`` entries in plan order, but every entry of
    a given type wraps the *same* underlying block object. A 12-layer hybrid with
    10 DeltaNet + 2 attention layers therefore stores 2 blocks, not 12, while the
    plan, the forward path, gradient checkpointing, offload and incremental
    decode all keep working untouched -- none of them can tell the difference.

    PyTorch de-duplicates repeated parameters by identity in ``.parameters()``,
    so the optimizer sees each shared weight once and gradients from every
    virtual layer accumulate into it, which is exactly ALBERT's update rule.

    Per-layer modulation
    --------------------
    Applying identical weights repeatedly with nothing to distinguish depth
    wastes the stack, so each virtual layer keeps two of its own scalars -- a
    RoPE position offset and a residual scale -- mirroring ``SharedWeightHVN``.
    Two floats per layer against a whole block's worth of weights.
    """

    def __init__(self, block: nn.Module, layer_idx: int):
        super().__init__()
        # Assigning a Module attribute registers it as a child, so the shared
        # block appears under several parents. That is intended and is what
        # makes the parameters shared rather than copied; state_dict will list
        # it once per position, all keys mapping to the same tensor.
        self.block = block
        self.layer_idx = layer_idx
        self.position_offset = nn.Parameter(torch.zeros(1))
        self.residual_scale = nn.Parameter(torch.ones(1))

    def forward(self, x, position_ids=None, **kw):
        if position_ids is not None:
            offset = self.position_offset.round().long()
            position_ids = (position_ids + offset).clamp(min=0)
        out = self.block(x, position_ids=position_ids, **kw)
        # Scale the block's CONTRIBUTION, not the whole activation: out is
        # already x + f(x), so scaling out directly would decay the residual
        # stream itself and make deep stacks unstable.
        return x + self.residual_scale * (out - x)


def share_blocks_by_type(blocks, plan):
    """Return a ModuleList where all layers of the same type share one block.

    ``blocks`` is the freshly built per-layer list and ``plan`` the block-type
    string per layer. The first block of each type is kept and every later layer
    of that type wraps it, so distinct types stay genuinely distinct.
    """
    if len(blocks) != len(plan):
        raise ValueError(f"plan has {len(plan)} entries for {len(blocks)} blocks")
    canonical: dict[str, nn.Module] = {}
    out = []
    for i, (blk, t) in enumerate(zip(blocks, plan)):
        if t not in canonical:
            canonical[t] = blk
        out.append(SharedBlockRef(canonical[t], i))
    return nn.ModuleList(out), canonical
