"""
2:4 Block-Sparse Weights (Issue #32, Phase 1.4)
=================================================

Enforces NVIDIA Ampere+ compatible 2:4 structured sparsity: within every
group of 4 consecutive weights along the input dimension, only the 2
largest-magnitude values survive.  This is the sparsity pattern NVIDIA
tensor cores accelerate natively (2x speedup, zero quality loss on
compatible hardware), and it halves weight storage since the sparsity
pattern needs only a small fixed-size index per group rather than a
per-element mask.

:class:`BlockSparseLinear` stores weights densely (masked) so it runs
correctly on any device including GPUs without sparse tensor core support;
:meth:`BlockSparseLinear.memory_bytes` reports the storage a packed 2:4
representation would actually use, for memory-budget accounting.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_2_4_sparsity(W: torch.Tensor) -> torch.Tensor:
    """Prune a weight matrix to 2:4 structured sparsity along the last dimension.

    For every group of 4 consecutive elements along ``dim=-1``, keeps the 2
    largest-magnitude values and zeros the rest.  If the last dimension is
    not a multiple of 4, it is zero-padded before pruning and the result is
    cropped back to the original size.

    Parameters
    ----------
    W : torch.Tensor
        Weight matrix, shape ``[..., in_features]``.

    Returns
    -------
    torch.Tensor
        Same shape as *W*, with 50% of each group of 4 elements zeroed.
    """
    *lead, cols = W.shape
    pad = (4 - cols % 4) % 4
    Wp = F.pad(W, (0, pad)) if pad else W
    groups = Wp.reshape(*lead, -1, 4)
    _, idx = groups.abs().topk(2, dim=-1)
    mask = torch.zeros_like(groups, dtype=torch.bool)
    mask.scatter_(-1, idx, True)
    pruned = torch.where(mask, groups, torch.zeros_like(groups))
    pruned = pruned.reshape(*lead, -1)
    return pruned[..., :cols]


class BlockSparseLinear(nn.Module):
    """``nn.Linear`` drop-in with a fixed 2:4 structured sparsity pattern.

    The sparsity mask is computed once (at construction, or from an
    existing dense layer via :meth:`from_linear`) and re-applied on every
    forward pass, so gradient updates during any subsequent fine-tuning
    cannot regrow pruned positions.

    Parameters
    ----------
    in_features, out_features : int
        As in ``nn.Linear``.
    bias : bool
        Whether to include a bias term (bias is never sparsified).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.register_buffer("mask", torch.ones_like(self.weight, dtype=torch.bool))
        self._apply_mask()

    def _apply_mask(self) -> None:
        with torch.no_grad():
            pruned = apply_2_4_sparsity(self.weight.data)
            self.mask = pruned != 0
            self.weight.data = pruned

    @classmethod
    def from_linear(cls, linear: nn.Linear) -> "BlockSparseLinear":
        """Build a :class:`BlockSparseLinear` from a trained ``nn.Linear``.

        Prunes the existing weights to 2:4 sparsity in-place (does not
        retrain), copies the bias if present.
        """
        module = cls(linear.in_features, linear.out_features, bias=linear.bias is not None)
        module.weight.data = linear.weight.data.clone()
        module._apply_mask()
        if module.bias is not None:
            module.bias.data = linear.bias.data.clone()
        module.to(linear.weight.device)
        return module

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight * self.mask  # re-enforce sparsity even if weight was fine-tuned
        return F.linear(x, w, self.bias)

    def sparsity(self) -> float:
        """Fraction of weights that are exactly zero (should be ~0.5 for 2:4)."""
        return 1.0 - (self.mask.sum().item() / self.mask.numel())

    def memory_bytes(self, value_bits: int = 16) -> int:
        """Estimated storage if packed as a real 2:4 sparse tensor.

        A hardware 2:4 representation stores only the ``nnz`` surviving
        values (at *value_bits* each) plus a 2-bit index per group of 4
        identifying which 2 positions survived (standard Ampere metadata
        format), rather than a full dense mask.
        """
        nnz = self.mask.sum().item()
        value_bytes = nnz * value_bits / 8.0
        n_groups = self.weight.numel() / 4.0
        index_bytes = n_groups * 4 / 8.0  # 2 indices x 2 bits per group
        bias_bytes = self.bias.numel() * 4 if self.bias is not None else 0
        return int(value_bytes + index_bytes + bias_bytes)


def convert_model_to_block_sparse(model: nn.Module, target_names: list[str] | None = None) -> dict[str, float]:
    """Replace named ``nn.Linear`` submodules in *model* with :class:`BlockSparseLinear`.

    Parameters
    ----------
    model : nn.Module
        Model to modify in-place.
    target_names : list[str], optional
        Dotted module names of the ``nn.Linear`` layers to convert.  If
        ``None``, every ``nn.Linear`` in the model is converted.

    Returns
    -------
    dict[str, float]
        Layer name -> achieved sparsity fraction.
    """
    linear_modules = {name: mod for name, mod in model.named_modules() if isinstance(mod, nn.Linear)}
    names = target_names or list(linear_modules.keys())
    results: dict[str, float] = {}

    for name in names:
        if name not in linear_modules:
            continue
        *parent_path, attr = name.split(".")
        parent = model
        for p in parent_path:
            parent = getattr(parent, p)
        sparse_layer = BlockSparseLinear.from_linear(linear_modules[name])
        setattr(parent, attr, sparse_layer)
        results[name] = sparse_layer.sparsity()

    return results
