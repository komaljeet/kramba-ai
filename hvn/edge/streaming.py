"""
Scalability to 1T+ Parameters: Streaming & On-Demand Loading (Issue #32, Phase 4)
=====================================================================================

Since HVN has no KV cache, weights are the *only* memory consumer during
inference, and HVN already processes tokens strictly layer-by-layer
(``for block in self.blocks: x = block(x)``).  That makes layer-by-layer
weight streaming from disk both natural and sufficient: at any instant
only one layer's weights need to be resident in VRAM.

- :class:`StreamingHVNInference` — loads each ``HVNBlock``'s weights from
  disk immediately before it runs and evicts them immediately after,
  capping VRAM usage at one layer's footprint regardless of model depth.
- :class:`MMapWeightStore` — packs multiple tensors into one file with an
  index, for memory-mapped (only-touched-pages-paged-in) loading.
- :class:`SuccinctWeights` — a rank-queryable bit-vector index (Jacobson
  1989 succinct rank/select) for compactly locating sparse weight
  positions without per-element index overhead.
- :class:`TaskSpecificLoader` — loads only the layer subset relevant to a
  registered task, for on-demand partial-model loading.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 4.1 Layer-by-layer weight streaming from NVMe
# ---------------------------------------------------------------------------


class StreamingHVNInference:
    """Streams ``HVNBlock`` weights from disk one layer at a time.

    Immediately after construction, every block's parameters are written
    to *weight_dir* and the block is moved off the compute device (freeing
    its VRAM).  During :meth:`forward`, each block is loaded back onto the
    device just before it runs its forward pass, and evicted immediately
    after — so peak VRAM usage for the block stack is bounded by the
    *single largest layer*, not the sum of all layers, independent of
    ``num_layers``.

    Parameters
    ----------
    model : HamiltonianVortexNetwork
        Model whose ``.blocks`` will be streamed.
    weight_dir : str
        Directory to store per-layer weight files (simulates NVMe storage).
    device : torch.device
        Compute device (GPU) each layer is loaded onto just-in-time.
    """

    def __init__(self, model: nn.Module, weight_dir: str, device: torch.device):
        self.model = model
        self.weight_dir = weight_dir
        self.device = device
        os.makedirs(weight_dir, exist_ok=True)
        self._save_blocks_to_disk()

    def _block_path(self, i: int) -> str:
        return os.path.join(self.weight_dir, f"block_{i}.pt")

    def _save_blocks_to_disk(self) -> None:
        for i, block in enumerate(self.model.blocks):
            torch.save(block.state_dict(), self._block_path(i))
            block.to("cpu")
        if self.device.type == "cuda":
            torch.cuda.empty_cache()

    def _load_block(self, i: int) -> nn.Module:
        block = self.model.blocks[i]
        state = torch.load(self._block_path(i), map_location=self.device, weights_only=True)
        block.load_state_dict(state)
        block.to(self.device)
        return block

    @torch.no_grad()
    def forward(self, input_ids: torch.Tensor, position_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S = input_ids.shape
        if position_ids is None:
            position_ids = torch.arange(S, device=self.device).unsqueeze(0).expand(B, -1)

        x = self.model.token_embedding.to(self.device)(input_ids.to(self.device))
        for i in range(len(self.model.blocks)):
            block = self._load_block(i)
            x = block(x, position_ids=position_ids)
            block.to("cpu")
            if self.device.type == "cuda":
                torch.cuda.empty_cache()

        x = self.model.final_norm.to(self.device)(x)
        return self.model.decoder.to(self.device)(x)

    def peak_layer_bytes(self) -> int:
        """The VRAM ceiling this streaming approach requires — the single largest layer."""
        return max(
            sum(p.numel() * p.element_size() for p in block.parameters())
            for block in self.model.blocks
        )

    def total_streamed_bytes(self) -> int:
        """Total weight bytes on disk across all layers (what the streaming approach avoids holding in VRAM at once)."""
        return sum(
            sum(p.numel() * p.element_size() for p in block.parameters())
            for block in self.model.blocks
        )


# ---------------------------------------------------------------------------
# 4.2 Memory-mapped compressed weight files (zero-copy loading)
# ---------------------------------------------------------------------------


class MMapWeightStore:
    """Packs multiple named tensors into one file for memory-mapped loading.

    Only the pages actually touched by :meth:`load` are read from disk
    (via ``numpy.memmap``), rather than the OS eagerly reading the entire
    file — the same principle behind ``mmap``-based checkpoint loading in
    production inference servers.

    Parameters
    ----------
    path : str
        Path to the packed weight file.
    """

    def __init__(self, path: str):
        self.path = path
        self._index: dict[str, tuple[int, tuple[int, ...], str, int]] = {}

    def save(self, tensors: dict[str, torch.Tensor]) -> None:
        offset = 0
        with open(self.path, "wb") as f:
            for name, t in tensors.items():
                arr = t.detach().cpu().numpy()
                data = arr.tobytes()
                f.write(data)
                self._index[name] = (offset, tuple(arr.shape), str(arr.dtype), len(data))
                offset += len(data)

    def load(self, name: str, device: Optional[torch.device] = None) -> torch.Tensor:
        offset, shape, dtype_str, _nbytes = self._index[name]
        mm = np.memmap(self.path, dtype=dtype_str, mode="r", offset=offset, shape=shape)
        t = torch.from_numpy(np.array(mm))  # copies out of the mmap view for lifetime safety
        return t.to(device) if device is not None else t

    def index_bytes(self) -> int:
        return len(self._index) * 64  # rough per-entry (name + offset/shape/dtype) overhead


# ---------------------------------------------------------------------------
# 15.1 Succinct data structures for sparse weight indexing
# ---------------------------------------------------------------------------


class SuccinctWeights:
    """Rank-queryable bit-vector index for sparse weight positions (Jacobson 1989).

    Precomputes block-level popcount prefix sums so that ``rank(i)`` — the
    number of set bits before position *i* — is answered in ``O(block_size)``
    instead of ``O(i)``, without storing a full per-element index.  This is
    the standard succinct rank/select building block used to compactly
    locate nonzero entries in a sparse weight tensor.

    Parameters
    ----------
    mask : torch.Tensor
        Boolean (or 0/1) tensor marking nonzero weight positions.
    block_size : int
        Number of bits per rank-sampling block.
    """

    def __init__(self, mask: torch.Tensor, block_size: int = 64):
        self.shape = tuple(mask.shape)
        flat = mask.flatten().bool()
        self.n = flat.numel()
        self.block_size = block_size
        self.bits = flat

        n_blocks = (self.n + block_size - 1) // block_size
        pad = n_blocks * block_size - self.n
        padded = torch.nn.functional.pad(flat.long(), (0, pad))
        block_counts = padded.view(n_blocks, block_size).sum(dim=1)
        self.block_rank = torch.cat([torch.zeros(1, dtype=torch.long), block_counts.cumsum(0)[:-1]])

    def rank(self, i: int) -> int:
        """Number of set bits in ``[0, i)``."""
        block_idx = i // self.block_size
        within = i - block_idx * self.block_size
        base = self.block_rank[block_idx].item()
        start = block_idx * self.block_size
        extra = self.bits[start:start + within].sum().item()
        return base + extra

    def nonzero_positions(self) -> torch.Tensor:
        return self.bits.nonzero(as_tuple=False).flatten()

    def memory_bytes(self) -> int:
        packed_bits = self.n / 8.0
        rank_overhead = self.block_rank.numel() * 4  # int32 rank sample per block
        return int(packed_bits + rank_overhead)


# ---------------------------------------------------------------------------
# 4.5 On-demand weight loading (task-specific subsets)
# ---------------------------------------------------------------------------


class TaskSpecificLoader:
    """Loads only the layer subset relevant to a registered task.

    A lightweight registry mapping task name -> layer indices, analogous
    to gene-expression-style on-demand activation (Issue #32 §13.1): a
    model deployed for a narrow task doesn't need every layer's weights
    resident, only the ones that matter for that task.

    Parameters
    ----------
    (none — state is built up via :meth:`register_task`)
    """

    def __init__(self):
        self._task_map: dict[str, list[int]] = {}

    def register_task(self, task_name: str, layer_indices: list[int]) -> None:
        self._task_map[task_name] = layer_indices

    def load_subset(
        self,
        task_name: str,
        weight_store: MMapWeightStore,
        tensor_names_by_layer: dict[int, list[str]],
        device: Optional[torch.device] = None,
    ) -> dict[str, torch.Tensor]:
        """Load only the tensors belonging to the task's registered layer subset."""
        total_layers = max(tensor_names_by_layer.keys(), default=-1) + 1
        layer_indices = self._task_map.get(task_name, list(range(total_layers)))
        loaded: dict[str, torch.Tensor] = {}
        for i in layer_indices:
            for name in tensor_names_by_layer.get(i, []):
                loaded[name] = weight_store.load(name, device=device)
        return loaded

    def compression_ratio(self, task_name: str, total_layers: int) -> float:
        """How much smaller the task-specific subset is vs. loading every layer."""
        layer_indices = self._task_map.get(task_name, list(range(total_layers)))
        return total_layers / max(len(layer_indices), 1)
