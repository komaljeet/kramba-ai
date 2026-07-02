"""
HVNBlock
========

The wrapper that links the fluid vortex step (VortexInjectionLayer) to the
quantum state evolution (HamiltonianMemoryBlock).

For each token in the sequence:
  1. VortexInjectionLayer computes the velocity field v_t = Γ · (H · ω_t) and
     produces the deflected representation r̃_t.
  2. HamiltonianMemoryBlock absorbs v_t into H via entanglement renormalization.

Key insight: the Hamiltonian H is updated *sequentially* token-by-token,
but each update is O(d_model²) — constant in N.

**Multi-head Hamiltonian**: Instead of a single H matrix, we maintain `n_heads`
parallel Hamiltonian matrices, each operating on a `d_head = d_model // n_heads`
slice of the hidden dimension.  This is analogous to multi-head attention and
allows the model to capture different types of associations simultaneously.

**Chunked parallel processing**: For training speed, tokens are processed in
chunks of `chunk_size`.  Within a chunk, all tokens share the same Hamiltonian
state (computed from the previous chunk), enabling parallel GPU computation.
The Hamiltonian is then updated sequentially with the chunk's velocities.  This
trades a small amount of causal accuracy (tokens within a chunk see the H from
*before* the chunk, not the intra-chunk updates) for a large speedup.
"""

import torch
import torch.nn as nn
from .vortex_injection import VortexInjectionLayer
from .hamiltonian_memory import HamiltonianMemoryBlock


class HVNBlock(nn.Module):
    """A single Hamiltonian Vortex Network block with multi-head memory.

    Combines fluid-dynamical vortex injection with quantum entanglement memory
    evolution into a single residual block.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    n_heads : int
        Number of parallel Hamiltonian heads (like multi-head attention).
        Each head operates on d_model // n_heads dimensions.
    bond_dimension : int, optional
        SVD truncation rank for quantum renormalization.  None = full rank.
    chunk_size : int | None
        Number of tokens to process in parallel within a chunk.  Larger = faster
        but less causally accurate within the chunk.  Defaults to 64, and can be
        set to None to use an adaptive chunk size based on the sequence length.
    dropout : float
        Dropout rate applied to the block output.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        bond_dimension: int | None = None,
        chunk_size: int | None = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        assert d_model % n_heads == 0, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size

        # Per-head vortex injection layers
        self.vortex_heads = nn.ModuleList([
            VortexInjectionLayer(self.d_head)
            for _ in range(n_heads)
        ])

        # Per-head quantum memory
        self.memory_heads = nn.ModuleList([
            HamiltonianMemoryBlock(self.d_head, bond_dimension=bond_dimension)
            for _ in range(n_heads)
        ])

        # Input projection: d_model → n_heads * d_head (identity if d_model == n_heads * d_head)
        self.input_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.input_proj.weight)

        # Output projection: n_heads * d_head → d_model
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.output_proj.weight)

        # Feed-forward sub-layer (2x expansion)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

        # Layer norms for pre-norm residual architecture
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        # Dropout for the vortex sub-layer output
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Process a sequence through the multi-head HVN block.

        Uses chunked parallel processing: tokens within a chunk share the same
        Hamiltonian state, enabling parallel GPU computation.  The Hamiltonian
        is updated between chunks.

        Parameters
        ----------
        x : torch.Tensor
            Input embeddings, shape [batch, seq_len, d_model].

        Returns
        -------
        output : torch.Tensor
            Output representations, shape [batch, seq_len, d_model].
        """
        B, S, D = x.shape

        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)

        # Pre-norm
        x_norm = self.norm1(x)  # [B, S, D]

        # Project input to per-head dimensions
        x_proj = self.input_proj(x_norm)  # [B, S, D]
        # Split into heads: [B, S, n_heads, d_head]
        x_heads = x_proj.view(B, S, self.n_heads, self.d_head)

        # Process each head
        head_outputs = []
        for h in range(self.n_heads):
            x_h = x_heads[:, :, h, :]  # [B, S, d_head]
            out_h = self._process_head_chunked(h, x_h, B, S, position_ids)
            head_outputs.append(out_h)

        # Concatenate heads and project: [B, S, D]
        deflected = torch.cat(head_outputs, dim=-1)  # [B, S, D]
        deflected = self.output_proj(deflected)  # [B, S, D]

        # Residual connection
        x = x + self.dropout(deflected)

        # Feed-forward sub-layer with residual
        x = x + self.ffn(self.norm2(x))

        return x

    def _resolve_chunk_size(self, seq_len: int) -> int:
        """Choose a chunk size that balances parallelism and accuracy."""
        if self.chunk_size is None:
            # For short prompts, use the full sequence to avoid extra bookkeeping.
            # For longer sequences, use a larger chunk to keep the GPU busy.
            return max(1, min(128, seq_len))
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be a positive integer, got {self.chunk_size}")
        return int(self.chunk_size)

    def _process_head_chunked(
        self,
        head_idx: int,
        x_h: torch.Tensor,
        B: int,
        S: int,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Process one head with chunked parallel vortex injection.

        Tokens within a chunk share the same Hamiltonian state (from before
        the chunk), enabling parallel computation.  After each chunk, the
        Hamiltonian is updated with all velocities from that chunk.
        """
        vortex = self.vortex_heads[head_idx]
        memory = self.memory_heads[head_idx]

        # Initialize Hamiltonian for this head
        H = memory.init_hamiltonian(B, device=x_h.device, dtype=x_h.dtype)  # [B, d_head, d_head]

        chunk_size = self._resolve_chunk_size(S)
        all_outputs = []

        # Process in chunks
        for chunk_start in range(0, S, chunk_size):
            chunk_end = min(chunk_start + chunk_size, S)
            x_chunk = x_h[:, chunk_start:chunk_end, :]  # [B, chunk_len, d_head]

            # All tokens in this chunk use the SAME Hamiltonian H
            # This enables parallel computation across the chunk
            chunk_position_ids = None
            if position_ids is not None:
                chunk_position_ids = position_ids[:, chunk_start:chunk_end]
            deflected_chunk, velocity_chunk = vortex(x_chunk, H, position_ids=chunk_position_ids)  # [B, chunk_len, d_head] x2
            all_outputs.append(deflected_chunk)

            # Update Hamiltonian with each token's velocity (sequential within chunk)
            chunk_len = chunk_end - chunk_start
            for t in range(chunk_len):
                H = memory.update(H, velocity_chunk[:, t, :])

        return torch.cat(all_outputs, dim=1)  # [B, S, d_head]

    def init_state(self, batch_size: int, device=None, dtype=None) -> list[list[torch.Tensor]]:
        """Initialize the causal Hamiltonian state for all heads (incremental generation).

        Returns
        -------
        state : list[list[torch.Tensor]]
            One Hamiltonian matrix per head: [[H_0], [H_1], ..., [H_{n_heads-1}]].
        """
        return [
            [head.init_hamiltonian(batch_size, device=device, dtype=dtype)]
            for head in self.memory_heads
        ]

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: list[list[torch.Tensor]],
        position_idx: int | None = None,
    ) -> tuple[torch.Tensor, list[list[torch.Tensor]]]:
        """Process a SINGLE token incrementally — O(1) per token generation.

        Parameters
        ----------
        x_t : torch.Tensor
            Single token embedding, shape [batch, 1, d_model].
        state : list[list[torch.Tensor]]
            Per-head Hamiltonian states: [[H_0], [H_1], ...].

        Returns
        -------
        output : torch.Tensor
            Output for this token, shape [batch, 1, d_model].
        new_state : list[list[torch.Tensor]]
            Updated per-head Hamiltonian states.
        """
        B, _, D = x_t.shape

        if position_idx is None:
            position_ids = torch.arange(1, device=x_t.device).unsqueeze(0).expand(B, -1)
        else:
            position_ids = torch.full((B, 1), int(position_idx), device=x_t.device, dtype=torch.long)

        # Pre-norm + project
        x_norm = self.norm1(x_t)  # [B, 1, D]
        x_proj = self.input_proj(x_norm)  # [B, 1, D]
        x_heads = x_proj.view(B, 1, self.n_heads, self.d_head)  # [B, 1, n_heads, d_head]

        head_outputs = []
        new_state = []
        for h in range(self.n_heads):
            x_h = x_heads[:, :, h, :]  # [B, 1, d_head]
            H = state[h][0]  # [B, d_head, d_head]

            # Vortex injection
            deflected_t, velocity_t = self.vortex_heads[h](x_h, H, position_ids=position_ids)

            # Update Hamiltonian
            H_new = self.memory_heads[h].update(H, velocity_t.squeeze(1))

            head_outputs.append(deflected_t)
            new_state.append([H_new])

        # Concatenate and project
        deflected = torch.cat(head_outputs, dim=-1)  # [B, 1, D]
        deflected = self.output_proj(deflected)

        # Residual + FFN
        x_t = x_t + self.dropout(deflected)
        x_t = x_t + self.ffn(self.norm2(x_t))

        return x_t, new_state