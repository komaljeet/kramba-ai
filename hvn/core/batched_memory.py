"""
Batched Multi-Head Hamiltonian Memory
======================================

A fully batched version of HamiltonianMemoryBlock that processes ALL heads
simultaneously by stacking per-head parameters into batched tensors.

Instead of looping over n_heads (each with its own eta, memory_gate, etc.),
this module uses batched operations so all heads update in parallel.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class MultiHeadHamiltonianMemory(nn.Module):
    """Batched multi-head Hamiltonian memory with deterministic MPS-style updates.

    Combines n_heads HamiltonianMemoryBlock instances into a single module
    that processes all heads in parallel using batched operations.

    Parameters
    ----------
    d_head : int
        Per-head dimensionality.
    n_heads : int
        Number of parallel heads.
    entanglement_rate : float, optional
        Initial eta value (one per head).
    bond_dimension : int, optional
        SVD truncation rank. None = full rank.
    """

    def __init__(
        self,
        d_head: int,
        n_heads: int,
        entanglement_rate: Optional[float] = None,
        bond_dimension: Optional[int] = None,
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.bond_dimension = bond_dimension

        if entanglement_rate is None:
            entanglement_rate = 1.0 / math.sqrt(d_head)

        # Per-head entanglement rate (learnable)
        self.eta = nn.Parameter(torch.full((n_heads,), entanglement_rate))

        # Per-head memory gate (learnable)
        self.memory_gate = nn.Parameter(torch.ones(n_heads))

    def init_hamiltonian(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Initialize Hamiltonian matrices for all heads.

        Returns
        -------
        H : torch.Tensor, shape [batch * n_heads, d_head, d_head]
            One Hamiltonian per (batch element, head) pair.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        BH = batch_size * self.n_heads
        H = torch.eye(self.d_head, device=device, dtype=dtype)
        H = H.unsqueeze(0).expand(BH, -1, -1).contiguous()
        H = H / self.d_head
        return H

    def update_batch(
        self,
        hamiltonian: torch.Tensor,
        velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Batched Hamiltonian update for all heads simultaneously.

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Per-head Hamiltonian matrices, shape [B*n_heads, d_head, d_head].
        velocities : torch.Tensor
            Velocity vectors, shape [B*n_heads, seq_len, d_head].

        Returns
        -------
        hamiltonian_new : torch.Tensor, shape [B*n_heads, d_head, d_head]
        """
        BH, S, D = velocities.shape

        # Per-head eta: [n_heads] → [BH] (each head gets its own eta)
        eta = torch.clamp(self.eta, min=0.0, max=1.0)  # [n_heads]
        # Expand to match BH: repeat each head's eta for each batch element
        eta_expanded = eta.repeat_interleave(BH // self.n_heads)  # [BH]
        eta_b = eta_expanded.view(BH, 1, 1)  # [BH, 1, 1]

        # --- Normalise all velocity vectors ---
        v_norms = velocities.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [BH, S, 1]
        v = velocities / v_norms  # [BH, S, D]

        # --- Batched rank-1 projectors sum ---
        P_sum = torch.bmm(v.transpose(-1, -2), v)  # [BH, D, D]

        # --- Destructive interference: H · P_sum · H (computed once) ---
        Hv = torch.bmm(hamiltonian, v.transpose(-1, -2))  # [BH, D, S]
        Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v.transpose(-1, -2))  # [BH, D, S]
        HPH_sum = torch.bmm(Hv, Ht_v.transpose(-1, -2))  # [BH, D, D]

        # --- Batched entanglement update ---
        hamiltonian_new = (1.0 - eta_b) * hamiltonian + eta_b * (P_sum - HPH_sum) / S

        # --- Spectral norm clamping ---
        fro_norm = hamiltonian_new.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        max_norm = (D ** 0.5)
        scale = torch.clamp(max_norm / fro_norm, max=1.0)
        hamiltonian_new = hamiltonian_new * scale

        # --- Optional: low-rank truncation ---
        if self.bond_dimension is not None and self.bond_dimension < D:
            hamiltonian_new = self._svd_truncate_batched(hamiltonian_new)

        # --- Per-head memory gate ---
        gate_expanded = self.memory_gate.repeat_interleave(BH // self.n_heads)  # [BH]
        hamiltonian_new = hamiltonian_new * gate_expanded.view(BH, 1, 1)

        return hamiltonian_new

    def _svd_truncate_batched(self, H: torch.Tensor) -> torch.Tensor:
        """Batched low-rank renormalization via randomized power iteration."""
        BH, D, _ = H.shape
        r = min(self.bond_dimension, D)
        n_iter = 3

        with torch.no_grad():
            Omega = torch.randn(BH, D, r, device=H.device, dtype=H.dtype)
            Omega = Omega / math.sqrt(r)

            G = torch.bmm(H, H.transpose(-1, -2))  # [BH, D, D]
            G = G + 1e-5 * torch.eye(D, device=H.device, dtype=H.dtype).unsqueeze(0)

            Q = Omega
            for _ in range(n_iter):
                Q = torch.bmm(G, Q)
                Q, _ = torch.linalg.qr(Q)

        Qt_H = torch.bmm(Q.transpose(-1, -2), H)
        Qt_H_Q = torch.bmm(Qt_H, Q)
        H_trunc = torch.bmm(Q, torch.bmm(Qt_H_Q, Q.transpose(-1, -2)))

        return H_trunc

    def update_single(
        self,
        hamiltonian: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Update Hamiltonian with a single velocity vector (for incremental generation).

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Shape [B*n_heads, d_head, d_head].
        velocity : torch.Tensor
            Shape [B*n_heads, d_head].
        """
        BH, D, _ = hamiltonian.shape

        eta = torch.clamp(self.eta, min=0.0, max=1.0)
        eta_expanded = eta.repeat_interleave(BH // self.n_heads)
        eta_b = eta_expanded.view(BH, 1, 1)

        v_norm = velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = (velocity / v_norm).unsqueeze(-1)  # [BH, D, 1]

        P = v * v.transpose(-1, -2)  # [BH, D, D]

        Hv = torch.bmm(hamiltonian, v)
        Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v)
        HPH = Hv * Ht_v.transpose(-1, -2)

        hamiltonian_new = (1.0 - eta_b) * hamiltonian + eta_b * (P - HPH)

        fro_norm = hamiltonian_new.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        max_norm = (D ** 0.5)
        scale = torch.clamp(max_norm / fro_norm, max=1.0)
        hamiltonian_new = hamiltonian_new * scale

        if self.bond_dimension is not None and self.bond_dimension < D:
            hamiltonian_new = self._svd_truncate_batched(hamiltonian_new)

        gate_expanded = self.memory_gate.repeat_interleave(BH // self.n_heads)
        hamiltonian_new = hamiltonian_new * gate_expanded.view(BH, 1, 1)

        return hamiltonian_new