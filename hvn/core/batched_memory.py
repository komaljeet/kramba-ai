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
        use_residual: bool = True,
        use_input_dependent_eta: bool = True,
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.bond_dimension = bond_dimension
        self.use_residual = use_residual
        self.use_input_dependent_eta = use_input_dependent_eta

        if entanglement_rate is None:
            entanglement_rate = 1.0 / math.sqrt(d_head)

        # Per-head entanglement rate (learnable, used as fallback or base rate)
        self.eta = nn.Parameter(torch.full((n_heads,), entanglement_rate))

        # Per-head memory gate (learnable)
        self.memory_gate = nn.Parameter(torch.ones(n_heads))

        # Input-dependent eta projection: η_t = σ(W_η · x_t)
        # Projects from d_head to n_heads (one eta per head per token)
        if use_input_dependent_eta:
            self.eta_proj = nn.Linear(d_head, n_heads, bias=True)
            nn.init.zeros_(self.eta_proj.weight)
            nn.init.zeros_(self.eta_proj.bias)  # initial eta ≈ 0.5 after sigmoid

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
        inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batched Hamiltonian update for all heads simultaneously.

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Per-head Hamiltonian matrices, shape [B*n_heads, d_head, d_head].
        velocities : torch.Tensor
            Velocity vectors, shape [B*n_heads, seq_len, d_head].
        inputs : torch.Tensor, optional
            Input embeddings for input-dependent eta, shape [B*n_heads, seq_len, d_head].
            Required if use_input_dependent_eta=True.

        Returns
        -------
        hamiltonian_new : torch.Tensor, shape [B*n_heads, d_head, d_head]
        """
        BH, S, D = velocities.shape

        # --- Compute eta (entanglement rate) ---
        if self.use_input_dependent_eta and inputs is not None:
            # Input-dependent: η_t = σ(W_η · x_t) → per-token, per-head
            # inputs: [BH, S, D] → eta_raw: [BH, S, n_heads]
            eta_raw = self.eta_proj(inputs)  # [BH, S, n_heads]
            # Average over sequence to get per-(batch,head) eta for the chunk
            eta_avg = torch.sigmoid(eta_raw).mean(dim=1)  # [BH, n_heads]
            # Each batch element bh corresponds to head (bh % n_heads)
            # Gather the diagonal: eta_b[bh] = eta_avg[bh, bh % n_heads]
            head_indices = torch.arange(BH, device=inputs.device) % self.n_heads
            eta_b = eta_avg.gather(1, head_indices.unsqueeze(1)).view(BH, 1, 1)  # [BH, 1, 1]
        else:
            # Fallback to learned per-head eta
            eta = torch.clamp(self.eta, min=0.0, max=1.0)  # [n_heads]
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

        # --- Entanglement update ---
        if self.use_residual:
            # Residual connection: H_new = H_old + η · (P - H·P·H) / S
            # This preserves long-range information better than the blend
            delta_H = eta_b * (P_sum - HPH_sum) / S
            hamiltonian_new = hamiltonian + delta_H
        else:
            # Original blend: H_new = (1-η)·H + η·(P - H·P·H) / S
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
        """Batched low-rank renormalization via randomized power iteration.

        Handles half-precision (fp16/bf16) by running the QR decomposition in
        float32, then casting the result back to the original dtype.
        """
        BH, D, _ = H.shape
        r = min(self.bond_dimension, D)
        n_iter = 3
        orig_dtype = H.dtype
        # torch.linalg.qr doesn't support fp16/bf16 — upcast for the no-grad section
        work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype

        with torch.no_grad():
            # Disable autocast to ensure QR runs in float32
            with torch.amp.autocast("cuda", enabled=False):
                H_f32 = H.to(work_dtype)
                Omega = torch.randn(BH, D, r, device=H.device, dtype=work_dtype)
                Omega = Omega / math.sqrt(r)

                G = torch.bmm(H_f32, H_f32.transpose(-1, -2))  # [BH, D, D]
                G = G + 1e-5 * torch.eye(D, device=H.device, dtype=work_dtype).unsqueeze(0)

                Q = Omega
                for _ in range(n_iter):
                    Q = torch.bmm(G, Q)
                    Q, _ = torch.linalg.qr(Q)

        # Project H in float32 (outside autocast) then cast back
        with torch.amp.autocast("cuda", enabled=False):
            H_work = H.to(work_dtype)
            Qt_H = torch.bmm(Q.transpose(-1, -2), H_work)
            Qt_H_Q = torch.bmm(Qt_H, Q)
            H_trunc = torch.bmm(Q, torch.bmm(Qt_H_Q, Q.transpose(-1, -2)))
        return H_trunc.to(orig_dtype)

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