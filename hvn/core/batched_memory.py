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
    rank_k : int, optional
        Rank-k entanglement update dimension.  When > 1, the projector
        P_t = Σ_{i=1}^{k} w_i ⊗ w_i^T is rank-k instead of rank-1,
        making each token's update k times more expressive.  Default: 1
        (backward-compatible rank-1 behaviour).
    use_nonlinear_update : bool, optional
        If True, apply a gated nonlinearity to the projector vectors before
        forming the outer product (Strategy 4).  This increases expressiveness
        analogously to how attention uses softmax (nonlinear) on QK^T.
    """

    def __init__(
        self,
        d_head: int,
        n_heads: int,
        entanglement_rate: Optional[float] = None,
        bond_dimension: Optional[int] = None,
        use_residual: bool = True,
        use_input_dependent_eta: bool = True,
        rank_k: int = 1,
        use_nonlinear_update: bool = False,
    ):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads
        self.bond_dimension = bond_dimension
        self.use_residual = use_residual
        self.use_input_dependent_eta = use_input_dependent_eta
        self.rank_k = rank_k
        self.use_nonlinear_update = use_nonlinear_update

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

        # --- Strategy 1: Rank-k projector ---
        # Learn k projection vectors per token from the input.
        # W_proj: [n_heads, d_head, k * d_head] → project x_t → k vectors
        if rank_k > 1:
            self.rank_k_proj = nn.Linear(d_head, rank_k * d_head, bias=False)
            nn.init.xavier_uniform_(self.rank_k_proj.weight)

        # --- Strategy 4: Nonlinear Hamiltonian update ---
        # Gated nonlinear projector: P_t = (v_t ⊙ σ(W_gate · x_t)) ⊗ (·)^T
        # The gate allows the model to selectively suppress certain directions.
        if use_nonlinear_update:
            self.nonlinear_gate = nn.Linear(d_head, d_head, bias=True)
            nn.init.zeros_(self.nonlinear_gate.weight)
            nn.init.zeros_(self.nonlinear_gate.bias)

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

        Supports rank-k projectors (Strategy 1), nonlinear gating
        (Strategy 4), and per-token entanglement rate (Strategy 5).

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
            # Strategy 5: Per-token η — don't average over the sequence.
            # Instead, apply per-token η in the sequential H update loop.
            # For the batched (chunk-level) update, we still use the average
            # as the chunk-level eta, but the per-token path uses individual
            # eta values (see update_batch_per_token below).
            eta_raw = self.eta_proj(inputs)  # [BH, S, n_heads]
            eta_avg = torch.sigmoid(eta_raw).mean(dim=1)  # [BH, n_heads]
            head_indices = torch.arange(BH, device=inputs.device) % self.n_heads
            eta_b = eta_avg.gather(1, head_indices.unsqueeze(1)).view(BH, 1, 1)  # [BH, 1, 1]
        else:
            eta = torch.clamp(self.eta, min=0.0, max=1.0)  # [n_heads]
            eta_expanded = eta.repeat_interleave(BH // self.n_heads)  # [BH]
            eta_b = eta_expanded.view(BH, 1, 1)  # [BH, 1, 1]

        # --- Normalise all velocity vectors ---
        v_norms = velocities.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [BH, S, 1]
        v = velocities / v_norms  # [BH, S, D]

        # --- Strategy 4: Nonlinear gating of velocity vectors ---
        if self.use_nonlinear_update and inputs is not None:
            # gate = σ(W_gate · x_t) — selectively suppress directions
            gate = torch.sigmoid(self.nonlinear_gate(inputs))  # [BH, S, D]
            v = v * gate  # gated velocity

        # --- Strategy 1: Rank-k projector ---
        if self.rank_k > 1 and inputs is not None:
            # Project x_t → k vectors w_1..w_k, each in ℝ^d_head
            # rank_k_proj: [d_head, k*d_head] → inputs: [BH, S, D] → [BH, S, k*D]
            w = self.rank_k_proj(inputs)  # [BH, S, k*D]
            w = w.view(BH, S, self.rank_k, D)  # [BH, S, k, D]
            # Normalise each w_i
            w_norms = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [BH, S, k, 1]
            w = w / w_norms  # [BH, S, k, D]

            # P_sum = Σ_t Σ_{i=1}^{k} w_{t,i} ⊗ w_{t,i}^T
            # Reshape for batched outer product sum: [BH, S*k, D]
            w_flat = w.view(BH, S * self.rank_k, D)  # [BH, S*k, D]
            P_sum = torch.bmm(w_flat.transpose(-1, -2), w_flat)  # [BH, D, D]
            # Normalise by S (not S*k) to keep the update magnitude comparable
            P_sum = P_sum / S

            # Destructive interference with rank-k projector
            # H · P · H = Σ_t Σ_i (H w_{t,i}) ⊗ (H^T w_{t,i})^T
            # = (H w_flat^T) (H^T w_flat^T)^T  ... but we need H^T
            Hv = torch.bmm(hamiltonian, w_flat.transpose(-1, -2))  # [BH, D, S*k]
            Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), w_flat.transpose(-1, -2))  # [BH, D, S*k]
            HPH_sum = torch.bmm(Hv, Ht_v.transpose(-1, -2))  # [BH, D, D]
            HPH_sum = HPH_sum / S
        else:
            # --- Original rank-1 projectors sum ---
            P_sum = torch.bmm(v.transpose(-1, -2), v)  # [BH, D, D]

            # --- Destructive interference: H · P_sum · H (computed once) ---
            Hv = torch.bmm(hamiltonian, v.transpose(-1, -2))  # [BH, D, S]
            Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v.transpose(-1, -2))  # [BH, D, S]
            HPH_sum = torch.bmm(Hv, Ht_v.transpose(-1, -2))  # [BH, D, D]

        # --- Entanglement update ---
        if self.use_residual:
            # Residual connection: H_new = H_old + η · (P - H·P·H) / S
            delta_H = eta_b * (P_sum - HPH_sum)
            hamiltonian_new = hamiltonian + delta_H
        else:
            # Original blend: H_new = (1-η)·H + η·(P - H·P·H) / S
            hamiltonian_new = (1.0 - eta_b) * hamiltonian + eta_b * (P_sum - HPH_sum)

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

    def update_batch_per_token(
        self,
        hamiltonian: torch.Tensor,
        velocities: torch.Tensor,
        inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-token sequential Hamiltonian update with per-token η (Strategy 5).

        Instead of averaging η over the chunk, this method loops over tokens
        and applies a per-token entanglement rate η_t = σ(W_η · x_t).

        This is the truly per-token path — each token gets its own η value,
        allowing the model to selectively decide how much each token should
        update memory (like Mamba's selective scan).

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Per-head Hamiltonian matrices, shape [B*n_heads, d_head, d_head].
        velocities : torch.Tensor
            Velocity vectors, shape [B*n_heads, seq_len, d_head].
        inputs : torch.Tensor, optional
            Input embeddings, shape [B*n_heads, seq_len, d_head].

        Returns
        -------
        hamiltonian_new : torch.Tensor, shape [B*n_heads, d_head, d_head]
        """
        BH, S, D = velocities.shape
        H = hamiltonian  # [BH, D, D]

        # Pre-compute per-token eta if input-dependent
        if self.use_input_dependent_eta and inputs is not None:
            eta_raw = self.eta_proj(inputs)  # [BH, S, n_heads]
            eta_all = torch.sigmoid(eta_raw)  # [BH, S, n_heads]
            head_indices = torch.arange(BH, device=inputs.device) % self.n_heads
            # For each bh, gather eta_all[bh, :, head_indices[bh]]
            idx = head_indices.view(BH, 1, 1).expand(BH, S, 1)  # [BH, S, 1]
            eta_per_token = eta_all.gather(2, idx).squeeze(-1)  # [BH, S]
        else:
            eta = torch.clamp(self.eta, min=0.0, max=1.0)
            eta_expanded = eta.repeat_interleave(BH // self.n_heads)
            eta_per_token = eta_expanded.view(BH, 1).expand(BH, S)  # [BH, S]

        for t in range(S):
            v_t = velocities[:, t, :]  # [BH, D]
            x_t = inputs[:, t, :] if inputs is not None else None
            eta_t = eta_per_token[:, t].view(BH, 1, 1)  # [BH, 1, 1]

            # Normalise velocity
            v_norm = v_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v = v_t / v_norm  # [BH, D]

            # Strategy 4: Nonlinear gating
            if self.use_nonlinear_update and x_t is not None:
                gate = torch.sigmoid(self.nonlinear_gate(x_t))  # [BH, D]
                v = v * gate

            # Strategy 1: Rank-k projector
            if self.rank_k > 1 and x_t is not None:
                w = self.rank_k_proj(x_t)  # [BH, k*D]
                w = w.view(BH, self.rank_k, D)  # [BH, k, D]
                w_norms = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                w = w / w_norms  # [BH, k, D]
                # P = Σ_i w_i ⊗ w_i^T = w^T @ w  (per batch element)
                P = torch.bmm(w.transpose(-1, -2), w)  # [BH, D, D]
                # H·P·H = Σ_i (H w_i) ⊗ (H^T w_i)^T
                Hv = torch.bmm(hamiltonian, w.transpose(-1, -2))  # [BH, D, k]
                Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), w.transpose(-1, -2))  # [BH, D, k]
                HPH = torch.bmm(Hv, Ht_v.transpose(-1, -2))  # [BH, D, D]
            else:
                # Rank-1 projector
                v_col = v.unsqueeze(-1)  # [BH, D, 1]
                P = v_col * v_col.transpose(-1, -2)  # [BH, D, D]
                Hv = torch.bmm(hamiltonian, v_col)  # [BH, D, 1]
                Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v_col)  # [BH, D, 1]
                HPH = Hv * Ht_v.transpose(-1, -2)  # [BH, D, D]

            # Update
            if self.use_residual:
                delta_H = eta_t * (P - HPH)
                H = H + delta_H
            else:
                H = (1.0 - eta_t) * H + eta_t * (P - HPH)

            # Spectral norm clamping
            fro_norm = H.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
            max_norm = (D ** 0.5)
            scale = torch.clamp(max_norm / fro_norm, max=1.0)
            H = H * scale

        # SVD truncation (once after all tokens)
        if self.bond_dimension is not None and self.bond_dimension < D:
            H = self._svd_truncate_batched(H)

        # Per-head memory gate
        gate_expanded = self.memory_gate.repeat_interleave(BH // self.n_heads)
        H = H * gate_expanded.view(BH, 1, 1)

        return H

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
        input_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update Hamiltonian with a single velocity vector (for incremental generation).

        Supports rank-k projectors (Strategy 1) and nonlinear gating
        (Strategy 4) when input_vec is provided.

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Shape [B*n_heads, d_head, d_head].
        velocity : torch.Tensor
            Shape [B*n_heads, d_head].
        input_vec : torch.Tensor, optional
            Input embedding for this token, shape [B*n_heads, d_head].
            Required for rank-k and nonlinear update.
        """
        BH, D, _ = hamiltonian.shape

        # Per-token eta (input-dependent if available)
        if self.use_input_dependent_eta and input_vec is not None:
            eta_raw = self.eta_proj(input_vec)  # [BH, n_heads]
            head_indices = torch.arange(BH, device=hamiltonian.device) % self.n_heads
            eta_t = torch.sigmoid(eta_raw.gather(1, head_indices.unsqueeze(1))).view(BH, 1, 1)
        else:
            eta = torch.clamp(self.eta, min=0.0, max=1.0)
            eta_expanded = eta.repeat_interleave(BH // self.n_heads)
            eta_t = eta_expanded.view(BH, 1, 1)

        v_norm = velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = velocity / v_norm  # [BH, D]

        # Strategy 4: Nonlinear gating
        if self.use_nonlinear_update and input_vec is not None:
            gate = torch.sigmoid(self.nonlinear_gate(input_vec))  # [BH, D]
            v = v * gate

        # Strategy 1: Rank-k projector
        if self.rank_k > 1 and input_vec is not None:
            w = self.rank_k_proj(input_vec)  # [BH, k*D]
            w = w.view(BH, self.rank_k, D)  # [BH, k, D]
            w_norms = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            w = w / w_norms  # [BH, k, D]
            P = torch.bmm(w.transpose(-1, -2), w)  # [BH, D, D]
            Hv = torch.bmm(hamiltonian, w.transpose(-1, -2))  # [BH, D, k]
            Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), w.transpose(-1, -2))  # [BH, D, k]
            HPH = torch.bmm(Hv, Ht_v.transpose(-1, -2))  # [BH, D, D]
        else:
            v_col = v.unsqueeze(-1)  # [BH, D, 1]
            P = v_col * v_col.transpose(-1, -2)  # [BH, D, D]
            Hv = torch.bmm(hamiltonian, v_col)
            Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v_col)
            HPH = Hv * Ht_v.transpose(-1, -2)

        # Update
        if self.use_residual:
            delta_H = eta_t * (P - HPH)
            hamiltonian_new = hamiltonian + delta_H
        else:
            hamiltonian_new = (1.0 - eta_t) * hamiltonian + eta_t * (P - HPH)

        fro_norm = hamiltonian_new.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        max_norm = (D ** 0.5)
        scale = torch.clamp(max_norm / fro_norm, max=1.0)
        hamiltonian_new = hamiltonian_new * scale

        if self.bond_dimension is not None and self.bond_dimension < D:
            hamiltonian_new = self._svd_truncate_batched(hamiltonian_new)

        gate_expanded = self.memory_gate.repeat_interleave(BH // self.n_heads)
        hamiltonian_new = hamiltonian_new * gate_expanded.view(BH, 1, 1)

        return hamiltonian_new