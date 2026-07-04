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
            # Batch is batch-outer (bh = b*n_heads + h), so tile per-head eta with
            # repeat (not repeat_interleave) to match — Issue #46.  This mirrors the
            # input-dependent branch above, which already indexes batch-outer via
            # ``arange(BH) % n_heads``.
            eta_expanded = eta.repeat(BH // self.n_heads)  # [BH]
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
        gate_expanded = self.memory_gate.repeat(BH // self.n_heads)  # [BH]
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
            eta_expanded = eta.repeat(BH // self.n_heads)
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
        gate_expanded = self.memory_gate.repeat(BH // self.n_heads)
        H = H * gate_expanded.view(BH, 1, 1)

        return H

    def parallel_update(
        self,
        hamiltonian: torch.Tensor,
        velocities: torch.Tensor,
        inputs: torch.Tensor | None = None,
        correction_pass: bool = True,
    ) -> torch.Tensor:
        """Parallel scan Hamiltonian update (Issue #33 Fix 3).

        Reformulates the sequential Hamiltonian update as an associative scan
        (parallel prefix sum) to eliminate the O(S) sequential loop.

        **Key insight**: The full update ``H_new = (1-η)·H + η·(P - H·P·H)``
        is NOT directly associative.  But it can be **linearized** by dropping
        the destructive interference term for the scan:

            H_t ≈ (1-η)·H_{t-1} + η·P_t

        This is a **linear recurrence**: ``H_t = α·H_{t-1} + β_t`` where
        ``α = (1-η)`` and ``β_t = η·P_t``.  Linear recurrences ARE associative
        and can be parallel-scanned in O(log S) parallel depth.

        **Correction pass**: After the scan, we optionally re-apply the
        destructive interference term (H·P·H) using the scanned H.

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Per-head Hamiltonian matrices, shape [B*n_heads, d_head, d_head].
        velocities : torch.Tensor
            Velocity vectors, shape [B*n_heads, seq_len, d_head].
        inputs : torch.Tensor, optional
            Input embeddings (for input-dependent eta).
        correction_pass : bool
            If True, apply a correction pass that re-introduces the H·P·H
            destructive interference term.

        Returns
        -------
        hamiltonian_new : torch.Tensor, shape [B*n_heads, d_head, d_head]
        """
        BH, S, D = velocities.shape

        # --- Compute eta ---
        if self.use_input_dependent_eta and inputs is not None:
            eta_raw = self.eta_proj(inputs)  # [BH, S, n_heads]
            eta_avg = torch.sigmoid(eta_raw).mean(dim=1)  # [BH, n_heads]
            head_indices = torch.arange(BH, device=inputs.device) % self.n_heads
            eta = eta_avg.gather(1, head_indices.unsqueeze(1)).view(BH)  # [BH]
        else:
            eta = torch.clamp(self.eta, min=0.0, max=1.0)
            eta = eta.repeat(BH // self.n_heads)  # [BH]

        # --- Normalize velocities ---
        v_norms = velocities.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = velocities / v_norms  # [BH, S, D]

        # --- Strategy 4: Nonlinear gating ---
        if self.use_nonlinear_update and inputs is not None:
            gate = torch.sigmoid(self.nonlinear_gate(inputs))
            v = v * gate

        # --- Strategy 1: Rank-k projector ---
        if self.rank_k > 1 and inputs is not None:
            w = self.rank_k_proj(inputs)  # [BH, S, k*D]
            w = w.view(BH, S, self.rank_k, D)
            w_norms = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            w = w / w_norms
            w_flat = w.view(BH, S * self.rank_k, D)
            P = torch.bmm(w_flat.transpose(-1, -2), w_flat)  # [BH, D, D]
            # For the scan, we need per-token P — use the rank-k sum per token
            P_per_token = torch.bmm(
                w.view(BH * S, self.rank_k, D).transpose(-1, -2),
                w.view(BH * S, self.rank_k, D)
            ).view(BH, S, D, D) / self.rank_k
        else:
            # Rank-1 projectors: P_t = v_t ⊗ v_t^T
            P = v.unsqueeze(-1) * v.unsqueeze(-2)  # [BH, S, D, D]

        # --- Linear recurrence scan: H_t = α·H_{t-1} + η·P_t ---
        alpha = (1.0 - eta).view(BH, 1)  # [BH, 1]

        # Compute alpha powers: α^0, α^1, ..., α^(S-1) per batch element
        # alpha_powers[bh, t] = alpha[bh]^t
        alpha_2d = alpha.expand(BH, S)
        alpha_powers = torch.cumprod(alpha_2d, dim=1)  # [BH, S] — α^1, α^2, ..., α^S
        # Prepend α^0 = 1
        ones = torch.ones(BH, 1, device=hamiltonian.device, dtype=hamiltonian.dtype)
        alpha_powers = torch.cat([ones, alpha_powers[:, :-1]], dim=1)  # [BH, S] — α^0, α^1, ..., α^(S-1)

        # Weights for each token's contribution to the final H:
        # w_t = α^(S-1-t) · η
        eta_2d = eta.view(BH, 1)
        weights = alpha_powers.flip(1) * eta_2d  # [BH, S]
        weighted_P = weights.unsqueeze(-1).unsqueeze(-1) * P  # [BH, S, D, D]

        # Cumulative sum (parallel scan via cumsum)
        cumsum_P = torch.cumsum(weighted_P, dim=1)  # [BH, S, D, D]

        # Final H from scan: H_S = α^(S-1) · H_0 + Σ_t α^(S-1-t) · η · P_t
        alpha_final = alpha_powers[:, -1].view(BH, 1, 1)  # α^(S-1)
        H_scanned = alpha_final * hamiltonian + cumsum_P[:, -1, :, :]  # [BH, D, D]

        # --- Correction pass: re-apply destructive interference ---
        if correction_pass:
            if self.rank_k > 1 and inputs is not None:
                Hv = torch.bmm(hamiltonian, w_flat.transpose(-1, -2))  # [BH, D, S*k]
                Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), w_flat.transpose(-1, -2))
                HPH_sum = torch.bmm(Hv, Ht_v.transpose(-1, -2))  # [BH, D, D]
                H_scanned = H_scanned - eta_2d.view(BH, 1, 1) * HPH_sum / S
            else:
                Hv = torch.bmm(hamiltonian, v.transpose(-1, -2))  # [BH, D, S]
                Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v.transpose(-1, -2))
                HPH_sum = torch.bmm(Hv, Ht_v.transpose(-1, -2))  # [BH, D, D]
                H_scanned = H_scanned - eta_2d.view(BH, 1, 1) * HPH_sum / S

        # --- Spectral norm clamping ---
        fro_norm = H_scanned.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        max_norm = (D ** 0.5)
        scale = torch.clamp(max_norm / fro_norm, max=1.0)
        H_scanned = H_scanned * scale

        # --- Optional SVD truncation ---
        if self.bond_dimension is not None and self.bond_dimension < D:
            H_scanned = self._svd_truncate_batched(H_scanned)

        # --- Per-head memory gate ---
        gate_expanded = self.memory_gate.repeat(BH // self.n_heads)
        H_scanned = H_scanned * gate_expanded.view(BH, 1, 1)

        return H_scanned

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
            eta_expanded = eta.repeat(BH // self.n_heads)
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

        gate_expanded = self.memory_gate.repeat(BH // self.n_heads)
        hamiltonian_new = hamiltonian_new * gate_expanded.view(BH, 1, 1)

        return hamiltonian_new

    def update_single_served(
        self,
        hamiltonian: torch.Tensor,
        velocity: torch.Tensor,
        input_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Concurrent-serving single-token H update with batch-outer ordering (Issue #43).

        Numerically equivalent to :meth:`update_single` for a single stream, but
        expands the per-head ``eta``/``memory_gate`` **batch-outer**
        (``bh = b*n_heads + h``) via ``repeat`` (tile) rather than
        ``repeat_interleave`` (head-outer).  The head-outer expansion in
        :meth:`update_single` silently applies the wrong head's gate to each
        stream when ``n_heads>1, B>1``; this method keeps concurrent-batch output
        identical to single-stream.  The input-dependent-eta branch already
        indexes batch-outer (``arange(BH) % n_heads``) and is reused unchanged.

        All the heavy ops (``H·v``, ``P``, ``H·P·H``) are per-stream and stay
        batched — the Hamiltonian is block-diagonal across streams by construction.
        """
        BH, D, _ = hamiltonian.shape
        B = BH // self.n_heads

        if self.use_input_dependent_eta and input_vec is not None:
            eta_raw = self.eta_proj(input_vec)  # [BH, n_heads]
            head_indices = torch.arange(BH, device=hamiltonian.device) % self.n_heads
            eta_t = torch.sigmoid(eta_raw.gather(1, head_indices.unsqueeze(1))).view(BH, 1, 1)
        else:
            eta = torch.clamp(self.eta, min=0.0, max=1.0)
            eta_t = eta.repeat(B).view(BH, 1, 1)  # batch-outer tile

        v_norm = velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = velocity / v_norm  # [BH, D]

        if self.use_nonlinear_update and input_vec is not None:
            gate = torch.sigmoid(self.nonlinear_gate(input_vec))
            v = v * gate

        if self.rank_k > 1 and input_vec is not None:
            w = self.rank_k_proj(input_vec).view(BH, self.rank_k, D)
            w_norms = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            w = w / w_norms
            P = torch.bmm(w.transpose(-1, -2), w)
            Hv = torch.bmm(hamiltonian, w.transpose(-1, -2))
            Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), w.transpose(-1, -2))
            HPH = torch.bmm(Hv, Ht_v.transpose(-1, -2))
        else:
            v_col = v.unsqueeze(-1)
            P = v_col * v_col.transpose(-1, -2)
            Hv = torch.bmm(hamiltonian, v_col)
            Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v_col)
            HPH = Hv * Ht_v.transpose(-1, -2)

        if self.use_residual:
            hamiltonian_new = hamiltonian + eta_t * (P - HPH)
        else:
            hamiltonian_new = (1.0 - eta_t) * hamiltonian + eta_t * (P - HPH)

        fro_norm = hamiltonian_new.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        max_norm = (D ** 0.5)
        scale = torch.clamp(max_norm / fro_norm, max=1.0)
        hamiltonian_new = hamiltonian_new * scale

        if self.bond_dimension is not None and self.bond_dimension < D:
            hamiltonian_new = self._svd_truncate_batched(hamiltonian_new)

        gate_expanded = self.memory_gate.repeat(B)  # batch-outer tile
        hamiltonian_new = hamiltonian_new * gate_expanded.view(BH, 1, 1)

        return hamiltonian_new


class MultiThreadHamiltonianMemory(nn.Module):
    """Multi-Thread Hamiltonian Memory (Issue #22).

    Maintains **K independent Hamiltonians per head**, each updated with a
    different projection of the input.  Different threads learn to track
    different types of relationships (syntax, semantics, repetition, etc.).

    Architecture::

        Current:   1 head → 1 Hamiltonian (d×d)     → 1 thread of associations
        Proposed:  1 head → K Hamiltonians (d×d each) → K threads of associations

    Each thread gets its own:
      - Input projection (W_proj_k: d_head → d_head) — different view of the input
      - Hamiltonian matrix H_k (d_head × d_head) — independent memory
      - Velocity computation: v_k = Γ_k · (H_k · ω_k)

    The K thread outputs are combined via a learned mixing weight (like
    multi-head attention's output projection).

    When ``n_threads=1`` this module is functionally equivalent to
    :class:`MultiHeadHamiltonianMemory` (backward-compatible).

    Parameters
    ----------
    d_head : int
        Per-head dimensionality.
    n_heads : int
        Number of parallel heads.
    n_threads : int
        Number of independent Hamiltonian threads per head (K).
    entanglement_rate : float, optional
        Initial eta value (one per head).
    bond_dimension : int, optional
        SVD truncation rank. None = full rank.
    rank_k : int, optional
        Rank-k entanglement update dimension (Issue #21 Strategy 1).
    use_nonlinear_update : bool, optional
        Gated nonlinearity in the projector (Issue #21 Strategy 4).
    use_residual : bool, optional
        Residual Hamiltonian connection (Issue #13).
    use_input_dependent_eta : bool, optional
        Per-token entanglement rate (Issue #21 Strategy 5).
    """

    def __init__(
        self,
        d_head: int,
        n_heads: int,
        n_threads: int = 1,
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
        self.n_threads = n_threads
        self.bond_dimension = bond_dimension
        self.use_residual = use_residual
        self.use_input_dependent_eta = use_input_dependent_eta
        self.rank_k = rank_k
        self.use_nonlinear_update = use_nonlinear_update

        if entanglement_rate is None:
            entanglement_rate = 1.0 / math.sqrt(d_head)

        # Effective number of (head, thread) pairs — this is the batch dimension
        self.n_units = n_heads * n_threads

        # Per-unit entanglement rate (learnable)
        self.eta = nn.Parameter(torch.full((self.n_units,), entanglement_rate))

        # Per-unit memory gate (learnable)
        self.memory_gate = nn.Parameter(torch.ones(self.n_units))

        # Input-dependent eta projection: η_t = σ(W_η · x_t)
        if use_input_dependent_eta:
            self.eta_proj = nn.Linear(d_head, self.n_units, bias=True)
            nn.init.zeros_(self.eta_proj.weight)
            nn.init.zeros_(self.eta_proj.bias)

        # --- Per-thread input projections: W_proj_k · x_t → ω_k ---
        # Each thread gets a different learned projection from d_head → d_head.
        # Shape: [n_threads, d_head, d_head]
        if n_threads > 1:
            self.thread_proj = nn.Parameter(
                torch.empty(n_threads, d_head, d_head)
            )
            for k in range(n_threads):
                nn.init.xavier_uniform_(self.thread_proj[k])
        else:
            self.thread_proj = None

        # --- Per-thread coupling constant Γ_k ---
        self.gamma = nn.Parameter(
            torch.full((self.n_units, 1), 1.0 / math.sqrt(d_head))
        )

        # --- Thread mixing weights: combine K thread outputs per head ---
        # mix_weight[h, k] determines how much thread k contributes to head h's output.
        # Initialized as uniform (equal mixing) so all threads start with equal weight.
        # Shape: [n_heads, n_threads]
        if n_threads > 1:
            self.thread_mix = nn.Parameter(
                torch.ones(n_heads, n_threads) / n_threads
            )
        else:
            self.thread_mix = None

        # --- Rank-k projector (Issue #21 Strategy 1) ---
        if rank_k > 1:
            self.rank_k_proj = nn.Linear(d_head, rank_k * d_head, bias=False)
            nn.init.xavier_uniform_(self.rank_k_proj.weight)

        # --- Nonlinear gating (Issue #21 Strategy 4) ---
        if use_nonlinear_update:
            self.nonlinear_gate = nn.Linear(d_head, d_head, bias=True)
            nn.init.zeros_(self.nonlinear_gate.weight)
            nn.init.zeros_(self.nonlinear_gate.bias)

    def init_hamiltonian(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Initialize Hamiltonian matrices for all (head, thread) pairs.

        Returns
        -------
        H : torch.Tensor, shape [batch * n_heads * n_threads, d_head, d_head]
            One Hamiltonian per (batch element, head, thread) triple.
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        BU = batch_size * self.n_units  # batch * heads * threads
        H = torch.eye(self.d_head, device=device, dtype=dtype)
        H = H.unsqueeze(0).expand(BU, -1, -1).contiguous()
        H = H / self.d_head
        return H

    def project_inputs(self, x: torch.Tensor) -> torch.Tensor:
        """Project inputs through per-thread projections.

        Parameters
        ----------
        x : torch.Tensor, shape [B*n_heads, S, d_head]
            Input embeddings (already split per head).

        Returns
        -------
        x_threaded : torch.Tensor, shape [B*n_heads*n_threads, S, d_head]
            Each thread's projected view of the input.
        """
        BH, S, D = x.shape
        B = BH // self.n_heads

        if self.thread_proj is not None:
            # thread_proj: [n_threads, D, D]
            # x: [BH, S, D] → expand for threads: [BH, n_threads, S, D]
            # Apply per-thread projection: [BH, n_threads, S, D]
            # Use einsum: (bh, k, s, d) = (bh, s, d) @ (k, d, d)^T
            x_expanded = x.unsqueeze(1).expand(BH, self.n_threads, S, D)  # [BH, T, S, D]
            x_threaded = torch.einsum(
                "btsd,kde->btse", x_expanded, self.thread_proj
            )  # [BH, T, S, D]
        else:
            # n_threads=1: just repeat the input
            x_threaded = x.unsqueeze(1).expand(BH, 1, S, D).contiguous()

        # Reshape to [BH*T, S, D]
        x_threaded = x_threaded.reshape(BH * self.n_threads, S, D)
        return x_threaded

    def mix_thread_velocities(self, velocities: torch.Tensor, B: int) -> torch.Tensor:
        """Mix K thread velocities into per-head velocities via learned weights.

        Parameters
        ----------
        velocities : torch.Tensor, shape [B*n_heads*n_threads, S, d_head]
            Velocity vectors from all threads.
        B : int
            Batch size.

        Returns
        -------
        mixed : torch.Tensor, shape [B*n_heads, S, d_head]
            Thread-mixed velocity vectors.
        """
        if self.n_threads == 1:
            return velocities

        BH = B * self.n_heads
        _, S, D = velocities.shape

        # Reshape: [B*n_heads*n_threads, S, D] → [B, n_heads, n_threads, S, D]
        v = velocities.view(B, self.n_heads, self.n_threads, S, D)

        # Apply learned mixing: mix_weight[h, k] → weighted sum over threads
        # thread_mix: [n_heads, n_threads]
        # mixed[b, h, s, d] = Σ_k thread_mix[h, k] * v[b, h, k, s, d]
        mixed = torch.einsum("hk,bhksd->bhsd", self.thread_mix, v)

        # Reshape to [B*n_heads, S, D]
        return mixed.reshape(BH, S, D)

    def update_batch(
        self,
        hamiltonian: torch.Tensor,
        velocities: torch.Tensor,
        inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Batched Hamiltonian update for all (head, thread) pairs simultaneously.

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Per-unit Hamiltonian matrices, shape [B*n_units, d_head, d_head].
        velocities : torch.Tensor
            Velocity vectors, shape [B*n_units, seq_len, d_head].
        inputs : torch.Tensor, optional
            Input embeddings, shape [B*n_units, seq_len, d_head].

        Returns
        -------
        hamiltonian_new : torch.Tensor, shape [B*n_units, d_head, d_head]
        """
        BU, S, D = velocities.shape

        # --- Compute eta (entanglement rate) ---
        if self.use_input_dependent_eta and inputs is not None:
            eta_raw = self.eta_proj(inputs)  # [BU, S, n_units]
            eta_avg = torch.sigmoid(eta_raw).mean(dim=1)  # [BU, n_units]
            unit_indices = torch.arange(BU, device=inputs.device) % self.n_units
            eta_b = eta_avg.gather(1, unit_indices.unsqueeze(1)).view(BU, 1, 1)
        else:
            eta = torch.clamp(self.eta, min=0.0, max=1.0)
            eta_expanded = eta.repeat(BU // self.n_units)
            eta_b = eta_expanded.view(BU, 1, 1)

        # --- Normalise all velocity vectors ---
        v_norms = velocities.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = velocities / v_norms

        # --- Strategy 4: Nonlinear gating ---
        if self.use_nonlinear_update and inputs is not None:
            gate = torch.sigmoid(self.nonlinear_gate(inputs))
            v = v * gate

        # --- Strategy 1: Rank-k projector ---
        if self.rank_k > 1 and inputs is not None:
            w = self.rank_k_proj(inputs)
            w = w.view(BU, S, self.rank_k, D)
            w_norms = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            w = w / w_norms
            w_flat = w.view(BU, S * self.rank_k, D)
            P_sum = torch.bmm(w_flat.transpose(-1, -2), w_flat)
            P_sum = P_sum / S
            Hv = torch.bmm(hamiltonian, w_flat.transpose(-1, -2))
            Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), w_flat.transpose(-1, -2))
            HPH_sum = torch.bmm(Hv, Ht_v.transpose(-1, -2))
            HPH_sum = HPH_sum / S
        else:
            P_sum = torch.bmm(v.transpose(-1, -2), v)
            Hv = torch.bmm(hamiltonian, v.transpose(-1, -2))
            Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v.transpose(-1, -2))
            HPH_sum = torch.bmm(Hv, Ht_v.transpose(-1, -2))

        # --- Entanglement update ---
        if self.use_residual:
            delta_H = eta_b * (P_sum - HPH_sum)
            hamiltonian_new = hamiltonian + delta_H
        else:
            hamiltonian_new = (1.0 - eta_b) * hamiltonian + eta_b * (P_sum - HPH_sum)

        # --- Spectral norm clamping ---
        fro_norm = hamiltonian_new.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        max_norm = D ** 0.5
        scale = torch.clamp(max_norm / fro_norm, max=1.0)
        hamiltonian_new = hamiltonian_new * scale

        # --- Optional: low-rank truncation ---
        if self.bond_dimension is not None and self.bond_dimension < D:
            hamiltonian_new = self._svd_truncate_batched(hamiltonian_new)

        # --- Per-unit memory gate ---
        gate_expanded = self.memory_gate.repeat(BU // self.n_units)
        hamiltonian_new = hamiltonian_new * gate_expanded.view(BU, 1, 1)

        return hamiltonian_new

    def update_batch_per_token(
        self,
        hamiltonian: torch.Tensor,
        velocities: torch.Tensor,
        inputs: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Per-token sequential Hamiltonian update with per-token η (Strategy 5).

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Per-unit Hamiltonian matrices, shape [B*n_units, d_head, d_head].
        velocities : torch.Tensor
            Velocity vectors, shape [B*n_units, seq_len, d_head].
        inputs : torch.Tensor, optional
            Input embeddings, shape [B*n_units, seq_len, d_head].
        """
        BU, S, D = velocities.shape
        H = hamiltonian

        if self.use_input_dependent_eta and inputs is not None:
            eta_raw = self.eta_proj(inputs)
            eta_all = torch.sigmoid(eta_raw)
            unit_indices = torch.arange(BU, device=inputs.device) % self.n_units
            idx = unit_indices.view(BU, 1, 1).expand(BU, S, 1)
            eta_per_token = eta_all.gather(2, idx).squeeze(-1)
        else:
            eta = torch.clamp(self.eta, min=0.0, max=1.0)
            eta_expanded = eta.repeat(BU // self.n_units)
            eta_per_token = eta_expanded.view(BU, 1).expand(BU, S)

        for t in range(S):
            v_t = velocities[:, t, :]
            x_t = inputs[:, t, :] if inputs is not None else None
            eta_t = eta_per_token[:, t].view(BU, 1, 1)

            v_norm = v_t.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            v = v_t / v_norm

            if self.use_nonlinear_update and x_t is not None:
                gate = torch.sigmoid(self.nonlinear_gate(x_t))
                v = v * gate

            if self.rank_k > 1 and x_t is not None:
                w = self.rank_k_proj(x_t).view(BU, self.rank_k, D)
                w_norms = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
                w = w / w_norms
                P = torch.bmm(w.transpose(-1, -2), w)
                Hv = torch.bmm(H, w.transpose(-1, -2))
                Ht_v = torch.bmm(H.transpose(-1, -2), w.transpose(-1, -2))
                HPH = torch.bmm(Hv, Ht_v.transpose(-1, -2))
            else:
                v_col = v.unsqueeze(-1)
                P = v_col * v_col.transpose(-1, -2)
                Hv = torch.bmm(H, v_col)
                Ht_v = torch.bmm(H.transpose(-1, -2), v_col)
                HPH = Hv * Ht_v.transpose(-1, -2)

            if self.use_residual:
                H = H + eta_t * (P - HPH)
            else:
                H = (1.0 - eta_t) * H + eta_t * (P - HPH)

            fro_norm = H.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
            max_norm = D ** 0.5
            scale = torch.clamp(max_norm / fro_norm, max=1.0)
            H = H * scale

        if self.bond_dimension is not None and self.bond_dimension < D:
            H = self._svd_truncate_batched(H)

        gate_expanded = self.memory_gate.repeat(BU // self.n_units)
        H = H * gate_expanded.view(BU, 1, 1)

        return H

    def _svd_truncate_batched(self, H: torch.Tensor) -> torch.Tensor:
        """Batched low-rank renormalization via randomized power iteration."""
        BU, D, _ = H.shape
        r = min(self.bond_dimension, D)
        n_iter = 3
        orig_dtype = H.dtype
        work_dtype = (
            torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype
        )

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=False):
                H_f32 = H.to(work_dtype)
                Omega = torch.randn(BU, D, r, device=H.device, dtype=work_dtype)
                Omega = Omega / math.sqrt(r)
                G = torch.bmm(H_f32, H_f32.transpose(-1, -2))
                G = G + 1e-5 * torch.eye(D, device=H.device, dtype=work_dtype).unsqueeze(0)
                Q = Omega
                for _ in range(n_iter):
                    Q = torch.bmm(G, Q)
                    Q, _ = torch.linalg.qr(Q)

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
        """Update Hamiltonian with a single velocity vector (incremental generation).

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Shape [B*n_units, d_head, d_head].
        velocity : torch.Tensor
            Shape [B*n_units, d_head].
        input_vec : torch.Tensor, optional
            Input embedding, shape [B*n_units, d_head].
        """
        BU, D, _ = hamiltonian.shape

        if self.use_input_dependent_eta and input_vec is not None:
            eta_raw = self.eta_proj(input_vec)
            unit_indices = torch.arange(BU, device=hamiltonian.device) % self.n_units
            eta_t = torch.sigmoid(
                eta_raw.gather(1, unit_indices.unsqueeze(1))
            ).view(BU, 1, 1)
        else:
            eta = torch.clamp(self.eta, min=0.0, max=1.0)
            eta_expanded = eta.repeat(BU // self.n_units)
            eta_t = eta_expanded.view(BU, 1, 1)

        v_norm = velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        v = velocity / v_norm

        if self.use_nonlinear_update and input_vec is not None:
            gate = torch.sigmoid(self.nonlinear_gate(input_vec))
            v = v * gate

        if self.rank_k > 1 and input_vec is not None:
            w = self.rank_k_proj(input_vec).view(BU, self.rank_k, D)
            w_norms = w.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            w = w / w_norms
            P = torch.bmm(w.transpose(-1, -2), w)
            Hv = torch.bmm(hamiltonian, w.transpose(-1, -2))
            Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), w.transpose(-1, -2))
            HPH = torch.bmm(Hv, Ht_v.transpose(-1, -2))
        else:
            v_col = v.unsqueeze(-1)
            P = v_col * v_col.transpose(-1, -2)
            Hv = torch.bmm(hamiltonian, v_col)
            Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v_col)
            HPH = Hv * Ht_v.transpose(-1, -2)

        if self.use_residual:
            delta_H = eta_t * (P - HPH)
            hamiltonian_new = hamiltonian + delta_H
        else:
            hamiltonian_new = (1.0 - eta_t) * hamiltonian + eta_t * (P - HPH)

        fro_norm = hamiltonian_new.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        max_norm = D ** 0.5
        scale = torch.clamp(max_norm / fro_norm, max=1.0)
        hamiltonian_new = hamiltonian_new * scale

        if self.bond_dimension is not None and self.bond_dimension < D:
            hamiltonian_new = self._svd_truncate_batched(hamiltonian_new)

        gate_expanded = self.memory_gate.repeat(BU // self.n_units)
        hamiltonian_new = hamiltonian_new * gate_expanded.view(BU, 1, 1)

        return hamiltonian_new