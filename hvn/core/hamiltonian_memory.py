"""
HamiltonianMemoryBlock
======================

A fixed-size Matrix Product State (Tensor Train) memory that updates its internal
entanglement state deterministically on a single forward pass — no backpropagation
or optimisation steps during inference.

Theoretical basis (see docs/THEORETICAL_FRAMEWORK.md §3):

    The Hamiltonian H ∈ ℝ^{d_model × d_model} is a *compressed summary* of all
    past tokens' vorticity interactions.  It evolves via **Quantum Entanglement
    Renormalization**:

        H_new = (1 - η) · H_old + η · (P_t - H_old · P_t · H_old)

    where P_t = v_t ⊗ v_t^T is the rank-1 interaction projector.

    - The P_t term provides **associative alignment** (reinforces new associations).
    - The H·P_t·H term provides **destructive interference** on noise / repeated
      patterns (cancels components already aligned with v_t).
    - The (1-η) blend preserves long-range memory.

    After each update, an optional **truncated SVD renormalization** keeps only
    the top-r singular values, discarding high-energy noise modes — the quantum
    renormalization step.

    All operations are O(d_model²) (or O(d_model · χ) in compressed MPS form),
    independent of sequence length N.  This is the key to O(1) per-token inference.
"""

import torch
import torch.nn as nn
import math
from typing import Optional


class HamiltonianMemoryBlock(nn.Module):
    """Fixed-size entanglement Hamiltonian with deterministic MPS-style updates.

    Parameters
    ----------
    d_model : int
        Dimensionality of the model.
    entanglement_rate : float
        Learning-free update rate η (default 1/√d_model).  Controls how strongly
        each new vortex vector alters the Hamiltonian.
    bond_dimension : int, optional
        Maximum rank r for SVD truncation (entanglement bond dimension).
        If None, no truncation is applied (full-rank updates).
        If set, after each token the Hamiltonian is decomposed via SVD and
        reconstructed from the top-r singular values — this is the quantum
        renormalization step that discards noise modes.
    """

    def __init__(
        self,
        d_model: int,
        entanglement_rate: Optional[float] = None,
        bond_dimension: Optional[int] = None,
    ):
        super().__init__()
        self.d_model = d_model

        # Entanglement rate η — can be learned or fixed.
        # Default: 1/√d_model (gentle updates that preserve long-range memory).
        if entanglement_rate is None:
            entanglement_rate = 1.0 / math.sqrt(d_model)
        self.eta = nn.Parameter(torch.tensor(entanglement_rate))

        # Entanglement bond dimension for SVD truncation.
        # If None, no truncation (full-rank).  If set, keep top-r singular values.
        self.bond_dimension = bond_dimension

        # Learnable scalar gate to control the strength of the Hamiltonian's
        # contribution (helps with training stability).
        self.memory_gate = nn.Parameter(torch.tensor(1.0))

    def init_hamiltonian(self, batch_size: int, device=None, dtype=None) -> torch.Tensor:
        """Initialise a fresh Hamiltonian matrix for a given batch.

        The Hamiltonian is initialised to a scaled identity — representing a
        "ground state" with no entanglement yet.

        Returns
        -------
        H : torch.Tensor, shape [batch, d_model, d_model]
        """
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype

        # Start from identity scaled by 1/d_model — a minimal-entanglement state.
        H = torch.eye(self.d_model, device=device, dtype=dtype)
        H = H.unsqueeze(0).expand(batch_size, -1, -1).contiguous()
        H = H / self.d_model
        return H

    def update(
        self,
        hamiltonian: torch.Tensor,
        velocity: torch.Tensor,
    ) -> torch.Tensor:
        """Update the Hamiltonian with a single velocity (vortex) vector.

        Implements the entanglement renormalization rule:

            H_new = (1 - η) · H_old + η · (P_t - H_old · P_t · H_old)

        where P_t = v_t ⊗ v_t^T.

        To prevent spectral norm explosion over long sequences, the velocity
        vector is L2-normalised and the Hamiltonian is spectrally clamped
        after each update.

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Current Hamiltonian, shape [batch, d_model, d_model].
        velocity : torch.Tensor
            Velocity vector for a single token, shape [batch, d_model].

        Returns
        -------
        hamiltonian_new : torch.Tensor, shape [batch, d_model, d_model]
        """
        B, D, _ = hamiltonian.shape
        eta = torch.clamp(self.eta, min=0.0, max=1.0)  # safety clamp

        # --- Normalise the velocity vector to prevent unbounded growth ---
        # ||v_t|| is normalised to 1 so that P_t = v_t ⊗ v_t^T has spectral norm 1.
        v_norm = velocity.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
        v = (velocity / v_norm).unsqueeze(-1)  # [B, D, 1]

        # --- Rank-1 interaction projector: P_t = v_t ⊗ v_t^T ---
        P = v * v.transpose(-1, -2)  # [B, D, D]  (outer product, spectral norm = 1)

        # --- Destructive interference term: H · P · H ---
        # Exploit rank-1 structure: H·P = (Hv)⊗v^T, then (Hv)⊗(v^T H) = (Hv)⊗(H^T v)^T
        # This is cheaper and more numerically stable than two full matmuls.
        Hv = torch.bmm(hamiltonian, v)             # [B, D, 1]  (H · v)
        Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v)  # [B, D, 1]  (H^T · v)
        HPH = Hv * Ht_v.transpose(-1, -2)          # [B, D, D]  (outer product)

        # --- Entanglement renormalization update ---
        # H_new = (1 - η)·H + η·(P - H·P·H)
        hamiltonian_new = (1.0 - eta) * hamiltonian + eta * (P - HPH)

        # --- Spectral norm clamping: prevent the Hamiltonian from exploding ---
        # Clamp the Frobenius norm to a maximum of sqrt(d_model).
        # This is a cheap proxy for spectral norm control.
        fro_norm = hamiltonian_new.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)  # [B, 1, 1]
        max_norm = (D ** 0.5)
        scale = torch.clamp(max_norm / fro_norm, max=1.0)  # only shrink, never grow
        hamiltonian_new = hamiltonian_new * scale

        # --- Optional: low-rank truncation (quantum renormalization) ---
        if self.bond_dimension is not None and self.bond_dimension < D:
            hamiltonian_new = self._svd_truncate(hamiltonian_new)

        # Apply learnable memory gate (controls Hamiltonian's influence strength)
        hamiltonian_new = hamiltonian_new * self.memory_gate

        return hamiltonian_new

    def _svd_truncate(self, H: torch.Tensor) -> torch.Tensor:
        """Low-rank renormalization via randomized power iteration.

        Replaces the expensive O(D³) eigh decomposition with a GPU-friendly
        randomized subspace iteration that costs only O(D² · r · n_iter).

        Algorithm:
            1. Draw a random Gaussian probe matrix Ω ∈ ℝ^{D×r}
            2. Power-iterate: Q = orth((H·H^T)^n_iter · Ω)  → top-r subspace
            3. Project: H_trunc = Q · (Q^T · H · Q) · Q^T

        This is fully differentiable (no eigh NaN issues), runs entirely on
        GPU with batched matmuls, and converges to the true top-r subspace
        in a few iterations.

        Parameters
        ----------
        H : torch.Tensor, shape [batch, d_model, d_model]

        Returns
        -------
        H_truncated : torch.Tensor, shape [batch, d_model, d_model]
        """
        B, D, _ = H.shape
        r = min(self.bond_dimension, D)
        n_iter = 3  # power iterations (3 is enough for good approximation)
        orig_dtype = H.dtype
        # torch.linalg.qr doesn't support fp16/bf16 — upcast for the no-grad section
        work_dtype = torch.float32 if orig_dtype in (torch.float16, torch.bfloat16) else orig_dtype

        # --- Randomized subspace iteration ---
        # Use a fixed random seed per call for reproducibility within a step
        # but different across batch elements
        with torch.no_grad():
            # Disable autocast to ensure QR runs in float32
            with torch.amp.autocast("cuda", enabled=False):
                H_f32 = H.to(work_dtype)
                # Draw random probe: [B, D, r]
                Omega = torch.randn(B, D, r, device=H.device, dtype=work_dtype)
                Omega = Omega / math.sqrt(r)

                # Gram matrix G = H · H^T  (we need the top-r eigenvectors of G)
                G = torch.bmm(H_f32, H_f32.transpose(-1, -2))  # [B, D, D]
                G = G + 1e-5 * torch.eye(D, device=H.device, dtype=work_dtype).unsqueeze(0)

                # Power iteration: Q ≈ top-r eigenvectors of G
                Q = Omega
                for _ in range(n_iter):
                    Q = torch.bmm(G, Q)  # [B, D, r]
                    # QR orthonormalize for numerical stability
                    Q, _ = torch.linalg.qr(Q)  # Q: [B, D, r]

        # --- Project H into the low-rank subspace ---
        # Gradients flow through H (Q is detached)
        # H_trunc = Q · (Q^T · H · Q) · Q^T
        with torch.amp.autocast("cuda", enabled=False):
            H_work = H.to(work_dtype)
            Qt_H = torch.bmm(Q.transpose(-1, -2), H_work)  # [B, r, D]
            Qt_H_Q = torch.bmm(Qt_H, Q)                 # [B, r, r]
            H_trunc = torch.bmm(Q, torch.bmm(Qt_H_Q, Q.transpose(-1, -2)))  # [B, D, D]

        return H_trunc.to(orig_dtype)

    def update_batch(
        self,
        hamiltonian: torch.Tensor,
        velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Batched Hamiltonian update for all tokens in a chunk simultaneously.

        Instead of looping token-by-token, this processes all tokens in a
        single batched operation using the linearised approximation:

            H_new ≈ (1 - η)·H + η · Σ_t P_t

        The destructive interference term (H·P·H) is computed once using the
        chunk-initial H, which is the same approximation used in the chunked
        forward pass (all tokens in a chunk see the same H).

        This eliminates the Python for-loop bottleneck and replaces ~10·S
        sequential kernel launches with a handful of batched matmuls.

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Current Hamiltonian, shape [batch, d_model, d_model].
        velocities : torch.Tensor
            Velocity vectors for all tokens in the chunk, shape [batch, seq_len, d_model].

        Returns
        -------
        hamiltonian_new : torch.Tensor, shape [batch, d_model, d_model]
        """
        B, S, D = velocities.shape
        eta = torch.clamp(self.eta, min=0.0, max=1.0)

        # --- Normalise all velocity vectors ---
        v_norms = velocities.norm(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, S, 1]
        v = velocities / v_norms  # [B, S, D]

        # --- Batched rank-1 projectors: P_t = v_t ⊗ v_t^T ---
        # Sum all projectors in one operation: Σ_t P_t = v^T · v (batched)
        # P_sum[b] = sum_t v[b,t,:] ⊗ v[b,t,:]^T = v[b].T @ v[b]  → [B, D, D]
        P_sum = torch.bmm(v.transpose(-1, -2), v)  # [B, D, D]

        # --- Destructive interference: H · P_sum · H (computed once) ---
        # Using the chunk-initial H for all tokens (same approximation as chunked forward)
        Hv = torch.bmm(hamiltonian, v.transpose(-1, -2))  # [B, D, S]  (H · v^T)
        # HPH_sum = Σ_t (Hv_t ⊗ (H^T v_t)^T) = (H v^T) (H v^T)^T ... but with H^T
        Ht_v = torch.bmm(hamiltonian.transpose(-1, -2), v.transpose(-1, -2))  # [B, D, S]
        HPH_sum = torch.bmm(Hv, Ht_v.transpose(-1, -2))  # [B, D, D]

        # --- Batched entanglement update ---
        # H_new = (1 - η)·H + η · (P_sum - HPH_sum) / S
        # Divide by S to normalise (average projector contribution)
        hamiltonian_new = (1.0 - eta) * hamiltonian + eta * (P_sum - HPH_sum) / S

        # --- Spectral norm clamping ---
        fro_norm = hamiltonian_new.norm(dim=(-1, -2), keepdim=True).clamp(min=1e-8)
        max_norm = (D ** 0.5)
        scale = torch.clamp(max_norm / fro_norm, max=1.0)
        hamiltonian_new = hamiltonian_new * scale

        # --- Optional: low-rank truncation ---
        if self.bond_dimension is not None and self.bond_dimension < D:
            hamiltonian_new = self._svd_truncate(hamiltonian_new)

        # Apply learnable memory gate
        hamiltonian_new = hamiltonian_new * self.memory_gate

        return hamiltonian_new

    def forward(
        self,
        hamiltonian: torch.Tensor,
        velocities: torch.Tensor,
    ) -> torch.Tensor:
        """Sequentially update the Hamiltonian over a sequence of velocity vectors.

        This processes the sequence token-by-token, updating H at each step.
        Because H is fixed-size, the cost is O(S · D²) — linear in sequence
        length but **constant per token**.

        Parameters
        ----------
        hamiltonian : torch.Tensor
            Initial Hamiltonian, shape [batch, d_model, d_model].
        velocities : torch.Tensor
            Velocity vectors for all tokens, shape [batch, seq_len, d_model].

        Returns
        -------
        hamiltonian_final : torch.Tensor, shape [batch, d_model, d_model]
            The Hamiltonian after absorbing all velocity vectors.
        """
        B, S, D = velocities.shape

        H = hamiltonian
        for t in range(S):
            H = self.update(H, velocities[:, t, :])  # [B, D, D]

        return H