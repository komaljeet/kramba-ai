"""
VortexInjectionLayer
=====================

Translates incoming dense token vectors into a discrete velocity field using a
modified Biot-Savart approximation.

Theoretical basis (see docs/THEORETICAL_FRAMEWORK.md §2):

    Each token embedding x_t is treated as a Lagrangian vortex particle with:
      - position  r_t = x_t
      - vorticity ω_t = W_ω · x_t        (learned projection)
      - velocity  v_t = Γ · (H · ω_t)    (Biot-Savart local approximation)

    The velocity field H is supplied by the HamiltonianMemoryBlock and is a
    fixed-size summary of all previously absorbed vorticity.  Because H has
    shape [d_model × d_model], the velocity computation is a single matrix-vector
    product — O(d_model²) regardless of sequence length N.

    The deflected (interacted) representation is:
        r̃_t = x_t + Γ · v_t

This layer is purely feed-forward (no recurrence, no attention).  It receives the
current Hamiltonian matrix H from the memory block and returns both the deflected
representation and the raw velocity vector (which the memory block needs for its
entanglement update).
"""

import torch
import torch.nn as nn
import math


class VortexInjectionLayer(nn.Module):
    """Translates token embeddings into vortex-particle velocity fields.

    Parameters
    ----------
    d_model : int
        Dimensionality of the model (embedding / hidden size).
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

        # Learned vorticity projection: ω_t = W_ω · x_t
        # Initialized with small values so initial vorticity is gentle.
        self.vorticity_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.vorticity_proj.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        # Coupling constant Γ (learnable scalar, initialized to 1/√d_model)
        self.gamma = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_model)))

        # Output projection back to d_model
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.output_proj.weight)

        # LayerNorm for stabilising the deflected representation
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        hamiltonian: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vortex-velocity-deflected token representations.

        Parameters
        ----------
        x : torch.Tensor
            Input token embeddings, shape [batch, seq_len, d_model].
        hamiltonian : torch.Tensor
            Current Hamiltonian matrix H, shape [batch, d_model, d_model].
            This is the fixed-size "velocity field" summarising all past
            vorticity interactions.

        Returns
        -------
        deflected : torch.Tensor
            Deflected (interacted) representations, shape [batch, seq_len, d_model].
        velocity : torch.Tensor
            Raw velocity vectors v_t, shape [batch, seq_len, d_model].
            These are needed by the HamiltonianMemoryBlock for the entanglement
            update.
        """
        B, S, D = x.shape

        # --- Step 1: Vorticity projection ---
        # ω_t = W_ω · x_t   →  [B, S, D]
        vorticity = self.vorticity_proj(x)

        # --- Step 2: Biot-Savart local velocity ---
        # v_t = Γ · (H · ω_t)
        # H: [B, D, D],  ω_t: [B, S, D] → need [B, S, D]
        # We compute bmm(H, ω_t^T) → [B, D, S] → transpose → [B, S, D]
        # Equivalently: einsum('bsd,bde->bse', ω_t, H)  (note H acts on the left)
        #
        # v_t[b, s, e] = Γ * Σ_d H[b, e, d] * ω_t[b, s, d]
        velocity = torch.einsum("bsd,bde->bse", vorticity, hamiltonian)
        velocity = self.gamma * velocity  # [B, S, D]

        # --- Step 3: Deflection ---
        # r̃_t = x_t + Γ · v_t
        # We apply the deflection and then project back.
        deflected_raw = x + velocity  # [B, S, D]
        deflected = self.output_proj(deflected_raw)  # [B, S, D]
        deflected = self.layer_norm(deflected)  # stabilise

        return deflected, velocity