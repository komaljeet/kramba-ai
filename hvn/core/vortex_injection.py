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
        r̃_t = x_t + g_t ⊙ (Γ · v_t)

    where g_t is a learned gate (sigmoid of a linear projection of x_t and v_t).
    This gating allows the model to selectively decide how much the vortex
    velocity field should deflect each token — analogous to the output gate in
    an LSTM.

This layer is purely feed-forward (no recurrence, no attention).  It receives the
current Hamiltonian matrix H from the memory block and returns both the deflected
representation and the raw velocity vector (which the memory block needs for its
entanglement update).
"""

import torch
import torch.nn as nn
import math


def _apply_rotary_embeddings(
    x: torch.Tensor,
    position_ids: torch.Tensor,
    frequency_scale: torch.Tensor | float = 1.0,
) -> torch.Tensor:
    """Apply rotary positional embeddings to the last dimension of x.

    The implementation rotates pairs of channels using sinusoidal frequencies
    based on absolute positions. This gives the vortex projection explicit
    sequence-order information before the Biot-Savart velocity computation.

    Parameters
    ----------
    frequency_scale : torch.Tensor | float
        Per-head frequency scaling factor.  A value > 1 increases the rotary
        frequency (captures local / high-frequency patterns), while a value
        < 1 decreases it (captures global / low-frequency patterns).
        Shape: scalar or [B*n_heads, 1] for per-head frequencies.
    """
    if x.shape[-1] < 2:
        return x

    rotary_dim = x.shape[-1] - (x.shape[-1] % 2)
    if rotary_dim < 2:
        return x

    device = x.device
    dtype = x.dtype
    positions = position_ids.to(device=device, dtype=dtype)
    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, rotary_dim, 2, device=device, dtype=dtype) / rotary_dim))
    # Apply per-head frequency scaling — different heads see different frequencies
    if isinstance(frequency_scale, torch.Tensor):
        # frequency_scale: [BH] → reshape to [BH, 1, 1] for broadcasting
        freqs = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1) * frequency_scale.view(-1, 1, 1)
    else:
        freqs = positions.unsqueeze(-1) * inv_freq.view(1, 1, -1) * frequency_scale

    cos = torch.cos(freqs)
    sin = torch.sin(freqs)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    rotated_even = x_even * cos - x_odd * sin
    rotated_odd = x_even * sin + x_odd * cos

    rotated = torch.empty_like(x)
    rotated[..., 0::2] = rotated_even
    rotated[..., 1::2] = rotated_odd

    if x.shape[-1] % 2 == 1:
        rotated[..., -1] = x[..., -1]

    return rotated


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
        self.vorticity_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.vorticity_proj.weight, mean=0.0, std=1.0 / math.sqrt(d_model))

        # Coupling constant Γ (learnable scalar, initialized to 1/√d_model)
        self.gamma = nn.Parameter(torch.tensor(1.0 / math.sqrt(d_model)))

        # Gating: g_t = σ(W_g · [x_t ; v_t]) — controls how much the vortex
        # velocity deflects the token.  This is analogous to an LSTM output gate
        # and lets the model selectively attend to or ignore the velocity field.
        self.gate_proj = nn.Linear(d_model * 2, d_model, bias=True)
        nn.init.xavier_uniform_(self.gate_proj.weight)
        nn.init.zeros_(self.gate_proj.bias)

        # Output projection back to d_model
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.output_proj.weight)

        # LayerNorm for stabilising the deflected representation
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        hamiltonian: torch.Tensor,
        position_ids: torch.Tensor | None = None,
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

        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(B, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, -1)
        else:
            position_ids = position_ids.to(device=x.device)

        # --- Step 1: Vorticity projection ---
        vorticity = self.vorticity_proj(x)  # [B, S, D]
        vorticity = _apply_rotary_embeddings(vorticity, position_ids)

        # --- Step 2: Biot-Savart local velocity ---
        # v_t = Γ · (H · ω_t)
        velocity = torch.einsum("bsd,bde->bse", vorticity, hamiltonian)
        velocity = self.gamma * velocity  # [B, S, D]

        # --- Step 3: Gated deflection ---
        # g_t = σ(W_g · [x_t ; v_t]) — gate decides how much velocity deflects
        gate_input = torch.cat([x, velocity], dim=-1)  # [B, S, 2D]
        gate = torch.sigmoid(self.gate_proj(gate_input))  # [B, S, D]

        # r̃_t = x_t + g_t ⊙ (Γ · v_t)
        deflected_raw = x + gate * velocity  # [B, S, D]
        deflected = self.output_proj(deflected_raw)  # [B, S, D]
        deflected = self.layer_norm(deflected)  # stabilise

        return deflected, velocity