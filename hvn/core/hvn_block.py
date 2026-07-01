"""
HVNBlock
========

The wrapper that links the fluid vortex step (VortexInjectionLayer) to the
quantum state evolution (HamiltonianMemoryBlock).

For each token in the sequence:
  1. VortexInjectionLayer computes the velocity field v_t = Γ · (H · ω_t) and
     produces the deflected representation r̃_t.
  2. HamiltonianMemoryBlock absorbs v_t into H via entanglement renormalization.

The key insight: the Hamiltonian H is updated *sequentially* token-by-token,
but each update is O(d_model²) — constant in N.  The velocity computation is
also O(d_model²) per token.  Therefore the entire block is O(S · d_model²) for
a sequence of length S, i.e. **O(d_model²) per token = O(1) in N**.

To produce per-token outputs (not just a final summary), HVNBlock maintains a
running Hamiltonian and applies the vortex injection at each step, collecting
outputs for every position.
"""

import torch
import torch.nn as nn
from .vortex_injection import VortexInjectionLayer
from .hamiltonian_memory import HamiltonianMemoryBlock


class HVNBlock(nn.Module):
    """A single Hamiltonian Vortex Network block.

    Combines fluid-dynamical vortex injection with quantum entanglement memory
    evolution into a single residual block.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    bond_dimension : int, optional
        SVD truncation rank for quantum renormalization.  None = full rank.
    dropout : float
        Dropout rate applied to the block output.
    """

    def __init__(
        self,
        d_model: int,
        bond_dimension: int | None = None,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model

        # Fluid dynamics component
        self.vortex = VortexInjectionLayer(d_model)

        # Quantum memory component
        self.memory = HamiltonianMemoryBlock(d_model, bond_dimension=bond_dimension)

        # Feed-forward sub-layer (lightweight, 2x expansion)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process a sequence through the HVN block.

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

        # --- Initialise the Hamiltonian (ground state) ---
        H = self.memory.init_hamiltonian(B, device=x.device, dtype=x.dtype)  # [B, D, D]

        # --- Pre-norm: normalise input before vortex injection ---
        x_norm = self.norm1(x)  # [B, S, D]

        # --- Sequential vortex injection + Hamiltonian evolution ---
        # We process token-by-token so that H evolves causally:
        # token t's velocity depends on H updated through tokens 0..t-1.
        outputs = []
        velocities_all = []

        for t in range(S):
            # Current token
            x_t = x_norm[:, t:t + 1, :]  # [B, 1, D]

            # Vortex injection: compute velocity from current H and deflect
            deflected_t, velocity_t = self.vortex(x_t, H)  # [B,1,D], [B,1,D]
            outputs.append(deflected_t.squeeze(1))  # [B, D]
            velocities_all.append(velocity_t.squeeze(1))  # [B, D]

            # Update Hamiltonian with this token's velocity (causal)
            H = self.memory.update(H, velocity_t.squeeze(1))  # [B, D, D]

        # Stack outputs: [B, S, D]
        deflected = torch.stack(outputs, dim=1)  # [B, S, D]

        # --- Residual connection ---
        x = x + self.dropout(deflected)

        # --- Feed-forward sub-layer with residual ---
        x = x + self.ffn(self.norm2(x))

        return x

    def init_state(self, batch_size: int, device=None, dtype=None) -> list[torch.Tensor]:
        """Initialize the causal Hamiltonian state for incremental generation.

        Returns a list of Hamiltonian matrices (one per block is handled by
        the model, but each block has one H).  This enables O(1) per-token
        generation without reprocessing the full context.

        Returns
        -------
        state : list[torch.Tensor]
            A list containing one Hamiltonian matrix [B, D, D].
        """
        H = self.memory.init_hamiltonian(batch_size, device=device, dtype=dtype)
        return [H]

    def forward_step(
        self,
        x_t: torch.Tensor,
        state: list[torch.Tensor],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Process a SINGLE token incrementally — O(1) per token generation.

        This is the key optimization for fast autoregressive generation.
        Instead of reprocessing the entire context, we maintain the Hamiltonian
        state and only process the new token.

        Parameters
        ----------
        x_t : torch.Tensor
            Single token embedding, shape [batch, 1, d_model].
        state : list[torch.Tensor]
            Hamiltonian state from previous tokens: [H] where H is [B, D, D].

        Returns
        -------
        output : torch.Tensor
            Output for this token, shape [batch, 1, d_model].
        new_state : list[torch.Tensor]
            Updated Hamiltonian state.
        """
        B, _, D = x_t.shape
        H = state[0]  # [B, D, D]

        # Pre-norm
        x_t_norm = self.norm1(x_t)  # [B, 1, D]

        # Vortex injection with current Hamiltonian
        deflected_t, velocity_t = self.vortex(x_t_norm, H)  # [B,1,D], [B,1,D]

        # Update Hamiltonian with this token's velocity
        H_new = self.memory.update(H, velocity_t.squeeze(1))  # [B, D, D]

        # Residual
        x_t = x_t + self.dropout(deflected_t)

        # FFN sub-layer
        x_t = x_t + self.ffn(self.norm2(x_t))

        return x_t, [H_new]