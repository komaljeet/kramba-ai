"""
Batched Multi-Head Vortex Injection
====================================

A fully batched version of VortexInjectionLayer that processes ALL heads
simultaneously by stacking per-head weights into batched tensors.

Instead of looping over n_heads (each with its own nn.Linear), this module
uses a single batched matmul (torch.bmm) to compute all heads in parallel.

Shapes use BH = B * n_heads as the batch dimension, so all heads are
processed as one large batch.
"""

import torch
import torch.nn as nn
import math

from .vortex_injection import _apply_rotary_embeddings


class MultiHeadVortexInjection(nn.Module):
    """Batched multi-head vortex injection layer.

    Combines n_heads VortexInjectionLayer instances into a single module
    that processes all heads in parallel using batched operations.

    Parameters
    ----------
    d_head : int
        Per-head dimensionality.
    n_heads : int
        Number of parallel heads.
    """

    def __init__(self, d_head: int, n_heads: int):
        super().__init__()
        self.d_head = d_head
        self.n_heads = n_heads

        # --- Batched vorticity projection: ω_t = W_ω · x_t ---
        # Weight shape: [n_heads, d_head, d_head] — one matrix per head
        self.vorticity_weight = nn.Parameter(
            torch.empty(n_heads, d_head, d_head)
        )
        for h in range(n_heads):
            nn.init.normal_(self.vorticity_weight[h], mean=0.0, std=1.0 / math.sqrt(d_head))

        # --- Batched coupling constant Γ (one per head) ---
        self.gamma = nn.Parameter(torch.full((n_heads, 1), 1.0 / math.sqrt(d_head)))

        # --- Batched gate projection: g_t = σ(W_g · [x_t ; v_t]) ---
        # Weight: [n_heads, d_head, 2*d_head], Bias: [n_heads, d_head]
        self.gate_weight = nn.Parameter(torch.empty(n_heads, d_head, 2 * d_head))
        self.gate_bias = nn.Parameter(torch.zeros(n_heads, d_head))
        for h in range(n_heads):
            nn.init.xavier_uniform_(self.gate_weight[h])

        # --- Batched output projection ---
        self.output_weight = nn.Parameter(torch.empty(n_heads, d_head, d_head))
        for h in range(n_heads):
            nn.init.xavier_uniform_(self.output_weight[h])

        # --- Batched LayerNorm (one per head, but LayerNorm doesn't batch easily) ---
        # We use a single GroupNorm-like approach: normalize over last dim per head
        self.ln_weight = nn.Parameter(torch.ones(n_heads, d_head))
        self.ln_bias = nn.Parameter(torch.zeros(n_heads, d_head))

    def forward(
        self,
        x: torch.Tensor,
        hamiltonian: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute vortex-velocity-deflected representations for all heads.

        Parameters
        ----------
        x : torch.Tensor
            Input, shape [B*n_heads, S, d_head].
        hamiltonian : torch.Tensor
            Per-head Hamiltonian matrices, shape [B*n_heads, d_head, d_head].
        position_ids : torch.Tensor, optional
            Position IDs, shape [B*n_heads, S] or [B, S].

        Returns
        -------
        deflected : torch.Tensor
            Deflected representations, shape [B*n_heads, S, d_head].
        velocity : torch.Tensor
            Raw velocity vectors, shape [B*n_heads, S, d_head].
        """
        BH, S, D = x.shape  # BH = B * n_heads

        if position_ids is None:
            position_ids = torch.arange(S, device=x.device).unsqueeze(0).expand(BH, -1)
        elif position_ids.ndim == 2 and position_ids.shape[0] == 1:
            position_ids = position_ids.expand(BH, -1)
        elif position_ids.ndim == 1:
            position_ids = position_ids.unsqueeze(0).expand(BH, -1)
        else:
            position_ids = position_ids.to(device=x.device)

        # --- Step 1: Batched vorticity projection ---
        # ω_t = W_ω · x_t  for all heads at once
        # vorticity_weight: [n_heads, D, D] → repeat for each batch element: [BH, D, D]
        # Batch ordering: [B0H0, B0H1, ..., B0H(n-1), B1H0, ...]
        B = BH // self.n_heads
        w_vort = self.vorticity_weight.repeat_interleave(B, dim=0)  # [BH, D, D]
        vorticity = torch.bmm(x, w_vort.transpose(-1, -2))  # [BH, S, D]
        # Apply rotary embeddings
        vorticity = _apply_rotary_embeddings(vorticity, position_ids)

        # --- Step 2: Batched Biot-Savart velocity ---
        # v_t = Γ · (H · ω_t)
        velocity = torch.bmm(vorticity, hamiltonian.transpose(-1, -2))  # [BH, S, D]
        gamma_expanded = self.gamma.repeat_interleave(B, dim=0)  # [BH, 1]
        velocity = gamma_expanded.view(BH, 1, 1) * velocity  # [BH, S, D]

        # --- Step 3: Batched gated deflection ---
        # g_t = σ(W_g · [x_t ; v_t])
        gate_input = torch.cat([x, velocity], dim=-1)  # [BH, S, 2D]
        w_gate = self.gate_weight.repeat_interleave(B, dim=0)  # [BH, D, 2D]
        b_gate = self.gate_bias.repeat_interleave(B, dim=0)  # [BH, D]
        gate = torch.bmm(gate_input, w_gate.transpose(-1, -2))  # [BH, S, D]
        gate = gate + b_gate.view(BH, 1, D)
        gate = torch.sigmoid(gate)  # [BH, S, D]

        # r̃_t = x_t + g_t ⊙ v_t
        deflected_raw = x + gate * velocity  # [BH, S, D]

        # Batched output projection
        w_out = self.output_weight.repeat_interleave(B, dim=0)  # [BH, D, D]
        deflected = torch.bmm(deflected_raw, w_out.transpose(-1, -2))  # [BH, S, D]

        # Batched LayerNorm: normalize over last dim
        mean = deflected.mean(dim=-1, keepdim=True)
        var = deflected.var(dim=-1, keepdim=True, unbiased=False)
        deflected = (deflected - mean) / torch.sqrt(var + 1e-5)
        ln_w = self.ln_weight.repeat_interleave(B, dim=0)  # [BH, D]
        ln_b = self.ln_bias.repeat_interleave(B, dim=0)  # [BH, D]
        deflected = deflected * ln_w.view(BH, 1, D) + ln_b.view(BH, 1, D)

        return deflected, velocity