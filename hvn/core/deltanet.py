"""
DeltaNetBlock — Readout/Update Dynamics Fix (Issue #34)
=======================================================

A drop-in alternative to :class:`HVNBlock` that replaces HVN's crippled
readout/update dynamics with **proven linear-attention fast-weight dynamics**
(DeltaNet / GLA / Gated-DeltaNet family).  It is fully self-contained and is
only used when ``use_deltanet=True`` is passed to
:class:`~hvn.core.model.HamiltonianVortexNetwork` — the original HVN path is
untouched and remains the default, so nothing that already worked regresses.

Why this exists
---------------
HVN's PPL gap (≈8.1 vs SSM ≈1.1) was long blamed on a "fixed d×d state
compression bottleneck".  The project's own benchmark disproves that: the SSM
reaches PPL ≈1.1 with a *smaller* fixed state.  The real culprits are the
**readout/update dynamics** (see ``related.md`` #34):

  1. The velocity L2-normalization + Frobenius clamp (``‖H‖_F ≤ √d``) — a NaN
     band-aid that throws away all magnitude information the readout depends on.
  2. The non-standard, ill-conditioned symmetric ``H·P·H`` update that couples
     the read vector and the write projector (both use ``v``).

Stripped of the fluid/quantum framing, HVN is a linear-attention fast-weight
net (``v = Γ·(H·ω)`` read, ``H ← (1-η)H + η(P − HPH)`` delta-style update), so
we borrow the proven fixes from that literature.

The three levers (highest impact first)
---------------------------------------
1. **Per-channel data-dependent decay gate** ``α_t ∈ (0,1)^d`` (GLA / Mamba-2
   style) replaces the global Frobenius clamp, plus a **per-head output
   RMSNorm normalizer** so the read is scale-normalized (a proper weighted
   average) rather than an unnormalized matmul.
2. **Canonical (gated) DeltaNet update** with **separate learned key ``k_t`` and
   value ``v_t`` projections** and a **separate query ``q_t``** for the read —
   decoupling read from write and dropping the symmetric ``HPH`` term::

       S_t = diag_k(α_t) · S_{t-1} + β_t · (v_t − S_{t-1}k_t) k_tᵀ
       o_t = S_t q_t

   ``k_t``/``q_t`` are L2-normalized (delta-rule stability), ``β_t = σ(W_β x)``
   is the write strength.
3. **Output gate** ``g_t = SiLU(W_g x_t)`` on the read plus a **short causal
   depthwise conv** on the input path (Mamba / RWKV-7 both do this).

O(1) inference is preserved: the recurrent state is a fixed ``d_head × d_head``
matrix per head (constant in sequence length N), updated in O(d_head²) per
token, with no KV cache.  ``forward`` (training) and ``forward_step``
(incremental) share the *identical* recurrence, so trained and generated
behaviour match.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hvn_block import RMSNorm, SwiGLU


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize over the last dimension."""
    return x / x.norm(dim=-1, keepdim=True).clamp_min(eps)


class DeltaNetBlock(nn.Module):
    """A single Gated-DeltaNet block with the same public interface as HVNBlock.

    Parameters
    ----------
    d_model : int
        Model dimensionality.
    n_heads : int
        Number of parallel heads (each operates on ``d_model // n_heads`` dims).
    dropout : float
        Dropout on the sub-layer output.
    conv_kernel : int
        Kernel size of the short causal depthwise conv (lever 3).  ``0``/``1``
        disables it.  Default 4 (Mamba-style).
    use_output_gate : bool
        Enable the SiLU output gate on the read (lever 3).  Default True.
    use_decay_gate : bool
        Enable the per-channel data-dependent decay gate (lever 1).  Default
        True.  When False the state uses no decay (pure DeltaNet).
    use_swiglu : bool
        Use a SwiGLU FFN instead of the standard GELU FFN.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        conv_kernel: int = 4,
        use_output_gate: bool = True,
        use_decay_gate: bool = True,
        use_swiglu: bool = False,
        # Accepted for interface compatibility with HVNBlock construction
        # (ignored — DeltaNet defines its own dynamics):
        bond_dimension: int | None = None,
        chunk_size: int | None = None,
        **_unused,
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.conv_kernel = int(conv_kernel) if conv_kernel and conv_kernel > 1 else 0
        self.use_output_gate = use_output_gate
        self.use_decay_gate = use_decay_gate

        # Pre-norm (parallel-residual, matches HVNBlock)
        self.norm = RMSNorm(d_model)

        # Short causal depthwise conv on the (normed) input path (lever 3)
        if self.conv_kernel:
            self.conv = nn.Conv1d(
                d_model, d_model, kernel_size=self.conv_kernel,
                groups=d_model, bias=True, padding=0,
            )

        # Separate q / k / v projections (lever 2)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)

        # Write strength β_t = σ(W_β x)  (per head)
        self.beta_proj = nn.Linear(d_model, n_heads, bias=True)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)  # β ≈ 0.5 initially

        # Per-channel data-dependent decay gate α_t = σ(W_α x) (lever 1)
        if use_decay_gate:
            self.alpha_proj = nn.Linear(d_model, d_model, bias=True)
            nn.init.zeros_(self.alpha_proj.weight)
            # Bias so α ≈ σ(3.0) ≈ 0.95 initially — retain memory, learnable.
            nn.init.constant_(self.alpha_proj.bias, 3.0)

        # Output gate g_t = SiLU(W_g x) (lever 3)
        if use_output_gate:
            self.gate_proj = nn.Linear(d_model, d_model, bias=True)

        # Per-head output RMSNorm normalizer (lever 1): scale-normalizes the
        # read so it behaves like a proper weighted average, replacing the
        # magnitude-destroying Frobenius clamp.
        self.read_norm_weight = nn.Parameter(torch.ones(n_heads, self.d_head))
        self.read_norm_eps = 1e-6

        # Output projection back to d_model
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # FFN branch (parallel residual)
        if use_swiglu:
            self.ffn = SwiGLU(d_model, expansion=3, dropout=dropout)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_model * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model * 2, d_model),
                nn.Dropout(dropout),
            )

        self.dropout = nn.Dropout(dropout)

    # ------------------------------------------------------------------
    # Shared projection / gating helpers
    # ------------------------------------------------------------------
    def _causal_conv(self, x: torch.Tensor) -> torch.Tensor:
        """Causal depthwise conv over a full sequence.  x: [B, S, D] → [B, S, D]."""
        if not self.conv_kernel:
            return x
        xt = x.transpose(1, 2)  # [B, D, S]
        xt = F.pad(xt, (self.conv_kernel - 1, 0))  # left-pad → causal
        y = self.conv(xt)  # [B, D, S]
        return y.transpose(1, 2)

    def _project(self, xc: torch.Tensor):
        """Compute q, k, v, α, β, g from the conv'd, normed input.

        Parameters
        ----------
        xc : [B, S, D]

        Returns
        -------
        q, k, v : [B, S, H, d]   (q, k are L2-normalized over d)
        alpha   : [B, S, H, d]   in (0,1)   (or None if decay disabled)
        beta    : [B, S, H, 1]   in (0,1)
        gate    : [B, S, D]      SiLU gate  (or None if disabled)
        """
        B, S, _ = xc.shape
        H, d = self.n_heads, self.d_head

        q = _l2norm(self.q_proj(xc).view(B, S, H, d))
        k = _l2norm(self.k_proj(xc).view(B, S, H, d))
        v = self.v_proj(xc).view(B, S, H, d)

        beta = torch.sigmoid(self.beta_proj(xc)).view(B, S, H, 1)

        if self.use_decay_gate:
            alpha = torch.sigmoid(self.alpha_proj(xc)).view(B, S, H, d)
        else:
            alpha = None

        gate = None
        if self.use_output_gate:
            gate = F.silu(self.gate_proj(xc))  # [B, S, D]

        return q, k, v, alpha, beta, gate

    def _read_normalize(self, o: torch.Tensor) -> torch.Tensor:
        """Per-head RMSNorm on the read output.  o: [B, S, H, d] → [B, S, H, d]."""
        rms = o.pow(2).mean(dim=-1, keepdim=True).add(self.read_norm_eps).rsqrt()
        return o * rms * self.read_norm_weight  # weight broadcasts [H, d]

    # ------------------------------------------------------------------
    # Training forward (sequential delta-rule scan, O(1) memory in N)
    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Process a sequence.  x: [B, S, D] → [B, S, D].

        Parallel-residual: ``x = x + deltanet(norm(x)) + ffn(norm(x))``.
        """
        B, S, D = x.shape
        H, d = self.n_heads, self.d_head
        N = B * H

        x_norm = self.norm(x)
        xc = self._causal_conv(x_norm)
        q, k, v, alpha, beta, gate = self._project(xc)

        # Move to [N, S, d] for the batched-over-heads recurrence
        def to_N(t):  # [B, S, H, *] → [N, S, *]
            return t.permute(0, 2, 1, 3).reshape(N, S, -1)

        qN = to_N(q)          # [N, S, d]
        kN = to_N(k)          # [N, S, d]
        vN = to_N(v)          # [N, S, d]
        betaN = to_N(beta)    # [N, S, 1]
        alphaN = to_N(alpha) if alpha is not None else None  # [N, S, d]

        S_state = torch.zeros(N, d, d, device=x.device, dtype=x.dtype)  # [d_v, d_k]
        outs = []
        for t in range(S):
            k_t = kN[:, t]           # [N, d]  (key space)
            if alphaN is not None:
                a_t = alphaN[:, t]   # [N, d]  per key-channel decay
                S_state = S_state * a_t.unsqueeze(1)   # decay columns (key dim)
            # prediction error: u = v - S k
            Sk = (S_state * k_t.unsqueeze(1)).sum(-1)          # [N, d_v]
            u = vN[:, t] - Sk                                  # [N, d_v]
            # delta write: S += β · u kᵀ
            S_state = S_state + betaN[:, t].unsqueeze(-1) * (
                u.unsqueeze(2) * k_t.unsqueeze(1)
            )
            # read with the query: o = S q
            o_t = (S_state * qN[:, t].unsqueeze(1)).sum(-1)    # [N, d_v]
            outs.append(o_t)

        o = torch.stack(outs, dim=1)  # [N, S, d]
        o = o.view(B, H, S, d).permute(0, 2, 1, 3)  # [B, S, H, d]
        o = self._read_normalize(o)                 # normalizer (lever 1)
        o = o.reshape(B, S, D)

        if gate is not None:
            o = o * gate                            # output gate (lever 3)

        deflected = self.out_proj(o)
        ffn_out = self.ffn(x_norm)
        return x + self.dropout(deflected) + ffn_out

    # ------------------------------------------------------------------
    # Incremental generation — O(1) per token, no KV cache
    # ------------------------------------------------------------------
    def init_state(self, batch_size: int, device=None, dtype=None):
        """Initialize the recurrent state for incremental generation."""
        if device is None:
            device = next(self.parameters()).device
        if dtype is None:
            dtype = next(self.parameters()).dtype
        d = self.d_head
        state = {
            "S": torch.zeros(batch_size * self.n_heads, d, d, device=device, dtype=dtype),
        }
        if self.conv_kernel:
            state["conv"] = torch.zeros(
                batch_size, self.d_model, self.conv_kernel - 1, device=device, dtype=dtype
            )
        return state

    # Batched multi-sequence path is identical (already fully batched over B).
    def init_state_batched(self, batch_size: int, device=None, dtype=None):
        return self.init_state(batch_size, device=device, dtype=dtype)

    def _conv_step(self, x_t: torch.Tensor, buf: torch.Tensor | None):
        """Single-token causal conv.  x_t: [B, 1, D] → ([B, 1, D], new_buf)."""
        if not self.conv_kernel:
            return x_t, None
        xt = x_t.transpose(1, 2)  # [B, D, 1]
        window = torch.cat([buf, xt], dim=2)  # [B, D, K]
        y = self.conv(window)  # [B, D, 1]
        new_buf = window[:, :, 1:]  # [B, D, K-1]
        return y.transpose(1, 2), new_buf

    def forward_step(self, x_t: torch.Tensor, state, position_idx: int | None = None):
        """Process a single token.  x_t: [B, 1, D] → ([B, 1, D], new_state)."""
        B, _, D = x_t.shape
        H, d = self.n_heads, self.d_head
        N = B * H

        x_norm = self.norm(x_t)  # [B, 1, D]
        xc, new_conv = self._conv_step(x_norm, state.get("conv") if self.conv_kernel else None)
        q, k, v, alpha, beta, gate = self._project(xc)  # [B, 1, H, d] etc.

        def to_N(t):
            return t.permute(0, 2, 1, 3).reshape(N, -1)  # [N, *] (S==1 collapsed)

        q_t = to_N(q)      # [N, d]
        k_t = to_N(k)      # [N, d]
        v_t = to_N(v)      # [N, d]
        beta_t = to_N(beta)  # [N, 1]

        S_state = state["S"]  # [N, d, d]
        if alpha is not None:
            a_t = to_N(alpha)  # [N, d]
            S_state = S_state * a_t.unsqueeze(1)
        Sk = (S_state * k_t.unsqueeze(1)).sum(-1)
        u = v_t - Sk
        S_state = S_state + beta_t.unsqueeze(-1) * (u.unsqueeze(2) * k_t.unsqueeze(1))
        o_t = (S_state * q_t.unsqueeze(1)).sum(-1)  # [N, d]

        o = o_t.view(B, H, 1, d).permute(0, 2, 1, 3)  # [B, 1, H, d]
        o = self._read_normalize(o).reshape(B, 1, D)
        if gate is not None:
            o = o * gate

        deflected = self.out_proj(o)
        x_out = x_t + self.dropout(deflected) + self.ffn(x_norm)

        new_state = {"S": S_state}
        if self.conv_kernel:
            new_state["conv"] = new_conv
        return x_out, new_state

    def forward_step_batched(self, x_t: torch.Tensor, state, position_idx: int | None = None):
        """Batched multi-sequence single-token step (identical recurrence)."""
        return self.forward_step(x_t, state, position_idx=position_idx)
