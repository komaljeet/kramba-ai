"""
TransformerBlock — causal multi-head attention baseline (A/B issue #73)
======================================================================

A matched **transformer baseline** for the EdgeHVN-vs-Transformer controlled
A/B. It is deliberately *structurally identical* to :class:`DeltaNetBlock` —
same :class:`RMSNorm`, same parallel-residual wiring
``x = x + dropout(mix(norm(x))) + ffn(norm(x))``, and the same GELU FFN style —
with the **only** difference being the token mixer: standard causal multi-head
softmax attention (+ RoPE) in place of DeltaNet's gated delta-rule recurrence.

Isolating that single variable is the whole point of the A/B: any quality
difference between the two runs is then attributable to the mixing mechanism,
not to incidental scaffolding differences.

Parameter matching
------------------
DeltaNet's mixer has 6 projections (q,k,v,α,gate,out ≈ 6·d²); MHA has 4
(q,k,v,out = 4·d²). To keep the *total* block parameter count equal, the FFN
expansion defaults to 3 (→ 6·d² in the FFN), giving 10·d²/block — the same as
DeltaNet's 6·d² mixer + 4·d² (2×) FFN. At d=512/12L/8H this matches the HVN's
83M to within ~0.3%.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hvn_block import RMSNorm  # identical norm to DeltaNetBlock — a control


def _rope_cos_sin(seq_len: int, d_head: int, device, dtype, base: float = 10000.0):
    """Rotary positional embedding tables. Returns (cos, sin), each [S, d_head]."""
    half = d_head // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(seq_len, device=device).float()
    freqs = torch.outer(t, inv_freq)            # [S, half]
    emb = torch.cat([freqs, freqs], dim=-1)     # [S, d_head]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def _apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply RoPE to x [B, H, S, d_head]; cos/sin broadcast as [S, d_head]."""
    half = x.shape[-1] // 2
    x1, x2 = x[..., :half], x[..., half:]
    rot = torch.cat([-x2, x1], dim=-1)
    return x * cos + rot * sin


class TransformerBlock(nn.Module):
    """Causal MHA + RoPE block with DeltaNetBlock's scaffold (A/B baseline).

    Parameters mirror DeltaNetBlock's public interface so it is a drop-in swap
    inside :class:`EdgeHVN`. Extra DeltaNet-only kwargs are accepted and ignored.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
        ffn_expansion: int = 3,
        rope_base: float = 10000.0,
        **_unused,  # swallow DeltaNet/HVN-only kwargs (conv_kernel, bond_dimension, …)
    ):
        super().__init__()
        assert d_model % n_heads == 0, (
            f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        )
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rope_base = rope_base

        # Pre-norm (parallel-residual, matches DeltaNetBlock)
        self.norm = RMSNorm(d_model)

        # Standard multi-head projections (bias-free, matching DeltaNet q/k/v/out)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.xavier_uniform_(self.out_proj.weight)

        # FFN branch — same GELU style as DeltaNetBlock; expansion 3 for param parity
        hidden = d_model * ffn_expansion
        self.ffn = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, d_model),
            nn.Dropout(dropout),
        )

        self.dropout = nn.Dropout(dropout)

    def _attention(self, x_norm: torch.Tensor) -> torch.Tensor:
        """Causal multi-head softmax attention with RoPE. x_norm: [B, S, D]."""
        B, S, D = x_norm.shape
        H, d = self.n_heads, self.d_head
        q = self.q_proj(x_norm).view(B, S, H, d).transpose(1, 2)  # [B, H, S, d]
        k = self.k_proj(x_norm).view(B, S, H, d).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, S, H, d).transpose(1, 2)

        cos, sin = _rope_cos_sin(S, d, x_norm.device, q.dtype, self.rope_base)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        # Fused causal scaled-dot-product attention (flash when available)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, S, d]
        o = o.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(o)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None,
                return_state: bool = False):
        """Process a sequence. x: [B, S, D] → [B, S, D].

        Parallel-residual, identical to DeltaNetBlock:
        ``x = x + dropout(attn(norm(x))) + ffn(norm(x))``.
        """
        x_norm = self.norm(x)
        attn_out = self._attention(x_norm)
        ffn_out = self.ffn(x_norm)
        out = x + self.dropout(attn_out) + ffn_out
        if return_state:
            return out, None  # transformer baseline keeps no recurrent state
        return out
