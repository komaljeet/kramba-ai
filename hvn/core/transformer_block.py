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


def _rope_cos_sin_at(position_idx: int, d_head: int, device, dtype,
                      base: float = 10000.0):
    """RoPE cos/sin for a single absolute position (incremental decode).

    Same formula as :func:`_rope_cos_sin` but for one explicit position
    rather than a 0..S-1 range — used by ``forward_step`` where the token
    being processed sits at an arbitrary offset into an ongoing conversation,
    not at the start of a fresh sequence. Returns (cos, sin), each [1, d_head].
    """
    half = d_head // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.tensor([float(position_idx)], device=device)
    freqs = torch.outer(t, inv_freq)          # [1, half]
    emb = torch.cat([freqs, freqs], dim=-1)   # [1, d_head]
    return emb.cos().to(dtype), emb.sin().to(dtype)


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
        window: int | None = None,
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
        # Sliding-window width (hybrid token-recall). None = full causal (the A/B
        # baseline + the training path). When set and the sequence is longer than
        # `window`, attention is masked to the last `window` keys per query — the
        # bounded-KV-cache design (see EdgeHVN block_plan). At training seq_len=512
        # (< window 4096) this path is NOT taken; is_causal=True stays the fast path.
        self.window = window

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

        if self.window is not None and S > self.window:
            # Sliding-window causal attention: query i attends to keys in
            # [i-window+1, i] (no future). This is the bounded-KV-cache design —
            # only taken when the sequence exceeds the window (inference/gate).
            # At training seq_len=512 (< W=4096) the else branch runs, so the
            # flash is_causal=True fast path is unchanged → no training regression.
            pos = torch.arange(S, device=x_norm.device)
            dist = pos[:, None] - pos[None, :]              # [S, S], dist[i,j]=i-j
            mask = torch.zeros(S, S, device=x_norm.device, dtype=q.dtype)
            mask = mask.masked_fill(dist < 0, float("-inf"))            # future keys
            mask = mask.masked_fill(dist >= self.window, float("-inf"))  # beyond window
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        else:
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

    # ------------------------------------------------------------------
    # Incremental decode (issue #97 Phase B) — bounded KV cache
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int, device=None, dtype=None) -> dict:
        """Empty KV cache for incremental decode. Matches DeltaNetBlock.init_state's
        signature so EdgeHVN can dispatch either block type identically."""
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        empty = torch.zeros(batch_size, self.n_heads, 0, self.d_head,
                            device=device, dtype=dtype)
        return {"k_cache": empty, "v_cache": empty.clone()}

    def forward_step(self, x_t: torch.Tensor, state: dict,
                     position_idx: int = 0) -> tuple[torch.Tensor, dict]:
        """Process one token incrementally. x_t: [B, 1, D] -> ([B, 1, D], new_state).

        Appends this token's K/V to the cache (truncating to the last
        ``window`` entries — the bounded, VRAM-flat KV cache the hybrid design
        promises), then attends over the cache. No mask is needed: the cache
        only ever holds this token and earlier ones, so it's causal by
        construction. State is a plain {str: Tensor} dict (no custom classes)
        so it round-trips through hvn/chat.py's generic state-tree
        encode/decode unmodified.
        """
        B, S, D = x_t.shape
        assert S == 1, f"forward_step expects one token at a time, got S={S}"
        H, d = self.n_heads, self.d_head

        x_norm = self.norm(x_t)
        q = self.q_proj(x_norm).view(B, 1, H, d).transpose(1, 2)  # [B,H,1,d]
        k = self.k_proj(x_norm).view(B, 1, H, d).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, 1, H, d).transpose(1, 2)

        cos, sin = _rope_cos_sin_at(position_idx, d, x_norm.device, q.dtype,
                                    self.rope_base)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        k_cache = torch.cat([state["k_cache"], k], dim=2)  # [B,H,L+1,d]
        v_cache = torch.cat([state["v_cache"], v], dim=2)
        if self.window is not None and k_cache.shape[2] > self.window:
            k_cache = k_cache[:, :, -self.window:, :]
            v_cache = v_cache[:, :, -self.window:, :]

        o = F.scaled_dot_product_attention(q, k_cache, v_cache, is_causal=False)
        o = o.transpose(1, 2).reshape(B, 1, D)
        attn_out = self.out_proj(o)

        ffn_out = self.ffn(x_norm)
        out = x_t + self.dropout(attn_out) + ffn_out
        return out, {"k_cache": k_cache, "v_cache": v_cache}
