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

    def _attention(self, x_norm: torch.Tensor,
                   doc_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Causal multi-head softmax attention with RoPE. x_norm: [B, S, D].

        ``doc_ids`` ([B, S], Issue #132) restricts attention to within-document
        keys for packed rows, by AND-ing a block-diagonal document mask into the
        attention mask (together with causality and, when set, the sliding
        window). This forces the explicit-mask path -- the fused
        ``is_causal=True`` flash kernel cannot express a per-batch segment mask
        -- which is why packing costs a little on the attention layers. Only the
        hybrid's 2 attention layers pay it; the other 10 are DeltaNet.

        RoPE needs no adjustment: it is relative, so a document starting at
        offset 300 still sees correct relative distances within itself, and
        cross-document keys are masked out here anyway.
        """
        B, S, D = x_norm.shape
        H, d = self.n_heads, self.d_head
        q = self.q_proj(x_norm).view(B, S, H, d).transpose(1, 2)  # [B, H, S, d]
        k = self.k_proj(x_norm).view(B, S, H, d).transpose(1, 2)
        v = self.v_proj(x_norm).view(B, S, H, d).transpose(1, 2)

        cos, sin = _rope_cos_sin(S, d, x_norm.device, q.dtype, self.rope_base)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)

        windowed = self.window is not None and S > self.window
        if windowed or doc_ids is not None:
            # Explicit-mask path. Taken when attention is sliding-window (the
            # bounded-KV-cache design, sequence longer than `window`) and/or the
            # row is packed (document isolation). At training seq_len=512
            # (< W=4096) with no packing, neither applies and the
            # is_causal=True flash fast path below is unchanged → no regression
            # for existing runs.
            pos = torch.arange(S, device=x_norm.device)
            dist = pos[:, None] - pos[None, :]              # [S, S], dist[i,j]=i-j
            mask = torch.zeros(S, S, device=x_norm.device, dtype=q.dtype)
            mask = mask.masked_fill(dist < 0, float("-inf"))            # future keys
            if windowed:
                mask = mask.masked_fill(dist >= self.window, float("-inf"))
            if doc_ids is not None:
                from .packing import segment_attn_mask
                same = segment_attn_mask(doc_ids)           # [B, 1, S, S] bool
                # Broadcast the [S, S] causal/window mask against the per-batch
                # document mask → [B, 1, S, S].
                mask = mask.view(1, 1, S, S).expand(B, 1, S, S).clone()
                mask = mask.masked_fill(~same, float("-inf"))
            o = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, is_causal=False)
        else:
            # Fused causal scaled-dot-product attention (flash when available)
            o = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # [B, H, S, d]
        o = o.transpose(1, 2).reshape(B, S, D)
        return self.out_proj(o)

    def forward(self, x: torch.Tensor, position_ids: torch.Tensor | None = None,
                return_state: bool = False,
                doc_ids: torch.Tensor | None = None):
        """Process a sequence. x: [B, S, D] → [B, S, D].

        Parallel-residual, identical to DeltaNetBlock:
        ``x = x + dropout(attn(norm(x))) + ffn(norm(x))``.

        ``doc_ids`` ([B, S], Issue #132) isolates packed documents — see
        :meth:`_attention`.
        """
        x_norm = self.norm(x)
        attn_out = self._attention(x_norm, doc_ids=doc_ids)
        ffn_out = self.ffn(x_norm)
        out = x + self.dropout(attn_out) + ffn_out
        if return_state:
            return out, None  # transformer baseline keeps no recurrent state
        return out

    # ------------------------------------------------------------------
    # Incremental decode (issue #97 Phase B) — bounded KV cache
    # ------------------------------------------------------------------

    def init_state(self, batch_size: int, device=None, dtype=None) -> dict:
        """KV cache for incremental decode. Matches DeltaNetBlock.init_state's
        signature so EdgeHVN can dispatch either block type identically.

        Issue #138: when ``window`` is set, this pre-allocates the FULL
        ``[B, H, window, d]`` buffer once and ``forward_step`` writes into it in
        place, instead of the previous ``torch.cat``-and-slice which allocated a
        fresh cache every token. At window 4096 that ``cat`` copied ~4 MB per
        tensor per layer per token in order to append ~4 KB — and, because every
        token produced new tensors at new addresses with a growing shape, it made
        CUDA-graph capture of the decode step impossible.

        The pre-allocation is also what makes the "VRAM-flat" claim literally
        true: peak cache memory is now the same on token 1 as on token 100,000,
        rather than climbing to it.

        ``written`` is a plain Python int; ``hvn/chat.py``'s state-tree codec
        passes non-tensor scalars through unchanged, so this still round-trips.
        """
        device = device if device is not None else next(self.parameters()).device
        dtype = dtype if dtype is not None else next(self.parameters()).dtype
        if self.window is None:
            # Unbounded attention keeps the growing cache: there is no fixed size
            # to pre-allocate, and this path is not the one serving takes.
            empty = torch.zeros(batch_size, self.n_heads, 0, self.d_head,
                                device=device, dtype=dtype)
            return {"k_cache": empty, "v_cache": empty.clone()}
        shape = (batch_size, self.n_heads, self.window, self.d_head)
        return {
            "k_cache": torch.zeros(shape, device=device, dtype=dtype),
            "v_cache": torch.zeros(shape, device=device, dtype=dtype),
            "written": 0,
        }

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

        if "written" in state:
            # --- Issue #138: fixed-size ring buffer, written in place ---------
            # No allocation and no copy of the existing cache per token. Order
            # inside the ring does not matter: RoPE is already baked into the
            # cached keys, and attention is a softmax over a SET of positions,
            # so a rotated ring attends to exactly the same tokens.
            #
            # It is mathematically equal but NOT bit-identical once wrapped: the
            # rotation changes the order of the attention reduction, and float
            # addition is not associative. Measured ~2.4e-07, against the ~3e-6
            # machine-precision bar #97 set for this path. Before the first wrap
            # the order is unchanged and the output IS bit-identical; both are
            # pinned in tests/test_kv_ring_buffer.py.
            k_cache, v_cache = state["k_cache"], state["v_cache"]
            written = state["written"]
            W = k_cache.shape[2]
            slot = written % W
            k_cache[:, :, slot:slot + 1, :] = k
            v_cache[:, :, slot:slot + 1, :] = v
            n_valid = min(written + 1, W)
            if n_valid < W:
                # Not yet wrapped: slots [0, n_valid) hold real tokens and the
                # rest are zeros. Slice rather than mask — attending over 4096
                # mostly-zero keys to serve 20 real ones would cost ~200x the
                # attention FLOPs. Dynamic shape here is fine: this branch is the
                # warm-up, and a future CUDA-graph capture targets the wrapped
                # steady state below, which has a fixed shape.
                k_att, v_att = k_cache[:, :, :n_valid, :], v_cache[:, :, :n_valid, :]
            else:
                k_att, v_att = k_cache, v_cache
            new_state = {"k_cache": k_cache, "v_cache": v_cache,
                         "written": written + 1}
        else:
            # Legacy growing cache: unbounded attention, and any state dict
            # saved before #138 (StateCheckpoint files on disk predate it).
            k_cache = torch.cat([state["k_cache"], k], dim=2)  # [B,H,L+1,d]
            v_cache = torch.cat([state["v_cache"], v], dim=2)
            if self.window is not None and k_cache.shape[2] > self.window:
                k_cache = k_cache[:, :, -self.window:, :]
                v_cache = v_cache[:, :, -self.window:, :]
            k_att, v_att = k_cache, v_cache
            new_state = {"k_cache": k_cache, "v_cache": v_cache}

        o = F.scaled_dot_product_attention(q, k_att, v_att, is_causal=False)
        o = o.transpose(1, 2).reshape(B, 1, D)
        attn_out = self.out_proj(o)

        ffn_out = self.ffn(x_norm)
        out = x_t + self.dropout(attn_out) + ffn_out
        return out, new_state
