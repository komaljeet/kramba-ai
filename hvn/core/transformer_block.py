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
    t = torch.arange(position_idx, position_idx + 1, device=device,
                     dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)          # [1, half]
    emb = torch.cat([freqs, freqs], dim=-1)   # [1, d_head]
    return emb.cos().to(dtype), emb.sin().to(dtype)


def build_rope_table(max_positions: int, d_head: int, device, dtype,
                     base: float = 10000.0):
    """Precomputed RoPE cos/sin for positions ``[0, max_positions)`` (Issue #138).

    Two reasons this replaces per-token construction on the decode path, and the
    second is the one that matters:

    1. ``torch.tensor([float(position_idx)], device=cuda)`` is a **CPU->GPU copy
       from unpinned memory**, which CUDA graph capture rejects outright.
    2. Even with the copy gone, ``position_idx`` changes every token, and a graph
       bakes in the values it captured. A graph recorded at position 17 would
       replay position 17's rotation forever — **silently wrong, not an error**,
       which is far worse than a failed capture.

    A table plus a device-resident index fixes both: the graph performs an
    ``index_select`` whose *input* moves, so each replay reads the correct row.
    Cost is trivial — 4096 positions x 64 d_head x 2 tables in fp32 is ~2 MB.

    Returns ``(cos, sin)``, each ``[max_positions, d_head]``.
    """
    half = d_head // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half, device=device).float() / half))
    t = torch.arange(max_positions, device=device, dtype=torch.float32)
    freqs = torch.outer(t, inv_freq)              # [P, half]
    emb = torch.cat([freqs, freqs], dim=-1)       # [P, d_head]
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

        ``written`` is a plain Python int by default; ``hvn/chat.py``'s state-tree
        codec passes non-tensor scalars through unchanged, so it round-trips.

        **Under ``graph_safe_decode`` it is a device tensor instead**, because a
        CUDA graph cannot touch Python. A graph captured with an int ``written``
        bakes in that token's write slot and attention extent and replays them
        forever — the context silently stops growing (see ``forward_step``).
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
        graph_safe = getattr(self, "graph_safe_decode", False)
        return {
            "k_cache": torch.zeros(shape, device=device, dtype=dtype),
            "v_cache": torch.zeros(shape, device=device, dtype=dtype),
            "written": (torch.zeros((), dtype=torch.long, device=device)
                        if graph_safe else 0),
        }

    def _ring_positions(self, W: int, device) -> torch.Tensor:
        """Cached ``arange(W)`` for the ring mask.

        Built once, not per token: ``torch.arange`` inside ``forward_step`` is an
        allocation, and an allocation inside a CUDA graph capture comes from the
        graph's private pool — legal, but it means one more kernel per token for
        a value that never changes.
        """
        buf = getattr(self, "_ring_pos", None)
        if buf is None or buf.numel() != W or buf.device != device:
            buf = torch.arange(W, device=device)
            self.register_buffer("_ring_pos", buf, persistent=False)
        return buf

    def enable_rope_table(self, max_positions: int | None = None,
                          device=None, dtype=None) -> int:
        """Precompute the RoPE table so decode can be CUDA-graph captured (#138).

        Opt-in and idempotent. Without it ``forward_step`` builds cos/sin per
        token, which (a) is a CPU->GPU copy that graph capture rejects and (b)
        would bake one position into a captured graph.

        ``max_positions`` defaults to the attention window, since a bounded
        window is the furthest back this block can ever attend. Returns the size
        so a caller can assert the range it needs is covered rather than
        discovering a silent clamp at token ``max_positions``.
        """
        p = next(self.parameters())
        device = device if device is not None else p.device
        dtype = dtype if dtype is not None else p.dtype
        if max_positions is None:
            max_positions = self.window if self.window is not None else 4096
        cos, sin = build_rope_table(max_positions, self.d_head, device, dtype,
                                    self.rope_base)
        self.register_buffer("_rope_cos", cos, persistent=False)
        self.register_buffer("_rope_sin", sin, persistent=False)
        # Makes init_state hand back a DEVICE-tensor `written`, which is what
        # lets a captured graph advance the ring instead of replaying one slot.
        self.graph_safe_decode = True
        return max_positions

    def _rope_at(self, position_idx, d_head, device, dtype):
        """cos/sin for one position, from the table when it exists.

        ``position_idx`` may be a Python int (eager) or a 0-d/1-element device
        tensor (graph-safe: the index moves, so each replay reads its own row).
        A Python int inside a capture would be constant-folded into the graph and
        every replay would silently reuse that one position's rotation.
        """
        cos_tab = getattr(self, "_rope_cos", None)
        if cos_tab is None:
            return _rope_cos_sin_at(int(position_idx), d_head, device, dtype,
                                    self.rope_base)
        n = cos_tab.shape[0]
        if torch.is_tensor(position_idx):
            idx = position_idx.reshape(1).to(torch.long) % n
        else:
            # Range-check only on the int path: reading a device tensor here
            # would force a host sync every token and defeat the purpose.
            if position_idx >= n:
                raise IndexError(
                    f"position {position_idx} exceeds the RoPE table ({n}); "
                    f"call enable_rope_table(max_positions=...) with a larger "
                    f"range rather than wrapping silently")
            idx = torch.as_tensor([position_idx], device=cos_tab.device,
                                  dtype=torch.long)
        return (cos_tab.index_select(0, idx).to(dtype),
                self._rope_sin.index_select(0, idx).to(dtype))

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

        cos, sin = self._rope_at(position_idx, d, x_norm.device, q.dtype)
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

            if torch.is_tensor(written):
                # --- graph-safe path (Issue #138) --------------------------
                # EVERYTHING the graph touches must be device-resident. With an
                # int `written`, a captured graph bakes in this token's write
                # slot AND the attention extent, so every replay overwrites one
                # slot and attends over a fixed prefix — the context silently
                # stops growing. Measured directly: `written` stayed at 16 across
                # three replays while the logits kept changing, so it looked
                # alive and was wrong.
                #
                # Two consequences, both deliberate:
                #   * the write is index_copy_ with a DEVICE index, not a Python
                #     slice;
                #   * attention reads the FULL ring at fixed shape with a mask,
                #     because a `[:, :, :n_valid]` slice is exactly the dynamic
                #     shape that got baked in. That costs attention over the whole
                #     window even early on, which is why the graphed path should
                #     use a window near the trained sequence length rather than
                #     4096.
                slot = (written % W).reshape(1)
                k_cache.index_copy_(2, slot, k)
                v_cache.index_copy_(2, slot, v)
                written.add_(1)
                idx = self._ring_positions(W, k_cache.device)
                valid = idx < written.clamp(max=W)          # [W] bool
                attn_mask = valid.view(1, 1, 1, W)
                o = F.scaled_dot_product_attention(q, k_cache, v_cache,
                                                   attn_mask=attn_mask)
                o = o.transpose(1, 2).reshape(B, 1, D)
                attn_out = self.out_proj(o)
                ffn_out = self.ffn(x_norm)
                return x_t + self.dropout(attn_out) + ffn_out, state

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
