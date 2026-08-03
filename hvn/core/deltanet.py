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

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hvn_block import RMSNorm, SwiGLU

try:
    # Fused Triton scan (#83). Optional — falls back to the eager scans if
    # unavailable (e.g. no CUDA/triton). Import guarded so CPU imports still work.
    from .deltanet_triton import gated_delta_scan as _triton_gated_delta_scan
    # Design B (#96): optimal sequential fwd + fused blockwise-recompute bwd.
    # Same math as gated_delta_scan (parity 5e-8), ~1.18x faster scan + 4x less
    # transient backward VRAM. Opt-in; gate-passed at d=64 but not yet verified
    # end-to-end under --fast.
    from .deltanet_triton import megakernel_scan as _triton_megakernel_scan
    _HAS_TRITON_SCAN = True
except Exception:  # pragma: no cover
    _triton_gated_delta_scan = None
    _triton_megakernel_scan = None
    _HAS_TRITON_SCAN = False


# Floor for the intra-chunk cumulative decay ``P`` in the chunked scan (Issue
# #44).  Keys/queries are L2-normalized (|·| ≤ 1), so ``1/P`` reaches at most
# ``1/_P_FLOOR = 1e16`` — well inside fp32's ~3.4e38 range even after summing
# over d_head — while a floor this small only touches channels that have already
# decayed to ≈0 (negligible contribution).
_P_FLOOR = 1e-16


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize over the last dimension.

    The sum-of-squares reduction is done in fp32 regardless of the input dtype:
    in bf16 the accumulation over a wide ``d_head`` loses enough precision that
    the normalized vectors drift, which — compounded through the delta-rule
    recurrence at large width — is one of the seeds of the bf16 loss spike
    (Issue #44).  The result is cast back to the input dtype.
    """
    xf = x.float()
    out = xf / xf.norm(dim=-1, keepdim=True).clamp_min(eps)
    return out.to(x.dtype)


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
    train_chunk : int
        Chunk length ``C`` for the chunked-parallel delta rule used in the
        training forward.  The recurrence is reassociated so only ``S/C``
        steps are sequential (the rest is batched matmuls + one triangular
        solve per chunk), giving a large speedup over the token-by-token scan
        while computing the *identical* result.  Kept modest (32) for numerical
        stability of the intra-chunk decay reparameterization.  Set to 0/1 to
        force the plain sequential scan.
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
        train_chunk: int = 32,
        use_triton_scan: bool = False,
        use_megakernel_scan: bool = False,
        # Accepted for interface compatibility with HVNBlock construction
        # (ignored — DeltaNet defines its own dynamics):
        bond_dimension: int | None = None,
        chunk_size: int | None = None,
        **_unused,
    ):
        super().__init__()
        self.train_chunk = int(train_chunk) if train_chunk and train_chunk > 1 else 0
        # Issue #95: grow the chunk to the largest size that stays numerically
        # EXACT for the current decay (no cumprod underflow) — fast at mild decay
        # (trained α~0.95 → near full-seq chunk), safe at hard decay. Opt-in.
        self.adaptive_chunk = False
        # Use the fused Triton scan (#83) for the training forward when available.
        # Identity-preserving (rel-diff < 2e-3 vs _scan_sequential,
        # tests/test_issue83.py) and MORE numerically reliable than the eager
        # _scan_chunked path it replaces in isolation (that path's own docstring
        # documents it can be silently wrong on decayed channels at large chunks
        # / hard decay). Kept opt-in at the class default (2026-08-01, Issue
        # #122): a broader default-flip attempt found a real, non-trivial loss
        # divergence (0.297 vs 0.010 for eager) specifically in the
        # BlockCoordinateScheduler weight-offload + buffer-caching combination
        # (tests/test_block_coordinate_offload.py::
        # test_lru_cache_hit_avoids_reload_and_preserves_parity) -- narrow to
        # that advanced feature (grad_checkpoint alone is unaffected,
        # tests/test_edgehvn_grad_checkpoint.py 5/5), not root-caused, so the
        # class-level default stays False until it is. Explicitly enabled
        # (and verified safe there -- no BCD/offload involved) in
        # continue_ft_hybrid.py / continue_ft_looped.py via enable_scan_flags.
        self.use_triton_scan = bool(use_triton_scan)
        # Design B (#96): when use_triton_scan is on, route the scan through the
        # fused blockwise-recompute backward instead of the 2-kernel sequential
        # backward. Loss bit-identical, grad diff ~5e-8 (scripts/ab_megakernel.py),
        # ~1.05x faster step + ~4x less transient backward VRAM -- BUT kept
        # opt-in (default False), unlike use_triton_scan: 2026-08-01 testing
        # found it introduces 2 torch.compile graph breaks that triton_scan
        # alone doesn't (tests/test_issue83_stacking.py::test_zero_graph_breaks
        # caught this when a broader default-flip attempt briefly defaulted
        # this True too). Correctness is unaffected either way -- this is a
        # compile-fusion caveat, not a numerics one -- but it's real enough
        # that "unconditionally on" isn't as clean a case as triton_scan's.
        # Still explicitly enabled where already proven (continue_ft_hybrid.py
        # / continue_ft_looped.py's enable_scan_flags), just not the bare default.
        self.use_megakernel_scan = bool(use_megakernel_scan)
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

        # Gates in fp32 (Issue #44): β is the write strength and α is the decay
        # that drives the ``P = cumprod(α)`` / ``1/P`` reparameterization in the
        # chunked scan — a bf16-quantized α near 1.0 (init ≈0.95) compounds error
        # over the chunk and helps seed the large-width divergence.  Sigmoid in
        # fp32 keeps them accurate; kept fp32 to match the fp32 scan.
        beta = torch.sigmoid(self.beta_proj(xc).float()).view(B, S, H, 1)

        if self.use_decay_gate:
            alpha = torch.sigmoid(self.alpha_proj(xc).float()).view(B, S, H, d)
        else:
            alpha = None

        gate = None
        if self.use_output_gate:
            gate = F.silu(self.gate_proj(xc))  # [B, S, D]

        return q, k, v, alpha, beta, gate

    def _read_normalize(self, o: torch.Tensor) -> torch.Tensor:
        """Per-head RMSNorm on the read output.  o: [B, S, H, d] → [B, S, H, d].

        Done in fp32 (Issue #44): the read ``o = S·q`` can grow large at wide
        ``d_head``, and squaring it in bf16 both loses precision and risks
        overflow (bf16 overflows above ≈1.8e19 after ``pow(2)``), which produces
        Inf → NaN.  Normalizing in fp32 keeps the readout finite and scale-stable
        at d≥512; the output is cast back to the block dtype.
        """
        of = o.float()
        rms = of.pow(2).mean(dim=-1, keepdim=True).add(self.read_norm_eps).rsqrt()
        out = of * rms * self.read_norm_weight.float()  # weight broadcasts [H, d]
        return out.to(o.dtype)

    # ------------------------------------------------------------------
    # Delta-rule recurrence (two implementations, identical result)
    # ------------------------------------------------------------------
    @staticmethod
    def _scan_sequential(qN, kN, vN, alphaN, betaN, return_state: bool = False):
        """Token-by-token gated delta-rule scan (reference / short sequences).

        All tensors are [N, S, d] (betaN is [N, S, 1]).  Returns oN [N, S, d].
        This is the exact recurrence that ``forward_step`` also implements.
        When ``return_state`` is True, also returns the final ``S_state`` [N, d, d]
        so a prompt can be prefilled and generation continued (Issue #36).
        """
        N, S, d = qN.shape
        S_state = torch.zeros(N, d, d, device=qN.device, dtype=qN.dtype)
        outs = []
        for t in range(S):
            k_t = kN[:, t]
            if alphaN is not None:
                S_state = S_state * alphaN[:, t].unsqueeze(1)  # decay key columns
            Sk = (S_state * k_t.unsqueeze(1)).sum(-1)
            u = vN[:, t] - Sk
            S_state = S_state + betaN[:, t].unsqueeze(-1) * (u.unsqueeze(2) * k_t.unsqueeze(1))
            outs.append((S_state * qN[:, t].unsqueeze(1)).sum(-1))
        oN = torch.stack(outs, dim=1)
        if return_state:
            return oN, S_state
        return oN

    @staticmethod
    def _scan_chunked(qN, kN, vN, alphaN, betaN, chunk, return_state: bool = False):
        """Chunked-parallel gated delta rule (Yang et al. 2024) — same result.

        Splits the sequence into length-``C`` chunks; only the cross-chunk state
        carry is sequential (``S/C`` steps).  Within a chunk the intra-chunk
        error correction is solved in closed form via a unit lower-triangular
        solve, and the reads/state-update are batched matmuls.  Per-channel
        decay ``α`` is folded in via an intra-chunk cumulative-decay
        reparameterization (GLA/gated-DeltaNet style).

        Derivation (per head, state ``S`` maps key→value, ``o = S q``):
            uᵢ = βᵢ(vᵢ − S̃ᵢ₋₁ kᵢ),  S_t = S̃ + βᵢ uᵢ kᵢᵀ,  S̃ = decay(·)
        With ``P_i`` = inclusive cumprod of α within the chunk, define
        ``k̄=k⊙P``, ``k̃=k/P``, ``q̄=q⊙P``; then the recurrence closes as
        ``(I+strictly_lower(A))U = R`` with ``A[i,j]=βᵢ(k̃ⱼ·k̄ᵢ)``,
        ``R[i]=βᵢ(vᵢ − S_in k̄ᵢ)``, and
        ``O = q̄ S_inᵀ + tril(q̄ k̃ᵀ)U``, ``S_out = S_in⊙P_last + Uᵀ(k⊙P_last/P)``.
        All causal (j≤i) interactions carry the bounded ratio ``P_i/P_j ≤ 1``.
        """
        N, S, d = qN.shape
        dev = qN.device
        orig_dtype = qN.dtype
        # Run the reassociation in fp32 for the 1/P reparameterization stability.
        with torch.autocast(device_type=dev.type, enabled=False):
            qN = qN.float(); kN = kN.float(); vN = vN.float(); betaN = betaN.float()
            if alphaN is not None:
                alphaN = alphaN.float()
            S_in = torch.zeros(N, d, d, device=dev, dtype=torch.float32)  # [d_v, d_k]
            eye = torch.eye(chunk, device=dev, dtype=torch.float32)
            outs = []
            for s in range(0, S, chunk):
                e = min(s + chunk, S)
                L = e - s
                q = qN[:, s:e]; k = kN[:, s:e]; v = vN[:, s:e]  # [N, L, d]
                beta = betaN[:, s:e]                             # [N, L, 1]

                if alphaN is not None:
                    a = alphaN[:, s:e].clamp_min(1e-6)           # [N, L, d]
                    # Inclusive cumprod, floored (Issue #44): the causal ratios
                    # that the algebra actually uses are ``P_i/P_j ≤ 1`` (j≤i),
                    # but the intermediate ``ktil = k/P`` overflows fp32 to Inf
                    # once ``P`` underflows — at large ``d_head`` (d≥512) enough
                    # decay channels shrink within a 32-token chunk that the raw
                    # ``(1e-6)^32`` cumprod underflows, giving Inf → NaN and
                    # (post loss-spike guard) a fully stalled model.  Flooring
                    # ``P`` at ``P_FLOOR`` bounds ``1/P`` while only perturbing
                    # channels that have already decayed to ≈0 (their
                    # contribution is negligible), so the read/state stay finite.
                    P = torch.cumprod(a, dim=1).clamp_min(_P_FLOOR)  # [N, L, d]
                else:
                    P = torch.ones(N, L, d, device=dev, dtype=torch.float32)
                P_last = P[:, -1, :]                             # [N, d]
                kbar = k * P                                     # read/target side
                ktil = k / P                                     # write/source side
                qbar = q * P

                # R[i] = beta_i (v_i - S_in kbar_i)
                Sk = torch.einsum('nvk,nlk->nlv', S_in, kbar)    # [N, L, d_v]
                R = beta * (v - Sk)                              # [N, L, d_v]

                # A[i,j] = beta_i (kbar_i · ktil_j), keep strictly lower (j<i)
                KK = torch.einsum('nik,njk->nij', kbar, ktil)    # [N, L, L]
                A = beta * KK                                    # beta_i scales row i
                M = eye[:L, :L] + torch.tril(A, diagonal=-1)     # unit lower-tri
                U = torch.linalg.solve_triangular(
                    M, R, upper=False, unitriangular=True)       # [N, L, d_v]

                # O = qbar S_inᵀ + tril(qbar · ktilᵀ) U
                QK = torch.tril(torch.einsum('nik,njk->nij', qbar, ktil), diagonal=0)
                O = torch.einsum('nvk,nlk->nlv', S_in, qbar) + torch.einsum('nij,njv->niv', QK, U)
                outs.append(O)

                # S_out = S_in ⊙ P_last(cols) + Σ_i u_i (k_i ⊙ P_last/P_i)ᵀ
                k_scaled = k * (P_last.unsqueeze(1) / P)          # [N, L, d]
                S_in = S_in * P_last.unsqueeze(1) + torch.einsum('niv,nik->nvk', U, k_scaled)

            oN = torch.cat(outs, dim=1)                          # [N, S, d]
        oN = oN.to(orig_dtype) if oN.dtype != orig_dtype else oN
        if return_state:
            return oN, S_in.to(orig_dtype)                       # final [N, d_v, d_k]
        return oN

    # ------------------------------------------------------------------
    # Stable chunked scan — bounded causal ratios, exact at ALL decay (#95)
    # ------------------------------------------------------------------
    @staticmethod
    def _scan_chunked_stable(qN, kN, vN, alphaN, betaN, chunk,
                             return_state: bool = False):
        """Chunked-parallel gated delta rule, numerically stable at ALL decay.

        Same recurrence/result as :meth:`_scan_chunked` and
        :meth:`_scan_sequential`, but the per-channel causal ratios are formed in
        log space as ``exp(logP_i,d − logP_j,d)`` and masked to the causal lower
        triangle **before** the ``exp`` — so every ratio is in ``(0, 1]`` and the
        computation never materializes ``1/P`` (the term that overflows fp32 to
        Inf once ``P = cumprod(α)`` underflows, the bug that makes
        :meth:`_scan_chunked` silently wrong on decayed channels at large chunks
        / hard decay). Large chunks are now exact at any ``α`` — including the
        untrained start (``α≈0.5``) and the hard-decay stress regime
        (``α≈0.1``) — so chunk size is a pure cost tradeoff, not a stability
        limit.

        This is the **math reference** for the fused Triton kernel (#95): it is
        written in plain PyTorch so autograd supplies the backward for free,
        which lets the tests check fwd+bwd parity against
        :meth:`_scan_sequential` at every decay before the kernel is trusted.
        It is NOT the training fast path — the ``[N, L, L, d]`` per-channel
        ratio it materializes is VRAM-prohibitive at the real shape
        (N=32, L=256 → ~0.5 GB/chunk); the fused kernel computes that ratio in
        SRAM tile-by-tile instead. See ``hvn/core/deltanet_triton.py``.

        Layout: ``qN,kN,vN,alphaN`` are ``[N,S,d]``, ``betaN`` is ``[N,S,1]``.
        Returns ``oN`` ``[N,S,d]`` (and the final ``[N,d,d]`` state if
        ``return_state``).
        """
        N, S, d = qN.shape
        dev = qN.device
        orig_dtype = qN.dtype
        with torch.autocast(device_type=dev.type, enabled=False):
            qN = qN.float(); kN = kN.float(); vN = vN.float(); betaN = betaN.float()
            if alphaN is not None:
                alphaN = alphaN.float()
            S_in = torch.zeros(N, d, d, device=dev, dtype=torch.float32)  # [d_v, d_k]
            idx = torch.arange(chunk, device=dev)
            # Causal masks (True = zero the ratio there). The exp only ever sees
            # <=0 (or -inf), so the result is in (0, 1] u {0} — never overflows.
            mask_strict = idx[None, :] >= idx[:, None]   # j >= i : A is strictly lower
            mask_lower = idx[None, :] > idx[:, None]     # j >  i : QK keeps the diagonal
            outs = []
            for s in range(0, S, chunk):
                e = min(s + chunk, S)
                L = e - s
                q = qN[:, s:e]; k = kN[:, s:e]; v = vN[:, s:e]   # [N, L, d]
                beta = betaN[:, s:e]                            # [N, L, 1]

                if alphaN is not None:
                    a = alphaN[:, s:e].clamp_min(1e-30)          # [N, L, d]
                    logP = torch.cumsum(torch.log(a), dim=1)      # inclusive [N, L, d]
                else:
                    logP = torch.zeros(N, L, d, device=dev, dtype=torch.float32)
                P = torch.exp(logP)                            # [N,L,d], <=1, safe
                P_last = P[:, -1, :]                            # [N, d]
                logP_last = logP[:, -1, :]                      # [N, d]

                # S_in read side: kbar = k⊙P, qbar = q⊙P (P <= 1, safe — no 1/P).
                kbar = k * P
                qbar = q * P
                Sk = torch.einsum('nvk,nlk->nlv', S_in, kbar)    # [N, L, d_v]
                R = beta * (v - Sk)                              # [N, L, d_v]

                # Per-channel causal ratio: ratio[i,j,d] = exp(logP[i,d]-logP[j,d]).
                # Masked to the causal triangle BEFORE exp so the exponent is
                # always <=0 (or -inf); the [N,L,L,d] tensor is the oracle-only
                # cost the fused kernel avoids by tiling it in SRAM.
                logR = logP[:, :, None] - logP[:, None, :]       # [N, L, L, d]
                ratio_strict = torch.exp(logR.masked_fill(mask_strict[:L, :L, None], float('-inf')))
                ratio_lower = torch.exp(logR.masked_fill(mask_lower[:L, :L, None], float('-inf')))

                # A[i,j] = beta_i * sum_d k_i,d k_j,d ratio[i,j,d]   (strictly lower)
                A = beta * torch.einsum('nid,njd,nijd->nij', k, k, ratio_strict)  # [N,L,L]
                M = torch.eye(L, device=dev, dtype=torch.float32) + A
                U = torch.linalg.solve_triangular(
                    M, R, upper=False, unitriangular=True)       # [N, L, d_v]

                # O = qbar·S_inᵀ + QK·U,  QK[i,j] = sum_d q_i,d k_j,d ratio[i,j,d]
                QK = torch.einsum('nid,njd,nijd->nij', q, k, ratio_lower)
                O = (torch.einsum('nvk,nlk->nlv', S_in, qbar)
                     + torch.einsum('nij,njv->niv', QK, U))
                outs.append(O)

                # S_out = S_in⊙P_last + Σ_i u_i (k_i ⊙ P_last/P_i)ᵀ
                #   P_last/P_i = exp(logP_last - logP_i) <= 1 (i <= last), safe.
                k_scaled = k * torch.exp(logP_last[:, None, :] - logP)
                S_in = (S_in * P_last.unsqueeze(1)
                        + torch.einsum('niv,nik->nvk', U, k_scaled))

            oN = torch.cat(outs, dim=1)                          # [N, S, d]
        oN = oN.to(orig_dtype) if oN.dtype != orig_dtype else oN
        if return_state:
            return oN, S_in.to(orig_dtype)
        return oN

    @staticmethod
    def _safe_chunk(alphaN, base_chunk: int, S: int) -> int:
        """Largest chunk whose worst-case intra-chunk cumprod-α stays above
        ``_P_FLOOR`` (#95).

        When the cumprod never underflows, ``_scan_chunked`` runs WITHOUT the
        ``P_FLOOR`` clamp and is therefore numerically exact (fp32 roundoff
        ~1e-7) at any decay. The safe length is ``C < log(_P_FLOOR)/log(α_min)``;
        we take a power-of-2 bucket ≤ that bound (with margin) so torch.compile
        sees only a handful of distinct trip counts. Grows when decay is mild
        (fast), shrinks when hard (safe), never below ``base_chunk``.
        """
        if alphaN is None:
            return min(base_chunk, S) if base_chunk else S
        a_min = float(alphaN.detach().min().clamp_min(1e-6))  # one host sync/forward
        if a_min >= 0.999:
            bound = S
        else:
            # a_min**C > _P_FLOOR → C < log(_P_FLOOR)/log(a_min); keep 1e4 margin
            # so the cumprod stays clear of the floor (→ no clamp → exact).
            bound = int(math.log(_P_FLOOR * 1e4) / math.log(a_min))
        bound = min(bound, S)
        # Largest power-of-2 bucket ≤ bound (bounded recompiles under compile).
        # NOTE: may go BELOW base_chunk at hard decay — correctness over speed;
        # 32 itself underflows once α_min ≲ 0.31, so we must be free to shrink.
        # Cap at 256: beyond it the O(chunk²) intra-chunk solve outweighs the
        # fewer-chunks win (measured sweet spot at S=512).
        chunk = 0
        for c in (16, 32, 64, 128, 256):
            if c <= bound:
                chunk = c
        if chunk == 0:            # bound < 16 (very hard decay, rare): exact raw
            chunk = max(2, bound)
        return chunk

    # ------------------------------------------------------------------
    # Training forward (sequential delta-rule scan, O(1) memory in N)
    # ------------------------------------------------------------------
    def forward(
        self,
        x: torch.Tensor,
        position_ids: torch.Tensor | None = None,
        return_state: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict]:
        """Process a sequence.  x: [B, S, D] → [B, S, D].

        Parallel-residual: ``x = x + deltanet(norm(x)) + ffn(norm(x))``.

        When ``return_state`` is True, also returns the recurrent state
        (``{"S": [B*n_heads, d, d], "conv": [B, D, K-1]}``) after absorbing the
        whole sequence, so generation can continue via ``forward_step``
        (prompt prefill for Issue #36).  The state matches ``init_state``.
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

        # Fused Triton scan (#83): 9x faster fwd+bwd, numerically stable at all
        # decay. Used for the training forward when enabled and available; not
        # for incremental/return_state generation (that path stays eager).
        if (self.use_triton_scan and _HAS_TRITON_SCAN and not return_state
                and S > 1 and qN.is_cuda):
            if self.use_megakernel_scan:
                oN = _triton_megakernel_scan(qN, kN, vN, alphaN, betaN)
            else:
                oN = _triton_gated_delta_scan(qN, kN, vN, alphaN, betaN)
        # Reassociated (chunked-parallel) delta rule for training speed; falls
        # back to the plain sequential scan for short sequences or when disabled.
        elif self.train_chunk and S > 1:
            chunk = (self._safe_chunk(alphaN, self.train_chunk, S)
                     if self.adaptive_chunk else self.train_chunk)
            oN = self._scan_chunked(qN, kN, vN, alphaN, betaN, chunk,
                                    return_state=return_state)
        else:
            oN = self._scan_sequential(qN, kN, vN, alphaN, betaN,
                                       return_state=return_state)
        if return_state:
            oN, S_final = oN  # unpack (oN, final recurrent state)

        o = oN.view(B, H, S, d).permute(0, 2, 1, 3)  # [B, S, H, d]
        o = self._read_normalize(o)                 # normalizer (lever 1)
        o = o.reshape(B, S, D)

        if gate is not None:
            o = o * gate                            # output gate (lever 3)

        deflected = self.out_proj(o)
        ffn_out = self.ffn(x_norm)
        out = x + self.dropout(deflected) + ffn_out

        if return_state:
            state = {"S": S_final}
            if self.conv_kernel:
                # Conv buffer = last (K-1) *pre-conv* tokens (x_norm), matching
                # the window that _conv_step concatenates each step.  Left-pad
                # with zeros when the prompt is shorter than K-1.
                k1 = self.conv_kernel - 1
                buf = x_norm.transpose(1, 2)  # [B, D, S]
                if S < k1:
                    pad = torch.zeros(B, D, k1 - S, device=x.device, dtype=x_norm.dtype)
                    buf = torch.cat([pad, buf], dim=2)
                else:
                    buf = buf[:, :, -k1:]
                state["conv"] = buf.contiguous()
            return out, state
        return out

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
