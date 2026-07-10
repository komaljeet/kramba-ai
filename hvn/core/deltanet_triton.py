"""
Custom Triton kernels for HVN's gated delta-rule scan (Issue #83).
====================================================================

Why a custom kernel (not fla): HVN's recurrence is

    S_t = S_{t-1}·diag(α_t)·(I − β_t k_t k_tᵀ) + β_t v_t k_tᵀ,      o_t = S_t q_t

with **per-channel** decay α (a d-vector per head) and scalar write strength β.
Its low-rank term −β(α⊙k)kᵀ is *asymmetric*, so fla's fast kernels do not fit
(``gated_delta_rule`` assumes scalar decay → changes identity; the fast DPLR
kernel assumes b=−a / symmetric low-rank). See #83. The only fla op that matched
our math (``dplr.naive``) is a slow Python loop.

This module implements the recurrence **directly** in Triton — the same
token-by-token math as :meth:`DeltaNetBlock._scan_sequential`, which is the
ground-truth reference. Because it never forms ``1/P``, it is numerically
**stable at all decay regimes**, unlike the eager ``_scan_chunked`` (which is
correct only for α≳0.5 at chunk=32 — the bug the #79 harness caught).

Milestones (this file grows across commits, each gated by tests/test_issue79.py):
  1. forward kernel  ✅ (this commit) — validated vs _scan_sequential at all α.
  2. backward kernel (training) — TODO.
  3. chunk-parallel tiling for throughput — TODO.

Layout: all tensors are [N, S, d] with N = B*n_heads, d = d_head; β is [N, S].
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
    _HAS_TRITON = True
except ImportError:  # pragma: no cover
    _HAS_TRITON = False


if _HAS_TRITON:

    @triton.jit
    def _gated_delta_fwd_kernel(
        q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, o_ptr,
        S: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
    ):
        """Forward gated delta-rule recurrence, one program per (n, v-block).

        State tile is [BV, D] (BV rows of the d_v axis, all D=d_k columns), held
        across the sequential scan over S. fp32 throughout — no reparameterization,
        so stable at any α. Grid = (N, D // BV).
        """
        pid_n = tl.program_id(0)
        pid_v = tl.program_id(1)

        offs_k = tl.arange(0, D)              # d_k indices (full)
        offs_v = pid_v * BV + tl.arange(0, BV)  # this program's d_v rows

        # Base offset for sequence n; per-step add t*D.
        base = pid_n * S * D
        beta_base = pid_n * S

        state = tl.zeros((BV, D), dtype=tl.float32)  # S_t : [d_v_block, d_k]

        for t in range(S):
            row = base + t * D
            k_t = tl.load(k_ptr + row + offs_k).to(tl.float32)          # [D]
            q_t = tl.load(q_ptr + row + offs_k).to(tl.float32)          # [D]
            a_t = tl.load(alpha_ptr + row + offs_k).to(tl.float32)      # [D]
            v_t = tl.load(v_ptr + row + offs_v).to(tl.float32)         # [BV]
            b_t = tl.load(beta_ptr + beta_base + t).to(tl.float32)      # scalar

            # decay key columns, then delta correction
            state = state * a_t[None, :]                                # [BV,D]
            Sk = tl.sum(state * k_t[None, :], axis=1)                   # [BV]
            u = v_t - Sk                                               # [BV]
            state = state + b_t * (u[:, None] * k_t[None, :])           # [BV,D]
            o_t = tl.sum(state * q_t[None, :], axis=1)                  # [BV]

            tl.store(o_ptr + row + offs_v, o_t.to(o_ptr.dtype.element_ty))


def gated_delta_forward(q, k, v, alpha, beta, BV: int = 16):
    """Triton forward of HVN's gated delta-rule scan (inference / no grad).

    Parameters
    ----------
    q, k, v, alpha : [N, S, d]   (q, k already L2-normalized; alpha in (0,1))
    beta           : [N, S, 1] or [N, S]   scalar write strength per head
    Returns o : [N, S, d].
    """
    assert _HAS_TRITON, "triton not available"
    assert q.is_cuda, "CUDA required"
    N, S, D = q.shape
    assert D % BV == 0, f"d_head {D} must be divisible by BV {BV}"
    q, k, v, alpha = (t.contiguous().float() for t in (q, k, v, alpha))
    beta = beta.reshape(N, S).contiguous().float()
    o = torch.empty_like(q)
    grid = (N, D // BV)
    _gated_delta_fwd_kernel[grid](q, k, v, alpha, beta, o, S=S, D=D, BV=BV)
    return o


if _HAS_TRITON:

    @triton.jit
    def _gd_fwd_states_kernel(
        q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, o_ptr, states_ptr,
        S: tl.constexpr, D: tl.constexpr,
    ):
        """Forward that also writes the *entering* state S_{t-1} per step, so the
        backward can recompute exactly. One program per sequence; full [D,D]
        state. states layout: [N, S, D(d_v), D(d_k)]."""
        n = tl.program_id(0)
        ov = tl.arange(0, D)[:, None]   # d_v axis
        ok = tl.arange(0, D)[None, :]   # d_k axis
        base = n * S * D
        bbase = n * S
        sbase = n * S * D * D
        state = tl.zeros((D, D), dtype=tl.float32)
        for t in range(S):
            row = base + t * D
            k_t = tl.load(k_ptr + row + tl.arange(0, D)).to(tl.float32)   # [D] d_k
            q_t = tl.load(q_ptr + row + tl.arange(0, D)).to(tl.float32)
            a_t = tl.load(alpha_ptr + row + tl.arange(0, D)).to(tl.float32)
            v_t = tl.load(v_ptr + row + tl.arange(0, D)).to(tl.float32)   # [D] d_v
            b_t = tl.load(beta_ptr + bbase + t).to(tl.float32)
            # save entering state S_{t-1}
            tl.store(states_ptr + sbase + t * D * D + ov * D + ok, state)
            state = state * a_t[None, :]
            Sk = tl.sum(state * k_t[None, :], axis=1)       # [D] d_v
            u = v_t - Sk
            state = state + b_t * (u[:, None] * k_t[None, :])
            o_t = tl.sum(state * q_t[None, :], axis=1)      # [D] d_v
            tl.store(o_ptr + row + tl.arange(0, D), o_t.to(o_ptr.dtype.element_ty))

    @triton.jit
    def _gd_bwd_kernel(
        q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, states_ptr, do_ptr,
        dq_ptr, dk_ptr, dv_ptr, dalpha_ptr, dbeta_ptr,
        S: tl.constexpr, D: tl.constexpr,
    ):
        """Reverse-time adjoint (validated vs autograd at 1e-16 in fp64). One
        program per sequence; full [D,D] state grad carried backward."""
        n = tl.program_id(0)
        rk = tl.arange(0, D)             # d_k axis (1-D)
        rv = tl.arange(0, D)             # d_v axis (1-D)
        ov = tl.arange(0, D)[:, None]    # d_v
        ok = tl.arange(0, D)[None, :]    # d_k
        base = n * S * D
        bbase = n * S
        sbase = n * S * D * D
        dS = tl.zeros((D, D), dtype=tl.float32)
        for t in range(S - 1, -1, -1):
            row = base + t * D
            Sprev = tl.load(states_ptr + sbase + t * D * D + ov * D + ok)  # [dv,dk]
            k_t = tl.load(k_ptr + row + rk).to(tl.float32)
            q_t = tl.load(q_ptr + row + rk).to(tl.float32)
            a_t = tl.load(alpha_ptr + row + rk).to(tl.float32)
            v_t = tl.load(v_ptr + row + rv).to(tl.float32)
            b_t = tl.load(beta_ptr + bbase + t).to(tl.float32)
            do_t = tl.load(do_ptr + row + rv).to(tl.float32)               # [dv]
            # recompute Sd, u, S_t
            Sd = Sprev * a_t[None, :]
            Sk = tl.sum(Sd * k_t[None, :], axis=1)          # [dv]
            u = v_t - Sk
            S_t = Sd + b_t * (u[:, None] * k_t[None, :])
            # o_t = S_t @ q_t
            dS_t = dS + do_t[:, None] * q_t[None, :]         # [dv,dk]
            dq_t = tl.sum(S_t * do_t[:, None], axis=0)       # sum dv -> [dk]
            tl.store(dq_ptr + row + rk, dq_t)
            # S_t = Sd + b*outer(u,k)
            dSd = dS_t
            dSk_via = tl.sum(dS_t * k_t[None, :], axis=1)    # [dv]
            dbeta_t = tl.sum(u * dSk_via)                    # scalar
            tl.store(dbeta_ptr + bbase + t, dbeta_t)
            du = b_t * dSk_via                               # [dv]
            dk_t = b_t * tl.sum(dS_t * u[:, None], axis=0)   # [dk]
            # u = v - Sd@k
            tl.store(dv_ptr + row + rv, du)                  # dv_t = du
            dSd = dSd - du[:, None] * k_t[None, :]
            dk_t = dk_t - tl.sum(Sd * du[:, None], axis=0)   # [dk]
            tl.store(dk_ptr + row + rk, dk_t)
            # Sd = Sprev * alpha
            dalpha_t = tl.sum(Sprev * dSd, axis=0)           # sum dv -> [dk]
            tl.store(dalpha_ptr + row + rk, dalpha_t)
            dS = dSd * a_t[None, :]


class _GatedDeltaScan(torch.autograd.Function):
    """Autograd-enabled HVN gated delta-rule scan via Triton fwd+bwd kernels."""

    @staticmethod
    def forward(ctx, q, k, v, alpha, beta):
        N, S, D = q.shape
        q, k, v, alpha = (t.contiguous().float() for t in (q, k, v, alpha))
        beta2 = beta.reshape(N, S).contiguous().float()
        # States-free forward: save only the (small) inputs. The [N,S,D,D] state
        # tensor is NOT kept — storing it for all layers blows past 6GB VRAM and
        # WDDM-spills (kills the speedup). Backward recomputes states transiently.
        o = torch.empty_like(q)
        BV = 16 if D % 16 == 0 else 1
        _gated_delta_fwd_kernel[(N, D // BV)](q, k, v, alpha, beta2, o, S=S, D=D, BV=BV)
        ctx.save_for_backward(q, k, v, alpha, beta2)
        ctx.beta_shape = beta.shape
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta2 = ctx.saved_tensors
        N, S, D = q.shape
        do = do.contiguous().float()
        # Recompute the entering states here (transient — freed when this layer's
        # backward returns, so peak = one layer's states, not all L).
        states = torch.empty(N, S, D, D, device=q.device, dtype=torch.float32)
        o_scratch = torch.empty_like(q)
        _gd_fwd_states_kernel[(N,)](q, k, v, alpha, beta2, o_scratch, states, S=S, D=D)
        dq = torch.empty_like(q); dk = torch.empty_like(q)
        dv = torch.empty_like(q); dalpha = torch.empty_like(q)
        dbeta2 = torch.empty_like(beta2)
        _gd_bwd_kernel[(N,)](q, k, v, alpha, beta2, states, do,
                             dq, dk, dv, dalpha, dbeta2, S=S, D=D)
        return dq, dk, dv, dalpha, dbeta2.reshape(ctx.beta_shape)


def gated_delta_scan(q, k, v, alpha, beta):
    """Autograd-enabled HVN gated delta-rule scan (training). Same math as
    DeltaNetBlock._scan_sequential, fused in Triton, stable at all decay.

    q, k, v, alpha : [N, S, d]  ;  beta : [N, S, 1]  ->  o : [N, S, d].
    """
    assert _HAS_TRITON and q.is_cuda, "CUDA + triton required"
    return _GatedDeltaScan.apply(q, k, v, alpha, beta)


# ===========================================================================
# Chunk-parallel scan (#95) — stable at all decay, fast at all decay.
# ===========================================================================
# The sequential kernel above is exact and alpha-independent but does S token
# steps in series (~60% of the HVN training step at --fast). This chunk-parallel
# path replaces those S tiny sequential state updates with S/C chunk carries +
# O(C^2 d) parallel matmuls / a fused triangular solve per chunk.
#
# The one numerically-novel op is the *per-channel causal Gram matrix*
#
#     G[i,j] = sum_d a[i,d] b[j,d] * exp(logP[i,d] - logP[j,d])     (j <= i)
#
# where logP = cumsum(log alpha) is the inclusive log cumprod of the per-channel
# decay. Forming the ratio as exp(logP_i - logP_j) (masked to the causal lower
# triangle *before* the exp) keeps every factor in (0, 1] -- it never
# materializes 1/P, the term that overflows fp32 to Inf once cumprod-alpha
# underflows and makes the eager _scan_chunked silently wrong at hard decay.
# That ratio is per-channel (a [L,L,d] tensor), so the fused Triton kernel
# computes it tile-by-tile in SRAM (a [BI,BJ,BD] register tile) and writes only
# the [N,L,L] result G to HBM. See DeltaNetBlock._scan_chunked_stable for the
# eager reference (validated to ~1e-7 fwd / ~1e-5 bwd at all decay).
#
# The triangular solve, the S_in reads, the QK@U read, and the cross-chunk state
# carry are all stock PyTorch ops (cuBLAS solve + matmul), so autograd composes
# the whole scan's backward from the single hand-written log-Gram adjoint.

if _HAS_TRITON:

    @triton.jit
    def _loggram_fwd_kernel(a_ptr, b_ptr, logP_ptr, out_ptr,
                            N, L,
                            D: tl.constexpr,
                            STRICT: tl.constexpr,
                            BLOCK_I: tl.constexpr,
                            BLOCK_J: tl.constexpr,
                            BLOCK_D: tl.constexpr):
        """Per-channel causal log-space Gram: G[i,j] = sum_d a[i,d] b[j,d]
        exp(logP[i,d]-logP[j,d]), masked to j<i (STRICT=1) or j<=i (STRICT=0).

        Grid (N, L//BI, L//BJ). Inputs [N, L, d] contiguous; out [N, L, L].
        """
        pid_n = tl.program_id(0)
        pid_i = tl.program_id(1)
        pid_j = tl.program_id(2)
        ri = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)      # [BI]
        rj = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)      # [BJ]
        li = ri < L
        lj = rj < L
        # causal validity over (i, j)
        if STRICT:
            ij = rj[None, :] < ri[:, None]
        else:
            ij = rj[None, :] <= ri[:, None]
        valid = ij & li[:, None] & lj[None, :]            # [BI, BJ]

        acc = tl.zeros((BLOCK_I, BLOCK_J), dtype=tl.float32)
        base = pid_n * L * D
        for d0 in range(0, D, BLOCK_D):
            rd = d0 + tl.arange(0, BLOCK_D)               # [BD]
            ld = rd < D
            a_ptrs = a_ptr + base + ri[:, None] * D + rd[None, :]
            b_ptrs = b_ptr + base + rj[:, None] * D + rd[None, :]
            p_i_ptrs = logP_ptr + base + ri[:, None] * D + rd[None, :]
            p_j_ptrs = logP_ptr + base + rj[:, None] * D + rd[None, :]
            a_id = tl.load(a_ptrs, mask=li[:, None] & ld[None, :], other=0.0)
            b_jd = tl.load(b_ptrs, mask=lj[:, None] & ld[None, :], other=0.0)
            lp_i = tl.load(p_i_ptrs, mask=li[:, None] & ld[None, :], other=0.0)
            lp_j = tl.load(p_j_ptrs, mask=lj[:, None] & ld[None, :], other=0.0)
            logr = lp_i[:, None, :] - lp_j[None, :, :]     # [BI, BJ, BD]
            logr = tl.where(valid[:, :, None], logr, float('-inf'))
            ratio = tl.exp(logr)                            # (0,1] u {0}
            prod = a_id[:, None, :] * b_jd[None, :, :] * ratio
            acc += tl.sum(prod, axis=2)
        out_ptrs = out_ptr + pid_n * L * L + ri[:, None] * L + rj[None, :]
        tl.store(out_ptrs, acc, mask=li[:, None] & lj[None, :])


def _log_gram_ref(a, b, logP, strict):
    """Eager fp32 reference for the log-Gram (the oracle's A/QK core)."""
    N, L, d = a.shape
    idx = torch.arange(L, device=a.device)
    mask = (idx[None, :] < idx[:, None]) if strict else (idx[None, :] <= idx[:, None])
    logR = logP[:, :, None] - logP[:, None, :]               # [N,L,L,d]
    ratio = torch.exp(logR.masked_fill(~mask[:, :, None], float('-inf')))
    return torch.einsum('nid,njd,nijd->nij', a, b, ratio)     # [N,L,L]


if _HAS_TRITON:

    @triton.jit
    def _loggram_bwd_da_kernel(dG_ptr, b_ptr, logP_ptr, da_ptr,
                               N, L,
                               D: tl.constexpr, STRICT: tl.constexpr,
                               BLOCK_I: tl.constexpr,
                               BLOCK_J: tl.constexpr,
                               BLOCK_D: tl.constexpr):
        """da[i,d] = sum_j dG[i,j] b[j,d] exp(logP[i,d]-logP[j,d]) over the
        causal mask. Grid (N, L//BI, D//BD); loops over j-blocks."""
        pid_n = tl.program_id(0)
        pid_i = tl.program_id(1)
        pid_d = tl.program_id(2)
        ri = pid_i * BLOCK_I + tl.arange(0, BLOCK_I)
        rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        li = ri < L
        ld = rd < D
        base = pid_n * L * D
        lp_i = tl.load(logP_ptr + base + ri[:, None] * D + rd[None, :],
                       mask=li[:, None] & ld[None, :], other=0.0)        # [BI, BD]
        acc = tl.zeros((BLOCK_I, BLOCK_D), dtype=tl.float32)
        for j0 in range(0, L, BLOCK_J):
            rj = j0 + tl.arange(0, BLOCK_J)
            lj = rj < L
            if STRICT:
                valid = rj[None, :] < ri[:, None]
            else:
                valid = rj[None, :] <= ri[:, None]
            valid = valid & li[:, None] & lj[None, :]
            b_jd = tl.load(b_ptr + base + rj[:, None] * D + rd[None, :],
                           mask=lj[:, None] & ld[None, :], other=0.0)    # [BJ, BD]
            lp_j = tl.load(logP_ptr + base + rj[:, None] * D + rd[None, :],
                           mask=lj[:, None] & ld[None, :], other=0.0)
            dG = tl.load(dG_ptr + pid_n * L * L + ri[:, None] * L + rj[None, :],
                         mask=li[:, None] & lj[None, :], other=0.0)      # [BI, BJ]
            logr = lp_i[:, None, :] - lp_j[None, :, :]                   # [BI, BJ, BD]
            logr = tl.where(valid[:, :, None], logr, float('-inf'))
            acc += tl.sum(dG[:, :, None] * b_jd[None, :, :] * tl.exp(logr), axis=1)
        tl.store(da_ptr + base + ri[:, None] * D + rd[None, :], acc,
                 mask=li[:, None] & ld[None, :])

    @triton.jit
    def _loggram_bwd_db_kernel(dG_ptr, a_ptr, logP_ptr, db_ptr,
                               N, L,
                               D: tl.constexpr, STRICT: tl.constexpr,
                               BLOCK_I: tl.constexpr,
                               BLOCK_J: tl.constexpr,
                               BLOCK_D: tl.constexpr):
        """db[j,d] = sum_i dG[i,j] a[i,d] exp(logP[i,d]-logP[j,d]) over the
        causal mask. Grid (N, L//BJ, D//BD); loops over i-blocks."""
        pid_n = tl.program_id(0)
        pid_j = tl.program_id(1)
        pid_d = tl.program_id(2)
        rj = pid_j * BLOCK_J + tl.arange(0, BLOCK_J)
        rd = pid_d * BLOCK_D + tl.arange(0, BLOCK_D)
        lj = rj < L
        ld = rd < D
        base = pid_n * L * D
        lp_j = tl.load(logP_ptr + base + rj[:, None] * D + rd[None, :],
                       mask=lj[:, None] & ld[None, :], other=0.0)        # [BJ, BD]
        acc = tl.zeros((BLOCK_J, BLOCK_D), dtype=tl.float32)
        for i0 in range(0, L, BLOCK_I):
            ri = i0 + tl.arange(0, BLOCK_I)
            li = ri < L
            if STRICT:
                valid = rj[None, :] < ri[:, None]
            else:
                valid = rj[None, :] <= ri[:, None]
            valid = valid & li[:, None] & lj[None, :]
            a_id = tl.load(a_ptr + base + ri[:, None] * D + rd[None, :],
                           mask=li[:, None] & ld[None, :], other=0.0)    # [BI, BD]
            lp_i = tl.load(logP_ptr + base + ri[:, None] * D + rd[None, :],
                           mask=li[:, None] & ld[None, :], other=0.0)
            dG = tl.load(dG_ptr + pid_n * L * L + ri[:, None] * L + rj[None, :],
                         mask=li[:, None] & lj[None, :], other=0.0)      # [BI, BJ]
            logr = lp_i[:, None, :] - lp_j[None, :, :]                   # [BI, BJ, BD]
            logr = tl.where(valid[:, :, None], logr, float('-inf'))
            acc += tl.sum(dG[:, :, None] * a_id[:, None, :] * tl.exp(logr), axis=0)
        tl.store(db_ptr + base + rj[:, None] * D + rd[None, :], acc,
                 mask=lj[:, None] & ld[None, :])


class _LogGram(torch.autograd.Function):
    """Autograd-enabled fused per-channel causal log-space Gram.

    Forward: G[i,j] = sum_d a[i,d] b[j,d] exp(logP[i,d]-logP[j,d]), masked.
    Backward: da/db (causal reductions) + dlogP = a*da - b*db (elementwise).
    Stable at all decay (the ratio is exp of a masked <=0 exponent).
    """

    @staticmethod
    def forward(ctx, a, b, logP, strict):
        N, L, D = a.shape
        a = a.contiguous().float()
        b = b.contiguous().float()
        logP = logP.contiguous().float()
        G = torch.empty(N, L, L, device=a.device, dtype=torch.float32)
        BLOCK_I, BLOCK_J, BLOCK_D = 16, 16, 16
        grid = (N, triton.cdiv(L, BLOCK_I), triton.cdiv(L, BLOCK_J))
        _loggram_fwd_kernel[grid](a, b, logP, G, N, L, D=D, STRICT=int(strict),
                                  BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J, BLOCK_D=BLOCK_D)
        ctx.save_for_backward(a, b, logP)
        ctx.strict = strict
        ctx.L = L
        ctx.D = D
        return G

    @staticmethod
    def backward(ctx, dG):
        a, b, logP = ctx.saved_tensors
        N, L, D = a.shape
        dG = dG.contiguous().float()
        BLOCK_I, BLOCK_J, BLOCK_D = 16, 16, 16
        da = torch.empty_like(a)
        db = torch.empty_like(b)
        grid_a = (N, triton.cdiv(L, BLOCK_I), triton.cdiv(D, BLOCK_D))
        _loggram_bwd_da_kernel[grid_a](dG, b, logP, da, N, L, D=D,
                                       STRICT=int(ctx.strict),
                                       BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
                                       BLOCK_D=BLOCK_D)
        grid_b = (N, triton.cdiv(L, BLOCK_J), triton.cdiv(D, BLOCK_D))
        _loggram_bwd_db_kernel[grid_b](dG, a, logP, db, N, L, D=D,
                                       STRICT=int(ctx.strict),
                                       BLOCK_I=BLOCK_I, BLOCK_J=BLOCK_J,
                                       BLOCK_D=BLOCK_D)
        dlogP = a * da - b * db
        return da, db, dlogP, None


def log_gram(a, b, logP, strict: bool):
    """Fused per-channel causal log-space Gram, autograd-enabled. a, b, logP :
    [N, L, d] fp32 on CUDA. Returns G : [N, L, L]."""
    assert _HAS_TRITON and a.is_cuda, "CUDA + triton required"
    return _LogGram.apply(a, b, logP, strict)


def chunked_scan_forward(q, k, v, alpha, beta, chunk: int = 128):
    """Chunk-parallel gated delta-rule scan — FORWARD only (#95). Same math as
    DeltaNetBlock._scan_chunked_stable, but the per-channel causal Gram (A, QK)
    is the fused Triton log-Gram and the solve / S_in matmuls / carry are stock
    cuBLAS ops. Exact at all decay; the [N,L,L,d] ratio never hits HBM.

    q, k, v, alpha : [N, S, d]  ;  beta : [N, S, 1]  ->  o : [N, S, d] (fp32).
    """
    assert _HAS_TRITON and q.is_cuda, "CUDA + triton required"
    N, S, d = q.shape
    dev = q.device
    with torch.autocast(device_type=dev.type, enabled=False):
        q, k, v = q.float(), k.float(), v.float()
        beta = beta.float()
        if alpha is not None:
            alpha = alpha.float()
        S_in = torch.zeros(N, d, d, device=dev, dtype=torch.float32)
        outs = []
        for s in range(0, S, chunk):
            e = min(s + chunk, S)
            L = e - s
            qc, kc, vc = q[:, s:e], k[:, s:e], v[:, s:e]      # [N, L, d]
            bc = beta[:, s:e]                                  # [N, L, 1]
            if alpha is not None:
                logP = torch.cumsum(torch.log(alpha[:, s:e].clamp_min(1e-30)), dim=1)
            else:
                logP = torch.zeros(N, L, d, device=dev, dtype=torch.float32)
            P = torch.exp(logP)
            P_last = P[:, -1, :]                                # [N, d]
            logP_last = logP[:, -1, :]                          # [N, d]
            kbar = kc * P
            qbar = qc * P
            Sk = torch.einsum('nvk,nlk->nlv', S_in, kbar)       # [N, L, d]
            R = bc * (vc - Sk)                                  # [N, L, d]
            A = bc * log_gram(kc, kc, logP, strict=True)        # [N, L, L] strictly lower
            M = torch.eye(L, device=dev, dtype=torch.float32) + A
            U = torch.linalg.solve_triangular(M, R, upper=False, unitriangular=True)
            QK = log_gram(qc, kc, logP, strict=False)           # [N, L, L] lower incl diag
            O = (torch.einsum('nvk,nlk->nlv', S_in, qbar)
                 + torch.einsum('nij,njv->niv', QK, U))         # [N, L, d]
            outs.append(O)
            k_scaled = kc * torch.exp(logP_last[:, None, :] - logP)
            S_in = (S_in * P_last[:, None, :]
                    + torch.einsum('niv,nik->nvk', U, k_scaled))
        oN = torch.cat(outs, dim=1)                            # [N, S, d]
    return oN


# ===========================================================================
# Fused chunk-parallel scan MEGAKERNEL (#96) — one kernel, SRAM-resident.
# ===========================================================================
# The hybrid `chunked_scan_forward` above was measured 2-4x SLOWER than the
# sequential Triton kernel at d=64: it launches ~40 kernels/chunk and writes
# the [N,L,L] A/QK to HBM between them. This megakernel fuses the whole chunk
# pipeline into ONE program per sequence so the [L,L] A/QK and the [L,d]
# U/O/R live in SRAM and never touch HBM (only the output O is streamed out).
#
# The triangular solve is the Neumann series  U = sum_{k<L} (-A)^k R  (A strictly
# lower => nilpotent) — a chain of plain `tl.dot` matmuls, because classical
# forward substitution (U[i]=R[i]-A[i,:i]·U[:i]) is not expressible in Triton
# (no dynamic tile indexing). Validated exact in fp32: scripts/probe_neumann_solve.py
# (solve alone, rel ~1e-8) and scripts/probe_neumann_scan.py (full pipeline vs
# _scan_sequential, rel ~1e-6) at all decay incl hard alpha=0.1, L=32..256.
#
# Grid (N,): one program owns its full chunk loop, with S_in [d,d] carried in
# registers across chunks (no HBM state carry). Per chunk of length CHUNK:
#   1. load q,k,v,alpha,beta; logP=cumsum(log alpha); P=exp(logP); P_last.
#   2. kbar=k*P, qbar=q*P; R = beta*(v - S_in@kbar).
#   3. A = beta * strict-log-Gram(k,k)   [L,L] resident (strict lower).
#   4. U = Neumann solve (L-1 tl.dot(A,.) matmuls) = (I+A)^{-1} R.
#   5. QK = lower-log-Gram(q,k)  [L,L] (reuses A's SRAM slot).
#   6. O = S_in@qbar + QK@U  ->  store O to HBM.
#   7. S_in = S_in*P_last + U^T@(k * exp(logP_last - logP))   (in registers).

if _HAS_TRITON:

    @triton.jit
    def _mk_fwd_kernel(
        q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, o_ptr,
        s_ptr, nneum_ptr,
        D: tl.constexpr, CHUNK: tl.constexpr, BV: tl.constexpr, BD: tl.constexpr,
    ):
        """Fused chunk-parallel gated-delta scan, forward.

        Grid (N, D//BV): one program per (sequence, d_v-block). The d_v axis is
        tiled by BV so the carried state S_in is [BV, D] (not [D, D]) — holding
        the full [D,D] state plus all [L,D]/[L,L] tiles in one program blows the
        register file and ptxas segfaults. This is the same tiling the sequential
        kernel uses (BV=16). Cost: the [L,L] A/QK Grams (independent of d_v) are
        recomputed per d_v-block (D//BV x redundant) — accepted to fit registers.

        S_in [BV, D] is carried in registers across CHUNK-sized chunks. The chunk
        count and Neumann term count are passed as 1-element TENSORS so Triton
        does not specialize them into compile-time constants (which would unroll
        the loops and crash ptxas on the resulting multi-10K-line PTX).
        """
        pid_n = tl.program_id(0)
        pid_v = tl.program_id(1)
        s_val = tl.load(s_ptr)            # runtime scalar: sequence length S
        nneum = tl.load(nneum_ptr)        # runtime scalar: CHUNK - 1 (Neumann terms)
        base = pid_n * s_val * D          # offset for sequence n in [N,S,D] tensors
        bbase = pid_n * s_val             # offset for sequence n in [N,S] beta

        ri = tl.arange(0, CHUNK)          # chunk-local row indices [L]
        rd = tl.arange(0, D)             # full d_k indices [D]
        offs_v = pid_v * BV + tl.arange(0, BV)   # this program's d_v rows [BV]
        rj = tl.arange(0, CHUNK)          # for the [L,L] Gram tiles
        mstrict = rj[None, :] < ri[:, None]      # [L,L]  j < i  (strict lower)
        mlower = rj[None, :] <= ri[:, None]      # [L,L]  j <= i (lower incl diag)

        S_in = tl.zeros((BV, D), dtype=tl.float32)   # [d_v_block, d_k] carried state

        s = tl.zeros((), dtype=tl.int32)
        while s < s_val:  # rolled chunk loop (S runtime); S % CHUNK == 0 by wrapper
            grow = base + (s + ri)[:, None] * D + rd[None, :]      # [L, D] (full d_k)
            vrow = base + (s + ri)[:, None] * D + offs_v[None, :]  # [L, BV] (this d_v block)

            kc = tl.load(k_ptr + grow).to(tl.float32)            # [L, D]
            qc = tl.load(q_ptr + grow).to(tl.float32)
            ac = tl.load(alpha_ptr + grow).to(tl.float32)
            vc = tl.load(v_ptr + vrow).to(tl.float32)            # [L, BV]
            bc = tl.load(beta_ptr + bbase + s + ri).to(tl.float32)  # [L]

            logA = tl.log(tl.maximum(ac, 1e-30))                 # [L, D]
            logP = tl.cumsum(logA, axis=0)                        # [L, D] inclusive
            P = tl.exp(logP)
            logP_last = tl.sum(logA, axis=0)                      # [D]  (= last row)
            P_last = tl.exp(logP_last)                            # [D]

            kbar = kc * P                                          # [L, D]
            qbar = qc * P                                          # [L, D]
            Sk = tl.dot(kbar, tl.trans(S_in), input_precision="ieee")  # [L, BV]
            R = bc[:, None] * (vc - Sk)                            # [L, BV]

            # --- A = beta * strict log-Gram(k, k)  [L, L] resident ---
            G = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
            for d0 in range(0, D, BD):
                rdb = d0 + tl.arange(0, BD)
                gb = base + (s + ri)[:, None] * D + rdb[None, :]
                kb = tl.load(k_ptr + gb).to(tl.float32)            # [L, BD]
                ab = tl.load(alpha_ptr + gb).to(tl.float32)
                lp = tl.cumsum(tl.log(tl.maximum(ab, 1e-30)), axis=0)  # [L, BD]
                logr = lp[:, None, :] - lp[None, :, :]            # [L, L, BD]
                logr = tl.where(mstrict[:, :, None], logr, float('-inf'))
                G += tl.sum(kb[:, None, :] * kb[None, :, :] * tl.exp(logr), axis=2)
            A = bc[:, None] * G                                    # [L, L] strict lower

            # --- U = sum_{k<L} (-A)^k R  (Neumann; A nilpotent) — rolled ---
            U = R                                                  # term_0 = R
            term = R
            k = tl.zeros((), dtype=tl.int32)
            while k < nneum:
                term = -tl.dot(A, term, input_precision="ieee")   # [L, BV]
                U = U + term
                k = k + 1

            # --- QK = lower log-Gram(q, k)  [L, L] (reuses G's slot) ---
            G = tl.zeros((CHUNK, CHUNK), dtype=tl.float32)
            for d0 in range(0, D, BD):
                rdb = d0 + tl.arange(0, BD)
                gb = base + (s + ri)[:, None] * D + rdb[None, :]
                qb = tl.load(q_ptr + gb).to(tl.float32)            # [L, BD]
                kb = tl.load(k_ptr + gb).to(tl.float32)
                ab = tl.load(alpha_ptr + gb).to(tl.float32)
                lp = tl.cumsum(tl.log(tl.maximum(ab, 1e-30)), axis=0)
                logr = lp[:, None, :] - lp[None, :, :]            # [L, L, BD]
                logr = tl.where(mlower[:, :, None], logr, float('-inf'))
                G += tl.sum(qb[:, None, :] * kb[None, :, :] * tl.exp(logr), axis=2)
            QK = G                                                 # [L, L] lower incl diag

            # --- O = S_in @ qbar + QK @ U ---
            O = (tl.dot(qbar, tl.trans(S_in), input_precision="ieee")
                 + tl.dot(QK, U, input_precision="ieee"))          # [L, BV]
            tl.store(o_ptr + vrow, O.to(o_ptr.dtype.element_ty))

            # --- cross-chunk carry: S_in = S_in*P_last + U^T @ (k*exp(logP_last-logP)) ---
            ratio_c = tl.exp(logP_last[None, :] - logP)           # [L, D]  = P_last/P (<=1)
            k_scaled = kc * ratio_c                                # [L, D]
            S_in = (S_in * P_last[None, :]
                    + tl.dot(tl.trans(U), k_scaled, input_precision="ieee"))  # [BV, D]

            s = s + CHUNK


def megakernel_forward(q, k, v, alpha, beta, chunk: int = 32, bv: int = 16, bd: int = 8):
    """Fused chunk-parallel gated-delta scan, FORWARD only (#96). One Triton
    kernel per (sequence, d_v-block); [L,L] A/QK live in SRAM, U/R/O are tiled
    over d_v (BV). Same math as `_scan_chunked_stable` with the triangular solve
    via the Neumann series (validated exact in fp32). Not autograd-enabled
    (fwd-only); the backward is the next step.

    q, k, v, alpha : [N, S, d]  ;  beta : [N, S, 1]  ->  o : [N, S, d].
    """
    assert _HAS_TRITON and q.is_cuda, "CUDA + triton required"
    N, S, D = q.shape
    assert S % chunk == 0, f"S={S} must be divisible by chunk={chunk}"
    assert D % bv == 0, f"d={D} must be divisible by bv={bv}"
    assert D % bd == 0, f"d={D} must be divisible by bd={bd}"
    q, k, v, alpha = (t.contiguous().float() for t in (q, k, v, alpha))
    beta2 = beta.reshape(N, S).contiguous().float()
    o = torch.empty_like(q)
    # Pass loop bounds as 1-element tensors so Triton does NOT specialize them
    # (specialized int args would unroll the loops and crash ptxas).
    s_t = torch.tensor([S], dtype=torch.int32, device=q.device)
    n_t = torch.tensor([chunk - 1], dtype=torch.int32, device=q.device)
    grid = (N, D // bv)
    _mk_fwd_kernel[grid](q, k, v, alpha, beta2, o, s_t, n_t,
                         D=D, CHUNK=chunk, BV=bv, BD=bd, num_warps=4)
    return o
