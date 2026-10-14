"""
Custom Triton kernels for HVN's gated delta-rule scan (Issue #83).
====================================================================

Why a custom kernel (not fla): HVN's recurrence is

    S_t = S_{t-1}*diag(alpha_t)*(I - beta_t k_t k_tT) + beta_t v_t k_tT,      o_t = S_t q_t

with **per-channel** decay alpha (a d-vector per head) and scalar write strength beta.
Its low-rank term -beta(alpha*k)kT is *asymmetric*, so fla's fast kernels do not fit
(``gated_delta_rule`` assumes scalar decay -> changes identity; the fast DPLR
kernel assumes b=-a / symmetric low-rank). See #83. The only fla op that matched
our math (``dplr.naive``) is a slow Python loop.

This module implements the recurrence **directly** in Triton -- the same
token-by-token math as :meth:`DeltaNetBlock._scan_sequential`, which is the
ground-truth reference. Because it never forms ``1/P``, it is numerically
**stable at all decay regimes**, unlike the eager ``_scan_chunked`` (which is
correct only for alpha>=0.5 at chunk=32 -- the bug the #79 harness caught).

Milestones (this file grows across commits, each gated by tests/test_issue79.py):
  1. forward kernel  [ok] (this commit) -- validated vs _scan_sequential at all alpha.
  2. backward kernel (training) -- TODO.
  3. chunk-parallel tiling for throughput -- TODO.

Layout: all tensors are [N, S, d] with N = B*n_heads, d = d_head; beta is [N, S].
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
        across the sequential scan over S. fp32 throughout -- no reparameterization,
        so stable at any alpha. Grid = (N, D // BV).
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
        # tensor is NOT kept -- storing it for all layers blows past 6GB VRAM and
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
        # Recompute the entering states here (transient -- freed when this layer's
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
# Chunk-parallel scan (#95) -- stable at all decay, fast at all decay.
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
    """Chunk-parallel gated delta-rule scan -- FORWARD only (#95). Same math as
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
# Fused chunk-parallel scan MEGAKERNEL (#96) -- one kernel, SRAM-resident.
# ===========================================================================
# The hybrid `chunked_scan_forward` above was measured 2-4x SLOWER than the
# sequential Triton kernel at d=64: it launches ~40 kernels/chunk and writes
# the [N,L,L] A/QK to HBM between them. This megakernel fuses the whole chunk
# pipeline into ONE program per sequence so the [L,L] A/QK and the [L,d]
# U/O/R live in SRAM and never touch HBM (only the output O is streamed out).
#
# The triangular solve is the Neumann series  U = sum_{k<L} (-A)^k R  (A strictly
# lower => nilpotent) -- a chain of plain `tl.dot` matmuls, because classical
# forward substitution (U[i]=R[i]-A[i,:i]*U[:i]) is not expressible in Triton
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
        tiled by BV so the carried state S_in is [BV, D] (not [D, D]) -- holding
        the full [D,D] state plus all [L,D]/[L,L] tiles in one program blows the
        register file and ptxas segfaults. This is the same tiling the sequential
        kernel uses (BV=16). Cost: the [L,L] A/QK Grams (independent of d_v) are
        recomputed per d_v-block (D//BV x redundant) -- accepted to fit registers.

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

            # --- U = sum_{k<L} (-A)^k R  (Neumann; A nilpotent) -- rolled ---
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


# ============================================================================
# Design B (#96): fused blockwise-recompute BACKWARD.
#
# Design A (chunk-parallel fwd megakernel) was killed by the speed gate -- its
# Neumann triangular solve is L-1 serial tl.dot MMAs, heavier than the
# sequential kernel's pipelined elementwise steps. Design B attacks a DIFFERENT
# bottleneck: the backward's HBM traffic. The current backward is 2 kernels:
#   _gd_fwd_states_kernel : forward recompute, writes [N,S,D,D] entering states
#                          (268 MB at N32 S512 d64) to HBM;
#   _gd_bwd_kernel        : reverse adjoint, reads those 268 MB back.
# Measured (scripts/probe_bwd_hbm.py) the backward runs at ~80% of peak HBM
# bandwidth -- it is memory-bound, dominated by the states roundtrip.
#
# Design B replaces that with blockwise recompute (flash-attention style):
#   _mk_ckpt_kernel : forward recompute, writes only block-BOUNDARY entering
#                    states [N, S/B, D, D] (67 MB at B=4 -- 4x less);
#   _mk_bwd_kernel  : reverse adjoint; per block, load the block's checkpoint
#                    state and RECOMPUTE the B entering states in SRAM, then
#                    run the (unchanged) per-step adjoint. Same compute as the
#                    2-kernel path, ~4x less states HBM.
#
# The adjoint body is VERBATIM from _gd_bwd_kernel (validated vs autograd at
# 1e-16 in fp64) -- only the Sprev source changes (SRAM recompute, not the full
# HBM states array). The forward is the unchanged optimal _gated_delta_fwd_kernel.
# ============================================================================

if _HAS_TRITON:

    @triton.jit
    def _mk_ckpt_kernel(
        k_ptr, v_ptr, alpha_ptr, beta_ptr, ckpt_ptr, nblk,
        S: tl.constexpr, D: tl.constexpr, B: tl.constexpr,
    ):
        """Forward recompute that checkpoints only block-BOUNDARY entering
        states. One program per sequence; full [D,D] state. ckpt layout:
        [N, nblocks, D(d_v), D(d_k)] where ckpt[b] = state before step b*B
        (the entering state of block b). nblocks is a runtime int (rolled
        block loop); B is constexpr (unrolled inner step loop)."""
        n = tl.program_id(0)
        base = n * S * D
        bbase = n * S
        cbase = n * nblk * D * D
        rk = tl.arange(0, D)
        ov = tl.arange(0, D)[:, None]
        ok = tl.arange(0, D)[None, :]
        state = tl.zeros((D, D), dtype=tl.float32)
        b = tl.zeros((), tl.int32)
        while b < nblk:
            # checkpoint the entering state of this block (before its 1st step)
            tl.store(ckpt_ptr + cbase + b * D * D + ov * D + ok, state)
            for j in range(B):  # unrolled (B constexpr)
                t = b * B + j
                row = base + t * D
                k_t = tl.load(k_ptr + row + rk).to(tl.float32)
                a_t = tl.load(alpha_ptr + row + rk).to(tl.float32)
                v_t = tl.load(v_ptr + row + rk).to(tl.float32)
                b_t = tl.load(beta_ptr + bbase + t).to(tl.float32)
                state = state * a_t[None, :]
                Sk = tl.sum(state * k_t[None, :], axis=1)
                u = v_t - Sk
                state = state + b_t * (u[:, None] * k_t[None, :])
            b = b + 1

    @triton.jit
    def _mk_bwd_kernel(
        q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, ckpt_ptr, do_ptr,
        dq_ptr, dk_ptr, dv_ptr, dalpha_ptr, dbeta_ptr, nblk,
        S: tl.constexpr, D: tl.constexpr, B: tl.constexpr,
    ):
        """Reverse-time adjoint with blockwise SRAM recompute. One program per
        sequence; full [D,D] state grad carried backward. For each block (in
        reverse): load the block's checkpoint entering state, recompute each
        step's entering state in SRAM (O(B^2) on-the-fly recompute -- no [B,D,D]
        buffer, since Triton cannot index a tile by a runtime step index), then
        run the validated per-step adjoint. The adjoint body is identical to
        _gd_bwd_kernel; only Sprev comes from recompute instead of HBM."""
        n = tl.program_id(0)
        base = n * S * D
        bbase = n * S
        cbase = n * nblk * D * D
        rk = tl.arange(0, D)
        rv = tl.arange(0, D)
        ov = tl.arange(0, D)[:, None]
        ok = tl.arange(0, D)[None, :]
        dS = tl.zeros((D, D), dtype=tl.float32)
        b = nblk - 1
        while b >= 0:
            s = b * B
            S_start = tl.load(ckpt_ptr + cbase + b * D * D + ov * D + ok)  # [D,D]
            for j in range(B - 1, -1, -1):  # unrolled (B constexpr), j = B-1..0
                t = s + j
                row = base + t * D
                # recompute Sprev = entering state of step t = S_start advanced
                # j steps. (j constexpr from the unrolled outer loop, so range(j)
                # unrolls too; empty when j == 0.)
                Sprev = S_start
                for jj in range(j):
                    tt = s + jj
                    rrow = base + tt * D
                    k_tt = tl.load(k_ptr + rrow + rk).to(tl.float32)
                    a_tt = tl.load(alpha_ptr + rrow + rk).to(tl.float32)
                    v_tt = tl.load(v_ptr + rrow + rk).to(tl.float32)
                    b_tt = tl.load(beta_ptr + bbase + tt).to(tl.float32)
                    Sprev = Sprev * a_tt[None, :]
                    Skk = tl.sum(Sprev * k_tt[None, :], axis=1)
                    uu = v_tt - Skk
                    Sprev = Sprev + b_tt * (uu[:, None] * k_tt[None, :])
                # --- validated adjoint body (verbatim from _gd_bwd_kernel) ---
                k_t = tl.load(k_ptr + row + rk).to(tl.float32)
                q_t = tl.load(q_ptr + row + rk).to(tl.float32)
                a_t = tl.load(alpha_ptr + row + rk).to(tl.float32)
                v_t = tl.load(v_ptr + row + rv).to(tl.float32)
                b_t = tl.load(beta_ptr + bbase + t).to(tl.float32)
                do_t = tl.load(do_ptr + row + rv).to(tl.float32)
                Sd = Sprev * a_t[None, :]
                Sk = tl.sum(Sd * k_t[None, :], axis=1)
                u = v_t - Sk
                S_t = Sd + b_t * (u[:, None] * k_t[None, :])
                dS_t = dS + do_t[:, None] * q_t[None, :]
                dq_t = tl.sum(S_t * do_t[:, None], axis=0)
                tl.store(dq_ptr + row + rk, dq_t)
                dSd = dS_t
                dSk_via = tl.sum(dS_t * k_t[None, :], axis=1)
                dbeta_t = tl.sum(u * dSk_via)
                tl.store(dbeta_ptr + bbase + t, dbeta_t)
                du = b_t * dSk_via
                dk_t = b_t * tl.sum(dS_t * u[:, None], axis=0)
                tl.store(dv_ptr + row + rv, du)
                dSd = dSd - du[:, None] * k_t[None, :]
                dk_t = dk_t - tl.sum(Sd * du[:, None], axis=0)
                tl.store(dk_ptr + row + rk, dk_t)
                dalpha_t = tl.sum(Sprev * dSd, axis=0)
                tl.store(dalpha_ptr + row + rk, dalpha_t)
                dS = dSd * a_t[None, :]
                # --- end adjoint body ---
            b = b - 1

    @triton.jit
    def _mk_bwd_kernel_buf(
        q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, ckpt_ptr, do_ptr,
        dq_ptr, dk_ptr, dv_ptr, dalpha_ptr, dbeta_ptr, nblk,
        S: tl.constexpr, D: tl.constexpr, B: tl.constexpr,
    ):
        """Buffered variant of _mk_bwd_kernel: O(B) recompute instead of O(B^2).
        Per block, ONE forward pass buffers the B entering states in SRAM (a
        [B,D,D] tile, written/read via tl.where slot-masking with constexpr j --
        Triton can't index a [D,D] tile by a runtime step), then ONE reverse
        adjoint pass reads them. Cuts the recompute compute + redundant input
        reads that capped the O(B^2) kernel at ~1.19x; trades B*16 KB of SRAM
        (B=4 -> 64 KB) -- the register-pressure risk. Adjoint body identical."""
        n = tl.program_id(0)
        base = n * S * D
        bbase = n * S
        cbase = n * nblk * D * D
        rk = tl.arange(0, D)
        rv = tl.arange(0, D)
        ov = tl.arange(0, D)[:, None]
        ok = tl.arange(0, D)[None, :]
        idx = tl.arange(0, B)[:, None, None]   # [B,1,1] slot selector
        dS = tl.zeros((D, D), dtype=tl.float32)
        b = nblk - 1
        while b >= 0:
            s = b * B
            S_start = tl.load(ckpt_ptr + cbase + b * D * D + ov * D + ok)  # [D,D]
            # --- forward pass: buffer the B entering states (O(B) recompute) ---
            buf = tl.where(idx == 0, S_start[None, :, :], tl.zeros((B, D, D), dtype=tl.float32))
            Sprev = S_start
            for jj in range(B - 1):  # unrolled; fills slots 1..B-1
                tt = s + jj
                rrow = base + tt * D
                k_tt = tl.load(k_ptr + rrow + rk).to(tl.float32)
                a_tt = tl.load(alpha_ptr + rrow + rk).to(tl.float32)
                v_tt = tl.load(v_ptr + rrow + rk).to(tl.float32)
                b_tt = tl.load(beta_ptr + bbase + tt).to(tl.float32)
                Sprev = Sprev * a_tt[None, :]
                Skk = tl.sum(Sprev * k_tt[None, :], axis=1)
                uu = v_tt - Skk
                Sprev = Sprev + b_tt * (uu[:, None] * k_tt[None, :])
                buf = tl.where(idx == (jj + 1), Sprev[None, :, :], buf)
            # --- reverse adjoint: read entering states from the buffer ---
            for j in range(B - 1, -1, -1):  # unrolled, j = B-1..0
                t = s + j
                row = base + t * D
                Sprev = tl.sum(tl.where(idx == j, buf, 0.0), axis=0)  # [D,D]
                # --- validated adjoint body (verbatim from _gd_bwd_kernel) ---
                k_t = tl.load(k_ptr + row + rk).to(tl.float32)
                q_t = tl.load(q_ptr + row + rk).to(tl.float32)
                a_t = tl.load(alpha_ptr + row + rk).to(tl.float32)
                v_t = tl.load(v_ptr + row + rv).to(tl.float32)
                b_t = tl.load(beta_ptr + bbase + t).to(tl.float32)
                do_t = tl.load(do_ptr + row + rv).to(tl.float32)
                Sd = Sprev * a_t[None, :]
                Sk = tl.sum(Sd * k_t[None, :], axis=1)
                u = v_t - Sk
                S_t = Sd + b_t * (u[:, None] * k_t[None, :])
                dS_t = dS + do_t[:, None] * q_t[None, :]
                dq_t = tl.sum(S_t * do_t[:, None], axis=0)
                tl.store(dq_ptr + row + rk, dq_t)
                dSd = dS_t
                dSk_via = tl.sum(dS_t * k_t[None, :], axis=1)
                dbeta_t = tl.sum(u * dSk_via)
                tl.store(dbeta_ptr + bbase + t, dbeta_t)
                du = b_t * dSk_via
                dk_t = b_t * tl.sum(dS_t * u[:, None], axis=0)
                tl.store(dv_ptr + row + rv, du)
                dSd = dSd - du[:, None] * k_t[None, :]
                dk_t = dk_t - tl.sum(Sd * du[:, None], axis=0)
                tl.store(dk_ptr + row + rk, dk_t)
                dalpha_t = tl.sum(Sprev * dSd, axis=0)
                tl.store(dalpha_ptr + row + rk, dalpha_t)
                dS = dSd * a_t[None, :]
                # --- end adjoint body ---
            b = b - 1


class _MegaScan(torch.autograd.Function):
    """Design B autograd wrapper: optimal sequential forward (unchanged
    `_gated_delta_fwd_kernel`) + fused blockwise-recompute backward.

    `.apply` takes ONLY tensor args (q,k,v,alpha,beta) -- matching
    `_GatedDeltaScan` -- so it composes with `torch.compile` (AOT-autograd treats
    an all-tensor custom Function the same as the working sequential scan;
    non-tensor kwargs break inductor's backward codegen). `block` and `buffered`
    are class attributes read in backward (baked in as constants at AOT trace
    time); set them via `megakernel_scan(block=..., buffered=...)` BEFORE the
    compiled forward is traced (i.e. once, at model build), not per-step.
    Defaults: block=4 (the measured sweet spot), buffered=False (the O(B^2)
    winner; the O(B) buffered variant is a documented dead end)."""

    block = 4
    buffered = False

    @staticmethod
    def forward(ctx, q, k, v, alpha, beta):
        N, S, D = q.shape
        q, k, v, alpha = (t.contiguous().float() for t in (q, k, v, alpha))
        beta2 = beta.reshape(N, S).contiguous().float()
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
        B = _MegaScan.block
        assert S % B == 0, f"S={S} must be divisible by block B={B}"
        nblocks = S // B
        do = do.contiguous().float()
        # nblocks goes to the kernels as a plain runtime int. It used to be
        # wrapped in a 1-element CUDA tensor, i.e. a pageable host->device copy
        # that synchronises the stream -- once per DeltaNet block per backward,
        # draining the launch queue every time. A scalar kernel arg is not a
        # copy and is still a runtime value, so the block loop stays rolled.
        ckpt = torch.empty(N, nblocks, D, D, device=q.device, dtype=torch.float32)
        _mk_ckpt_kernel[(N,)](k, v, alpha, beta2, ckpt, nblocks, S=S, D=D, B=B)
        dq = torch.empty_like(q); dk = torch.empty_like(q)
        dv = torch.empty_like(q); dalpha = torch.empty_like(q)
        dbeta2 = torch.empty_like(beta2)
        bwd = _mk_bwd_kernel_buf if _MegaScan.buffered else _mk_bwd_kernel
        bwd[(N,)](q, k, v, alpha, beta2, ckpt, do,
                  dq, dk, dv, dalpha, dbeta2, nblocks, S=S, D=D, B=B)
        return dq, dk, dv, dalpha, dbeta2.reshape(ctx.beta_shape)


# ============================================================================
# Strided-IO variant (2026-10-14): read q/k/v/alpha/beta straight from the
# block's [B, S, H, d] projections, in their own dtype.
# ============================================================================
# `_MegaScan` takes [N, S, d] fp32: the caller permutes [B,S,H,d] -> [N,S,d]
# (a copy) and the wrapper casts bf16 -> fp32 (a second copy), and on the way
# back autograd casts dq/dk/dv fp32 -> bf16 and permutes again. The kernels
# already upcast every load with `.to(tl.float32)`, so those copies are pure
# layout/dtype prep. Here the kernels address the [B,S,H,d] tensors directly
#     n = b*H + h;  base = b*sB + h*sH;  row = base + t*sS
# and store dq/dk/dv/dalpha/dbeta in the inputs' dtypes, in contiguous
# [B,S,H,d] (the layout the l2norm / projection backward wants). `o` stays the
# fp32 [N,S,d] the fused read-norm consumes. The math -- every load, every
# fp32 op, in order -- is the validated kernels' verbatim; only addresses and
# the final store dtype change. Selected by DeltaNetBlock.use_fused_scan_io.
# ============================================================================

if _HAS_TRITON:

    @triton.jit
    def _gated_delta_fwd_kernel_s(
        q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, o_ptr,
        H, sB, sS, sH,
        S: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
    ):
        """`_gated_delta_fwd_kernel` reading [B,S,H,D] inputs by stride.
        beta is contiguous [B,S,H]; o is contiguous fp32 [B*H, S, D]."""
        pid_n = tl.program_id(0)
        pid_v = tl.program_id(1)
        bi = pid_n // H
        hi = pid_n % H

        offs_k = tl.arange(0, D)
        offs_v = pid_v * BV + tl.arange(0, BV)

        base = bi * sB + hi * sH
        obase = pid_n * S * D

        state = tl.zeros((BV, D), dtype=tl.float32)

        for t in range(S):
            row = base + t * sS
            k_t = tl.load(k_ptr + row + offs_k).to(tl.float32)
            q_t = tl.load(q_ptr + row + offs_k).to(tl.float32)
            a_t = tl.load(alpha_ptr + row + offs_k).to(tl.float32)
            v_t = tl.load(v_ptr + row + offs_v).to(tl.float32)
            b_t = tl.load(beta_ptr + (bi * S + t) * H + hi).to(tl.float32)

            state = state * a_t[None, :]
            Sk = tl.sum(state * k_t[None, :], axis=1)
            u = v_t - Sk
            state = state + b_t * (u[:, None] * k_t[None, :])
            o_t = tl.sum(state * q_t[None, :], axis=1)

            tl.store(o_ptr + obase + t * D + offs_v, o_t.to(o_ptr.dtype.element_ty))

    @triton.jit
    def _mk_ckpt_kernel_s(
        k_ptr, v_ptr, alpha_ptr, beta_ptr, ckpt_ptr, nblk,
        H, sB, sS, sH,
        S: tl.constexpr, D: tl.constexpr, B: tl.constexpr,
    ):
        """`_mk_ckpt_kernel` reading [B,S,H,D] inputs by stride."""
        n = tl.program_id(0)
        bi = n // H
        hi = n % H
        base = bi * sB + hi * sH
        cbase = n * nblk * D * D
        rk = tl.arange(0, D)
        ov = tl.arange(0, D)[:, None]
        ok = tl.arange(0, D)[None, :]
        state = tl.zeros((D, D), dtype=tl.float32)
        b = tl.zeros((), tl.int32)
        while b < nblk:
            tl.store(ckpt_ptr + cbase + b * D * D + ov * D + ok, state)
            for j in range(B):
                t = b * B + j
                row = base + t * sS
                k_t = tl.load(k_ptr + row + rk).to(tl.float32)
                a_t = tl.load(alpha_ptr + row + rk).to(tl.float32)
                v_t = tl.load(v_ptr + row + rk).to(tl.float32)
                b_t = tl.load(beta_ptr + (bi * S + t) * H + hi).to(tl.float32)
                state = state * a_t[None, :]
                Sk = tl.sum(state * k_t[None, :], axis=1)
                u = v_t - Sk
                state = state + b_t * (u[:, None] * k_t[None, :])
            b = b + 1

    @triton.jit
    def _mk_bwd_kernel_s(
        q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, ckpt_ptr, do_ptr,
        dq_ptr, dk_ptr, dv_ptr, dalpha_ptr, dbeta_ptr, nblk,
        H, sB, sS, sH,
        S: tl.constexpr, D: tl.constexpr, B: tl.constexpr,
    ):
        """`_mk_bwd_kernel` reading [B,S,H,D] inputs by stride. Gradients go
        to contiguous [B,S,H,D] (dbeta: [B,S,H]) in each pointer's dtype; do
        is contiguous [B*H, S, D]."""
        n = tl.program_id(0)
        bi = n // H
        hi = n % H
        base = bi * sB + hi * sH
        gbase = (bi * S * H + hi) * D       # contiguous [B,S,H,D] grads
        obase = n * S * D                   # contiguous [N,S,D] do
        cbase = n * nblk * D * D
        rk = tl.arange(0, D)
        rv = tl.arange(0, D)
        ov = tl.arange(0, D)[:, None]
        ok = tl.arange(0, D)[None, :]
        dS = tl.zeros((D, D), dtype=tl.float32)
        b = nblk - 1
        while b >= 0:
            s = b * B
            S_start = tl.load(ckpt_ptr + cbase + b * D * D + ov * D + ok)
            for j in range(B - 1, -1, -1):
                t = s + j
                row = base + t * sS
                grow = gbase + t * H * D
                bidx = (bi * S + t) * H + hi
                Sprev = S_start
                for jj in range(j):
                    tt = s + jj
                    rrow = base + tt * sS
                    k_tt = tl.load(k_ptr + rrow + rk).to(tl.float32)
                    a_tt = tl.load(alpha_ptr + rrow + rk).to(tl.float32)
                    v_tt = tl.load(v_ptr + rrow + rk).to(tl.float32)
                    b_tt = tl.load(beta_ptr + (bi * S + tt) * H + hi).to(tl.float32)
                    Sprev = Sprev * a_tt[None, :]
                    Skk = tl.sum(Sprev * k_tt[None, :], axis=1)
                    uu = v_tt - Skk
                    Sprev = Sprev + b_tt * (uu[:, None] * k_tt[None, :])
                # --- validated adjoint body (verbatim from _mk_bwd_kernel) ---
                k_t = tl.load(k_ptr + row + rk).to(tl.float32)
                q_t = tl.load(q_ptr + row + rk).to(tl.float32)
                a_t = tl.load(alpha_ptr + row + rk).to(tl.float32)
                v_t = tl.load(v_ptr + row + rv).to(tl.float32)
                b_t = tl.load(beta_ptr + bidx).to(tl.float32)
                do_t = tl.load(do_ptr + obase + t * D + rv).to(tl.float32)
                Sd = Sprev * a_t[None, :]
                Sk = tl.sum(Sd * k_t[None, :], axis=1)
                u = v_t - Sk
                S_t = Sd + b_t * (u[:, None] * k_t[None, :])
                dS_t = dS + do_t[:, None] * q_t[None, :]
                dq_t = tl.sum(S_t * do_t[:, None], axis=0)
                tl.store(dq_ptr + grow + rk, dq_t.to(dq_ptr.dtype.element_ty))
                dSd = dS_t
                dSk_via = tl.sum(dS_t * k_t[None, :], axis=1)
                dbeta_t = tl.sum(u * dSk_via)
                tl.store(dbeta_ptr + bidx, dbeta_t.to(dbeta_ptr.dtype.element_ty))
                du = b_t * dSk_via
                dk_t = b_t * tl.sum(dS_t * u[:, None], axis=0)
                tl.store(dv_ptr + grow + rv, du.to(dv_ptr.dtype.element_ty))
                dSd = dSd - du[:, None] * k_t[None, :]
                dk_t = dk_t - tl.sum(Sd * du[:, None], axis=0)
                tl.store(dk_ptr + grow + rk, dk_t.to(dk_ptr.dtype.element_ty))
                dalpha_t = tl.sum(Sprev * dSd, axis=0)
                tl.store(dalpha_ptr + grow + rk, dalpha_t.to(dalpha_ptr.dtype.element_ty))
                dS = dSd * a_t[None, :]
                # --- end adjoint body ---
            b = b - 1


SCAN_IO_STATS = {"fallback_copies": 0}


class _MegaScanBSHD(torch.autograd.Function):
    """`_MegaScan` on [B,S,H,d] inputs in their own dtype (no permute/cast
    copies). Same kernels' math, same `_MegaScan.block`. Returns o as fp32
    [B*H, S, d] -- exactly what `_MegaScan` returns for the permuted inputs."""

    @staticmethod
    def forward(ctx, q, k, v, alpha, beta):
        Bt, S, H, D = q.shape
        ts = [t if t.stride(-1) == 1 else t.contiguous() for t in (q, k, v, alpha)]
        # One stride set is passed to the kernels (a size-1 dim's stride is
        # never used). Mixed layouts -- never the block's own -- fall back to
        # contiguous copies, counted so a test can assert the block avoids it.
        if len({tuple(st if sz > 1 else 0 for st, sz in zip(t.stride(), t.shape))
                for t in ts}) != 1:
            SCAN_IO_STATS["fallback_copies"] += 1
            ts = [t.contiguous() for t in ts]
        q, k, v, alpha = ts
        beta3 = beta.reshape(Bt, S, H)
        if not beta3.is_contiguous():
            beta3 = beta3.contiguous()
        sB, sS, sH, _ = q.stride()
        o = torch.empty(Bt * H, S, D, device=q.device, dtype=torch.float32)
        BV = 16 if D % 16 == 0 else 1
        _gated_delta_fwd_kernel_s[(Bt * H, D // BV)](
            q, k, v, alpha, beta3, o, H, sB, sS, sH, S=S, D=D, BV=BV)
        ctx.save_for_backward(q, k, v, alpha, beta3)
        ctx.beta_shape = beta.shape
        return o

    @staticmethod
    def backward(ctx, do):
        q, k, v, alpha, beta3 = ctx.saved_tensors
        Bt, S, H, D = q.shape
        N = Bt * H
        B = _MegaScan.block
        assert S % B == 0, f"S={S} must be divisible by block B={B}"
        nblocks = S // B
        do = do.contiguous().float()
        sB, sS, sH, _ = q.stride()
        ckpt = torch.empty(N, nblocks, D, D, device=q.device, dtype=torch.float32)
        _mk_ckpt_kernel_s[(N,)](k, v, alpha, beta3, ckpt, nblocks, H, sB, sS, sH,
                                S=S, D=D, B=B)
        shp = (Bt, S, H, D)
        dq = torch.empty(shp, device=q.device, dtype=q.dtype)
        dk = torch.empty(shp, device=q.device, dtype=k.dtype)
        dv = torch.empty(shp, device=q.device, dtype=v.dtype)
        dalpha = torch.empty(shp, device=q.device, dtype=alpha.dtype)
        dbeta3 = torch.empty_like(beta3)
        _mk_bwd_kernel_s[(N,)](q, k, v, alpha, beta3, ckpt, do,
                               dq, dk, dv, dalpha, dbeta3, nblocks, H, sB, sS, sH,
                               S=S, D=D, B=B)
        return dq, dk, dv, dalpha, dbeta3.reshape(ctx.beta_shape)


def megakernel_scan_bshd(q, k, v, alpha, beta):
    """Strided-IO megakernel scan. q, k, v, alpha : [B, S, H, d] (any float
    dtype, last dim unit-stride); beta : [B, S, H, 1]. Returns o : fp32
    [B*H, S, d], bit-identical to `megakernel_scan` on the permuted, fp32-cast
    inputs; gradients come back in each input's dtype."""
    assert _HAS_TRITON and q.is_cuda, "CUDA + triton required"
    assert not _MegaScan.buffered, "strided IO implements the O(B^2) backward only"
    return _MegaScanBSHD.apply(q, k, v, alpha, beta)


def megakernel_scan(q, k, v, alpha, beta, block: int | None = None,
                    buffered: bool | None = None):
    """Design B: optimal sequential forward + fused blockwise-recompute
    backward. Drop-in for `gated_delta_scan` (autograd-enabled, 5 tensor args).

    `block`/`buffered` override the class defaults (block=4, buffered=False) and
    are set as class attributes on `_MegaScan` -- pass them ONLY for eager
    micro-benchmarking (e.g. the B-sweep). The compiled training path must call
    this with no overrides (defaults), so the class attrs are stable at AOT
    trace time. block=4 is the measured sweet spot; buffered=False is the O(B^2)
    winner (the O(B) buffered variant is a documented dead end -- Triton can't
    cheaply index a [B,D,D] tile, so the per-step slot read costs a full-buffer
    masked reduction that offsets the O(B) saving, plus register pressure)."""
    if block is not None:
        _MegaScan.block = block
    if buffered is not None:
        _MegaScan.buffered = buffered
    return _MegaScan.apply(q, k, v, alpha, beta)
