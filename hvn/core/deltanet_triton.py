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
