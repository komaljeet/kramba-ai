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
    """Triton forward of HVN's gated delta-rule scan.

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
