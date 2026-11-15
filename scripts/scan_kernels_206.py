"""#206 study kernels: layout variants of the gated delta-rule scan (not wired into hvn/).

Author: Rambarun Komaljeet

The production kernels hold the state as [d_v, d_k] and reduce over d_k (axis 1). With
Triton's default blocked layout the 32 lanes of a warp lie along the last axis, so every
per-step reduction over d_k is a cross-lane shuffle tree (5 stages per row) and, with
more than one warp, a shared-memory round trip with a barrier.

Here the state is held TRANSPOSED, [d_k, d_v]: lanes lie along d_v and each thread keeps
a whole d_k column in registers, so the forward's two reductions over d_k are in-thread
adds (no shuffles, no barriers). Same recurrence, different summation order, so the
results are not bit-identical to production (checked by the study script).

  fwdT_kernel   forward; optional WRITE_O (output) and WRITE_CKPT (block-boundary
                entering states, the backward's checkpoint), so one kernel can be the
                plain forward, the checkpoint pass, or the fused recompute forward.
  bwdT_kernel   backward adjoint with the same O(B^2) blockwise recompute as
                `_mk_bwd_kernel_s`, transposed layout, d_v optionally split over
                programs (SPLIT > 1 writes partial dq/dk/dalpha/dbeta that are summed).
"""
import triton
import triton.language as tl


@triton.jit
def fwdT_kernel(
    q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, o_ptr, ckpt_ptr, nblk,
    H, sB, sS, sH,
    S: tl.constexpr, D: tl.constexpr, BV: tl.constexpr, B: tl.constexpr,
    WRITE_O: tl.constexpr, WRITE_CKPT: tl.constexpr,
):
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)
    bi = pid_n // H
    hi = pid_n % H
    offs_k = tl.arange(0, D)                      # d_k (axis 0)
    offs_v = pid_v * BV + tl.arange(0, BV)        # d_v (axis 1)
    base = bi * sB + hi * sH
    obase = pid_n * S * D
    cbase = pid_n * nblk * D * D
    st = tl.zeros((D, BV), dtype=tl.float32)      # S^T : [d_k, d_v_block]
    blk = 0
    while blk < nblk:
        if WRITE_CKPT:
            # ckpt layout [N, nblk, d_v, d_k] as in production
            tl.store(ckpt_ptr + cbase + blk * D * D + offs_v[None, :] * D + offs_k[:, None], st)
        for j in tl.static_range(B):
            t = blk * B + j
            row = base + t * sS
            k_t = tl.load(k_ptr + row + offs_k).to(tl.float32)
            a_t = tl.load(alpha_ptr + row + offs_k).to(tl.float32)
            v_t = tl.load(v_ptr + row + offs_v).to(tl.float32)
            b_t = tl.load(beta_ptr + (bi * S + t) * H + hi).to(tl.float32)
            st = st * a_t[:, None]
            Sk = tl.sum(st * k_t[:, None], axis=0)            # [BV]
            u = v_t - Sk
            st = st + b_t * (k_t[:, None] * u[None, :])
            if WRITE_O:
                q_t = tl.load(q_ptr + row + offs_k).to(tl.float32)
                o_t = tl.sum(st * q_t[:, None], axis=0)
                tl.store(o_ptr + obase + t * D + offs_v, o_t)
        blk += 1


@triton.jit
def bwdT_kernel(
    q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, ckpt_ptr, do_ptr,
    dq_ptr, dk_ptr, dv_ptr, dalpha_ptr, dbeta_ptr, nblk,
    H, sB, sS, sH,
    S: tl.constexpr, D: tl.constexpr, B: tl.constexpr, BV: tl.constexpr,
    SPLIT: tl.constexpr,
):
    """Transposed-layout adjoint. Program (n, pid_v) owns d_v rows pid_v*BV..+BV of
    the state and its gradient. dv is exact per row. dq/dk/dalpha (sums over d_v)
    and dbeta (sum over d_v) are written as fp32 partials [SPLIT, ...] when SPLIT > 1,
    else stored directly in the gradient dtype."""
    n = tl.program_id(0)
    pid_v = tl.program_id(1)
    bi = n // H
    hi = n % H
    base = bi * sB + hi * sH
    gbase = (bi * S * H + hi) * D
    obase = n * S * D
    cbase = n * nblk * D * D
    rk = tl.arange(0, D)
    rv = pid_v * BV + tl.arange(0, BV)
    # partial-output offsets: [SPLIT, B*S*H*D] / [SPLIT, B*S*H]
    pofs = pid_v * (S * H * D) * (tl.num_programs(0) // H)
    pbofs = pid_v * (S * H) * (tl.num_programs(0) // H)
    dS = tl.zeros((D, BV), dtype=tl.float32)        # dS^T [d_k, d_v_block]
    b = nblk - 1
    while b >= 0:
        s = b * B
        S_start = tl.load(ckpt_ptr + cbase + b * D * D + rv[None, :] * D + rk[:, None])  # [dk, dv]
        for j in tl.static_range(B - 1, -1, -1):
            t = s + j
            row = base + t * sS
            grow = gbase + t * H * D
            bidx = (bi * S + t) * H + hi
            Sprev = S_start
            for jj in tl.static_range(j):
                tt = s + jj
                rrow = base + tt * sS
                k_tt = tl.load(k_ptr + rrow + rk).to(tl.float32)
                a_tt = tl.load(alpha_ptr + rrow + rk).to(tl.float32)
                v_tt = tl.load(v_ptr + rrow + rv).to(tl.float32)
                b_tt = tl.load(beta_ptr + (bi * S + tt) * H + hi).to(tl.float32)
                Sprev = Sprev * a_tt[:, None]
                Skk = tl.sum(Sprev * k_tt[:, None], axis=0)
                uu = v_tt - Skk
                Sprev = Sprev + b_tt * (k_tt[:, None] * uu[None, :])
            k_t = tl.load(k_ptr + row + rk).to(tl.float32)
            q_t = tl.load(q_ptr + row + rk).to(tl.float32)
            a_t = tl.load(alpha_ptr + row + rk).to(tl.float32)
            v_t = tl.load(v_ptr + row + rv).to(tl.float32)
            b_t = tl.load(beta_ptr + bidx).to(tl.float32)
            do_t = tl.load(do_ptr + obase + t * D + rv).to(tl.float32)
            Sd = Sprev * a_t[:, None]
            Sk = tl.sum(Sd * k_t[:, None], axis=0)                 # [dv]
            u = v_t - Sk
            S_t = Sd + b_t * (k_t[:, None] * u[None, :])
            dS_t = dS + q_t[:, None] * do_t[None, :]
            dq_t = tl.sum(S_t * do_t[None, :], axis=1)             # [dk]
            dSk_via = tl.sum(dS_t * k_t[:, None], axis=0)          # [dv]
            dbeta_t = tl.sum(u * dSk_via)
            du = b_t * dSk_via
            dk_t = b_t * tl.sum(dS_t * u[None, :], axis=1)
            dSd = dS_t - k_t[:, None] * du[None, :]
            dk_t = dk_t - tl.sum(Sd * du[None, :], axis=1)
            dalpha_t = tl.sum(Sprev * dSd, axis=1)
            tl.store(dv_ptr + grow + rv, du.to(dv_ptr.dtype.element_ty))
            if SPLIT == 1:
                tl.store(dq_ptr + grow + rk, dq_t.to(dq_ptr.dtype.element_ty))
                tl.store(dk_ptr + grow + rk, dk_t.to(dk_ptr.dtype.element_ty))
                tl.store(dalpha_ptr + grow + rk, dalpha_t.to(dalpha_ptr.dtype.element_ty))
                tl.store(dbeta_ptr + bidx, dbeta_t.to(dbeta_ptr.dtype.element_ty))
            else:
                tl.store(dq_ptr + pofs + grow + rk, dq_t)
                tl.store(dk_ptr + pofs + grow + rk, dk_t)
                tl.store(dalpha_ptr + pofs + grow + rk, dalpha_t)
                tl.store(dbeta_ptr + pbofs + bidx, dbeta_t)
            dS = dSd * a_t[:, None]
        b = b - 1
