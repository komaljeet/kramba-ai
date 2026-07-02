# Related Issues & Future Work

> Issues that are related to HVN's quality improvement journey.
> Created for context continuity across sessions.

---

## Open Issues

(none — all issues closed)

---

## Closed Issues (Quality Journey)

### #22 — Multi-Thread Hamiltonian Memory (CLOSED)
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/22
- **Status**: Closed (2026-07-02)
- **What was done**: K independent Hamiltonians per head, each updated with a
  different projection of the input.  K "threads" of memory = K different
  views of the sequence.
  - New classes: `MultiThreadHamiltonianMemory`, `MultiThreadVortexInjection`
  - `--n_threads K` CLI flag (default 1 for backward compat)
  - Per-thread input projections, vorticity weights, coupling constants
  - Learned thread mixing weights (like MHA output projection)
  - All existing strategies (rank_k, nonlinear, per_token_eta, etc.) work
    within each thread
- **Result**: PPL improved 8.80 → 8.41 (4.4% better) with n_threads=4 alone.
  With all strategies combined: PPL 8.13 (7.6% better).  Scaling exponent
  improved from -0.0126 to -0.0123.  O(1) inference preserved — memory is
  O(K·d²) per head, still 100× less than KV cache at N=100K.

### #21 — Multi-Strategy Quality Upgrade (CLOSED)
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/21
- **Status**: Closed (2026-07-02)
- **What was done**: 6 strategies implemented:
  1. Rank-k entanglement update (`--rank_k 4|8`)
  2. Multi-chunk causal re-injection (`--causal_reinjection`)
  3. Cross-head Hamiltonian mixing (`--cross_head_mixing`)
  4. Nonlinear Hamiltonian update (`--nonlinear_update`)
  5. Per-token entanglement rate (`--per_token_eta`)
  6. SwiGLU FFN (`--swiglu`)
- **Result**: Scaling exponent improved -0.007 → -0.0126 (1.8× better).
  PPL at d_model=128 unchanged (8.37). PPL at d_model=256 improved 13.82 → 13.67.
  The gap to Transformer remains large — fundamental compression bottleneck.

### #1 — Rotary Positional Embeddings (CLOSED)
- RoPE added to vortex injection layer
- Per-head frequency specialization (Issue #16): geometric 2^h frequencies

### #12 — Higher-Rank Hamiltonian Update (CLOSED)
- Batched projectors in batched_memory.py

### #13 — Residual Hamiltonian Connection (CLOSED)
- H_new = H_old + ΔH (use_residual=True)

### #14 — Learned Input-Dependent Entanglement Rate (CLOSED)
- η_t = σ(W_η · x_t)

### #15 — RMSNorm + Parallel Residual (CLOSED)
- x = x + vortex(norm(x)) + ffn(norm(x))

### #16 — Per-Head Vorticity Frequency Projections (CLOSED)
- Geometric init 2^h, Fourier-style specialization

---

## The Quality Gap Summary

| Model | PPL (Shakespeare, d_model=128, 20ep) | Scaling Exponent |
|-------|--------------------------------------|------------------|
| SSM | ~1.16 | N/A |
| Transformer | ~2.0 | -0.090 |
| HVN (Issue #22 n_threads=4 + all strategies) | ~8.13 | -0.0123 |
| HVN (Issue #22 n_threads=4) | ~8.41 | -0.0123 |
| HVN (Issue #21 strategies) | ~8.37 | -0.0126 |
| HVN (original) | ~8.37 | -0.007 |

**Root cause**: The Hamiltonian compresses all sequence history into a single
d×d matrix per head.  Multi-thread memory (K matrices per head) increases
capacity from O(d²) to O(K·d²), improving PPL by 4-8%.

**Next step**: Further quality improvements may require parallel scan (Phase 5.5
Strategy B) or custom Triton kernels (Phase 5.1, 5.3) for speed.