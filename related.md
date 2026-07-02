# Related Issues & Future Work

> Issues that are related to HVN's quality improvement journey.
> Created for context continuity across sessions.

---

## Open Issues

### #22 — Multi-Thread Hamiltonian Memory (NEXT)
- **GitLab**: https://gitlab.com/komalbarun/kramba-ai/-/work_items/22
- **Status**: Open
- **Goal**: Maintain K independent Hamiltonians per head, each tracking different
  patterns. K "threads" of memory = K different views of the sequence.
- **Key idea**: One d×d matrix stores one view; K matrices store K views —
  like K people taking notes from different angles.
- **Expected impact**: PPL ~8.4 → target < 6.0; scaling exponent -0.0126 → < -0.03
- **Preserves**: O(1) inference, no KV cache, < 20% Transformer FLOPs, 6GB VRAM
- **Files to modify**: `batched_memory.py`, `batched_vortex.py`, `hvn_block.py`,
  `model.py`, `hf_model.py`, `train.py`, `quality_benchmark.py`, `eval/scaling_laws.py`
- **CLI flag**: `--n_threads K` (default 1 for backward compat)

---

## Closed Issues (Quality Journey)

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
| HVN (Issue #21 strategies) | ~8.37 | -0.0126 |
| HVN (original) | ~8.37 | -0.007 |

**Root cause**: The Hamiltonian compresses all sequence history into a single
d×d matrix per head. This lossy compression loses too much information compared
to attention's full pairwise recall (all N key-value pairs stored uncompressed).

**Next step**: Issue #22 (Multi-Thread Hamiltonian Memory) — K independent
Hamiltonians per head to increase memory capacity from O(d²) to O(K·d²)
while preserving O(1) inference.