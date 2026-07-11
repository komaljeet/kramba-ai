# Theoretical Framework & FLOP Analysis

> **Moved from README:** §2 Theoretical Framework & Mathematical Formalism · §3 FLOP
> Analysis: HVN vs Multi-Head Attention. See [index.md](index.md) for the project overview.

## 2. Theoretical Framework & Mathematical Formalism

The architecture fuses two theoretical pillars:

| Pillar | Source Field | Role in HVN |
|--------|-------------|-------------|
| **Lagrangian Vortex Particles** | Fluid Dynamics | Translates each token into a discrete vortex particle that interacts *locally* with a background velocity field via a modified Biot-Savart law. |
| **Matrix Product State (MPS) Entanglement Hamiltonian** | Quantum Mechanics | A fixed-size tensor-train state that evolves deterministically on every forward pass, absorbing vortex vectors through quantum entanglement renormalization. |

### 2.1 Lagrangian Vortex Particle Mapping

Each input token embedding **x**_t ∈ ℝ^{d_model} is interpreted as a **Lagrangian vortex
particle** characterized by:

- **Position**: r_t ∈ ℝ^{d_model} — the token's embedding vector (static within a step).
- **Vorticity (circulation)**: ω_t ∈ ℝ^{d_model} — derived via a learned linear projection.
- **Velocity**: v_t ∈ ℝ^{d_model} — the velocity field induced at the particle's location
  by all *previously absorbed* vorticity stored in the Hamiltonian memory.

**Modified Biot-Savart Law (Local O(1) Approximation).** Classical fluid dynamics
computes induced velocity by a global integral (O(N) over all prior particles):

```
v(r) = (1/4π) ∫ (ω(r') × (r - r')) / |r - r'|³  dV'
```

HVN replaces it with a **local, fixed-size approximation**: `v_t = Γ · (H · ω_t)`

- **H** ∈ ℝ^{d_model × d_model} is the **Entanglement Hamiltonian** — a fixed-size matrix
  summarizing the entire history of absorbed vorticity.
- **Γ** is a learned scalar coupling constant (initialized to 1/√d_model).
- **ω_t** is the current token's vorticity vector.

Because **H** has fixed shape, velocity is a single matrix-vector product: **O(d_model²)**
regardless of N. This is the key to O(1) per-token inference.

**Vortex Deflection.** The velocity deflects the token's position, producing the
*interacted* representation injected into quantum memory: `r̃_t = r_t + Γ · v_t`. This is
the fluid-dynamical analogue of the query-key interaction in Attention, but computed in
constant time against a summary rather than against all keys.

### 2.2 Entanglement Hamiltonian Matrix (Fixed-Size Quantum Memory)

**MPS (Tensor Train) Structure.** The Hamiltonian memory is stored as an MPS — a tensor
train of fixed bond dimension χ: `H = A_1 · A_2 · ... · A_K` (K = d_model / χ), each core
**A_k** ∈ ℝ^{χ × χ}. HVN **never explicitly reconstructs** the full d×d matrix; operations
run on the compressed cores, keeping the footprint at O(K · χ²) = O(d_model · χ). (The
reference implementation stores H as a single d×d matrix but updates it via low-rank
MPS-style renormalization.)

**Quantum Entanglement Renormalization.** When a new vortex vector **v_t** arrives, it is
absorbed into the Hamiltonian (analogous to a DMRG step). The update rule:

```
H_new = (1 - η) · H_old + η · (v_t ⊗ v_t^T - H_old · v_t ⊗ v_t^T · H_old)
```

Decomposed: (1) **Outer product** P_t = v_t ⊗ v_t^T — rank-1 interaction projector; (2)
**Destructive interference** subtract H_old·P_t·H_old — cancels components already aligned
with v_t; (3) **Associative alignment** add P_t — reinforces the new association; (4)
**Renormalization** blend with (1-η)·H_old. η is a learned or fixed entanglement rate
(default 1/√d_model). This is a **deterministic, single-pass update** — no optimization,
runs in O(d_model²) (or O(d_model·χ) compressed), independent of sequence length.

**Low-Rank MPS Compression.** After each update a truncated-SVD renormalization keeps the
top-r singular values (`H = U_r · S_r · V_r^T`), discarding high-energy noise modes. Cost:
O(d_model² · r) — still O(1) in N.

### 2.3 The HVN Block (per token t)

```
1. VortexInjection:
   ω_t = W_ω · x_t                    (vorticity projection)
   v_t = Γ · (H · ω_t)                (Biot-Savart local velocity)
   r̃_t = x_t + Γ · v_t                (deflected representation)

2. HamiltonianMemoryUpdate:
   P_t = v_t ⊗ v_t^T                  (rank-1 interaction projector)
   H ← (1-η)·H + η·(P_t - H·P_t·H)    (entanglement renormalization)
   [optional: SVD truncation to rank r]

3. Output:
   y_t = W_o · r̃_t                    (project back to d_model)
```

**Total per-token cost**: O(d_model²) — constant in N. The Hamiltonian is a compressed
holographic summary of all past tokens; correlated concepts reinforce (via P_t) while noise
cancels (via H·P_t·H), and each new token "feels" the entire history through H·ω_t without
scanning it.

## 3. FLOP Analysis: HVN vs Multi-Head Attention

Notation: N=seq len (up to 100,000), d=d_model (512), h=heads (8), r=SVD bond dim (32–128),
L=layers (12), V=vocab (32,000). All counts are inference (forward), per token, per layer.

### 3.1 Standard Multi-Head Attention (MHA)

- **Without KV cache**: 8d² + 4Nd + 3N per token per layer.
- **With KV cache**: 4d² + 4Nd + 3N. Even with a KV cache, per-token cost is **O(Nd)** — the
  new query must attend to all N cached keys/values. The cache reduces *recomputation*,
  not *scan complexity*.
- **KV cache memory**: 2·N·d·bytes per layer. For N=100K, d=512, fp16: ~200 MB/layer →
  **~2.4 GB** for 12 layers.

### 3.2 HVN Per-Token Cost (exploiting rank-1 structure)

Because P_t = v_t ⊗ v_t^T is rank-1, `H·P_t·H = (Hv) ⊗ (H^T v)^T`, computed via two matvecs
(2d² each) + one outer product (d²) = **5d²** instead of the naive 2d³.

| Operation | FLOPs |
|-----------|-------|
| Vorticity projection | 2d² |
| Biot-Savart velocity (H · ω_t) | 2d² |
| Deflection + output proj + norm | 2d² + 2d |
| Interaction projector (outer product) | d² |
| H · v_t and H^T · v_t (destructive interference) | 4d² |
| Outer product (Hv ⊗ (H^Tv)^T) | d² |
| Renormalization blend | 2d² |
| SVD truncation (optional, rank r) | O(d²r) |
| FFN sub-layer (2× expansion) | 6d² |
| **Total per token per layer** | **~19d² + O(d²r)** |

For d=512: with SVD r=128 ≈ **38.6M** FLOPs/layer; full rank ≈ 139.2M. HVN Hamiltonian
memory = d²·bytes/layer → for d=512, fp16, 12 layers: **~6.29 MB, constant in N**.

### 3.3 Head-to-Head at N = 100,000 (d=512, L=12, r=128)

| Architecture | Total Per-Token FLOPs (all 12 layers) |
|---|---|
| MHA with KV cache | ~**2,474M** |
| HVN (with SVD, r=128) | ~**463M** |
| HVN (no SVD) | ~1,670M |

| Variant | Ratio at N=100K |
|---|---|
| **HVN with SVD (r=128)** | 463 / 2,474 ≈ **18.7%** ✓ (< 20% target) |
| HVN without SVD | ≈ 67.5% |

### 3.4 Scaling vs Sequence Length (d=512, L=12, r=128)

| N | MHA (KV cache) | HVN (SVD r=128) | Ratio |
|---|---|---|---|
| 1,000 | 36M | 463M | 12.8× |
| 10,000 | 258M | 463M | 0.56× |
| 100,000 | 2,474M | 463M | **0.187×** |
| 1,000,000 | 24,588M | 463M | 0.019× |

MHA is **O(Nd)** (grows linearly); HVN is **O(d²r)** (constant in N). Crossover ≈ N ≈
5,000; HVN's advantage grows unboundedly beyond. Memory: HVN uses ~390× less at N=100K
(6.29 MB vs 2.46 GB), improving further as N grows. Empirical scaling results in
[benchmarks_scaling.md](benchmarks_scaling.md).