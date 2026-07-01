# Hamiltonian Vortex Network (HVN) — Theoretical Framework & Mathematical Formalism

## 1. Overview

The Hamiltonian Vortex Network (HVN) is a novel sequence-modeling architecture that
replaces standard Self-Attention with a **fixed-size quantum-fluid dynamical system**.
The design goal is to achieve long-range associative recall comparable to Attention
while maintaining **O(1) inference cost per token** — no KV cache, no quadratic scan,
no gradient-descent test-time-training loop.

The architecture fuses two theoretical pillars:

| Pillar | Source Field | Role in HVN |
|--------|-------------|-------------|
| **Lagrangian Vortex Particles** | Fluid Dynamics | Translates each token into a discrete vortex particle that interacts *locally* with a background velocity field via a modified Biot-Savart law. |
| **Matrix Product State (MPS) Entanglement Hamiltonian** | Quantum Mechanics | A fixed-size tensor-train state that evolves deterministically on every forward pass, absorbing vortex vectors through quantum entanglement renormalization. |

---

## 2. Lagrangian Vortex Particle Mapping

### 2.1 Token → Vortex Particle

Each input token embedding **x**_t ∈ ℝ^{d_model} is interpreted as a **Lagrangian vortex
particle** characterized by:

- **Position**: r_t ∈ ℝ^{d_model} — the token's embedding vector (static within a step).
- **Vorticity (circulation)**: ω_t ∈ ℝ^{d_model} — derived from the token via a learned
  linear projection.
- **Velocity**: v_t ∈ ℝ^{d_model} — the velocity field induced at the particle's location
  by all *previously absorbed* vorticity stored in the Hamiltonian memory.

### 2.2 Modified Biot-Savart Law (Local O(1) Approximation)

In classical fluid dynamics, the Biot-Savart law computes the velocity induced at a
point by a continuous vorticity distribution:

```
v(r) = (1/4π) ∫ (ω(r') × (r - r')) / |r - r'|³  dV'
```

This is a global integral — O(N) over all prior particles.  HVN replaces it with a
**local, fixed-size approximation**:

```
v_t = Γ · (H · ω_t)
```

where:

- **H** ∈ ℝ^{d_model × d_model} is the **Entanglement Hamiltonian** — a fixed-size matrix
  that *summarizes* the entire history of absorbed vorticity.  It is the effective
  "velocity field" at the current point.
- **Γ** is a learned scalar coupling constant (initialized to 1/√d_model).
- **ω_t** is the current token's vorticity vector.

Because **H** has fixed shape (d_model × d_model), the velocity computation is a single
matrix-vector product: **O(d_model²)** regardless of sequence length N.  This is the
key to O(1) per-token inference.

### 2.3 Vortex Deflection

The new token's velocity **v_t** deflects its position, producing the *interacted*
representation that will be injected into the quantum memory:

```
r̃_t = r_t + Γ · v_t
```

This deflection encodes *how the existing memory field alters the new token* — the
fluid-dynamical analogue of the query-key interaction in Attention, but computed in
constant time against a summary rather than against all keys.

---

## 3. Entanglement Hamiltonian Matrix (Fixed-Size Quantum Memory)

### 3.1 Matrix Product State (Tensor Train) Structure

The Hamiltonian memory is stored as a **Matrix Product State (MPS)** — a tensor train
of fixed bond dimension χ:

```
H = A_1 · A_2 · ... · A_K      (K = d_model / χ)
```

Each **A_k** ∈ ℝ^{χ × χ} is a core tensor.  The full reconstructed matrix is:

```
H_reconstructed = A_1 ⊗ A_2 ⊗ ... ⊗ A_K   (Kronecker product chain)
```

However, HVN **never explicitly reconstructs** the full d_model × d_model matrix.
All operations are performed on the **compressed MPS cores** directly, keeping the
memory footprint at O(K · χ²) = O(d_model · χ) rather than O(d_model²).

For implementation simplicity (and because χ is a hyperparameter), the reference
implementation stores H as a single d_model × d_model matrix but updates it via
**low-rank MPS-style renormalization** (see §3.3).  The theoretical complexity
argument in §FLOP_ANALYSIS holds for the compressed form.

### 3.2 Quantum Entanglement Renormalization

When a new vortex vector **v_t** arrives, it must be *absorbed* into the Hamiltonian.
In quantum mechanics, this is analogous to a **density-matrix renormalization group
(DMRG)** step: the environment interacts with the state, and the state is renormalized
to preserve entanglement structure while discarding noise.

The update rule is:

```
H_new = (1 - η) · H_old + η · (v_t ⊗ v_t^T - H_old · v_t ⊗ v_t^T · H_old)
```

Decomposed:

1. **Outer product**: P_t = v_t ⊗ v_t^T  — the "interaction projector" (rank-1).
2. **Destructive interference**: subtract H_old · P_t · H_old — this term cancels
   components of H that are *already aligned* with v_t, preventing redundant
   reinforcement (destructive interference on noise / repeated patterns).
3. **Associative alignment**: add P_t — this reinforces the new association.
4. **Renormalization**: blend with (1 - η) · H_old to preserve prior memory.
   η is a learned or fixed entanglement rate (default 1/√d_model).

This is a **deterministic, single-pass update** — no optimization step, no gradient
descent.  It runs in O(d_model²) time (or O(d_model · χ) in compressed MPS form),
independent of sequence length.

### 3.3 Low-Rank MPS Compression

To prevent H from growing unboundedly in spectral norm, after each update we apply
a **truncated SVD renormalization**:

```
H = U · S · V^T   →   keep top-r singular values   →   H = U_r · S_r · V_r^T
```

where r is the **entanglement bond dimension** (hyperparameter).  This is the quantum
renormalization step: high-energy (noise) modes are truncated, preserving only the
top-r entangled modes.  Cost: O(d_model² · r) — still O(1) in N.

In the MPS (tensor-train) formulation, this truncation is applied to each bond
individually, yielding O(d_model · χ · r) cost.

---

## 4. Putting It Together: The HVN Block

For each token t in the sequence:

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

**Total per-token cost**: O(d_model²) — constant in N.

---

## 5. Why This Achieves Long-Range Associative Recall

- The Hamiltonian **H** is a *compressed summary* of all past tokens' vorticity
  interactions.  Like a KV cache, it stores associations — but in fixed O(d_model²)
  space rather than O(N · d_model).
- The **entanglement renormalization** ensures that correlated concepts reinforce
  each other (associative alignment via the P_t term) while uncorrelated noise
  cancels out (destructive interference via the H·P_t·H term).
- The **Biot-Savart local approximation** means each new token "feels" the entire
  history through H without scanning it — the velocity field H·ω_t is a global
  summary computed in one matmul.

This is analogous to how a fluid's velocity field at any point encodes the
integrated effect of all upstream vortices — you don't need to re-sum every vortex;
the field already contains their cumulative influence.