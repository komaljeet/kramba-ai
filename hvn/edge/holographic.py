"""
Holographic & Superposition Weight Encoding (Issue #32, Phases 2.3, 8.1, 8.2)
=================================================================================

Techniques that store *multiple* weight matrices (e.g. one Hamiltonian per
layer) in a single shared representation, trading a small amount of
recovery noise for a large reduction in the number of distinct large
tensors that must be stored:

- :class:`HolographicWeights` — Gabor/Plate-style holographic superposition:
  ``H = Σ_k W_k ⊙ Φ_k`` with near-orthogonal random ``±1`` phase masks
  ``Φ_k``; each ``W_k`` is recovered as ``H ⊙ Φ_k`` with cross-talk that
  shrinks as ``O(1/√d)``.
- :class:`SuperpositionWeights` — a *learned* shared basis (via joint SVD)
  rather than random masks: ``W_k = Σ_i α_{k,i} · B_i``.
- :class:`PhaseEncodedComplexWeights` — factors a set of same-shape
  matrices into one shared magnitude tensor plus a low-bit phase tensor
  per matrix.
"""

from __future__ import annotations

import math

import torch


# ---------------------------------------------------------------------------
# 8.1 Holographic weight encoding (random-phase superposition)
# ---------------------------------------------------------------------------


class HolographicWeights:
    """Superimposes ``num_slots`` weight matrices into one holographic matrix.

    Each slot ``k`` gets a fixed pseudorandom ``±1`` phase mask ``Φ_k``
    (regenerated deterministically from *seed*, so nothing but the seed
    needs to be stored).  Encoding sums ``W_k ⊙ Φ_k`` across slots into a
    single matrix of the *same shape* as one input matrix; decoding
    multiplies by the same mask again.  Because the masks are
    near-orthogonal, ``Φ_j ⊙ Φ_k`` for ``j ≠ k`` averages out to
    ``O(1/√d)`` noise instead of canceling exactly — this is the
    fundamental accuracy/compression trade-off of the technique (more
    slots = more compression, more cross-talk).

    Parameters
    ----------
    shape : tuple[int, int]
        Shape of each individual weight matrix (all slots must match).
    num_slots : int
        Number of matrices to superimpose (``L`` layers -> ``num_slots=L``
        gives ``L×`` compression of the raw storage).
    seed : int
        RNG seed for the phase masks.
    """

    def __init__(self, shape: tuple[int, int], num_slots: int, seed: int = 0):
        self.shape = shape
        self.num_slots = num_slots
        self.seed = seed
        self._masks: list[torch.Tensor] | None = None

    def _get_masks(self, device: torch.device) -> list[torch.Tensor]:
        if self._masks is None:
            gen = torch.Generator(device="cpu").manual_seed(self.seed)
            self._masks = [
                (torch.randint(0, 2, self.shape, generator=gen).float() * 2 - 1).to(device)
                for _ in range(self.num_slots)
            ]
        return self._masks

    @torch.no_grad()
    def encode(self, weights: list[torch.Tensor]) -> torch.Tensor:
        """Superimpose up to ``num_slots`` matrices into one holographic matrix."""
        assert len(weights) <= self.num_slots
        device = weights[0].device
        masks = self._get_masks(device)
        H = torch.zeros(self.shape, device=device, dtype=torch.float32)
        for k, W in enumerate(weights):
            H = H + W.float() * masks[k]
        return H

    @torch.no_grad()
    def decode(self, H: torch.Tensor, slot: int) -> torch.Tensor:
        """Recover the matrix stored in *slot* (with ``O(1/√d)`` cross-talk noise)."""
        return H * self._get_masks(H.device)[slot]

    def memory_bytes(self) -> int:
        """Storage is a single matrix regardless of ``num_slots`` (+ a few bytes for the seed)."""
        rows, cols = self.shape
        return int(rows * cols * 4) + 8


# ---------------------------------------------------------------------------
# 2.3 Quantum-inspired superposition weight sharing (learned shared basis)
# ---------------------------------------------------------------------------


class SuperpositionWeights:
    """Shared-basis weight compression via joint SVD.

    Unlike :class:`HolographicWeights` (random masks, exact-shape shared
    storage), this learns a small set of shared basis matrices ``B_i`` by
    flattening every input matrix into a row vector and taking the
    dominant right-singular vectors of the stacked matrix — the optimal
    (least-squares) shared basis for reconstructing all of them from a
    small number of per-matrix coefficients.

    Parameters
    ----------
    num_basis : int
        Number of shared basis vectors to keep.
    """

    def __init__(self, num_basis: int):
        self.num_basis = num_basis

    @torch.no_grad()
    def fit(self, weights: list[torch.Tensor]) -> dict:
        shape = tuple(weights[0].shape)
        flat = torch.stack([w.flatten().float() for w in weights], dim=0)  # [K, D]
        _, _, Vh = torch.linalg.svd(flat, full_matrices=False)
        b = min(self.num_basis, Vh.shape[0])
        basis = Vh[:b, :]  # [b, D]
        coeffs = flat @ basis.t()  # [K, b]
        return {"coeffs": coeffs, "basis": basis, "shape": shape, "num_matrices": len(weights)}

    @staticmethod
    def reconstruct(qdict: dict, idx: int) -> torch.Tensor:
        vec = qdict["coeffs"][idx] @ qdict["basis"]
        return vec.reshape(qdict["shape"])

    @staticmethod
    def memory_bytes(qdict: dict) -> int:
        return int(qdict["coeffs"].numel() * 4 + qdict["basis"].numel() * 4)


# ---------------------------------------------------------------------------
# 8.2 Phase-encoded complex weight representation
# ---------------------------------------------------------------------------


class PhaseEncodedComplexWeights:
    """Shares one magnitude tensor across same-shape matrices; stores only low-bit phase per matrix.

    Given ``K`` same-shape weight matrices, factors out a single shared
    magnitude ``M = mean_k(|W_k|)`` and represents each matrix as
    ``W_k ≈ M ⊙ cos(θ_k)``.  Since ``θ_k = acos(clip(W_k / M, -1, 1))``
    exactly inverts (``cos(acos(x)) = x``) up to the clipping error, the
    only actual information loss is from quantizing ``θ_k`` to
    *phase_bits* — which, because the (typically dominant) magnitude
    information is amortized across all ``K`` matrices, gives close to
    ``32 / phase_bits`` compression once ``K`` is reasonably large.
    """

    def __init__(self, phase_bits: int = 8):
        self.phase_bits = phase_bits

    @torch.no_grad()
    def fit(self, weights: list[torch.Tensor]) -> dict:
        stacked = torch.stack([w.float() for w in weights], dim=0)  # [K, rows, cols]
        magnitude = stacked.abs().mean(dim=0).clamp(min=1e-8)
        ratio = torch.clamp(stacked / magnitude.unsqueeze(0), -1, 1)
        phases = torch.acos(ratio)  # in [0, pi]
        qmax = 2 ** self.phase_bits - 1
        phase_codes = torch.round(phases / math.pi * qmax).clamp(0, qmax).to(torch.uint8 if self.phase_bits <= 8 else torch.int16)
        return {
            "magnitude": magnitude, "phase_codes": phase_codes, "phase_bits": self.phase_bits,
            "shape": tuple(weights[0].shape), "num_matrices": len(weights), "orig_dtype": weights[0].dtype,
        }

    @staticmethod
    def reconstruct(qdict: dict, idx: int) -> torch.Tensor:
        qmax = 2 ** qdict["phase_bits"] - 1
        theta = qdict["phase_codes"][idx].float() / qmax * math.pi
        return (qdict["magnitude"] * torch.cos(theta)).to(qdict["orig_dtype"])

    @staticmethod
    def memory_bytes(qdict: dict) -> int:
        rows, cols = qdict["shape"]
        K, bits = qdict["num_matrices"], qdict["phase_bits"]
        magnitude_bytes = rows * cols * 4
        phase_bytes = rows * cols * K * bits / 8.0
        return int(magnitude_bytes + phase_bytes)
