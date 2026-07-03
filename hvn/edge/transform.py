"""
Harmonic-Transform Weight Compression (Issue #32, Phase 3.1/3.2)
====================================================================

Weight matrices, like natural images, have smooth/blocky structure whose
energy concentrates in a handful of transform-domain coefficients.
Transforming to that domain and discarding near-zero coefficients gives a
sparse representation that reconstructs the original matrix almost exactly.

- :class:`WaveletCompressor` — multi-level Haar wavelet transform +
  magnitude thresholding (implemented from scratch with ``torch`` FFT/matmul
  primitives — no ``pywavelets`` dependency).
- :class:`DCTCompressor` — global 2D DCT-II transform + thresholding
  (the same energy-concentration principle JPEG uses, simplified from
  block-wise 8×8 DCT to whole-matrix DCT for implementation tractability).
"""

from __future__ import annotations

import math

import torch

# ---------------------------------------------------------------------------
# 3.1 Haar wavelet decomposition
# ---------------------------------------------------------------------------


def _haar_forward(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """1D Haar transform along the last dimension. Returns (approx, detail)."""
    evens, odds = x[..., 0::2], x[..., 1::2]
    return (evens + odds) / math.sqrt(2), (evens - odds) / math.sqrt(2)


def _haar_inverse(avg: torch.Tensor, diff: torch.Tensor) -> torch.Tensor:
    evens = (avg + diff) / math.sqrt(2)
    odds = (avg - diff) / math.sqrt(2)
    return torch.stack([evens, odds], dim=-1).reshape(*avg.shape[:-1], -1)


class WaveletCompressor:
    """Multi-level 2D Haar wavelet weight compression.

    Recursively splits a matrix into an approximation ("LL") subband and
    three detail subbands (LH, HL, HH), keeping only the globally
    largest-magnitude coefficients across all levels.

    Parameters
    ----------
    levels : int
        Number of wavelet decomposition levels (stops early if a
        dimension becomes odd).
    keep_fraction : float
        Fraction of coefficients to retain (e.g. 0.25 keeps the top 25%).
    """

    def __init__(self, levels: int = 2, keep_fraction: float = 0.25):
        self.levels = levels
        self.keep_fraction = keep_fraction

    def _forward_2d(self, W: torch.Tensor, levels: int) -> list:
        coeffs = []
        cur = W
        for _ in range(levels):
            rows, cols = cur.shape
            if rows % 2 or cols % 2:
                break
            avg_c, diff_c = _haar_forward(cur)  # transform along columns
            combined = torch.cat([avg_c, diff_c], dim=-1)  # [rows, cols]
            avg_r, diff_r = _haar_forward(combined.transpose(0, 1))
            avg_r, diff_r = avg_r.transpose(0, 1), diff_r.transpose(0, 1)  # [rows/2, cols]
            half_cols = cols // 2
            LL, LH = avg_r[:, :half_cols], avg_r[:, half_cols:]
            HL, HH = diff_r[:, :half_cols], diff_r[:, half_cols:]
            coeffs.append((LH, HL, HH))
            cur = LL
        coeffs.append(cur)
        return coeffs

    def _inverse_2d(self, coeffs: list) -> torch.Tensor:
        cur = coeffs[-1]
        for LH, HL, HH in reversed(coeffs[:-1]):
            avg_r = torch.cat([cur, LH], dim=-1)
            diff_r = torch.cat([HL, HH], dim=-1)
            combined = _haar_inverse(avg_r.transpose(0, 1), diff_r.transpose(0, 1)).transpose(0, 1)
            cols = combined.shape[1]
            half_cols = cols // 2
            cur = _haar_inverse(combined[:, :half_cols], combined[:, half_cols:])
        return cur

    @torch.no_grad()
    def compress(self, W: torch.Tensor) -> dict:
        coeffs = self._forward_2d(W.float(), self.levels)
        all_vals = torch.cat([
            c.flatten() for group in coeffs for c in (group if isinstance(group, tuple) else (group,))
        ])
        k = max(1, int(all_vals.numel() * self.keep_fraction))
        threshold = torch.topk(all_vals.abs(), k).values.min()

        pruned = []
        for group in coeffs:
            if isinstance(group, tuple):
                pruned.append(tuple(torch.where(c.abs() >= threshold, c, torch.zeros_like(c)) for c in group))
            else:
                pruned.append(torch.where(group.abs() >= threshold, group, torch.zeros_like(group)))
        return {"coeffs": pruned, "shape": tuple(W.shape), "orig_dtype": W.dtype}

    @torch.no_grad()
    def decompress(self, cdict: dict) -> torch.Tensor:
        return self._inverse_2d(cdict["coeffs"]).to(cdict["orig_dtype"])

    @staticmethod
    def memory_bytes(cdict: dict) -> int:
        total = 0
        for group in cdict["coeffs"]:
            for t in (group if isinstance(group, tuple) else (group,)):
                nnz = int((t != 0).sum().item())
                total += nnz * (4 + 4)  # value + sparse index, rough estimate
        return int(total)


# ---------------------------------------------------------------------------
# 3.2 DCT (Discrete Cosine Transform) weight compression
# ---------------------------------------------------------------------------


def _dct_matrix(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Orthonormal DCT-II matrix: ``y = M @ x`` transforms, ``x = Mᵀ @ y`` inverts."""
    k = torch.arange(n, device=device, dtype=dtype).unsqueeze(1)
    i = torch.arange(n, device=device, dtype=dtype).unsqueeze(0)
    M = torch.cos(math.pi / n * (i + 0.5) * k)
    M[0, :] *= 1.0 / math.sqrt(2)
    M *= math.sqrt(2.0 / n)
    return M


class DCTCompressor:
    """Global 2D DCT-II weight compression (JPEG-style energy concentration).

    Simplified from block-wise (8×8/16×16) DCT to a single whole-matrix
    transform for implementation tractability — the underlying principle
    is identical: most of a weight matrix's energy concentrates in a small
    number of low-frequency coefficients, which survive thresholding while
    near-zero high-frequency coefficients are discarded.

    Parameters
    ----------
    quality : float
        Fraction of DCT coefficients to keep (higher = better quality,
        larger storage — analogous to a JPEG quality factor).
    """

    def __init__(self, quality: float = 0.25):
        self.quality = quality

    @torch.no_grad()
    def compress(self, W: torch.Tensor) -> dict:
        rows, cols = W.shape
        Mr, Mc = _dct_matrix(rows, W.device), _dct_matrix(cols, W.device)
        Y = Mr @ W.float() @ Mc.t()
        k = max(1, int(Y.numel() * self.quality))
        threshold = torch.topk(Y.abs().flatten(), k).values.min()
        Y_pruned = torch.where(Y.abs() >= threshold, Y, torch.zeros_like(Y))
        return {"Y": Y_pruned.to_sparse(), "shape": (rows, cols), "orig_dtype": W.dtype}

    @torch.no_grad()
    def decompress(self, cdict: dict) -> torch.Tensor:
        rows, cols = cdict["shape"]
        Y = cdict["Y"]
        Y = Y.to_dense() if Y.is_sparse else Y
        Mr, Mc = _dct_matrix(rows, Y.device), _dct_matrix(cols, Y.device)
        return (Mr.t() @ Y @ Mc).to(cdict["orig_dtype"])

    @staticmethod
    def memory_bytes(cdict: dict) -> int:
        Y = cdict["Y"]
        nnz = Y._nnz() if Y.is_sparse else int((Y != 0).sum().item())
        return int(nnz * (4 + 4 + 4))
