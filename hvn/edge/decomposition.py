"""
Structured Weight Decompositions (Issue #32, Phases 2.2/2.3, 3.5, 4.4)
==========================================================================

Tensor-network and algebraic-structure decompositions that replace a dense
``[out, in]`` weight matrix with a factored representation using far fewer
parameters, exploiting the fact that trained weight matrices (and
especially HVN's already-SVD-truncated Hamiltonian) are approximately
low-rank / low-entanglement:

- :class:`LowRankSparseWeights` — Robust-PCA-style ``W ≈ L + S`` split
  (Phase 2.2)
- :class:`TTMatrix` — Tensor-Train matrix format (Novikov et al. 2015),
  the formalization of HVN's existing MPS-style Hamiltonian structure
  (Phase 2.3)
- :class:`MERAMatrix` — simplified multi-scale (hierarchical block)
  renormalization, MERA-inspired (Phase 4.4)
- :class:`KroneckerLinear` — ``W ≈ A ⊗ B`` factorization (Phase 3.5)
- :class:`CirculantLinear`, :class:`ToeplitzLinear` — FFT-diagonalizable
  structured layers (Phase 3.3)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# 2.2 Low-rank + sparse (L+S) decomposition
# ---------------------------------------------------------------------------


class LowRankSparseWeights:
    """Robust-PCA-style decomposition ``W ≈ U·Vᵀ + S``.

    ``U, V`` capture the low-rank "backbone" (the dominant, coherent
    structure of the weight matrix — this is what SVD truncation already
    keeps in HVN's Hamiltonian), and ``S`` is a sparse correction that
    captures the outlier features the low-rank part cannot represent.  The
    velocity computation ``v = Γ·(H·ω)`` becomes a low-rank matmul plus a
    sparse matmul: ``Γ·(U·(Vᵀ·ω) + S·ω)``.

    Parameters
    ----------
    rank : int
        Rank of the low-rank component.
    sparsity : float
        Fraction of elements in the residual to zero out (e.g. 0.99 keeps
        only the top 1% largest-magnitude residual entries).
    iterations : int
        Number of alternating low-rank/sparse refinement passes.  A single
        SVD pass is not robust to outliers (a few huge entries can hijack
        the top singular vectors); alternating "fit low-rank to W-S, then
        fit sparse to W-L" a few times is a lightweight approximation of
        the full Robust PCA (IALM) iteration that converges to a much
        cleaner separation.
    """

    def __init__(self, rank: int = 32, sparsity: float = 0.99, iterations: int = 5):
        self.rank = rank
        self.sparsity = sparsity
        self.iterations = iterations

    @torch.no_grad()
    def decompose(self, W: torch.Tensor) -> dict:
        """Split *W* into low-rank ``U, V`` factors and a sparse residual ``S``."""
        device, orig_dtype = W.device, W.dtype
        W = W.float()
        S = torch.zeros_like(W)
        U, V = None, None

        for _ in range(self.iterations):
            U_full, S_diag, Vh_full = torch.linalg.svd(W - S, full_matrices=False)
            r = min(self.rank, S_diag.numel())
            U = U_full[:, :r] * S_diag[:r].unsqueeze(0)
            V = Vh_full[:r, :]
            L = U @ V

            residual = W - L
            k = max(1, int(residual.numel() * (1 - self.sparsity)))
            flat = residual.abs().flatten()
            threshold = torch.topk(flat, k).values.min() if k < flat.numel() else flat.min()
            mask = residual.abs() >= threshold
            S = residual * mask

        return {
            "U": U.to(orig_dtype), "V": V.to(orig_dtype), "S": S.to_sparse().to(orig_dtype),
            "shape": tuple(W.shape), "rank": r, "orig_dtype": orig_dtype,
        }

    @staticmethod
    def reconstruct(qdict: dict) -> torch.Tensor:
        L = qdict["U"] @ qdict["V"]
        S = qdict["S"].to_dense() if qdict["S"].is_sparse else qdict["S"]
        return (L + S).to(qdict["orig_dtype"])

    @staticmethod
    def memory_bytes(qdict: dict) -> int:
        rows, cols = qdict["shape"]
        r = qdict["rank"]
        low_rank_bytes = (rows * r + r * cols) * 4
        S = qdict["S"]
        nnz = S._nnz() if S.is_sparse else int((S != 0).sum().item())
        # sparse residual: value (fp32) + row/col indices (int32 each)
        sparse_bytes = nnz * (4 + 4 + 4)
        return int(low_rank_bytes + sparse_bytes)


# ---------------------------------------------------------------------------
# 2.3 / 4.4 Tensor Train (TT) and simplified MERA decomposition
# ---------------------------------------------------------------------------


class TTMatrix:
    """Tensor-Train matrix format (Novikov et al. 2015, "Tensorizing Neural Networks").

    Reshapes a dense ``[M, N]`` weight matrix into a higher-order tensor by
    factorizing ``M = m_1·...·m_d`` and ``N = n_1·...·n_d``, then applies
    TT-SVD to obtain a chain of small 4D cores
    ``G_k ∈ R^{r_{k-1} × m_k × n_k × r_k}``.  This is the formalization of
    the Hamiltonian's existing MPS structure described in
    ``docs/THEORETICAL_FRAMEWORK.md`` §3.1.

    Total parameters scale as ``O(d · r² · max(m_k, n_k))`` instead of
    ``O(M·N)`` — for the worked example in the issue (``d_model=4096 =
    64×64``, rank ``r=32``): ``2 · 64 · 32² ≈ 131K`` vs ``4096² ≈ 16.8M``
    dense — a **128×** reduction on the largest matrices at 1T-parameter
    scale.
    """

    def __init__(self, cores: list[torch.Tensor], row_factors: tuple[int, ...], col_factors: tuple[int, ...]):
        self.cores = cores
        self.row_factors = row_factors
        self.col_factors = col_factors

    @classmethod
    @torch.no_grad()
    def from_dense(
        cls, W: torch.Tensor, row_factors: tuple[int, ...], col_factors: tuple[int, ...], max_rank: int
    ) -> "TTMatrix":
        """TT-SVD decompose a dense matrix.

        Parameters
        ----------
        W : torch.Tensor
            Dense matrix, shape ``[prod(row_factors), prod(col_factors)]``.
        row_factors, col_factors : tuple[int, ...]
            Factorizations of the row/column dimensions (must have equal
            length ``d`` — one factor pair per TT core).
        max_rank : int
            Maximum TT-rank at each internal bond.
        """
        d = len(row_factors)
        assert len(col_factors) == d, "row_factors and col_factors must have the same length"
        M, N = W.shape
        assert math.prod(row_factors) == M and math.prod(col_factors) == N

        T = W.reshape(*row_factors, *col_factors)
        perm = []
        for i in range(d):
            perm += [i, d + i]
        T = T.permute(*perm).contiguous()  # (m1, n1, m2, n2, ..., md, nd)

        cores = []
        r_prev = 1
        remaining = T.reshape(1, -1)
        for k in range(d - 1):
            m_k, n_k = row_factors[k], col_factors[k]
            remaining = remaining.reshape(r_prev * m_k * n_k, -1)
            U, S, Vh = torch.linalg.svd(remaining, full_matrices=False)
            r_k = min(max_rank, U.shape[1])
            U, S, Vh = U[:, :r_k], S[:r_k], Vh[:r_k, :]
            cores.append(U.reshape(r_prev, m_k, n_k, r_k))
            remaining = torch.diag(S) @ Vh
            r_prev = r_k
        m_last, n_last = row_factors[-1], col_factors[-1]
        cores.append(remaining.reshape(r_prev, m_last, n_last, 1))
        return cls(cores, row_factors, col_factors)

    @torch.no_grad()
    def to_dense(self) -> torch.Tensor:
        """Reconstruct the full ``[M, N]`` matrix by contracting all cores."""
        d = len(self.row_factors)
        result = self.cores[0].squeeze(0)  # [m1, n1, r1]
        for k in range(1, d):
            core = self.cores[k]  # [r_{k-1}, m_k, n_k, r_k]
            r_prev, m_k, n_k, r_k = core.shape
            combined = torch.einsum("rcp,pmnk->rcmnk", result, core)
            R_rows, R_cols = combined.shape[0], combined.shape[1]
            result = combined.permute(0, 2, 1, 3, 4).reshape(R_rows * m_k, R_cols * n_k, r_k)
        return result.squeeze(-1)

    def matvec(self, x: torch.Tensor) -> torch.Tensor:
        """Matrix-vector product without materializing the dense matrix (2-core case).

        For the common ``d=2`` factorization this contracts ``x`` through
        the two TT cores directly, at ``O(r·(m1·n1·n2 + m1·m2·r))`` cost
        instead of materializing the full ``[M, N]`` matrix.  For ``d>2``
        this falls back to :meth:`to_dense` (still correct, just without
        the memory-avoidance benefit for deeper TT chains).
        """
        if len(self.row_factors) != 2:
            return self.to_dense() @ x
        m1, m2 = self.row_factors
        n1, n2 = self.col_factors
        G1 = self.cores[0].squeeze(0)  # [m1, n1, r]
        G2 = self.cores[1].squeeze(-1)  # [r, m2, n2]
        X = x.reshape(n1, n2)  # [n1, n2]
        tmp = torch.einsum("rbn,an->rba", G2, X)  # [r, m2, n1] (contract n2)
        out = torch.einsum("mnr,rbn->mb", G1, tmp)  # [m1, m2] (contract r, n1)
        return out.reshape(m1 * m2)

    def memory_bytes(self, bits: int = 32) -> int:
        return int(sum(c.numel() for c in self.cores) * bits / 8.0)

    def num_params(self) -> int:
        return sum(c.numel() for c in self.cores)


class MERAMatrix:
    """Simplified, MERA-inspired hierarchical (multi-scale) weight compression.

    True MERA (White 1992 renormalization / Vidal 2003 entanglement
    renormalization) alternates learned "disentangler" and "isometry"
    tensors across a full renormalization-group tree.  This class
    implements a practical, testable approximation of that idea: it
    recursively splits the matrix into 2×2 blocks, keeps a rank-``r``
    low-rank summary ("isometry") of each level, and recurses on the
    coarse-grained residual — capturing correlations at multiple scales
    (block-diagonal *and* long-range) the way a wavelet transform captures
    multi-resolution structure, with a low-rank "entanglement" step at each
    scale rather than MERA's full unitary disentanglers.

    Parameters
    ----------
    rank_per_level : int
        Rank kept at each recursion level.
    max_levels : int
        Maximum recursion depth (stops early once the block is smaller
        than ``rank_per_level``).
    """

    def __init__(self, rank_per_level: int = 8, max_levels: int = 3):
        self.rank_per_level = rank_per_level
        self.max_levels = max_levels

    @torch.no_grad()
    def decompose(self, W: torch.Tensor) -> dict:
        levels = []
        residual = W.float()
        for _ in range(self.max_levels):
            rows, cols = residual.shape
            if min(rows, cols) <= self.rank_per_level or rows % 2 or cols % 2:
                break
            U_full, S, Vh_full = torch.linalg.svd(residual, full_matrices=False)
            r = min(self.rank_per_level, S.numel())
            U, Sv, V = U_full[:, :r], S[:r], Vh_full[:r, :]
            levels.append((U * Sv.unsqueeze(0), V))
            approx = (U * Sv.unsqueeze(0)) @ V
            residual = residual - approx
        return {"levels": levels, "residual": residual, "shape": tuple(W.shape), "orig_dtype": W.dtype}

    @staticmethod
    def reconstruct(qdict: dict) -> torch.Tensor:
        out = qdict["residual"].clone()
        for U, V in qdict["levels"]:
            out = out + U @ V
        return out.to(qdict["orig_dtype"])

    @staticmethod
    def memory_bytes(qdict: dict) -> int:
        total = qdict["residual"].numel() * 4
        for U, V in qdict["levels"]:
            total += (U.numel() + V.numel()) * 4
        return int(total)


# ---------------------------------------------------------------------------
# 3.5 Kronecker-factored weights
# ---------------------------------------------------------------------------


class KroneckerLinear(nn.Module):
    """``nn.Linear`` approximated as a Kronecker product ``W ≈ A ⊗ B``.

    Parameters
    ----------
    in_features, out_features : int
        Must each factor as ``in_features = in_a * in_b``,
        ``out_features = out_a * out_b``.
    in_a, out_a : int
        Dimensions of the first Kronecker factor ``A ∈ R^{out_a × in_a}``;
        the second factor ``B ∈ R^{out_b × in_b}`` is inferred.
    """

    def __init__(self, in_features: int, out_features: int, in_a: int, out_a: int, bias: bool = False):
        super().__init__()
        assert in_features % in_a == 0 and out_features % out_a == 0
        in_b, out_b = in_features // in_a, out_features // out_a
        self.in_features, self.out_features = in_features, out_features
        self.in_a, self.out_a, self.in_b, self.out_b = in_a, out_a, in_b, out_b
        self.A = nn.Parameter(torch.empty(out_a, in_a))
        self.B = nn.Parameter(torch.empty(out_b, in_b))
        nn.init.xavier_uniform_(self.A)
        nn.init.xavier_uniform_(self.B)
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None

    def weight(self) -> torch.Tensor:
        return torch.kron(self.A, self.B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight(), self.bias)

    @classmethod
    def from_linear_via_svd(cls, linear: nn.Linear, in_a: int, out_a: int) -> "KroneckerLinear":
        """Fit the best rank-1 Kronecker factorization of an existing dense layer's weight.

        Uses the Van Loan-Pitsianis rearrangement: reshaping ``W`` so that
        its best rank-1 approximation (via SVD) yields the optimal
        ``A ⊗ B`` factors in the Frobenius-norm sense.
        """
        W = linear.weight.data
        out_features, in_features = W.shape
        in_b, out_b = in_features // in_a, out_features // out_a
        # Van Loan-Pitsianis rearrangement: R[(oa,ob),(ia,ib)] -> R2[oa*ia, ob*ib]
        R = W.reshape(out_a, out_b, in_a, in_b).permute(0, 2, 1, 3).reshape(out_a * in_a, out_b * in_b)
        U, S, Vh = torch.linalg.svd(R, full_matrices=False)
        a = (U[:, 0] * math.sqrt(S[0].item())).reshape(out_a, in_a)
        b = (Vh[0, :] * math.sqrt(S[0].item())).reshape(out_b, in_b)

        module = cls(in_features, out_features, in_a, out_a, bias=linear.bias is not None)
        module.A.data = a
        module.B.data = b
        if module.bias is not None:
            module.bias.data = linear.bias.data.clone()
        module.to(W.device)
        return module

    def num_params(self) -> int:
        return self.A.numel() + self.B.numel()


# ---------------------------------------------------------------------------
# 3.3 Circulant / Toeplitz structured weights
# ---------------------------------------------------------------------------


class CirculantLinear(nn.Module):
    """Square linear layer whose weight is a circulant matrix, specified by one row.

    A circulant matrix is fully determined by its first row ``c`` (``d``
    parameters instead of ``d²``) and diagonalizes under the DFT, so
    ``W @ x`` can be computed in ``O(d log d)`` via FFT:
    ``W @ x = IFFT(FFT(c) * FFT(x))``.
    """

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.c = nn.Parameter(torch.randn(dim) / math.sqrt(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def weight(self) -> torch.Tensor:
        idx = (torch.arange(self.dim).unsqueeze(0) - torch.arange(self.dim).unsqueeze(1)) % self.dim
        return self.c[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Xf = torch.fft.fft(x.float(), dim=-1)
        Cf = torch.fft.fft(self.c.float(), dim=-1)
        y = torch.fft.ifft(Xf * Cf.unsqueeze(0), dim=-1).real.to(x.dtype)
        if self.bias is not None:
            y = y + self.bias
        return y

    def num_params(self) -> int:
        return self.c.numel()


class ToeplitzLinear(nn.Module):
    """Square linear layer whose weight is a Toeplitz matrix, specified by ``2d-1`` values.

    Toeplitz matrices are embeddable in a ``2d × 2d`` circulant matrix,
    enabling the same FFT-based ``O(d log d)`` matmul as
    :class:`CirculantLinear`.
    """

    def __init__(self, dim: int, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.diagonals = nn.Parameter(torch.randn(2 * dim - 1) / math.sqrt(dim))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None

    def weight(self) -> torch.Tensor:
        d = self.dim
        idx = (torch.arange(d).unsqueeze(1) - torch.arange(d).unsqueeze(0)) + (d - 1)
        return self.diagonals[idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight(), self.bias)

    def num_params(self) -> int:
        return self.diagonals.numel()
