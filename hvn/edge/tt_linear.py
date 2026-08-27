"""Trainable Tensor-Train linear layer (Issue #161).

``hvn/edge/decomposition.py::TTMatrix`` already implements TT-SVD, but it is a
post-hoc *analysis* tool and cannot be put in a model:

* it is a plain class, not an ``nn.Module`` -- its cores are not ``nn.Parameter``
* ``from_dense`` and ``to_dense`` are both ``@torch.no_grad()``
* ``matvec`` takes a single vector (no batch dimension), and for ``d > 2`` it
  falls back to ``to_dense()``, which would silently break gradients

:class:`TTLinear` is the trainable counterpart: the TT cores **are** the
parameters, the forward pass contracts a batched input through them without
ever materialising the dense ``[out, in]`` matrix, and it is a drop-in for
``nn.Linear(in_features, out_features, bias=...)``.

What this is for
----------------
TT is **not** a training-speed lever at the sizes this repo can train. For a
``[d, d]`` matrix with two cores at rank ``r``, the contraction costs
``B*r*n1*m2*(n2+m1)`` against dense ``B*d^2``:

===========  ===================
``d_model``  TT vs dense FLOPs
===========  ===================
384          3.3x MORE
2048         1.5x MORE
4096         break-even
===========  ===================

and #153 measured the training pipeline at 97.2% GPU-busy, so there is no idle
time for a cheaper layer to reclaim. TT is a **weight-storage / parameter-
efficiency** lever, which is on-thesis for the edge-inference goal but has to be
graded against ternary (~10.7x storage at *zero* extra FLOPs), not against fp32.

Interaction with ternary
------------------------
``apply_ternary_to_model`` replaces ``nn.Linear`` submodules. :class:`TTLinear`
is deliberately **not** an ``nn.Linear`` subclass, so it is skipped and TT layers
stay full precision unless ``ternary_cores=True`` is passed explicitly. Silently
stacking two compression schemes on the same weight would make any A/B measure
both at once.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["TTLinear", "balanced_factors", "full_tt_rank", "tt_num_params"]


def balanced_factors(n: int) -> tuple[int, int]:
    """Split *n* into ``(a, b)`` with ``a * b == n`` and ``b / a`` as small as possible.

    TT compresses best when the two cores are of comparable size, so a balanced
    split is the sensible default. ``384 -> (16, 24)``, ``768 -> (24, 32)``.
    """
    if n < 1:
        raise ValueError(f"cannot factor n={n}")
    best = (1, n)
    for a in range(1, int(math.isqrt(n)) + 1):
        if n % a == 0:
            best = (a, n // a)
    return best


def full_tt_rank(row_factors: tuple[int, ...], col_factors: tuple[int, ...]) -> int:
    """Largest bond rank that carries information, i.e. the exact-reconstruction rank.

    Beyond this the extra columns are numerically zero, so a "TT" at this rank is
    a lossless but *larger-than-dense* reparameterisation -- useful only as a
    known-answer cell, never as a compression setting.
    """
    if len(row_factors) != 2:
        # For deeper chains the bound differs per bond; the first bond is the
        # binding one for the 2-core case this helper is used for.
        raise ValueError("full_tt_rank is defined here for the 2-core case only")
    return min(row_factors[0] * col_factors[0], row_factors[1] * col_factors[1])


def tt_num_params(row_factors: tuple[int, ...], col_factors: tuple[int, ...],
                  rank: int) -> int:
    """Parameter count of the TT cores for this factorisation and rank."""
    d = len(row_factors)
    total = 0
    r_prev = 1
    for k in range(d):
        r_k = 1 if k == d - 1 else min(rank, r_prev * row_factors[k] * col_factors[k])
        total += r_prev * row_factors[k] * col_factors[k] * r_k
        r_prev = r_k
    return total


class TTLinear(nn.Module):
    """``nn.Linear`` drop-in whose weight is stored as a Tensor-Train.

    Parameters
    ----------
    in_features, out_features, bias
        As in ``nn.Linear``.
    rank
        Maximum TT bond rank. Must be small enough that the cores are genuinely
        smaller than the dense matrix -- construction refuses otherwise, because
        a "compression" that is larger than what it replaces is a bug that
        otherwise shows up only as an unexplained memory number.
    row_factors, col_factors
        Factorisations of ``out_features`` / ``in_features``. Default: a
        balanced two-way split of each.
    ternary_cores
        Apply BitNet b1.58 ternary quantisation to the cores in the forward pass
        (straight-through). Default ``False``: TT and ternary compress the same
        weight, so stacking them silently would confound any measurement of
        either. Per-tensor scale only.
    allow_expansion
        Permit a rank at which the cores are *larger* than the dense weight.
        Only exact-reconstruction parity tests should set this: for a square
        matrix with balanced factors, the rank needed for lossless
        reconstruction always exceeds the dense parameter count, so the
        known-answer cell cannot be built without it. Never a training setting.

    Notes
    -----
    The cores are initialised randomly and then rescaled **once**, at
    construction, so the composed matrix has the same standard deviation Xavier
    would have given the dense layer. A hand-derived per-core variance formula
    is easy to get subtly wrong for arbitrary rank and factor shapes; measuring
    the realised scale and correcting it is exact by construction.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = False,
                 rank: int = 32,
                 row_factors: tuple[int, ...] | None = None,
                 col_factors: tuple[int, ...] | None = None,
                 ternary_cores: bool = False,
                 allow_expansion: bool = False):
        super().__init__()
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.rank = int(rank)
        self.ternary_cores = bool(ternary_cores)

        rf = tuple(row_factors) if row_factors is not None else balanced_factors(out_features)
        cf = tuple(col_factors) if col_factors is not None else balanced_factors(in_features)
        if len(rf) != len(cf):
            raise ValueError(f"row_factors {rf} and col_factors {cf} must have equal length")
        if math.prod(rf) != self.out_features:
            raise ValueError(f"row_factors {rf} multiply to {math.prod(rf)}, not out_features={out_features}")
        if math.prod(cf) != self.in_features:
            raise ValueError(f"col_factors {cf} multiply to {math.prod(cf)}, not in_features={in_features}")
        self.row_factors, self.col_factors = rf, cf

        n_tt = tt_num_params(rf, cf, self.rank)
        n_dense = self.out_features * self.in_features
        if n_tt >= n_dense and not allow_expansion:
            raise ValueError(
                f"TT at rank {self.rank} needs {n_tt:,} params against {n_dense:,} dense "
                f"for [{out_features}, {in_features}] with factors {rf} x {cf} -- it is "
                f"BIGGER, not a compression. Lower the rank (max useful here is "
                f"{full_tt_rank(rf, cf) if len(rf) == 2 else 'n/a'})."
            )
        self._num_core_params = n_tt

        # Cores: G_k in R^{r_{k-1} x m_k x n_k x r_k}
        cores = []
        r_prev = 1
        d = len(rf)
        for k in range(d):
            r_k = 1 if k == d - 1 else min(self.rank, r_prev * rf[k] * cf[k])
            cores.append(nn.Parameter(torch.randn(r_prev, rf[k], cf[k], r_k)))
            r_prev = r_k
        self.cores = nn.ParameterList(cores)

        self.bias = nn.Parameter(torch.zeros(self.out_features)) if bias else None
        self._rescale_cores_to_xavier()

    # ------------------------------------------------------------------
    @torch.no_grad()
    def _rescale_cores_to_xavier(self) -> None:
        """Scale the cores so the composed matrix matches Xavier-uniform's std."""
        target_std = math.sqrt(2.0 / (self.in_features + self.out_features))
        actual = self.to_dense().std().item()
        if actual <= 0 or not math.isfinite(actual):  # pragma: no cover - degenerate init
            return
        factor = (target_std / actual) ** (1.0 / len(self.cores))
        for c in self.cores:
            c.mul_(factor)

    def to_dense(self) -> torch.Tensor:
        """Materialise the effective ``[out_features, in_features]`` matrix.

        For inspection, parity tests and checkpoint conversion -- **not** used by
        :meth:`forward`, which is the whole point of the layer.
        """
        d = len(self.row_factors)
        result = self.cores[0].squeeze(0)  # [m1, n1, r1]
        for k in range(1, d):
            core = self.cores[k]
            r_prev, m_k, n_k, r_k = core.shape
            combined = torch.einsum("rcp,pmnk->rcmnk", result, core)
            R_rows, R_cols = combined.shape[0], combined.shape[1]
            result = combined.permute(0, 2, 1, 3, 4).reshape(R_rows * m_k, R_cols * n_k, r_k)
        return result.squeeze(-1)

    def _effective_cores(self) -> list[torch.Tensor]:
        if not self.ternary_cores:
            return list(self.cores)
        from .ternary import ternary_quantize_ste
        return [ternary_quantize_ste(c) for c in self.cores]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: ``[..., in_features]`` -> ``[..., out_features]``.

        Contracts through the cores one at a time. The dense matrix is never
        formed, so peak memory is set by the cores plus the running carry.
        """
        if x.shape[-1] != self.in_features:
            raise ValueError(
                f"expected last dim {self.in_features}, got {tuple(x.shape)}"
            )
        lead = x.shape[:-1]
        cores = self._effective_cores()
        carry = x.reshape(-1, 1, 1, self.in_features)  # [B, out_so_far, r_prev, rest]
        B = carry.shape[0]

        for k, G in enumerate(cores):
            r_prev, m_k, n_k, r_k = G.shape
            rest = carry.shape[-1]
            if rest % n_k:  # pragma: no cover - guarded by factor validation
                raise RuntimeError(f"core {k}: rest {rest} not divisible by n_k {n_k}")
            rest_next = rest // n_k
            carry = carry.reshape(B, -1, r_prev, n_k, rest_next)
            carry = torch.einsum("zorns,rmnk->zomks", carry, G.to(carry.dtype))
            carry = carry.reshape(B, -1, r_k, rest_next)

        out = carry.reshape(*lead, self.out_features)
        if self.bias is not None:
            out = out + self.bias.to(out.dtype)
        return out

    # ------------------------------------------------------------------
    def num_params(self) -> int:
        """Core parameters (excludes bias), for compression accounting."""
        return self._num_core_params

    def compression_vs_dense(self) -> float:
        """How many times smaller the cores are than the dense weight."""
        return (self.out_features * self.in_features) / self._num_core_params

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"rank={self.rank}, factors={self.row_factors}x{self.col_factors}, "
                f"bias={self.bias is not None}, ternary_cores={self.ternary_cores}, "
                f"compression={self.compression_vs_dense():.2f}x")

    # ------------------------------------------------------------------
    @classmethod
    def from_linear(cls, linear: nn.Linear, rank: int = 32,
                    row_factors: tuple[int, ...] | None = None,
                    col_factors: tuple[int, ...] | None = None,
                    ternary_cores: bool = False,
                    allow_expansion: bool = False) -> "TTLinear":
        """TT-SVD warm-start from a trained dense layer.

        The route for compressing an existing checkpoint (followed by a short
        recovery fine-tune, as in ``hvn/edge/recovery.py``) rather than training
        TT from scratch.
        """
        from .decomposition import TTMatrix

        module = cls(linear.in_features, linear.out_features,
                     bias=linear.bias is not None, rank=rank,
                     row_factors=row_factors, col_factors=col_factors,
                     ternary_cores=ternary_cores,
                     allow_expansion=allow_expansion)
        tt = TTMatrix.from_dense(
            linear.weight.data.float(), module.row_factors, module.col_factors, rank
        )
        with torch.no_grad():
            for param, core in zip(module.cores, tt.cores):
                if param.shape != core.shape:
                    raise RuntimeError(
                        f"TT-SVD produced core {tuple(core.shape)} but the layer "
                        f"expects {tuple(param.shape)} -- rank truncation disagreed"
                    )
                param.copy_(core.to(param.dtype))
            if module.bias is not None:
                module.bias.copy_(linear.bias.data)
        return module.to(linear.weight.device)
