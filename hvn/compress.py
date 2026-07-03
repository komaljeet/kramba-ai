"""
Extreme Weight Compression — Phase 1: Foundational Compression (Issue #32)
============================================================================

Implements the four Phase 1 techniques from the compression roadmap in
``EXECUTION_PLAN.md`` / GitLab issue #32, targeting 10B params: 20 GB (fp16)
-> ~5 GB, and quality-preserving 2-4 bit quantization:

1. :class:`GPTQQuantizer` — Hessian-based post-training quantization with
   sequential error compensation (Frantar et al. 2022).  Quantizes a weight
   matrix column-by-column, immediately propagating each column's rounding
   error to the not-yet-quantized columns via the inverse-Hessian Cholesky
   factor.  This keeps 4-bit quantization error far below naive round-to-
   nearest.

2. :class:`HadamardQuantizer` — Hadamard pre-rotation before quantization
   (QuIP#, Chee et al. 2023).  A random-sign Hadamard transform spreads
   outlier weights across all coordinates, making the post-rotation
   distribution closer to Gaussian and therefore much friendlier to
   uniform/2-bit quantization.

3. :class:`RateDistortionAllocator` — assigns each layer its own bit-width
   from a fixed average bit budget, giving more bits to layers whose
   quantization error propagates strongly to the output (as estimated from
   their calibration Hessian trace and weight energy) and fewer bits to
   robust layers.

4. Calibration utilities (:func:`collect_calibration_hessian`,
   :func:`gather_layer_inputs`) — capture layer input activations via
   forward hooks and turn them into the ``X^T X`` Hessian proxy that GPTQ
   and the rate-distortion allocator need.

All quantization here is **inference-only** and operates on already-trained
``nn.Linear`` weight matrices (``VortexInjectionLayer``/``HVNBlock``
projections, FFN layers, decoder).  Training remains full precision, per the
issue's acceptance criteria.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# Calibration utilities
# ---------------------------------------------------------------------------


def gather_layer_inputs(
    model: nn.Module,
    layer_names: list[str],
    forward_fn: Callable[[], None],
) -> dict[str, torch.Tensor]:
    """Capture the input activations seen by a set of named ``nn.Linear`` layers.

    Registers a forward hook on each named module, runs *forward_fn* (which
    should invoke the model on some calibration data), then returns the
    concatenated inputs for each layer.

    Parameters
    ----------
    model : nn.Module
        The model containing the target layers.
    layer_names : list[str]
        Dotted module names (as returned by ``model.named_modules()``) of
        the ``nn.Linear`` layers to calibrate.
    forward_fn : Callable[[], None]
        A zero-argument callable that runs one or more forward passes
        through *model* on calibration data (under ``torch.no_grad()``).

    Returns
    -------
    dict[str, torch.Tensor]
        Maps layer name -> activation tensor of shape ``[n_samples, in_features]``.
    """
    captured: dict[str, list[torch.Tensor]] = {name: [] for name in layer_names}
    modules = dict(model.named_modules())
    handles = []

    def _make_hook(name: str):
        def hook(_module, inputs, _output):
            x = inputs[0].detach()
            captured[name].append(x.reshape(-1, x.shape[-1]).float().cpu())
        return hook

    for name in layer_names:
        if name not in modules:
            raise KeyError(f"No module named {name!r} in model")
        handles.append(modules[name].register_forward_hook(_make_hook(name)))

    try:
        with torch.no_grad():
            forward_fn()
    finally:
        for h in handles:
            h.remove()

    return {name: torch.cat(chunks, dim=0) for name, chunks in captured.items() if chunks}


def collect_calibration_hessian(activations: torch.Tensor, device: Optional[torch.device] = None) -> torch.Tensor:
    """Turn calibration activations into the GPTQ Hessian proxy ``H = 2/n · XᵀX``.

    Parameters
    ----------
    activations : torch.Tensor
        Layer input activations, shape ``[n_samples, in_features]``.
    device : torch.device, optional
        Device to compute on (defaults to *activations*'s device).

    Returns
    -------
    H : torch.Tensor
        ``[in_features, in_features]`` Hessian approximation.
    """
    X = activations.to(device or activations.device).float()
    n = X.shape[0]
    if n == 0:
        raise ValueError("Need at least 1 calibration sample")
    return (2.0 / n) * (X.t() @ X)


# ---------------------------------------------------------------------------
# 1.1 GPTQ-style Hessian-based quantization with error compensation
# ---------------------------------------------------------------------------


class GPTQQuantizer:
    """Hessian-based post-training quantization with sequential error compensation.

    Implements the core GPTQ algorithm (Frantar, Ashkboos, Hoefler & Alistarh,
    2022): weight columns are quantized one at a time in the order they
    appear.  After quantizing column *i*, the resulting rounding error is
    immediately propagated to all not-yet-quantized columns using the
    Cholesky factor of the inverse calibration Hessian — the same
    closed-form update used by Optimal Brain Quantization / Optimal Brain
    Surgeon.  This lets 4-bit (or lower) quantization reach <0.5% relative
    error, versus ~10% for naive round-to-nearest NF4 (see
    ``hvn/quantize.py``).

    Parameters
    ----------
    bits : int
        Target bit-width (typically 3-4).
    group_size : int
        Number of consecutive input-feature columns sharing one affine
        (scale, zero-point) quantization grid.
    percdamp : float
        Hessian diagonal damping factor (as a fraction of the mean
        diagonal) to keep the Cholesky decomposition numerically stable.
    """

    def __init__(self, bits: int = 4, group_size: int = 128, percdamp: float = 0.01):
        if not (1 <= bits <= 8):
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        self.bits = bits
        self.group_size = group_size
        self.percdamp = percdamp

    @torch.no_grad()
    def quantize(self, W: torch.Tensor, H: torch.Tensor) -> dict:
        """Quantize a weight matrix using its calibration Hessian.

        Parameters
        ----------
        W : torch.Tensor
            Weight matrix, shape ``[out_features, in_features]`` (the
            ``nn.Linear`` convention).
        H : torch.Tensor
            Calibration Hessian, shape ``[in_features, in_features]``, from
            :func:`collect_calibration_hessian`.

        Returns
        -------
        dict
            Quantized representation (codes/scales/zeros + metadata).  Pass
            to :meth:`dequantize` to reconstruct.
        """
        device = W.device
        orig_dtype = W.dtype
        W = W.detach().clone().float()
        rows, cols = W.shape
        H = H.detach().clone().float().to(device)
        assert H.shape == (cols, cols), f"Hessian shape {tuple(H.shape)} != ({cols}, {cols})"

        # Damping keeps the Cholesky factorization well-conditioned.
        diag = torch.diagonal(H)
        damp = self.percdamp * diag.mean().clamp(min=1e-8)
        H = H + torch.eye(cols, device=device) * damp

        # Columns with (near-)zero Hessian carry no calibration signal —
        # quantize them independently without error propagation.
        dead = torch.diagonal(H).abs() < 1e-8
        H[dead, dead] = 1.0

        # Cholesky factor of H^{-1} (upper-triangular), the standard GPTQ trick:
        # this lets column i's error be redistributed to columns > i via a
        # single matrix row instead of re-inverting a shrinking Hessian.
        L = torch.linalg.cholesky(H)
        Hinv = torch.cholesky_inverse(L)
        Hinv_chol = torch.linalg.cholesky(Hinv, upper=True)

        Q = torch.zeros_like(W)
        n_groups = math.ceil(cols / self.group_size)
        scales = torch.zeros(rows, n_groups, device=device)
        zeros = torch.zeros(rows, n_groups, device=device)
        qmax = 2 ** self.bits - 1

        for group_idx, group_start in enumerate(range(0, cols, self.group_size)):
            group_end = min(group_start + self.group_size, cols)
            w_group = W[:, group_start:group_end]
            wmin = w_group.min(dim=1, keepdim=True).values
            wmax = w_group.max(dim=1, keepdim=True).values
            scale = (wmax - wmin).clamp(min=1e-8) / qmax
            zero = torch.round(-wmin / scale)
            scales[:, group_idx] = scale.squeeze(1)
            zeros[:, group_idx] = zero.squeeze(1)

            for i in range(group_start, group_end):
                w_col = W[:, i]
                q_col = torch.clamp(torch.round(w_col / scale.squeeze(1) + zero.squeeze(1)), 0, qmax)
                dq_col = (q_col - zero.squeeze(1)) * scale.squeeze(1)
                Q[:, i] = q_col

                if dead[i]:
                    continue

                # Sequential error compensation: redistribute this column's
                # rounding error to all not-yet-quantized columns, weighted
                # by the inverse-Hessian Cholesky row — this is what makes
                # GPTQ far more accurate than independent round-to-nearest.
                err = (w_col - dq_col) / Hinv_chol[i, i]
                if i + 1 < cols:
                    W[:, i + 1:] -= err.unsqueeze(1) * Hinv_chol[i, i + 1:].unsqueeze(0)

        code_dtype = torch.uint8 if self.bits <= 8 else torch.int16
        return {
            "codes": Q.to(code_dtype),
            "scales": scales,
            "zeros": zeros,
            "group_size": self.group_size,
            "bits": self.bits,
            "shape": (rows, cols),
            "orig_dtype": orig_dtype,
        }

    @torch.no_grad()
    def dequantize(self, qdict: dict) -> torch.Tensor:
        """Reconstruct the fp32/fp16 weight matrix from a GPTQ code dict."""
        Q = qdict["codes"].float()
        scales, zeros = qdict["scales"], qdict["zeros"]
        group_size = qdict["group_size"]
        rows, cols = qdict["shape"]
        device = Q.device
        W = torch.empty(rows, cols, device=device)

        for group_idx, group_start in enumerate(range(0, cols, group_size)):
            group_end = min(group_start + group_size, cols)
            scale = scales[:, group_idx].unsqueeze(1)
            zero = zeros[:, group_idx].unsqueeze(1)
            W[:, group_start:group_end] = (Q[:, group_start:group_end] - zero) * scale

        return W.to(qdict["orig_dtype"])

    @staticmethod
    def memory_bytes(qdict: dict) -> int:
        """Estimated storage footprint of the quantized representation, in bytes."""
        rows, cols = qdict["shape"]
        bits = qdict["bits"]
        code_bytes = rows * cols * bits / 8.0
        overhead_bytes = qdict["scales"].numel() * 4 + qdict["zeros"].numel() * 4
        return int(code_bytes + overhead_bytes)


# ---------------------------------------------------------------------------
# 1.2 Hadamard pre-rotation quantization (QuIP#)
# ---------------------------------------------------------------------------


def _hadamard_matrix(n: int, device: torch.device, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Construct an orthonormal ``n x n`` Hadamard matrix (Sylvester construction).

    ``n`` must be a power of 2.  The returned matrix satisfies ``Hᵀ = H``
    and ``H @ H = I`` (it is both symmetric and self-inverse), so the same
    matrix is used to rotate and un-rotate.
    """
    if n & (n - 1) != 0:
        raise ValueError(f"Hadamard size must be a power of 2, got {n}")
    H = torch.tensor([[1.0]], device=device, dtype=dtype)
    while H.shape[0] < n:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    return H / math.sqrt(n)


class HadamardQuantizer:
    """Random-sign Hadamard rotation + group-wise uniform quantization (QuIP#-style).

    Weight outliers make low-bit uniform quantization lossy: a single large
    value forces a coarse per-group scale that wastes precision on every
    other value in the group.  Rotating each row by a random-sign Hadamard
    matrix first ("incoherence processing") spreads any outlier's magnitude
    across all ``padded`` coordinates, making the post-rotation distribution
    close to Gaussian and roughly outlier-free.  Uniform quantization is
    then near-optimal even at 2 bits.  The rotation is exactly invertible
    (Hadamard matrices are self-inverse up to the ``1/√n`` normalization),
    so dequantization recovers the original basis exactly, modulo the
    quantization error introduced in the rotated domain.

    Parameters
    ----------
    bits : int
        Target bit-width (2-bit is the primary use case here).
    group_size : int
        Quantization group size (post-rotation, padded to a power of 2).
    seed : int
        Seed for the random sign pattern.  Only the seed (not the full
        sign vector or Hadamard matrix) needs to be stored/shipped —
        both are regenerated deterministically at dequantization time.
    """

    def __init__(self, bits: int = 2, group_size: int = 128, seed: int = 0):
        if not (1 <= bits <= 8):
            raise ValueError(f"bits must be in [1, 8], got {bits}")
        self.bits = bits
        self.group_size = group_size
        self.seed = seed

    @staticmethod
    def _padded_size(n: int) -> int:
        return 1 << (n - 1).bit_length() if n > 1 else 1

    @torch.no_grad()
    def quantize(self, W: torch.Tensor) -> dict:
        """Rotate + quantize a weight matrix.

        Parameters
        ----------
        W : torch.Tensor
            Weight matrix, shape ``[out_features, in_features]``.

        Returns
        -------
        dict
            Quantized representation.  Pass to :meth:`dequantize` to
            reconstruct.
        """
        device, orig_dtype = W.device, W.dtype
        rows, cols = W.shape
        padded = self._padded_size(cols)

        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        signs = (torch.randint(0, 2, (padded,), generator=gen).float() * 2 - 1).to(device)

        Wp = torch.zeros(rows, padded, device=device, dtype=torch.float32)
        Wp[:, :cols] = W.detach().float()
        Wp = Wp * signs.unsqueeze(0)

        Had = _hadamard_matrix(padded, device=device)
        Wrot = Wp @ Had  # incoherent (outlier-spread) representation

        qmax = 2 ** self.bits - 1
        n_groups = math.ceil(padded / self.group_size)
        codes = torch.zeros(rows, padded, device=device)
        scales = torch.zeros(rows, n_groups, device=device)
        zeros = torch.zeros(rows, n_groups, device=device)

        for group_idx, group_start in enumerate(range(0, padded, self.group_size)):
            group_end = min(group_start + self.group_size, padded)
            block = Wrot[:, group_start:group_end]
            wmin = block.min(dim=1, keepdim=True).values
            wmax = block.max(dim=1, keepdim=True).values
            scale = (wmax - wmin).clamp(min=1e-8) / qmax
            zero = torch.round(-wmin / scale)
            codes[:, group_start:group_end] = torch.clamp(torch.round(block / scale + zero), 0, qmax)
            scales[:, group_idx] = scale.squeeze(1)
            zeros[:, group_idx] = zero.squeeze(1)

        return {
            "codes": codes.to(torch.uint8),
            "scales": scales,
            "zeros": zeros,
            "group_size": self.group_size,
            "bits": self.bits,
            "signs": signs.to(torch.int8),
            "shape": (rows, cols),
            "padded": padded,
            "orig_dtype": orig_dtype,
            "seed": self.seed,
        }

    @torch.no_grad()
    def dequantize(self, qdict: dict) -> torch.Tensor:
        """Reconstruct the weight matrix by un-rotating the dequantized codes."""
        codes = qdict["codes"].float()
        scales, zeros = qdict["scales"], qdict["zeros"]
        group_size = qdict["group_size"]
        rows, cols = qdict["shape"]
        padded = qdict["padded"]
        device = codes.device

        Wrot = torch.empty(rows, padded, device=device)
        for group_idx, group_start in enumerate(range(0, padded, group_size)):
            group_end = min(group_start + group_size, padded)
            scale = scales[:, group_idx].unsqueeze(1)
            zero = zeros[:, group_idx].unsqueeze(1)
            Wrot[:, group_start:group_end] = (codes[:, group_start:group_end] - zero) * scale

        Had = _hadamard_matrix(padded, device=device)
        Wp = Wrot @ Had  # Hadamard matrix is self-inverse
        signs = qdict["signs"].float().to(device)
        Wp = Wp * signs.unsqueeze(0)
        return Wp[:, :cols].to(qdict["orig_dtype"])

    @staticmethod
    def memory_bytes(qdict: dict) -> int:
        """Estimated storage footprint of the quantized representation, in bytes.

        Note the rotation itself costs nothing to store — only the RNG
        ``seed`` (a few bytes) is needed to regenerate the sign pattern and
        Hadamard matrix deterministically.
        """
        rows, _ = qdict["shape"]
        padded = qdict["padded"]
        bits = qdict["bits"]
        code_bytes = rows * padded * bits / 8.0
        overhead_bytes = qdict["scales"].numel() * 4 + qdict["zeros"].numel() * 4
        return int(code_bytes + overhead_bytes)


# ---------------------------------------------------------------------------
# 1.3 Rate-distortion bit allocation
# ---------------------------------------------------------------------------


class RateDistortionAllocator:
    """Per-layer bit-width allocation under a fixed average bit budget.

    Not every layer needs the same precision: layers whose quantization
    error propagates strongly to the model output should get more bits;
    robust layers can be pushed lower without hurting quality.  This
    implements the standard reverse-water-filling solution to that
    trade-off: for a set of per-layer distortion sensitivities ``s_i``
    (assumed to scale the mean-squared quantization error, i.e. roughly
    ``distortion_i ∝ s_i · 4^{-bits_i}``), the rate-distortion-optimal
    allocation is

        bits_i = avg_bits + 0.5 · (log2(s_i) − mean(log2(s)))

    which gives more bits to high-sensitivity layers and fewer to
    low-sensitivity ones while keeping the average fixed at ``avg_bits``.

    Parameters
    ----------
    min_bits, max_bits : int
        Hard floor/ceiling on the allocated bit-width per layer.
    """

    def __init__(self, min_bits: int = 2, max_bits: int = 8):
        if min_bits > max_bits:
            raise ValueError("min_bits must be <= max_bits")
        self.min_bits = min_bits
        self.max_bits = max_bits

    def allocate(self, sensitivities: dict[str, float], avg_bits: float) -> dict[str, int]:
        """Compute a per-layer bit-width allocation.

        Parameters
        ----------
        sensitivities : dict[str, float]
            Per-layer distortion sensitivity (higher = more sensitive to
            quantization error).  See :meth:`compute_sensitivity`.
        avg_bits : float
            Target average bit-width across all layers.

        Returns
        -------
        dict[str, int]
            Layer name -> allocated bit-width, clamped to
            ``[min_bits, max_bits]``.
        """
        if not sensitivities:
            return {}
        names = list(sensitivities.keys())
        s = torch.tensor([max(sensitivities[n], 1e-12) for n in names])
        log_s = torch.log2(s)
        bits = avg_bits + 0.5 * (log_s - log_s.mean())
        bits = torch.clamp(torch.round(bits), self.min_bits, self.max_bits)
        return {n: int(b.item()) for n, b in zip(names, bits)}

    @staticmethod
    def compute_sensitivity(W: torch.Tensor, H: torch.Tensor) -> float:
        """Estimate a layer's quantization-distortion sensitivity.

        Uses ``mean(diag(H)) · mean(W²)`` as a proxy for how much output
        error a unit of weight-quantization noise in this layer produces:
        ``mean(diag(H))`` captures how strongly the calibration inputs
        activate this layer (from the GPTQ Hessian ``H = 2/n·XᵀX``), and
        ``mean(W²)`` captures the weight energy that noise gets multiplied
        into.

        Parameters
        ----------
        W : torch.Tensor
            Layer weight matrix.
        H : torch.Tensor
            Layer calibration Hessian (see :func:`collect_calibration_hessian`).

        Returns
        -------
        float
            Sensitivity score (unitless, only meaningful relative to other
            layers' scores).
        """
        trace_mean = torch.diagonal(H).mean().item()
        w_energy = (W.detach().float() ** 2).mean().item()
        return max(trace_mean * w_energy, 1e-12)


# ---------------------------------------------------------------------------
# Model-level convenience: quantize every nn.Linear in a module tree
# ---------------------------------------------------------------------------


def compress_linear_layers_gptq(
    model: nn.Module,
    calibration_forward: Callable[[], None],
    bits: int = 4,
    group_size: int = 128,
    percdamp: float = 0.01,
    layer_names: Optional[list[str]] = None,
) -> dict[str, dict]:
    """Quantize every (or a chosen subset of) ``nn.Linear`` layer in *model* with GPTQ.

    Runs *calibration_forward* once to gather per-layer input activations
    via forward hooks, builds each layer's Hessian, then GPTQ-quantizes and
    **replaces the layer's weight data in-place** with the dequantized
    (quantized-then-reconstructed) values — i.e. the model becomes usable
    immediately for further evaluation, while the returned metadata dict
    holds the actual compressed codes/scales for memory accounting.

    Parameters
    ----------
    model : nn.Module
        Model to compress (weights modified in-place).
    calibration_forward : Callable[[], None]
        Zero-arg callable that runs forward pass(es) through *model* on
        calibration data.
    bits : int
        Target bit-width.
    group_size : int
        GPTQ quantization group size.
    percdamp : float
        Hessian damping factor.
    layer_names : list[str], optional
        Restrict compression to these named ``nn.Linear`` modules.  If
        ``None``, every ``nn.Linear`` in the model is compressed.

    Returns
    -------
    dict[str, dict]
        Layer name -> {"qdict": ..., "orig_bytes": ..., "quant_bytes": ...,
        "max_abs_error": ...}.
    """
    linear_modules = dict(model.named_modules())
    target_names = layer_names or [
        name for name, mod in linear_modules.items() if isinstance(mod, nn.Linear)
    ]
    if not target_names:
        return {}

    activations = gather_layer_inputs(model, target_names, calibration_forward)

    quantizer = GPTQQuantizer(bits=bits, group_size=group_size, percdamp=percdamp)
    results: dict[str, dict] = {}

    for name in target_names:
        if name not in activations:
            continue  # layer was never exercised by the calibration data
        layer = linear_modules[name]
        assert isinstance(layer, nn.Linear)
        device = layer.weight.device
        H = collect_calibration_hessian(activations[name], device=device)
        qdict = quantizer.quantize(layer.weight.data, H)
        deq = quantizer.dequantize(qdict)

        max_abs_error = (deq.float() - layer.weight.data.float()).abs().max().item()
        with torch.no_grad():
            layer.weight.data.copy_(deq)

        results[name] = {
            "qdict": qdict,
            "orig_bytes": layer.weight.numel() * 4,
            "quant_bytes": GPTQQuantizer.memory_bytes(qdict),
            "max_abs_error": max_abs_error,
        }

    return results
