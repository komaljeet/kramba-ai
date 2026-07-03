"""
Quality Recovery at Extreme Compression (Issue #32, Phase 5)
================================================================

Techniques for maintaining quality once weights have been pushed to
ternary/2-bit/heavily-pruned representations:

- :class:`AnnealingPruner` — simulated-annealing search for a pruning
  mask (escapes the local minima that greedy magnitude pruning gets stuck
  in).
- :class:`TopologicalPruner` — multi-scale connectivity ("0-dimensional
  persistent-homology-inspired") pruning: keeps weights that remain part
  of a connected high-magnitude structure across many filtration
  thresholds, prunes short-lived/isolated ("topological noise") weights.
- :class:`FreeEnergyQuantizer` — Lloyd-Max optimal scalar quantization
  levels for the empirical weight distribution, instead of fixed
  uniform/NF4 levels.
- :class:`PopulationCoding` — redundant low-bit encoding across several
  independently-dithered "population members," reconstructed by
  averaging (reduces effective quantization noise).
- :func:`recovery_finetune_step` — a thin, documented pointer to the
  existing QLoRA support (``hvn/lora.py``, Issue #29) as the recommended
  post-compression fine-tuning recovery path.

Note on scope: a full ``gudhi``/``ripser``-based multi-dimensional
persistent homology pipeline, and a full LDPC belief-propagation decoder,
are substantial standalone projects; per the issue's own effort/benefit
estimates for these techniques (1.5-2x, 1.1-1.3x respectively) they are
implemented here as honestly-simplified but functioning versions of the
same underlying idea rather than the full literature implementation.
"""

from __future__ import annotations

import math
from collections import Counter

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# 5.1 Simulated annealing for optimal weight pruning
# ---------------------------------------------------------------------------


class AnnealingPruner:
    """Simulated-annealing search for a pruning mask (Boltzmann 1877).

    Greedy magnitude-based pruning is a single-pass local heuristic.
    Starting from the greedy solution, this performs randomized
    swap proposals (move one currently-pruned weight back to "kept" while
    pruning a currently-kept weight instead), accepting improvements
    always and worse swaps with Metropolis probability ``exp(-Δ/T)`` as
    the temperature ``T`` is cooled — escaping local minima that pure
    greedy pruning cannot.

    Saliency uses the OBD/OBS-style proxy ``w_i² · H_ii`` (squared weight
    times its calibration-Hessian diagonal), the same second-order
    signal GPTQ uses for quantization error compensation.

    Parameters
    ----------
    sparsity : float
        Target fraction of weights to prune.
    iterations : int
        Number of annealing swap proposals.
    T0, T_min : float
        Initial and final temperature (exponential cooling schedule).
    seed : int
        RNG seed.
    """

    def __init__(self, sparsity: float = 0.9, iterations: int = 500, T0: float = 1.0, T_min: float = 0.01, seed: int = 0):
        self.sparsity = sparsity
        self.iterations = iterations
        self.T0 = T0
        self.T_min = T_min
        self.seed = seed

    @torch.no_grad()
    def prune(self, W: torch.Tensor, H_diag: torch.Tensor) -> torch.Tensor:
        """Find a pruning mask using simulated annealing over OBD/OBS saliency.

        Parameters
        ----------
        W : torch.Tensor
            Weight matrix, shape ``[out_features, in_features]``.
        H_diag : torch.Tensor
            Diagonal of the calibration Hessian, shape ``[in_features]``.
        """
        device = W.device
        flat_w = W.detach().flatten().float()
        h_expanded = H_diag.detach().float().repeat(W.shape[0])
        saliency = (flat_w ** 2) * h_expanded
        n = flat_w.numel()
        k_prune = int(n * self.sparsity)

        order = torch.argsort(saliency)  # ascending: lowest saliency first
        pruned_idx = order[:k_prune].tolist()
        kept_idx = order[k_prune:].tolist()

        cur_energy = sum(saliency[i].item() for i in pruned_idx)
        best_energy = cur_energy
        best_pruned_idx = list(pruned_idx)

        gen = torch.Generator().manual_seed(self.seed)
        for it in range(self.iterations):
            T = self.T0 * (self.T_min / self.T0) ** (it / max(1, self.iterations - 1))
            pos_i = torch.randint(0, len(pruned_idx), (1,), generator=gen).item()
            pos_j = torch.randint(0, len(kept_idx), (1,), generator=gen).item()
            i, j = pruned_idx[pos_i], kept_idx[pos_j]
            delta = (saliency[j] - saliency[i]).item()  # energy change if we keep i, prune j instead
            if delta < 0 or torch.rand(1, generator=gen).item() < math.exp(-delta / max(T, 1e-8)):
                pruned_idx[pos_i], kept_idx[pos_j] = j, i
                cur_energy += delta
                if cur_energy < best_energy:
                    best_energy = cur_energy
                    best_pruned_idx = list(pruned_idx)

        # Simulated annealing explores away from the greedy start (that's the point —
        # it can escape local minima) but the *final* state after cooling isn't
        # guaranteed to be the best one visited, so we track and return the best.
        mask = torch.ones(n, dtype=torch.bool, device=device)
        mask[torch.tensor(best_pruned_idx, device=device)] = False
        return mask.reshape(W.shape)


# ---------------------------------------------------------------------------
# 5.2 / 10.1 Topological (multi-scale connectivity) pruning
# ---------------------------------------------------------------------------


class TopologicalPruner:
    """Multi-scale connectivity pruning, inspired by 0-dimensional persistent homology.

    Sweeps a magnitude filtration across several thresholds; at each
    threshold, computes 4-connected components of the "active"
    (above-threshold) weight positions.  A weight's **persistence score**
    is how many threshold levels it remains part of a connected structure
    of size ``>= min_component_size`` (rather than an isolated point).
    Weights with low persistence are "topological noise" — magnitude
    spikes that never connect into a larger structure — and are pruned
    first, keeping small-magnitude weights that are structurally
    persistent over ones with slightly larger magnitude but no structural
    support.

    Parameters
    ----------
    sparsity : float
        Target fraction of weights to prune.
    num_levels : int
        Number of filtration thresholds to sweep.
    min_component_size : int
        Minimum connected-component size to count as "structural" (not noise).
    """

    def __init__(self, sparsity: float = 0.5, num_levels: int = 8, min_component_size: int = 3):
        self.sparsity = sparsity
        self.num_levels = num_levels
        self.min_component_size = min_component_size

    @staticmethod
    def _component_sizes(active: torch.Tensor) -> torch.Tensor:
        rows, cols = active.shape
        n = rows * cols
        parent = list(range(n))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

        active_flat = active.flatten().tolist()
        for r in range(rows):
            for c in range(cols):
                idx = r * cols + c
                if not active_flat[idx]:
                    continue
                if c + 1 < cols and active_flat[idx + 1]:
                    union(idx, idx + 1)
                if r + 1 < rows and active_flat[idx + cols]:
                    union(idx, idx + cols)

        roots = [find(i) for i in range(n)]
        counts = Counter(roots[i] for i in range(n) if active_flat[i])
        sizes = torch.zeros(n, device=active.device)
        for i in range(n):
            if active_flat[i]:
                sizes[i] = counts[roots[i]]
        return sizes.reshape(rows, cols)

    @torch.no_grad()
    def prune(self, W: torch.Tensor) -> torch.Tensor:
        mag = W.detach().abs()
        quantile_probs = torch.linspace(0.1, 0.95, self.num_levels, device=mag.device)
        thresholds = torch.quantile(mag.flatten().float(), quantile_probs)
        persistence = torch.zeros_like(mag)
        for t in thresholds:
            active = mag >= t
            sizes = self._component_sizes(active)
            persistence += (active & (sizes >= self.min_component_size)).float()

        k_prune = int(mag.numel() * self.sparsity)
        # Rank by persistence first (ascending = noisiest first), magnitude as tiebreak.
        score = persistence.flatten() * (mag.max() + 1) + mag.flatten()
        order = torch.argsort(score)
        mask = torch.ones(mag.numel(), dtype=torch.bool, device=W.device)
        mask[order[:k_prune]] = False
        return mask.reshape(W.shape)


# ---------------------------------------------------------------------------
# 6.2 Free-energy-optimal quantization levels (Lloyd-Max)
# ---------------------------------------------------------------------------


class FreeEnergyQuantizer:
    """Lloyd-Max optimal scalar quantizer.

    Minimizing the expected quantization distortion for a fixed number of
    levels is exactly the free-energy-minimization argument in the issue:
    the Boltzmann-optimal quantization levels for an empirical
    distribution are its Lloyd-Max centroids, found here by iterative
    1D k-means (assign each weight to its nearest level, recompute each
    level as the mean of its assigned weights, repeat to convergence).
    This adapts to the actual weight distribution instead of assuming
    fixed uniform or NF4 levels.

    Parameters
    ----------
    num_levels : int
        Number of quantization levels (e.g. 16 for 4-bit).
    iterations : int
        Maximum Lloyd-Max iterations.
    """

    def __init__(self, num_levels: int = 16, iterations: int = 30):
        self.num_levels = num_levels
        self.iterations = iterations

    @torch.no_grad()
    def fit_quantize(self, W: torch.Tensor) -> dict:
        flat = W.detach().flatten().float()
        quantiles = torch.linspace(0, 1, self.num_levels + 2, device=flat.device)[1:-1]
        levels = torch.quantile(flat, quantiles)

        for _ in range(self.iterations):
            dist = (flat.unsqueeze(1) - levels.unsqueeze(0)).abs()
            assign = dist.argmin(dim=1)
            new_levels = levels.clone()
            for k in range(self.num_levels):
                sel = flat[assign == k]
                if sel.numel() > 0:
                    new_levels[k] = sel.mean()
            if torch.allclose(new_levels, levels, atol=1e-6):
                levels = new_levels
                break
            levels = new_levels

        dist = (flat.unsqueeze(1) - levels.unsqueeze(0)).abs()
        code_dtype = torch.uint8 if self.num_levels <= 256 else torch.int16
        codes = dist.argmin(dim=1).to(code_dtype)
        return {"codes": codes.reshape(W.shape), "levels": levels, "shape": tuple(W.shape), "orig_dtype": W.dtype}

    @staticmethod
    def dequantize(qdict: dict) -> torch.Tensor:
        return qdict["levels"][qdict["codes"].long()].to(qdict["orig_dtype"])

    @staticmethod
    def memory_bytes(qdict: dict) -> int:
        bits = max(1, math.ceil(math.log2(qdict["levels"].numel())))
        code_bytes = qdict["codes"].numel() * bits / 8.0
        level_bytes = qdict["levels"].numel() * 4
        return int(code_bytes + level_bytes)


# ---------------------------------------------------------------------------
# 13.2 Population coding (noise-robust redundant weight representation)
# ---------------------------------------------------------------------------


class PopulationCoding:
    """Redundant low-bit encoding across several independently-dithered "population members".

    Encodes the same weight matrix ``population_size`` times, each with an
    independent random dither offset applied before rounding.  Averaging
    the ``population_size`` dequantized copies at reconstruction time
    reduces the effective quantization noise variance by roughly
    ``1/population_size`` — the same principle behind population coding
    in neuroscience, where a population of individually-noisy neurons
    collectively encodes a value more precisely than any single neuron.
    This lets more aggressive (lower-bit) per-member quantization be used
    while still hitting a target reconstruction accuracy, at the cost of
    ``population_size×`` the storage of a single low-bit copy.

    Parameters
    ----------
    bits : int
        Bits per population member.
    population_size : int
        Number of redundant (dithered) encodings.
    seed : int
        RNG seed for the dither pattern.
    """

    def __init__(self, bits: int = 2, population_size: int = 4, seed: int = 0):
        self.bits = bits
        self.population_size = population_size
        self.seed = seed

    @torch.no_grad()
    def encode(self, W: torch.Tensor) -> dict:
        device = W.device
        qmax = 2 ** self.bits - 1
        wmin, wmax = W.min(), W.max()
        scale = (wmax - wmin).clamp(min=1e-8) / qmax
        gen = torch.Generator(device="cpu").manual_seed(self.seed)

        codes_list = []
        for _ in range(self.population_size):
            dither = (torch.rand(W.shape, generator=gen).to(device) - 0.5) * scale
            q = torch.clamp(torch.round((W - wmin + dither) / scale), 0, qmax)
            codes_list.append(q.to(torch.uint8))

        return {
            "codes": torch.stack(codes_list, dim=0), "scale": scale, "wmin": wmin,
            "bits": self.bits, "shape": tuple(W.shape), "orig_dtype": W.dtype,
            "population_size": self.population_size,
        }

    @staticmethod
    def decode(qdict: dict) -> torch.Tensor:
        codes = qdict["codes"].float()
        recon = (codes * qdict["scale"] + qdict["wmin"]).mean(dim=0)
        return recon.to(qdict["orig_dtype"])

    @staticmethod
    def memory_bytes(qdict: dict) -> int:
        return int(qdict["codes"].numel() * qdict["bits"] / 8.0)


# ---------------------------------------------------------------------------
# 5.5 QLoRA fine-tuning recovery (thin pointer to existing hvn/lora.py, Issue #29)
# ---------------------------------------------------------------------------


def recovery_finetune_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    batch_x: torch.Tensor,
    batch_y: torch.Tensor,
    criterion: nn.Module,
) -> float:
    """One recovery fine-tuning step after extreme compression.

    A thin wrapper — the actual QLoRA machinery (quantized base +
    trainable low-rank adapters) already exists in ``hvn/lora.py``
    (``apply_qlora``, Issue #29).  The recommended Phase 5.5 recovery
    recipe is: compress the model (ternary/GPTQ/pruning/etc.), call
    ``model.apply_qlora()`` (or ``apply_lora`` for a non-quantized base),
    then run a short fine-tune using this step function to recover most
    of the quality lost to compression, without re-training from scratch.
    """
    model.train()
    logits = model(batch_x)
    loss = criterion(logits.reshape(-1, logits.size(-1)), batch_y.reshape(-1))
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return loss.item()
