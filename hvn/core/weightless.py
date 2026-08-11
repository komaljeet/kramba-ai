"""
Weightless (seed-regenerated) projections for the DeltaNet block — Issue #40
============================================================================

**FROZEN: research legacy (2026-08-11).** Imported only by the frozen
``hvn/core/model.py``; ``EdgeHVN`` — the production model — does not reach it.
No new features, no parity expectations, kept importable. See
docs/architecture.md § "Layering".

Goal (from the issue): *"instead of using weights maybe use something else to
keep file size and memory used low? We already won the no-qkv battle."*

The Hamiltonian ``H`` is already weightless — it is runtime state, never trained
or stored (that was the no-KV / no-qkv win).  What remains on disk are the
*trained* tensors: the embedding, the decoder, and the per-block projections
(``q/k/v/out`` and the small gate projections).

Feasibility (per the maintainer's decision note on #40): a **fully** weightless
net is not viable without handing back most of the DeltaNet quality win, because
that win came *specifically* from the **learned, data-dependent dynamics**
(``α_t = σ(W_α x)``, ``β_t = σ(W_β x)``, the SiLU output gate, and the causal
conv).  So we go **hybrid**:

* **Freeze + seed** the four big linear "address/content" projections of the
  DeltaNet block — ``q_proj``, ``k_proj``, ``v_proj``, ``out_proj`` (each
  ``d_model × d_model``).  Each becomes a fixed random matrix regenerated from a
  PRNG **seed** on load, so it costs *one integer* on disk instead of ``d²``
  floats.  This is the classic random-features trick: ``q``/``k`` are
  L2-normalized *directions* (a random basis still spans the space), and the
  data-dependent gates on top can still steer the dynamics.
* **Keep learned** the tiny high-leverage core: the decay gate ``α``, the write
  gate ``β``, the output gate ``g``, the depthwise conv, and the per-head read
  RMSNorm.  These are small (``O(d)`` or ``O(d·H)``) but carry the DeltaNet win.

Result: O(1) inference / no-KV / fixed d×d state are all untouched (freezing a
projection is orthogonal to them), the four ``d²`` projection matrices per layer
vanish from the checkpoint, and only a fraction of the learned quality is at
risk — quantified by the A/B in ``test_issue40.py``.

Everything here is opt-in (``weightless=True``); the default DeltaNet and HVN
paths are byte-for-byte unchanged.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .deltanet import DeltaNetBlock


# A large odd stride so per-projection seeds derived from one base seed are
# well separated (avoids accidental collisions between layers/projections).
_SEED_STRIDE = 0x9E3779B1  # 32-bit golden-ratio constant


def _derive_seed(base: int, salt: int) -> int:
    """Deterministically mix a base seed with a per-projection salt (0..2^31)."""
    return (int(base) + (int(salt) + 1) * _SEED_STRIDE) & 0x7FFFFFFF


class SeededLinear(nn.Module):
    """A frozen, bias-free linear layer whose weight is regenerated from a seed.

    Call-compatible with ``nn.Linear(in_features, out_features, bias=False)`` so
    it can be dropped into :class:`DeltaNetBlock` without touching its forward.

    The weight is a **non-persistent buffer** (``persistent=False``) so it is
    *excluded* from ``state_dict`` — nothing about it is written to the
    checkpoint except the tiny integer ``seed`` (kept as a Python attribute and
    round-tripped via the model config).  On construction the exact same matrix
    is regenerated with a dedicated ``torch.Generator`` seeded by ``seed``, so
    the layer is deterministic across save/load and CPU/GPU.

    Initialization matches ``nn.Linear``'s default (Kaiming-uniform,
    ``a=sqrt(5)``) so a weightless projection is drawn from the same
    distribution a trained one started from.
    """

    def __init__(self, in_features: int, out_features: int, seed: int):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.seed = int(seed)
        weight = self._generate()
        # Non-persistent: stays out of state_dict (the file-size win).
        self.register_buffer("weight", weight, persistent=False)

    def _generate(self) -> torch.Tensor:
        # Generate on CPU with a seeded generator for cross-device determinism,
        # then .to() moves it with the module in the standard way.
        gen = torch.Generator(device="cpu").manual_seed(self.seed)
        weight = torch.empty(self.out_features, self.in_features)
        # Same scheme as nn.Linear.reset_parameters (Kaiming-uniform a=sqrt(5)).
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5, generator=gen)
        return weight

    def _apply(self, fn, recurse=True):
        # Ensure the regenerated buffer follows .to()/.cuda()/.float() like the
        # rest of the module (buffers already do; this is just belt-and-braces
        # for subclasses that override _apply).
        return super()._apply(fn, recurse=recurse)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight.to(x.dtype))

    def extra_repr(self) -> str:
        return (f"in_features={self.in_features}, out_features={self.out_features}, "
                f"bias=False, seeded=True, seed={self.seed}")


class WeightlessDeltaNetBlock(DeltaNetBlock):
    """DeltaNet block with **seed-regenerated (frozen) q/k/v/out projections**.

    Identical dynamics and forward/forward_step to :class:`DeltaNetBlock` — the
    only change is that the four ``d_model × d_model`` projections are replaced
    by :class:`SeededLinear`, so they are frozen random features stored as a
    single seed rather than ``4·d²`` trained floats.  The learned core (decay
    gate ``α``, write gate ``β``, output gate ``g``, causal conv, read RMSNorm,
    FFN, norm) is untouched.

    Parameters
    ----------
    seed : int
        Base PRNG seed.  Per-projection seeds are derived deterministically
        (``layer_idx`` + projection index) so every frozen matrix is distinct
        yet reproducible from this one integer.
    layer_idx : int
        Index of this block in the stack (mixes into the derived seeds so
        different layers get different frozen projections).
    """

    # Which of the block's projections to freeze/seed.  All four d×d matrices
    # (the bulk of the block's parameters) — the learned dynamics are the small
    # gate projections, which stay trainable.
    _FROZEN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "out_proj")

    def __init__(self, *args, seed: int = 1234, layer_idx: int = 0, **kwargs):
        super().__init__(*args, **kwargs)
        self.weightless_seed = int(seed)
        self.layer_idx = int(layer_idx)
        d = self.d_model
        for i, name in enumerate(self._FROZEN_PROJECTIONS):
            salt = self.layer_idx * (len(self._FROZEN_PROJECTIONS) + 1) + i
            proj_seed = _derive_seed(seed, salt)
            setattr(self, name, SeededLinear(d, d, seed=proj_seed))

    def num_frozen_params(self) -> int:
        """Number of float weights removed from the checkpoint by seeding."""
        return len(self._FROZEN_PROJECTIONS) * self.d_model * self.d_model
