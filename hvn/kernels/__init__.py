"""
HVN Custom Kernels (Issue #33)
===============================

Fused Triton kernels for the Hamiltonian Vortex Network.

Key kernels:
  - ``fused_vortex_step``: Vorticity projection + RoPE + Biot-Savart velocity +
    gated deflection + output projection + LayerNorm in one kernel.
  - ``fused_hamiltonian_update``: Velocity normalization + projector + H·P·H +
    entanglement blend + Frobenius norm clamp in one kernel.

All kernels are **dtype-agnostic** — they accept a ``dtype`` parameter that
selects between fp16, fp32, int8, and ternary weight formats.  This forward-
compatibility design means Issue #32 (Extreme Weight Compression) can add
new dtype branches without rewriting the kernel infrastructure.
"""

from .hamiltonian_kernel import (
    fused_vortex_step,
    fused_hamiltonian_update,
    FusedHVNStep,
)

__all__ = [
    "fused_vortex_step",
    "fused_hamiltonian_update",
    "FusedHVNStep",
]