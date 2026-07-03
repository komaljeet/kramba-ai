"""
Hamiltonian Vortex Network (HVN) — Core Engine
==============================================

This package implements the HVN architecture, a novel sequence-modeling
approach that fuses Lagrangian Vortex Fluid Dynamics with Quantum Matrix
Product State entanglement to achieve O(1) per-token inference cost.

Modules
-------
vortex_injection   : Translates token embeddings into discrete velocity fields
                     via a modified Biot-Savart law.
hamiltonian_memory : Fixed-size Matrix Product State (Tensor Train) that
                     evolves deterministically on each forward pass.
hvn_block          : Wrapper linking the fluid vortex step to quantum state
                     evolution.
decoder            : Shallow feed-forward decoder projecting the Hamiltonian
                     state to vocabulary logits.
model              : Full HVN language model assembling all components.
"""

from .vortex_injection import VortexInjectionLayer
from .hamiltonian_memory import HamiltonianMemoryBlock
from .batched_memory import MultiHeadHamiltonianMemory, MultiThreadHamiltonianMemory
from .batched_vortex import MultiHeadVortexInjection, MultiThreadVortexInjection
from .hvn_block import HVNBlock
from .decoder import HVNDecoder
from .model import HamiltonianVortexNetwork, CUDAGraphRunner

__all__ = [
    "VortexInjectionLayer",
    "HamiltonianMemoryBlock",
    "MultiHeadHamiltonianMemory",
    "MultiThreadHamiltonianMemory",
    "MultiHeadVortexInjection",
    "MultiThreadVortexInjection",
    "HVNBlock",
    "HVNDecoder",
    "HamiltonianVortexNetwork",
    "CUDAGraphRunner",
]