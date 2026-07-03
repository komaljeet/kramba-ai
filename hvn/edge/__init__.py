"""
HVN Edge — Extreme Weight Compression Architecture Variant (Issue #32)
========================================================================

A separate, edge-deployment-oriented variant of the Hamiltonian Vortex
Network that stacks every compression technique from the issue #32 roadmap
(``EXECUTION_PLAN.md`` / ``related.md``) on top of the reference
architecture in ``hvn/core``.  It lives in its own sub-package rather than
as flags on the reference model because several of these techniques
(ternary weight training, shared weights, tensor-train Hamiltonians)
genuinely change training dynamics and are not safe defaults for the
reference implementation's benchmarked quality numbers.

Sub-modules (one per phase of the roadmap; Phase 1 already lives in
``hvn/compress.py`` and ``hvn/sparse.py``):

- ``ternary``       — Phase 2.1: BitNet b1.58 ternary weight training
- ``decomposition``  — Phase 2.2/2.3, 3.5: low-rank+sparse, Tensor Train,
  MERA, Kronecker-factored and circulant/Toeplitz structured weights
- ``shared``         — Phase 2.4: ALBERT-style weight sharing across layers
- ``entropy``        — Phase 3.1, 9.1: entropy (Huffman) and LDPC coding
- ``transform``      — Phase 3.1/3.2: wavelet and DCT domain compression
- ``holographic``    — Phase 3.3/2.3: holographic and superposition weight
  encoding across layers
- ``streaming``       — Phase 4: NVMe layer streaming, memory-mapped
  weight files, succinct sparse indexing, task-specific loading
- ``recovery``        — Phase 5: simulated-annealing pruning, topological
  pruning, free-energy-optimal quantization levels, population coding
- ``model``           — ``EdgeHVN``: assembles the above into a runnable
  model variant
"""

from .model import EdgeHVN
from .shared import SharedWeightHVN
from .ternary import TernaryLinear, TernaryHamiltonian, apply_ternary_to_model, ternary_quantize_ste

__all__ = [
    "EdgeHVN",
    "SharedWeightHVN",
    "TernaryLinear",
    "TernaryHamiltonian",
    "apply_ternary_to_model",
    "ternary_quantize_ste",
]
