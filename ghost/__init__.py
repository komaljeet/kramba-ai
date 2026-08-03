"""GHOST — Gradient Heuristic Omission via Step Teleportation.

Written by: Rambarun Komaljeet

Training-acceleration research package (GitLab epic #124). Most training steps
between phase transitions are hypothesised to follow predictable weight-update
patterns that can be extrapolated rather than computed; only phase transitions
need real backpropagation.

Phase 1 (#125): ``ghost.detector``     — phase-transition detection from loss logs.
Phase 2 (#126): ``ghost.extrapolator`` — weight-delta extrapolation (not yet built).
Phase 3 (#127): ``ghost.trainer``      — the teleporting training loop (not yet built).
"""

__all__ = ["detector"]
