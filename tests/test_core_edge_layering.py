"""The `core/` vs `edge/` layering, pinned so it cannot drift (Issue #151).

Author: Rambarun Komaljeet

docs/architecture.md § "Layering" records that `hvn/core/` is a **shared block
library**, not a second architecture: EdgeHVN reaches 13 of its 16 modules, and
only `model.py` (HamiltonianVortexNetwork) and `weightless.py` are legacy-only.

A doc that says this will go stale silently. These tests recompute it from the
import graph, so the day someone makes `core/model.py` reachable from EdgeHVN —
or moves a block into `edge/` and inverts the dependency — the suite says so
instead of the docs quietly becoming wrong.
"""

import collections
import pathlib
import re

CORE = pathlib.Path(__file__).resolve().parent.parent / "hvn" / "core"

#: EdgeHVN's direct core imports (hvn/edge/model.py) plus the training/eval
#: machinery it uses. Kept explicit rather than parsed so a silent removal of an
#: import from edge/model.py cannot silently shrink the "must stay live" set.
EDGE_ROOTS = ["decoder", "hvn_block", "deltanet", "transformer_block",
              "packing", "block_coordinate", "moe", "activations"]
LEGACY_ROOTS = ["model"]


def _graph():
    mods = {p.stem: p for p in CORE.glob("*.py") if p.stem != "__init__"}
    edges = collections.defaultdict(set)
    for name, p in mods.items():
        src = p.read_text(encoding="utf-8", errors="replace")
        for m in re.finditer(r"^\s*from \.(\w+) import", src, re.M):
            if m.group(1) in mods:
                edges[name].add(m.group(1))
    return mods, edges


def _reach(roots, edges):
    seen, stack = set(), list(roots)
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        stack.extend(edges[n] - seen)
    return seen


def test_edge_does_not_depend_on_the_frozen_model():
    """The freeze is only meaningful if production never reaches it."""
    _mods, edges = _graph()
    assert "model" not in _reach(EDGE_ROOTS, edges), (
        "EdgeHVN now reaches hvn/core/model.py, which is marked FROZEN. Either "
        "the dependency is a mistake, or the freeze in docs/architecture.md and "
        "the module docstring is now wrong.")
    assert "weightless" not in _reach(EDGE_ROOTS, edges)


def test_legacy_only_set_is_exactly_model_and_weightless():
    """Pins the documented table. If this changes, the docs must change."""
    _mods, edges = _graph()
    legacy_only = _reach(LEGACY_ROOTS, edges) - _reach(EDGE_ROOTS, edges)
    assert legacy_only == {"model", "weightless"}, (
        f"legacy-only set changed to {sorted(legacy_only)}; update "
        f"docs/architecture.md § Layering")


def test_the_shared_blocks_really_are_shared():
    """Guards the claim that motivated NOT splitting the packages.

    If a future refactor moved DeltaNet into edge/, this fails -- which is the
    point: duplicating deltanet.py + deltanet_triton.py is the failure mode the
    layering note exists to prevent.
    """
    _mods, edges = _graph()
    shared = _reach(EDGE_ROOTS, edges) & _reach(LEGACY_ROOTS, edges)
    for must in ("deltanet", "hvn_block", "decoder"):
        assert must in shared, f"{must} is no longer shared by both models"


def test_frozen_files_say_so():
    """A freeze nobody can see is not a freeze."""
    for name in ("model", "weightless"):
        head = (CORE / f"{name}.py").read_text(encoding="utf-8",
                                               errors="replace")[:1200]
        assert "FROZEN" in head, f"hvn/core/{name}.py lost its freeze marker"
