"""Known-answer and anti-vacuity tests for TTLinear (Issue #161).

Run: ``python tests/test_tt_linear.py``

The falsifiability question these exist to answer, asked before they were run:
**if the TT contraction ordering were wrong, would these catch it?**

Yes. A wrong permute/reshape in the core contraction produces a *different
matrix*, so the parity test's error would be O(1), not 1e-5. That is the single
likeliest bug in TT code and it is what test 1 is aimed at.

Test 4 exists so the parity test cannot become vacuous: at exact-reconstruction
rank the TT is deliberately larger than dense, which is a valid known-answer
setting but NOT a compression, and a reader could mistake one for the other.
"""

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hvn.edge.tt_linear import (  # noqa: E402
    TTLinear, balanced_factors, full_tt_rank, tt_num_params,
)

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


# ---------------------------------------------------------------------------
def test_known_answer_forward_matches_dense() -> None:
    """FULL-rank TT built from a dense layer must reproduce its forward exactly."""
    print("\n[1] known-answer: full-rank TT == dense nn.Linear (forward)")
    torch.manual_seed(0)
    for (out_f, in_f) in [(384, 384), (256, 128), (768, 384)]:
        lin = nn.Linear(in_f, out_f, bias=False)
        rf, cf = balanced_factors(out_f), balanced_factors(in_f)
        r_full = full_tt_rank(rf, cf)
        tt = TTLinear.from_linear(lin, rank=r_full, allow_expansion=True)

        x = torch.randn(4, 7, in_f)
        err = (tt(x) - lin(x)).abs().max().item()
        check(f"forward parity [{out_f},{in_f}] r={r_full}", err < 1e-4, f"max|err|={err:.3e}")

        w_err = (tt.to_dense() - lin.weight.data).abs().max().item()
        check(f"weight reconstruction [{out_f},{in_f}]", w_err < 1e-4, f"max|err|={w_err:.3e}")


def test_known_answer_backward_matches_dense() -> None:
    """Gradients w.r.t. the INPUT must match dense too -- forward parity alone
    would pass even if the backward graph were wrong."""
    print("\n[2] known-answer: full-rank TT == dense nn.Linear (backward)")
    torch.manual_seed(0)
    in_f, out_f = 128, 256
    lin = nn.Linear(in_f, out_f, bias=False)
    rf, cf = balanced_factors(out_f), balanced_factors(in_f)
    tt = TTLinear.from_linear(lin, rank=full_tt_rank(rf, cf), allow_expansion=True)

    x1 = torch.randn(3, 5, in_f, requires_grad=True)
    x2 = x1.detach().clone().requires_grad_(True)
    lin(x1).pow(2).sum().backward()
    tt(x2).pow(2).sum().backward()
    err = (x1.grad - x2.grad).abs().max().item()
    check("input-gradient parity", err < 1e-3, f"max|err|={err:.3e}")

    check("cores receive gradient",
          all(c.grad is not None and c.grad.abs().sum() > 0 for c in tt.cores))


def test_truncated_rank_compresses_and_trains() -> None:
    """A realistic (truncated) rank must actually be smaller AND learnable."""
    print("\n[3] truncated rank: compresses, and gradients flow")
    torch.manual_seed(0)
    tt = TTLinear(384, 384, rank=32)
    dense_params = 384 * 384
    check("cores smaller than dense",
          tt.num_params() < dense_params,
          f"{tt.num_params():,} vs {dense_params:,} ({tt.compression_vs_dense():.2f}x)")

    x = torch.randn(2, 6, 384)
    tt(x).pow(2).sum().backward()
    check("all cores get nonzero grad",
          all(c.grad is not None and c.grad.abs().sum() > 0 for c in tt.cores))

    # A few optimiser steps must reduce a fixed target loss -- a layer that is
    # smaller but cannot learn is worthless, and shape tests alone miss that.
    torch.manual_seed(1)
    layer = TTLinear(64, 64, rank=8)
    xb, yb = torch.randn(16, 64), torch.randn(16, 64)
    opt = torch.optim.Adam(layer.parameters(), lr=1e-2)
    first = torch.nn.functional.mse_loss(layer(xb), yb).item()
    for _ in range(50):
        opt.zero_grad()
        loss = torch.nn.functional.mse_loss(layer(xb), yb)
        loss.backward()
        opt.step()
    check("loss decreases under training", loss.item() < first,
          f"{first:.4f} -> {loss.item():.4f}")


def test_fused_and_direct_paths_agree() -> None:
    """The two contraction orders must be the same function.

    'fused' folds the cores then GEMMs; 'direct' streams the batch through the
    cores. They are different code paths over the same math, so a divergence
    here is a bug in one of them -- and would otherwise show up only as an arm
    of an A/B quietly computing something else.
    """
    print("\n[4] fused and direct contraction agree")
    torch.manual_seed(0)
    for (out_f, in_f, rank) in [(384, 384, 32), (256, 128, 16), (768, 384, 8)]:
        fused = TTLinear(in_f, out_f, rank=rank, bias=True, contract="fused")
        direct = TTLinear(in_f, out_f, rank=rank, bias=True, contract="direct")
        with torch.no_grad():
            for a, b in zip(direct.cores, fused.cores):
                a.copy_(b)
            direct.bias.copy_(fused.bias)

        x = torch.randn(3, 9, in_f)
        err = (fused(x) - direct(x)).abs().max().item()
        check(f"forward agrees [{out_f},{in_f}] r={rank}", err < 1e-4,
              f"max|err|={err:.3e}")

    # Gradients into the CORES must agree too -- forward parity alone would pass
    # if one path's backward were wrong.
    xg = torch.randn(2, 5, 128)
    f2 = TTLinear(128, 128, rank=8, contract="fused")
    d2 = TTLinear(128, 128, rank=8, contract="direct")
    with torch.no_grad():
        for a, b in zip(d2.cores, f2.cores):
            a.copy_(b)
    f2(xg).pow(2).sum().backward()
    d2(xg).pow(2).sum().backward()
    gerr = max((a.grad - b.grad).abs().max().item() for a, b in zip(f2.cores, d2.cores))
    check("core gradients agree", gerr < 1e-3, f"max|err|={gerr:.3e}")


def test_anti_vacuity_guards() -> None:
    """The guards whose only job is to fail."""
    print("\n[4] anti-vacuity: the size guard and the rank bound")
    rf, cf = balanced_factors(384), balanced_factors(384)
    r_full = full_tt_rank(rf, cf)

    # The parity rank is deliberately an EXPANSION -- assert that, so nobody
    # reads test 1 as evidence that TT compresses.
    check("exact-reconstruction rank is larger than dense",
          tt_num_params(rf, cf, r_full) > 384 * 384,
          f"{tt_num_params(rf, cf, r_full):,} vs {384*384:,}")

    refused = False
    try:
        TTLinear(384, 384, rank=r_full)
    except ValueError:
        refused = True
    check("constructor refuses a non-compressing rank", refused)

    # And the compressing rank must genuinely truncate, or test 3 is a no-op.
    check("rank 32 truncates (r < full rank)", 32 < r_full, f"32 < {r_full}")


def test_factorisation_helpers() -> None:
    print("\n[5] factorisation helpers")
    check("balanced_factors(384) == (16, 24)", balanced_factors(384) == (16, 24))
    check("balanced_factors(768) == (24, 32)", balanced_factors(768) == (24, 32))
    for n in (64, 128, 256, 384, 512, 768, 896, 1024, 2048):
        a, b = balanced_factors(n)
        if a * b != n:
            check(f"balanced_factors({n}) multiplies back", False)
            return
    check("balanced_factors multiplies back for all tested widths", True)


def test_ternary_cores_optin() -> None:
    """Ternary on the cores is OFF by default -- stacking it silently would
    confound any TT-vs-ternary measurement."""
    print("\n[6] ternary-core stacking is opt-in")
    tt = TTLinear(128, 128, rank=8)
    check("default is ternary_cores=False", tt.ternary_cores is False)

    torch.manual_seed(0)
    plain = TTLinear(128, 128, rank=8)
    tern = TTLinear(128, 128, rank=8, ternary_cores=True)
    with torch.no_grad():
        for a, b in zip(tern.cores, plain.cores):
            a.copy_(b)
    x = torch.randn(2, 128)
    diff = (tern(x) - plain(x)).abs().max().item()
    check("ternary_cores=True changes the forward", diff > 1e-6, f"max|diff|={diff:.3e}")

    tern(x).pow(2).sum().backward()
    check("STE keeps gradient flowing through ternary cores",
          all(c.grad is not None and c.grad.abs().sum() > 0 for c in tern.cores))


if __name__ == "__main__":
    print("=" * 70)
    print("TTLinear known-answer tests (Issue #161)")
    print("=" * 70)
    test_known_answer_forward_matches_dense()
    test_known_answer_backward_matches_dense()
    test_truncated_rank_compresses_and_trains()
    test_fused_and_direct_paths_agree()
    test_anti_vacuity_guards()
    test_factorisation_helpers()
    test_ternary_cores_optin()
    print("\n" + "=" * 70)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    if FAIL:
        for n in FAIL:
            print(f"  FAILED: {n}")
    print("=" * 70)
    sys.exit(1 if FAIL else 0)
