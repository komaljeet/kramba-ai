"""TTLinear wired into DeltaNetBlock / EdgeHVN (Issue #161).

Run: ``python tests/test_tt_deltanet.py``

Falsifiability, asked before running: if the wiring silently did nothing -- the
most likely failure, since a model with dense layers trains and loads perfectly
well -- would these catch it? Yes: test 2 asserts the targeted modules are
actually ``TTLinear`` instances AND that the parameter count drops. Either alone
is weak (a flag could be recorded without taking effect; a count could drop for
an unrelated reason), so both are checked.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hvn.core.deltanet import DeltaNetBlock  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402
from hvn.edge.tt_linear import TTLinear  # noqa: E402

PASS, FAIL = [], []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  -- {detail}" if detail else ""))


def count(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ---------------------------------------------------------------------------
def test_default_is_byte_identical_dense() -> None:
    """No tt_targets must leave the block exactly as it was."""
    print("\n[1] default (no tt_targets) is unchanged")
    torch.manual_seed(0)
    blk = DeltaNetBlock(d_model=128, n_heads=4)
    check("q_proj still nn.Linear", isinstance(blk.q_proj, nn.Linear))
    check("out_proj still nn.Linear", isinstance(blk.out_proj, nn.Linear))
    check("ffn[0] still nn.Linear", isinstance(blk.ffn[0], nn.Linear))
    check("tt_targets empty", blk.tt_targets == ())


def test_targets_actually_take_effect() -> None:
    """The wiring must change the modules AND shrink the parameter count."""
    print("\n[2] tt_targets replace the right modules and shrink the block")
    torch.manual_seed(0)
    dense = DeltaNetBlock(d_model=384, n_heads=6)
    torch.manual_seed(0)
    tt = DeltaNetBlock(d_model=384, n_heads=6,
                       tt_targets=("qkv", "out", "ffn"), tt_rank=32)

    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        check(f"{name} is TTLinear", isinstance(getattr(tt, name), TTLinear))
    check("ffn[0] is TTLinear", isinstance(tt.ffn[0], TTLinear))
    check("ffn[3] is TTLinear", isinstance(tt.ffn[3], TTLinear))

    # Untargeted layers must be untouched -- especially the zero-init gates.
    check("alpha_proj left dense", isinstance(tt.alpha_proj, nn.Linear))
    check("beta_proj left dense", isinstance(tt.beta_proj, nn.Linear))
    check("alpha_proj still zero-initialised",
          tt.alpha_proj.weight.abs().max().item() == 0.0)

    d_n, t_n = count(dense), count(tt)
    check("parameter count drops", t_n < d_n,
          f"{d_n:,} -> {t_n:,} ({d_n / t_n:.2f}x smaller)")


def test_forward_and_backward_run() -> None:
    print("\n[3] TT block trains end-to-end")
    torch.manual_seed(0)
    blk = DeltaNetBlock(d_model=128, n_heads=4, tt_targets=("qkv", "out", "ffn"), tt_rank=8)
    x = torch.randn(2, 16, 128)
    out = blk(x)
    check("forward shape preserved", out.shape == x.shape, str(tuple(out.shape)))
    check("forward is finite", torch.isfinite(out).all().item())

    out.pow(2).sum().backward()
    tt_cores = [p for m in blk.modules() if isinstance(m, TTLinear) for p in m.cores]
    check("every TT core receives gradient",
          all(c.grad is not None and c.grad.abs().sum() > 0 for c in tt_cores),
          f"{len(tt_cores)} cores")


def test_edgehvn_plumbing() -> None:
    print("\n[4] EdgeHVN plumbs tt_targets to its DeltaNet layers")
    torch.manual_seed(0)
    common = dict(vocab_size=256, d_model=128, num_layers=4, n_heads=4,
                  use_deltanet=True)
    dense = EdgeHVN(**common)
    tt = EdgeHVN(**common, tt_targets=("qkv", "out", "ffn"), tt_rank=8)

    n_tt = sum(1 for m in tt.modules() if isinstance(m, TTLinear))
    check("TTLinear layers present in EdgeHVN", n_tt > 0, f"{n_tt} layers")
    check("config recorded on the model", tt.tt_targets == ("qkv", "out", "ffn"))
    check("rank recorded on the model", tt.tt_rank == 8)

    d_n, t_n = count(dense), count(tt)
    check("EdgeHVN parameter count drops", t_n < d_n,
          f"{d_n:,} -> {t_n:,} ({d_n / t_n:.2f}x smaller)")

    ids = torch.randint(0, 256, (2, 16))
    logits = tt(ids)
    logits = logits[0] if isinstance(logits, tuple) else logits
    check("EdgeHVN forward runs", logits.shape[:2] == (2, 16), str(tuple(logits.shape)))
    check("EdgeHVN forward finite", torch.isfinite(logits).all().item())


def test_hybrid_plan_survives() -> None:
    """TT must not disturb attention layers in a hybrid plan."""
    print("\n[5] hybrid block_plan: TT touches DeltaNet layers only")
    torch.manual_seed(0)
    from hvn.core.transformer_block import TransformerBlock
    m = EdgeHVN(vocab_size=256, d_model=128, num_layers=4, n_heads=4,
                block_plan=["deltanet", "transformer", "deltanet", "transformer"],
                tt_targets=("qkv",), tt_rank=8)
    attn = [b for b in m.blocks if isinstance(b, TransformerBlock)]
    check("attention layers present", len(attn) == 2, f"{len(attn)} found")
    tt_in_attn = [mm for b in attn for mm in b.modules() if isinstance(mm, TTLinear)]
    check("attention layers untouched by TT", len(tt_in_attn) == 0,
          f"{len(tt_in_attn)} TTLinear found inside attention")


def test_invalid_targets_are_refused() -> None:
    """Bad targets must raise, not be silently dropped."""
    print("\n[6] invalid targets raise rather than silently no-op")
    for bad in ("alpha", "alpha_proj", "beta", "everything"):
        raised = False
        try:
            DeltaNetBlock(d_model=128, n_heads=4, tt_targets=(bad,), tt_rank=8)
        except ValueError:
            raised = True
        check(f"tt_targets=({bad!r},) refused", raised)

    raised = False
    try:
        DeltaNetBlock(d_model=128, n_heads=4, use_swiglu=True,
                      tt_targets=("ffn",), tt_rank=8)
    except ValueError:
        raised = True
    check("'ffn' target refused on SwiGLU FFN", raised)


def test_ternary_does_not_silently_stack() -> None:
    """apply_ternary_to_model must skip TT layers -- two compression schemes on
    one weight would confound any measurement of either."""
    print("\n[7] ternary skips TT layers (no silent double-compression)")
    from hvn.edge.ternary import apply_ternary_to_model, TernaryLinear

    torch.manual_seed(0)
    blk = DeltaNetBlock(d_model=128, n_heads=4, tt_targets=("qkv",), tt_rank=8)
    apply_ternary_to_model(blk)

    check("q_proj stayed TTLinear", isinstance(blk.q_proj, TTLinear))
    check("q_proj did NOT become TernaryLinear", not isinstance(blk.q_proj, TernaryLinear))
    # A dense, untargeted layer in the same block SHOULD have been converted --
    # otherwise this test would pass even if apply_ternary_to_model did nothing.
    check("untargeted alpha_proj was ternarised (control)",
          isinstance(blk.alpha_proj, TernaryLinear))


if __name__ == "__main__":
    print("=" * 70)
    print("TT wiring into DeltaNet / EdgeHVN (Issue #161)")
    print("=" * 70)
    test_default_is_byte_identical_dense()
    test_targets_actually_take_effect()
    test_forward_and_backward_run()
    test_edgehvn_plumbing()
    test_hybrid_plan_survives()
    test_invalid_targets_are_refused()
    test_ternary_does_not_silently_stack()
    print("\n" + "=" * 70)
    print(f"{len(PASS)} passed, {len(FAIL)} failed")
    for n in FAIL:
        print(f"  FAILED: {n}")
    print("=" * 70)
    sys.exit(1 if FAIL else 0)
