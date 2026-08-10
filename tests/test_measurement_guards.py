"""Guards that stop a measurement from being silently unrepresentative (#136).

Written by: Rambarun Komaljeet

Two failures on 2026-08-10 motivated these, and neither raised an error at the
time -- both produced confident numbers:

  - `eval/vram_envelope.py` measured `HamiltonianVortexNetwork` (no block_plan,
    no attention layers) at vocab 8000, and its rows were quoted in
    docs/training.md as the sizing table for a hybrid model at vocab 50257.
  - `scripts/size_sweep.py` called `build_model_and_config` directly, so it never
    reached main() where --fast is applied, and reported eager throughput as if
    it were training throughput. It published "#136's config spills at 105% of
    VRAM, 30.6 days". Correct answer: 82.6%, 9.0 days.

The head-dim rule is the third: it was documented in
hvn/kernels/hamiltonian_kernel.py:293 and enforced nowhere, so a bad shape
crashed mid-run with an InductorError naming `tl.arange`.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_edge_conversational import apply_fast_stack, check_head_dim  # noqa: E402


# --------------------------------------------------------------------------
# check_head_dim
# --------------------------------------------------------------------------

@pytest.mark.parametrize("d_model,n_heads,head_dim", [
    (512, 8, 64),
    (768, 12, 64),
    (896, 14, 64),
    (1024, 8, 128),
    (1024, 16, 64),
])
def test_power_of_two_head_dims_pass(d_model, n_heads, head_dim):
    assert d_model // n_heads == head_dim
    check_head_dim(d_model, n_heads)  # must not raise


@pytest.mark.parametrize("d_model,n_heads,bad_head_dim", [
    (640, 8, 80),
    (768, 8, 96),    # the exact config #136 proposed -- would die under --fast
    (896, 8, 112),
])
def test_non_power_of_two_head_dims_raise(d_model, n_heads, bad_head_dim):
    with pytest.raises(ValueError, match="not a power of 2"):
        check_head_dim(d_model, n_heads)


def test_error_names_the_actual_numbers():
    """A guard that fires with an unreadable message just moves the confusion."""
    with pytest.raises(ValueError) as exc:
        check_head_dim(768, 8)
    msg = str(exc.value)
    assert "768" in msg and "8" in msg and "96" in msg
    assert "tl.arange" in msg          # names the real failure site
    assert "n_heads=12" in msg          # and the fix


def test_suggestion_is_actually_valid():
    """The suggested n_heads must itself pass the check -- an unchecked
    suggestion is how a helpful message sends someone into the same crash."""
    with pytest.raises(ValueError) as exc:
        check_head_dim(896, 8)
    msg = str(exc.value)
    import re
    m = re.search(r"n_heads=(\d+)", msg)
    assert m, f"no suggestion in: {msg}"
    check_head_dim(896, int(m.group(1)))  # must not raise


def test_indivisible_is_caught_separately():
    with pytest.raises(ValueError, match="not divisible"):
        check_head_dim(768, 7)


def test_check_is_skipped_without_triton():
    """The constraint belongs to the Triton kernel. Without it, 96 is fine."""
    check_head_dim(768, 8, triton_scan=False)


def test_guard_is_not_vacuous():
    """Fails if check_head_dim is ever turned into a no-op.

    Every other test here would still pass against `def check_head_dim(*a): pass`
    except the ones asserting a raise -- so this states the intent directly: the
    common, plausible-looking config MUST be rejected.
    """
    raised = False
    try:
        check_head_dim(768, 8)
    except ValueError:
        raised = True
    assert raised, "check_head_dim accepted head_dim=96 -- the guard is dead"


# --------------------------------------------------------------------------
# apply_fast_stack
# --------------------------------------------------------------------------

class _FakeBlock(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.use_triton_scan = False
        self.use_megakernel_scan = False


class _FakeModel(torch.nn.Module):
    def __init__(self, n=4):
        super().__init__()
        self.blocks = torch.nn.ModuleList([_FakeBlock() for _ in range(n)])


def test_fast_stack_enables_triton_on_every_block():
    m = _FakeModel()
    assert not any(b.use_triton_scan for b in m.blocks)
    apply_fast_stack(m, compile_model=False, verbose=False)
    assert all(b.use_triton_scan for b in m.blocks), \
        "a block was left on the slow path -- partial application is the bug"


def test_megakernel_requires_triton():
    """Without the Triton scan there is no scan backward to swap."""
    m = _FakeModel()
    apply_fast_stack(m, triton_scan=False, megakernel_scan=True,
                     compile_model=False, verbose=False)
    assert not any(b.use_megakernel_scan for b in m.blocks)


def test_megakernel_applies_with_triton():
    m = _FakeModel()
    apply_fast_stack(m, triton_scan=True, megakernel_scan=True,
                     compile_model=False, verbose=False)
    assert all(b.use_megakernel_scan for b in m.blocks)


def test_tf32_toggle():
    apply_fast_stack(_FakeModel(), tf32=True, compile_model=False, verbose=False)
    assert torch.backends.cuda.matmul.allow_tf32 is True


def test_disabling_everything_changes_nothing():
    """The off path must be a genuine no-op, so an A/B arm that needs bare
    eager (the fp32 comparison) is not silently accelerated."""
    m = _FakeModel()
    out = apply_fast_stack(m, triton_scan=False, megakernel_scan=False,
                           tf32=False, compile_model=False, verbose=False)
    assert out is m
    assert not any(b.use_triton_scan for b in m.blocks)
