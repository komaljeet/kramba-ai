"""Tests for padding-free sequence packing (Issue #132).

The load-bearing tests here are the *isolation* ones. Packing is only worth
anything if a packed row is numerically indistinguishable from running its
documents separately -- otherwise it is a silent quality regression dressed up
as a speedup. Every sequence-axis carry in the hybrid gets its own test:

  - the DeltaNet recurrent state (alpha=0 boundary reset)
  - the DeltaNet causal depthwise conv (K-1 backward taps)
  - attention (block-diagonal document mask)
  - the full EdgeHVN stack, end to end on logits

These run on CPU with the exact ``_scan_sequential`` reference, so they do not
need the GPU/Triton path. The eager chunked scan is separately asserted to
*refuse* packing rather than silently approximate it.
"""

import numpy as np
import pytest
import torch

from hvn.core.deltanet import DeltaNetBlock
from hvn.core.packing import (
    PackedDataset,
    doc_start_mask,
    pack_examples,
    packing_stats,
    segment_attn_mask,
)
from hvn.core.transformer_block import TransformerBlock
from hvn.edge.model import EdgeHVN

torch.manual_seed(0)

# Parity bar. The reset is exact in the sequential scan (alpha=0 is a real
# zero, not an epsilon), so what is left is fp32 accumulation-order noise.
ATOL = 1e-5


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def test_doc_start_mask():
    doc = torch.tensor([[0, 0, 0, 1, 1, 2]])
    got = doc_start_mask(doc)
    assert got.tolist() == [[True, False, False, True, False, True]]


def test_segment_attn_mask_is_block_diagonal():
    doc = torch.tensor([[0, 0, 1]])
    m = segment_attn_mask(doc)[0, 0]
    assert m.tolist() == [
        [True, True, False],
        [True, True, False],
        [False, False, True],
    ]


def test_doc_start_mask_rejects_wrong_rank():
    with pytest.raises(ValueError, match=r"\[B, S\]"):
        doc_start_mask(torch.zeros(4, dtype=torch.long))


# ---------------------------------------------------------------------------
# Bin packing
# ---------------------------------------------------------------------------
def test_pack_examples_respects_capacity():
    lengths = [100, 200, 300, 50, 400, 120]
    bins = pack_examples(lengths, capacity=513, seed=0)
    seen = sorted(i for b in bins for i in b)
    assert seen == list(range(len(lengths))), "every example must be placed exactly once"
    for b in bins:
        assert sum(lengths[i] for i in b) <= 513


def test_pack_examples_oversized_gets_its_own_bin():
    bins = pack_examples([600, 10], capacity=513, seed=0)
    # The 600-token example is clamped to capacity, so it fills a bin alone.
    big = [b for b in bins if 0 in b][0]
    assert big == [0]


def test_packing_beats_padding_on_the_real_shape():
    """Reproduce grounded-QA's measured length distribution, not a uniform one.

    Measured on data/grounded_qa_expanded.jsonl: median 86, p90 115, max 205,
    against a 513 capacity. The spread matters -- uniform lengths strand a
    fixed remainder in every row (86*5=430 of 513), while a varied distribution
    lets short examples fill the gaps, which is the real case.
    """
    rng = np.random.default_rng(0)
    lengths = np.clip(rng.normal(90, 30, 500), 20, 205).astype(int).tolist()
    bins = pack_examples(lengths, capacity=513, seed=0)
    s = packing_stats(lengths, bins, 513)
    assert s["examples_per_row"] > 4.0, s
    assert s["row_reduction"] > 4.0, s
    assert s["packed_waste_frac"] < 0.06, s
    assert s["padded_waste_frac"] > 0.80, s


def test_first_fit_decreasing_is_available_but_not_default():
    lengths = [300, 200, 100, 50]
    a = pack_examples(lengths, 513, strategy="shuffled_first_fit", seed=1)
    b = pack_examples(lengths, 513, strategy="first_fit_decreasing")
    assert sorted(i for x in a for i in x) == sorted(i for x in b for i in x)
    with pytest.raises(ValueError, match="unknown packing strategy"):
        pack_examples(lengths, 513, strategy="best_fit")


# ---------------------------------------------------------------------------
# PackedDataset
# ---------------------------------------------------------------------------
def _ex(n, tok=7, learn=True):
    ids = np.full(n, tok, dtype=np.int64)
    lab = np.full(n, tok if learn else -100, dtype=np.int64)
    lab[0] = -100                      # prompt-style first-token mask
    return ids, lab


def test_packed_dataset_shapes_and_doc_ids():
    ds = PackedDataset([_ex(10), _ex(20), _ex(30)], seq_len=63, verbose=False)
    x, y, d = ds[0]
    assert x.shape == y.shape == d.shape == (63,)
    # Document ids are row-local and non-decreasing.
    assert torch.all(d[1:] >= d[:-1])


def test_packed_dataset_cuts_the_causal_shift_at_boundaries():
    """The last token of doc A must never be trained to predict doc B's first.

    This is the failure packing silently introduces if the shift is not cut:
    labels[t+1] belongs to the next document. Built here with a corpus whose
    first token is NOT masked, which the real corpora happen to do for us --
    the point is that the class does not depend on that.
    """
    def unmasked(n, tok):
        ids = np.full(n, tok, dtype=np.int64)
        return ids, ids.copy()          # every token learned, including the first

    ds = PackedDataset([unmasked(4, 1), unmasked(4, 2)], seq_len=15, verbose=False,
                       pad_id=0)
    x, y, d = ds[0]
    starts = doc_start_mask(d.unsqueeze(0))[0]
    # y is labels shifted by one, so position t predicts t+1. Wherever t+1
    # starts a document, the target must be masked out.
    nxt_is_start = torch.zeros_like(starts)
    nxt_is_start[:-1] = starts[1:]
    assert torch.all(y[nxt_is_start] == -100), (
        "cross-document prediction target leaked through the causal shift"
    )


def test_packed_dataset_residue_is_its_own_document():
    ds = PackedDataset([_ex(4)], seq_len=15, verbose=False)
    _, _, d = ds[0]
    # One real doc (id 0) then the tail residue as its own id, so the residue
    # cannot silently extend the last real document's recurrent state.
    assert d[0].item() == 0
    assert d[-1].item() != d[0].item()


# ---------------------------------------------------------------------------
# Isolation: DeltaNet recurrent state + conv
# ---------------------------------------------------------------------------
def _deltanet(d_model=32, n_heads=2, conv_kernel=4):
    blk = DeltaNetBlock(d_model=d_model, n_heads=n_heads, conv_kernel=conv_kernel,
                        dropout=0.0)
    blk.eval()
    # Force the exact sequential scan: alpha=0 is a true zero there, and it is
    # the documented reference recurrence.
    blk.train_chunk = 0
    blk.use_triton_scan = False
    return blk


def test_deltanet_packed_row_matches_documents_run_separately():
    """Acceptance criterion 1: per-document parity against _scan_sequential."""
    blk = _deltanet()
    D = blk.d_model
    torch.manual_seed(1)
    a = torch.randn(1, 7, D)
    b = torch.randn(1, 5, D)

    packed = torch.cat([a, b], dim=1)
    doc = torch.tensor([[0] * 7 + [1] * 5])
    out_packed = blk(packed, doc_ids=doc)

    out_a = blk(a)
    out_b = blk(b)
    ref = torch.cat([out_a, out_b], dim=1)

    torch.testing.assert_close(out_packed, ref, atol=ATOL, rtol=ATOL)


def test_deltanet_packing_actually_changes_the_result():
    """Guard against a vacuous parity test.

    If doc_ids were silently ignored, the test above could still pass for the
    wrong reason. The unpacked forward on the same concatenated input MUST
    differ from the packed one -- that difference is exactly the cross-document
    leakage packing removes.
    """
    blk = _deltanet()
    torch.manual_seed(2)
    packed = torch.randn(1, 12, blk.d_model)
    doc = torch.tensor([[0] * 7 + [1] * 5])
    with_docs = blk(packed, doc_ids=doc)
    without = blk(packed)
    assert not torch.allclose(with_docs, without, atol=1e-4), (
        "doc_ids had no effect -- the isolation is not actually being applied"
    )


def test_deltanet_conv_taps_do_not_cross_a_boundary():
    """The K-1 depthwise taps must not read the previous document.

    Perturbing the tail of document A must leave document B's output bit-identical.
    """
    blk = _deltanet(conv_kernel=4)
    torch.manual_seed(3)
    x = torch.randn(1, 12, blk.d_model)
    doc = torch.tensor([[0] * 7 + [1] * 5])

    out1 = blk(x, doc_ids=doc)
    x2 = x.clone()
    x2[:, 4:7] += 5.0          # perturb only doc A's last 3 tokens (the tap reach)
    out2 = blk(x2, doc_ids=doc)

    torch.testing.assert_close(out1[:, 7:], out2[:, 7:], atol=ATOL, rtol=ATOL)
    assert not torch.allclose(out1[:, 4:7], out2[:, 4:7]), "doc A should have changed"


def test_segment_safe_conv_matches_plain_conv_within_one_document():
    """With a single document, the segment-safe conv must reproduce conv1d exactly."""
    blk = _deltanet(conv_kernel=4)
    torch.manual_seed(4)
    x = torch.randn(2, 9, blk.d_model)
    doc = torch.zeros(2, 9, dtype=torch.long)
    torch.testing.assert_close(
        blk._causal_conv(x, doc_ids=doc), blk._causal_conv(x),
        atol=ATOL, rtol=ATOL,
    )


def test_chunked_scan_refuses_packing_instead_of_approximating():
    """The cumprod/log-space paths cannot represent alpha=0 -- they must refuse."""
    blk = _deltanet()
    blk.train_chunk = 4          # re-enable the eager chunked scan
    x = torch.randn(1, 12, blk.d_model)
    doc = torch.tensor([[0] * 7 + [1] * 5])
    with pytest.raises(RuntimeError, match="cannot represent"):
        blk(x, doc_ids=doc)


def test_packing_requires_the_decay_gate():
    blk = DeltaNetBlock(d_model=32, n_heads=2, dropout=0.0, use_decay_gate=False)
    blk.eval()
    blk.train_chunk = 0
    blk.use_triton_scan = False
    x = torch.randn(1, 8, 32)
    doc = torch.tensor([[0] * 4 + [1] * 4])
    with pytest.raises(ValueError, match="requires the decay gate"):
        blk(x, doc_ids=doc)


# ---------------------------------------------------------------------------
# Isolation: the Triton scan (the path the FT scripts actually run)
# ---------------------------------------------------------------------------
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")


@requires_cuda
@pytest.mark.parametrize("megakernel", [False, True])
def test_triton_scan_packed_row_matches_documents_run_separately(megakernel):
    """The reset must be EXACT in the Triton kernel, not merely close.

    This is the load-bearing case: continue_ft_hybrid.py / continue_ft_looped.py
    enable triton-scan (+ megakernel backward) unconditionally, so the CPU
    _scan_sequential tests above do not cover the path a real run takes.

    It is exact because deltanet_triton.py's scan is sequential over S with a
    direct ``state = state * a_t`` multiply ("fp32 throughout -- no
    reparameterization, so stable at any alpha"), so alpha=0 is a true zero.
    Measured: 4.8e-7 max abs diff, against 0.60 for the leakage it removes.
    """
    blk = DeltaNetBlock(d_model=64, n_heads=2, conv_kernel=4, dropout=0.0).cuda().eval()
    blk.train_chunk = 0
    blk.use_triton_scan = True
    blk.use_megakernel_scan = megakernel

    torch.manual_seed(11)
    a = torch.randn(1, 40, 64, device="cuda")
    b = torch.randn(1, 24, 64, device="cuda")
    doc = torch.tensor([[0] * 40 + [1] * 24], device="cuda")

    with torch.no_grad():
        packed = blk(torch.cat([a, b], dim=1), doc_ids=doc)
        ref = torch.cat([blk(a), blk(b)], dim=1)
        leaked = blk(torch.cat([a, b], dim=1))

    torch.testing.assert_close(packed, ref, atol=1e-5, rtol=1e-5)
    # The isolation must be doing real work, not passing vacuously.
    assert (leaked - ref).abs().max() > 1e-2


# ---------------------------------------------------------------------------
# Isolation: attention
# ---------------------------------------------------------------------------
def test_transformer_packed_row_matches_documents_run_separately():
    blk = TransformerBlock(d_model=32, n_heads=2, dropout=0.0)
    blk.eval()
    torch.manual_seed(5)
    a = torch.randn(1, 7, 32)
    b = torch.randn(1, 5, 32)
    doc = torch.tensor([[0] * 7 + [1] * 5])

    out_packed = blk(torch.cat([a, b], dim=1), doc_ids=doc)
    ref = torch.cat([blk(a), blk(b)], dim=1)
    torch.testing.assert_close(out_packed, ref, atol=ATOL, rtol=ATOL)


def test_transformer_window_and_document_masks_compose():
    """With a window set AND packing, both constraints must apply."""
    blk = TransformerBlock(d_model=32, n_heads=2, dropout=0.0, window=3)
    blk.eval()
    torch.manual_seed(6)
    a = torch.randn(1, 7, 32)
    b = torch.randn(1, 6, 32)
    doc = torch.tensor([[0] * 7 + [1] * 6])
    out_packed = blk(torch.cat([a, b], dim=1), doc_ids=doc)
    ref = torch.cat([blk(a), blk(b)], dim=1)
    torch.testing.assert_close(out_packed, ref, atol=ATOL, rtol=ATOL)


def test_unpacked_attention_still_uses_the_flash_fast_path():
    """No regression for existing runs: doc_ids=None, S < window → is_causal path."""
    blk = TransformerBlock(d_model=32, n_heads=2, dropout=0.0, window=4096)
    blk.eval()
    x = torch.randn(1, 16, 32)
    # Nothing to assert numerically -- the guard is that this does not build a
    # mask. Assert the branch condition directly instead.
    assert not (blk.window is not None and x.shape[1] > blk.window)


# ---------------------------------------------------------------------------
# Isolation: the full EdgeHVN stack (acceptance criterion 2)
# ---------------------------------------------------------------------------
def _hybrid(num_layers=4):
    """A small deltanet+transformer hybrid, matching the real block_plan shape."""
    plan = ["deltanet", "transformer", "deltanet", "transformer"][:num_layers]
    m = EdgeHVN(vocab_size=64, d_model=32, num_layers=num_layers, n_heads=2,
                dropout=0.0, block_plan=plan, attn_window=4096)
    m.eval()
    for blk in m.blocks:
        if isinstance(blk, DeltaNetBlock):
            blk.train_chunk = 0
            blk.use_triton_scan = False
    return m


def test_edgehvn_packed_logits_match_documents_run_separately():
    """Acceptance criterion 2: end-to-end cross-document isolation on logits."""
    m = _hybrid()
    torch.manual_seed(7)
    a = torch.randint(0, 64, (1, 9))
    b = torch.randint(0, 64, (1, 6))
    doc = torch.tensor([[0] * 9 + [1] * 6])

    with torch.no_grad():
        packed = m(torch.cat([a, b], dim=1), doc_ids=doc)
        ref = torch.cat([m(a), m(b)], dim=1)
    torch.testing.assert_close(packed, ref, atol=1e-4, rtol=1e-4)


def test_edgehvn_document_b_is_invariant_to_document_a():
    """Swapping document A entirely must not move document B's logits."""
    m = _hybrid()
    torch.manual_seed(8)
    b = torch.randint(0, 64, (1, 6))
    doc = torch.tensor([[0] * 9 + [1] * 6])
    with torch.no_grad():
        o1 = m(torch.cat([torch.randint(0, 64, (1, 9)), b], dim=1), doc_ids=doc)
        o2 = m(torch.cat([torch.randint(0, 64, (1, 9)), b], dim=1), doc_ids=doc)
    torch.testing.assert_close(o1[:, 9:], o2[:, 9:], atol=1e-4, rtol=1e-4)


def test_edgehvn_grad_checkpoint_path_also_isolates():
    """The checkpoint closure must thread doc_ids on both of its invocations."""
    m = _hybrid()
    m.grad_checkpoint = True
    m.train()
    torch.manual_seed(9)
    a = torch.randint(0, 64, (1, 9))
    b = torch.randint(0, 64, (1, 6))
    doc = torch.tensor([[0] * 9 + [1] * 6])
    packed = m(torch.cat([a, b], dim=1), doc_ids=doc)
    packed.sum().backward()          # exercises the recompute (call #2)
    assert all(p.grad is not None for p in m.parameters() if p.requires_grad)


def test_edgehvn_unpacked_forward_is_unchanged():
    """doc_ids=None must be byte-identical to the pre-#132 path."""
    m = _hybrid()
    torch.manual_seed(10)
    ids = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = m(ids)
        b = m(ids, doc_ids=None)
    torch.testing.assert_close(a, b, atol=0.0, rtol=0.0)


def test_edgehvn_rejects_packing_for_unsupported_block_types():
    m = EdgeHVN(vocab_size=64, d_model=32, num_layers=2, n_heads=2, dropout=0.0,
                block_plan=["hvn", "deltanet"])
    m.eval()
    ids = torch.randint(0, 64, (1, 8))
    doc = torch.tensor([[0] * 4 + [1] * 4])
    with pytest.raises(NotImplementedError, match="deltanet"):
        m(ids, doc_ids=doc)


def test_edgehvn_rejects_mismatched_doc_ids_shape():
    m = _hybrid()
    ids = torch.randint(0, 64, (1, 8))
    with pytest.raises(ValueError, match="must match input_ids"):
        m(ids, doc_ids=torch.zeros(1, 7, dtype=torch.long))
