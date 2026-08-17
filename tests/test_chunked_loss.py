"""Chunked cross-entropy must be EXACT, not merely close (#158).

Author: Rambarun Komaljeet

This replaces the loss computation of every training run. The failure that
matters is silent: an off-by-one at a chunk boundary drops or double-counts one
position in 4096, which moves the loss by ~0.02% -- far below anything visible in
a training curve, and indistinguishable from noise in a comparison. The same
shape as #164, where the one skill being measured was 0.02% of the signal and the
curve could not see it.

So the gates are equality, not tolerance, and the suite includes deliberately
broken variants that it must reject. A test that only checked "the loss is about
the same" would pass on every bug this file exists to catch.
"""

from __future__ import annotations

import os
import sys

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN           # noqa: E402
from hvn.edge.chunked_loss import (          # noqa: E402
    chunked_cross_entropy, peak_logit_bytes,
)

V, D, S, B = 512, 64, 48, 2


def _model():
    torch.manual_seed(0)
    plan = ["transformer" if i in (1, 3) else "deltanet" for i in range(4)]
    m = EdgeHVN(vocab_size=V, d_model=D, num_layers=4, n_heads=4,
                block_plan=plan, attn_window=64)
    # eval(): dropout would make two forward passes differ for reasons that have
    # nothing to do with chunking, and the earlier version of this test was
    # fooled by exactly that (0.319 max "error" that was entirely dropout).
    m.eval()
    return m


def _batch():
    g = torch.Generator().manual_seed(1)
    x = torch.randint(0, V, (B, S), generator=g)
    return x, torch.roll(x, -1, 1)


def _reference(m, hidden, y):
    return F.cross_entropy(m.decoder(hidden).reshape(-1, V), y.reshape(-1),
                           reduction="none")


@pytest.mark.parametrize("chunk", [7, 16, 24, 48, 64])
def test_bit_identical_to_the_unchunked_loss(chunk):
    """Equality, not allclose. Chunk sizes include one that does not divide S
    (7 and 24 leave a short final slice) and one larger than S."""
    m = _model()
    x, y = _batch()
    with torch.no_grad():
        h = m(x, return_hidden=True)
        ref = _reference(m, h, y)
        got = chunked_cross_entropy(h, y, m.decoder, chunk=chunk)
    assert torch.equal(ref, got), (
        f"chunk={chunk}: max |diff| {(ref - got).abs().max().item():.3e}")


def test_every_position_is_scored_exactly_once():
    """The off-by-one gate, asserted on shape and ordering rather than on a
    scalar. A dropped or duplicated position can leave the MEAN almost unchanged
    while the per-token vector is wrong."""
    m = _model()
    x, y = _batch()
    with torch.no_grad():
        h = m(x, return_hidden=True)
        ref = _reference(m, h, y)
        got = chunked_cross_entropy(h, y, m.decoder, chunk=16)
    assert got.shape == ref.shape == (B * S,)
    assert torch.equal(got, ref), "per-token ordering differs"


def test_gradients_match_including_the_tied_table():
    """The tied embedding table takes gradient from BOTH the lookup and the
    output head, so it is where a chunked implementation is most likely to drop a
    term -- and a dropped term there would still train, just worse."""
    torch.manual_seed(0)
    plan = ["transformer" if i in (1, 3) else "deltanet" for i in range(4)]
    a = EdgeHVN(vocab_size=V, d_model=D, num_layers=4, n_heads=4,
                block_plan=plan, attn_window=64, tie_embeddings=True).eval()
    b = EdgeHVN(vocab_size=V, d_model=D, num_layers=4, n_heads=4,
                block_plan=plan, attn_window=64, tie_embeddings=True).eval()
    b.load_state_dict(a.state_dict())
    assert a.token_embedding.weight is a.decoder.decoder.weight, "not tied"

    x, y = _batch()
    _reference(a, a(x, return_hidden=True), y).mean().backward()
    chunked_cross_entropy(b(x, return_hidden=True), y, b.decoder,
                          chunk=16).mean().backward()

    named_b = dict(b.named_parameters())
    worst, where = 0.0, ""
    for n, pa in a.named_parameters():
        pb = named_b[n]
        assert (pa.grad is None) == (pb.grad is None), f"{n}: gradient presence differs"
        if pa.grad is not None:
            d = (pa.grad - pb.grad).abs().max().item()
            if d > worst:
                worst, where = d, n
    assert worst < 1e-5, f"gradients diverge at {where}: {worst:.3e}"

    tbl = "token_embedding.weight"
    assert dict(a.named_parameters())[tbl].grad.abs().sum() > 0, \
        "the tied table received no gradient in the reference -- test is vacuous"


def test_recompute_and_chunking_are_separable():
    """use_checkpoint=False keeps the chunking but drops the recompute. Both
    paths must be exact, so a future failure points at which half broke."""
    m = _model()
    x, y = _batch()
    with torch.no_grad():
        h = m(x, return_hidden=True)
        ref = _reference(m, h, y)
        assert torch.equal(ref, chunked_cross_entropy(h, y, m.decoder, 16,
                                                      use_checkpoint=False))


def test_mismatched_targets_are_refused_not_silently_scored():
    """Passing targets of the wrong length would score the wrong positions and
    return a perfectly plausible number."""
    m = _model()
    x, y = _batch()
    with torch.no_grad():
        h = m(x, return_hidden=True)
    with pytest.raises(ValueError, match="do not match"):
        chunked_cross_entropy(h, y[:, :-1], m.decoder, chunk=16)


def test_peak_logit_memory_actually_drops():
    """The point of the exercise. At 4096 x 32003 in fp32 the logits are ~500 MB;
    chunk 512 must cut that 8x."""
    full = peak_logit_bytes(4096, 32003, chunk=4096)
    chunked = peak_logit_bytes(4096, 32003, chunk=512)
    assert full / chunked == pytest.approx(8.0)
    assert full / 2 ** 20 > 400, "the premise (logits are large) no longer holds"


# ---------------------------------------------------------------------------
# Anti-vacuity: the gates above must REJECT a broken implementation.
# Without these, they could pass for reasons unrelated to correctness.
# ---------------------------------------------------------------------------

def _broken_last_slice_dropped(hidden, targets, decoder, chunk):
    """Skips a short final slice -- the classic off-by-one."""
    Bn, Sn, _ = hidden.shape
    out = []
    for start in range(0, Sn, chunk):
        end = start + chunk
        if end > Sn:            # the bug: drop the remainder instead of clamping
            break
        lg = decoder(hidden[:, start:end])
        out.append(F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                   targets[:, start:end].reshape(-1),
                                   reduction="none").view(Bn, -1))
    return torch.cat(out, dim=1).reshape(-1)


def _broken_batch_interleaved(hidden, targets, decoder, chunk):
    """Concatenates slices without reshaping to [B, slice] first -- correct for
    B=1, silently wrong for B>1, which is how it would reach production."""
    Bn, Sn, _ = hidden.shape
    out = []
    for start in range(0, Sn, chunk):
        end = min(start + chunk, Sn)
        lg = decoder(hidden[:, start:end])
        out.append(F.cross_entropy(lg.reshape(-1, lg.size(-1)),
                                   targets[:, start:end].reshape(-1),
                                   reduction="none"))
    return torch.cat(out)


def test_the_gates_reject_a_dropped_final_slice():
    m = _model()
    x, y = _batch()
    with torch.no_grad():
        h = m(x, return_hidden=True)
        ref = _reference(m, h, y)
        bad = _broken_last_slice_dropped(h, y, m.decoder, chunk=7)
    assert not torch.equal(ref[:bad.numel()], bad) or bad.numel() != ref.numel(), \
        "a dropped final slice is not detected -- the equality gate is vacuous"
    assert bad.numel() < ref.numel()


def test_the_gates_reject_batch_interleaving():
    """The subtle one: same length, same values, wrong ORDER. A mean-based check
    would pass this happily -- the per-token equality gate is what catches it."""
    m = _model()
    x, y = _batch()
    with torch.no_grad():
        h = m(x, return_hidden=True)
        ref = _reference(m, h, y)
        bad = _broken_batch_interleaved(h, y, m.decoder, chunk=16)
    assert bad.numel() == ref.numel(), "test needs same-length outputs to be meaningful"
    assert bad.mean().item() == pytest.approx(ref.mean().item(), abs=1e-5), (
        "the means already differ, so this variant would be caught by a weaker "
        "check and proves nothing about the per-token gate")
    assert not torch.equal(ref, bad), \
        "batch interleaving is not detected -- the per-token gate is vacuous"
