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


# ---------------------------------------------------------------------------
# Fast loss head (2026-10-12): weight cast once per call + vocab padded to a
# multiple of 64 with -inf bias columns. bf16 CUDA autocast only.
# ---------------------------------------------------------------------------

_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA autocast")


def _head(V_, D_, frozen=True):
    from hvn.core.decoder import HVNDecoder
    torch.manual_seed(0)
    dec = HVNDecoder(D_, V_, dropout=0.0).cuda()
    dec.decoder.weight.requires_grad_(not frozen)
    return dec


def _run_head(dec, hid, y, fast, amp=True, chunk=128):
    import hvn.edge.chunked_loss as cl
    h = hid.clone().requires_grad_(True)
    dec.zero_grad(set_to_none=True)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        tok = cl.chunked_cross_entropy(h, y, dec, chunk=chunk, fast_head=fast)
    tok.mean().backward()
    return tok.detach().float(), h.grad.float(), dec.decoder.bias.grad.float()


def _head_inputs(V_, D_, S_=600, B_=2):
    g = torch.Generator(device="cuda").manual_seed(3)
    hid = torch.randn(B_, S_, D_, device="cuda", generator=g)
    y = torch.randint(0, V_, (B_, S_), device="cuda", generator=g)
    y[0, :7] = V_ - 1       # the last real column, right next to the padding
    return hid, y


@_cuda
def test_fast_head_is_taken_only_where_it_is_safe():
    """The speedup only exists if the fast path actually runs in training, and
    it must NOT run where it would change numerics (fp32; trainable weight)."""
    import hvn.edge.chunked_loss as cl
    dec = _head(50257, 64)
    h = torch.randn(1, 8, 64, device="cuda")
    assert cl._fast_head(h, dec) is None, "fast head ran outside autocast (fp32 must stay exact)"
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        fast = cl._fast_head(h, dec)
        assert fast is not None, "fast head not taken for a frozen bf16 head -- lever is dead"
        w_pad, _, vocab = fast
        assert vocab == 50257 and w_pad.shape == (50304, 64) and w_pad.dtype == torch.bfloat16
        assert torch.equal(w_pad[:vocab], dec.decoder.weight.to(torch.bfloat16))
        assert not w_pad[vocab:].any(), "padding rows are not zero"
        dec.decoder.weight.requires_grad_(True)
        assert cl._fast_head(h, dec) is None, \
            "fast head ran with a trainable weight: its slice grads would sum in bf16"
        with torch.no_grad():
            assert cl._fast_head(h, dec) is not None, "no-grad eval should use the fast head"


@_cuda
@pytest.mark.parametrize("V_", [50257, 32003, 4096])
def test_fast_head_per_token_loss_bit_identical(V_):
    """Per-token losses, not the mean -- a padding column leaking into the
    softmax shifts every token's loss. 4096 is already a multiple of 64 (no
    padding; pre-cast only)."""
    dec = _head(V_, 128)
    hid, y = _head_inputs(V_, 128)
    t0, _, _ = _run_head(dec, hid, y, fast=False)
    t1, _, _ = _run_head(dec, hid, y, fast=True)
    assert torch.isfinite(t1).all()
    assert torch.equal(t0, t1), f"max |dloss| {(t0 - t1).abs().max().item():.3e}"


def _rel_errs(D_, V_, S_, B_, chunk):
    dec = _head(V_, D_)
    hid, y = _head_inputs(V_, D_, S_, B_)
    _, g0, b0 = _run_head(dec, hid, y, fast=False, chunk=chunk)
    _, g1, b1 = _run_head(dec, hid, y, fast=True, chunk=chunk)
    _, gr, br = _run_head(dec, hid, y, fast=False, amp=False, chunk=chunk)
    assert torch.isfinite(g1).all() and torch.isfinite(b1).all()
    out = {}
    for name, base, fast, ref in (("hidden", g0, g1, gr), ("bias", b0, b1, br)):
        n = ref.norm().item()
        out[name] = ((base - ref).norm().item() / n, (fast - ref).norm().item() / n)
        assert out[name][0] > 0, f"{name}: bf16 path equals fp32 -- autocast not active, test vacuous"
    return out


# Production loss-head GEMM shapes: chunk 512, batch 1. (d_model, vocab)
_PROD_HEADS = [(512, 32003), (896, 32003), (2048, 50257)]


@_cuda
@pytest.mark.parametrize("D_,V_", _PROD_HEADS)
def test_fast_head_gradients_match_when_reductions_are_fp32(D_, V_):
    """The gradient math is the same: with cuBLAS's bf16 split-K reductions
    disabled, both paths are the same distance from fp32 to 1% (measured: equal
    to four digits at every shape tried). So any difference under the default
    setting is cuBLAS's kernel choice, not the padding or the pre-cast."""
    flag = torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    try:
        e = _rel_errs(D_, V_, 1024, 1, 512)
    finally:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = flag
    e_base, e_fast = e["hidden"]
    assert abs(e_fast - e_base) <= 0.01 * e_base, (
        f"d={D_}: fast {e_fast:.4e} vs old {e_base:.4e} from fp32 with fp32 "
        f"reductions -- the padded head computes something different")


@_cuda
@pytest.mark.parametrize("D_,V_", _PROD_HEADS)
def test_fast_head_gradient_change_is_within_bf16_noise(D_, V_):
    """Gradients are NOT bit-identical (max 8.9e-8 at 2048 x 50257). Under
    PyTorch's default `allow_bf16_reduced_precision_reduction=True`, cuBLAS may
    pick a split-K kernel that reduces in bf16, and which path gets one depends
    on the GEMM shape. Measured hidden-gradient distance from fp32 (relative
    norm, old -> fast), 2026-10-12:

        d=256  v=50257   8.0e-3 -> 5.6e-3   (old path worse)
        d=512  v=32003   7.0e-3 -> 6.1e-3
        d=896  v=32003   3.5e-3 -> 5.1e-3   (fast worse, x1.45)
        d=1024 v=50257   3.2e-3 -> 5.1e-3   (x1.60)
        d=2048 v=50257   3.5e-3 -> 3.5e-3   (the 1B config: identical)

    So the fast path can be up to ~1.6x further from fp32 at some widths, but it
    never leaves the band the OLD path already produces at other production
    widths (up to 8e-3), i.e. ~1-2 bf16 ulps (2^-8 = 3.9e-3) relative. The gate
    is that band: the fast path must stay within 2x of the old one AND under
    the old path's own worst production value. Max-abs is not used: it is one
    element and swings 5x with the kernel choice in either direction."""
    e = _rel_errs(D_, V_, 1024, 1, 512)
    for name, (e_base, e_fast) in e.items():
        assert e_fast <= 2.0 * e_base and e_fast <= 8.5e-3, (
            f"{name} d={D_}: fast head {e_fast:.3e} from fp32, old path {e_base:.3e} "
            f"-- outside the bf16 band the old path already has")


@_cuda
def test_fast_head_rejects_zero_bias_padding():
    """Anti-vacuity for the loss gate: padding columns with bias 0 instead of
    -inf must be caught, or the equality test above proves nothing about the
    padding."""
    import hvn.edge.chunked_loss as cl
    dec = _head(50257, 128)
    hid, y = _head_inputs(50257, 128)
    t0, _, _ = _run_head(dec, hid, y, fast=False)
    orig = cl._slice_loss_padded

    def _bad(h, t, w, b, vocab):
        logits = F.linear(h, w, torch.cat([b.to(w.dtype), b.new_zeros(w.shape[0] - vocab).to(w.dtype)]))
        return F.cross_entropy(logits.reshape(-1, logits.size(-1)), t.reshape(-1), reduction="none")
    cl._slice_loss_padded = _bad
    try:
        tb, _, _ = _run_head(dec, hid, y, fast=True)
    finally:
        cl._slice_loss_padded = orig
    assert not torch.equal(t0, tb), "zero-bias padding not detected -- the gate is vacuous"


@_cuda
def test_fast_head_leaves_model_vocab_and_checkpoint_untouched():
    """Padding is local to the loss call: the tied table, decoder and
    state_dict keep the real vocab size."""
    m = EdgeHVN(vocab_size=50257, d_model=64, num_layers=2, n_heads=4,
                block_plan=["deltanet", "deltanet"], tie_embeddings=True,
                dropout=0.0).cuda()
    before = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    m.token_embedding.weight.requires_grad_(False)
    x = torch.randint(0, 50257, (1, 64), device="cuda")
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        chunked_cross_entropy(m(x, return_hidden=True), x, m.decoder, chunk=16).mean().backward()
    after = {k: tuple(v.shape) for k, v in m.state_dict().items()}
    assert before == after
    assert m.decoder.decoder.weight.shape[0] == 50257
    assert m.decoder.decoder.weight is m.token_embedding.weight, "tie broken"
    assert m.decoder.decoder.weight.dtype == torch.float32
