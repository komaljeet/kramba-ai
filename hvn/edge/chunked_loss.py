"""Chunked, recomputed cross-entropy: ~1 GB back on a 6 GB card (#158).

Author: Rambarun Komaljeet

Measured on a live 48M run, 3.27 GB in use:

    computing the loss (4096 positions x 32003 vocab)   ~1500 MB
    CUDA context / kernels / workspaces                  ~400 MB
    weights                                               183 MB
    optimizer state                                       183 MB
    layer activations                                    ~ 50 MB

**The loss costs more than the model** -- about three times the weights. Layer
activations are already tiny because gradient checkpointing recomputes them; this
term is what remains, and nothing had been done to it.

Why the obvious fix is not enough
---------------------------------
Chunking only the cross-entropy saves nothing, because ``model(x)`` has already
materialised the full ``[B, S, V]`` logits tensor before the loss is ever called.
The projection itself has to be chunked too. So this takes the model's HIDDEN
states -- ``[B, S, D]``, 4 MB at 4096x512 against 500 MB of logits -- and for
each slice of positions projects, computes the loss, and lets the slice go.

Backward would still need every slice's logits at once, which would undo the
saving, so each slice is wrapped in ``torch.utils.checkpoint``: forward keeps only
the hidden states, and backward recomputes one slice's logits at a time. That is
exactly what gradient checkpointing already does for every other layer of this
model -- it simply was never applied to the loss.

Peak logit memory drops from ``S x V`` to ``chunk x V``: at chunk=512 that is an
8x reduction on the dominant term, for one extra projection per chunk in backward.

Exactness, stated precisely
---------------------------
Every position is scored exactly once against the full vocabulary. In fp32 the
returned per-token losses are BIT-IDENTICAL to the unchunked path -- ``tests/test_chunked_loss.py`` asserts bit-level equality of the loss and
of every gradient, including into the tied embedding table, which takes gradient
from both the lookup and the output head and is where a chunked implementation is
most likely to drop a term.

Under **bf16 autocast** they are not bit-identical, and that is expected rather
than a defect: each slice reduces separately, so the accumulation order differs.
Measured at the production shape (48M, seq 4096, accum 8) the gap is **3.8e-03 on
a loss of 10.5 -- 3.6e-04 relative**, and two different chunk sizes differ from
each other by 3.8e-04. Do not describe this path as bit-identical without saying
"in fp32".

Measured cost and benefit at that shape (`results/chunked_loss/bench.json`):

    unchunked   2275 MB alloc / 2584 MB reserved / 3.21 s per step
    chunk 512   1003 MB alloc / 1148 MB reserved / 3.43 s per step
    chunk 256   1006 MB alloc / 1056 MB reserved / 3.53 s per step

**-1436 MB reserved for 1.07x step time.** The extra time is the one additional
projection per chunk in backward.

The failure this must not have is silent: an off-by-one at a chunk boundary drops
or double-counts one position in 4096, moving the loss by ~0.02% -- invisible in
any training curve. The test suite therefore includes a deliberately-broken
variant and requires the gates to reject it.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

#: Positions scored per slice. 512 gives an 8x cut at seq_len 4096 while keeping
#: each projection large enough to saturate the GPU; smaller chunks save little
#: more and add launch overhead.
DEFAULT_CHUNK = 512


#: The fast loss head pads the vocabulary dimension up to a multiple of this.
#: 50257 (GPT-2) is odd, which forces cuBLAS onto an `align1` GEMM kernel; at
#: 50304 = 64 * 786 it picks an aligned one. Measured at 2048 x 50257, seq 4096:
#: 243.7 -> 148.8 ms per micro-batch for the loss head (pre-cast + pad).
VOCAB_PAD_MULTIPLE = 64


def _slice_loss(hidden_slice: torch.Tensor, target_slice: torch.Tensor,
                decoder: nn.Module) -> torch.Tensor:
    """Project one slice to vocabulary space and score it. Recomputed in backward."""
    logits = decoder(hidden_slice)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_slice.reshape(-1),
        reduction="none",
    )


def _slice_loss_padded(hidden_slice: torch.Tensor, target_slice: torch.Tensor,
                       weight_pad: torch.Tensor, bias: torch.Tensor | None,
                       vocab: int) -> torch.Tensor:
    """`_slice_loss` against a pre-cast, vocab-padded weight (see `_fast_head`).

    The bias is cast and padded HERE, per slice, not once outside: it may be
    trainable (it lives in the BCD tail block), and casting it once would sum
    its per-slice gradients in bf16 instead of fp32. It is 50K elements, so
    this costs microseconds. Padding columns get -inf, so their softmax weight
    is exactly 0: they add exact zeros to the normaliser, are never a target,
    and receive zero gradient.
    """
    pad = weight_pad.shape[0] - vocab
    if bias is None:
        b = weight_pad.new_zeros(vocab)
    else:
        b = bias.to(weight_pad.dtype)
    b = torch.cat([b, b.new_full((pad,), float("-inf"))])
    logits = F.linear(hidden_slice, weight_pad, b)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)),
        target_slice.reshape(-1),
        reduction="none",
    )


def _fast_head(hidden: torch.Tensor, decoder: nn.Module):
    """Return ``(weight_pad, bias, vocab)`` for the pre-cast padded loss head, or
    None when the plain per-slice path must be used.

    Two costs this removes, both measured at 2048 x 50257 (2026-10-12):

    1. The decoder weight was cast fp32 -> bf16 by autocast on EVERY slice
       call, forward and recompute: 16 casts of a 206 MB tensor per
       micro-batch, ~3.6 ms each. Autocast's own weight-cast cache does not
       help, by design: it only caches leaf tensors with ``requires_grad``
       (``cached_cast`` in ATen's autocast_mode), and under BCD the tied table
       is frozen on every step except the embedding block's. Here it is cast
       once per call and reused by every slice.
    2. The odd vocab forces an ``align1`` GEMM. The cast copy is written
       straight into a buffer padded to a multiple of 64 rows (zeros), so the
       pad costs no extra copy and no extra memory over the cast itself.

    Used only when ALL of these hold, otherwise the old path runs unchanged:
    - CUDA autocast to a 16-bit dtype is on. In fp32 the chunked path is
      bit-identical to the unchunked one and the tests assert that; padding
      changes the GEMM shape, so it stays off there.
    - The decoder is a plain ``nn.Linear`` (directly or as ``HVNDecoder``'s
      ``.decoder``). A TernaryLinear or an MLP head has its own forward.
    - The weight needs no gradient. When it is trainable (the tied table, on
      the embedding block's turn), a single shared cast would sum the slices'
      weight gradients in bf16; the per-slice autocast cast sums them in fp32.
      That step keeps the old path.

    The padding is local to this call: the model, its parameters, its
    state_dict and its checkpoints keep the true vocab size.
    """
    if not hidden.is_cuda or not torch.is_autocast_enabled("cuda"):
        return None
    dtype = torch.get_autocast_dtype("cuda")
    if dtype not in (torch.bfloat16, torch.float16):
        return None
    lin = getattr(decoder, "decoder", decoder) if not isinstance(decoder, nn.Linear) else decoder
    if type(lin) is not nn.Linear:
        return None
    w = lin.weight
    if w.requires_grad and torch.is_grad_enabled():
        return None
    if not w.is_cuda:
        return None
    vocab, d = w.shape
    vp = -(-vocab // VOCAB_PAD_MULTIPLE) * VOCAB_PAD_MULTIPLE
    with torch.no_grad():
        weight_pad = torch.empty(vp, d, dtype=dtype, device=w.device)
        weight_pad[:vocab].copy_(w)
        weight_pad[vocab:].zero_()
    return weight_pad, lin.bias, vocab


def chunked_cross_entropy(hidden: torch.Tensor, targets: torch.Tensor,
                          decoder: nn.Module, chunk: int = DEFAULT_CHUNK,
                          use_checkpoint: bool = True,
                          fast_head: bool = True) -> torch.Tensor:
    """Per-token cross-entropy over ``hidden``, computed ``chunk`` positions at a time.

    Parameters
    ----------
    hidden : ``[B, S, D]`` final hidden states -- the model's output BEFORE the
        vocabulary projection.
    targets : ``[B, S]`` target ids.
    decoder : the module projecting ``D -> vocab``.
    chunk : positions per slice.
    use_checkpoint : recompute each slice's logits in backward instead of keeping
        them. False keeps the chunking but not the saving -- used by the tests to
        separate "chunking is exact" from "recompute is exact".
    fast_head : under 16-bit CUDA autocast with a frozen plain-Linear decoder,
        cast the weight once and pad the vocab (see `_fast_head`). False forces
        the per-slice path -- used by the tests and for A/B measurement.

    Returns
    -------
    ``[B*S]`` per-token losses, in the same order and with the same values as
    ``F.cross_entropy(..., reduction="none")`` on the unchunked logits.
    """
    if hidden.dim() != 3:
        raise ValueError(f"expected [B, S, D] hidden states, got {tuple(hidden.shape)}")
    if targets.shape[:2] != hidden.shape[:2]:
        raise ValueError(
            f"targets {tuple(targets.shape)} do not match hidden "
            f"{tuple(hidden.shape[:2])} -- a mismatch here would score the wrong "
            f"positions and still return a plausible number")
    if chunk < 1:
        raise ValueError("chunk must be >= 1")

    B, S, _ = hidden.shape
    fast = _fast_head(hidden, decoder) if fast_head else None
    if fast is not None:
        fn, extra = _slice_loss_padded, fast
    else:
        fn, extra = _slice_loss, (decoder,)
    out = []
    for start in range(0, S, chunk):
        end = min(start + chunk, S)          # the last slice is short, not skipped
        h = hidden[:, start:end]
        t = targets[:, start:end]
        if use_checkpoint and torch.is_grad_enabled() and hidden.requires_grad:
            piece = checkpoint(fn, h, t, *extra, use_reentrant=False)
        else:
            piece = fn(h, t, *extra)
        out.append(piece)

    # Slices are [B*chunk] each, laid out batch-major within the slice, so a plain
    # concatenation would interleave batches wrongly for B > 1. Reshape back to
    # [B, slice] first, join along the sequence axis, then flatten -- giving the
    # same [B*S] ordering the unchunked path produces.
    pieces = [p.view(B, -1) for p in out]
    return torch.cat(pieces, dim=1).reshape(-1)


def peak_logit_bytes(seq_len: int, vocab: int, chunk: int, dtype_bytes: int = 4) -> int:
    """Peak bytes held by the logits, chunked vs not -- used by the VRAM gate."""
    return min(chunk, seq_len) * vocab * dtype_bytes
