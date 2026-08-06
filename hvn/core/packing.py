"""Padding-free sequence packing for the fine-tune path (Issue #132).

The fine-tune datasets pad every example to ``seq_len + 1`` with EOS tokens whose
labels are ``-100``.  Those tokens contribute nothing to the gradient, so the
compute spent on them is pure waste.  Measured on the real corpora with the real
GPT-2 tokenizer at ``seq_len=512``:

===========================================  =====  ============  ==============
corpus                                           n  median tokens  padding waste
===========================================  =====  ============  ==============
``data/grounded_qa_expanded.jsonl``          5,240            86          83.0%
``data/tool_use_train.jsonl``                  600           180          63.6%
===========================================  =====  ============  ==============

Packing concatenates several short examples into one ``seq_len`` row so a step
carries ~6 real examples instead of one.  **This does not change what the model
learns per real token** -- it stops paying for the dead ones.  Same epochs, far
less wall-clock.

Document isolation
------------------
A packed row holds several unrelated documents, so every mechanism that carries
information along the sequence axis must be cut at the boundaries.  There are
exactly three in this architecture, and each is handled where it lives:

1. **The DeltaNet recurrent state** -- reset by forcing the per-channel decay
   gate ``alpha_t = 0`` at each document's first token, which makes
   ``S_t = S_{t-1}*diag(0)*(...) + beta_t v_t k_t^T`` depend on nothing before
   ``t``.  See :func:`doc_start_mask` and ``DeltaNetBlock.forward``.

2. **The DeltaNet causal depthwise conv** -- its ``K-1`` taps reach back across a
   boundary.  Handled in ``DeltaNetBlock._causal_conv``.

3. **Attention** in the hybrid's ``TransformerBlock`` layers -- handled by
   AND-ing a block-diagonal document mask into the attention mask.

RoPE needs nothing: it is relative, and cross-document keys are masked out
anyway, so a document starting at offset 300 still sees correct relative
distances within itself.

Why the reset is exact, not an epsilon
--------------------------------------
``hvn/core/deltanet_triton.py``'s scan is a *sequential-over-S* recurrence with a
direct ``state = state * a_t`` multiply ("fp32 throughout -- no
reparameterization, so stable at any alpha"), so ``alpha = 0`` is a
mathematically exact reset there.  The eager :meth:`_scan_chunked` /
:meth:`_scan_chunked_stable` paths fold ``alpha`` into a ``cumprod``/log-space
reparameterization that cannot represent a hard zero (they clamp at ``1e-6``),
so they would leak a ~1e-6 fraction of the previous document's state.  Rather
than ship a silently-approximate path, ``DeltaNetBlock.forward`` raises when it
is asked to pack under the eager chunked scan.  The FT scripts this targets
already default to the Triton scan.
"""

from __future__ import annotations

import random

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = [
    "doc_start_mask",
    "segment_attn_mask",
    "pack_examples",
    "PackedDataset",
    "packing_stats",
]


# ---------------------------------------------------------------------------
# Mask helpers
# ---------------------------------------------------------------------------
def doc_start_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """``[B, S]`` bool, True at each position that begins a new document.

    Position 0 always starts a document (the recurrent state enters the row
    zeroed anyway, so resetting there is a no-op, but keeping it True makes the
    mask self-describing and costs nothing).
    """
    if doc_ids.dim() != 2:
        raise ValueError(f"doc_ids must be [B, S], got shape {tuple(doc_ids.shape)}")
    starts = torch.ones_like(doc_ids, dtype=torch.bool)
    starts[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    return starts


def segment_attn_mask(doc_ids: torch.Tensor) -> torch.Tensor:
    """``[B, 1, S, S]`` bool, True where query i may attend to key j.

    True iff the two positions belong to the same document.  The caller ANDs
    this with causality (and the sliding window, when set).
    """
    same = doc_ids[:, :, None] == doc_ids[:, None, :]   # [B, S, S]
    return same.unsqueeze(1)                            # [B, 1, S, S]


# ---------------------------------------------------------------------------
# Bin packing
# ---------------------------------------------------------------------------
def pack_examples(
    lengths: list[int],
    capacity: int,
    strategy: str = "shuffled_first_fit",
    seed: int = 0,
) -> list[list[int]]:
    """Group example indices into bins of at most ``capacity`` tokens.

    Parameters
    ----------
    lengths : list[int]
        Token length of each example (including its own EOS).
    capacity : int
        Row capacity in tokens, i.e. ``seq_len + 1`` (one extra for the causal
        shift, matching the padded datasets' convention).
    strategy : {"shuffled_first_fit", "first_fit_decreasing"}
        ``shuffled_first_fit`` (default) shuffles before packing, so example
        length is uncorrelated with position in the row and with which examples
        share a step.  ``first_fit_decreasing`` wastes fewer tokens but sorts
        long examples to the front, which correlates length with batch
        composition -- a real training-dynamics risk, so it is not the default.
    seed : int
        Shuffle seed (``shuffled_first_fit`` only), for reproducible packing.

    Returns
    -------
    list[list[int]]
        One list of example indices per packed row.

    Notes
    -----
    Examples longer than ``capacity`` get a bin to themselves; the caller is
    responsible for truncating them (the datasets already do).
    """
    if strategy not in ("shuffled_first_fit", "first_fit_decreasing"):
        raise ValueError(f"unknown packing strategy {strategy!r}")

    order = list(range(len(lengths)))
    if strategy == "shuffled_first_fit":
        random.Random(seed).shuffle(order)
    else:
        order.sort(key=lambda i: lengths[i], reverse=True)

    bins: list[list[int]] = []
    remaining: list[int] = []
    for i in order:
        L = min(lengths[i], capacity)
        placed = False
        for b in range(len(bins)):
            if remaining[b] >= L:
                bins[b].append(i)
                remaining[b] -= L
                placed = True
                break
        if not placed:
            bins.append([i])
            remaining.append(capacity - L)
    return bins


def packing_stats(lengths: list[int], bins: list[list[int]], capacity: int) -> dict:
    """Real-token / padding accounting for a packing, for logging."""
    real = sum(min(L, capacity) for L in lengths)
    budget = len(bins) * capacity
    padded_budget = len(lengths) * capacity
    return {
        "n_examples": len(lengths),
        "n_rows": len(bins),
        "examples_per_row": len(lengths) / max(len(bins), 1),
        "real_tokens": real,
        "packed_budget": budget,
        "padded_budget": padded_budget,
        "packed_waste_frac": 1.0 - real / max(budget, 1),
        "padded_waste_frac": 1.0 - real / max(padded_budget, 1),
        "row_reduction": padded_budget / max(budget, 1),
    }


# ---------------------------------------------------------------------------
# Dataset wrapper
# ---------------------------------------------------------------------------
class PackedDataset(Dataset):
    """Packs a list of variable-length ``(ids, labels)`` examples into rows.

    Each item is ``(x, y, doc_ids)`` with ``x``/``y``/``doc_ids`` all length
    ``seq_len``.  ``x``/``y`` carry the same causal shift the padded datasets
    use (``x = row[:-1]``, ``y = labels[1:]``), and ``doc_ids`` is aligned to
    ``x`` so a block can cut its sequence-axis carries at the boundaries.

    Any leftover tail in a row is filled with ``pad_id`` and label ``-100`` --
    that residue is what packing could not remove (a few percent), versus the
    60-85% the padded datasets spend.

    Parameters
    ----------
    examples : list[tuple[np.ndarray, np.ndarray]]
        Variable-length ``(ids, labels)`` pairs, already truncated to at most
        ``seq_len + 1`` and already loss-masked with ``-100``.
    seq_len : int
        Training sequence length; rows hold ``seq_len + 1`` tokens.
    pad_id : int
        Token id for the unavoidable tail residue.
    """

    def __init__(
        self,
        examples: list[tuple[np.ndarray, np.ndarray]],
        seq_len: int = 512,
        pad_id: int = 50256,
        strategy: str = "shuffled_first_fit",
        seed: int = 0,
        verbose: bool = True,
    ):
        self.seq_len = seq_len
        capacity = seq_len + 1

        lengths = [len(ids) for ids, _ in examples]
        bins = pack_examples(lengths, capacity, strategy=strategy, seed=seed)
        self.stats = packing_stats(lengths, bins, capacity)

        self.rows: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for b in bins:
            row_ids = np.full(capacity, pad_id, dtype=np.int64)
            row_lab = np.full(capacity, -100, dtype=np.int64)
            # Document id is *row-local* and starts at 0. The tail residue gets
            # its own id (n_docs) so it is isolated from the last real document
            # rather than silently extending it.
            row_doc = np.zeros(capacity, dtype=np.int64)
            off = 0
            for doc_i, ex_i in enumerate(b):
                ids, lab = examples[ex_i]
                L = min(len(ids), capacity - off)
                row_ids[off:off + L] = ids[:L]
                row_lab[off:off + L] = lab[:L]
                row_doc[off:off + L] = doc_i
                off += L
            if off < capacity:
                row_doc[off:] = len(b)      # residue = its own document

            # Cut the causal shift at every boundary. Without this, the last
            # token of document A is trained to predict the FIRST token of
            # document B -- a cross-document prediction target that packing
            # would silently introduce. (The current corpora all mask their
            # first token anyway, since the prompt is -100, so this is a
            # no-op on them -- but it is what makes the class correct for a
            # corpus that does not.)
            starts = np.ones(capacity, dtype=bool)
            starts[1:] = row_doc[1:] != row_doc[:-1]
            row_lab[starts] = -100

            self.rows.append((row_ids, row_lab, row_doc))

        if verbose:
            s = self.stats
            print(f"  Packed {s['n_examples']:,} examples into {s['n_rows']:,} rows "
                  f"({s['examples_per_row']:.2f} ex/row, "
                  f"{s['row_reduction']:.2f}x fewer rows); "
                  f"padding waste {s['padded_waste_frac']:.1%} -> "
                  f"{s['packed_waste_frac']:.1%}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        ids, lab, doc = self.rows[idx]
        # Causal shift, matching the padded datasets: predict token t+1 from <=t.
        x = torch.from_numpy(ids[:-1])
        y = torch.from_numpy(lab[1:])
        # doc_ids aligns with x (the positions actually fed to the blocks).
        d = torch.from_numpy(doc[:-1])
        return x, y, d
