"""Looped Early Exit for EdgeHVN (Issue #122) — decode-time-only integration.

Wraps an *already-trained* :class:`~hvn.edge.model.EdgeHVN` instance so its
existing block stack (DeltaNet + windowed attention) is applied up to
``max_loops`` times per token during incremental decode, exiting as soon as
the normalized entropy of the predicted-token distribution drops below
``entropy_threshold``. This is architectural integration + measurement only
— see ``scripts/sweep_looped_early_exit.py`` for the N/T hyperparameter
search on an existing checkpoint. No new weights are introduced, so nothing
here requires retraining to run (whether it helps without retraining is an
open empirical question the sweep answers, not an assumption baked in here).

Design (Issue #122 spec, resolved ambiguities noted):

* **Exit signal = normalized entropy of the softmax over the vocab**
  (``H(p) / log(vocab_size)``, so the threshold ``T`` lives in ``[0, 1]``
  independent of vocab size), NOT a learned classifier. A 2-layer MLP
  "confidence" head would need to be trained to mean anything; entropy is
  computable directly from logits that already exist, which is the only
  option consistent with "do not retrain."
* **Per-loop refinement**: loop *i+1* re-runs the same block stack on loop
  *i*'s OUTPUT hidden state (weight-tied depth, "looped transformer" style)
  — not on a fresh re-embedding of the same token. Re-embedding every loop
  would make every loop's output byte-identical (state is reset too), so
  there would be nothing to refine and entropy would never change loop to
  loop.
* **State reset per loop, for the current token position only**: every
  loop starts from a clone of the state that existed *before* this token
  (the DeltaNet recurrent matrix / attention KV cache as of the end of the
  previous committed token) — a rejected loop's candidate state is
  discarded, never blended in.
* **The state carried forward to the NEXT token is always LOOP 1's**,
  regardless of which loop the exit fired on. Only loop 1 sees the real
  token embedding; loops 2+ run the blocks on an already-transformed hidden
  state, so their k/v are not derived from this token, and writing them into
  the DeltaNet memory corrupts what the next token must read back. Measured
  on the frozen checkpoint: committing a later loop's state drops near-window
  recall 40% -> 0% *with the logits held identical*, while loop-4 logits over
  a clean loop-1 state hold at 35%
  (``scripts/test_state_commit_hypothesis.py``). Extra loops refine the
  *prediction*; they do not rewrite *memory*. An earlier version of this file
  committed the last loop's state instead, which silently invalidated every
  experiment run against it (see docs/state.md).
* ``forward_step`` / ``init_state`` / ``init_conversation_state`` /
  ``chat_step`` keep their exact existing signatures — this monkeypatches
  only ``forward_step`` (via ``types.MethodType``), so
  ``hvn/chat.py::ChatSession`` (which calls only those four methods) needs
  zero changes, per the Issue #122 spec.
"""

from __future__ import annotations

import math
import types

import torch


def _clone_state(state):
    """Deep-clone a per-block incremental-decode state (dict/list-of-tensor
    nesting used by DeltaNetBlock/TransformerBlock/HVNBlock — see their
    ``init_state``/``forward_step`` implementations), so a discarded loop's
    in-place-updated tensors never leak into the next loop's attempt."""
    if isinstance(state, dict):
        return {k: _clone_state(v) for k, v in state.items()}
    if isinstance(state, list):
        return [_clone_state(v) for v in state]
    if isinstance(state, torch.Tensor):
        return state.clone()
    return state


def normalized_entropy(logits: torch.Tensor, vocab_size: int) -> float:
    """Entropy of softmax(logits) over the last dim, divided by log(vocab_size)
    so the result is in [0, 1] regardless of vocab size. Takes the max over
    every other dim (batch/position) — the LEAST confident stream gates the
    exit, so a batch of concurrent streams doesn't exit early on one easy
    member while others are still uncertain.
    """
    probs = torch.softmax(logits.float(), dim=-1)
    logp = torch.log(probs.clamp_min(1e-12))
    ent = -(probs * logp).sum(dim=-1)
    max_ent = math.log(vocab_size)
    return (ent / max_ent).max().item()


def _looped_forward_step(self, input_ids, states, position_idx=None):
    """Replacement for ``EdgeHVN.forward_step`` — same signature and return
    shape (``logits [B,1,V], new_states``), internally looped. See module
    docstring for the loop/reset/exit semantics.
    """
    if self.blocks is None:
        raise NotImplementedError(
            "looped early exit is not implemented for shared_weights=True "
            "models -- no hybrid checkpoint uses SharedWeightHVN."
        )

    base_states = states
    x = self.token_embedding(input_ids)  # [B, 1, D] — only re-embedded on loop 1
    logits = None
    committed_states = None
    loops_used = 0

    for i in range(self._looped_max_loops):
        loop_states = _clone_state(base_states)
        h = x
        candidate_states = []
        for j, block in enumerate(self.blocks):
            h, st = block.forward_step(h, loop_states[j], position_idx=position_idx)
            candidate_states.append(st)
        logits = self.decoder(self.final_norm(h))
        loops_used = i + 1

        # Commit LOOP 1's state, always -- never a later loop's.
        #
        # Measured (scripts/test_state_commit_hypothesis.py, frozen checkpoint,
        # 20 trials, near-window synthetic recall):
        #     logits L1 / state L1  ->  40.0%   (the N=1 baseline)
        #     logits L1 / state L4  ->   0.0%   <- committing a later loop's state
        #     logits L4 / state L1  ->  35.0%   <- drifted logits are FINE
        #     logits L4 / state L4  ->   0.0%   (the old behaviour here)
        # Committing a later loop's state destroys recall on its own, with the
        # logits held identical; drifted logits with a clean state do not.
        #
        # Why: only loop 1 runs the blocks on the actual token embedding. Loops
        # 2+ run them on an already-transformed hidden state, so their k/v are
        # not derived from this token at all. For DeltaNet the state IS the
        # memory (S <- diag(a)S + b(v - Sk)k^T), so committing a later loop's
        # state writes an association keyed on noise -- and does it once per
        # extra loop, N times per token instead of once. It corrupts exactly
        # what the next token needs to read back.
        #
        # Semantically this is also the right contract: extra loops exist to
        # refine the PREDICTION for the current token, not to rewrite memory.
        # What the model records is "I saw token X at position t", which is
        # what loop 1 -- and only loop 1 -- actually computes.
        if i == 0:
            committed_states = candidate_states

        x = h  # feed the refined hidden state into the next loop, if any

        ent = normalized_entropy(logits, self.vocab_size)
        if ent <= self._looped_entropy_threshold:
            break

    self._loop_log.append(loops_used)
    return logits, committed_states


def enable_looped_early_exit(model, max_loops: int, entropy_threshold: float):
    """Attach (or reconfigure) looped early exit on an EdgeHVN instance.

    Idempotent: calling this again on an already-wrapped model just updates
    ``max_loops``/``entropy_threshold`` in place rather than re-wrapping —
    so a sweep can call this once per (N, T) combination on the same loaded
    model/checkpoint without reloading weights.
    """
    if not hasattr(model, "_looped_base_forward_step"):
        model._looped_base_forward_step = model.forward_step
        model.forward_step = types.MethodType(_looped_forward_step, model)
        model._loop_log = []
    model._looped_max_loops = max_loops
    model._looped_entropy_threshold = entropy_threshold
    return model


def disable_looped_early_exit(model):
    """Restore the original (unlooped) ``forward_step``. No-op if not enabled."""
    if hasattr(model, "_looped_base_forward_step"):
        model.forward_step = model._looped_base_forward_step
        del model._looped_base_forward_step
        del model._looped_max_loops
        del model._looped_entropy_threshold
        del model._loop_log
    return model


def pop_loop_log(model) -> list[int]:
    """Return and clear the per-forward_step-call loop-count log."""
    log = list(getattr(model, "_loop_log", []))
    if hasattr(model, "_loop_log"):
        model._loop_log.clear()
    return log


def looped_forward_train(model, input_ids, position_ids, k: int, deep_supervise: bool = True,
                         doc_ids=None):
    """Full-sequence (training-time) counterpart to _looped_forward_step.

    Runs the block stack k times over the WHOLE sequence via each block's
    ``forward()`` (not ``forward_step``) -- there is no explicit external
    state to clone/reset here, since a plain ``forward()`` call always
    computes its own recurrence/attention from scratch internally; looping
    is just re-feeding the previous loop's output hidden state back in,
    exactly mirroring the inference-time "no fresh re-embedding, state
    naturally resets every loop" semantics.

    Returns a list of per-loop logits: length k if deep_supervise (train on
    EVERY loop's prediction, not just the last -- otherwise the model is
    never taught that an early loop's output alone should be usable, which
    is exactly the property entropy-gated early exit depends on), else
    length 1 (only the final loop).

    ``doc_ids`` ([B, S], Issue #132) isolates packed documents. It is forwarded
    to every block on every loop: each loop recomputes the full
    recurrence/attention from scratch, so the boundary cut has to be reapplied
    each time rather than once up front.
    """
    if model.blocks is None:
        raise NotImplementedError(
            "looped training is not implemented for shared_weights=True models."
        )
    blk_kw = {} if doc_ids is None else {"doc_ids": doc_ids}
    h = model.token_embedding(input_ids)
    all_logits = []
    for i in range(k):
        for block in model.blocks:
            h = block(h, position_ids=position_ids, **blk_kw)
        logits = model.decoder(model.final_norm(h))
        if deep_supervise or i == k - 1:
            all_logits.append(logits)
    return all_logits
