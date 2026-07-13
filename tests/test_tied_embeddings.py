"""Tests for tied token_embedding / decoder output-projection weights
(Issue #106, the classic GPT-2/ALBERT trick). Opt-in (`tie_embeddings=False`
default) since, unlike most of epic #78, this changes what's actually being
trained, not just memory bookkeeping -- these tests are the parity/
correctness sign-off the issue itself flagged as required before landing.
No GPU needed -- runs on CPU with tiny dims.

Covers:
  1. The two weights are the SAME tensor object (the actual memory-savings
     mechanism -- not just equal values, genuinely shared storage).
  2. Tying halves the embedding+output-projection param count vs untied.
  3. Backward populates the shared weight's gradient correctly (accumulated
     from both usage sites: as an embedding lookup AND as an output
     projection) -- a plausible failure mode for a naive tying
     implementation is a stale/`None` grad on one side.
  4. Checkpoint save/load roundtrip preserves the tie -- rebuilding from a
     saved checkpoint with tie_embeddings=True must still end up with the
     SAME shared tensor, not two independently-loaded copies that happen to
     start equal.
  5. Training convergence smoke test: loss decreases with tying enabled,
     same as any other config (correctness, not a performance claim).

Run:  python -m pytest tests/test_tied_embeddings.py -q
  or:  python tests/test_tied_embeddings.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.edge.model import EdgeHVN

VOCAB, D_MODEL = 64, 32


def _tiny_model(tie_embeddings: bool, seed: int = 0):
    torch.manual_seed(seed)
    return EdgeHVN(vocab_size=VOCAB, d_model=D_MODEL, num_layers=2, n_heads=4,
                   use_deltanet=True, chunk_size=8, dropout=0.0,
                   tie_embeddings=tie_embeddings)


def test_weights_are_the_same_tensor():
    model = _tiny_model(tie_embeddings=True)
    assert model.decoder.decoder.weight is model.token_embedding.weight


def test_untied_weights_are_different_tensors():
    """Regression guard: the default path must NOT accidentally tie."""
    model = _tiny_model(tie_embeddings=False)
    assert model.decoder.decoder.weight is not model.token_embedding.weight


def test_tying_halves_embedding_output_param_count():
    untied = _tiny_model(tie_embeddings=False)
    tied = _tiny_model(tie_embeddings=True)
    untied_total = sum(p.numel() for p in untied.parameters())
    tied_total = sum(p.numel() for p in tied.parameters())
    embedding_params = VOCAB * D_MODEL
    assert untied_total - tied_total == embedding_params


def test_gradient_flows_to_shared_weight():
    model = _tiny_model(tie_embeddings=True)
    ids = torch.randint(0, VOCAB, (2, 16))
    logits = model(ids)
    loss = F.cross_entropy(logits.reshape(-1, VOCAB), ids.reshape(-1))
    loss.backward()
    grad = model.token_embedding.weight.grad
    assert grad is not None
    assert torch.isfinite(grad).all()
    assert grad.abs().sum().item() > 0  # not a dead/all-zero gradient
    assert model.decoder.decoder.weight.grad is grad  # same tensor -> same .grad


def test_checkpoint_roundtrip_preserves_tie(tmp_path):
    model = _tiny_model(tie_embeddings=True)
    ckpt_path = tmp_path / "tied_ckpt.pt"
    torch.save({"model_state_dict": model.state_dict()}, ckpt_path)

    rebuilt = EdgeHVN(vocab_size=VOCAB, d_model=D_MODEL, num_layers=2, n_heads=4,
                       use_deltanet=True, chunk_size=8, dropout=0.0,
                       tie_embeddings=True)
    ckpt = torch.load(ckpt_path, weights_only=False)
    rebuilt.load_state_dict(ckpt["model_state_dict"])
    assert rebuilt.decoder.decoder.weight is rebuilt.token_embedding.weight
    assert torch.equal(rebuilt.token_embedding.weight, model.token_embedding.weight)


def test_tied_init_is_rescaled_not_embedding_default():
    """Regression guard for the divergence bug found via scripts/bench_harness.py:
    naively tying (keeping nn.Embedding's default init, std~1.0) hands the
    decoder a ~20x-too-large weight (nn.Linear's default init here is
    std~0.05), producing huge logits and a DIVERGING loss (25+ and climbing
    in the original CPU parity run, vs untied's stable ~9). The fix re-inits
    the shared tensor to GPT-2's standard std=0.02 after tying -- assert
    that directly so this can't silently regress back to the embedding's
    default scale."""
    model = _tiny_model(tie_embeddings=True)
    std = model.token_embedding.weight.std().item()
    assert abs(std - 0.02) < 0.01, f"tied weight std={std:.4f}, expected ~0.02"


def test_training_converges_with_tied_embeddings():
    model = _tiny_model(tie_embeddings=True)
    opt = torch.optim.AdamW(model.parameters(), lr=5e-3)

    torch.manual_seed(1)
    ids = torch.randint(0, VOCAB, (4, 16))
    losses = []
    for _ in range(60):
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, VOCAB), ids.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    early_mean = sum(losses[:10]) / 10
    late_mean = sum(losses[-10:]) / 10
    assert late_mean < early_mean, (
        f"loss did not decrease with tied embeddings: early={early_mean:.4f} late={late_mean:.4f}"
    )


if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            if "tmp_path" in t.__code__.co_varnames[:t.__code__.co_argcount]:
                import tempfile
                from pathlib import Path
                with tempfile.TemporaryDirectory() as d:
                    t(Path(d))
            else:
                t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    raise SystemExit(1 if failed else 0)
