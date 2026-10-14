"""#168: a tied EdgeHVN no longer allocates a decoder weight just to discard it.

The decoder of a tied model is now built on ``meta`` and the CPU RNG is advanced
by hand instead of materialising a [vocab, d_model] fp32 matrix (0.38 GiB at 1B)
that the tie immediately throws away. What could go silently wrong:

* the generator ending in a different state -- every later draw (dropout, the
  DataLoader shuffle) would then differ from every earlier seeded run;
* the bias or the shared table taking different values;
* a meta tensor surviving into the finished model.

The reference is the untouched untied path plus the tie's own ``normal_``, which
is exactly what the old tied constructor drew, in the same order.
"""
from __future__ import annotations

import torch
import torch.nn as nn

import hvn.edge.model as edge_model
from hvn.edge.model import EdgeHVN

# vocab * d_model = 1,050,304: above the RNG helper's 1<<20 chunk and not a
# multiple of it, so the chunked advance actually crosses a chunk boundary.
_KW = dict(vocab_size=16411, d_model=64, num_layers=2, n_heads=4, use_deltanet=True)


def _build(tie: bool, seed: int = 1234) -> tuple[EdgeHVN, torch.Tensor]:
    torch.manual_seed(seed)
    m = EdgeHVN(tie_embeddings=tie, **_KW)
    return m, torch.get_rng_state()


def _reference(seed: int = 1234) -> tuple[EdgeHVN, torch.Tensor]:
    """What the pre-#168 tied constructor produced: untied build, then the tie's normal_."""
    torch.manual_seed(seed)
    m = EdgeHVN(tie_embeddings=False, **_KW)
    with torch.no_grad():
        nn.init.normal_(m.token_embedding.weight, mean=0.0, std=0.02)
    return m, torch.get_rng_state()


def test_tied_ctor_bit_identical_to_old_path():
    m, rng = _build(tie=True)
    ref, ref_rng = _reference()
    assert torch.equal(rng, ref_rng), "CPU RNG left in a different state"
    sd, rsd = m.state_dict(), ref.state_dict()
    assert sd.keys() == rsd.keys()
    for k in rsd:
        if k == "decoder.decoder.weight":
            continue  # the reference is untied; the tied value is the embedding
        assert torch.equal(sd[k], rsd[k]), k
    assert torch.equal(m.decoder.decoder.weight, ref.token_embedding.weight)


def test_tied_ctor_shares_storage_and_has_no_meta():
    m, _ = _build(tie=True)
    assert m.decoder.decoder.weight is m.token_embedding.weight
    for name, p in list(m.named_parameters()) + list(m.named_buffers()):
        assert p.device.type == "cpu", name
    logits = m(torch.randint(0, _KW["vocab_size"], (1, 8)))
    logits = logits[0] if isinstance(logits, tuple) else logits
    assert torch.isfinite(logits).all()


def test_lean_path_is_taken(monkeypatch):
    calls = []
    real = edge_model._advance_cpu_rng_like_uniform
    monkeypatch.setattr(edge_model, "_advance_cpu_rng_like_uniform",
                        lambda n, *a, **k: (calls.append(n), real(n, *a, **k)))
    _build(tie=True)
    assert calls == [_KW["vocab_size"] * _KW["d_model"]]
    calls.clear()
    _build(tie=False)
    assert calls == []


def test_rng_check_can_fail(monkeypatch):
    """Guard against the main test going vacuous: drop the RNG advance and the
    comparison must notice."""
    monkeypatch.setattr(edge_model, "_advance_cpu_rng_like_uniform", lambda n, *a, **k: None)
    m, rng = _build(tie=True)
    _, ref_rng = _reference()
    assert not torch.equal(rng, ref_rng)
    assert not torch.equal(m.decoder.decoder.bias, _reference()[0].decoder.decoder.bias)
