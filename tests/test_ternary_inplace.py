"""In-place ternary conversion: no weight copies, same RNG stream, ties unchanged.

``apply_ternary_to_model`` used to clone every weight on top of a fresh xavier
init and a zeros probe, while a dict kept every old ``nn.Linear`` alive -- ~4x
the model's fp32 size in host RAM at peak on a 1B --offload build. The fix
adopts each dense Parameter. These tests pin the three things that could go
silently wrong with that:

* a copy sneaking back in (checked by storage pointer, not by values -- a copy
  has the same values, so a value check cannot fail on it);
* the global CPU RNG ending in a different state than before, which would change
  a seeded run's DataLoader shuffle versus every earlier run;
* a tied weight becoming shared where the old code broke the tie.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from hvn.edge.ternary import GROUP_PER_TENSOR, TernaryLinear, apply_ternary_to_model


def _legacy_from_linear(linear: nn.Linear, group_size: int = GROUP_PER_TENSOR):
    """The pre-fix implementation, verbatim in effect: zeros probe, xavier, clone."""
    m = TernaryLinear(linear.in_features, linear.out_features,
                      bias=linear.bias is not None, group_size=group_size)
    torch.zeros(linear.out_features, linear.in_features)
    m.weight.data = linear.weight.data.clone()
    if m.bias is not None:
        m.bias.data = linear.bias.data.clone()
    return m


def _legacy_apply(model: nn.Module, names):
    mods = dict(model.named_modules())
    for name in names:
        *pp, attr = name.split(".")
        parent = model
        for p in pp:
            parent = getattr(parent, p)
        setattr(parent, attr, _legacy_from_linear(mods[name]))


class _Net(nn.Module):
    def __init__(self, tie: bool = False):
        super().__init__()
        self.emb = nn.Embedding(50, 24)
        self.a = nn.Linear(24, 40)
        self.b = nn.Linear(40, 24, bias=False)
        self.inner = nn.Sequential(nn.Linear(24, 24), nn.GELU(), nn.Linear(24, 24))
        self.decoder = nn.Linear(24, 50, bias=False)
        if tie:
            self.decoder.weight = self.emb.weight

    def forward(self, ids):
        h = self.emb(ids)
        h = self.b(torch.relu(self.a(h)))
        return self.decoder(self.inner(h))


def test_adopts_storage_no_copy():
    torch.manual_seed(0)
    m = _Net()
    ptrs = {n: (mod.weight.data_ptr(), mod.bias.data_ptr() if mod.bias is not None else None)
            for n, mod in m.named_modules() if isinstance(mod, nn.Linear)}
    vals = {n: mod.weight.detach().clone() for n, mod in m.named_modules()
            if isinstance(mod, nn.Linear)}
    out = apply_ternary_to_model(m)
    assert len(out) == 5  # vacuity guard: every Linear actually went through
    for n, layer in out.items():
        assert isinstance(layer, TernaryLinear)
        assert layer.weight.data_ptr() == ptrs[n][0], f"{n}: weight was copied"
        if ptrs[n][1] is not None:
            assert layer.bias.data_ptr() == ptrs[n][1], f"{n}: bias was copied"
        assert torch.equal(layer.weight.detach(), vals[n])
        assert layer.weight.requires_grad
    assert not any(isinstance(x, nn.Linear) for x in m.modules())


def test_matches_legacy_rng_values_and_output():
    """Same seed -> same CPU RNG state, same weights, same forward as the old path."""
    names = ["a", "b", "inner.0", "inner.2"]  # decoder excluded, like the trainer
    torch.manual_seed(123)
    ref = _Net()
    _legacy_apply(ref, names)
    rng_ref = torch.get_rng_state()
    x_ref = torch.randint(0, 50, (2, 7))

    torch.manual_seed(123)
    new = _Net()
    apply_ternary_to_model(new, target_names=names)
    rng_new = torch.get_rng_state()
    x_new = torch.randint(0, 50, (2, 7))

    assert torch.equal(rng_ref, rng_new), "conversion changed the CPU RNG stream"
    assert torch.equal(x_ref, x_new)
    sd_ref, sd_new = ref.state_dict(), new.state_dict()
    assert sd_ref.keys() == sd_new.keys()
    for k in sd_ref:
        assert torch.equal(sd_ref[k], sd_new[k]), k
    assert torch.equal(ref(x_ref), new(x_new))


def test_rng_guard_is_not_vacuous():
    """If the RNG advance were dropped, the state WOULD differ -- so the test above can fail."""
    torch.manual_seed(123)
    _Net()
    skipped = torch.get_rng_state()
    torch.manual_seed(123)
    m = _Net()
    _legacy_apply(m, ["a"])
    assert not torch.equal(skipped, torch.get_rng_state())


def test_tied_weight_is_cloned_as_before():
    torch.manual_seed(0)
    m = _Net(tie=True)
    assert m.decoder.weight is m.emb.weight
    out = apply_ternary_to_model(m)
    dec = out["decoder"]
    assert dec.weight is not m.emb.weight
    assert dec.weight.data_ptr() != m.emb.weight.data_ptr()
    assert torch.equal(dec.weight.detach(), m.emb.weight.detach())
    # the untied ones were still adopted
    assert out["a"].weight.data_ptr() != dec.weight.data_ptr()


def test_shared_module_is_not_treated_as_tied():
    """One Linear reachable by two paths (shared_weights) is adopted, converted once."""
    lin = nn.Linear(8, 8)
    m = nn.Module()
    m.p = nn.Module()
    m.q = nn.Module()
    m.p.l = lin
    m.q.l = lin
    ptr = lin.weight.data_ptr()
    out = apply_ternary_to_model(m)
    assert list(out) == ["p.l"]
    assert out["p.l"].weight.data_ptr() == ptr
