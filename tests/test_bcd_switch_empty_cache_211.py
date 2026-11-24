"""#211: ``--bcd-switch-empty-cache`` empties the CUDA cache at each BCD switch.

Author: Rambarun Komaljeet

Claims and the test for each:
* Same numbers: emptying the cache only moves memory, so losses and every active
  gradient over several steps and switches are bit-identical (torch.equal) to the
  run without it.
* It really releases memory at the switch: with the flag, empty_cache runs once per
  switch and a freed 64 MiB block cached before each switch is gone right after
  it (reserved drops by >= 64 MiB). Must-fail partner: without the flag the same
  counters show no call and no drop, so the release check is not vacuous.
* #212: the flag is ON by default; --no-bcd-switch-empty-cache turns it off and
  the old --bcd-switch-empty-cache still parses on.
* #212: the DEFAULT parsed args, turned into the flag the way main() does, really
  empty the cache at each switch; the opt-out's parsed args do not. Must-fail
  partner: flipping the parser default back to False makes the default test fail
  (checked by hand when #212 landed).
"""
import argparse
import os
import sys

import pytest
import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import train_edge_conversational as T  # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="offload needs CUDA")
L, V, GA = 6, 64, 2


def _run(flag, monkeypatch, steps=8):
    calls = {"n": 0}
    orig = torch.cuda.empty_cache

    def counted():
        calls["n"] += 1
        orig()
    monkeypatch.setattr(torch.cuda, "empty_cache", counted)
    torch.cuda.empty_cache()
    calls["n"] = 0
    torch.manual_seed(0)
    dev = torch.device("cuda")
    m = EdgeHVN(vocab_size=V, d_model=32, num_layers=L, n_heads=4, use_deltanet=True,
                chunk_size=8, dropout=0.0, grad_checkpoint=True).cuda()
    opt = torch.optim.AdamW(m.parameters(), lr=1e-2)
    sched = BlockCoordinateScheduler(m, opt, steps_per_block=2, offload_inactive_weights=True,
                                     device=dev, offload_prefetch=True,
                                     partition="balanced", layers_per_block=3)
    calls["n"] = 0          # the scheduler's own constructor-time empty_cache
    g = torch.Generator().manual_seed(1)
    ids = torch.randint(0, V, (steps, GA, 2, 16), generator=g)
    losses, grads, drops = [], [], []
    for s in range(steps):
        for mb in range(GA):
            x = ids[s, mb].cuda()
            loss = F.cross_entropy(m(x).reshape(-1, V), x.reshape(-1)) / GA
            loss.backward()
            losses.append(loss.detach().clone())
        grads.append([p.grad.detach().clone() for p in m.parameters()
                      if p.requires_grad and p.grad is not None])
        opt.step()
        opt.zero_grad()
        # A freed 64 MiB block sits in the cache at every step end, standing in for
        # the activation segments that are free at a real switch (a tiny model's
        # own segments are shared with live tensors, so it would release nothing).
        tmp = torch.empty(64 * 2**20, dtype=torch.uint8, device=dev)
        del tmp
        before = torch.cuda.memory_reserved()
        if T.bcd_step(sched, flag, dev):
            drops.append(before - torch.cuda.memory_reserved())
    return losses, grads, drops, calls["n"]


@cuda_only
def test_same_bits_and_cache_released_at_each_switch(monkeypatch):
    l0, g0, d0, n0 = _run(False, monkeypatch)
    l1, g1, d1, n1 = _run(True, monkeypatch)
    assert len(l0) == len(l1) and all(torch.equal(a, b) for a, b in zip(l0, l1)), "loss moved"
    for s, (a, b) in enumerate(zip(g0, g1)):
        assert len(a) == len(b) > 0, f"step {s}: no active gradients to compare"
        for x, y in zip(a, b):
            assert torch.equal(x, y), f"step {s}: max |d| {(x - y).abs().max().item():.3e}"
    assert len(d1) >= 3, "too few switches for the release check to mean anything"
    assert n1 == len(d1), (n1, len(d1))
    assert all(d >= 64 * 2**20 for d in d1), f"reserved did not drop at a switch: {d1}"


@cuda_only
def test_must_fail_partner_no_flag_no_release(monkeypatch):
    """Without the flag the same counters must show nothing: no call, no drop."""
    _, _, d0, n0 = _run(False, monkeypatch)
    assert len(d0) >= 3
    assert n0 == 0
    assert all(d <= 0 for d in d0), d0


@cuda_only
def test_equality_check_sees_one_ulp(monkeypatch):
    """The bit-identity check above is sensitive at the last bit."""
    _, g0, _, _ = _run(False, monkeypatch, steps=1)
    x = g0[0][0].clone()
    x.view(-1)[0] = torch.nextafter(x.view(-1)[0], x.view(-1)[0] + 1)
    assert not torch.equal(g0[0][0], x)


_ORIG_PARSE_ARGS = argparse.ArgumentParser.parse_args


def _trainer_args(monkeypatch, argv):
    got = {}

    def stop(self, *a, **k):
        got["ns"] = _ORIG_PARSE_ARGS(self, *a, **k)
        raise SystemExit(0)
    monkeypatch.setattr(argparse.ArgumentParser, "parse_args", stop)
    monkeypatch.setattr(sys, "argv", ["train_edge_conversational.py"] + argv)
    with pytest.raises(SystemExit):
        T.main()
    return got["ns"]


def test_flag_on_by_default_opt_out_parses_off(monkeypatch):
    assert _trainer_args(monkeypatch, []).bcd_switch_empty_cache is True
    assert _trainer_args(monkeypatch, ["--bcd-switch-empty-cache"]).bcd_switch_empty_cache is True
    assert _trainer_args(monkeypatch,
                         ["--no-bcd-switch-empty-cache"]).bcd_switch_empty_cache is False


def _flag_as_main_sets_it(argv):
    """The value main() stores in config["bcd_switch_empty_cache"] for these argv."""
    return bool(T.build_arg_parser().parse_args(argv).bcd_switch_empty_cache)


@cuda_only
def test_default_args_empty_cache_at_each_switch(monkeypatch):
    """#212: with NO flag given, the switch really releases the cache."""
    _, _, d, n = _run(_flag_as_main_sets_it([]), monkeypatch)
    assert len(d) >= 3, "too few switches for the release check to mean anything"
    assert n == len(d), (n, len(d))
    assert all(x >= 64 * 2**20 for x in d), f"default did not release at a switch: {d}"


@cuda_only
def test_opt_out_args_do_not_empty_cache(monkeypatch):
    """#212: --no-bcd-switch-empty-cache really turns the release off."""
    _, _, d, n = _run(_flag_as_main_sets_it(["--no-bcd-switch-empty-cache"]), monkeypatch)
    assert len(d) >= 3
    assert n == 0
    assert all(x <= 0 for x in d), d
