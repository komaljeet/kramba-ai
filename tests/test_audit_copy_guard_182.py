"""#182: pre_training_audit's Check 2/3 skip when their extra copies cannot fit.

Both checks hold two extra weight-sized copies of the model on GPU (Check 2: a
deepcopy + its grads; Check 3: the state dict loaded onto the device + the
rebuilt model). At ~1.1B on a 6 GB card that OOMed inside the audit itself,
before training ever started. The guard must skip those cases and must NOT skip
ones that fit -- the second half is what keeps the real check alive for small
models, so both directions are pinned here on a faked card.
"""
from __future__ import annotations

import os
import sys
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_edge_conversational as tec  # noqa: E402

GiB = 2**30
MiB = 2**20
TOTAL = 6 * GiB


def _fake_card(monkeypatch, driver_free, reserved=0, allocated=0):
    monkeypatch.setattr(tec.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(tec.torch.cuda, "get_device_properties",
                        lambda d=None: SimpleNamespace(total_memory=TOTAL))
    monkeypatch.setattr(tec.torch.cuda, "mem_get_info",
                        lambda d=None: (driver_free, TOTAL))
    monkeypatch.setattr(tec.torch.cuda, "memory_reserved", lambda d=None: reserved)
    monkeypatch.setattr(tec.torch.cuda, "memory_allocated", lambda d=None: allocated)


def _model_4mib():
    m = nn.Linear(1024, 1024, bias=False)  # 1024*1024*4 B = exactly 4 MiB fp32
    assert sum(p.numel() * p.element_size() for p in m.parameters()) == 4 * MiB
    return m


CUDA = torch.device("cuda")


def _driver_free_for_room(room):
    # room = 0.90*TOTAL - (TOTAL - free)  =>  free = room + 0.10*TOTAL
    return int(room + (1 - tec.AUDIT_COPY_MAX_VRAM_FRAC) * TOTAL)


def test_small_model_with_room_still_runs_the_checks(monkeypatch):
    _fake_card(monkeypatch, driver_free=5 * GiB)
    fits, need, _ = tec._audit_extra_copies_fit(_model_4mib(), CUDA, n_copies=2)
    assert fits
    assert need == pytest.approx(8 * MiB / GiB)


def test_skips_when_two_copies_exceed_the_margin(monkeypatch):
    _fake_card(monkeypatch, driver_free=_driver_free_for_room(8 * MiB - 1024))
    fits, _, _ = tec._audit_extra_copies_fit(_model_4mib(), CUDA, n_copies=2)
    assert not fits


def test_just_enough_room_is_not_skipped(monkeypatch):
    _fake_card(monkeypatch, driver_free=_driver_free_for_room(8 * MiB + 1024))
    fits, _, _ = tec._audit_extra_copies_fit(_model_4mib(), CUDA, n_copies=2)
    assert fits


def test_pytorch_cached_memory_counts_as_reusable(monkeypatch):
    # Driver sees too little free, but 10 MiB sits in PyTorch's cache unused.
    _fake_card(monkeypatch, driver_free=_driver_free_for_room(0),
               reserved=10 * MiB, allocated=0)
    fits, _, _ = tec._audit_extra_copies_fit(_model_4mib(), CUDA, n_copies=2)
    assert fits


def test_unwraps_compiled_module(monkeypatch):
    # A torch.compile OptimizedModule exposes the real model as _orig_mod.
    inner = _model_4mib()
    wrapper = nn.Module()
    wrapper._orig_mod = inner
    _fake_card(monkeypatch, driver_free=5 * GiB)
    _, need, _ = tec._audit_extra_copies_fit(wrapper, CUDA, n_copies=2)
    assert need == pytest.approx(8 * MiB / GiB)


def test_cpu_device_never_skips():
    fits, _, _ = tec._audit_extra_copies_fit(_model_4mib(), torch.device("cpu"), 2)
    assert fits
