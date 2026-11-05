"""#186: Check 2 must free its throwaway model copy before Check 3 sizes itself.

Check 2's `loss` kept the autograd graph alive, and that graph kept every
parameter of the deepcopy alive. Check 3's fit guard then saw that memory as in
use and could skip a check that fits.
"""
from __future__ import annotations

import gc
import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_edge_conversational as tec  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")

VOCAB = 1000
SEQ = 32
DEVICE = torch.device("cuda")


def _audit(tmp_path, capsys, monkeypatch, chunked_loss):
    """Run the audit; return (bytes Check 2 left allocated, one model copy's bytes)."""
    gc.collect()
    torch.cuda.empty_cache()
    model = EdgeHVN(
        vocab_size=VOCAB, d_model=128, num_layers=2, n_heads=4,
        dropout=0.0, use_deltanet=True,
    ).to(DEVICE)
    copy_bytes = sum(p.numel() * p.element_size() for p in model.parameters())

    allocated_at_guard = []

    def spy(m, d, n_copies):
        allocated_at_guard.append(torch.cuda.memory_allocated(d))
        return True, 0.0, float("inf")

    monkeypatch.setattr(tec, "_audit_extra_copies_fit", spy)
    x = torch.randint(0, VOCAB, (1, SEQ))
    y = torch.randint(0, VOCAB, (1, SEQ))
    config = {"batch_size": 1, "seq_len": SEQ, "precision": "bf16",
              "chunked_loss": chunked_loss}
    assert tec.pre_training_audit(model, [(x, y)], config, DEVICE, str(tmp_path))

    # Guards against a vacuous pass: Check 2 must have really run a backward.
    assert "Backward pass correct" in capsys.readouterr().out
    before_check2, before_check3 = allocated_at_guard
    return before_check3 - before_check2, copy_bytes


@pytest.mark.parametrize("chunked_loss", [0, 8])
def test_check2_frees_its_copy(monkeypatch, tmp_path, capsys, chunked_loss):
    # The first audit in a process also allocates ~16 MiB once, which would hide
    # or fake a leak of this small model, so only the second run is measured.
    _audit(tmp_path, capsys, monkeypatch, chunked_loss)
    left, copy_bytes = _audit(tmp_path, capsys, monkeypatch, chunked_loss)
    assert left < copy_bytes / 2, (
        f"Check 2 left {left / 2**20:.2f} MiB allocated; "
        f"one model copy is {copy_bytes / 2**20:.2f} MiB"
    )
