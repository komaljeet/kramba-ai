"""#174: a resume must continue the torch RNG stream, not restart it.

EdgeHVN trains with dropout on (default p=0.1). Nothing saved the RNG, so a
resumed process drew different dropout masks from the first step on, even
with bit-identical weights, optimizer state, LR schedule and batches. Measured
on the real model (4L d128 hybrid, GPU, --deterministic, Muon, bf16): CUDA
Philox offset 3840 at the step-20 save vs 0 after resume, loss +1.6e-2 at the
first resumed step, all 59 weight tensors different 20 steps later.

These tests drive the real train() loop with a dropout model through a
mid-interval kill and compare the final WEIGHTS of the resumed run against an
uninterrupted one -- exact equality, since every other input is identical.
"""
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_edge_conversational as tec  # noqa: E402

VOCAB, SEQ, N_SAMPLES, BS, ACCUM = 64, 8, 50, 2, 2
SAVE_EVERY, TOTAL, KILL_AFTER_MICRO = 5, 20, 2 * 12 + 1   # killed inside step 13


class _Windows(Dataset):
    def __init__(self):
        g = torch.Generator().manual_seed(7)
        self.data = torch.randint(0, VOCAB, (N_SAMPLES, SEQ + 1), generator=g)

    def __len__(self):
        return N_SAMPLES

    def __getitem__(self, i):
        return self.data[i, :-1], self.data[i, 1:]


class _Killed(Exception):
    pass


class _DropLM(nn.Module):
    def __init__(self, kill_after=None):
        super().__init__()
        self.vocab_size, self.d_model, self.num_layers = VOCAB, 16, 1
        self.emb = nn.Embedding(VOCAB, 16)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(16, VOCAB)
        self.kill_after, self.calls = kill_after, 0

    def forward(self, x):
        if self.kill_after is not None and self.calls >= self.kill_after:
            raise _Killed
        self.calls += 1
        return self.out(self.drop(self.emb(x)))


def _config(tmp):
    return {"lr": 1e-2, "weight_decay": 0.0, "warmup_steps": 0, "grad_accum": ACCUM,
            "log_file": os.path.join(tmp, "log.csv"), "save_every_steps": SAVE_EVERY,
            "log_every_steps": 10_000, "batch_size": BS, "seq_len": SEQ,
            "optimizer": "adamw", "precision": "fp32"}


def _run(tmp, model, resume_step=0, resume_state=None):
    os.makedirs(tmp, exist_ok=True)
    ds = _Windows()
    loader = DataLoader(ds, batch_size=BS, drop_last=True,
                        sampler=tec.ResumableShuffleSampler(len(ds), 123))
    try:
        tec.train(model, loader, _config(tmp), torch.device("cpu"), tmp,
                  max_steps=TOTAL, resume_step=resume_step, resume_state=resume_state)
    except _Killed:
        pass
    return model


def _kill_and_resume(tmp_path, drop_rng_state=False):
    torch.manual_seed(0)
    ref = _run(str(tmp_path / "ref"), _DropLM())

    torch.manual_seed(0)
    ck = str(tmp_path / "killed")
    _run(ck, _DropLM(kill_after=KILL_AFTER_MICRO))
    resume_step = (KILL_AFTER_MICRO // ACCUM) // SAVE_EVERY * SAVE_EVERY
    state = tec._load_resume_state(ck, resume_step)
    assert state is not None
    if drop_rng_state:
        state.pop("rng_state", None)
    ckpt = torch.load(os.path.join(ck, f"edge_conversational_step{resume_step}.pt"),
                      weights_only=False)
    # A fresh process: a different RNG history before training resumes.
    torch.manual_seed(999)
    torch.rand(1000)
    resumed = _DropLM()
    resumed.load_state_dict(ckpt["model_state_dict"])
    _run(ck, resumed, resume_step=resume_step, resume_state=state)
    return ref, resumed, resume_step


def test_resumed_run_matches_uninterrupted_bit_for_bit(tmp_path):
    ref, resumed, resume_step = _kill_and_resume(tmp_path)
    assert resume_step == 10 and resumed.calls == (TOTAL - resume_step) * ACCUM
    for (n, a), (_, b) in zip(ref.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), n


def test_test_can_fail_without_the_saved_rng(tmp_path):
    """Control: drop the saved RNG (the pre-#174 behaviour) and the same
    harness must see different weights -- else the test above proves nothing."""
    ref, resumed, _ = _kill_and_resume(tmp_path, drop_rng_state=True)
    assert not torch.equal(ref.out.weight, resumed.out.weight)


def test_pre_174_state_is_tolerated():
    assert tec._restore_rng_state({"global_step": 5}) is False
    assert tec._restore_rng_state(None) is False
