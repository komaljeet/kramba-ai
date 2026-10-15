"""#172: a resume must continue the EXACT batch sequence, not reshuffle.

The train loader used ``shuffle=True``. RandomSampler draws its seed from the
global torch RNG on every ``iter()``, so the order depended on how much RNG the
process had consumed, and nothing saved how far into the epoch the run was. A
hard-killed + resumed run therefore saw a different batch sequence from step 1
of the resume (measured on the real CLI: step-505 loss 7.90 vs 8.24).

These tests drive the real ``train()`` loop with a model that records every
batch it is fed, kill it mid-interval, resume from the saved state, and compare
the SEQUENCE of batches -- not a loss, which can coincide. The comparison spans
an epoch boundary so the next epoch's order is checked too.
"""
import hashlib
import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_edge_conversational as tec  # noqa: E402

VOCAB, SEQ, N_SAMPLES, BS, ACCUM = 64, 8, 50, 2, 2
# 50 samples / bs 2 = 25 micro-batches per epoch; with accum 2 an epoch ends
# mid-step 13, so the 30-step run below crosses two epoch boundaries.
SAVE_EVERY, TOTAL, KILL_AFTER_MICRO = 5, 30, 2 * 12 + 1   # killed inside step 13


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


class _Recorder(nn.Module):
    """Tiny LM that logs a hash of every batch it is fed, then 'dies'."""

    def __init__(self, log, kill_after=None):
        super().__init__()
        self.vocab_size, self.d_model, self.num_layers = VOCAB, 8, 1
        self.emb = nn.Embedding(VOCAB, 8)
        self.out = nn.Linear(8, VOCAB)
        self.log, self.kill_after = log, kill_after

    def forward(self, x):
        if self.kill_after is not None and len(self.log) >= self.kill_after:
            raise _Killed
        self.log.append(hashlib.sha1(x.numpy().tobytes()).hexdigest())
        return self.out(self.emb(x))


def _loader(seed):
    ds = _Windows()
    return DataLoader(ds, batch_size=BS, drop_last=True,
                      sampler=tec.ResumableShuffleSampler(len(ds), seed))


def _config(tmp):
    return {"lr": 1e-3, "weight_decay": 0.0, "warmup_steps": 0, "grad_accum": ACCUM,
            "log_file": os.path.join(tmp, "log.csv"), "save_every_steps": SAVE_EVERY,
            "log_every_steps": 10_000, "batch_size": BS, "seq_len": SEQ,
            "optimizer": "adamw", "precision": "fp32"}


def _run(tmp, loader, log, resume_step=0, resume_state=None, kill_after=None):
    os.makedirs(tmp, exist_ok=True)
    try:
        tec.train(_Recorder(log, kill_after), loader, _config(tmp), torch.device("cpu"),
                  tmp, max_steps=TOTAL, resume_step=resume_step,
                  resume_state=resume_state)
    except _Killed:
        pass


def _kill_and_resume(tmp_path, *, resume_seed, drop_data_state=False):
    ref = []
    _run(str(tmp_path / "ref"), _loader(seed=123), ref)
    assert len(ref) == TOTAL * ACCUM

    ck = str(tmp_path / "killed")
    _run(ck, _loader(seed=123), [], kill_after=KILL_AFTER_MICRO)
    resume_step = (KILL_AFTER_MICRO // ACCUM) // SAVE_EVERY * SAVE_EVERY
    state = tec._load_resume_state(ck, resume_step)
    assert state is not None, "the killed run left no resume state at its last save"
    if drop_data_state:
        state["data_state"] = None
    # A fresh process: different default torch seed -> different loader seed.
    resumed = []
    _run(ck, _loader(seed=resume_seed), resumed, resume_step=resume_step,
         resume_state=state)
    return ref, resumed, resume_step


def test_resumed_batch_sequence_matches_uninterrupted(tmp_path):
    ref, resumed, resume_step = _kill_and_resume(tmp_path, resume_seed=999)
    tail = ref[resume_step * ACCUM:]
    # Guards against a vacuous pass: the compared stretch must be long and must
    # cross an epoch boundary (25 micro-batches/epoch).
    assert resume_step == 10 and len(tail) == 40
    assert (resume_step * ACCUM) // 25 != (TOTAL * ACCUM - 1) // 25
    assert resumed == tail


def test_test_can_fail_without_the_saved_position(tmp_path):
    """Control: the same harness, with the pre-#172 behaviour (no saved data
    position), must detect the mismatch -- else the test above proves nothing."""
    ref, resumed, resume_step = _kill_and_resume(tmp_path, resume_seed=999,
                                                 drop_data_state=True)
    tail = ref[resume_step * ACCUM:]
    assert len(resumed) == len(tail)
    mismatches = sum(a != b for a, b in zip(resumed, tail))
    assert mismatches > len(tail) // 2


def test_sampler_restart_skips_without_reading_and_epochs_differ():
    s = tec.ResumableShuffleSampler(1000, seed=5)
    full = list(iter(s))                       # epoch 0
    nxt = list(iter(s))                        # auto-advances to epoch 1
    assert sorted(full) == list(range(1000)) and full != nxt
    s.set_position(0, 640)
    assert list(iter(s)) == full[640:]
    s.set_position(1, 0)
    assert list(iter(s)) == nxt


def test_audit_peek_does_not_move_training_order(tmp_path):
    """pre_training_audit calls next(iter(dataloader)) before train(); that iter
    must not consume the epoch the training loop starts on."""
    a, b = [], []
    _run(str(tmp_path / "a"), _loader(seed=1), a)
    peeked = _loader(seed=1)
    next(iter(peeked))
    _run(str(tmp_path / "b"), peeked, b)
    assert a == b


def test_pipeline_uses_resumable_sampler_seeded_from_initial_seed(monkeypatch):
    class _Stub(Dataset):
        def __init__(self, tokenizer, seq_len, max_chars):
            self.token_ids = list(range(200))

        def __len__(self):
            return 20

        def __getitem__(self, i):
            return torch.zeros(4, dtype=torch.long), torch.zeros(4, dtype=torch.long)

    monkeypatch.setattr(tec, "WikiText103LocalDataset", _Stub)
    torch.manual_seed(4242)
    loader, _v, _s = tec.build_dataset_pipeline(None, seq_len=4, batch_size=2, smoke=True)
    assert isinstance(loader.sampler, tec.ResumableShuffleSampler)
    assert loader.sampler.seed == 4242
