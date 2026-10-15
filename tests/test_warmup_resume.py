"""#173: a resume taken mid-warmup must finish the warmup, not skip it.

train() used to set ``warmup_done = True`` on every resume. Since #112/#165 the
cosine is CONTINUED from its saved position, and mid-warmup that position is 0,
so the forced flag started the cosine early, anchored on the half-warmed LR: the
run never reached peak LR again (real CLI, warmup 100, resumed at step 40: LR
stuck at ~0.4x the uninterrupted run for the rest of the schedule).

These tests drive the real ``train()`` loop, kill it mid-warmup, resume from the
saved state, and compare the LR of EVERY step after the resume -- through the
warmup boundary and well into the cosine -- against an uninterrupted run.
"""
import csv
import os
import sys

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_edge_conversational as tec  # noqa: E402

VOCAB, SEQ, N_SAMPLES, BS, ACCUM = 64, 8, 200, 2, 2
WARMUP, SAVE_EVERY, TOTAL = 20, 5, 40
KILL_AFTER_MICRO = 2 * 12 + 1        # killed inside step 13 -> resumes from step 10
RESUME_STEP = 10


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


class _TinyLM(nn.Module):
    def __init__(self, kill_after=None):
        super().__init__()
        self.vocab_size, self.d_model, self.num_layers = VOCAB, 8, 1
        self.emb = nn.Embedding(VOCAB, 8)
        self.out = nn.Linear(8, VOCAB)
        self.calls, self.kill_after = 0, kill_after

    def forward(self, x):
        if self.kill_after is not None and self.calls >= self.kill_after:
            raise _Killed
        self.calls += 1
        return self.out(self.emb(x))


def _loader():
    ds = _Windows()
    return DataLoader(ds, batch_size=BS, drop_last=True,
                      sampler=tec.ResumableShuffleSampler(len(ds), 123))


def _config(tmp, warmup):
    return {"lr": 1e-3, "weight_decay": 0.0, "warmup_steps": warmup, "grad_accum": ACCUM,
            "log_file": os.path.join(tmp, "log.csv"), "save_every_steps": SAVE_EVERY,
            "log_every_steps": 1, "batch_size": BS, "seq_len": SEQ,
            "optimizer": "adamw", "precision": "fp32"}


def _run(tmp, resume_step=0, resume_state=None, kill_after=None, warmup=WARMUP):
    os.makedirs(tmp, exist_ok=True)
    try:
        tec.train(_TinyLM(kill_after), _loader(), _config(tmp, warmup),
                  torch.device("cpu"), tmp, max_steps=TOTAL,
                  resume_step=resume_step, resume_state=resume_state)
    except _Killed:
        pass


def _lrs(tmp):
    # The resumed run appends to the same CSV; the last row for a step wins.
    out = {}
    with open(os.path.join(tmp, "log.csv"), encoding="utf-8") as f:
        for row in csv.DictReader(f):
            out[int(row["step"])] = float(row["lr"])
    return out


def _kill_and_resume(tmp_path, *, resume_warmup=WARMUP):
    ref_dir, ck = str(tmp_path / "ref"), str(tmp_path / "killed")
    _run(ref_dir)
    _run(ck, kill_after=KILL_AFTER_MICRO)
    state = tec._load_resume_state(ck, RESUME_STEP)
    assert state is not None, "the killed run left no resume state at its last save"
    _run(ck, resume_step=RESUME_STEP, resume_state=state, warmup=resume_warmup)
    ref, got = _lrs(ref_dir), _lrs(ck)
    steps = list(range(RESUME_STEP + 1, TOTAL + 1))
    return [ref[s] for s in steps], [got[s] for s in steps]


def test_mid_warmup_resume_matches_uninterrupted_lr(tmp_path):
    ref, got = _kill_and_resume(tmp_path)
    # Vacuity guards: the resume is mid-warmup, the compared stretch crosses the
    # warmup boundary, and the LR actually moves on both sides of it.
    assert 0 < RESUME_STEP < WARMUP < TOTAL
    assert len(ref) == TOTAL - RESUME_STEP
    assert ref[WARMUP - RESUME_STEP - 1] > 1.5 * ref[0]        # still warming up
    assert ref[-1] < 0.5 * max(ref)                            # cosine decaying
    for r, g in zip(ref, got):
        assert abs(r - g) <= 1e-7, (ref, got)


def test_test_can_fail_when_warmup_is_skipped_on_resume(tmp_path):
    """Control: resuming with warmup 0 reproduces the pre-#173 forced
    ``warmup_done = True``. The harness must see that, else the test above
    proves nothing."""
    ref, got = _kill_and_resume(tmp_path, resume_warmup=0)
    mismatches = sum(abs(r - g) > 1e-7 for r, g in zip(ref, got))
    assert mismatches > len(ref) // 2
    # The pre-fix signature: the run never gets back to peak LR.
    assert max(got) < 0.75 * max(ref)
