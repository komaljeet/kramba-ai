"""#213: the trainer's --val-every eval under BCD --offload.

Author: Rambarun Komaljeet

Before the fix, ``evaluate()`` put the model in eval() mode, and EdgeHVN.forward
only streams offloaded (CPU-parked) frozen layers when ``training and
grad_enabled``. The eval forward fell through to the plain path and crashed on
the first frozen layer: ``Expected all tensors to be on the same device, but
got weight is on cpu`` (reproduced on the real 1B k4 run, origin/main 4611a2f,
step 1, ``hvn/core/hvn_block.py`` rms_norm of blocks.0).

What these tests pin, each with a partner that shows it can fail:

  runs + right number  evaluate() on an offloaded model runs, and its per-window
                       losses equal BOTH the #209 harness walk
                       (scripts/ab_heldout_209.py ``seg``) and a plain GPU-resident
                       copy of the same weights, bit for bit. Fails on main (the
                       crash). Non-vacuous: frozen weights are asserted to be on the
                       CPU before the eval and some frozen block to have trained.
  leaves no trace      after an eval every frozen weight is back on the CPU, the
                       pool holds nothing, train mode and requires_grad are back.
  training unchanged   the REAL train() loop with eval after every step vs eval off:
                       per-step loss, gradient and RNG sha256 identical at every BCD
                       block position (2 steps each, nonzero LR, dropout on). A
                       determinism control runs eval-off twice. Falsifiers: an eval
                       that skips the RNG restore, and one that nudges the optimizer
                       state, must each flip the hashes.
  CPU, real train()    eval every step vs off on a dropout model on the CPU: final
                       weights identical. On main they differ (iter(val_loader)
                       draws from the CPU RNG, dropout then draws different masks).
  resume               a run with eval on, killed and resumed from its checkpoint,
                       ends bit-identical to the uninterrupted eval-on run.
"""
import contextlib
import hashlib
import os
import sys

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import train_edge_conversational as tec  # noqa: E402

CUDA = torch.cuda.is_available()
needs_cuda = pytest.mark.skipif(not CUDA, reason="offload needs a CUDA device")

VOCAB, SEQ, SPB = 256, 32, 2


# --------------------------------------------------------------------------
# Tiny offloaded EdgeHVN, built the way train() builds the real one
# --------------------------------------------------------------------------

def _tiny_model(dropout=0.0, ternary=True):
    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model
    torch.manual_seed(0)
    m = EdgeHVN(vocab_size=VOCAB, d_model=64, num_layers=4, n_heads=4,
                block_plan=["deltanet", "transformer", "deltanet", "deltanet"],
                attn_window=SEQ, use_deltanet=True, dropout=dropout,
                tie_embeddings=True, grad_checkpoint=True)
    if ternary:
        # Block Linears only (the tied head stays a plain Linear, as in the trainer).
        apply_ternary_to_model(m, [n for n, mod in m.named_modules()
                                   if isinstance(mod, nn.Linear) and n.startswith("blocks.")])
    return m


def _batches(n, seed):
    g = torch.Generator().manual_seed(seed)
    x = torch.randint(0, VOCAB, (n, SEQ + 1), generator=g)
    return x[:, :-1].contiguous(), x[:, 1:].contiguous()


def _val_loader(n=6):
    x, y = _batches(n, 99)
    src = torch.zeros(n, dtype=torch.long)
    # A real DataLoader: iter() draws a base seed from the global CPU RNG.
    return DataLoader(TensorDataset(x, y, src), batch_size=1, shuffle=False)


class _Windows(Dataset):
    def __init__(self, n=40):
        self.x, self.y = _batches(n, 7)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]


def _gpu_config(tmp, val_every):
    return {"lr": 3e-4, "weight_decay": 0.0, "warmup_steps": 0, "grad_accum": 2,
            "log_file": os.path.join(tmp, "log.csv"), "save_every_steps": 100_000,
            "log_every_steps": 100_000, "batch_size": 1, "seq_len": SEQ,
            "optimizer": "muon", "muon_lr": 0.02, "precision": "bf16",
            "chunked_loss": 16, "block_coordinate": True, "offload": True,
            "bcd_partition": "layer", "bcd_steps_per_block": SPB,
            "val_every_steps": val_every, "val_batches": 4,
            # The trainer's 85% spill guard reads the process-wide peak, which
            # earlier tests in a full run have already raised; a toy needs none.
            "max_vram_pct": 0}


def _offloaded(steps):
    """An offloaded model + its scheduler after ``steps`` real training steps
    (through train(), so the BCD/offload set-up is the trainer's own)."""
    import hvn.core.block_coordinate as bc
    got = {}
    orig = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *a, **k):
        orig(self, model, optimizer, *a, **k)
        got["sched"], got["opt"] = self, optimizer
    bc.BlockCoordinateScheduler.__init__ = init
    try:
        import tempfile
        tmp = tempfile.mkdtemp()
        m = _tiny_model()
        loader = DataLoader(_Windows(), batch_size=1,
                            sampler=tec.ResumableShuffleSampler(40, 123))
        tec.train(m, loader, _gpu_config(tmp, 100_000), torch.device("cuda"), tmp,
                  max_steps=steps)
    finally:
        bc.BlockCoordinateScheduler.__init__ = orig
    return m, got["sched"]


def _frozen_offloadable(m):
    pool, act = m._offload_pool, m._offload_active_names
    out = []
    for name, p in m.named_parameters():
        seg = ".".join(name.split(".")[:2]) if name.startswith("blocks.") else None
        if seg is not None and seg not in act and pool._offloadable([p]):
            out.append((name, p))
    return out


# --------------------------------------------------------------------------
# References: the #209 harness walk, and a GPU-resident copy
# --------------------------------------------------------------------------

@torch.no_grad()
def _harness_losses(model, loader, chunk):
    """scripts/ab_heldout_209.py heldout(), verbatim walk."""
    from hvn.edge.chunked_loss import chunked_cross_entropy as cce
    pool, act = model._offload_pool, model._offload_active_names
    was = model.training
    model.eval()

    def seg(name, mods, fn):
        params = [p for m in mods for p in m.parameters()]
        if name in act or not pool._offloadable(params):
            return fn()
        pool.load(name, params)
        out = fn()
        pool.evict(name)
        return out
    per, dev = [], torch.device("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        for x, y, _ in loader:
            x, y = x.to(dev), y.to(dev)
            pos = torch.arange(x.shape[1], device=dev).unsqueeze(0)
            z = seg("token_embedding", [model.token_embedding], lambda: model.token_embedding(x))
            for i, blk in enumerate(model.blocks):
                z = seg(f"blocks.{i}", [blk], lambda: blk(z, pos))
            tl = seg("final_norm+decoder", [model.final_norm, model.decoder],
                     lambda: cce(model.final_norm(z), y, model.decoder,
                                 chunk=chunk, use_checkpoint=False))
            per.append(tl.float().mean())
    if was:
        model.train()
    return torch.stack(per)


def _evaluate_per_window(model, loader, chunk, monkeypatch):
    """Run tec.evaluate and capture each window's mean loss as it computes it."""
    import hvn.edge.chunked_loss as CL
    orig = CL.chunked_cross_entropy
    seen = []

    def spy(*a, **k):
        out = orig(*a, **k)
        seen.append(out.float().mean())
        return out
    monkeypatch.setattr(CL, "chunked_cross_entropy", spy)
    res = tec.evaluate(model, loader, torch.device("cuda"), precision="bf16",
                       chunked_loss=chunk)
    monkeypatch.setattr(CL, "chunked_cross_entropy", orig)
    return res, torch.stack(seen)


def _resident_copy(model):
    """A plain (no offload) GPU model holding exactly the weights the offloaded
    model would stream: frozen ones at their parked dtype, read from their CPU
    homes; active ones as they are."""
    ref = _tiny_model()
    ref.load_state_dict({k: v.float() for k, v in model.state_dict().items()})
    ref = ref.cuda()
    src = dict(model.named_parameters())
    for name, p in ref.named_parameters():
        p.data = src[name].data.detach().to("cuda", copy=True)
    ref.eval()
    return ref


# --------------------------------------------------------------------------
# 1. It runs, and reports the harness's number
# --------------------------------------------------------------------------

@needs_cuda
def test_offloaded_eval_runs_and_matches_harness_and_resident_copy(monkeypatch):
    # 2*SPB+1 steps: blocks 0 and 1 have each trained and been re-parked, so a
    # stale CPU home (the #117 failure) would show up as a mismatch below.
    m, sched = _offloaded(2 * SPB + 1)
    frozen = _frozen_offloadable(m)
    assert frozen, "no CPU-parked frozen weight: the test would not exercise offload"
    assert all(p.device.type == "cpu" for _, p in frozen)

    loader = _val_loader()
    (overall, _), mine = _evaluate_per_window(m, loader, 16, monkeypatch)
    harness = _harness_losses(m, loader, 16)
    ref = _resident_copy(m)
    with torch.no_grad():
        (r_overall, _), resident = _evaluate_per_window(ref, loader, 16, monkeypatch)

    assert len(mine) == len(harness) == len(loader)
    assert torch.equal(mine, harness), (mine - harness).abs().max()
    assert torch.equal(mine, resident), (mine - resident).abs().max()
    assert overall == pytest.approx(float(harness.double().mean()), rel=1e-6)
    assert torch.isfinite(mine).all()


@needs_cuda
def test_resident_copy_check_can_fail(monkeypatch):
    """Partner: perturb one frozen weight in the resident copy only, and the
    bit-equality used above must break -- else it could not see stale weights."""
    m, _ = _offloaded(2 * SPB + 1)
    loader = _val_loader()
    _, mine = _evaluate_per_window(m, loader, 16, monkeypatch)
    ref = _resident_copy(m)
    name = _frozen_offloadable(m)[0][0]
    with torch.no_grad():
        dict(ref.named_parameters())[name].mul_(1.01)
    _, resident = _evaluate_per_window(ref, loader, 16, monkeypatch)
    assert not torch.equal(mine, resident)


@needs_cuda
def test_offloaded_forward_with_grad_on_refuses_clearly():
    m, _ = _offloaded(1)
    m.eval()
    x = torch.randint(0, VOCAB, (1, SEQ), device="cuda")
    with pytest.raises(RuntimeError, match="no_grad"):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            m(x)
    m.train()


# --------------------------------------------------------------------------
# 2. It leaves no trace on the model or the pool
# --------------------------------------------------------------------------

@needs_cuda
def test_eval_leaves_weights_parked_and_pool_empty():
    m, sched = _offloaded(SPB + 1)
    req = {n: p.requires_grad for n, p in m.named_parameters()}
    tec.evaluate(m, _val_loader(), torch.device("cuda"), precision="bf16", chunked_loss=16)
    assert m.training
    assert {n: p.requires_grad for n, p in m.named_parameters()} == req
    assert all(p.device.type == "cpu" for _, p in _frozen_offloadable(m))
    assert len(m._offload_pool._resident) == 0
    assert not m._offload_pool._pending_events
    # No GPU copy of a frozen ternary weight survives the eval (#131 cache).
    from hvn.edge.ternary import TernaryLinear
    for i, blk in enumerate(m.blocks):
        if f"blocks.{i}" in m._offload_active_names:
            continue
        for mod in blk.modules():
            if isinstance(mod, TernaryLinear):
                assert mod._wcache is None


# --------------------------------------------------------------------------
# 3. Training bits unchanged, through the real train() loop
# --------------------------------------------------------------------------

def _hashed_run(tmp, val_every, eval_hook=None, steps=None):
    """Real train() on the tiny offloaded model, dropout on. Returns one
    (block, lr, loss_sha, grad_sha, rng_sha, grad_nonzero) row per step."""
    from torch.optim.optimizer import (register_optimizer_step_post_hook,
                                       register_optimizer_step_pre_hook)
    import hvn.core.block_coordinate as bc
    import hvn.edge.chunked_loss as CL
    S = {"loss": hashlib.sha256(), "grad": None, "nz": 0}
    rows, sched = [], []
    orig_init = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *a, **k):
        orig_init(self, model, optimizer, *a, **k)
        sched.append(self)
    orig_cce = CL.chunked_cross_entropy

    def cce(*a, **k):
        out = orig_cce(*a, **k)
        if torch.is_grad_enabled():
            S["loss"].update(out.detach().float().cpu().numpy().tobytes())
        return out

    def pre(opt, args, kw):
        h, nz = hashlib.sha256(), 0
        for g in opt.param_groups:
            for p in g["params"]:
                if p.grad is not None:
                    h.update(p.grad.detach().float().cpu().numpy().tobytes())
                    nz += int((p.grad != 0).sum())
        S["grad"], S["nz"] = h.hexdigest(), nz

    def post(opt, args, kw):
        r = hashlib.sha256()
        r.update(torch.get_rng_state().numpy().tobytes())
        r.update(torch.cuda.get_rng_state().numpy().tobytes())
        rows.append((sched[0].active_block_name, opt.param_groups[0]["lr"],
                     S["loss"].hexdigest(), S["grad"], r.hexdigest(), S["nz"]))
        S["loss"] = hashlib.sha256()

    orig_eval = tec.evaluate

    def ev(*a, **k):
        out = orig_eval(*a, **k)
        if eval_hook is not None:
            eval_hook(sched[0])
        return out

    bc.BlockCoordinateScheduler.__init__ = init
    CL.chunked_cross_entropy = cce
    tec.evaluate = ev
    h1 = register_optimizer_step_pre_hook(pre)
    h2 = register_optimizer_step_post_hook(post)
    try:
        torch.manual_seed(0)
        m = _tiny_model(dropout=0.1)
        torch.manual_seed(1234)
        loader = DataLoader(_Windows(), batch_size=1,
                            sampler=tec.ResumableShuffleSampler(40, 123))
        n_blocks_steps = steps or 0
        tec.train(m, loader, _gpu_config(tmp, val_every), torch.device("cuda"), tmp,
                  max_steps=n_blocks_steps, val_loader=_val_loader())
    finally:
        h1.remove(); h2.remove()
        bc.BlockCoordinateScheduler.__init__ = orig_init
        CL.chunked_cross_entropy = orig_cce
        tec.evaluate = orig_eval
    return rows, sched[0].num_blocks


def _n_steps():
    m = _tiny_model()
    from hvn.core.block_coordinate import BlockCoordinateScheduler
    from hvn.optimizers import build_optimizer
    m = m.cuda()
    opt, _ = build_optimizer(m, "muon", lr=3e-4, weight_decay=0.0, muon_lr=0.02)
    s = BlockCoordinateScheduler(m, opt, steps_per_block=SPB, offload_inactive_weights=True,
                                 device=torch.device("cuda"),
                                 offload_park_dtype=torch.bfloat16, partition="layer")
    return s.num_blocks * SPB


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if not CUDA:
        pytest.skip("offload needs a CUDA device")
    steps = _n_steps()
    out = {}
    for tag, every, hook in (
            ("off", 100_000, None),
            ("off2", 100_000, None),
            ("on", 1, None),
    ):
        out[tag] = _hashed_run(str(tmp_path_factory.mktemp(tag)), every, hook, steps)
    return out


@needs_cuda
def test_runs_are_not_vacuous(runs):
    rows, nb = runs["off"]
    assert len(rows) == nb * SPB
    blocks = [r[0] for r in rows]
    assert len(set(blocks)) == nb, "not every BCD block position was visited"
    assert all(blocks.count(b) == SPB for b in set(blocks))
    assert all(r[1] > 0 for r in rows), "zero LR: weights would not move"
    assert all(r[5] > 0 for r in rows), "all-zero gradient at some step"


@needs_cuda
def test_determinism_control(runs):
    """Two eval-off runs agree, else the identity test below means nothing."""
    assert runs["off"][0] == runs["off2"][0]


@needs_cuda
def test_eval_every_step_leaves_training_bits_unchanged(runs):
    off, on = runs["off"][0], runs["on"][0]
    for i, (a, b) in enumerate(zip(off, on)):
        assert a == b, f"step {i + 1} ({a[0]}): eval changed training"


@needs_cuda
def test_hash_catches_eval_that_moves_the_rng(tmp_path, monkeypatch):
    """Falsifier: the same eval without the RNG restore (origin/main had none)
    must flip the per-step RNG hash."""
    monkeypatch.setattr(torch.random, "fork_rng",
                        lambda *a, **k: contextlib.nullcontext())
    rows, _ = _hashed_run(str(tmp_path), 1, None, 2 * SPB)
    off = _hashed_run(str(tmp_path / "ref"), 100_000, None, 2 * SPB)[0]
    assert [r[4] for r in rows] != [r[4] for r in off]


@needs_cuda
def test_hash_catches_eval_that_touches_optimizer_state(tmp_path):
    """Falsifier: an eval that nudges the active block's momentum must flip the
    loss/gradient hashes of a later step."""
    def nudge(sched):
        st = sched.optimizer.state
        for p, s in st.items():
            for v in s.values():
                if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 1:
                    v.view(-1)[0] += 1e-3
                    return
    rows, _ = _hashed_run(str(tmp_path), 1, nudge, 2 * SPB)
    off = _hashed_run(str(tmp_path / "ref"), 100_000, None, 2 * SPB)[0]
    assert [r[2:4] for r in rows] != [r[2:4] for r in off]


@needs_cuda
def test_cache_emptied_after_eval_only_at_a_switch(tmp_path, monkeypatch):
    """Memory placement (measured at 1B, see docs/state.md #213): an eval right
    after a BCD switch is followed by empty_cache (the switch had just emptied
    it, so this restores the no-eval state); a mid-turn eval is not (emptying
    there made the next step peak 97%). Records the order of evals, switches
    and empty_cache calls through the real train() loop."""
    import hvn.core.block_coordinate as bc
    events = []
    o_empty, o_eval, o_step = torch.cuda.empty_cache, tec.evaluate, bc.BlockCoordinateScheduler.step

    def empty():
        events.append("empty")
        o_empty()

    def ev(*a, **k):
        events.append("eval")
        return o_eval(*a, **k)

    def step(self):
        s = o_step(self)
        events.append("switch" if s else "nostep")
        return s
    monkeypatch.setattr(torch.cuda, "empty_cache", empty)
    monkeypatch.setattr(tec, "evaluate", ev)
    monkeypatch.setattr(bc.BlockCoordinateScheduler, "step", step)
    m = _tiny_model()
    loader = DataLoader(_Windows(), batch_size=1, sampler=tec.ResumableShuffleSampler(40, 123))
    tec.train(m, loader, _gpu_config(str(tmp_path), 1), torch.device("cuda"), str(tmp_path),
              max_steps=3 * SPB, val_loader=_val_loader())
    # Per step: switch|nostep, [empty from the switch], eval, [empty after eval].
    seq = [e for e in events if e in ("switch", "nostep", "eval", "empty")]
    after = {}
    i, n = 0, 0
    while i < len(seq):
        if seq[i] in ("switch", "nostep"):
            kind = seq[i]
            j = i + 1
            if kind == "switch":
                assert seq[j] == "empty", seq[i:i + 4]   # #211/#212 switch empty
                j += 1
            assert seq[j] == "eval", seq[i:i + 4]
            after[n] = (kind, j + 1 < len(seq) and seq[j + 1] == "empty")
            n += 1
            i = j + 1
        else:
            i += 1
    assert n == 3 * SPB
    assert any(k == "switch" for k, _ in after.values())
    assert any(k == "nostep" for k, _ in after.values())
    for step_i, (kind, emptied) in after.items():
        assert emptied == (kind == "switch"), (step_i, kind, emptied)


# --------------------------------------------------------------------------
# 4. CPU, real train(): the RNG half, which fails on main
# --------------------------------------------------------------------------

CPU_VOCAB, CPU_SEQ = 64, 8


class _DropLM(nn.Module):
    def __init__(self, kill_after=None):
        super().__init__()
        self.vocab_size, self.d_model, self.num_layers = CPU_VOCAB, 16, 1
        self.emb = nn.Embedding(CPU_VOCAB, 16)
        self.drop = nn.Dropout(0.3)
        self.out = nn.Linear(16, CPU_VOCAB)
        self.kill_after, self.calls = kill_after, 0

    def forward(self, x):
        if self.training and self.kill_after is not None and self.calls >= self.kill_after:
            raise _Killed
        if self.training:
            self.calls += 1
        return self.out(self.drop(self.emb(x)))


class _Killed(Exception):
    pass


class _CpuWindows(Dataset):
    def __init__(self, n=40, seed=7):
        g = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, CPU_VOCAB, (n, CPU_SEQ + 1), generator=g)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i, :-1], self.data[i, 1:], 0


def _cpu_run(tmp, model, val_every, resume_step=0, resume_state=None, total=12):
    os.makedirs(tmp, exist_ok=True)
    ds = _CpuWindows()
    loader = DataLoader(ds, batch_size=2, drop_last=True,
                        sampler=tec.ResumableShuffleSampler(len(ds), 123))
    val = DataLoader(_CpuWindows(6, seed=3), batch_size=2, shuffle=False)
    cfg = {"lr": 1e-2, "weight_decay": 0.0, "warmup_steps": 0, "grad_accum": 2,
           "log_file": os.path.join(tmp, "log.csv"), "save_every_steps": 5,
           "log_every_steps": 10_000, "batch_size": 2, "seq_len": CPU_SEQ,
           "optimizer": "adamw", "precision": "fp32", "val_every_steps": val_every}
    try:
        tec.train(model, loader, cfg, torch.device("cpu"), tmp, max_steps=total,
                  resume_step=resume_step, resume_state=resume_state, val_loader=val)
    except _Killed:
        pass
    return model


def test_cpu_eval_every_step_leaves_weights_unchanged(tmp_path):
    """Fails on origin/main: iter(val_loader) drew from the CPU RNG, so every
    eval shifted the dropout masks of the steps after it."""
    torch.manual_seed(0)
    off = _cpu_run(str(tmp_path / "off"), _DropLM(), 100_000)
    torch.manual_seed(0)
    on = _cpu_run(str(tmp_path / "on"), _DropLM(), 1)
    rows = open(tmp_path / "on" / "val_log.csv").read().strip().splitlines()
    assert len(rows) == 1 + 12, "the eval did not run every step"
    for (n, a), (_, b) in zip(off.state_dict().items(), on.state_dict().items()):
        assert torch.equal(a, b), n


def test_cpu_check_can_fail(tmp_path, monkeypatch):
    """Partner: drop the RNG restore and the same comparison sees a difference."""
    monkeypatch.setattr(torch.random, "fork_rng",
                        lambda *a, **k: contextlib.nullcontext())
    torch.manual_seed(0)
    off = _cpu_run(str(tmp_path / "off"), _DropLM(), 100_000)
    torch.manual_seed(0)
    on = _cpu_run(str(tmp_path / "on"), _DropLM(), 1)
    assert not torch.equal(off.out.weight, on.out.weight)


def test_cpu_resume_with_eval_on_matches_uninterrupted(tmp_path):
    """(e) checkpoint/resume with eval on: kill inside step 8, resume from the
    step-5 checkpoint, end bit-identical to the uninterrupted eval-on run."""
    torch.manual_seed(0)
    ref = _cpu_run(str(tmp_path / "ref"), _DropLM(), 1)
    torch.manual_seed(0)
    ck = str(tmp_path / "killed")
    _cpu_run(ck, _DropLM(kill_after=2 * 7 + 1), 1)
    state = tec._load_resume_state(ck, 5)
    assert state is not None
    ckpt = torch.load(os.path.join(ck, "edge_conversational_step5.pt"), weights_only=False)
    torch.manual_seed(999)
    torch.rand(1000)
    resumed = _DropLM()
    resumed.load_state_dict(ckpt["model_state_dict"])
    _cpu_run(ck, resumed, 1, resume_step=5, resume_state=state)
    assert resumed.calls == (12 - 5) * 2
    for (n, a), (_, b) in zip(ref.state_dict().items(), resumed.state_dict().items()):
        assert torch.equal(a, b), n


def test_defaults_unchanged():
    """--val-every keeps its default (None -> the checkpoint interval, 500);
    --val-batches defaults to the old fixed cap of 100 batches."""
    import inspect
    a = tec.build_arg_parser().parse_args([])
    assert a.val_every is None and a.val_batches == 100 and a.val_fraction == 0.0
    assert a.save_every_steps is None
    assert 'config.get("val_every_steps", config["save_every_steps"])' in inspect.getsource(tec.train)
