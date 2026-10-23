"""The fused ternary quantizer, as ``--fast`` wires it, must not move training.

Author: Rambarun Komaljeet

``tests/test_ternary_fused.py`` gates the kernel on single tensors. This gates
the WIRING: a real hybrid EdgeHVN (DeltaNet + attention, tied embeddings, grad
checkpoint, Triton scan + megakernel, bf16 autocast) trained for several Muon
steps with ``apply_fast_stack(fused_ternary=True)`` against the same model on the
eager quantizer. Every step's loss and the final weights must be bit-identical.
A sequence, not one step, because a quantizer difference that only shows once
the shadow weights move would pass a one-step check.

``test_fused_path_is_actually_taken`` exists because the comparison above passes
trivially if the switch is silently ignored (eager vs eager).
"""

import copy

import pytest
import torch
import torch.nn as nn

# Imported at module load, BEFORE any CUDA work: in this venv, importing pandas
# (pulled in by the trainer) after CUDA is initialised kills the process with a
# Windows heap-corruption fault (0xc0000374). Reproduces in three lines with
# no repo code: torch.zeros(1, device="cuda"); import pandas.
from train_edge_conversational import apply_fast_stack

cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")

V, D, L, H, S = 512, 128, 4, 2, 64


def _model():
    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model
    torch.manual_seed(0)
    plan = ["transformer" if i == 2 else "deltanet" for i in range(L)]
    m = EdgeHVN(vocab_size=V, d_model=D, num_layers=L, n_heads=H, block_plan=plan,
                attn_window=S, tie_embeddings=True, grad_checkpoint=True,
                dropout=0.0).cuda()
    lin = [n for n, x in m.named_modules() if isinstance(x, nn.Linear)]
    ex = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(m, target_names=[n for n in lin if not any(p in n for p in ex)])
    return m


def _train(m, steps=4, accum=2):
    from hvn.optimizers import build_optimizer
    opt, _ = build_optimizer(m, "muon", lr=3e-4, weight_decay=0.01, muon_lr=0.02)
    g = torch.Generator().manual_seed(1)
    losses = []
    for _ in range(steps):
        tot = 0.0
        for _ in range(accum):
            x = torch.randint(0, V, (2, S), generator=g).cuda()
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = m(x)
                loss = nn.functional.cross_entropy(
                    logits.reshape(-1, V).float(), torch.roll(x, -1, 1).reshape(-1)) / accum
            loss.backward()
            tot += loss.item()
        opt.step()
        opt.zero_grad(set_to_none=True)
        losses.append(tot)
    return losses


@pytest.fixture(autouse=True)
def _restore_tf32():
    # apply_fast_stack(tf32=True) flips TF32 process-wide. Put it back so later
    # tests in the same pytest run do not inherit it (#170: it broke
    # test_bcd_balanced_offload[True-1]/[True-3] when run after this file).
    saved = (torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32)
    yield
    torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32 = saved


def _fast(m, fused):
    return apply_fast_stack(m, triton_scan=True, megakernel_scan=True, tf32=True,
                            compile_model=False, verbose=False, fused_ternary=fused)


@cuda_only
def test_fused_path_is_actually_taken(monkeypatch):
    import hvn.edge.ternary_fused as tf
    calls = {"n": 0}
    real = tf.ternary_quantize_ste_fused

    def counting(w, eps=1e-6):
        calls["n"] += 1
        return real(w, eps)

    monkeypatch.setattr(tf, "ternary_quantize_ste_fused", counting)
    m = _fast(_model(), fused=True)
    _train(m, steps=1, accum=1)
    assert calls["n"] > 0, "fused_ternary=True never reached the fused kernel"

    calls["n"] = 0
    m = _fast(_model(), fused=False)
    _train(m, steps=1, accum=1)
    assert calls["n"] == 0, "fused_ternary=False still used the fused kernel"


@cuda_only
def test_fused_training_trajectory_is_bit_identical():
    base = _model()
    eager = _fast(copy.deepcopy(base), fused=False)
    fused = _fast(copy.deepcopy(base), fused=True)

    le = _train(eager)
    lf = _train(fused)
    assert le == lf, f"loss trajectory moved: eager {le} vs fused {lf}"
    for (n, pe), (_, pf) in zip(eager.named_parameters(), fused.named_parameters()):
        assert torch.equal(pe, pf), f"{n} diverged after {len(le)} steps"
    # Non-vacuity: the weights must actually have moved, or "identical after
    # training" says nothing.
    moved = any(not torch.equal(p0, p1) for p0, p1 in
                zip(base.parameters(), eager.parameters()))
    assert moved, "training did not change any weight -- the comparison is vacuous"
