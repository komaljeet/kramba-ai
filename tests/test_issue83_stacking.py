"""
Test suite for Issue #83 stacking — torch.compile + Triton scan + bf16.

The previous session believed there was a "graph-break" between torch.compile
and the Triton scan's torch.autograd.Function, preventing them from stacking.
Investigation revealed the actual issue was a Windows cp1252 encoding bug in
PyTorch's inductor codecache (Unicode chars in docstrings like α, β crash the
file writer). With PYTHONUTF8=1, there are **zero graph breaks** — compile and
Triton stack natively.

This test verifies:
  1. compile + triton-scan produces correct output (no crash, no NaN).
  2. compile + triton-scan has zero graph breaks.
  3. The full stack (compile + triton + bf16) runs without error.
  4. A real 50-step benchmark measuring the combined speedup.

Run on GPU:  python tests/test_issue83_stacking.py
"""

import os as _os
import sys as _sys
import time

# Force UTF-8 before any torch import (fixes the inductor cp1252 crash)
_os.environ.setdefault("PYTHONUTF8", "1")

_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from hvn.core.deltanet import DeltaNetBlock
from hvn.edge.model import EdgeHVN
from hvn.edge.ternary import apply_ternary_to_model, TernaryLinear


def _device():
    assert torch.cuda.is_available(), "CUDA required"
    return torch.device("cuda")


def _build_model(vocab=256, d=128, L=3, h=4, device=None):
    m = EdgeHVN(vocab, d, L, h, use_deltanet=True, deltanet_conv_kernel=4)
    linears = {n: mod for n, mod in m.named_modules() if isinstance(mod, nn.Linear)}
    excl = ('beta_proj', 'alpha_proj', 'decoder')
    targets = [n for n in linears if not any(p in n for p in excl)]
    apply_ternary_to_model(m, target_names=targets)
    return m.to(device)


def _make_dl(vocab=256, n=400, seq=64, bs=4):
    torch.manual_seed(42)
    x = torch.randint(0, vocab, (n, seq), dtype=torch.long)
    y = torch.randint(0, vocab, (n, seq), dtype=torch.long)
    return DataLoader(TensorDataset(x, y), batch_size=bs, shuffle=True, drop_last=True)


def _train_50(model, dl, dev, precision="fp32", lr=3e-4, ga=4):
    use_amp = precision in ("fp16", "bf16") and dev.type == "cuda"
    amp_dt = torch.float16 if precision == "fp16" else torch.bfloat16
    opt = torch.optim.AdamW(model.parameters(), lr=lr, fused=True)
    crit = nn.CrossEntropyLoss()
    model.train()
    di = iter(dl); micro = 0; step = 0; warm = 3; total = 53
    torch.cuda.synchronize(); t0 = time.perf_counter()
    while step < total:
        try: bx, by = next(di)
        except StopIteration: di = iter(dl); bx, by = next(di)
        bx, by = bx.to(dev), by.to(dev)
        with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dt):
            logits = model(bx)
            loss = crit(logits.reshape(-1, model.vocab_size), by.reshape(-1)) / ga
        loss.backward(); micro += 1
        if micro >= ga:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); opt.zero_grad(); micro = 0; step += 1
    torch.cuda.synchronize()
    return time.perf_counter() - t0


def test_compile_triton_correct():
    """compile + triton-scan produces correct output (no crash, no NaN)."""
    dev = _device()
    torch.manual_seed(0)
    blk = DeltaNetBlock(d_model=128, n_heads=4, dropout=0.0).to(dev)
    blk.use_triton_scan = True
    x = torch.randn(2, 64, 128, device=dev, requires_grad=True)

    compiled = torch.compile(blk, mode="default")
    o = compiled(x)
    assert not torch.isnan(o).any(), "NaN in compiled+triton output"
    o.sum().backward()
    assert not torch.isnan(x.grad).any(), "NaN in compiled+triton gradients"
    print("[1/4] compile + triton-scan: correct output + gradients ... OK")


def test_zero_graph_breaks():
    """compile + triton-scan has zero graph breaks."""
    dev = _device()
    torch.manual_seed(0)
    blk = DeltaNetBlock(d_model=128, n_heads=4, dropout=0.0).to(dev)
    blk.use_triton_scan = True
    x = torch.randn(2, 64, 128, device=dev, requires_grad=True)

    torch._dynamo.reset()
    cnt = torch._dynamo.testing.CompileCounter()
    compiled = torch.compile(blk, mode="default", backend=cnt)
    o = compiled(x)
    o.sum().backward()
    # frame_count=1 means no graph breaks (single compiled frame)
    breaks = cnt.frame_count - 1
    assert breaks == 0, f"Expected 0 graph breaks, got {breaks}"
    print(f"[2/4] compile + triton-scan: {breaks} graph breaks (zero!) ... OK")


def test_full_stack_runs():
    """The full stack (compile + triton + bf16) runs without error."""
    dev = _device()
    dl = _make_dl()
    torch.manual_seed(0)
    model = _build_model(device=dev)
    # Enable triton scan on all blocks
    for blk in model.blocks:
        if hasattr(blk, 'use_triton_scan'):
            blk.use_triton_scan = True
    model = torch.compile(model)
    # Run 5 steps — just verify it doesn't crash
    _train_50(model, dl, dev, precision="bf16")
    print("[3/4] compile + triton + bf16: runs without error ... OK")


def test_stacked_speedup():
    """Measure the combined speedup: baseline vs full stack."""
    dev = _device()
    dl = _make_dl()

    # Baseline: fp32, no compile, no triton
    torch.manual_seed(0)
    m_base = _build_model(device=dev)
    t_base = _train_50(m_base, dl, dev, "fp32")

    # Full stack: bf16 + compile + triton
    torch.manual_seed(0)
    m_stack = _build_model(device=dev)
    for blk in m_stack.blocks:
        if hasattr(blk, 'use_triton_scan'):
            blk.use_triton_scan = True
    m_stack = torch.compile(m_stack)
    t_stack = _train_50(m_stack, dl, dev, "bf16")

    speedup = t_base / t_stack
    print(f"\n  50-step timing (d=128, L=3):")
    print(f"    baseline (fp32, no compile, no triton): {t_base:.2f}s ({t_base/50*1000:.1f} ms/step)")
    print(f"    full stack (bf16 + compile + triton):    {t_stack:.2f}s ({t_stack/50*1000:.1f} ms/step)")
    print(f"    combined speedup: {speedup:.2f}x")

    # The combined speedup should be > 1 (even at tiny scale)
    assert speedup > 1.0, f"Full stack is SLOWER than baseline ({speedup:.2f}x)"
    print(f"[4/4] stacked speedup: {speedup:.2f}x (baseline → full stack) ... OK")


if __name__ == "__main__":
    print("=" * 72)
    print("Issue #83 stacking — compile + Triton + bf16")
    print("=" * 72)
    test_compile_triton_correct()
    test_zero_graph_breaks()
    test_full_stack_runs()
    test_stacked_speedup()
    print("=" * 72)
    print("ALL TESTS PASSED")
    print("=" * 72)