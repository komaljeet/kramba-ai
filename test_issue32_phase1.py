"""
Test script for Extreme Weight Compression — Phase 1 (Issue #32).

Verifies the four Phase 1 foundational compression techniques:
  1. GPTQQuantizer — Hessian-based 4-bit quantization with error compensation
  2. HadamardQuantizer — Hadamard pre-rotation for 2-bit quantization (QuIP#)
  3. RateDistortionAllocator — per-layer sensitivity-based bit allocation
  4. BlockSparseLinear — 2:4 structured sparsity

Includes an end-to-end check: train a tiny HVN model, measure baseline
perplexity, GPTQ-quantize every nn.Linear layer to 4-bit, and verify the
perplexity degradation stays under the issue's 5% acceptance criterion.

All tests run on GPU (CUDA required).
"""

import math
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, ".")

from hvn.compress import (
    GPTQQuantizer,
    HadamardQuantizer,
    RateDistortionAllocator,
    collect_calibration_hessian,
    compress_linear_layers_gptq,
)
from hvn.sparse import BlockSparseLinear, apply_2_4_sparsity, convert_model_to_block_sparse
from hvn.core import HamiltonianVortexNetwork
from train import CharTokenizer, TextDataset, SAMPLE_TEXT, train_model

DEVICE = torch.device("cuda")


def relative_error(orig: torch.Tensor, recon: torch.Tensor) -> float:
    return (orig.float() - recon.float()).abs().mean().item() / orig.float().abs().mean().clamp(min=1e-8).item()


def test_gptq_quantizer():
    print("\n" + "=" * 60)
    print("TEST 1: GPTQQuantizer (Hessian-based 4-bit + error compensation)")
    print("=" * 60)

    torch.manual_seed(0)
    out_features, in_features = 256, 256
    W = torch.randn(out_features, in_features, device=DEVICE) * 0.05
    # Simulate calibration activations with a few high-variance ("hot") features and
    # strong feature correlation — this is exactly the regime where independent
    # round-to-nearest quantization wastes precision but GPTQ's error compensation
    # (which optimizes *output* distortion, not raw weight error) pays off.
    X = torch.randn(2000, in_features, device=DEVICE)
    X = X + 0.5 * torch.roll(X, shifts=1, dims=1)  # correlate neighboring features
    X[:, :8] *= 8.0  # a handful of high-activation ("hot") features
    H = collect_calibration_hessian(X)

    quantizer = GPTQQuantizer(bits=4, group_size=64, percdamp=0.01)
    qdict = quantizer.quantize(W, H)
    W_deq = quantizer.dequantize(qdict)

    def output_distortion(W_recon: torch.Tensor) -> float:
        # (W - Ŵ) H (W - Ŵ)ᵀ — the quantity GPTQ actually minimizes: how much the
        # quantization error perturbs this layer's output on calibration-like inputs.
        err = (W - W_recon).float()
        return torch.einsum("oi,ij,oj->o", err, H, err).mean().item()

    err_gptq = relative_error(W, W_deq)
    dist_gptq = output_distortion(W_deq)
    print(f"  GPTQ 4-bit relative weight error: {err_gptq:.4%}")
    print(f"  GPTQ 4-bit output distortion (Hessian-weighted): {dist_gptq:.6e}")

    # Compare against naive round-to-nearest (no error compensation) at the same bit-width
    qmax = 2 ** 4 - 1
    wmin = W.min(dim=1, keepdim=True).values
    wmax = W.max(dim=1, keepdim=True).values
    scale = (wmax - wmin).clamp(min=1e-8) / qmax
    zero = torch.round(-wmin / scale)
    q_naive = torch.clamp(torch.round(W / scale + zero), 0, qmax)
    W_naive = (q_naive - zero) * scale
    err_naive = relative_error(W, W_naive)
    dist_naive = output_distortion(W_naive)
    print(f"  Naive round-to-nearest 4-bit relative weight error: {err_naive:.4%}")
    print(f"  Naive round-to-nearest 4-bit output distortion (Hessian-weighted): {dist_naive:.6e}")
    print(f"  GPTQ reduces output distortion by {(1 - dist_gptq / dist_naive) * 100:.1f}% vs naive")

    memory = GPTQQuantizer.memory_bytes(qdict)
    fp32_bytes = W.numel() * 4
    print(f"  Memory: fp32={fp32_bytes} bytes, GPTQ 4-bit={memory} bytes "
          f"({fp32_bytes / memory:.2f}x compression)")

    # GPTQ optimizes output distortion (its actual objective), not raw weight error —
    # 4-bit round-to-nearest has an inherent ~10% weight-noise floor regardless of
    # error compensation, so we assert on the metric GPTQ is designed to minimize.
    assert dist_gptq < dist_naive, "GPTQ error compensation should reduce output distortion vs naive RTN"
    assert memory < fp32_bytes, "Quantized representation should use less memory than fp32"
    print("  ✓ PASSED")


def test_hadamard_quantizer():
    print("\n" + "=" * 60)
    print("TEST 2: HadamardQuantizer (2-bit, QuIP#-style incoherence processing)")
    print("=" * 60)

    torch.manual_seed(1)
    out_features, in_features = 128, 192  # non-power-of-2 to test padding
    W = torch.randn(out_features, in_features, device=DEVICE) * 0.05
    # Inject a few large outliers (the scenario Hadamard rotation is meant to fix)
    W[:, 0] *= 20.0
    W[:, 5] *= 15.0

    quantizer = HadamardQuantizer(bits=2, group_size=64, seed=42)
    qdict = quantizer.quantize(W)
    W_deq = quantizer.dequantize(qdict)
    assert W_deq.shape == W.shape

    err_hadamard = relative_error(W, W_deq)
    print(f"  Hadamard 2-bit relative error: {err_hadamard:.4%}")

    # Compare against naive per-tensor 2-bit quantization directly on W (no rotation)
    qmax = 2 ** 2 - 1
    wmin = W.min(dim=1, keepdim=True).values
    wmax = W.max(dim=1, keepdim=True).values
    scale = (wmax - wmin).clamp(min=1e-8) / qmax
    zero = torch.round(-wmin / scale)
    q_naive = torch.clamp(torch.round(W / scale + zero), 0, qmax)
    W_naive = (q_naive - zero) * scale
    err_naive = relative_error(W, W_naive)
    print(f"  Naive (unrotated) 2-bit relative error:  {err_naive:.4%}")
    print(f"  Hadamard rotation reduces outlier-driven error by "
          f"{(1 - err_hadamard / err_naive) * 100:.1f}%" if err_naive > 0 else "")

    fp32_bytes = W.numel() * 4
    memory = HadamardQuantizer.memory_bytes(qdict)
    print(f"  Memory: fp32={fp32_bytes} bytes, Hadamard 2-bit={memory} bytes "
          f"({fp32_bytes / memory:.2f}x compression)")

    assert err_hadamard < err_naive, "Hadamard rotation should reduce outlier-driven quantization error"
    assert memory < fp32_bytes
    print("  ✓ PASSED")


def test_rate_distortion_allocator():
    print("\n" + "=" * 60)
    print("TEST 3: RateDistortionAllocator (sensitivity-based bit allocation)")
    print("=" * 60)

    torch.manual_seed(2)
    allocator = RateDistortionAllocator(min_bits=2, max_bits=8)

    # Simulate 4 layers with very different sensitivities
    sensitivities = {
        "layer_sensitive": 100.0,
        "layer_medium_a": 10.0,
        "layer_medium_b": 8.0,
        "layer_robust": 0.5,
    }
    allocation = allocator.allocate(sensitivities, avg_bits=4.0)
    print(f"  Sensitivities: {sensitivities}")
    print(f"  Allocation (avg_bits=4.0): {allocation}")

    avg_allocated = sum(allocation.values()) / len(allocation)
    print(f"  Average allocated bits: {avg_allocated:.2f}")

    assert allocation["layer_sensitive"] >= allocation["layer_medium_a"]
    assert allocation["layer_medium_a"] >= allocation["layer_robust"]
    assert all(allocator.min_bits <= b <= allocator.max_bits for b in allocation.values())
    assert abs(avg_allocated - 4.0) <= 1.0, "Average allocation should track the requested budget"

    # compute_sensitivity should rank a high-activation/high-energy layer above a quiet one
    W_hot = torch.randn(64, 64, device=DEVICE) * 1.0
    W_quiet = torch.randn(64, 64, device=DEVICE) * 0.01
    X_hot = torch.randn(500, 64, device=DEVICE) * 5.0
    X_quiet = torch.randn(500, 64, device=DEVICE) * 0.1
    H_hot = collect_calibration_hessian(X_hot)
    H_quiet = collect_calibration_hessian(X_quiet)
    s_hot = RateDistortionAllocator.compute_sensitivity(W_hot, H_hot)
    s_quiet = RateDistortionAllocator.compute_sensitivity(W_quiet, H_quiet)
    print(f"  compute_sensitivity: hot={s_hot:.4f}, quiet={s_quiet:.6f}")
    assert s_hot > s_quiet

    print("  ✓ PASSED")


def test_block_sparse_linear():
    print("\n" + "=" * 60)
    print("TEST 4: BlockSparseLinear (2:4 structured sparsity)")
    print("=" * 60)

    torch.manual_seed(3)
    dense = nn.Linear(128, 64, bias=True).to(DEVICE)
    sparse = BlockSparseLinear.from_linear(dense)

    sparsity = sparse.sparsity()
    print(f"  Achieved sparsity: {sparsity:.4%} (target: 50%)")
    assert abs(sparsity - 0.5) < 1e-6, f"2:4 sparsity should be exactly 50%, got {sparsity:.4%}"

    # Verify the 2:4 pattern: every group of 4 has exactly 2 nonzeros
    groups = sparse.mask.view(sparse.out_features, -1, 4)
    nnz_per_group = groups.sum(dim=-1)
    assert torch.all(nnz_per_group == 2), "Every group of 4 must have exactly 2 nonzeros"

    x = torch.randn(8, 128, device=DEVICE)
    y_sparse = sparse(x)
    y_dense = dense(x)
    assert y_sparse.shape == y_dense.shape == (8, 64)

    rel_diff = relative_error(y_dense, y_sparse)
    print(f"  Output shape: {tuple(y_sparse.shape)} (matches dense)")
    print(f"  Relative output difference vs dense (expected, from pruning): {rel_diff:.2%}")

    fp32_bytes = dense.weight.numel() * 4 + dense.bias.numel() * 4
    sparse_bytes = sparse.memory_bytes(value_bits=16)
    print(f"  Memory: dense fp32={fp32_bytes} bytes, 2:4 sparse (fp16 values)={sparse_bytes} bytes "
          f"({fp32_bytes / sparse_bytes:.2f}x compression)")
    assert sparse_bytes < fp32_bytes

    # apply_2_4_sparsity should be idempotent
    twice = apply_2_4_sparsity(apply_2_4_sparsity(dense.weight.data))
    once = apply_2_4_sparsity(dense.weight.data)
    assert torch.equal(twice, once)

    print("  ✓ PASSED")


def test_end_to_end_gptq_ppl():
    print("\n" + "=" * 60)
    print("TEST 5: End-to-end — train tiny HVN, GPTQ-4bit compress all Linear layers,")
    print("        verify perplexity degradation stays under the 5% acceptance criterion")
    print("=" * 60)

    torch.manual_seed(4)
    text = SAMPLE_TEXT * 20
    tokenizer = CharTokenizer(text, add_tool_tokens=False)
    token_ids = tokenizer.encode(text)

    seq_len = 64
    split = int(len(token_ids) * 0.9)
    train_ids, eval_ids = token_ids[:split], token_ids[split:]

    train_dataset = TextDataset(train_ids, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    model = HamiltonianVortexNetwork(
        vocab_size=tokenizer.vocab_size,
        d_model=64,
        num_layers=2,
        n_heads=4,
        bond_dimension=16,
        chunk_size=32,
        dropout=0.0,
    ).to(DEVICE)

    print(f"  Model: {sum(p.numel() for p in model.parameters()):,} params, "
          f"vocab={tokenizer.vocab_size}")
    print("  Training for 15 epochs...")
    t0 = time.time()
    train_model(model, train_loader, epochs=15, lr=3e-4, device=DEVICE,
                save_path="test_issue32_checkpoint.pt")
    print(f"  Training took {time.time() - t0:.1f}s")

    eval_dataset = TextDataset(eval_ids, seq_len)
    eval_loader = DataLoader(eval_dataset, batch_size=16, shuffle=False)
    criterion = nn.CrossEntropyLoss()

    def eval_ppl(m: nn.Module) -> float:
        m.eval()
        total_loss, n_batches = 0.0, 0
        with torch.no_grad():
            for x, y in eval_loader:
                x, y = x.to(DEVICE), y.to(DEVICE)
                logits = m(x)
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
                total_loss += loss.item()
                n_batches += 1
        return math.exp(total_loss / max(n_batches, 1))

    ppl_before = eval_ppl(model)
    print(f"  Perplexity before compression: {ppl_before:.4f}")

    # Calibration forward pass: run a handful of training batches through the model
    # under the hooks so GPTQ can build a Hessian for every nn.Linear layer.
    calibration_batches = [next(iter(train_loader)) for _ in range(4)]

    def calibration_forward():
        for cx, _cy in calibration_batches:
            model(cx.to(DEVICE))

    linear_layers_before = sum(1 for m in model.modules() if isinstance(m, nn.Linear))
    print(f"  Compressing {linear_layers_before} nn.Linear layers with GPTQ (4-bit, group_size=32)...")

    metadata = compress_linear_layers_gptq(
        model, calibration_forward, bits=4, group_size=32, percdamp=0.01,
    )

    total_orig = sum(m["orig_bytes"] for m in metadata.values())
    total_quant = sum(m["quant_bytes"] for m in metadata.values())
    max_err = max(m["max_abs_error"] for m in metadata.values())
    print(f"  Compressed {len(metadata)} layers: {total_orig} -> {total_quant} bytes "
          f"({total_orig / total_quant:.2f}x compression)")
    print(f"  Max abs weight reconstruction error across all layers: {max_err:.6f}")

    ppl_after = eval_ppl(model)
    degradation_pct = (ppl_after - ppl_before) / ppl_before * 100
    print(f"  Perplexity after GPTQ 4-bit compression: {ppl_after:.4f}")
    print(f"  Degradation: {degradation_pct:+.2f}% (acceptance criterion: < 5%)")

    assert total_quant < total_orig, "GPTQ compression should reduce weight storage"
    assert degradation_pct < 5.0, f"PPL degradation {degradation_pct:.2f}% exceeds the 5% acceptance criterion"
    print("  ✓ PASSED")


if __name__ == "__main__":
    assert torch.cuda.is_available(), "This test suite requires a CUDA GPU"
    print(f"Running on: {torch.cuda.get_device_name(0)}")

    test_gptq_quantizer()
    test_hadamard_quantizer()
    test_rate_distortion_allocator()
    test_block_sparse_linear()
    test_end_to_end_gptq_ppl()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED — Issue #32 Phase 1 (Foundational Compression)")
    print("=" * 60)
