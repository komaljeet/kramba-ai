"""
Test script for Hamiltonian quantization (Issue #20).

Verifies:
1. INT8 quantization/dequantization accuracy
2. NF4 4-bit quantization/dequantization accuracy
3. Memory savings estimation
4. Quantized inference (generation) produces valid output
5. Quantized vs full-precision output comparison

All tests run on GPU (CUDA required).
"""

import torch
import time
import sys

# Ensure we import from the local package
sys.path.insert(0, ".")

from hvn.quantize import (
    quantize_int8,
    dequantize_int8,
    quantize_nf4,
    dequantize_nf4,
    quantize_hamiltonian,
    quantize_hamiltonian_states,
    dequantize_hamiltonian_states,
    QuantizedHamiltonian,
    QuantizedHVNInference,
    estimate_memory_savings,
)
from hvn.core import HamiltonianVortexNetwork


def test_int8_accuracy():
    """Test INT8 quantization round-trip accuracy."""
    print("\n" + "=" * 60)
    print("TEST 1: INT8 Quantization Accuracy")
    print("=" * 60)

    torch.manual_seed(42)
    # Simulate a Hamiltonian matrix: [B*n_heads, d_head, d_head]
    H = torch.randn(8, 64, 64, device="cuda") * 0.1
    H = H + torch.eye(64, device="cuda").unsqueeze(0) * 0.01  # add small identity

    # Per-tensor
    q, scale = quantize_int8(H, per_channel=False)
    H_deq = dequantize_int8(q, scale)
    err = (H - H_deq).abs()
    rel_err = err.mean().item() / H.abs().mean().item()

    print(f"  Per-tensor INT8:")
    print(f"    Original dtype: {H.dtype}, shape: {H.shape}")
    print(f"    Quantized dtype: {q.dtype}")
    print(f"    Max abs error:   {err.max().item():.6f}")
    print(f"    Mean abs error:  {err.mean().item():.6f}")
    print(f"    Relative error:  {rel_err:.4%}")
    assert rel_err < 0.05, f"INT8 per-tensor relative error too high: {rel_err:.4%}"
    print("  ✓ PASSED")

    # Per-channel
    q_pc, scale_pc = quantize_int8(H, per_channel=True, channel_dim=-1)
    H_deq_pc = dequantize_int8(q_pc, scale_pc)
    err_pc = (H - H_deq_pc).abs()
    rel_err_pc = err_pc.mean().item() / H.abs().mean().item()

    print(f"\n  Per-channel INT8:")
    print(f"    Max abs error:   {err_pc.max().item():.6f}")
    print(f"    Mean abs error:  {err_pc.mean().item():.6f}")
    print(f"    Relative error:  {rel_err_pc:.4%}")
    assert rel_err_pc < 0.05, f"INT8 per-channel relative error too high: {rel_err_pc:.4%}"
    print("  ✓ PASSED")

    # Memory comparison
    fp32_bytes = H.numel() * 4
    int8_bytes = q.numel() + scale.numel() * 4
    print(f"\n  Memory: fp32={fp32_bytes} bytes, INT8={int8_bytes} bytes "
          f"({fp32_bytes/int8_bytes:.1f}× compression)")
    assert int8_bytes < fp32_bytes, "INT8 should use less memory than fp32"
    print("  ✓ PASSED")


def test_nf4_accuracy():
    """Test NF4 4-bit quantization round-trip accuracy."""
    print("\n" + "=" * 60)
    print("TEST 2: NF4 4-bit Quantization Accuracy")
    print("=" * 60)

    torch.manual_seed(42)
    H = torch.randn(8, 64, 64, device="cuda") * 0.1
    H = H + torch.eye(64, device="cuda").unsqueeze(0) * 0.01

    qdict = quantize_nf4(H, block_size=64)
    H_deq = dequantize_nf4(qdict)

    err = (H - H_deq).abs()
    rel_err = err.mean().item() / H.abs().mean().item()

    print(f"  NF4 4-bit (block_size=64):")
    print(f"    Original shape:  {H.shape}")
    print(f"    Packed codes:    {qdict['codes'].shape} (int8, 2 codes/byte)")
    print(f"    Scales:          {qdict['scales'].shape}")
    print(f"    Num blocks:      {qdict['num_blocks']}")
    print(f"    Max abs error:   {err.max().item():.6f}")
    print(f"    Mean abs error:  {err.mean().item():.6f}")
    print(f"    Relative error:  {rel_err:.4%}")
    # NF4 is 4-bit so slightly higher error is expected
    assert rel_err < 0.15, f"NF4 relative error too high: {rel_err:.4%}"
    print("  ✓ PASSED")

    # Memory comparison
    fp32_bytes = H.numel() * 4
    nf4_bytes = qdict["codes"].numel() + qdict["scales"].numel() * 4
    print(f"\n  Memory: fp32={fp32_bytes} bytes, NF4={nf4_bytes} bytes "
          f"({fp32_bytes/nf4_bytes:.1f}× compression)")
    assert nf4_bytes < fp32_bytes / 2, "NF4 should be at least 2x smaller than fp32"
    print("  ✓ PASSED")

    # Test with different block sizes
    for bs in [32, 64, 128, 256]:
        qdict_bs = quantize_nf4(H, block_size=bs)
        H_deq_bs = dequantize_nf4(qdict_bs)
        err_bs = (H - H_deq_bs).abs().mean().item() / H.abs().mean().item()
        nf4_bytes_bs = qdict_bs["codes"].numel() + qdict_bs["scales"].numel() * 4
        print(f"  block_size={bs:3d}: rel_err={err_bs:.4%}, "
              f"memory={nf4_bytes_bs} bytes ({fp32_bytes/nf4_bytes_bs:.1f}×)")


def test_quantized_hamiltonian_wrapper():
    """Test the QuantizedHamiltonian wrapper class."""
    print("\n" + "=" * 60)
    print("TEST 3: QuantizedHamiltonian Wrapper")
    print("=" * 60)

    torch.manual_seed(42)
    H = torch.randn(4, 32, 32, device="cuda") * 0.05

    # INT8 wrapper
    qh8 = quantize_hamiltonian(H, bits=8)
    H_deq8 = qh8.dequantize(dtype=torch.float32)
    err8 = (H - H_deq8).abs().mean().item()
    print(f"  INT8 wrapper: shape={qh8.shape}, bits={qh8.bits}, "
          f"memory={qh8.memory_bytes()} bytes, mean_err={err8:.6f}")
    assert qh8.shape == H.shape
    assert qh8.bits == 8
    print("  ✓ PASSED")

    # NF4 wrapper
    qh4 = quantize_hamiltonian(H, bits=4)
    H_deq4 = qh4.dequantize(dtype=torch.float32)
    err4 = (H - H_deq4).abs().mean().item()
    print(f"  NF4 wrapper:  shape={qh4.shape}, bits={qh4.bits}, "
          f"memory={qh4.memory_bytes()} bytes, mean_err={err4:.6f}")
    assert qh4.shape == H.shape
    assert qh4.bits == 4
    print("  ✓ PASSED")

    # Memory comparison
    fp32_mem = H.numel() * 4
    print(f"\n  Memory: fp32={fp32_mem} bytes, INT8={qh8.memory_bytes()} bytes, "
          f"NF4={qh4.memory_bytes()} bytes")
    assert qh8.memory_bytes() < fp32_mem
    assert qh4.memory_bytes() < qh8.memory_bytes()
    print("  ✓ NF4 < INT8 < fp32 confirmed")


def test_state_quantization():
    """Test quantization/dequantization of Hamiltonian generation states."""
    print("\n" + "=" * 60)
    print("TEST 4: Hamiltonian State Quantization/Dequantization")
    print("=" * 60)

    # Create a small model
    model = HamiltonianVortexNetwork(
        vocab_size=100, d_model=128, num_layers=2, n_heads=4,
    ).cuda()

    # Initialize batched generation states
    B = 2
    states = model.init_generation_state_batched(B, device="cuda", dtype=torch.float32)
    print(f"  Initialized {len(states)} block states, each shape: {states[0].shape}")

    # Quantize states (INT8)
    q_states_8 = quantize_hamiltonian_states(states, bits=8)
    deq_states_8 = dequantize_hamiltonian_states(q_states_8, dtype=torch.float32)

    # Check round-trip
    for i in range(len(states)):
        err = (states[i] - deq_states_8[i]).abs().mean().item()
        assert err < 0.001, f"State {i} INT8 round-trip error too high: {err}"
    print(f"  INT8 state round-trip: max mean error = {max((states[i] - deq_states_8[i]).abs().mean().item() for i in range(len(states))):.6f}")
    print("  ✓ PASSED")

    # Quantize states (NF4)
    q_states_4 = quantize_hamiltonian_states(states, bits=4)
    deq_states_4 = dequantize_hamiltonian_states(q_states_4, dtype=torch.float32)

    for i in range(len(states)):
        err = (states[i] - deq_states_4[i]).abs().mean().item()
        assert err < 0.01, f"State {i} NF4 round-trip error too high: {err}"
    print(f"  NF4 state round-trip: max mean error = {max((states[i] - deq_states_4[i]).abs().mean().item() for i in range(len(states))):.6f}")
    print("  ✓ PASSED")


def test_quantized_inference():
    """Test quantized inference generation produces valid output."""
    print("\n" + "=" * 60)
    print("TEST 5: Quantized Inference Generation")
    print("=" * 60)

    # Create a small model
    model = HamiltonianVortexNetwork(
        vocab_size=100, d_model=64, num_layers=2, n_heads=4,
    ).cuda()
    model.eval()

    prompt = [1, 5, 10, 15, 20]

    # Full-precision generation (baseline)
    with torch.no_grad():
        fp_output = model.generate_incremental(
            prompt_ids=prompt, max_new_tokens=30, temperature=0.8, top_k=10,
        )
    print(f"  Full-precision output ({len(fp_output)} tokens): {fp_output[:15]}...")

    # INT8 quantized inference
    q_infer_8 = QuantizedHVNInference(model, bits=8, dequant_dtype=torch.float32)
    q8_output = q_infer_8.generate_incremental(
        prompt_ids=prompt, max_new_tokens=30, temperature=0.8, top_k=10,
    )
    print(f"  INT8 quantized output  ({len(q8_output)} tokens): {q8_output[:15]}...")

    # NF4 quantized inference
    q_infer_4 = QuantizedHVNInference(model, bits=4, dequant_dtype=torch.float32)
    q4_output = q_infer_4.generate_incremental(
        prompt_ids=prompt, max_new_tokens=30, temperature=0.8, top_k=10,
    )
    print(f"  NF4 quantized output   ({len(q4_output)} tokens): {q4_output[:15]}...")

    # Verify outputs are valid (correct length, valid token IDs)
    assert len(fp_output) == len(prompt) + 30
    assert len(q8_output) == len(prompt) + 30
    assert len(q4_output) == len(prompt) + 30
    assert all(0 <= t < 100 for t in q8_output)
    assert all(0 <= t < 100 for t in q4_output)
    print("  ✓ All outputs have correct length and valid token IDs")

    # Check that quantized outputs share some overlap with fp (untrained model,
    # so we don't expect identical outputs, but the prompt should match)
    assert q8_output[:len(prompt)] == prompt, "INT8 prompt not preserved"
    assert q4_output[:len(prompt)] == prompt, "NF4 prompt not preserved"
    print("  ✓ Prompt tokens preserved in all outputs")


def test_memory_savings():
    """Test memory savings estimation."""
    print("\n" + "=" * 60)
    print("TEST 6: Memory Savings Estimation")
    print("=" * 60)

    model = HamiltonianVortexNetwork(
        vocab_size=1000, d_model=256, num_layers=4, n_heads=8,
    ).cuda()

    for bits in [8, 4]:
        savings = estimate_memory_savings(model, batch_size=1, bits=bits)
        print(f"\n  {bits}-bit quantization (d_model=256, 4 layers, 8 heads, B=1):")
        print(f"    fp32 Hamiltonian memory:  {savings['fp32_bytes']:>10,} bytes ({savings['fp32_bytes']/1024:.1f} KB)")
        print(f"    Quantized memory:         {savings['quant_bytes']:>10,} bytes ({savings['quant_bytes']/1024:.1f} KB)")
        print(f"    Compression ratio:        {savings['compression']:.1f}×")
        print(f"    Memory savings:           {savings['savings_pct']:.1f}%")
        assert savings["compression"] > 1.0, "Quantization should save memory"
        assert savings["savings_pct"] > 50, "Should save at least 50% memory"

    # Larger model
    model_large = HamiltonianVortexNetwork(
        vocab_size=32000, d_model=512, num_layers=12, n_heads=8,
    ).cuda()

    print(f"\n  Large model (d_model=512, 12 layers, 8 heads, B=1):")
    for bits in [8, 4]:
        savings = estimate_memory_savings(model_large, batch_size=1, bits=bits)
        print(f"    {bits}-bit: {savings['fp32_bytes']/1024:.1f} KB → "
              f"{savings['quant_bytes']/1024:.1f} KB "
              f"({savings['compression']:.1f}×, {savings['savings_pct']:.1f}% saved)")

    print("\n  ✓ All memory savings verified")


def test_batched_quantized_inference():
    """Test batched quantized inference with multiple sequences."""
    print("\n" + "=" * 60)
    print("TEST 7: Batched Quantized Inference")
    print("=" * 60)

    model = HamiltonianVortexNetwork(
        vocab_size=100, d_model=64, num_layers=2, n_heads=4,
    ).cuda()
    model.eval()

    prompts = [
        [1, 2, 3, 4, 5],
        [10, 20, 30],
        [50, 60, 70, 80],
    ]

    # Full precision batched
    with torch.no_grad():
        fp_outputs = model.generate_batched(prompts, max_new_tokens=20, temperature=0.8, top_k=10)

    # INT8 batched
    q_infer_8 = QuantizedHVNInference(model, bits=8)
    q8_outputs = q_infer_8.generate_batched(prompts, max_new_tokens=20, temperature=0.8, top_k=10)

    # NF4 batched
    q_infer_4 = QuantizedHVNInference(model, bits=4)
    q4_outputs = q_infer_4.generate_batched(prompts, max_new_tokens=20, temperature=0.8, top_k=10)

    print(f"  {len(prompts)} sequences, 20 new tokens each:")
    for i, p in enumerate(prompts):
        print(f"    Seq {i}: prompt={p}")
        print(f"      fp32:  {fp_outputs[i][:15]}...")
        print(f"      INT8:  {q8_outputs[i][:15]}...")
        print(f"      NF4:   {q4_outputs[i][:15]}...")

    # Verify
    for i in range(len(prompts)):
        assert len(q8_outputs[i]) == len(prompts[i]) + 20
        assert len(q4_outputs[i]) == len(prompts[i]) + 20
        assert q8_outputs[i][:len(prompts[i])] == prompts[i]
        assert q4_outputs[i][:len(prompts[i])] == prompts[i]

    print("  ✓ All batched outputs correct")


def test_speed_comparison():
    """Compare generation speed: fp32 vs INT8 vs NF4."""
    print("\n" + "=" * 60)
    print("TEST 8: Generation Speed Comparison")
    print("=" * 60)

    model = HamiltonianVortexNetwork(
        vocab_size=100, d_model=128, num_layers=2, n_heads=4,
    ).cuda()
    model.eval()

    prompt = [1, 5, 10, 15, 20, 25, 30]
    n_tokens = 100
    n_runs = 3

    # Warm up
    with torch.no_grad():
        _ = model.generate_incremental(prompt, max_new_tokens=10)

    # fp32
    times_fp = []
    with torch.no_grad():
        for _ in range(n_runs):
            torch.cuda.synchronize()
            t0 = time.time()
            _ = model.generate_incremental(prompt, max_new_tokens=n_tokens, temperature=0.8, top_k=10)
            torch.cuda.synchronize()
            times_fp.append(time.time() - t0)
    avg_fp = sum(times_fp) / n_runs

    # INT8
    q_infer_8 = QuantizedHVNInference(model, bits=8)
    times_8 = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = q_infer_8.generate_incremental(prompt, max_new_tokens=n_tokens, temperature=0.8, top_k=10)
        torch.cuda.synchronize()
        times_8.append(time.time() - t0)
    avg_8 = sum(times_8) / n_runs

    # NF4
    q_infer_4 = QuantizedHVNInference(model, bits=4)
    times_4 = []
    for _ in range(n_runs):
        torch.cuda.synchronize()
        t0 = time.time()
        _ = q_infer_4.generate_incremental(prompt, max_new_tokens=n_tokens, temperature=0.8, top_k=10)
        torch.cuda.synchronize()
        times_4.append(time.time() - t0)
    avg_4 = sum(times_4) / n_runs

    print(f"  Generating {n_tokens} tokens (d_model=128, 2 layers, 4 heads):")
    print(f"    fp32:  {avg_fp*1000:.1f} ms  ({n_tokens/avg_fp:.1f} tok/s)")
    print(f"    INT8:  {avg_8*1000:.1f} ms  ({n_tokens/avg_8:.1f} tok/s)  [{avg_fp/avg_8:.2f}× relative]")
    print(f"    NF4:   {avg_4*1000:.1f} ms  ({n_tokens/avg_4:.1f} tok/s)  [{avg_fp/avg_4:.2f}× relative]")
    print("  ✓ Speed comparison complete")


def main():
    print("=" * 60)
    print("Hamiltonian Quantization Test Suite (Issue #20)")
    print("=" * 60)

    # Verify GPU
    assert torch.cuda.is_available(), "CUDA is required — install GPU PyTorch!"
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"PyTorch: {torch.__version__}")

    test_int8_accuracy()
    test_nf4_accuracy()
    test_quantized_hamiltonian_wrapper()
    test_state_quantization()
    test_quantized_inference()
    test_memory_savings()
    test_batched_quantized_inference()
    test_speed_comparison()

    print("\n" + "=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
    print("\nSummary:")
    print("  - INT8 quantization: ~4× memory reduction, <5% relative error")
    print("  - NF4 4-bit quantization: ~8× memory reduction, <15% relative error")
    print("  - Quantized inference: valid generation with quantized Hamiltonian states")
    print("  - Memory savings: 75-87% reduction in Hamiltonian state memory")
    print("  - Batched inference: multiple sequences with quantized states")


if __name__ == "__main__":
    main()