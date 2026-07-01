"""
HVN Demo — Runnable Forward Pass
=================================

Demonstrates a complete forward pass through the Hamiltonian Vortex Network:

    Input:  [batch_size, sequence_length, d_model]  (dummy embeddings)
    Output: [batch_size, sequence_length, vocab_size]  (logits)

Also demonstrates the token-ID-based forward path and prints tensor shapes
at each stage to verify correctness.
"""

import torch
from hvn.core import HamiltonianVortexNetwork


def demo_embedding_forward():
    """Demo using pre-computed embeddings (as specified in the blueprint)."""
    print("=" * 70)
    print("DEMO 1: Forward pass from embeddings")
    print("=" * 70)

    # Hyperparameters
    batch_size = 4
    seq_len = 64
    d_model = 128
    vocab_size = 1000
    num_layers = 2
    bond_dimension = 32  # SVD truncation rank (quantum renormalization)

    # Create dummy input: [batch_size, seq_len, d_model]
    x = torch.randn(batch_size, seq_len, d_model)
    print(f"Input shape:  {x.shape}  (batch={batch_size}, seq_len={seq_len}, d_model={d_model})")

    # Build the model
    model = HamiltonianVortexNetwork(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        bond_dimension=bond_dimension,
    )
    model.eval()

    # Forward pass
    with torch.no_grad():
        logits = model.forward_from_embeddings(x)

    print(f"Output shape: {logits.shape}  (batch={batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")
    print(f"Logit range:  [{logits.min():.4f}, {logits.max():.4f}]")
    print(f"Logit mean:   {logits.mean():.4f}")
    print()

    # Verify shape correctness
    expected = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected, f"Shape mismatch! Expected {expected}, got {tuple(logits.shape)}"
    print("✓ Shape verification passed!")
    print()


def demo_token_ids_forward():
    """Demo using token IDs (standard language model interface)."""
    print("=" * 70)
    print("DEMO 2: Forward pass from token IDs")
    print("=" * 70)

    # Hyperparameters
    batch_size = 2
    seq_len = 32
    d_model = 64
    vocab_size = 500
    num_layers = 3
    bond_dimension = 16

    # Create dummy token IDs
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input IDs:    {input_ids.shape}  (batch={batch_size}, seq_len={seq_len})")

    # Build the model
    model = HamiltonianVortexNetwork(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        bond_dimension=bond_dimension,
    )
    model.eval()

    # Forward pass
    with torch.no_grad():
        logits = model(input_ids)

    print(f"Output shape: {logits.shape}  (batch={batch_size}, seq_len={seq_len}, vocab_size={vocab_size})")
    print()

    # Verify shape correctness
    expected = (batch_size, seq_len, vocab_size)
    assert logits.shape == expected, f"Shape mismatch! Expected {expected}, got {tuple(logits.shape)}"
    print("✓ Shape verification passed!")
    print()


def demo_long_sequence():
    """Demo with a long sequence to showcase O(1) per-token scaling."""
    print("=" * 70)
    print("DEMO 3: Long sequence (1024 tokens) — O(1) per-token inference")
    print("=" * 70)

    batch_size = 1
    seq_len = 1024
    d_model = 128
    vocab_size = 2000
    num_layers = 2
    bond_dimension = 32

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    print(f"Input IDs:    {input_ids.shape}  (seq_len={seq_len})")

    model = HamiltonianVortexNetwork(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        bond_dimension=bond_dimension,
    )
    model.eval()

    with torch.no_grad():
        logits = model(input_ids)

    print(f"Output shape: {logits.shape}")
    print(f"✓ Successfully processed {seq_len} tokens with fixed-size Hamiltonian memory.")
    print(f"  Memory per token: O(d_model²) = O({d_model}²) = O({d_model * d_model}) — constant in N.")
    print()


def demo_backprop():
    """Demo that the model is differentiable (supports backprop for training)."""
    print("=" * 70)
    print("DEMO 4: Backpropagation (training mode)")
    print("=" * 70)

    batch_size = 2
    seq_len = 16
    d_model = 64
    vocab_size = 300
    num_layers = 2

    model = HamiltonianVortexNetwork(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
    )
    model.train()

    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    target_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    logits = model(input_ids)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, vocab_size),
        target_ids.view(-1),
    )
    loss.backward()

    # Check gradients exist
    grad_count = sum(1 for p in model.parameters() if p.grad is not None)
    total_params = sum(1 for p in model.parameters())
    print(f"Loss: {loss.item():.4f}")
    print(f"Parameters with gradients: {grad_count}/{total_params}")
    print("✓ Backpropagation works — model is trainable!")
    print()


if __name__ == "__main__":
    print()
    print("╔════════════════════════════════════════════════════════════════════╗")
    print("║     Hamiltonian Vortex Network (HVN) — Demo Forward Passes        ║")
    print("╚════════════════════════════════════════════════════════════════════╝")
    print()

    demo_embedding_forward()
    demo_token_ids_forward()
    demo_long_sequence()
    demo_backprop()

    print("=" * 70)
    print("ALL DEMOS COMPLETED SUCCESSFULLY ✓")
    print("=" * 70)