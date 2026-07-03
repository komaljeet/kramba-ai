#!/usr/bin/env python3
"""
Multimodal Fusion Demo (Issue #27)
===================================

Demonstrates joint image-text training and cross-modal retrieval with HVN.

Tests:
1. Gated fusion layer — image + text → fused embeddings
2. Modality embeddings — [IMAGE]/[TEXT] type embeddings
3. Cross-modal retriever — contrastive learning + retrieval
4. MultimodalFusionHVN forward — image + text → logits
5. Joint training loss — LM + contrastive retrieval
6. Image captioning — image → text generation
7. Visual QA — image + question → answer
8. Text-to-image retrieval
9. Image-to-text retrieval
10. O(1) inference verification
11. Memory comparison

Issue #27 — https://gitlab.com/komalbarun/kramba-ai/-/work_items/27
"""

from __future__ import annotations

import argparse
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from hvn.core import HamiltonianVortexNetwork
from hvn.vision import (
    ImageEncoder,
    CrossModalProjector,
    GatedFusionLayer,
    ModalityEmbedding,
    CrossModalRetriever,
    MultimodalFusionHVN,
    generate_synthetic_image_text_pairs,
)


def test_passed(name: str, condition: bool, detail: str = "") -> bool:
    status = "✅ PASS" if condition else "❌ FAIL"
    print(f"  {status}: {name}" + (f" — {detail}" if detail else ""))
    return condition


def main():
    parser = argparse.ArgumentParser(description="Multimodal Fusion Demo (Issue #27)")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--in_channels", type=int, default=3, help="Image channels")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of HVN blocks")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocabulary size")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--text_seq_len", type=int, default=16, help="Text sequence length")
    parser.add_argument("--n_threads", type=int, default=1, help="Hamiltonian threads")
    parser.add_argument("--rank_k", type=int, default=1, help="Rank-k update")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    # Verify GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    assert torch.cuda.is_available(), "CUDA is required — install GPU PyTorch!"
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    torch.manual_seed(args.seed)

    all_passed = True
    print("\n" + "=" * 60)
    print("MULTIMODAL FUSION DEMO (Issue #27)")
    print("=" * 60)

    # --- Build model ---
    model = HamiltonianVortexNetwork(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        n_threads=args.n_threads,
        rank_k=args.rank_k,
    ).to(device)

    vision_encoder = ImageEncoder(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        d_model=args.d_model,
    ).to(device)

    multimodal = MultimodalFusionHVN(
        vision_encoder=vision_encoder,
        hvn=model,
        use_cross_modal_proj=True,
        use_gated_fusion=True,
        use_modality_embed=True,
        use_retrieval_head=True,
        retrieval_dim=args.d_model,
        dropout=0.1,
    ).to(device)

    # Generate synthetic data
    images, text_ids = generate_synthetic_image_text_pairs(
        num_pairs=args.batch_size,
        image_size=args.image_size,
        in_channels=args.in_channels,
        text_vocab_size=args.vocab_size,
        text_seq_len=args.text_seq_len,
        seed=args.seed,
    )
    images = images.to(device)
    text_ids = text_ids.to(device)

    num_patches = vision_encoder.num_patches
    print(f"\nModel: d_model={args.d_model}, layers={args.num_layers}, heads={args.n_heads}")
    print(f"Vision: {args.image_size}×{args.image_size} → {num_patches} patches")
    print(f"Data: {images.shape} images, {text_ids.shape} text")

    # --- Test 1: Gated Fusion Layer ---
    print("\n--- Test 1: Gated Fusion Layer ---")
    gated_fusion = GatedFusionLayer(args.d_model).to(device)
    img_embeds = torch.randn(args.batch_size, num_patches, args.d_model, device=device)
    txt_embeds = torch.randn(args.batch_size, args.text_seq_len, args.d_model, device=device)
    fused = gated_fusion(img_embeds, txt_embeds)
    expected_shape = (args.batch_size, num_patches + args.text_seq_len, args.d_model)
    all_passed &= test_passed(
        "Gated fusion output shape",
        fused.shape == expected_shape,
        f"got {fused.shape}, expected {expected_shape}",
    )

    # --- Test 2: Modality Embedding ---
    print("\n--- Test 2: Modality Embedding ---")
    mod_embed = ModalityEmbedding(args.d_model).to(device)
    img_with_mod = mod_embed(img_embeds, modality=0)
    txt_with_mod = mod_embed(txt_embeds, modality=1)
    all_passed &= test_passed(
        "Modality embedding preserves shape",
        img_with_mod.shape == img_embeds.shape and txt_with_mod.shape == txt_embeds.shape,
    )
    # Verify modality embeddings are different
    diff = (img_with_mod - img_embeds).abs().mean().item()
    all_passed &= test_passed(
        "Image modality embedding is non-zero",
        diff > 0,
        f"mean diff = {diff:.6f}",
    )

    # --- Test 3: Cross-Modal Retriever ---
    print("\n--- Test 3: Cross-Modal Retriever ---")
    retriever = CrossModalRetriever(args.d_model).to(device)
    img_vec = retriever.encode_image(img_embeds)  # [B, retrieval_dim]
    txt_vec = retriever.encode_text(txt_embeds)  # [B, retrieval_dim]
    all_passed &= test_passed(
        "Image retrieval vector shape",
        img_vec.shape == (args.batch_size, args.d_model),
        f"got {img_vec.shape}",
    )
    all_passed &= test_passed(
        "Text retrieval vector shape",
        txt_vec.shape == (args.batch_size, args.d_model),
        f"got {txt_vec.shape}",
    )
    # Check L2 normalization
    norms = img_vec.norm(dim=-1)
    all_passed &= test_passed(
        "Retrieval vectors are L2-normalized",
        torch.allclose(norms, torch.ones_like(norms), atol=1e-5),
        f"norms = {norms.tolist()}",
    )
    # Test retrieval
    indices, sims = retriever.retrieve(img_vec, txt_vec, top_k=2)
    all_passed &= test_passed(
        "Retrieval indices shape",
        indices.shape == (args.batch_size, 2),
        f"got {indices.shape}",
    )
    # Test contrastive loss
    loss = retriever.contrastive_loss(img_vec, txt_vec)
    all_passed &= test_passed(
        "Contrastive loss is a scalar",
        loss.dim() == 0 and loss.item() > 0,
        f"loss = {loss.item():.4f}",
    )

    # --- Test 4: MultimodalFusionHVN Forward ---
    print("\n--- Test 4: MultimodalFusionHVN Forward ---")
    logits = multimodal(images, text_ids)
    expected_logits_shape = (args.batch_size, num_patches + args.text_seq_len, args.vocab_size)
    all_passed &= test_passed(
        "Forward output shape (image + text)",
        logits.shape == expected_logits_shape,
        f"got {logits.shape}, expected {expected_logits_shape}",
    )

    # Image-only forward
    logits_img_only = multimodal(images)
    expected_img_shape = (args.batch_size, num_patches, args.vocab_size)
    all_passed &= test_passed(
        "Forward output shape (image only)",
        logits_img_only.shape == expected_img_shape,
        f"got {logits_img_only.shape}, expected {expected_img_shape}",
    )

    # --- Test 5: Joint Training Loss ---
    print("\n--- Test 5: Joint Training Loss ---")
    loss_dict = multimodal.compute_loss(images, text_ids, retrieval_weight=0.5)
    all_passed &= test_passed(
        "Loss dict has total, lm, retrieval",
        "total" in loss_dict and "lm" in loss_dict and "retrieval" in loss_dict,
    )
    all_passed &= test_passed(
        "Total loss is positive scalar",
        loss_dict["total"].item() > 0 and loss_dict["total"].dim() == 0,
        f"total = {loss_dict['total'].item():.4f}",
    )
    print(f"     LM loss: {loss_dict['lm'].item():.4f} | Retrieval loss: {loss_dict['retrieval'].item():.4f}")

    # Verify gradients flow (check that at least the key parameters get gradients)
    loss_dict["total"].backward()
    params_with_grad = sum(
        1 for p in multimodal.parameters() if p.requires_grad and p.grad is not None and p.grad.abs().sum().item() > 0
    )
    total_trainable = sum(1 for p in multimodal.parameters() if p.requires_grad)
    grad_ok = params_with_grad > total_trainable * 0.3  # At least 30% of params get gradients
    all_passed &= test_passed(
        "Gradients flow through model parameters",
        grad_ok,
        f"{params_with_grad}/{total_trainable} params have non-zero gradients",
    )

    # --- Test 6: Image Captioning ---
    print("\n--- Test 6: Image Captioning ---")
    single_image = images[:1]
    caption_ids = multimodal.generate_caption(
        single_image, max_new_tokens=20, temperature=0.8, top_k=10, device=device,
    )
    all_passed &= test_passed(
        "Captioning generates tokens",
        len(caption_ids) > 0 and len(caption_ids) <= 20,
        f"generated {len(caption_ids)} tokens",
    )

    # --- Test 7: Visual QA ---
    print("\n--- Test 7: Visual Question Answering ---")
    question_ids = text_ids[0, :args.text_seq_len // 2].tolist()
    answer_ids = multimodal.answer_question(
        single_image, question_ids, max_new_tokens=15, temperature=0.8, top_k=10, device=device,
    )
    all_passed &= test_passed(
        "VQA generates answer tokens",
        len(answer_ids) > 0 and len(answer_ids) <= 15,
        f"generated {len(answer_ids)} tokens",
    )

    # --- Test 8: Text-to-Image Retrieval ---
    print("\n--- Test 8: Text-to-Image Retrieval ---")
    query_ids = text_ids[0].tolist()
    candidate_images = images[:args.batch_size]
    indices, sims = multimodal.retrieve_images(
        query_ids, candidate_images, top_k=3, device=device,
    )
    all_passed &= test_passed(
        "T2I retrieval returns indices",
        len(indices) == 3 and len(sims) == 3,
        f"top-3 indices: {indices.tolist()}",
    )

    # --- Test 9: Image-to-Text Retrieval ---
    print("\n--- Test 9: Image-to-Text Retrieval ---")
    text_candidates = [text_ids[i].tolist() for i in range(args.batch_size)]
    indices, sims = multimodal.retrieve_text(
        single_image, text_candidates, top_k=3, device=device,
    )
    all_passed &= test_passed(
        "I2T retrieval returns indices",
        len(indices) == 3 and len(sims) == 3,
        f"top-3 indices: {indices.tolist()}",
    )

    # --- Test 10: O(1) Inference Verification ---
    print("\n--- Test 10: O(1) Inference Verification ---")
    multimodal.eval()
    patch_counts = [16, 64, 256]
    timings = []
    with torch.no_grad():
        for num_p in patch_counts:
            x = torch.randn(1, num_p, args.d_model, device=device)
            param_dtype = next(multimodal.parameters()).dtype
            x = x.to(param_dtype)
            states = model.init_generation_state(1, device=device, dtype=param_dtype)

            # Warmup
            for t in range(min(3, num_p)):
                patch_t = x[:, t:t + 1, :]
                new_states = []
                for i, block in enumerate(model.blocks):
                    patch_t, s = block.forward_step(patch_t, states[i], position_idx=t)
                    new_states.append(s)
                states = new_states

            if device.type == "cuda":
                torch.cuda.synchronize()

            start = time.perf_counter()
            for t in range(num_p):
                patch_t = x[:, t:t + 1, :]
                new_states = []
                for i, block in enumerate(model.blocks):
                    patch_t, s = block.forward_step(patch_t, states[i], position_idx=t)
                    new_states.append(s)
                states = new_states

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            ms_per_patch = (elapsed / num_p) * 1000
            timings.append(ms_per_patch)
            print(f"     {num_p:4d} patches: {ms_per_patch:.3f} ms/patch (total: {elapsed * 1000:.1f} ms)")

    if len(timings) >= 2:
        ratio = max(timings) / max(min(timings), 1e-6)
        all_passed &= test_passed(
            "O(1) scaling (max/min ratio < 3x)",
            ratio < 3.0,
            f"ratio = {ratio:.2f}",
        )

    # --- Test 11: Memory Comparison ---
    print("\n--- Test 11: Memory Comparison ---")
    d_head = args.d_model // args.n_heads
    hvn_mem = d_head * d_head * args.n_heads * args.n_threads * args.num_layers * 4  # fp32 bytes
    hvn_mem_mb = hvn_mem / (1024 ** 2)

    print(f"     HVN Hamiltonian memory: {hvn_mem_mb:.4f} MB")
    for N in [1000, 10000, 100000]:
        kv_mb = (2 * N * args.d_model * args.num_layers * 2) / (1024 ** 2)  # fp16
        ratio = hvn_mem_mb / kv_mb if kv_mb > 0 else 0
        print(f"     Transformer KV cache @ N={N:6d}: {kv_mb:.2f} MB (HVN = {ratio:.4%})")

    all_passed &= test_passed(
        "HVN memory < 1% of KV cache at N=100K",
        hvn_mem_mb / ((2 * 100000 * args.d_model * args.num_layers * 2) / (1024 ** 2)) < 0.01,
    )

    # --- Summary ---
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED — Issue #27 Multimodal Fusion verified!")
    else:
        print("⚠️  SOME TESTS FAILED — review output above")
    print("=" * 60)

    # Parameter count
    total_params = sum(p.numel() for p in multimodal.parameters())
    hvn_params = sum(p.numel() for p in model.parameters())
    vision_params = sum(p.numel() for p in vision_encoder.parameters())
    fusion_params = total_params - hvn_params - vision_params
    print(f"\nParameter count:")
    print(f"  HVN core:     {hvn_params:>10,}")
    print(f"  Vision:       {vision_params:>10,}")
    print(f"  Fusion heads: {fusion_params:>10,}")
    print(f"  Total:        {total_params:>10,}")

    sys.exit(0 if all_passed else 1)


if __name__ == "__main__":
    main()