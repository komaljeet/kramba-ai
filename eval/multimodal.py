"""
Multimodal Evaluation Harness
==============================

Cross-modal evaluation for the Hamiltonian Vortex Network (HVN).

Tests:
1. Image captioning quality (image → text generation)
2. Visual question answering (image + question → answer)
3. Cross-modal retrieval (text-to-image and image-to-text)
4. O(1) inference verification for multimodal
5. Memory scaling comparison

Issue #27 — https://gitlab.com/komalbarun/kramba-ai/-/work_items/27
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

# Ensure the project root is on sys.path so we can import sibling modules
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from hvn.core import HamiltonianVortexNetwork
from hvn.vision import (
    ImageEncoder,
    MultimodalFusionHVN,
    generate_synthetic_image_text_pairs,
)


def evaluate_captioning(
    model: MultimodalFusionHVN,
    images: torch.Tensor,
    text_ids: torch.Tensor,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 30,
) -> dict:
    """Evaluate image captioning quality.

    Generates captions from images and computes perplexity against
    the ground-truth captions.

    Parameters
    ----------
    model : MultimodalFusionHVN
        Multimodal model.
    images : torch.Tensor
        Test images, shape ``[B, C, H, W]``.
    text_ids : torch.Tensor
        Ground-truth caption token IDs, shape ``[B, S_text]``.
    tokenizer : object
        Tokenizer for decoding.
    device : torch.device
        Device to run on.
    max_new_tokens : int
        Maximum tokens to generate per caption.

    Returns
    -------
    results : dict
        Evaluation results including perplexity and sample captions.
    """
    model.eval()
    results = {"captions": [], "ppl": float("inf")}

    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for i in range(min(len(images), 8)):
            img = images[i:i + 1]
            # Generate caption
            gen_ids = model.generate_caption(
                img, max_new_tokens=max_new_tokens, temperature=0.8, top_k=10, device=device,
            )
            gen_text = tokenizer.decode(gen_ids)
            gt_text = tokenizer.decode(text_ids[i].tolist())

            results["captions"].append({
                "image_idx": i,
                "generated": gen_text,
                "ground_truth": gt_text,
            })

            # Compute LM loss on ground-truth caption
            logits = model(img, text_ids[i:i + 1])
            num_patches = model.vision_encoder.num_patches
            text_logits = logits[:, num_patches:, :]
            if text_logits.size(1) > 1:
                pred = text_logits[:, :-1, :].contiguous()
                target = text_ids[i:i + 1, 1:].contiguous()
                loss = nn.functional.cross_entropy(
                    pred.view(-1, pred.size(-1)), target.view(-1),
                )
                total_loss += loss.item()
                n_samples += 1

    if n_samples > 0:
        results["ppl"] = math.exp(min(total_loss / n_samples, 20))

    return results


def evaluate_vqa(
    model: MultimodalFusionHVN,
    images: torch.Tensor,
    questions: torch.Tensor,
    answers: torch.Tensor,
    tokenizer,
    device: torch.device,
    max_new_tokens: int = 20,
) -> dict:
    """Evaluate visual question answering.

    Parameters
    ----------
    model : MultimodalFusionHVN
        Multimodal model.
    images : torch.Tensor
        Test images, shape ``[B, C, H, W]``.
    questions : torch.Tensor
        Question token IDs, shape ``[B, S_q]``.
    answers : torch.Tensor
        Ground-truth answer token IDs, shape ``[B, S_a]``.
    tokenizer : object
        Tokenizer for decoding.
    device : torch.device
        Device to run on.
    max_new_tokens : int
        Maximum answer tokens to generate.

    Returns
    -------
    results : dict
        Evaluation results including sample Q&A pairs.
    """
    model.eval()
    results = {"qa_pairs": [], "ppl": float("inf")}

    total_loss = 0.0
    n_samples = 0

    with torch.no_grad():
        for i in range(min(len(images), 8)):
            img = images[i:i + 1]
            q_ids = questions[i].tolist()
            # Generate answer
            ans_ids = model.answer_question(
                img, q_ids, max_new_tokens=max_new_tokens, temperature=0.8, top_k=10, device=device,
            )
            gen_answer = tokenizer.decode(ans_ids)
            gt_answer = tokenizer.decode(answers[i].tolist())

            results["qa_pairs"].append({
                "image_idx": i,
                "question": tokenizer.decode(q_ids),
                "generated_answer": gen_answer,
                "ground_truth_answer": gt_answer,
            })

            # Compute LM loss on ground-truth answer
            # Feed image + question + answer through model
            combined_ids = torch.tensor([q_ids + answers[i].tolist()], dtype=torch.long, device=device)
            logits = model(img, combined_ids)
            num_patches = model.vision_encoder.num_patches
            text_logits = logits[:, num_patches + len(q_ids):, :]
            if text_logits.size(1) > 1:
                pred = text_logits[:, :-1, :].contiguous()
                target = answers[i:i + 1, 1:].contiguous()
                if pred.size(1) == target.size(1):
                    loss = nn.functional.cross_entropy(
                        pred.view(-1, pred.size(-1)), target.view(-1),
                    )
                    total_loss += loss.item()
                    n_samples += 1

    if n_samples > 0:
        results["ppl"] = math.exp(min(total_loss / n_samples, 20))

    return results


def evaluate_retrieval(
    model: MultimodalFusionHVN,
    images: torch.Tensor,
    text_ids: torch.Tensor,
    device: torch.device,
    top_k: int = 5,
) -> dict:
    """Evaluate cross-modal retrieval.

    Tests both text-to-image and image-to-text retrieval.
    Computes Recall@1 and Recall@k metrics.

    Parameters
    ----------
    model : MultimodalFusionHVN
        Multimodal model with retrieval head.
    images : torch.Tensor
        Test images, shape ``[B, C, H, W]``.
    text_ids : torch.Tensor
        Text token IDs, shape ``[B, S_text]``.
    device : torch.device
        Device to run on.
    top_k : int
        K for Recall@K.

    Returns
    -------
    results : dict
        Retrieval metrics including Recall@1, Recall@k.
    """
    if not model.use_retrieval_head:
        return {"error": "Retrieval head not enabled"}

    model.eval()
    B = len(images)
    results = {}

    with torch.no_grad():
        # Encode all images and texts
        img_embeds = model.encode_images(images.to(device))
        txt_embeds = model.encode_text(text_ids.to(device))

        img_vecs = model.retriever.encode_image(img_embeds)  # [B, retrieval_dim]
        txt_vecs = model.retriever.encode_text(txt_embeds)   # [B, retrieval_dim]

        # Text-to-image retrieval: for each text, find matching image
        # Ground truth: text[i] matches image[i]
        sim_matrix_t2i = txt_vecs @ img_vecs.T  # [B, B]
        _, topk_indices_t2i = torch.topk(sim_matrix_t2i, min(top_k, B), dim=-1)

        correct_t2i_1 = 0
        correct_t2i_k = 0
        for i in range(B):
            if topk_indices_t2i[i, 0].item() == i:
                correct_t2i_1 += 1
            if i in topk_indices_t2i[i].tolist():
                correct_t2i_k += 1

        results["t2i_recall@1"] = correct_t2i_1 / B
        results["t2i_recall@k"] = correct_t2i_k / B
        results["t2i_k"] = top_k

        # Image-to-text retrieval: for each image, find matching text
        sim_matrix_i2t = img_vecs @ txt_vecs.T  # [B, B]
        _, topk_indices_i2t = torch.topk(sim_matrix_i2t, min(top_k, B), dim=-1)

        correct_i2t_1 = 0
        correct_i2t_k = 0
        for i in range(B):
            if topk_indices_i2t[i, 0].item() == i:
                correct_i2t_1 += 1
            if i in topk_indices_i2t[i].tolist():
                correct_i2t_k += 1

        results["i2t_recall@1"] = correct_i2t_1 / B
        results["i2t_recall@k"] = correct_i2t_k / B
        results["i2t_k"] = top_k

        # Mean similarity on diagonal (should be high for good retrieval)
        diag_sim = sim_matrix_t2i.diag().mean().item()
        results["mean_diag_similarity"] = diag_sim

    return results


def evaluate_o1_inference(
    model: MultimodalFusionHVN,
    images: torch.Tensor,
    device: torch.device,
    patch_counts: list[int] | None = None,
) -> dict:
    """Verify O(1) per-token inference for multimodal model.

    Measures per-patch processing time as the number of patches increases.
    O(1) means the per-patch time should remain constant.

    Parameters
    ----------
    model : MultimodalFusionHVN
        Multimodal model.
    images : torch.Tensor
        Test images.
    device : torch.device
        Device to run on.
    patch_counts : list[int], optional
        Patch counts to test (defaults to [16, 64, 256, 1024]).

    Returns
    -------
    results : dict
        Timing results showing O(1) scaling.
    """
    if patch_counts is None:
        patch_counts = [16, 64, 256, 1024]

    model.eval()
    results = {"timings": [], "o1_confirmed": False}

    d_model = model.d_model
    B = 1

    with torch.no_grad():
        for num_patches in patch_counts:
            # Create synthetic patch embeddings
            x = torch.randn(B, num_patches, d_model, device=device)
            param_dtype = next(model.parameters()).dtype
            x = x.to(param_dtype)

            states = model.hvn.init_generation_state(B, device=device, dtype=param_dtype)

            # Warmup
            for t in range(min(5, num_patches)):
                patch_t = x[:, t:t + 1, :]
                new_states = []
                for i, block in enumerate(model.hvn.blocks):
                    patch_t, s = block.forward_step(patch_t, states[i], position_idx=t)
                    new_states.append(s)
                states = new_states

            if device.type == "cuda":
                torch.cuda.synchronize()

            # Timed run
            start = time.perf_counter()
            for t in range(num_patches):
                patch_t = x[:, t:t + 1, :]
                new_states = []
                for i, block in enumerate(model.hvn.blocks):
                    patch_t, s = block.forward_step(patch_t, states[i], position_idx=t)
                    new_states.append(s)
                states = new_states

            if device.type == "cuda":
                torch.cuda.synchronize()

            elapsed = time.perf_counter() - start
            ms_per_patch = (elapsed / num_patches) * 1000

            results["timings"].append({
                "num_patches": num_patches,
                "total_ms": elapsed * 1000,
                "ms_per_patch": ms_per_patch,
            })

    # Check if per-patch time is roughly constant (O(1))
    if len(results["timings"]) >= 2:
        times = [t["ms_per_patch"] for t in results["timings"]]
        max_ratio = max(times) / max(min(times), 1e-6)
        results["o1_confirmed"] = max_ratio < 3.0  # Allow 3x variance
        results["max_min_ratio"] = max_ratio

    return results


def evaluate_memory(
    model: MultimodalFusionHVN,
    device: torch.device,
) -> dict:
    """Compare multimodal HVN memory vs Transformer KV cache.

    Parameters
    ----------
    model : MultimodalFusionHVN
        Multimodal model.
    device : torch.device
        Device.

    Returns
    -------
    results : dict
        Memory comparison results.
    """
    d_model = model.d_model
    num_layers = model.hvn.num_layers
    num_patches = model.vision_encoder.num_patches

    # HVN Hamiltonian memory: d_head^2 * n_heads * n_threads * num_layers
    n_heads = model.hvn.n_heads
    n_threads = model.hvn.n_threads
    d_head = d_model // n_heads

    hvn_memory = d_head * d_head * n_heads * n_threads * num_layers * 4  # bytes (fp32)
    hvn_memory_mb = hvn_memory / (1024 ** 2)

    # Transformer KV cache at various sequence lengths
    # KV cache = 2 * N * d_model * num_layers * 2 bytes (fp16)
    results = {"hvn_memory_mb": hvn_memory_mb, "transformer_kv_cache": {}}

    for N in [1000, 10000, 100000]:
        kv_cache = 2 * N * d_model * num_layers * 2  # fp16
        kv_cache_mb = kv_cache / (1024 ** 2)
        results["transformer_kv_cache"][N] = {
            "kv_cache_mb": kv_cache_mb,
            "ratio": hvn_memory_mb / kv_cache_mb if kv_cache_mb > 0 else 0,
        }

    results["num_patches"] = num_patches
    results["d_model"] = d_model
    results["num_layers"] = num_layers
    results["n_heads"] = n_heads
    results["n_threads"] = n_threads

    return results


def run_full_evaluation(
    model: MultimodalFusionHVN,
    images: torch.Tensor,
    text_ids: torch.Tensor,
    tokenizer,
    device: torch.device,
    top_k: int = 5,
) -> dict:
    """Run the full multimodal evaluation suite.

    Parameters
    ----------
    model : MultimodalFusionHVN
        Multimodal model.
    images : torch.Tensor
        Test images.
    text_ids : torch.Tensor
        Text token IDs.
    tokenizer : object
        Tokenizer.
    device : torch.device
        Device.
    top_k : int
        K for retrieval Recall@K.

    Returns
    -------
    results : dict
        Full evaluation results.
    """
    print("=" * 60)
    print("MULTIMODAL EVALUATION (Issue #27)")
    print("=" * 60)

    results = {}

    # 1. Captioning
    print("\n1. Image Captioning Evaluation")
    print("-" * 40)
    cap_results = evaluate_captioning(model, images, text_ids, tokenizer, device)
    results["captioning"] = cap_results
    print(f"   Captioning PPL: {cap_results['ppl']:.2f}")
    for cap in cap_results["captions"][:3]:
        print(f"   Image {cap['image_idx']}: gen='{cap['generated'][:50]}...' gt='{cap['ground_truth'][:50]}...'")

    # 2. VQA
    print("\n2. Visual Question Answering Evaluation")
    print("-" * 40)
    # Use first half of text as "question", second half as "answer"
    S = text_ids.size(1)
    mid = S // 2
    questions = text_ids[:, :mid]
    answers = text_ids[:, mid:]
    vqa_results = evaluate_vqa(model, images, questions, answers, tokenizer, device)
    results["vqa"] = vqa_results
    print(f"   VQA PPL: {vqa_results['ppl']:.2f}")
    for qa in vqa_results["qa_pairs"][:3]:
        print(f"   Q: '{qa['question'][:40]}...' A: '{qa['generated_answer'][:40]}...'")

    # 3. Retrieval
    print("\n3. Cross-Modal Retrieval Evaluation")
    print("-" * 40)
    ret_results = evaluate_retrieval(model, images, text_ids, device, top_k=top_k)
    results["retrieval"] = ret_results
    if "error" not in ret_results:
        print(f"   Text→Image Recall@1: {ret_results['t2i_recall@1']:.2%}")
        print(f"   Text→Image Recall@{top_k}: {ret_results['t2i_recall@k']:.2%}")
        print(f"   Image→Text Recall@1: {ret_results['i2t_recall@1']:.2%}")
        print(f"   Image→Text Recall@{top_k}: {ret_results['i2t_recall@k']:.2%}")
        print(f"   Mean diagonal similarity: {ret_results['mean_diag_similarity']:.4f}")
    else:
        print(f"   {ret_results['error']}")

    # 4. O(1) inference
    print("\n4. O(1) Inference Verification")
    print("-" * 40)
    o1_results = evaluate_o1_inference(model, images, device)
    results["o1_inference"] = o1_results
    for t in o1_results["timings"]:
        print(f"   {t['num_patches']:6d} patches: {t['ms_per_patch']:.3f} ms/patch "
              f"(total: {t['total_ms']:.1f} ms)")
    print(f"   O(1) confirmed: {o1_results['o1_confirmed']} "
          f"(max/min ratio: {o1_results.get('max_min_ratio', 'N/A'):.2f})")

    # 5. Memory
    print("\n5. Memory Comparison")
    print("-" * 40)
    mem_results = evaluate_memory(model, device)
    results["memory"] = mem_results
    print(f"   HVN Hamiltonian memory: {mem_results['hvn_memory_mb']:.4f} MB")
    for N, data in mem_results["transformer_kv_cache"].items():
        print(f"   Transformer KV cache @ N={N}: {data['kv_cache_mb']:.2f} MB "
              f"(HVN = {data['ratio']:.4%} of KV cache)")

    print("\n" + "=" * 60)
    return results


def main():
    parser = argparse.ArgumentParser(description="Multimodal evaluation for HVN (Issue #27)")
    parser.add_argument("--d_model", type=int, default=128, help="Model dimension")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of HVN blocks")
    parser.add_argument("--n_heads", type=int, default=4, help="Number of heads")
    parser.add_argument("--bond_dim", type=int, default=32, help="Bond dimension")
    parser.add_argument("--image_size", type=int, default=64, help="Image size")
    parser.add_argument("--patch_size", type=int, default=16, help="Patch size")
    parser.add_argument("--in_channels", type=int, default=3, help="Image channels")
    parser.add_argument("--vocab_size", type=int, default=100, help="Vocabulary size")
    parser.add_argument("--num_pairs", type=int, default=32, help="Number of image-text pairs")
    parser.add_argument("--text_seq_len", type=int, default=16, help="Text sequence length")
    parser.add_argument("--top_k", type=int, default=5, help="K for retrieval Recall@K")
    parser.add_argument("--n_threads", type=int, default=1, help="Hamiltonian threads per head")
    parser.add_argument("--rank_k", type=int, default=1, help="Rank-k entanglement update")
    parser.add_argument("--save", type=str, default=None, help="Save results to JSON file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    assert torch.cuda.is_available(), "CUDA is required — install GPU PyTorch!"

    # Build model
    model = HamiltonianVortexNetwork(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        bond_dimension=args.bond_dim,
        n_threads=args.n_threads,
        rank_k=args.rank_k,
    )

    vision_encoder = ImageEncoder(
        image_size=args.image_size,
        patch_size=args.patch_size,
        in_channels=args.in_channels,
        d_model=args.d_model,
    )

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
        num_pairs=args.num_pairs,
        image_size=args.image_size,
        in_channels=args.in_channels,
        text_vocab_size=args.vocab_size,
        text_seq_len=args.text_seq_len,
        seed=args.seed,
    )
    images = images.to(device)
    text_ids = text_ids.to(device)

    # Simple tokenizer for evaluation
    class SimpleTokenizer:
        def __init__(self, vocab_size):
            self.vocab_size = vocab_size
            self.id_to_char = {i: chr(97 + (i % 26)) for i in range(vocab_size)}

        def decode(self, ids):
            if isinstance(ids, torch.Tensor):
                ids = ids.tolist()
            return "".join(self.id_to_char.get(i, "?") for i in ids)

    tokenizer = SimpleTokenizer(args.vocab_size)

    # Run evaluation
    results = run_full_evaluation(
        multimodal, images, text_ids, tokenizer, device, top_k=args.top_k,
    )

    # Save results
    if args.save:
        # Convert tensors to floats for JSON
        def make_json_safe(obj):
            if isinstance(obj, torch.Tensor):
                return obj.item() if obj.numel() == 1 else obj.tolist()
            elif isinstance(obj, dict):
                return {k: make_json_safe(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [make_json_safe(v) for v in obj]
            elif isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
                return None
            return obj

        safe_results = make_json_safe(results)
        with open(args.save, "w") as f:
            json.dump(safe_results, f, indent=2)
        print(f"\n✓ Results saved to {args.save}")


if __name__ == "__main__":
    main()