"""EdgeHVN hybrid grounded continue-FT — Phase A4 (token-recall).

Short, targeted continue-fine-tune that teaches the hybrid's **fresh attention
layers** the one skill the A5 gate (``eval/grounded_qa.py``) measures: *given a
passage + a question, emit the answer span that is in the passage*. Pure DeltaNet
recurrence structurally cannot recall exact tokens; the 2 sliding-attention
layers can. This run proves it.

It warm-starts the hybrid's 10 DeltaNet layers from a pure-HVN Chinchilla-floor
trunk (``--hybrid-trunk-init``), leaves the 2 attention layers at fresh init, and
fine-tunes on grounded-QA (context->answer) data rendered in the gate's *exact*
open-book chat format with loss only on the answer turn. An optional
``--chat-mix`` blends in OpenAssistant conversations so the pure-base trunk also
learns chat format (the gate emits HTML garbage otherwise — the trunk was never
chat-FT'd).

Reuses (no new training infra):
  * ``train_edge_conversational`` — ``_save_edge_checkpoint`` (block_plan
    metadata), ``load_hybrid_trunk_into_model`` (filtered trunk warm-start).
  * ``finetune_edge_conversational`` — ``InstructionDataset`` (chat format +
    loss masking), ``build_oasst_conversations``, the chat constants
    (USER_PREFIX/ASSISTANT_PREFIX/EOS_ID — identical to the gate's).
  * The train script's proven ordering: build hybrid -> apply ternary ->
    trunk-init -> set triton-scan + megakernel-scan -> torch.compile.

The A5 gate runs on the saved checkpoint with NO serving lift (the gate uses
full-sequence ``model(x)`` re-feed generation, not a KV cache).

Usage (the canonical A4 run, ~5-10K steps)::

    python continue_ft_hybrid.py --fast --max-steps 8000 \\
        --hybrid-trunk-init checkpoints/edge_chinchilla83m/edge_conversational_step51000.pt \\
        --checkpoint-dir checkpoints/hybrid_a4

Then the gate::

    python -m eval.grounded_qa --checkpoint checkpoints/hybrid_a4/hybrid_a4_step8000.pt
    # pass = retrieval lift > 0 (pure HVN = 0/5; 50K trunk = 1/5 partial)
"""

from __future__ import annotations

import argparse
import csv
import itertools
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

from train_edge_conversational import (
    GPT2_VOCAB_SIZE,
    _save_edge_checkpoint,
    load_hybrid_trunk_into_model,
)
from finetune_edge_conversational import (
    InstructionDataset,
    build_oasst_conversations,
    USER_PREFIX,
    ASSISTANT_PREFIX,
    EOS_ID,
    GroundedQADataset,
    OPEN_BOOK_INSTRUCTION,
)
from hvn.edge.model import EdgeHVN
from hvn.edge.ternary import apply_ternary_to_model

# GroundedQADataset / OPEN_BOOK_INSTRUCTION moved to finetune_edge_conversational.py
# (2026-07-15, issue #108) so that module can reuse them too without a circular
# import (it already imports InstructionDataset/build_oasst_conversations FROM
# here). Re-imported above so `from continue_ft_hybrid import GroundedQADataset`
# (e.g. tests/test_continue_ft_hybrid.py) keeps working unchanged.


# ---------------------------------------------------------------------------
# 2. Model build (hybrid arch + ternary + trunk warm-start + scan + compile)
# ---------------------------------------------------------------------------

def build_hybrid_model(trunk_ckpt_path: str, attn_layers: list[int],
                       attn_window: int, device: torch.device) -> EdgeHVN:
    """Build the hybrid EdgeHVN, apply ternary, warm-start the DeltaNet trunk.

    Architecture dims (d_model/num_layers/n_heads/vocab) are read from the trunk
    checkpoint so the hybrid exactly matches the pretrain arch. The 2 attention
    positions are fresh-init; the 10 DeltaNet positions are copied from the trunk.
    """
    ckpt = torch.load(trunk_ckpt_path, map_location=device, weights_only=False)
    num_layers = ckpt["num_layers"]
    block_plan = [
        "transformer" if i in set(attn_layers) else "deltanet"
        for i in range(num_layers)
    ]
    for li in attn_layers:
        if not 0 <= li < num_layers:
            raise SystemExit(f"--attn-layers position {li} out of range for "
                             f"num_layers={num_layers}")
    print(f"  Hybrid block_plan: {block_plan} (window={attn_window})")

    model = EdgeHVN(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=num_layers,
        n_heads=ckpt.get("n_heads", 4),
        use_deltanet=False,           # block_plan drives dispatch, not use_deltanet
        deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        block_plan=block_plan,
        attn_window=attn_window,
    )
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Hybrid params: {param_count:,}")

    # Apply ternary BitNet b1.58 (match the trunk; excludes beta/alpha/decoder).
    all_linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    target_names = [n for n in all_linears if not any(p in n for p in exclude)]
    apply_ternary_to_model(model, target_names=target_names)
    print(f"  Ternary applied ({len(target_names)} linears -> TernaryLinear STE)")

    # Warm-start the 10 DeltaNet layers from the trunk (2 attn layers stay fresh).
    trunk_step = load_hybrid_trunk_into_model(model, trunk_ckpt_path, device)
    print(f"  Trunk warm-started (trunk was at step {trunk_step}); "
          f"attention layers at fresh init.")
    return model, trunk_step


def enable_scan_flags(model: EdgeHVN, *, triton_scan: bool, megakernel_scan: bool):
    """Set triton-scan + megakernel-scan on each DeltaNet block (mirror train script)."""
    blocks = (model.blocks if getattr(model, "blocks", None) is not None
              else ([model.shared_block.block]
                    if getattr(model, "shared_block", None) is not None else []))
    if triton_scan:
        n = 0
        for blk in blocks:
            if hasattr(blk, "use_triton_scan"):
                blk.use_triton_scan = True
                n += 1
        print(f"  Fused Triton scan enabled on {n} DeltaNet block(s) (#83).")
    if megakernel_scan and triton_scan:
        n = 0
        for blk in blocks:
            if hasattr(blk, "use_megakernel_scan"):
                blk.use_megakernel_scan = True
                n += 1
        print(f"  Design B blockwise-recompute backward enabled on {n} "
              f"DeltaNet block(s) (#96) -- ~1.05x faster step.")
    elif megakernel_scan and not triton_scan:
        print("  WARNING: --megakernel-scan requires --triton-scan; ignoring.")


# ---------------------------------------------------------------------------
# 3. Continue-FT loop (step-based, loss-masked, deferred loss accumulator)
# ---------------------------------------------------------------------------

def continue_ft(model, dataloader, sampler, device, *, lr, max_steps, grad_accum,
                save_every, checkpoint_dir, log_path, warmup_frac=0.03,
                optimizer_name="adamw"):
    print("\n" + "=" * 70)
    print("A4 — HYBRID GROUNDED CONTINUE-FT")
    print("=" * 70)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,} | max_steps: {max_steps} | lr: {lr} | "
          f"grad_accum: {grad_accum}")

    # Issue #134: the parameter-free path. Prodigy beat a hand-tuned LR by 0.18
    # nats on exactly this kind of continue-FT, at a step size 30x smaller than
    # the hand-set one — and every FT in this repo, including the one that
    # produced the 35.5% recall number, ran at the hand-set value.
    is_prodigy = optimizer_name in ("prodigy", "muon-prodigy")
    if is_prodigy:
        from hvn.optimizers import build_optimizer
        optimizer, desc = build_optimizer(model, optimizer_name, lr=1.0,
                                          weight_decay=0.01)
        # No warmup, no cosine: Prodigy's d-ramp IS the warmup, and a decay
        # schedule fights an estimator that is monotone by construction.
        warmup_steps = 0
        scheduler = None
        print(f"  Optimizer: {desc}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                      fused=(device.type == "cuda"))
        warmup_steps = max(1, int(max_steps * warmup_frac))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, max_steps - warmup_steps))
        print(f"  Optimizer: AdamW, warmup {warmup_steps}, cosine over {max_steps}")
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    os.makedirs(checkpoint_dir, exist_ok=True)
    csv_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if os.path.getsize(log_path) == 0:
        writer.writerow(["step", "loss", "ppl", "lr", "timestamp"])
    csv_file.flush()

    model.train()
    global_step = 0
    micro = 0
    start = time.time()
    optimizer.zero_grad()
    accum_loss_t = torch.zeros((), device=device)
    n_steps_accum = 0  # optimizer steps since last log (to average the loss)
    data_iter = iter(dataloader)

    while global_step < max_steps:
        try:
            batch = next(data_iter)
        except StopIteration:
            # Epoch boundary: re-create the iterator. WeightedRandomSampler
            # reshuffles fresh each iter() call, so the next epoch is a new mix.
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Packed rows (Issue #132) carry a third element, doc_ids.
        if len(batch) == 3:
            x, y, doc_ids = batch
            doc_ids = doc_ids.to(device, non_blocking=True)
        else:
            x, y = batch
            doc_ids = None

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x) if doc_ids is None else model(x, doc_ids=doc_ids)
            loss = criterion(logits.reshape(-1, GPT2_VOCAB_SIZE), y.reshape(-1))
            loss = loss / grad_accum
        loss.backward()
        accum_loss_t += loss.detach()
        micro += 1

        if micro >= grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if global_step < warmup_steps:
                scale = (global_step + 1) / warmup_steps
                for pg in optimizer.param_groups:
                    pg["lr"] = lr * scale
            optimizer.step()
            if scheduler is not None and global_step >= warmup_steps:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            micro = 0
            n_steps_accum += 1

            if global_step % 20 == 0:
                # accum_loss_t holds the SUM of per-step mean losses since the
                # last log (loss/grad_accum added per micro-batch). Divide by the
                # optimizer-step count to get the true mean loss — otherwise the
                # displayed value is ~20x inflated and ppl=exp(loss) is nonsense.
                mean_loss = float(accum_loss_t.item()) / max(1, n_steps_accum)
                accum_loss_t.zero_()
                n_steps_accum = 0
                cur_lr = optimizer.param_groups[0]["lr"]
                ppl = math.exp(min(mean_loss, 20))
                print(f"  Step {global_step:5d}/{max_steps} | loss {mean_loss:.4f} | "
                      f"ppl {ppl:.2f} | lr {cur_lr:.2e} | "
                      f"{(time.time()-start)/max(1,global_step):.2f}s/step", flush=True)
                writer.writerow([global_step, f"{mean_loss:.6f}", f"{ppl:.4f}",
                                 f"{cur_lr:.8f}", time.strftime("%Y-%m-%d %H:%M:%S")])
                csv_file.flush()

            if save_every and global_step % save_every == 0:
                out = os.path.join(checkpoint_dir, f"hybrid_a4_step{global_step}.pt")
                _save_edge_checkpoint(model, out, global_step=global_step)
                print(f"  ✓ Saved {out}", flush=True)

    # Final save.
    out = os.path.join(checkpoint_dir, f"hybrid_a4_step{global_step}.pt")
    _save_edge_checkpoint(model, out, global_step=global_step)
    print(f"\n  ✓ A4 complete: {global_step} steps, "
          f"{(time.time()-start)/60:.1f} min — saved {out}")
    csv_file.close()


# ---------------------------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hybrid-trunk-init", type=str,
                    default="checkpoints/edge_chinchilla83m/edge_conversational_step51000.pt",
                    help="pure-HVN trunk to warm-start the 10 DeltaNet layers from")
    ap.add_argument("--grounded-data", type=str,
                    default="data/grounded_qa_train.jsonl",
                    help="JSONL of {passage,question,answer} (see scripts/build_grounded_qa_data.py)")
    ap.add_argument("--attn-layers", type=str, default="3,10",
                    help="0-indexed layer positions for the 2 attention blocks")
    ap.add_argument("--attn-window", type=int, default=4096)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--pack", action=argparse.BooleanOptionalAction, default=True,
                    help="padding-free sequence packing (Issue #132), ON BY DEFAULT -- "
                         "the grounded-QA corpus is 83%% padding at seq_len=512, so this "
                         "is a large wall-clock win for the SAME training (same real "
                         "tokens, ~6 examples per row). Documents are isolated inside a "
                         "row (DeltaNet state reset + conv taps + attention mask), so it "
                         "is exact, not an approximation. Measured 5.15x faster on "
                         "grounded-QA and 1.45x on OASST, with held-out loss within 0.4%% "
                         "of the padded control -- see docs/state.md. Pass --no-pack to "
                         "restore the old padded path.")
    ap.add_argument("--pack-seed", type=int, default=0,
                    help="seed for the packing shuffle (--pack)")
    ap.add_argument("--pack-strategy", type=str, default="shuffled_first_fit",
                    choices=["shuffled_first_fit", "first_fit_decreasing"],
                    help="bin-packing policy for --pack")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5,
                    help="Only used by --optimizer adamw. The prodigy variants "
                         "ignore it entirely — that is the point (#134).")
    ap.add_argument("--optimizer", type=str, default="adamw",
                    choices=["adamw", "prodigy", "muon-prodigy"],
                    help="Issue #134: 'prodigy' takes NO learning rate and beat "
                         "a hand-tuned one by 0.18 nats on a 5,000-step "
                         "continue-FT, at a step size 30x smaller. Default stays "
                         "adamw so existing invocations are unchanged.")
    ap.add_argument("--max-steps", type=int, default=8000)
    ap.add_argument("--save-every", type=int, default=1000)
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints/hybrid_a4")
    ap.add_argument("--chat-mix", type=float, default=0.3,
                    help="fraction of batches from OpenAssistant chat (0 = pure grounded-QA)")
    ap.add_argument("--max-conversations", type=int, default=None,
                    help="cap OASST conversations (e.g. --smoke uses 200)")
    ap.add_argument("--hf-cache-dir", type=str, default=None)
    ap.add_argument("--fast", action="store_true",
                    help="bf16 + TF32 + torch.compile + triton-scan + megakernel-scan")
    ap.add_argument("--smoke", action="store_true",
                    help="tiny run (few steps) — pipeline validation on CPU/GPU")
    args = ap.parse_args()

    if sys.platform == "win32":
        os.system("chcp 65001 >nul 2>&1")

    # triton-scan + megakernel-scan are unconditionally on (2026-08-01): both are
    # validated identity-preserving vs the exact sequential scan (tests/test_issue83.py
    # 7/7, rel-diff < 2e-3; megakernel backward "bit-identical loss", grad diff ~5e-8,
    # scripts/ab_megakernel.py) AND, contrary to earlier assumption, MORE numerically
    # reliable than the default eager `_scan_chunked` path they replace -- that path's
    # own docstring documents it can be "silently wrong on decayed channels at large
    # chunks / hard decay" (see docs/state.md, Issue #122). There is no real accuracy
    # tradeoff to justify gating this behind an opt-in flag; only real known caveat is
    # requiring a Triton-capable CUDA GPU (gracefully falls back via _HAS_TRITON_SCAN
    # if unavailable). --fast now controls ONLY torch.compile, a genuinely separate
    # tradeoff (compile-time warmup cost, less proven interaction with the loop-aware
    # training path in continue_ft_looped.py).
    use_fast = args.fast
    triton_scan = True
    megakernel_scan = True
    compile_model = use_fast

    if use_fast and torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("[--fast] bf16 + TF32 + torch.compile (triton-scan + megakernel-scan are always on)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    from transformers import GPT2Tokenizer
    tok_kwargs = {"cache_dir": args.hf_cache_dir} if args.hf_cache_dir else {}
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", **tok_kwargs)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Data ---
    print("\n" + "=" * 70)
    print("DATA — grounded-QA (+ optional OpenAssistant chat mix)")
    print("=" * 70)
    import json
    records = [json.loads(line) for line in open(args.grounded_data, encoding="utf-8")]
    print(f"  Grounded-QA: {len(records)} records from {args.grounded_data}")
    grounded_ds = GroundedQADataset(tokenizer, records, seq_len=args.seq_len,
                                    pad=not args.pack)

    chat_ds = None
    if args.chat_mix > 0:
        try:
            max_conv = 200 if args.smoke else args.max_conversations
            convos = build_oasst_conversations(max_conversations=max_conv)
            chat_ds = InstructionDataset(tokenizer, convos, seq_len=args.seq_len,
                                         pad=not args.pack)
            print(f"  Chat mix: {len(chat_ds)} OASST examples (target chat fraction={args.chat_mix})")
        except Exception as e:
            print(f"  Chat mix DISABLED (OASST cache unavailable: {e})")
            print(f"  -> running pure grounded-QA. Set --chat-mix 0 to silence this.")
            chat_ds = None

    if args.pack:
        # Issue #132: pack each corpus independently so a row never mixes
        # corpora -- that keeps the chat_mix fraction meaning what the flag says.
        from hvn.core.packing import PackedDataset
        print("\n  Packing (Issue #132) — padding-free rows:")
        print("    grounded:", end=" ")
        grounded_ds = PackedDataset(grounded_ds.examples, seq_len=args.seq_len,
                                    pad_id=EOS_ID, seed=args.pack_seed,
                                    strategy=args.pack_strategy)
        if chat_ds is not None:
            print("    chat:", end=" ")
            chat_ds = PackedDataset(chat_ds.examples, seq_len=args.seq_len,
                                    pad_id=EOS_ID, seed=args.pack_seed,
                                    strategy=args.pack_strategy)

    # Combine via a WeightedRandomSampler so P(chat)=chat_mix when chat_ds exists.
    if chat_ds is not None and args.chat_mix > 0:
        dataset = ConcatDataset([grounded_ds, chat_ds])
        n_g, n_c = len(grounded_ds), len(chat_ds)
        p_chat = args.chat_mix
        weights = ([1.0 / n_g * (1 - p_chat)] * n_g) + ([1.0 / n_c * p_chat] * n_c) \
            if n_c > 0 else [1.0 / n_g] * n_g
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset),
                                        replacement=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                drop_last=True, pin_memory=(device.type == "cuda"))
        print(f"  Combined dataset: {len(dataset)} examples, sampled to "
              f"{p_chat:.0%} chat / {1-p_chat:.0%} grounded")
    else:
        dataset = grounded_ds
        sampler = None
        dataloader = DataLoader(grounded_ds, batch_size=args.batch_size, shuffle=True,
                                drop_last=True, pin_memory=(device.type == "cuda"))

    # --- Model ---
    print("\n" + "=" * 70)
    print("MODEL — hybrid arch + trunk warm-start")
    print("=" * 70)
    if not os.path.exists(args.hybrid_trunk_init):
        print(f"❌ Trunk checkpoint not found: {args.hybrid_trunk_init}")
        sys.exit(1)
    model, trunk_step = build_hybrid_model(
        args.hybrid_trunk_init,
        attn_layers=sorted(set(int(x) for x in args.attn_layers.split(","))),
        attn_window=args.attn_window, device=device)
    model = model.to(device)

    enable_scan_flags(model, triton_scan=triton_scan, megakernel_scan=megakernel_scan)
    if compile_model:
        print("\n  Compiling model with torch.compile (first steps will be slow)...")
        model = torch.compile(model)

    # --- Continue-FT ---
    max_steps = 20 if args.smoke else args.max_steps
    log_path = os.path.join(args.checkpoint_dir, "a4_ft_log.csv")
    continue_ft(model, dataloader, sampler, device,
                lr=args.lr, max_steps=max_steps, grad_accum=args.grad_accum,
                save_every=args.save_every, checkpoint_dir=args.checkpoint_dir,
                log_path=log_path, optimizer_name=args.optimizer)

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()