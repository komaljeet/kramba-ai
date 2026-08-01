"""continue_ft_looped.py — loop-aware continued fine-tune of the real EdgeHVN
hybrid checkpoint (Issue #122, stage 2).

Stage 1 (scripts/train_looped_sandbox.py) tested the loop-aware training
recipe (random loop count k per step + deep supervision across every loop's
prediction) on a disposable ~780K-param model and got an ambiguous result
(a memorization-regime confound -- the tiny repeated corpus was fully
solvable at k=1, so there was no training pressure for extra loops to do
anything useful). That confound does not apply here: `data/
grounded_qa_expanded.jsonl` (5,240 real examples) is real, non-repeated
data the model cannot simply memorize at k=1.

This is a CONTINUED fine-tune, not a retrain: it warm-starts directly from
`checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt` (the same checkpoint the
scripts/sweep_looped_early_exit.py grid search ran on, and the one the
35.5%/8.5% recall baseline was originally measured on) via
`train_edge_conversational.load_checkpoint_for_resume` -- no fresh trunk
warm-start, no architecture change. Only the training OBJECTIVE changes:
each step samples k ~ Uniform[1, max_loops] and trains on the loop-aware,
deep-supervised loss (hvn.edge.looped.looped_forward_train) instead of a
single plain forward() pass, teaching the SAME weights to be a genuine
multi-step refiner instead of leaving them untrained for looping (which is
what scripts/sweep_looped_early_exit.py proved does not work).

The fused Triton scan + megakernel backward (#83/#96) are always on (no --fast
flag needed) -- verified identity-preserving AND more numerically reliable
than the default eager scan they replace (see docs/state.md, Issue #122);
measured ~42x faster in this script. torch.compile is intentionally NOT used:
looped_forward_train() bypasses model.forward() to implement the custom
multi-loop logic, so compiling the top-level model would compile a path
that's never invoked here -- a no-op, not a speedup.

Usage::

    python continue_ft_looped.py --max-steps 2000 --max-loops 4 \\
        --checkpoint-init checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt \\
        --checkpoint-dir checkpoints/hybrid_looped_ft
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from train_edge_conversational import GPT2_VOCAB_SIZE, _save_edge_checkpoint, load_checkpoint_for_resume
from finetune_edge_conversational import GroundedQADataset
from continue_ft_hybrid import enable_scan_flags
from hvn.edge.looped import looped_forward_train


def continue_ft_looped(model, dataloader, device, *, lr, max_steps, max_loops,
                       grad_accum, save_every, checkpoint_dir, log_path, warmup_frac=0.03):
    print("\n" + "=" * 70)
    print("LOOP-AWARE CONTINUE-FT (Issue #122 stage 2)")
    print("=" * 70)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,} | max_steps: {max_steps} | max_loops: {max_loops} | "
         f"lr: {lr} | grad_accum: {grad_accum}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                  fused=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    warmup_steps = max(1, int(max_steps * warmup_frac))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, max_steps - warmup_steps))

    os.makedirs(checkpoint_dir, exist_ok=True)
    csv_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if os.path.getsize(log_path) == 0:
        writer.writerow(["step", "k", "loss", "ppl", "lr", "timestamp"])
    csv_file.flush()

    model.train()
    global_step = 0
    micro = 0
    start = time.time()
    optimizer.zero_grad()
    accum_loss_t = torch.zeros((), device=device)
    n_steps_accum = 0
    data_iter = iter(dataloader)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    while global_step < max_steps:
        try:
            x, y = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            x, y = next(data_iter)

        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        position_ids = torch.arange(x.size(1), device=device).unsqueeze(0).expand(x.size(0), -1)
        k = int(torch.randint(1, max_loops + 1, (1,)).item())

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            per_loop_logits = looped_forward_train(model, x, position_ids, k, deep_supervise=True)
            loss = sum(criterion(lg.reshape(-1, GPT2_VOCAB_SIZE), y.reshape(-1))
                      for lg in per_loop_logits) / len(per_loop_logits)
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
            if global_step >= warmup_steps:
                scheduler.step()
            optimizer.zero_grad()
            global_step += 1
            micro = 0
            n_steps_accum += 1

            if global_step % 20 == 0:
                mean_loss = float(accum_loss_t.item()) / max(1, n_steps_accum)
                accum_loss_t.zero_()
                n_steps_accum = 0
                cur_lr = optimizer.param_groups[0]["lr"]
                ppl = math.exp(min(mean_loss, 20))
                mem_str = ""
                if device.type == "cuda":
                    peak_alloc = torch.cuda.max_memory_allocated(device) / (1024 ** 2)
                    peak_reserved = torch.cuda.max_memory_reserved(device) / (1024 ** 2)
                    mem_str = f" | peak_alloc {peak_alloc:.0f}MiB | peak_reserved {peak_reserved:.0f}MiB"
                print(f"  Step {global_step:5d}/{max_steps} | k~U[1,{max_loops}] | "
                     f"loss {mean_loss:.4f} | ppl {ppl:.2f} | lr {cur_lr:.2e} | "
                     f"{(time.time()-start)/max(1,global_step):.2f}s/step{mem_str}", flush=True)
                writer.writerow([global_step, k, f"{mean_loss:.6f}", f"{ppl:.4f}",
                                 f"{cur_lr:.8f}", time.strftime("%Y-%m-%d %H:%M:%S")])
                csv_file.flush()

            if save_every and global_step % save_every == 0:
                out = os.path.join(checkpoint_dir, f"hybrid_looped_step{global_step}.pt")
                _save_edge_checkpoint(model, out, global_step=global_step)
                print(f"  Saved {out}", flush=True)

    out = os.path.join(checkpoint_dir, f"hybrid_looped_step{global_step}.pt")
    _save_edge_checkpoint(model, out, global_step=global_step)
    print(f"\n  Loop-aware continue-FT complete: {global_step} steps, "
         f"{(time.time()-start)/60:.1f} min -- saved {out}")
    csv_file.close()
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint-init", type=str,
                    default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt",
                    help="existing hybrid checkpoint to warm-start from (NOT a fresh trunk)")
    ap.add_argument("--grounded-data", type=str, default="data/grounded_qa_expanded.jsonl")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--max-loops", type=int, default=4)
    ap.add_argument("--save-every", type=int, default=500)
    ap.add_argument("--checkpoint-dir", type=str, default="checkpoints/hybrid_looped_ft")
    args = ap.parse_args()

    if sys.platform == "win32":
        os.system("chcp 65001 >nul 2>&1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  device: {device}")

    from transformers import GPT2Tokenizer
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    print("\n" + "=" * 70)
    print("DATA — grounded-QA (real, non-repeated -- no memorization-regime confound)")
    print("=" * 70)
    records = [json.loads(line) for line in open(args.grounded_data, encoding="utf-8")]
    print(f"  Grounded-QA: {len(records)} records from {args.grounded_data}")
    dataset = GroundedQADataset(tokenizer, records, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, pin_memory=(device.type == "cuda"))

    print("\n" + "=" * 70)
    print("MODEL — warm-start from EXISTING hybrid checkpoint (continued-FT, not fresh trunk)")
    print("=" * 70)
    if not os.path.exists(args.checkpoint_init):
        print(f"Checkpoint not found: {args.checkpoint_init}")
        sys.exit(1)
    model, base_step, _ = load_checkpoint_for_resume(args.checkpoint_init, device)
    print(f"  Warm-started from step {base_step} of {args.checkpoint_init}")
    model = model.to(device)
    # triton-scan + megakernel-scan are unconditionally on (2026-08-01) -- both are
    # validated identity-preserving AND more numerically reliable than the default
    # eager scan path they replace (see docs/state.md, Issue #122). No accuracy
    # tradeoff to gate behind a flag; measured ~42x faster in this script alone.
    enable_scan_flags(model, triton_scan=True, megakernel_scan=True)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    log_path = os.path.join(args.checkpoint_dir, "looped_ft_log.csv")
    out_ckpt = continue_ft_looped(
        model, dataloader, device,
        lr=args.lr, max_steps=args.max_steps, max_loops=args.max_loops,
        grad_accum=args.grad_accum, save_every=args.save_every,
        checkpoint_dir=args.checkpoint_dir, log_path=log_path,
    )
    print(f"\nFinal checkpoint: {out_ckpt}")

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
