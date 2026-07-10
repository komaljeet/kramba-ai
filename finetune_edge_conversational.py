"""
EdgeHVN Conversational Fine-Tuning (Issue #72 — fine-tune phase)
================================================================

Instruction / chat fine-tuning of a pre-trained EdgeHVN checkpoint on
OpenAssistant (oasst1), continuing from the Run 2 base
(``checkpoints/edge_conversational/edge_conversational_step10000.pt``).

Key differences from pre-training (``train_edge_conversational.py``):

  1. **Instruction format** — oasst1 message *trees* are reconstructed into
     multi-turn conversations (root prompt → highest-ranked assistant reply →
     …), then rendered with a GPT-2-friendly chat template (no new vocab):

         <|user|>\n{user text}\n<|assistant|>\n{assistant text}<|endoftext|>

  2. **Loss masking** — only the assistant response tokens (and the trailing
     end-of-text) contribute to the loss (``ignore_index=-100``). The model
     learns to *respond*, not to parrot the user's turn or the role markers.

  3. **Full fine-tuning, low LR** — carries the pre-trained ternary + DeltaNet
     weights forward and continues training with a small cosine-decayed LR.
     (Not LoRA: the ternary-STE latent weights make full FT both simpler and
     cleaner for keeping the all-ternary inference story.)

Usage
-----
    # Full run from the Run 2 base
    python finetune_edge_conversational.py

    # Fast smoke test (tiny subset, 1 epoch)
    python finetune_edge_conversational.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Reuse the pre-training helpers so the checkpoint format stays identical.
from train_edge_conversational import (
    GPT2_VOCAB_SIZE,
    _save_edge_checkpoint,
    load_checkpoint_for_resume,
)

# GPT-2 end-of-text id (also used as pad); role markers are plain text.
EOS_ID = 50256
USER_PREFIX = "<|user|>\n"
ASSISTANT_PREFIX = "<|assistant|>\n"


# ---------------------------------------------------------------------------
# 1. OpenAssistant conversation reconstruction
# ---------------------------------------------------------------------------

def build_oasst_conversations(max_conversations: int | None = None,
                              lang: str = "en") -> list[list[tuple[str, str]]]:
    """Reconstruct oasst1 message trees into linear multi-turn conversations.

    For each tree, walk from the root prompter message downward, always taking
    the highest-ranked (rank 0 = best) child reply at each step. This yields a
    single best-path conversation of alternating (role, text) turns.

    Returns a list of conversations, each a list of ``(role, text)`` tuples
    where role is "prompter" or "assistant".

    Reads the locally-cached oasst1 arrow file directly with pyarrow. We avoid
    ``datasets.load_dataset`` on purpose: it segfaults on this Windows setup
    (same background-thread/native issue that forced local-file loading during
    pre-training), and its network path also hits the environment's TLS
    interception. The arrow cache from a prior run is all we need.
    """
    import glob
    import pyarrow as pa

    cache_root = os.path.expanduser(
        "~/.cache/huggingface/datasets/OpenAssistant___oasst1")
    matches = glob.glob(os.path.join(cache_root, "**", "oasst1-train.arrow"),
                        recursive=True)
    if not matches:
        raise FileNotFoundError(
            f"Cached oasst1 arrow not found under {cache_root}. "
            "load_dataset segfaults in this environment, so a prior arrow "
            "cache is required (it was created when oasst1.txt was first built).")
    path = matches[0]
    print(f"  Reading oasst1 from cache: {path}")
    try:
        tbl = pa.ipc.open_stream(pa.memory_map(path, "r")).read_all()
    except Exception:
        tbl = pa.ipc.open_file(pa.memory_map(path, "r")).read_all()

    cols = {c: tbl.column(c).to_pylist() for c in
            ("message_id", "parent_id", "role", "rank", "lang", "deleted", "text")}

    # Index messages by id and group children by parent.
    by_id: dict[str, dict] = {}
    children: dict[str | None, list[dict]] = {}
    for i in range(tbl.num_rows):
        if cols["deleted"][i]:
            continue
        if lang and cols["lang"][i] != lang:
            continue
        row = {"message_id": cols["message_id"][i],
               "parent_id": cols["parent_id"][i],
               "role": cols["role"][i],
               "rank": cols["rank"][i],
               "text": cols["text"][i]}
        by_id[row["message_id"]] = row
        children.setdefault(row["parent_id"], []).append(row)

    def _rank_key(m):
        r = m.get("rank")
        bad = r is None or (isinstance(r, float) and math.isnan(r))
        return (bad, r if not bad else 0.0)  # lower rank = better; bad ranks last

    def best_child(parent_id: str | None) -> dict | None:
        kids = children.get(parent_id, [])
        return min(kids, key=_rank_key) if kids else None

    conversations: list[list[tuple[str, str]]] = []
    # Roots: prompter messages with no parent.
    roots = [m for m in children.get(None, []) if m.get("role") == "prompter"]
    for root in roots:
        convo: list[tuple[str, str]] = []
        node: dict | None = root
        while node is not None:
            text = (node.get("text") or "").strip()
            if not text:
                break
            convo.append((node["role"], text))
            node = best_child(node["message_id"])
        # Keep conversations that have at least one prompter→assistant pair.
        if len(convo) >= 2:
            conversations.append(convo)
        if max_conversations is not None and len(conversations) >= max_conversations:
            break

    print(f"  Reconstructed {len(conversations):,} conversations "
          f"(lang={lang}) from {len(by_id):,} messages")
    return conversations


# ---------------------------------------------------------------------------
# 2. Instruction dataset with chat template + loss masking
# ---------------------------------------------------------------------------

class InstructionDataset(Dataset):
    """Tokenized, loss-masked instruction examples from oasst1 conversations.

    Each conversation is rendered with the chat template and tokenized once.
    Labels are set to ``-100`` for everything except assistant response tokens
    (plus the trailing EOS), so cross-entropy with ``ignore_index=-100`` trains
    the model only to produce assistant turns.

    Examples longer than ``seq_len`` are truncated (keeping the start);
    shorter ones are padded with EOS (label ``-100``).
    """

    def __init__(self, tokenizer, conversations: list[list[tuple[str, str]]],
                 seq_len: int = 512):
        self.seq_len = seq_len
        self.examples: list[tuple[np.ndarray, np.ndarray]] = []

        user_pref_ids = tokenizer.encode(USER_PREFIX)
        asst_pref_ids = tokenizer.encode(ASSISTANT_PREFIX)

        n_masked_only = 0
        for convo in conversations:
            ids: list[int] = []
            labels: list[int] = []
            for role, text in convo:
                if role == "prompter":
                    seg = user_pref_ids + tokenizer.encode(text + "\n")
                    ids.extend(seg)
                    labels.extend([-100] * len(seg))
                else:  # assistant — this is what we train on
                    ids.extend(asst_pref_ids)
                    labels.extend([-100] * len(asst_pref_ids))
                    ans = tokenizer.encode(text) + [EOS_ID]
                    ids.extend(ans)
                    labels.extend(ans)  # learned tokens

            # Truncate to seq_len + 1 (one extra for the causal shift).
            ids = ids[: seq_len + 1]
            labels = labels[: seq_len + 1]

            # Skip degenerate examples with no learnable (assistant) tokens.
            if not any(l != -100 for l in labels[1:]):
                n_masked_only += 1
                continue

            # Pad to seq_len + 1.
            pad = (seq_len + 1) - len(ids)
            if pad > 0:
                ids.extend([EOS_ID] * pad)
                labels.extend([-100] * pad)

            self.examples.append((
                np.asarray(ids, dtype=np.int64),
                np.asarray(labels, dtype=np.int64),
            ))

        print(f"  Built {len(self.examples):,} training examples "
              f"(seq_len={seq_len}, skipped {n_masked_only} with no assistant tokens)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, labels = self.examples[idx]
        # Causal shift: predict token t+1 from tokens ≤ t.
        x = torch.from_numpy(ids[:-1])
        y = torch.from_numpy(labels[1:])
        return x, y


# ---------------------------------------------------------------------------
# 3. Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(model, dataloader, device, *, lr, epochs, grad_accum,
             out_path, log_path, warmup_frac=0.03):
    print("\n" + "=" * 70)
    print("FINE-TUNING")
    print("=" * 70)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,} | epochs: {epochs} | lr: {lr} | "
          f"grad_accum: {grad_accum}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                  fused=(device.type == "cuda"))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)

    steps_per_epoch = max(1, len(dataloader) // grad_accum)
    total_steps = steps_per_epoch * epochs
    warmup_steps = max(1, int(total_steps * warmup_frac))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, total_steps - warmup_steps))
    print(f"  Optimizer steps: {total_steps} ({steps_per_epoch}/epoch), "
          f"warmup {warmup_steps}")

    csv_file = open(log_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if os.path.getsize(log_path) == 0:
        writer.writerow(["step", "epoch", "loss", "ppl", "lr", "timestamp"])
    csv_file.flush()

    model.train()
    global_step = 0
    start = time.time()
    optimizer.zero_grad()

    # On-device loss accumulator (deferred host sync): accumulate the
    # micro-batch losses on the GPU and sync to the host ONLY at log time
    # (every 20 opt steps). The old `accum_loss += loss.item()` forced a
    # GPU->CPU sync every micro-batch, stalling the async compute/H2D
    # pipeline. Mirrors the pretraining loop's deferred-accumulator pattern.
    accum_loss_t = torch.zeros((), device=device)

    for epoch in range(epochs):
        micro = 0
        for x, y in dataloader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = criterion(logits.reshape(-1, GPT2_VOCAB_SIZE), y.reshape(-1))
                loss = loss / grad_accum
            loss.backward()
            accum_loss_t += loss.detach()
            micro += 1

            if micro >= grad_accum:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                # Linear warmup, then cosine.
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

                if global_step % 20 == 0:
                    # Single host sync per log interval (was grad_accum syncs).
                    accum_loss = float(accum_loss_t.item())
                    accum_loss_t.zero_()
                    cur_lr = optimizer.param_groups[0]["lr"]
                    ppl = math.exp(min(accum_loss, 20))
                    print(f"  Step {global_step:5d}/{total_steps} (epoch {epoch}) | "
                          f"loss {accum_loss:.4f} | ppl {ppl:.2f} | lr {cur_lr:.2e} | "
                          f"{(time.time()-start)/max(1,global_step):.2f}s/step", flush=True)
                    writer.writerow([global_step, epoch, f"{accum_loss:.6f}",
                                     f"{ppl:.4f}", f"{cur_lr:.8f}",
                                     time.strftime("%Y-%m-%d %H:%M:%S")])
                    csv_file.flush()

        # Save at the end of each epoch.
        _save_edge_checkpoint(model, out_path, global_step=global_step)
        print(f"  ✓ Epoch {epoch} done — saved {out_path}")

    csv_file.close()
    print(f"\n  Fine-tuning complete: {global_step} steps, "
          f"{(time.time()-start)/60:.1f} min")


# ---------------------------------------------------------------------------
# 4. Quick generation smoke check
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_reply(model, tokenizer, device, user_msg, max_new=60):
    model.eval()
    prompt = USER_PREFIX + user_msg + "\n" + ASSISTANT_PREFIX
    ids = tokenizer.encode(prompt)
    x = torch.tensor([ids], device=device)
    for _ in range(max_new):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        nxt = int(logits[0, -1].argmax())
        if nxt == EOS_ID:
            break
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    out = tokenizer.decode(x[0, len(ids):].tolist())
    return out


def main():
    parser = argparse.ArgumentParser(description="EdgeHVN conversational fine-tuning")
    parser.add_argument("--base-checkpoint", type=str,
                        default="checkpoints/edge_conversational/edge_conversational_step10000.pt")
    parser.add_argument("--out", type=str, default="hvn_edge_chat.pt")
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    parser.add_argument("--smoke", action="store_true",
                        help="Tiny subset, 1 epoch — pipeline validation")
    args = parser.parse_args()

    if sys.platform == "win32":
        os.system("chcp 65001 >nul 2>&1")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from transformers import GPT2Tokenizer
    tok_kwargs = {"cache_dir": args.hf_cache_dir} if args.hf_cache_dir else {}
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2", **tok_kwargs)
    tokenizer.pad_token = tokenizer.eos_token

    # --- Data ---
    print("\n" + "=" * 70)
    print("DATA — OpenAssistant instruction format")
    print("=" * 70)
    max_conv = 200 if args.smoke else args.max_conversations
    conversations = build_oasst_conversations(max_conversations=max_conv)
    dataset = InstructionDataset(tokenizer, conversations, seq_len=args.seq_len)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                            drop_last=True, pin_memory=(device.type == "cuda"))

    # --- Model (load pre-trained base) ---
    print("\n" + "=" * 70)
    print("MODEL — loading base checkpoint")
    print("=" * 70)
    if not os.path.exists(args.base_checkpoint):
        print(f"❌ Base checkpoint not found: {args.base_checkpoint}")
        sys.exit(1)
    model, base_step, _ = load_checkpoint_for_resume(args.base_checkpoint, device)
    print(f"  Loaded {args.base_checkpoint} (pre-trained to step {base_step})")

    # --- Fine-tune ---
    epochs = 1 if args.smoke else args.epochs
    log_path = os.path.splitext(args.out)[0] + "_finetune_log.csv"
    finetune(model, dataloader, device,
             lr=args.lr, epochs=epochs, grad_accum=args.grad_accum,
             out_path=args.out, log_path=log_path)

    # --- Smoke generation ---
    print("\n  Sample replies from the fine-tuned model:")
    for q in ["What is the capital of France?", "Write a haiku about the sea."]:
        print(f"\n  USER: {q}")
        print(f"  ASSISTANT: {sample_reply(model, tokenizer, device, q)}")

    _save_edge_checkpoint(model, args.out)
    tok_out = os.path.splitext(args.out)[0] + "_tokenizer.json"
    tokenizer.save_pretrained(os.path.dirname(os.path.abspath(tok_out)) or ".")
    print(f"\n✓ Saved fine-tuned model: {args.out}")
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
