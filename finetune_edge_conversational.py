"""
EdgeHVN Conversational Fine-Tuning (Issue #72 — fine-tune phase)
================================================================

Instruction / chat fine-tuning of a pre-trained EdgeHVN checkpoint on
OpenAssistant (oasst1). Pass ``--base-checkpoint`` to point at a pre-trained
checkpoint — the old Run 2 default
(``checkpoints/edge_conversational/edge_conversational_step10000.pt``) was
undertrained and has been removed from LFS; use a Chinchilla-floor
checkpoint under ``checkpoints/edge_chinchilla83m/`` instead.

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
from torch.utils.data import (
    Dataset, DataLoader, ConcatDataset, WeightedRandomSampler,
)

# Reuse the pre-training helpers so the checkpoint format stays identical.
from train_edge_conversational import (
    GPT2_VOCAB_SIZE,
    _save_edge_checkpoint,
    load_checkpoint_for_resume,
)

from hvn.vocab import GPT2_EOS_ID, resolve_eos_id  # noqa: E402

# Legacy fallback for checkpoints written before eos_token_id was persisted.
# Every dataset and the generation loop resolve EOS from their own tokenizer or
# model instead (#136); this constant is the last resort, not the answer.
EOS_ID = GPT2_EOS_ID
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
                 seq_len: int = 512, pad: bool = True):
        # #136: EOS comes from THIS tokenizer, not the module constant. Local on
        # purpose, so every use below is covered rather than the ones someone
        # remembered to change. Training a 32K model with GPT-2's 50256 appended
        # would teach it to emit an id outside its own vocabulary.
        EOS_ID = resolve_eos_id(tokenizer=tokenizer)
        self.eos_id = EOS_ID
        self.seq_len = seq_len
        self.pad = pad
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

            # Pad to seq_len + 1 — unless packing (Issue #132), which keeps
            # examples variable-length so PackedDataset can concatenate them.
            if self.pad:
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
        if not self.pad:
            raise RuntimeError(
                f"{type(self).__name__} was built with pad=False (variable-length "
                "examples for Issue #132 packing) and cannot be iterated directly "
                "-- the default collate would fail on ragged rows. Wrap it: "
                "PackedDataset(ds.examples, seq_len=...)."
            )
        ids, labels = self.examples[idx]
        # Causal shift: predict token t+1 from tokens ≤ t.
        x = torch.from_numpy(ids[:-1])
        y = torch.from_numpy(labels[1:])
        return x, y


# ---------------------------------------------------------------------------
# 2b. Grounded-QA dataset (gate-format, loss-masked) — anti-forgetting replay
# ---------------------------------------------------------------------------
#
# Moved here from continue_ft_hybrid.py (2026-07-15, issue #108) so this
# module doesn't need to import from continue_ft_hybrid.py, which already
# imports FROM this module (InstructionDataset/build_oasst_conversations) --
# the reverse import would cycle. continue_ft_hybrid.py now imports these two
# names back from here instead.

# The gate's exact open-book prompt template (must match eval/grounded_qa.py
# and eval/synthetic_recall_probe.py).
OPEN_BOOK_INSTRUCTION = "Answer the question using only the context below.\n\n"


class GroundedQADataset(Dataset):
    """Tokenized, loss-masked grounded-QA examples in the gate's open-book format.

    Each (passage, question, answer) triple is rendered as::

        <USER> Answer the question using only the context below.

        Context: {passage}

        Question: {question}
        <ASSISTANT> {answer}<EOS>

    Labels are -100 everywhere except the answer tokens + trailing EOS, so CE
    with ``ignore_index=-100`` trains the model only to *produce the answer from
    the context* — the exact skill the gate measures. The prompt text matches
    ``eval/grounded_qa.py`` verbatim (same USER/ASSISTANT markers + instruction).
    """

    def __init__(self, tokenizer, records: list[dict], seq_len: int = 512,
                 pad: bool = True):
        # #136: EOS from this tokenizer, not the module constant. See
        # InstructionDataset for why this is a local.
        EOS_ID = resolve_eos_id(tokenizer=tokenizer)
        self.eos_id = EOS_ID
        self.seq_len = seq_len
        self.pad = pad
        self.examples: list[tuple[np.ndarray, np.ndarray]] = []

        user_pref_ids = tokenizer.encode(USER_PREFIX)
        asst_pref_ids = tokenizer.encode(ASSISTANT_PREFIX)
        instr_ids = tokenizer.encode(OPEN_BOOK_INSTRUCTION)

        n_truncated = 0
        for r in records:
            passage_ids = tokenizer.encode("Context: " + r["passage"] + "\n\n")
            question_ids = tokenizer.encode("Question: " + r["question"] + "\n")
            answer_ids = tokenizer.encode(r["answer"]) + [EOS_ID]

            # Prompt (masked) = USER_PREFIX + instruction + context + question + ASST_PREFIX
            prompt_ids = (user_pref_ids + instr_ids + passage_ids + question_ids
                          + asst_pref_ids)
            ids = prompt_ids + answer_ids
            labels = [-100] * len(prompt_ids) + list(answer_ids)

            if len(ids) > seq_len + 1:
                # The answer is at the END, so keep the TAIL (last seq_len+1
                # tokens): the answer + EOS + question + assistant marker + the
                # tail of the passage survive; the front of a long passage (and
                # possibly the "Answer using context" instruction) is dropped.
                # The loss is on the answer, which is never cut this way.
                ids = ids[-(seq_len + 1):]
                labels = labels[-(seq_len + 1):]
                n_truncated += 1

            # Pad to seq_len + 1 with EOS (label -100) — skipped when packing.
            if self.pad:
                pad = (seq_len + 1) - len(ids)
                if pad > 0:
                    ids.extend([EOS_ID] * pad)
                    labels.extend([-100] * pad)

            self.examples.append((
                np.asarray(ids, dtype=np.int64),
                np.asarray(labels, dtype=np.int64),
            ))

        print(f"  Built {len(self.examples):,} grounded-QA examples "
              f"(seq_len={seq_len}, {n_truncated} truncated to fit)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if not self.pad:
            raise RuntimeError(
                f"{type(self).__name__} was built with pad=False (variable-length "
                "examples for Issue #132 packing) and cannot be iterated directly "
                "-- the default collate would fail on ragged rows. Wrap it: "
                "PackedDataset(ds.examples, seq_len=...)."
            )
        ids, labels = self.examples[idx]
        # Causal shift: predict token t+1 from tokens ≤ t.
        x = torch.from_numpy(ids[:-1])
        y = torch.from_numpy(labels[1:])
        return x, y


# ---------------------------------------------------------------------------
# 2c. Tool-use dataset (gate-format, loss-masked) — issue #110 / #97 Phase C
# ---------------------------------------------------------------------------

class ToolUseDataset(Dataset):
    """Tokenized, loss-masked tool-use examples (see
    ``scripts/build_tool_use_data.py`` for the corpus format).

    Each record is rendered as::

        <USER> {query}
        <ASSISTANT> [TOOL_CALL]
        {compact json}
        [/TOOL_CALL]
        [TOOL_RESULT]
        {result}
        [/TOOL_RESULT]
        ... (repeated per step) ...
        {answer}<EOS>

    **Three-way masking** (the fix for a real flaw in the pre-existing
    ``hvn.tools.generate_tool_training_data``, which trained the model to
    *generate* the tool-result content -- i.e. to hallucinate what a tool
    will return, rather than to call it and use a real injected result):
      - the query (masked, input)
      - each ``[TOOL_CALL]...[/TOOL_CALL]`` block -- **learned** (the actual
        skill: emit a well-formed tool call)
      - each ``[TOOL_RESULT]...[/TOOL_RESULT]`` block -- **masked**
        (environment-injected context at real inference time, via
        ``EdgeToolCallingSession``; never a generation target)
      - the final answer + EOS -- **learned** (use the tool result to answer)
    """

    def __init__(self, tokenizer, records: list[dict], seq_len: int = 512,
                 pad: bool = True):
        from hvn.tools import TOOL_TOKENS
        import json as _json

        # #136: EOS from this tokenizer, not the module constant.
        EOS_ID = resolve_eos_id(tokenizer=tokenizer)
        self.eos_id = EOS_ID
        self.seq_len = seq_len
        self.pad = pad
        self.examples: list[tuple[np.ndarray, np.ndarray]] = []

        user_pref_ids = tokenizer.encode(USER_PREFIX)
        asst_pref_ids = tokenizer.encode(ASSISTANT_PREFIX)
        call_start_ids = tokenizer.encode(TOOL_TOKENS["tool_call_start"] + "\n")
        call_end_ids = tokenizer.encode("\n" + TOOL_TOKENS["tool_call_end"] + "\n")
        result_start_ids = tokenizer.encode(TOOL_TOKENS["tool_result_start"] + "\n")
        result_end_ids = tokenizer.encode("\n" + TOOL_TOKENS["tool_result_end"] + "\n")

        n_truncated = 0
        for r in records:
            prompt_ids = user_pref_ids + tokenizer.encode(r["query"] + "\n") + asst_pref_ids
            ids = list(prompt_ids)
            labels = [-100] * len(prompt_ids)

            for step in r["steps"]:
                call_json = _json.dumps(step["tool_call"], separators=(",", ":"))
                call_ids = tokenizer.encode(call_json)
                # [TOOL_CALL] block: LEARNED.
                seg = call_start_ids + call_ids + call_end_ids
                ids.extend(seg)
                labels.extend(seg)

                result_ids = tokenizer.encode(step["tool_result"])
                # [TOOL_RESULT] block: MASKED (environment-injected context).
                seg = result_start_ids + result_ids + result_end_ids
                ids.extend(seg)
                labels.extend([-100] * len(seg))

            # Final answer + EOS: LEARNED.
            answer_ids = tokenizer.encode(r["answer"]) + [EOS_ID]
            ids.extend(answer_ids)
            labels.extend(answer_ids)

            if len(ids) > seq_len + 1:
                # Keep the TAIL: the answer (always last) is never cut this
                # way. Same truncation strategy as GroundedQADataset.
                ids = ids[-(seq_len + 1):]
                labels = labels[-(seq_len + 1):]
                n_truncated += 1

            if self.pad:
                pad = (seq_len + 1) - len(ids)
                if pad > 0:
                    ids.extend([EOS_ID] * pad)
                    labels.extend([-100] * pad)

            self.examples.append((
                np.asarray(ids, dtype=np.int64),
                np.asarray(labels, dtype=np.int64),
            ))

        print(f"  Built {len(self.examples):,} tool-use examples "
              f"(seq_len={seq_len}, {n_truncated} truncated to fit)")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        if not self.pad:
            raise RuntimeError(
                f"{type(self).__name__} was built with pad=False (variable-length "
                "examples for Issue #132 packing) and cannot be iterated directly "
                "-- the default collate would fail on ragged rows. Wrap it: "
                "PackedDataset(ds.examples, seq_len=...)."
            )
        ids, labels = self.examples[idx]
        x = torch.from_numpy(ids[:-1])
        y = torch.from_numpy(labels[1:])
        return x, y


# ---------------------------------------------------------------------------
# 3. Fine-tuning loop
# ---------------------------------------------------------------------------

def finetune(model, dataloader, device, *, lr, epochs, grad_accum,
             out_path, log_path, warmup_frac=0.03, optimizer_name="adamw",
             save_every=0):
    print("\n" + "=" * 70)
    print("FINE-TUNING")
    print("=" * 70)

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Params: {param_count:,} | epochs: {epochs} | lr: {lr} | "
          f"grad_accum: {grad_accum}")

    steps_per_epoch = max(1, len(dataloader) // grad_accum)
    total_steps = steps_per_epoch * epochs

    # Issue #134: the parameter-free path, same wiring as continue_ft_hybrid.py.
    # Prodigy picks its own step size (landed 30x below the hand-set 2e-5-era
    # values on a continue-FT); no warmup and no cosine, because its d-ramp IS
    # the warmup and a decay schedule fights an estimator that is monotone.
    is_prodigy = optimizer_name == "prodigy"
    if is_prodigy:
        from hvn.optimizers import build_optimizer
        optimizer, desc = build_optimizer(model, "prodigy", lr=1.0,
                                          weight_decay=0.01)
        warmup_steps = 0
        scheduler = None
        print(f"  Optimizer: {desc}")
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01,
                                      fused=(device.type == "cuda"))
        warmup_steps = max(1, int(total_steps * warmup_frac))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, total_steps - warmup_steps))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
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
    #
    # accum_loss_t holds the SUM of per-micro-batch (loss/grad_accum) terms,
    # i.e. one per-optimizer-step mean loss added each opt step -- but is only
    # zeroed every 20 opt steps. Must divide by the opt-step count since the
    # last zero (n_steps_accum below), or the logged value is ~20x the true
    # mean loss and ppl=exp(min(loss,20)) reads as nonsense/catastrophic (the
    # exact bug found + fixed in continue_ft_hybrid.py, commit a023625 --
    # this sibling script predates that fix and had the same bug).
    accum_loss_t = torch.zeros((), device=device)
    n_steps_accum = 0

    for epoch in range(epochs):
        micro = 0
        for batch in dataloader:
            # Packed rows (Issue #132) carry a third element, doc_ids; padded
            # rows are the original 2-tuple.
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
                loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
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
                if scheduler is not None and global_step >= warmup_steps:
                    scheduler.step()
                optimizer.zero_grad()
                global_step += 1
                micro = 0
                n_steps_accum += 1

                if global_step % 20 == 0:
                    # Single host sync per log interval (was grad_accum syncs).
                    # Divide by n_steps_accum -- see the accumulator comment above.
                    mean_loss = float(accum_loss_t.item()) / max(1, n_steps_accum)
                    accum_loss_t.zero_()
                    n_steps_accum = 0
                    # Prodigy's param_groups["lr"] is the pinned 1.0 multiplier;
                    # log the step size it actually used, or the CSV is a constant.
                    cur_lr = (optimizer.effective_lr if is_prodigy
                              else optimizer.param_groups[0]["lr"])
                    ppl = math.exp(min(mean_loss, 20))
                    print(f"  Step {global_step:5d}/{total_steps} (epoch {epoch}) | "
                          f"loss {mean_loss:.4f} | ppl {ppl:.2f} | lr {cur_lr:.2e} | "
                          f"{(time.time()-start)/max(1,global_step):.2f}s/step", flush=True)
                    writer.writerow([global_step, epoch, f"{mean_loss:.6f}",
                                     f"{ppl:.4f}", f"{cur_lr:.8f}",
                                     time.strftime("%Y-%m-%d %H:%M:%S")])
                    csv_file.flush()

                # Numbered snapshots: crash insurance on long runs, and a way
                # back to an earlier point if a small corpus (600 tool-use
                # examples) starts being memorised -- the #134 follow-up saw
                # recall fall monotonically with FT steps.
                if save_every and global_step % save_every == 0:
                    snap = (os.path.splitext(out_path)[0]
                            + f"_step{global_step}.pt")
                    _save_edge_checkpoint(model, snap, global_step=global_step)
                    print(f"  ✓ Saved {snap}", flush=True)

        # Save at the end of each epoch.
        _save_edge_checkpoint(model, out_path, global_step=global_step)
        print(f"  ✓ Epoch {epoch} done — saved {out_path}")

    csv_file.close()
    print(f"\n  Fine-tuning complete: {global_step} steps, "
          f"{(time.time()-start)/60:.1f} min")
    # Issue #142: peak RESERVED, not allocated -- see the note in
    # continue_ft_hybrid.py. A config can read "fits" on allocated while
    # reserving past the card and silently spilling to host RAM.
    # #178: WDDM spills BELOW the literal card size (the desktop and other
    # processes hold part of it). A real spill was measured at 5,752 MiB
    # reserved on a 6,140 MiB card (93.7%), which "reserved > card" missed.
    # 90% catches that with ~225 MiB of margin, while configs measured to run
    # clean at 86-87% (docs/state.md) stay quiet. The 85% pretrain guard
    # (#163) is stricter on purpose: it refuses to start; this only warns.
    if torch.cuda.is_available():
        card = torch.cuda.get_device_properties(0).total_memory / 2**20
        alloc = torch.cuda.max_memory_allocated() / 2**20
        reserved = torch.cuda.max_memory_reserved() / 2**20
        warn = ("  ⚠ LIKELY SPILLED TO HOST RAM (>90% of card reserved)"
                if reserved > 0.90 * card else "")
        print(f"  peak VRAM: {alloc:.0f} MiB allocated / {reserved:.0f} MiB "
              f"reserved of {card:.0f} MiB card{warn}")


# ---------------------------------------------------------------------------
# 4. Quick generation smoke check
# ---------------------------------------------------------------------------

@torch.no_grad()
def sample_reply(model, tokenizer, device, user_msg, max_new=60):
    model.eval()
    EOS_ID = resolve_eos_id(model, tokenizer)  # #136: not the module constant
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
    parser.add_argument("--base-checkpoint", type=str, default=None,
                        help="Path to a pre-trained EdgeHVN checkpoint to fine-tune "
                             "from (required). The old default "
                             "checkpoints/edge_conversational/edge_conversational_step10000.pt "
                             "was the undertrained Run 2 base and has been removed from "
                             "LFS; use a Chinchilla-floor checkpoint under "
                             "checkpoints/edge_chinchilla83m/ instead.")
    parser.add_argument("--out", type=str, default="hvn_edge_chat.pt")
    parser.add_argument("--seq-len", type=int, default=512)
    # Issue #142: micro-batch 2 -> 4 with grad_accum 8 -> 4, so the EFFECTIVE
    # batch stays 16 and this is a pure throughput change, not a change to
    # training dynamics. #141 measured micro-batch 2 -> 4 at +35.4% end-to-end
    # (peak 4,136 MiB reserved, comfortable on the 6 GiB card). Do NOT raise this
    # to 8: it reserves 6,512 MiB, and Windows WDDM backs the overflow with host
    # RAM instead of OOM-ing -- 40% SLOWER than batch 2 while looking healthy.
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-5,
                        help="Only used by --optimizer adamw; prodigy ignores it.")
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw", "prodigy"],
                        help="Issue #134: 'prodigy' takes NO learning rate and beat "
                             "a hand-tuned one by 0.18 nats on a continue-FT, at a "
                             "step size 30x smaller. Default stays adamw so "
                             "existing invocations are unchanged.")
    parser.add_argument("--save-every", type=int, default=0,
                        help="Also save <out>_step{N}.pt every N optimizer steps "
                             "(0 = per-epoch save only, the old behaviour).")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--max-conversations", type=int, default=None)
    parser.add_argument("--grounded-replay", type=str, default=None,
                        help="JSONL of {passage,question,answer} (see "
                             "scripts/build_grounded_qa_data.py / expand_grounded_qa.py) "
                             "mixed in at --grounded-replay-frac to guard against "
                             "catastrophically forgetting recall during a long chat FT "
                             "(issue #108/#109 lesson).")
    parser.add_argument("--grounded-replay-frac", type=float, default=0.15,
                        help="P(grounded) when --grounded-replay is set.")
    parser.add_argument("--tool-use-data", type=str, default=None,
                        help="JSONL of {query,steps,answer} (see "
                             "scripts/build_tool_use_data.py) mixed in at "
                             "--tool-use-frac -- the agentic tool-calling FT "
                             "stage (issue #110 / #97 Phase C).")
    parser.add_argument("--tool-use-frac", type=float, default=0.5,
                        help="P(tool-use) when --tool-use-data is set.")
    parser.add_argument("--pack", action=argparse.BooleanOptionalAction, default=True,
                        help="Padding-free sequence packing (Issue #132), ON BY DEFAULT: "
                             "concatenate several short examples into each seq_len row "
                             "instead of padding each one out with EOS. The measured "
                             "corpora are 63-83%% padding at seq_len=512, so this is a "
                             "large wall-clock win for the SAME training (same real "
                             "tokens, ~6 examples per row). Documents are isolated "
                             "inside a row (DeltaNet state reset + conv taps + attention "
                             "mask), so it is exact, not an approximation. Measured "
                             "5.15x faster on grounded-QA and 1.45x on OASST, with "
                             "held-out loss within 0.4%% of the padded control -- see "
                             "docs/state.md. Pass --no-pack to restore the old padded "
                             "path. Enables the Triton scan (required for the exact "
                             "document-boundary state reset).")
    parser.add_argument("--pack-seed", type=int, default=0,
                        help="Seed for the packing shuffle (--pack), for reproducible rows.")
    parser.add_argument("--pack-strategy", type=str, default="shuffled_first_fit",
                        choices=["shuffled_first_fit", "first_fit_decreasing"],
                        help="Bin-packing policy. The default shuffles first so example "
                             "length is uncorrelated with batch composition; "
                             "first_fit_decreasing wastes fewer tokens but sorts long "
                             "examples to the front.")
    parser.add_argument("--hf-cache-dir", type=str, default=None)
    # Issue #143: default decided by measurement in THIS script's loop, not
    # inherited from continue_ft_hybrid's -- see docs/state.md.
    fast_group = parser.add_mutually_exclusive_group()
    fast_group.add_argument("--fast", dest="fast", action="store_true", default=True,
                            help="TF32 + torch.compile on the training path "
                                 "(DEFAULT ON since #143)")
    fast_group.add_argument("--no-fast", dest="fast", action="store_false",
                            help="disable torch.compile")
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
    chat_ds = InstructionDataset(tokenizer, conversations, seq_len=args.seq_len,
                                 pad=not args.pack)

    import json

    # (dataset, target_fraction, label) for every extra replay/task mix
    # requested -- chat gets whatever fraction remains. Generalizes the
    # original single-grounded-replay mixing (issue #108) to also support
    # tool-use data (issue #110) mixed in alongside it, via the same
    # ConcatDataset + WeightedRandomSampler pattern.
    extra: list[tuple[Dataset, float, str]] = []
    if args.grounded_replay:
        records = [json.loads(line) for line in
                   open(args.grounded_replay, encoding="utf-8")]
        grounded_ds = GroundedQADataset(tokenizer, records, seq_len=args.seq_len,
                                        pad=not args.pack)
        print(f"  Grounded-QA replay: {len(grounded_ds)} records from "
              f"{args.grounded_replay} (target fraction={args.grounded_replay_frac})")
        extra.append((grounded_ds, args.grounded_replay_frac, "grounded"))
    if args.tool_use_data:
        records = [json.loads(line) for line in
                   open(args.tool_use_data, encoding="utf-8")]
        tool_ds = ToolUseDataset(tokenizer, records, seq_len=args.seq_len,
                                 pad=not args.pack)
        print(f"  Tool-use data: {len(tool_ds)} records from "
              f"{args.tool_use_data} (target fraction={args.tool_use_frac})")
        extra.append((tool_ds, args.tool_use_frac, "tool-use"))

    if args.pack:
        # Issue #132: pack each corpus INDEPENDENTLY, then mix as before. A row
        # therefore only ever holds documents from one corpus, which keeps the
        # WeightedRandomSampler mix interpretable (a sampled row is still
        # unambiguously "a grounded row" or "a chat row"); mixing corpora inside
        # a row would make the requested fractions mean something subtly
        # different from what the flags say.
        from hvn.core.packing import PackedDataset
        # Pad with the SAME id the datasets appended as EOS. Taking it from the
        # dataset rather than the module constant keeps padding and EOS in
        # agreement for any vocabulary (#136).
        pack_pad_id = getattr(chat_ds, "eos_id", EOS_ID)
        print("\n  Packing (Issue #132) — padding-free rows:")
        print("    chat:", end=" ")
        chat_ds = PackedDataset(chat_ds.examples, seq_len=args.seq_len,
                                pad_id=pack_pad_id, seed=args.pack_seed, strategy=args.pack_strategy)
        packed_extra = []
        for ds, frac, label in extra:
            print(f"    {label}:", end=" ")
            packed_extra.append(
                (PackedDataset(ds.examples, seq_len=args.seq_len, pad_id=pack_pad_id,
                               seed=args.pack_seed, strategy=args.pack_strategy), frac, label)
            )
        extra = packed_extra

    if extra:
        extra_frac_total = sum(f for _, f, _ in extra)
        if extra_frac_total >= 1.0:
            raise SystemExit(f"--grounded-replay-frac + --tool-use-frac must sum to "
                             f"< 1.0 (chat needs a nonzero share), got {extra_frac_total}")
        chat_frac = 1.0 - extra_frac_total
        all_ds = [(chat_ds, chat_frac, "chat")] + extra
        dataset = ConcatDataset([ds for ds, _, _ in all_ds])
        weights = []
        for ds, frac, _ in all_ds:
            weights.extend([frac / len(ds)] * len(ds))
        sampler = WeightedRandomSampler(weights, num_samples=len(dataset),
                                        replacement=True)
        dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                                drop_last=True, pin_memory=(device.type == "cuda"))
        mix_str = " / ".join(f"{frac:.0%} {label}" for _, frac, label in all_ds)
        print(f"  Combined dataset: {len(dataset)} examples, sampled to {mix_str}")
    else:
        dataset = chat_ds
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                                drop_last=True, pin_memory=(device.type == "cuda"))

    # --- Model (load pre-trained base) ---
    print("\n" + "=" * 70)
    print("MODEL — loading base checkpoint")
    print("=" * 70)
    if args.base_checkpoint is None:
        print("❌ --base-checkpoint is required. The old default "
              "(checkpoints/edge_conversational/edge_conversational_step10000.pt) "
              "was the undertrained Run 2 base and has been removed from LFS; "
              "use a Chinchilla-floor checkpoint under "
              "checkpoints/edge_chinchilla83m/ instead.")
        sys.exit(1)
    if not os.path.exists(args.base_checkpoint):
        print(f"❌ Base checkpoint not found: {args.base_checkpoint}")
        sys.exit(1)
    model, base_step, _ = load_checkpoint_for_resume(args.base_checkpoint, device)
    print(f"  Loaded {args.base_checkpoint} (pre-trained to step {base_step})")

    if args.pack:
        # Issue #132: the boundary reset is expressed as alpha=0, which only the
        # sequential-over-S scans represent exactly. The Triton kernel is one
        # (direct `state = state * a_t` multiply, fp32, no reparameterization);
        # the eager chunked scan folds alpha into a cumprod/log-space form that
        # clamps at 1e-6 and would leak state across documents, so DeltaNetBlock
        # refuses it. Unlike continue_ft_hybrid.py / continue_ft_looped.py, this
        # script does not otherwise set the scan flags, so set them here.
        # (enable_scan_flags lives in continue_ft_hybrid.py, which imports the
        # datasets FROM this module -- importing it back would cycle.)
        n_set = 0
        for blk in (model.blocks or []):
            if hasattr(blk, "use_triton_scan"):
                if device.type == "cuda":
                    blk.use_triton_scan = True
                    blk.use_megakernel_scan = True
                else:
                    blk.train_chunk = 0     # CPU: exact sequential scan
                n_set += 1
        which = "triton + megakernel scan" if device.type == "cuda" else "sequential scan"
        print(f"  [--pack] enabled {which} on {n_set} DeltaNet blocks "
              f"(required for the exact document-boundary state reset)")

    # --- Fine-tune ---
    epochs = 1 if args.smoke else args.epochs
    log_path = os.path.splitext(args.out)[0] + "_finetune_log.csv"

    # Issue #143: torch.compile for the TRAINING path only.
    #
    # Deliberately not applied to `model` itself. `sample_reply` below generates
    # token-by-token with a growing sequence, so a compiled module would hit a
    # new shape every token and recompile on each one -- far slower than eager,
    # and it would be charged to "compile" in any benchmark. The compiled
    # wrapper shares parameters with `model`, so training through it updates the
    # same weights; generation and saving keep using the plain module.
    # (`_save_edge_checkpoint` unwraps `_orig_mod` anyway, so per-epoch saves
    # inside finetune() are safe either way.)
    train_model = model
    if args.fast and device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print("  [--fast] TF32 + torch.compile on the training path "
              "(first steps are slow while it compiles)")
        train_model = torch.compile(model)

    finetune(train_model, dataloader, device,
             lr=args.lr, epochs=epochs, grad_accum=args.grad_accum,
             out_path=args.out, log_path=log_path,
             optimizer_name=args.optimizer, save_every=args.save_every)

    # --- Smoke generation ---
    print("\n  Sample replies from the fine-tuned model:")
    for q in ["hi", "What's 2+2?", "What is the capital of France?",
              "Write a haiku about the sea."]:
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
