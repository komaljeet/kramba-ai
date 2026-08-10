"""Held-out loss on the grounded-QA corpus — a generalization probe for the
OASST packing arms (Issue #132 follow-up).

**Why this corpus is genuinely held out.** All three OASST re-gate arms ran
`continue_ft_hybrid.py --chat-mix 1.0`. At `chat_mix=1.0` the grounded examples'
sampler weight is `1/n_g * (1 - chat_mix) == 0`, so the 5,240 grounded-QA
examples were **never sampled** by any arm ("sampled to 100% chat / 0% grounded"
in every log). The trunk they all warm-start from
(`edge_conversational_step101000.pt`) is a pure SlimPajama+Wiki pretrain, never
grounded-FT'd, and the 2 attention layers start fresh. So this corpus is equally
unseen by all three checkpoints — a symmetric comparison, not a rigged one.

**Why it is needed.** The equal-wall-clock arm takes 26.7 passes over a fixed
3,465-example OASST corpus against the control's 18.5. Lowering *train* loss is
what extra epochs do almost by construction, and the FT harness logs no
validation loss (every arm trained on all 3,465 conversations, so there is no
held-out OASST split to appeal to). Without a held-out number, "packing's
speedup converts to quality" is unfalsifiable.

**What it measures, stated honestly:** transfer to a different task in the same
chat template, not OASST conversational fluency itself. A model that merely
memorised OASST harder should not transfer better; one that genuinely improved
plausibly should. It is a real signal where train loss has none, not a perfect
proxy for the thing the arms were optimising.

Evaluation is identical for every arm — the corpus is built **padded**
(`pad=True`) regardless of how the checkpoint was trained, in a fixed order with
no sampling. Padding carries `-100` labels, so the reported figure is a
per-real-token mean either way; using one code path for all arms removes packing
from the measurement side entirely.

Usage::

    python -m eval.heldout_grounded_loss --checkpoints a.pt b.pt c.pt
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_edge_conversational import (  # noqa: E402
    GPT2_VOCAB_SIZE,
    load_checkpoint_for_resume,
)
from finetune_edge_conversational import GroundedQADataset  # noqa: E402


class AllTokenView(torch.utils.data.Dataset):
    """Same examples, but supervising EVERY real token instead of the answer span.

    ``GroundedQADataset`` masks the prompt to ``-100`` so training optimises only
    "emit the answer from the context". That leaves ~2.8 supervised tokens per
    example (~14K over the corpus) -- a legitimate measurement of the grounded
    skill, but a high-variance one, and at a loss near ``ln(50257)=10.82`` it is
    measuring a near-random quantity on checkpoints that never trained it.

    This view instead scores next-token prediction over the whole rendered
    example (~450K tokens): a plain held-out language-modelling number on text no
    arm has seen, which is a much lower-variance instrument for "did the extra
    passes buy generalization or memorization".

    Padding must still be excluded, and it cannot be found by matching ``EOS_ID``
    (a real EOS also terminates the answer). The last non-``-100`` label marks
    that terminating EOS, so everything after it is padding.
    """

    def __init__(self, examples):
        self.examples = examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ids, labels = self.examples[idx]
        real = int(np.flatnonzero(labels != -100)[-1])  # index of the answer's EOS
        y = ids[1:].copy()
        y[real:] = -100                                  # labels[1:] is shifted by one
        return torch.from_numpy(ids[:-1]), torch.from_numpy(y)


@torch.no_grad()
def corpus_loss(model, loader, device, use_bf16: bool) -> tuple[float, int]:
    """Token-weighted mean CE over the corpus.

    Sums CE over every non-``-100`` label and divides once at the end, rather
    than averaging per-batch means -- batches hold different numbers of real
    tokens, so a mean-of-means would silently weight short examples more.
    """
    crit = nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    total, n_tok = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=use_bf16):
            logits = model(x)
        loss = crit(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        total += float(loss.item())
        n_tok += int((y != -100).sum().item())
    return total / max(1, n_tok), n_tok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="+", required=True)
    ap.add_argument("--labels", nargs="*", default=None,
                    help="display names, parallel to --checkpoints")
    ap.add_argument("--grounded-data", type=str,
                    default="data/grounded_qa_expanded.jsonl")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="evaluate only the first N examples (debug)")
    ap.add_argument("--all-tokens", action="store_true",
                    help="score every real token (held-out LM, ~450K tokens) "
                         "instead of only the answer span (~14K)")
    ap.add_argument("--bf16", action="store_true",
                    help="match the FT runs' autocast dtype (default fp32)")
    ap.add_argument("--json-out", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    records = [json.loads(l) for l in open(args.grounded_data, encoding="utf-8")]
    if args.limit:
        records = records[:args.limit]
    ds = GroundedQADataset(tok, records, seq_len=args.seq_len, pad=True)
    if args.all_tokens:
        ds = AllTokenView(ds.examples)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        drop_last=False, pin_memory=(device.type == "cuda"))
    print(f"Scoring: {'ALL real tokens (held-out LM)' if args.all_tokens else 'answer span only'}")
    print(f"Held-out corpus: {len(ds)} grounded-QA examples from {args.grounded_data}")
    print(f"  (weighted to 0 by --chat-mix 1.0 in every arm -- unseen by all)\n")

    labels = args.labels or args.checkpoints
    results = []
    for label, ckpt in zip(labels, args.checkpoints):
        model, step, _ = load_checkpoint_for_resume(ckpt, device)
        model.eval()
        loss, n_tok = corpus_loss(model, loader, device, args.bf16)
        results.append({"label": label, "checkpoint": ckpt, "step": step,
                        "heldout_loss": loss, "heldout_ppl": math.exp(min(loss, 20)),
                        "real_tokens": n_tok})
        print(f"  {label:34s} step {step:5d} | held-out loss {loss:.4f} | "
              f"ppl {math.exp(min(loss, 20)):7.2f}  ({n_tok:,} real tokens)")
        del model
        torch.cuda.empty_cache()

    if len(results) > 1:
        base = results[0]
        print(f"\nRelative to {base['label']!r}:")
        for r in results[1:]:
            d = r["heldout_loss"] - base["heldout_loss"]
            pct = (r["heldout_ppl"] / base["heldout_ppl"] - 1) * 100
            print(f"  {r['label']:34s} {d:+.4f} nats | ppl {pct:+.1f}%")

    if args.json_out:
        os.makedirs(os.path.dirname(args.json_out), exist_ok=True)
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote {args.json_out}")


if __name__ == "__main__":
    main()
