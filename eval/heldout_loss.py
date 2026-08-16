"""Held-out loss on a FIXED token set, comparable across sizes and checkpoints (#159).

Author: Rambarun Komaljeet

Every loss number the training runs report is TRAINING loss -- the code corpus is
built with 0 held-out windows. #132's verdict was retracted for exactly this:
grading arms on train loss in a multi-pass regime measures memorisation, not
quality. The 48M passed 1.28x its Chinchilla budget, so it is already re-seeing
data.

This evaluates any checkpoint on the last 20 MB of the corpus -- the slice
``eval/fim_completion_gate.py`` already reserves and no training run touches --
and it must produce the SAME token set for every call, or a scaling curve
measures the eval data rather than the model.

Determinism, and why it is asserted rather than assumed
-------------------------------------------------------
The token set is derived from a fixed byte offset with a fixed length and no
sampling, and ``--expect-tokens`` lets a caller pin the count it got last time.
A curve assembled from evaluations of slightly different data is the failure this
whole file exists to prevent, and it would look like a perfectly smooth result.

Usage::

    python -m eval.heldout_loss --checkpoint <ckpt.pt> --tokenizer <dir>
    python -m eval.heldout_loss --sweep checkpoints/code_48m_v2 --out results/x.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_edge_conversational as tec  # noqa: E402

#: Byte window reserved for evaluation. Matches eval/fim_completion_gate.py's
#: "last 20 MB", so the two gates never disagree about what is held out.
HELDOUT_BYTES = 20_000_000
#: Tokens actually scored. Capped so a full sweep is minutes, not hours; the cap
#: is part of the comparability contract, not a performance knob.
MAX_TOKENS = 400_000


def heldout_ids(corpus: str, tokenizer, max_tokens: int = MAX_TOKENS) -> list[int]:
    size = os.path.getsize(corpus)
    with open(corpus, "r", encoding="utf-8", errors="replace") as fh:
        fh.seek(max(0, size - HELDOUT_BYTES))
        text = fh.read()
    # Drop the first partial document so the window starts on a boundary.
    sep = "<|endoftext|>"
    if sep in text:
        text = text.split(sep, 1)[1]
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return ids[:max_tokens]


@torch.no_grad()
def evaluate(model, ids: list[int], seq_len: int, device) -> dict:
    """Mean next-token NLL over non-overlapping windows.

    Windows do not overlap and the final partial window is dropped, so the token
    count scored is identical for any two models given the same ids -- a model
    scored on more tokens than another is not comparable to it.
    """
    model.eval()
    n_win = len(ids) // seq_len
    total, counted = 0.0, 0
    for w in range(n_win):
        chunk = ids[w * seq_len:(w + 1) * seq_len]
        x = torch.tensor([chunk], device=device)
        logits = model(x)
        if isinstance(logits, tuple):
            logits = logits[0]
        # Predict token t+1 from t: drop the last position, shift the targets.
        lp = logits[0, :-1].float()
        tgt = torch.tensor(chunk[1:], device=device)
        total += F.cross_entropy(lp, tgt, reduction="sum").item()
        counted += tgt.numel()
    nll = total / max(1, counted)
    return {"heldout_nll": nll, "heldout_ppl": float(torch.tensor(nll).exp()),
            "tokens_scored": counted, "windows": n_win}


def _step_of(path: str) -> int:
    m = re.search(r"step(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 10 ** 9


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--sweep", help="directory of edge_conversational_step*.pt")
    p.add_argument("--tokenizer", default="data_cache/tokenizer_32k_fim")
    p.add_argument("--corpus", default="data_cache/code_corpus/stack_code.txt")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    p.add_argument("--expect-tokens", type=int, default=None,
                   help="Fail if the held-out set is not exactly this many "
                        "tokens. Pin it once a curve has started.")
    p.add_argument("--out")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")

    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.tokenizer)
    ids = heldout_ids(args.corpus, tok, args.max_tokens)
    print(f"held-out: {len(ids):,} tokens from the last "
          f"{HELDOUT_BYTES / 1e6:.0f} MB, seq_len {args.seq_len}")
    if args.expect_tokens is not None and len(ids) != args.expect_tokens:
        sys.exit(f"held-out set changed: {len(ids)} tokens, expected "
                 f"{args.expect_tokens}. The curve would not be comparable.")

    targets = ([args.checkpoint] if args.checkpoint else
               sorted(glob.glob(os.path.join(args.sweep,
                                             "edge_conversational_step*.pt")),
                      key=_step_of))
    if not targets:
        sys.exit("no checkpoints found")

    rows = []
    print(f"\n{'step':>8}{'held-out NLL':>15}{'ppl':>10}")
    for path in targets:
        model, step, _ = tec.load_checkpoint_for_resume(path, device)
        r = evaluate(model, ids, args.seq_len, device)
        r["step"] = step
        r["checkpoint"] = path
        r["params"] = sum(p_.numel() for p_ in model.parameters())
        r["body"] = r["params"] - model.token_embedding.weight.numel()
        r["tokens_seen"] = step * 4096 * 8
        r["tokens_per_body_param"] = r["tokens_seen"] / max(1, r["body"])
        rows.append(r)
        print(f"{step:>8}{r['heldout_nll']:>15.4f}{r['heldout_ppl']:>10.3f}")
        del model
        torch.cuda.empty_cache()

    if args.out:
        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"heldout_tokens": len(ids), "seq_len": args.seq_len,
                       "rows": rows}, fh, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
