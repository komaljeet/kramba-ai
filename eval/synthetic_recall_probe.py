"""Synthetic associative-recall probe — the clean, high-N R&D go/no-go.

`grounded_qa.py` is a real-world grounding test: 5 hand-written QA items scored by
keyword match. It's right for "does this ship as a product" but the WRONG
instrument for an R&D decision that gates a 1B-parameter investment — 5 items is
dominated by noise (one lucky keyword flips the verdict), and its answers are
confounded by the trunk's weak real-world knowledge/fluency, not just recall.

This probe fixes both problems: run hundreds of trials (real statistic) with
INVENTED entities/values (pseudo-words like "Zorvex"/"Blenthar" — unambiguous,
zero real-world-knowledge confound, cannot be answered closed-book by chance).

v2 note: an earlier version rendered pairs as raw random GPT-2 token ids glued
together with ad hoc separators (":", "\n", "?"). That produced a flat 0%
(including top-5) on EVERY checkpoint and EVERY condition, including the hybrid
INSIDE its own attention window — too uniform to be an architecture verdict.
Debugging showed why: random subword ids decode to garbled non-words the model
never saw a copy-task pattern for, so it just free-associates ('\n', ' I', ' A'
were the top completions; correct answer ranked ~41,000th). That was a format
artifact, not a recall measurement. Fixed by rendering every trial in the SAME
chat/QA template the model was fine-tuned on (`OPEN_BOOK_INSTRUCTION`, matching
`grounded_qa.py` and `continue_ft_hybrid.py` exactly) — in-distribution phrasing,
synthetic content.

Usage:
    python -m eval.synthetic_recall_probe --checkpoint <ckpt> --n-trials 300
    python -m eval.synthetic_recall_probe --checkpoint <ckpt> --n-trials 300 \
        --probe-window 128 --far
"""

from __future__ import annotations

import argparse
import os
import re
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402
from eval.grounded_qa import set_attention_window, EOS             # noqa: E402

USER_PREFIX = "<|user|>\n"
ASST_PREFIX = "<|assistant|>\n"
OPEN_BOOK_INSTRUCTION = "Answer the question using only the context below.\n\n"

_SYL_A = ["Zor", "Blen", "Krav", "Fim", "Dros", "Wex", "Nal", "Yur", "Plith", "Gorn",
          "Squi", "Thav", "Mox", "Ren", "Xelt", "Corv", "Bru", "Fen", "Ost", "Vim"]
_SYL_B = ["vex", "thar", "ilo", "unda", "eth", "arn", "orix", "ella", "ux", "ith",
          "ova", "and", "esk", "orum", "ina", "ock", "yle", "wen", "atra", "ist"]


def make_word(rng: torch.Generator) -> str:
    a = _SYL_A[int(torch.randint(0, len(_SYL_A), (1,), generator=rng).item())]
    b = _SYL_B[int(torch.randint(0, len(_SYL_B), (1,), generator=rng).item())]
    return a + b


def make_pair(rng: torch.Generator, used: set[str]) -> tuple[str, str]:
    """A distinct (entity, value) pseudo-word pair, both unused so far."""
    while True:
        entity, value = make_word(rng), make_word(rng)
        if entity != value and entity not in used and value not in used:
            used.add(entity)
            used.add(value)
            return entity, value


def whole_word_hit(text: str, word: str) -> bool:
    return re.search(rf"\b{re.escape(word)}\b", text, flags=re.IGNORECASE) is not None


@torch.no_grad()
def generate(model, tok, device, prompt, max_new=20, temp=0.4, top_p=0.9, rep_pen=1.3):
    ids = tok.encode(prompt)
    x = torch.tensor([ids], device=device)
    out: list[int] = []
    for _ in range(max_new):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)[0, -1].float()
        for t in set(out):
            logits[t] /= rep_pen
        probs = F.softmax(logits / temp, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp[cum - sp > top_p] = 0
        sp = sp / sp.sum()
        nxt = int(si[torch.multinomial(sp, 1)])
        if nxt == EOS:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(out).strip()


def build_prompt(rng, n_distractors: int) -> tuple[str, str]:
    """One trial: n_distractors unrelated (entity, value) facts + 1 target fact,
    target placed FIRST (so distractors act as filler pushing it back — the
    near/far lever, matching the --probe-window override on the model side).
    Returns (open_book_prompt, correct_value_word).
    """
    used: set[str] = set()
    tgt_entity, tgt_value = make_pair(rng, used)
    facts = [f"The {tgt_entity} is called {tgt_value}."]
    for _ in range(n_distractors):
        e, v = make_pair(rng, used)
        facts.append(f"The {e} is called {v}.")
    context = " ".join(facts)
    prompt = (USER_PREFIX + OPEN_BOOK_INSTRUCTION +
              f"Context: {context}\n\nQuestion: What is the {tgt_entity}?\n" + ASST_PREFIX)
    return prompt, tgt_value


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--n-distractors", type=int, default=0,
                    help="unrelated filler facts appended after the target fact "
                         "(pushes it toward/beyond --probe-window)")
    ap.add_argument("--probe-window", type=int, default=None,
                    help="override every TransformerBlock's window (e.g. 128)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model, step, _ = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    print(f"Loaded {args.checkpoint} (step {step})")

    if args.probe_window is not None:
        n = set_attention_window(model, args.probe_window)
        print(f"  window overridden to {args.probe_window} on {n} attention block(s)")

    rng = torch.Generator().manual_seed(args.seed)
    hits = 0
    for i in range(args.n_trials):
        prompt, correct = build_prompt(rng, args.n_distractors)
        answer = generate(model, tok, device, prompt)
        ok = whole_word_hit(answer, correct)
        hits += ok
        if i < 3:  # print a few for a sanity spot-check
            print(f"  [{'HIT' if ok else 'miss'}] expect={correct!r} got={answer!r}")

    acc = hits / args.n_trials
    print(f"\nSynthetic-recall result over {args.n_trials} trials "
          f"({args.n_distractors} distractor facts):")
    print(f"  exact whole-word accuracy: {acc:.1%}  ({hits}/{args.n_trials})")
    if acc > 0.5:
        print("  -> STRONG recall: reliably retrieves the exact invented value.")
    elif acc > 0.05:
        print("  -> PARTIAL recall: better than a garbled-output baseline, not reliable.")
    else:
        print("  -> NO recall: the model never reproduces the exact invented value.")


if __name__ == "__main__":
    main()
