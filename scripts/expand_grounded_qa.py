"""Scale the grounded-QA continue-FT corpus from ~40 seed pairs to a few thousand.

The A4 fine-tune (`continue_ft_hybrid.py`) failed the A5 gate because it trained the
2 attention layers on only **40** (passage, question, answer) triples — far too few
to learn the *general* "find the answer span in the passage" skill; it memorised 40
pairs and degraded the trunk's weak reading (gate lift +1 -> -1). The seed builder's
own header flags the fix: programmatic cloze generation from WikiText-103 for scale.

This does exactly that. For each clean WikiText-103 sentence we pick one salient,
unambiguous content word (a proper noun or number occurring exactly once), blank it
in a copy-cloze *question*, and keep the full sentence as the *passage* — so the
answer span is present in the context and the model must retrieve it. That is the
same primitive the gate measures (copy the right token out of the passage), at scale
and with real-text diversity. The 40 semantic seed pairs are appended (up-weighted)
so definitional QA is not drowned out.

Output rows match `continue_ft_hybrid.GroundedQADataset` exactly:
    {"passage": ..., "question": ..., "answer": ...}

Usage:
    python scripts/expand_grounded_qa.py \
        --wikitext ../../wikitext-103-v1/wiki.train.tokens \
        --seed data/grounded_qa_train.jsonl \
        --out data/grounded_qa_expanded.jsonl --n 5000
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re

# Capitalised function words that are poor answers (sentence-initial or generic).
_STOP_CAP = {
    "The", "A", "An", "He", "She", "It", "They", "We", "You", "I", "This",
    "That", "These", "Those", "In", "On", "At", "As", "By", "For", "From",
    "With", "And", "But", "Or", "Its", "His", "Her", "Their", "There", "When",
    "While", "After", "Before", "During", "However", "Although", "Because",
    "Some", "Many", "Most", "One", "Two", "Three", "First", "Second", "Later",
    "Both", "Each", "Other", "Such", "Then", "Also", "Between", "About",
}


def detokenize(line: str) -> str:
    """Turn a WikiText-103 token line into readable prose."""
    s = line.strip()
    s = s.replace(" @-@ ", "-").replace(" @,@ ", ",").replace(" @.@ ", ".")
    s = s.replace(" @-@", "-").replace("@-@ ", "-")
    # spacing fixes: no space before closing punctuation, drop bracket padding
    s = re.sub(r"\s+([.,;:!?%])", r"\1", s)
    s = re.sub(r"\(\s+", "(", s)
    s = re.sub(r"\s+\)", ")", s)
    s = s.replace(" 's", "'s").replace(" n't", "n't")
    s = re.sub(r"\s{2,}", " ", s)
    return s.strip()


def split_sentences(text: str) -> list[str]:
    # split on sentence-final punctuation followed by a space + capital
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text)
    return [p.strip() for p in parts if p.strip()]


def pick_answer(sentence: str) -> str | None:
    """Choose one unambiguous salient token (proper noun or number) to blank."""
    words = sentence.split()
    if not (8 <= len(words) <= 40):
        return None
    cands: list[str] = []
    for i, w in enumerate(words):
        core = w.strip(".,;:!?()\"'")
        if i == 0:
            continue  # never the sentence-initial word
        if re.fullmatch(r"[A-Z][a-zA-Z]{2,}", core) and core not in _STOP_CAP:
            cands.append(core)
        elif re.fullmatch(r"\d[\d,.]*\d|\d", core):
            cands.append(core)
    # require the candidate to occur exactly once, so the blank is unambiguous
    for c in cands:
        if sentence.count(c) == 1:
            return c
    return None


def make_example(sentence: str) -> dict | None:
    ans = pick_answer(sentence)
    if ans is None:
        return None
    blanked = sentence.replace(ans, "_____", 1)
    question = "Fill in the blank using the passage above: " + blanked
    return {"passage": sentence, "question": question, "answer": ans}


def iter_sentences(path: str):
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s or s.startswith("="):        # blank lines / headers
                continue
            if "<unk>" in s:
                continue
            text = detokenize(s)
            for sent in split_sentences(text):
                if "<unk>" in sent or "=" in sent:
                    continue
                yield sent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wikitext", required=True, help="wiki.train.tokens path")
    ap.add_argument("--seed", default="data/grounded_qa_train.jsonl",
                    help="the 40 curated semantic-QA pairs to fold in")
    ap.add_argument("--out", default="data/grounded_qa_expanded.jsonl")
    ap.add_argument("--n", type=int, default=5000, help="cloze examples to keep")
    ap.add_argument("--seed-dups", type=int, default=6,
                    help="times to repeat the semantic seed (up-weighting)")
    ap.add_argument("--rng", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.rng)
    seen_passages: set[str] = set()
    cloze: list[dict] = []
    for sent in iter_sentences(args.wikitext):
        if sent in seen_passages:
            continue
        # light length filter for clean, self-contained passages
        if not (40 <= len(sent) <= 240):
            continue
        ex = make_example(sent)
        if ex is None:
            continue
        seen_passages.add(sent)
        cloze.append(ex)
        # sample a broad pool then downselect, so we're not just the first N sents
        if len(cloze) >= args.n * 4:
            break

    rng.shuffle(cloze)
    cloze = cloze[: args.n]

    # fold in the curated semantic-QA seed (up-weighted so it isn't drowned)
    seed_rows: list[dict] = []
    if os.path.exists(args.seed):
        with open(args.seed, "r", encoding="utf-8") as fh:
            base = [json.loads(l) for l in fh if l.strip()]
        seed_rows = base * args.seed_dups

    rows = cloze + seed_rows
    rng.shuffle(rows)

    # validate: answer must appear verbatim in its passage
    bad = [r for r in rows if r["answer"] not in r["passage"]]
    if bad:
        raise SystemExit(f"{len(bad)} rows have answer not in passage (bug)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"wrote {len(rows)} rows -> {args.out}")
    print(f"  cloze (WikiText copy): {len(cloze)}")
    print(f"  semantic seed: {len(seed_rows)} ({len(seed_rows)//max(1,args.seed_dups)} x {args.seed_dups})")
    print("  sample cloze:", json.dumps(cloze[0], ensure_ascii=False)[:200] if cloze else "none")


if __name__ == "__main__":
    main()
