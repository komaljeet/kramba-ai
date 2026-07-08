"""
Grounded-QA evaluation for the fine-tuned EdgeHVN chat model.

The core question for the "tiny model + retrieval" thesis: given a retrieved
passage that contains the answer, can an 83M model *ground* its response in that
passage — even when it has no idea closed-book?

For each item we generate twice:
  - **closed-book**: just the question (the model must recall from weights)
  - **open-book**:   the question + a gold passage (RAG-style)

If open-book answers are correct where closed-book ones are not, retrieval is
doing its job and the model can read/synthesize. That is the decisive signal for
whether to build the RAG product on 83M or scale up first.

Usage:
    python -m eval.grounded_qa --checkpoint hvn_edge_chat.pt
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402

EOS = 50256
USER_PREFIX = "<|user|>\n"
ASST_PREFIX = "<|assistant|>\n"

# (question, gold passage the "retriever" returns, answer keywords that must appear)
ITEMS = [
    ("What is the capital of France?",
     "France is a country in Western Europe. Its capital and largest city is "
     "Paris, situated on the river Seine.",
     ["paris"]),
    ("Who wrote Romeo and Juliet?",
     "Romeo and Juliet is a tragedy written by the English playwright William "
     "Shakespeare early in his career.",
     ["shakespeare"]),
    ("What is the boiling point of water?",
     "At standard atmospheric pressure, water boils at 100 degrees Celsius, "
     "which is 212 degrees Fahrenheit.",
     ["100", "celsius"]),
    ("What language is spoken in Brazil?",
     "Brazil is the largest country in South America. Its official and most "
     "widely spoken language is Portuguese.",
     ["portuguese"]),
    ("What is a black hole?",
     "A black hole is a region of spacetime where gravity is so strong that "
     "nothing, not even light, can escape from it.",
     ["gravity", "light", "escape"]),
]


@torch.no_grad()
def generate(model, tok, device, prompt, max_new=60, temp=0.4, top_p=0.9,
             rep_pen=1.3):
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


def hit(answer: str, keywords: list[str]) -> bool:
    a = answer.lower()
    return any(k.lower() in a for k in keywords)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="hvn_edge_chat.pt")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if sys.platform == "win32":
        os.system("chcp 65001 >nul 2>&1")
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    model, step, _ = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    print(f"Loaded {args.checkpoint} (step {step})\n")

    closed_hits = open_hits = 0
    for q, passage, kw in ITEMS:
        closed_prompt = USER_PREFIX + q + "\n" + ASST_PREFIX
        open_prompt = (USER_PREFIX +
                       "Answer the question using only the context below.\n\n"
                       f"Context: {passage}\n\nQuestion: {q}\n" + ASST_PREFIX)
        closed = generate(model, tok, device, closed_prompt)
        opened = generate(model, tok, device, open_prompt)
        c_ok, o_ok = hit(closed, kw), hit(opened, kw)
        closed_hits += c_ok
        open_hits += o_ok
        print("=" * 72)
        print(f"Q: {q}   (expect: {kw})")
        print(f"  closed-book [{'HIT' if c_ok else 'miss'}]: {closed}")
        print(f"  open-book   [{'HIT' if o_ok else 'miss'}]: {opened}")

    n = len(ITEMS)
    print("\n" + "=" * 72)
    print(f"GROUNDING RESULT: closed-book {closed_hits}/{n}  |  "
          f"open-book {open_hits}/{n}")
    print(f"  retrieval lift: +{open_hits - closed_hits}/{n}")
    if open_hits >= closed_hits + max(1, n // 2):
        print("  -> Retrieval clearly helps: the model can ground in passages.")
    elif open_hits > closed_hits:
        print("  -> Partial grounding: retrieval helps but reading is shaky.")
    else:
        print("  -> No grounding lift: 83M reading capacity likely insufficient.")


if __name__ == "__main__":
    main()
