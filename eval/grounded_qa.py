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

--far-probe adds a **beyond-window** test that discriminates the hybrid from a
plain sliding-window (SWA) transformer. The default gate places the passage next
to the question, so it sits inside any window — an SWA transformer reads it too.
The probe shrinks the attention window and pushes the passage past it with
neutral filler, so only the DeltaNet recurrence can carry the answer. far-lift
(far-open - closed) is recall a windowed transformer cannot produce.

Usage:
    python -m eval.grounded_qa --checkpoint hvn_edge_chat.pt
    python -m eval.grounded_qa --checkpoint hvn_edge_chat.pt --far-probe
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402
from hvn.vocab import GPT2_EOS_ID, resolve_eos_id  # noqa: E402

# Legacy alias only. `generate` resolves EOS from the model/tokenizer it is
# actually given (#136) -- a 32K model's EOS is 0, and stopping on 50256 would
# mean never stopping at all.
EOS = GPT2_EOS_ID
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
    eos = resolve_eos_id(model, tok)
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
        if nxt == eos:
            break
        out.append(nxt)
        x = torch.cat([x, torch.tensor([[nxt]], device=device)], dim=1)
    return tok.decode(out).strip()


def hit(answer: str, keywords: list[str]) -> bool:
    a = answer.lower()
    return any(k.lower() in a for k in keywords)


# ---------------------------------------------------------------------------
# Beyond-window probe (discriminates the hybrid from a plain sliding-window
# transformer). The default gate above puts the gold passage right next to the
# question (~30 tokens), so it always sits inside any sliding window — a pure
# SWA transformer would read it too. To prove the DeltaNet *recurrence* carries
# recall a window cannot, we push the passage > W tokens back with neutral
# filler and shrink the attention window so the passage falls OUTSIDE it. Only
# the O(1) recurrent state can then carry the answer forward.
#
# Kept within the 512 training length (small window, e.g. 128) so RoPE never
# extrapolates — this isolates the architecture, not long-context behaviour.
# ---------------------------------------------------------------------------

# Neutral distractor — must contain none of the ITEMS answer keywords
# (paris/shakespeare/100/celsius/portuguese/gravity/light/escape/...).
_FILLER_SENT = ("Quarterly logistics notes: the shipping schedule was revised, "
                "warehouse inventory counts were reconciled, and the vendor "
                "onboarding forms were filed on time by the operations team. ")


def build_filler(tok, n_tokens: int) -> str:
    """Neutral filler text of at least `n_tokens` GPT-2 tokens."""
    unit = tok.encode(_FILLER_SENT)
    reps = (n_tokens // max(1, len(unit))) + 1
    return tok.decode(unit * reps)


def set_attention_window(model, window: int | None) -> int:
    """Override every TransformerBlock's sliding window in place; return count."""
    n = 0
    for m in model.modules():
        if type(m).__name__ == "TransformerBlock" and hasattr(m, "window"):
            m.window = window
            n += 1
    return n


def far_probe(model, tok, device, args):
    """Beyond-window discrimination: closed vs near-open vs far-open.

    far-lift (= far - closed) is the metric a plain sliding-window transformer
    CANNOT earn: with the passage beyond the window, only the recurrent state
    carries it. near-open (passage inside the window) confirms the windowed
    attention still reads, isolating the window as the only difference.
    """
    W = args.probe_window
    n_attn = set_attention_window(model, W)
    filler = build_filler(tok, W + args.far_margin)
    filler_toks = len(tok.encode(filler))
    print(f"Beyond-window probe: window={W}, {n_attn} attention block(s) "
          f"overridden, filler={filler_toks} tokens.")
    if n_attn == 0:
        print("  (note: no TransformerBlock found — this is a pure-recurrent "
              "baseline; far-lift here measures the recurrence alone.)")
    print()

    closed_hits = near_hits = far_hits = 0
    for q, passage, kw in ITEMS:
        closed_prompt = USER_PREFIX + q + "\n" + ASST_PREFIX
        near_prompt = (USER_PREFIX +
                       "Answer the question using only the context below.\n\n"
                       f"Context: {passage}\n\nQuestion: {q}\n" + ASST_PREFIX)
        far_prompt = (USER_PREFIX +
                      "Answer the question using only the context below.\n\n"
                      f"Context: {passage}\n\n{filler}\n\n"
                      f"Question: {q}\n" + ASST_PREFIX)

        # sanity: how far is the passage end from the generation frontier?
        # (everything after the passage — filler + question + prefix — must
        # exceed W for the passage to fall outside the sliding window).
        tail = tok.encode(f"\n\n{filler}\n\nQuestion: {q}\n" + ASST_PREFIX)
        gap = len(tail)  # tokens between passage end and first generated token

        closed = generate(model, tok, device, closed_prompt)
        near = generate(model, tok, device, near_prompt)
        far = generate(model, tok, device, far_prompt)
        c_ok, n_ok, f_ok = hit(closed, kw), hit(near, kw), hit(far, kw)
        closed_hits += c_ok
        near_hits += n_ok
        far_hits += f_ok
        in_window = "IN-window" if gap < W else "BEYOND-window"
        print("=" * 72)
        print(f"Q: {q}   (expect: {kw})   passage {gap} toks back [{in_window}]")
        print(f"  closed-book   [{'HIT' if c_ok else 'miss'}]: {closed}")
        print(f"  near-open     [{'HIT' if n_ok else 'miss'}]: {near}")
        print(f"  far-open      [{'HIT' if f_ok else 'miss'}]: {far}")

    n = len(ITEMS)
    print("\n" + "=" * 72)
    print(f"PROBE RESULT (window={W}): closed {closed_hits}/{n}  |  "
          f"near-open {near_hits}/{n}  |  far-open {far_hits}/{n}")
    near_lift = near_hits - closed_hits
    far_lift = far_hits - closed_hits
    print(f"  near-lift (windowed attention reads): +{near_lift}/{n}")
    print(f"  far-lift  (recurrence beyond window): +{far_lift}/{n}   <-- the "
          "hybrid-vs-SWA discriminator")
    if far_lift > 0:
        print("  -> Recurrence carries recall the sliding window CANNOT: the "
              "hybrid earns its keep over a plain windowed transformer.")
    elif near_lift > 0:
        print("  -> Reads only inside the window (like a plain SWA transformer); "
              "the recurrence adds no beyond-window recall here.")
    else:
        print("  -> No grounding lift at all under the small window.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="hvn_edge_chat.pt")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--far-probe", action="store_true",
                    help="Add the beyond-window probe: push the gold passage "
                         "> --probe-window tokens back so only the DeltaNet "
                         "recurrence (not windowed attention) can carry it.")
    ap.add_argument("--probe-window", type=int, default=128,
                    help="Attention window used for --far-probe (kept < 512 so "
                         "RoPE never extrapolates). Passage is pushed beyond it.")
    ap.add_argument("--far-margin", type=int, default=48,
                    help="Extra filler tokens beyond the window, to keep the "
                         "passage outside it as generation extends the sequence.")
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

    if args.far_probe:
        return far_probe(model, tok, device, args)

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
