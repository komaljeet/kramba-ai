"""RAG probe — can retrieval supply the knowledge the weights don't have? (Issue #137)

Written by: Rambarun Komaljeet

Three arms over the SAME questions, so the comparison is paired:

``closed``   question only, no context. Must fail — the facts are about INVENTED
             entities that appear in no pretraining corpus. This arm is the
             control that proves the questions are genuinely unanswerable from
             the weights, which is what makes any lift attributable to retrieval.
``rag``      BM25 retrieves top-k from a corpus of thousands of facts.
``oracle``   the correct fact handed over directly, bypassing retrieval.

**The oracle arm is the point of the design.** It is the ceiling RAG can reach on
this model, and it separates "the retriever missed" from "83M cannot use a
retrieved passage well enough for this to work". If the oracle arm does not clear
the bar, no retriever will rescue it and the track closes there — cheaply, before
anyone builds an embedding pipeline.

**Why invented pseudo-words rather than a real QA set.** `data/
grounded_qa_expanded.jsonl` restates its passage inside the question ("Fill in
the blank using the passage above: ..."), so retrieval would be trivial and
closed-book would be meaningless — the answer is already in the prompt. Real-world
QA carries the opposite problem: a hit may come from pretraining rather than from
the passage. Invented entities remove both confounds at once, which is the same
reasoning that made `eval/synthetic_recall_probe.py` the primary gate in #122.

Reported per arm: exact whole-word accuracy, retrieval recall@k, the token budget
actually consumed, and latency. Retrieval quality is reported SEPARATELY from
answer quality — without that split, "RAG didn't help" cannot be attributed.

Usage
-----
    python eval/rag_probe.py --checkpoint checkpoints/a4_prodigy/hybrid_a4_step2000.pt \
        --n-trials 300 --corpus-size 2000 --top-k 3
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.synthetic_recall_probe import (  # noqa: E402
    ASST_PREFIX, OPEN_BOOK_INSTRUCTION, USER_PREFIX,
    generate, make_pair, whole_word_hit,
)
from hvn.retrieval import BM25Index, fit_context  # noqa: E402
from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402


def build_corpus(n_facts: int, seed: int):
    """A knowledge base of invented facts, plus the (entity, value) key."""
    rng = torch.Generator().manual_seed(seed)
    used: set[str] = set()
    facts, pairs = [], []
    for _ in range(n_facts):
        e, v = make_pair(rng, used)
        facts.append(f"The {e} is called {v}.")
        pairs.append((e, v))
    return facts, pairs


def prompt_for(entity: str, context_passages) -> str:
    """Rendered in the model's ACTUAL trained template.

    #122 recorded a fully retracted result caused by probing outside the trained
    format — a raw-token version scored a flat 0% everywhere, which was a format
    artifact and not a recall measurement. So this reuses
    ``synthetic_recall_probe``'s exact strings rather than inventing a prompt.
    """
    if not context_passages:
        return (USER_PREFIX + f"Question: What is the {entity}?\n" + ASST_PREFIX)
    context = " ".join(context_passages)
    return (USER_PREFIX + OPEN_BOOK_INSTRUCTION +
            f"Context: {context}\n\nQuestion: What is the {entity}?\n" + ASST_PREFIX)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-trials", type=int, default=300)
    ap.add_argument("--corpus-size", type=int, default=2000,
                    help="facts in the retrieval database (the haystack)")
    ap.add_argument("--top-k", type=int, default=3)
    ap.add_argument("--context-budget", type=int, default=320,
                    help="token budget for retrieved passages. The checkpoints "
                         "were trained at seq_len=512 and this must leave room "
                         "for the instruction, question and answer (#135).")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--arms", nargs="+", default=["closed", "rag", "oracle"])
    ap.add_argument("--out", default="results/rag/rag_probe.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md).")
    device = torch.device("cuda")

    facts, pairs = build_corpus(args.corpus_size, args.seed)
    t0 = time.time()
    index = BM25Index(facts)
    build_sec = time.time() - t0
    print(f"[rag] indexed {len(index)} facts in {build_sec:.2f}s")

    model, _step, _ck = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")

    # The trial questions are a fixed subset of the corpus, identical for every
    # arm, so the arms are paired rather than merely comparable.
    step = max(1, len(pairs) // args.n_trials)
    trials = [pairs[i * step] for i in range(min(args.n_trials, len(pairs) // step))]
    print(f"[rag] {len(trials)} trials over a {len(index)}-fact corpus, "
          f"top-k={args.top_k}, context budget {args.context_budget} tokens")

    results = {}
    for arm in args.arms:
        hits = 0
        retrieval_hits = 0
        ctx_tokens = []
        n_passages = []
        t_start = time.time()
        for entity, value in trials:
            if arm == "closed":
                passages = []
            elif arm == "oracle":
                passages = [f"The {entity} is called {value}."]
            else:
                got = index.retrieve(f"What is the {entity}?", k=args.top_k)
                # Retrieval scored independently of the answer: this is the
                # number that says whether a miss was the retriever's fault.
                if any(f"The {entity} is called" in p for p in got):
                    retrieval_hits += 1
                passages = fit_context(got, tok, args.context_budget)
            prompt = prompt_for(entity, passages)
            ctx_tokens.append(len(tok.encode(prompt)))
            n_passages.append(len(passages))
            out = generate(model, tok, device, prompt)
            if whole_word_hit(out, value):
                hits += 1
        elapsed = time.time() - t_start
        acc = hits / max(len(trials), 1)
        results[arm] = {
            "accuracy": acc,
            "hits": hits,
            "n_trials": len(trials),
            "retrieval_recall_at_k": (retrieval_hits / len(trials)) if arm == "rag" else None,
            "mean_prompt_tokens": sum(ctx_tokens) / max(len(ctx_tokens), 1),
            "max_prompt_tokens": max(ctx_tokens) if ctx_tokens else 0,
            "mean_passages": sum(n_passages) / max(len(n_passages), 1),
            "sec_per_query": elapsed / max(len(trials), 1),
        }
        extra = ""
        if arm == "rag":
            extra = f" | retrieval recall@{args.top_k}: {retrieval_hits/len(trials):.1%}"
        print(f"[rag] {arm:<7} accuracy {acc:>6.1%} ({hits}/{len(trials)})"
              f" | prompt {results[arm]['mean_prompt_tokens']:.0f} tok"
              f" | {results[arm]['sec_per_query']:.2f} s/query{extra}", flush=True)

    print("\n[rag] ================= VERDICT =================")
    if "closed" in results and "rag" in results:
        lift = results["rag"]["accuracy"] - results["closed"]["accuracy"]
        print(f"[rag] retrieval lift (rag - closed): {lift:+.1%}")
    if "oracle" in results and "rag" in results:
        gap = results["oracle"]["accuracy"] - results["rag"]["accuracy"]
        print(f"[rag] oracle - rag: {gap:+.1%} "
              f"-> {'retrieval is the bottleneck' if gap > 0.05 else 'retrieval is NOT the bottleneck'}")
    if "oracle" in results:
        print(f"[rag] CEILING: with a perfect passage this model answers "
              f"{results['oracle']['accuracy']:.1%} of the time. No retriever "
              f"can beat that.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({
            "checkpoint": args.checkpoint.replace("\\", "/"),
            "corpus_size": args.corpus_size,
            "top_k": args.top_k,
            "context_budget": args.context_budget,
            "index_build_sec": build_sec,
            "arms": results,
        }, fh, indent=2)
    print(f"[rag] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
