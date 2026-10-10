"""Fixed chat-coherence eval: score a chat model on a frozen prompt set.

Written by: Rambarun Komaljeet

Loss and perplexity cannot tell a model that answers the question from one that
emits fluent, generic assistant boilerplate (docs/state.md, "content coherence,
not decoding mechanics"). This scores replies to ``eval/chat_prompts_v1.json``
on three criteria, per prompt:

  on_topic       HEURISTIC. At least ``min_hits`` of the prompt's answer keywords
                 appear in the reply, the reply is not near-identical to the
                 model's replies to OTHER prompts (boilerplate detector: word
                 3-gram Jaccard >= GENERIC_JACCARD), and it respects the prompt's
                 ``max_words`` if set. Lexical only: a reply that says "Paris is
                 not the capital" passes. Correctness and real relevance need a
                 human (or LLM-judge) read of the skim file.
  non_repetitive AUTOMATIC. Distinct word-3-gram ratio >= REP_MIN_DISTINCT3
                 (replies under REP_MIN_TOKENS tokens pass by definition).
  terminates     AUTOMATIC. The model emitted its own stop token before
                 ``--max-new`` tokens. Hitting the limit = rambled or looped.

``coherent`` = all three. Also reported: the **chance on-topic rate** -- each
reply scored against every OTHER prompt's keywords. If matched on-topic is not
far above chance, the keyword metric is not measuring anything; say so rather
than trust it.

Generation goes through ``serve_openai.generate_ids`` (the local serving loop:
full re-forward, same sampler) for BOTH EdgeHVN checkpoints and HF reference
models, so the two sides differ only in weights and chat template. Default
decode is greedy (deterministic: re-running gives the same score) with the
serving default repetition penalty 1.3.

Usage
-----
    python eval/chat_coherence.py --checkpoint checkpoints/hybrid_chat_v1.pt
    python eval/chat_coherence.py --hf-model checkpoints/hf/SmolLM2-135M-Instruct
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)

DEFAULT_PROMPTS = os.path.join(_ROOT, "eval", "chat_prompts_v1.json")
REP_MIN_DISTINCT3 = 0.75
REP_MIN_TOKENS = 12
GENERIC_JACCARD = 0.5
GENERIC_MIN_TOKENS = 6


# ---------------------------------------------------------------------------
# Scoring (pure functions -- unit-tested in tests/test_chat_coherence_eval.py)
# ---------------------------------------------------------------------------

def load_prompts(path: str = DEFAULT_PROMPTS) -> tuple[dict, str]:
    with open(path, "rb") as fh:
        raw = fh.read()
    return json.loads(raw), hashlib.sha256(raw).hexdigest()


def words(text: str) -> list[str]:
    return re.findall(r"\w+|[^\w\s]", text.lower())


def keyword_hits(text: str, keywords: list[str]) -> list[str]:
    """Keywords present at a word start. Numeric keywords must also END at a
    word boundary ("4" must not match "40"); alphabetic ones may be prefixes
    ("scatter" matches "scattering")."""
    low = text.lower()
    hits = []
    for kw in keywords:
        pat = r"(?<!\w)" + re.escape(kw.lower())
        if kw[-1].isdigit():
            pat += r"(?!\d)"
        if re.search(pat, low):
            hits.append(kw)
    return hits


def distinct_ngram_ratio(toks: list[str], n: int = 3) -> float:
    grams = [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 1.0


def _ngrams(toks: list[str], n: int = 3) -> set:
    return {tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)}


def generic_similarity(texts: list[str]) -> list[float]:
    """For each reply, max word-3-gram Jaccard vs the model's OTHER replies.
    High = the reply is (near-)the same text regardless of the question."""
    grams = [_ngrams(words(t)) for t in texts]
    out = []
    for i, gi in enumerate(grams):
        best = 0.0
        if gi and len(words(texts[i])) >= GENERIC_MIN_TOKENS:
            for j, gj in enumerate(grams):
                if j != i and gj:
                    best = max(best, len(gi & gj) / len(gi | gj))
        out.append(best)
    return out


def score_replies(prompts: list[dict], replies: list[dict]) -> dict:
    """``replies[i]`` = {"text": str, "hit_eos": bool} for ``prompts[i]``."""
    texts = [r["text"] for r in replies]
    gen_sim = generic_similarity(texts)
    rows = []
    for p, r, gs in zip(prompts, replies, gen_sim):
        toks = words(r["text"])
        n_words = len(r["text"].split())
        hits = keyword_hits(r["text"], p["keywords"])
        kw_ok = len(hits) >= p.get("min_hits", 1)
        len_ok = n_words <= p["max_words"] if "max_words" in p else True
        generic = gs >= GENERIC_JACCARD
        d3 = distinct_ngram_ratio(toks)
        row = {
            "id": p["id"], "category": p["category"],
            "keyword_hits": hits, "generic_sim": round(gs, 3),
            "distinct3": round(d3, 3), "n_words": n_words,
            "hit_eos": bool(r["hit_eos"]),
            "on_topic": bool(kw_ok and len_ok and not generic and toks),
            "non_repetitive": bool(len(toks) < REP_MIN_TOKENS
                                   or d3 >= REP_MIN_DISTINCT3),
            "terminates": bool(r["hit_eos"]),
        }
        row["coherent"] = row["on_topic"] and row["non_repetitive"] and row["terminates"]
        rows.append(row)

    n = len(rows)
    # Chance control: reply i against every other prompt's keywords.
    pairs = hits = 0
    for i, r in enumerate(replies):
        for j, p in enumerate(prompts):
            if j != i:
                pairs += 1
                hits += len(keyword_hits(r["text"], p["keywords"])) >= p.get("min_hits", 1)
    summary = {c: sum(r[c] for r in rows) / n
               for c in ("on_topic", "non_repetitive", "terminates", "coherent")}
    summary["chance_on_topic"] = hits / pairs if pairs else 0.0
    summary["by_category"] = {
        cat: {"n": len(rs), "coherent": sum(r["coherent"] for r in rs),
              "on_topic": sum(r["on_topic"] for r in rs)}
        for cat in sorted({r["category"] for r in rows})
        for rs in [[r for r in rows if r["category"] == cat]]}
    return {"rows": rows, "summary": summary}


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _load(args, device):
    """Returns (model, render(messages)->ids, decode(ids)->str, eos)."""
    import serve_openai
    if args.checkpoint:
        from transformers import GPT2TokenizerFast
        from train_edge_conversational import load_checkpoint_for_resume
        model, _step, _ck = load_checkpoint_for_resume(args.checkpoint, device)
        tok = GPT2TokenizerFast.from_pretrained("gpt2")
        eos = getattr(model, "eos_token_id", 50256)
        serve_openai._STATE["index"] = None          # no RAG in this eval
        render = lambda msgs: tok.encode(serve_openai.build_prompt(msgs))
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.hf_model)
        model = AutoModelForCausalLM.from_pretrained(args.hf_model).to(device)
        eos = tok.eos_token_id
        render = lambda msgs: tok.encode(tok.apply_chat_template(
            msgs, add_generation_prompt=True, tokenize=False),
            add_special_tokens=False)
    model.eval()
    return model, render, (lambda ids: tok.decode(ids).strip()), eos


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--checkpoint", help="EdgeHVN .pt (GPT-2 tokenizer, "
                     "<|user|>/<|assistant|> template)")
    src.add_argument("--hf-model", help="HF causal LM dir/name (uses its own "
                     "chat template)")
    ap.add_argument("--prompts", default=DEFAULT_PROMPTS)
    ap.add_argument("--name", default=None, help="output file stem")
    ap.add_argument("--out-dir", default=os.path.join(_ROOT, "results", "chat_eval"))
    ap.add_argument("--max-new", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.0,
                    help="0 = greedy (deterministic, the default)")
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.3)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    if args.checkpoint:
        # Must precede CUDA init: it imports pandas/pyarrow, which abort the
        # process silently (0xC0000374) once CUDA is armed -- see its header.
        import train_edge_conversational  # noqa: F401
    import torch
    from serve_openai import generate_ids
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data, sha = load_prompts(args.prompts)
    prompts = data["prompts"]
    name = args.name or os.path.splitext(os.path.basename(
        (args.checkpoint or args.hf_model).rstrip("/\\")))[0]

    model, render, decode, eos = _load(args, device)
    print(f"[chat-eval] {name}: {len(prompts)} prompts, eos={eos}, "
          f"{data['version']} sha256={sha[:12]}", flush=True)
    replies = []
    t0 = time.time()
    for i, p in enumerate(prompts):
        ids, stopped = generate_ids(
            model, render(p["messages"]), eos=eos, device=device,
            max_new=args.max_new, temperature=args.temperature,
            top_p=args.top_p, repetition_penalty=args.repetition_penalty,
            decode=decode)
        replies.append({"text": decode(ids), "hit_eos": stopped,
                        "n_tokens": len(ids)})
        print(f"[chat-eval] {i + 1}/{len(prompts)} {p['id']} "
              f"({len(ids)} tok, eos={stopped})", flush=True)

    scored = score_replies(prompts, replies)
    s = scored["summary"]
    os.makedirs(args.out_dir, exist_ok=True)
    stem = os.path.join(args.out_dir, name)
    meta = {"model": args.checkpoint or args.hf_model, "prompts": args.prompts,
            "prompts_version": data["version"], "prompts_sha256": sha,
            "decode": {k: getattr(args, k) for k in
                       ("max_new", "temperature", "top_p", "repetition_penalty", "seed")},
            "seconds": round(time.time() - t0, 1)}
    with open(stem + ".json", "w", encoding="utf-8") as fh:
        json.dump({"meta": meta, "summary": s,
                   "results": [dict(row, prompt=p["messages"], reply=r["text"])
                               for row, p, r in zip(scored["rows"], prompts, replies)]},
                  fh, indent=2, ensure_ascii=False)

    # Skim file: one block per prompt, verdicts up front.
    mark = lambda b: "ok " if b else "--"
    lines = [f"{name}  |  {data['version']}  |  decode {meta['decode']}",
             f"coherent {s['coherent']:.0%}  on_topic {s['on_topic']:.0%} "
             f"(chance {s['chance_on_topic']:.0%})  non_repetitive "
             f"{s['non_repetitive']:.0%}  terminates {s['terminates']:.0%}",
             "on_topic is a keyword heuristic -- read the replies to judge "
             "correctness/relevance.", ""]
    for row, p, r in zip(scored["rows"], prompts, replies):
        lines.append(f"=== {row['id']} [{row['category']}]  on_topic:{mark(row['on_topic'])} "
                     f"non_rep:{mark(row['non_repetitive'])} "
                     f"terminates:{mark(row['terminates'])}  "
                     f"(hits={row['keyword_hits']} generic={row['generic_sim']} "
                     f"d3={row['distinct3']} words={row['n_words']})")
        for m in p["messages"]:
            lines.append(f"  {m['role'].upper()}: {m['content']}")
        lines.append(f"  REPLY: {r['text']}")
        lines.append("")
    with open(stem + ".txt", "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    print(f"[chat-eval] {name}: coherent {s['coherent']:.0%} | on_topic "
          f"{s['on_topic']:.0%} (chance {s['chance_on_topic']:.0%}) | "
          f"non_repetitive {s['non_repetitive']:.0%} | terminates "
          f"{s['terminates']:.0%} | {meta['seconds']}s", flush=True)
    print(f"[chat-eval] wrote {stem}.json and {stem}.txt", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
