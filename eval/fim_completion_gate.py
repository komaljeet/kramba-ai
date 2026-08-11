"""Fill-in-the-Middle code-completion gate — what "does it complete code" means.

Author: Rambarun Komaljeet

Written **before** any code model is trained, deliberately. A 9-day run without a
success criterion produces a checkpoint nobody can grade, which is the position
this project has repeatedly ended up in.

Takes held-out code files the model never trained on, masks a span in the middle,
asks the model to fill it, and scores the result at three increasing bars:

``exact``
    the infill matches the deleted span character-for-character. Weak on its own:
    an LSP scores here, and so does memorisation.
``parses``
    the reconstructed file compiles. Still weak -- a wrong-but-syntactically-valid
    line passes -- but it separates "produces code" from "produces noise".
``behaviour``
    NOT scored here. ``eval/coding.py`` already runs HumanEval-style problems with
    hidden asserts in a sandboxed subprocess; that is the behavioural tier and is
    reused rather than rebuilt. Run it alongside this gate.

**The spread between the bars is the finding, not any single number.** A model at
90% parse and 5% exact is doing LSP-grade bracket-closing, which is not worth
demonstrating. That distinction is the whole reason this gate reports all of them.

Two controls, because a score without them cannot be interpreted:

* **known-answer** -- the model is shown the *unmasked* file and asked to
  reproduce the span it can already see. If that does not score near 100%, the
  harness is broken and no model number below it means anything.
* **floor** -- a trivial non-model baseline (empty completion, and copy-the-
  previous-line). If the model cannot beat the floor, the score is noise. Every
  trustworthy measurement in this repo has one; the ones that lacked one were
  retracted (#122, #132).

Usage::

    python -m eval.fim_completion_gate --checkpoint <ckpt> \
        --tokenizer data_cache/tokenizer_32k_fim \
        --corpus data_cache/code_corpus/stack_code.txt
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.fim import FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE      # noqa: E402
from eval.coding import _balanced, _valid_python             # noqa: E402

SEP = "<|endoftext|>"

#: The corpus is a deliberate 13-language mix (`results/codecomplete/corpus_plan.json`:
#: 30% python, 15% javascript, 10% java, ...), so a Python-only validator is
#: meaningless on it -- measured, 2 of 21,329 sampled files parse as Python. The
#: first version of this gate used `_valid_python` alone and its known-answer cell
#: caught it immediately: the ORACLE scored 0% parses. Kept as a comment because
#: "the validator must match the corpus" is the kind of thing that looks obvious
#: only after a control fails.
#:
#: `_balanced` (brackets) is language-agnostic and weak; `_valid_python` is strong
#: but applies to a third of the corpus. A case is only used if its ORIGINAL file
#: passes the chosen validator, which makes the oracle 100% by construction --
#: without that, a low parse score cannot be told apart from a broken metric.
VALIDATORS = {
    "python": _valid_python,
    "balanced": lambda s: _balanced(s),
}


def load_heldout(corpus: str, sep: str, n: int, tail_bytes: int,
                 min_tokens: int, tok) -> list[str]:
    """Files from the END of the corpus, which training reads from the start.

    Disjointness matters more than sample size here: scoring a model on files it
    trained on measures memorisation. The tail offset is recorded in the output
    so the split is reproducible and auditable rather than merely asserted.
    """
    size = os.path.getsize(corpus)
    start = max(0, size - tail_bytes)
    with open(corpus, "r", encoding="utf-8", errors="replace") as fh:
        fh.seek(start)
        text = fh.read()
    docs = [d for d in text.split(sep) if d.strip()]
    docs = docs[1:-1]          # both ends are truncated by the byte seek
    out = []
    for d in docs:
        if len(tok(d, add_special_tokens=False)["input_ids"]) >= min_tokens:
            out.append(d)
        if len(out) >= n:
            break
    return out


def make_fim_case(doc: str, tok, rng, max_ctx: int, span_lines: int):
    """Mask a whole-line span in the middle; return (prefix, middle, suffix).

    Lines, not tokens: a token-boundary cut produces half-identifiers that no
    model or human could complete, which would measure tokenisation rather than
    code understanding.
    """
    lines = doc.split("\n")
    if len(lines) < span_lines + 6:
        return None
    lo = rng.randint(3, max(3, len(lines) - span_lines - 3))
    full_pre = "\n".join(lines[:lo]) + "\n"
    middle = "\n".join(lines[lo:lo + span_lines]) + "\n"
    full_suf = "\n".join(lines[lo + span_lines:])

    # Keep the PROMPT inside the model's context: trim the OUTER ends, keeping
    # the text adjacent to the hole, which is what the completion depends on.
    # The FULL halves are returned as well and are what validity is scored on --
    # scoring the trimmed version would compile a mid-file fragment.
    def trim(s, budget, keep_end):
        ids = tok(s, add_special_tokens=False)["input_ids"]
        if len(ids) <= budget:
            return s
        ids = ids[-budget:] if keep_end else ids[:budget]
        return tok.decode(ids)

    half = max_ctx // 2
    return (trim(full_pre, half, keep_end=True), middle,
            trim(full_suf, half, keep_end=False), full_pre, full_suf)


@torch.no_grad()
def infill(model, tok, device, prefix, suffix, max_new, eos_ids):
    """Greedy FIM decode. Greedy on purpose: a gate must be reproducible, and
    sampling would make the score depend on a seed rather than the model."""
    prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
    ids = tok(prompt, add_special_tokens=False)["input_ids"]
    x = torch.tensor([ids], device=device)
    out = []
    states = model.init_state(1, device=device)
    logits = None
    for t in range(x.size(1)):
        logits, states = model.forward_step(x[:, t:t + 1], states, position_idx=t)
    for i in range(max_new):
        nxt = int(logits[0, -1].argmax())
        if nxt in eos_ids:
            break
        out.append(nxt)
        logits, states = model.forward_step(
            torch.tensor([[nxt]], device=device), states, position_idx=x.size(1) + i)
    return tok.decode(out)


def score(cases, produce, validator) -> dict:
    """produce(prefix, suffix, middle) -> predicted middle.

    Validity is checked on the FULL original document with the span replaced --
    not on the trimmed prompt. The prompt is trimmed to fit the model's context,
    so reassembling it would compile a mid-file fragment, which fails for reasons
    that have nothing to do with the completion.
    """
    n = exact = valid = 0
    for pre, middle, suf, full_pre, full_suf in cases:
        pred = produce(pre, suf, middle)
        n += 1
        if pred.strip() == middle.strip():
            exact += 1
        if validator(full_pre + pred + full_suf):
            valid += 1
    return {"n": n, "exact_pct": exact / n * 100 if n else 0.0,
            "valid_pct": valid / n * 100 if n else 0.0}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=None,
                   help="omit to run controls only -- useful for validating the "
                        "gate itself before any code model exists")
    p.add_argument("--tokenizer", default="data_cache/tokenizer_32k_fim")
    p.add_argument("--corpus", default="data_cache/code_corpus/stack_code.txt")
    p.add_argument("--separator", default=SEP)
    p.add_argument("--n-cases", type=int, default=100)
    p.add_argument("--span-lines", type=int, default=2)
    p.add_argument("--min-tokens", type=int, default=200)
    p.add_argument("--max-ctx", type=int, default=768)
    p.add_argument("--max-new", type=int, default=80)
    p.add_argument("--tail-bytes", type=int, default=20_000_000)
    p.add_argument("--validator", choices=list(VALIDATORS), default="balanced",
                   help="'python' is strong but the corpus is only ~30%% Python; "
                        "'balanced' (brackets) is language-agnostic and weak. Cases "
                        "are filtered to files their validator already accepts.")
    p.add_argument("--seed", type=int, default=11)
    p.add_argument("--out", default="results/fim_gate/gate.json")
    args = p.parse_args()

    import random
    rng = random.Random(args.seed)
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained(args.tokenizer)

    validator = VALIDATORS[args.validator]
    docs = load_heldout(args.corpus, args.separator, args.n_cases * 40,
                        args.tail_bytes, args.min_tokens, tok)
    # Only files whose ORIGINAL passes the validator become cases. That makes the
    # oracle 100% valid by construction -- without it, a low score cannot be told
    # apart from a validator that does not fit the corpus, which is exactly the
    # failure the first run of this gate hit (python validator on a 13-language mix).
    kept = [d for d in docs if validator(d)]
    cases = []
    for d in kept:
        c = make_fim_case(d, tok, rng, args.max_ctx, args.span_lines)
        if c:
            cases.append(c)
        if len(cases) >= args.n_cases:
            break
    print(f"[fim] {len(cases)} cases from {len(kept)}/{len(docs)} held-out files "
          f"passing the '{args.validator}' validator, last "
          f"{args.tail_bytes/1e6:.0f} MB of {args.corpus}")
    if not cases:
        sys.exit("no usable cases -- check the separator and --min-tokens")

    results = {}

    # ---- floor arms: no model at all -------------------------------------
    results["floor_empty"] = score(cases, lambda pre, suf, mid: "", validator)
    results["floor_copy_prev_line"] = score(
        cases, lambda pre, suf, mid: (pre.rstrip("\n").split("\n")[-1] + "\n"),
        validator)
    print(f"[fim] floor (empty)            "
          f"exact {results['floor_empty']['exact_pct']:5.1f}%  "
          f"valid {results['floor_empty']['valid_pct']:5.1f}%")
    print(f"[fim] floor (copy prev line)   "
          f"exact {results['floor_copy_prev_line']['exact_pct']:5.1f}%  "
          f"valid {results['floor_copy_prev_line']['valid_pct']:5.1f}%")

    # ---- known-answer: hand back the span that was removed ---------------
    results["known_answer"] = score(cases, lambda pre, suf, mid: mid, validator)
    ka = results["known_answer"]
    print(f"[fim] known-answer (oracle)    "
          f"exact {ka['exact_pct']:5.1f}%  valid {ka['valid_pct']:5.1f}%")
    void = ka["exact_pct"] < 99.0
    if void:
        print("[fim] *** GATE BROKEN: feeding back the true span does not score "
              "~100% exact. The scorer or the case construction is wrong; no "
              "model number from this harness is meaningful.")

    # ---- the model -------------------------------------------------------
    if args.checkpoint and not void:
        import train_edge_conversational as tec
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, step, _ = tec.load_checkpoint_for_resume(args.checkpoint, device)
        model.eval()

        # A vocabulary mismatch is silent and catastrophic: the checkpoint's ids
        # mean different strings under a different tokenizer, so the model would
        # produce fluent-looking garbage and the gate would report a real-looking
        # low score. Checked, not assumed. (The existing 83M is GPT-2/50257 and
        # the code tokenizer is 32003 with three FIM sentinels, so this fires on
        # the most obvious thing anyone would try first.)
        n_vocab = int(getattr(model, "vocab_size", 0))
        n_tok = len(tok)
        if n_vocab != n_tok:
            sys.exit(
                f"vocabulary mismatch: checkpoint has {n_vocab:,} embeddings, "
                f"tokenizer '{args.tokenizer}' has {n_tok:,}. The ids would mean "
                f"different strings and the score would be meaningless. This gate "
                f"needs a model TRAINED with the FIM tokenizer -- FIM is a "
                f"training-data transform (hvn/fim.py) and cannot be added to a "
                f"finished model.")
        missing = [t for t in (FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE)
                   if len(tok(t, add_special_tokens=False)["input_ids"]) != 1]
        if missing:
            sys.exit(f"tokenizer lacks single-token FIM sentinels: {missing}. "
                     f"Build one with scripts/build_fim_tokenizer.py.")
        eos_ids = set(tok(SEP, add_special_tokens=False)["input_ids"])
        print(f"[fim] model {args.checkpoint} (step {step}) on {device}")
        results["model"] = score(
            cases, lambda pre, suf, mid: infill(model, tok, device, pre, suf,
                                                args.max_new, eos_ids), validator)
        m = results["model"]
        print(f"[fim] MODEL                    "
              f"exact {m['exact_pct']:5.1f}%  valid {m['valid_pct']:5.1f}%")
        best_floor = max(results["floor_empty"]["exact_pct"],
                         results["floor_copy_prev_line"]["exact_pct"])
        if m["exact_pct"] <= best_floor:
            verdict = (f"AT OR BELOW THE FLOOR ({m['exact_pct']:.1f}% vs "
                       f"{best_floor:.1f}%) -- the model is not completing code.")
        elif m["exact_pct"] < 10 and m["valid_pct"] > 60:
            verdict = (f"LSP-GRADE: {m['valid_pct']:.1f}% valid but only "
                       f"{m['exact_pct']:.1f}% exact -- it produces syntactically "
                       f"valid code that is usually not the right code. This is "
                       f"the regime an editor already covers; not presentable.")
        else:
            verdict = (f"exact {m['exact_pct']:.1f}%, valid {m['valid_pct']:.1f}% "
                       f"against a {best_floor:.1f}% floor. Run eval/coding.py's "
                       f"executable pass@k for the behavioural tier before calling "
                       f"it real.")
        print(f"\n[fim] VERDICT: {verdict}")
        results["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "cases": len(cases),
                   "gate_broken": void, "results": results}, fh, indent=2)
    print(f"[fim] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
