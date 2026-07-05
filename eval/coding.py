"""
Coding evaluation harness — per-language held-out completion scoring (Issue #55)
================================================================================

Milestone 2, Issue E of 5. Gives each curriculum stage a **measurable** score
instead of vibes: a per-language, HumanEval-style held-out code-completion eval
for the curriculum order

    HTML → CSS → JS → PHP → Python → DevOps        (the #53 stage order)

so the #54 orchestrator can log a real number after each stage and curriculum
gains become quantifiable.

Metric strategy (per the issue brief) — pick the strongest signal a language's
tooling allows, and *always* report the tokenization-invariant fallback:

* **Exec-based pass@k** where a runtime exists (Python always; JavaScript when
  ``node`` is installed; PHP when ``php`` is installed). A small bundled set of
  HumanEval-style problems (a function stub + hidden asserts) is completed by the
  model, executed in an isolated subprocess with a timeout, and graded pass/fail.
* **Structural / compile validity** where no runtime is available (HTML, CSS,
  DevOps, and as the fallback for JS/PHP off-box). The model completes a held-out
  prefix and the concatenation is checked for structural validity (balanced tags
  / braces, ``compile()`` for Python, ``yaml.safe_load`` for manifests, ...).
* **Held-out bits-per-char / PPL** is *always* computed (reusing the #51 metric)
  — tokenization-invariant, so it compares fairly across tokenizers and gives a
  smooth gradient even when a tiny loop-validation model can't yet emit runnable
  code.

Two entry points
----------------
* :func:`score_stage` / :func:`evaluate_coding` — the **callable API** the #54
  orchestrator plugs into (``auto_train.evaluate_stage`` delegates to it). Given a
  loaded model + tokenizer + a language's held-out text, returns a JSON-
  serialisable metrics dict.
* CLI — ``python -m eval.coding --checkpoint X --lang python`` (or ``--lang all``)
  loads a checkpoint, pulls each language's held-out split from the #53 loader,
  and prints a report in the same style as the other ``eval/`` harnesses.

Safety note. Exec-based scoring runs **model-generated** code. It is always run in
a throwaway temp dir via ``subprocess`` with a short timeout, and can be disabled
with ``allow_exec=False`` (the orchestrator path defaults to structural-only so
unattended curriculum runs stay fast and side-effect-free; the exec pass@k is an
explicit ``python -m eval.coding`` pass). O(1) inference is untouched — this only
reads a trained checkpoint.

Usage::

    python -m eval.coding --checkpoint hvn_chat_checkpoint.pt --lang python
    python -m eval.coding --checkpoint hvn_chat_checkpoint.pt --lang all --json baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Callable

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass

import torch

from hvn.core import HamiltonianVortexNetwork
# Reuse the #51 held-out metric + the shared completion helper rather than
# duplicating them (keeps the #55 score comparable to #51/#54 numbers).
from build_chat_checkpoint import heldout_bpc, complete, is_non_degenerate

# The curriculum stages are exactly the languages we score.
CODING_LANGS: list[str] = ["html", "css", "js", "php", "python", "devops"]


# ---------------------------------------------------------------------------
# Structural / compile validators (no external runtime required)
# ---------------------------------------------------------------------------
#
# Each validator answers a single question: *is this a structurally plausible
# snippet of the language?* They are deliberately coarse — they reward a model
# that learned the language's shape (balanced tags/braces, valid syntax) and
# reject gibberish, giving a smooth gradient as a stage trains. They are NOT
# full parsers.

_HTML_VOID = {
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
}
_HTML_TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z][a-zA-Z0-9-]*)[^>]*?(/?)\s*>")


def _valid_html(code: str) -> bool:
    """True if the (open/close) tags balance on a stack, ignoring void/self-closing."""
    tags = _HTML_TAG_RE.findall(code)
    if not tags:
        return False
    stack: list[str] = []
    for closing, name, selfclose in tags:
        name = name.lower()
        if name in _HTML_VOID or selfclose:
            continue
        if closing:
            if stack and stack[-1] == name:
                stack.pop()
            else:
                return False  # mismatched / stray close tag
        else:
            stack.append(name)
    return not stack


def _balanced(code: str, pairs: str = "(){}[]") -> bool:
    """True if the given bracket pairs are balanced (quote/comment-agnostic — a
    coarse but robust check for the tiny-model regime)."""
    opening = pairs[0::2]
    closing = pairs[1::2]
    match = {c: o for o, c in zip(opening, closing)}
    stack: list[str] = []
    for ch in code:
        if ch in opening:
            stack.append(ch)
        elif ch in closing:
            if not stack or stack[-1] != match[ch]:
                return False
            stack.pop()
    return not stack


def _valid_css(code: str) -> bool:
    """Balanced braces/parens and at least one ``selector { prop: value }`` block."""
    if not _balanced(code, "(){}"):
        return False
    return bool(re.search(r"\{[^{}]*:[^{}]*\}", code, re.S))


def _valid_js(code: str) -> bool:
    """Balanced brackets + a JS-ish keyword (structural fallback when node absent)."""
    if not _balanced(code, "(){}[]"):
        return False
    return bool(re.search(r"\b(function|const|let|var|class|return|export|=>)\b", code))


def _valid_php(code: str) -> bool:
    """Contains a PHP open tag and balanced braces (structural fallback)."""
    if "<?php" not in code and "<?=" not in code:
        return False
    return _balanced(code, "(){}[]")


def _valid_python(code: str) -> bool:
    """True if the snippet compiles (no SyntaxError)."""
    try:
        compile(code, "<coding-eval>", "exec")
        return True
    except (SyntaxError, ValueError):
        return False


def _valid_devops(code: str) -> bool:
    """Valid if it parses as YAML, or looks like a Dockerfile / shell script.

    DevOps stage mixes YAML manifests, Dockerfiles and shell — accept any of the
    three shapes so a model that learned the ops idiom scores, while free-text
    gibberish (no ``key:`` lines, no ``FROM``, no shebang) is rejected.
    """
    stripped = code.strip()
    if not stripped:
        return False
    # Dockerfile
    if re.search(r"^\s*(FROM|RUN|CMD|COPY|WORKDIR|ENTRYPOINT|EXPOSE)\b", code, re.M):
        return True
    # Shell script
    if stripped.startswith("#!") or re.search(r"^\s*(echo|export|cd|set)\s", code, re.M):
        return True
    # YAML (best-effort — pyyaml is optional)
    try:
        import yaml  # type: ignore

        obj = yaml.safe_load(code)
        if isinstance(obj, (dict, list)):
            return True
    except Exception:
        pass
    # Structural YAML fallback: at least a couple of ``key:`` lines.
    return len(re.findall(r"^\s*[\w.-]+:", code, re.M)) >= 2


STRUCTURAL_VALIDATORS: dict[str, Callable[[str], bool]] = {
    "html": _valid_html,
    "css": _valid_css,
    "js": _valid_js,
    "php": _valid_php,
    "python": _valid_python,
    "devops": _valid_devops,
}


# ---------------------------------------------------------------------------
# Runtime detection + exec-based HumanEval-style problems
# ---------------------------------------------------------------------------

def _runtime_for(lang: str) -> str | None:
    """Return the runtime label available for *lang* (exec-capable), else None."""
    if lang == "python":
        return "python"
    if lang == "js" and shutil.which("node"):
        return "node"
    if lang == "php" and shutil.which("php"):
        return "php"
    return None


# A small bundled HumanEval-style set: a stub the model must complete, plus hidden
# tests. Kept tiny so a full six-language eval runs on the 6 GB box in seconds.
PYTHON_PROBLEMS: list[dict] = [
    {"prompt": "def add(a, b):\n    \"\"\"Return the sum of a and b.\"\"\"\n",
     "test": "assert add(2, 3) == 5\nassert add(-1, 1) == 0\n"},
    {"prompt": "def is_even(n):\n    \"\"\"Return True if n is even, else False.\"\"\"\n",
     "test": "assert is_even(4) is True\nassert is_even(7) is False\n"},
    {"prompt": "def largest(xs):\n    \"\"\"Return the largest element of the non-empty list xs.\"\"\"\n",
     "test": "assert largest([1, 5, 2]) == 5\nassert largest([-3, -1, -7]) == -1\n"},
    {"prompt": "def reverse_string(s):\n    \"\"\"Return the string s reversed.\"\"\"\n",
     "test": "assert reverse_string('abc') == 'cba'\nassert reverse_string('') == ''\n"},
    {"prompt": "def factorial(n):\n    \"\"\"Return n! for integer n >= 0.\"\"\"\n",
     "test": "assert factorial(0) == 1\nassert factorial(4) == 24\n"},
]

JS_PROBLEMS: list[dict] = [
    {"prompt": "// Return the sum of a and b.\nfunction add(a, b) {\n",
     "test": "if (add(2, 3) !== 5) throw new Error('add');\n"},
    {"prompt": "// Return true if n is even.\nfunction isEven(n) {\n",
     "test": "if (isEven(4) !== true || isEven(7) !== false) throw new Error('isEven');\n"},
    {"prompt": "// Return the largest element of the array xs.\nfunction largest(xs) {\n",
     "test": "if (largest([1, 5, 2]) !== 5) throw new Error('largest');\n"},
]

PHP_PROBLEMS: list[dict] = [
    {"prompt": "<?php\n// Return the sum of $a and $b.\nfunction add($a, $b) {\n",
     "test": "if (add(2, 3) !== 5) { throw new Exception('add'); }\n"},
    {"prompt": "<?php\n// Return true if $n is even.\nfunction is_even($n) {\n",
     "test": "if (is_even(4) !== true || is_even(7) !== false) { throw new Exception('is_even'); }\n"},
]

EXEC_PROBLEMS: dict[str, list[dict]] = {
    "python": PYTHON_PROBLEMS,
    "js": JS_PROBLEMS,
    "php": PHP_PROBLEMS,
}


def _run_subprocess(cmd: list[str], cwd: str, timeout: float) -> bool:
    """Run *cmd*; return True iff it exits 0 within *timeout* (else False)."""
    try:
        proc = subprocess.run(
            cmd, cwd=cwd, capture_output=True, timeout=timeout,
        )
        return proc.returncode == 0
    except (subprocess.TimeoutExpired, OSError):
        return False


def _exec_passes(lang: str, runtime: str, source: str, test: str, timeout: float) -> bool:
    """Write ``source + test`` to a temp file and execute it; True iff it passes.

    Executed in a throwaway temp dir with a timeout — model output is untrusted.
    """
    if lang == "python":
        code = source + "\n" + test
        fname, cmd_head = "solution.py", [sys.executable]
    elif lang == "js":
        code = source + "\n" + test
        fname, cmd_head = "solution.js", ["node"]
    elif lang == "php":
        # PHP: strip a leading <?php in the test (source already opened the tag).
        code = source + "\n" + test
        fname, cmd_head = "solution.php", ["php"]
    else:  # pragma: no cover - only exec langs reach here
        return False

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, fname)
        try:
            with open(path, "w", encoding="utf-8") as f:
                f.write(code)
        except OSError:  # pragma: no cover
            return False
        return _run_subprocess(cmd_head + [path], cwd=d, timeout=timeout)


# ---------------------------------------------------------------------------
# Completion sampling from held-out text
# ---------------------------------------------------------------------------

def _held_out_prefixes(eval_text: str, n_prompts: int, prefix_chars: int) -> list[str]:
    """Split held-out text into per-document prefixes to complete.

    Documents are joined by a blank line by the #53 loader, so split on that;
    fall back to fixed-size windows for a single-blob corpus. Each prefix is the
    document's first ``prefix_chars`` characters (a code-completion prompt).
    """
    docs = [d for d in re.split(r"\n\s*\n", eval_text) if len(d.strip()) >= 20]
    if not docs:
        docs = [eval_text[i:i + prefix_chars * 3]
                for i in range(0, len(eval_text), prefix_chars * 3)]
    prefixes: list[str] = []
    for doc in docs:
        if len(prefixes) >= n_prompts:
            break
        # Prompt with the first part of the doc; keep at least a little so the
        # model has context but must generate the rest.
        cut = min(prefix_chars, max(10, len(doc) // 2))
        prefixes.append(doc[:cut])
    return prefixes


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_coding(
    model: HamiltonianVortexNetwork,
    tokenizer,
    lang: str,
    eval_text: str,
    seq_len: int,
    device: torch.device,
    n_prompts: int = 8,
    prefix_chars: int = 160,
    max_new_tokens: int = 80,
    temperature: float = 0.2,
    top_k: int = 5,
    allow_exec: bool = True,
    exec_timeout: float = 5.0,
    exec_k: int = 1,
) -> dict:
    """Score one language for a loaded checkpoint. Returns a metrics dict.

    Always reports held-out ``ppl`` / ``bits_per_char`` (tokenization-invariant)
    and a structural ``completion_valid_rate`` (fraction of held-out completions
    that are structurally valid *and* non-degenerate). For a language with an
    available runtime and ``allow_exec=True`` it also runs a HumanEval-style
    ``exec_pass_at_k`` and uses it as the headline ``score``; otherwise the
    structural rate is the headline.
    """
    if lang not in CODING_LANGS:
        raise ValueError(f"unknown language {lang!r}; expected one of {CODING_LANGS}")

    model.eval()
    validator = STRUCTURAL_VALIDATORS[lang]

    # --- 1. Held-out language-model metric (always) ------------------------
    ppl, bpc = heldout_bpc(model, tokenizer, eval_text, seq_len, device)

    # --- 2. Structural completion-validity over held-out prefixes ----------
    prefixes = _held_out_prefixes(eval_text, n_prompts, prefix_chars)
    n_valid = 0
    samples: list[dict] = []
    for prefix in prefixes:
        completion = complete(
            model, tokenizer, prefix, max_new_tokens=max_new_tokens,
            temperature=temperature, top_k=top_k, device=device, stop_tokens=(),
        )
        combined = prefix + completion
        ok = validator(combined) and is_non_degenerate(completion, prefix)
        n_valid += int(ok)
        if len(samples) < 3:
            samples.append({"prefix": prefix[:60], "completion": completion[:80], "valid": ok})
    n = len(prefixes)
    completion_valid_rate = (n_valid / n) if n else 0.0

    # --- 3. Exec-based HumanEval-style pass@k (exec langs only) -------------
    runtime = _runtime_for(lang)
    exec_pass_at_k: float | None = None
    exec_problems = EXEC_PROBLEMS.get(lang, [])
    if allow_exec and runtime and exec_problems:
        solved = 0
        for prob in exec_problems:
            passed = False
            for _ in range(max(1, exec_k)):
                completion = complete(
                    model, tokenizer, prob["prompt"], max_new_tokens=max_new_tokens,
                    temperature=(0.0 if exec_k == 1 else max(0.2, temperature)),
                    top_k=(1 if exec_k == 1 else top_k), device=device, stop_tokens=(),
                )
                if _exec_passes(lang, runtime, prob["prompt"] + completion,
                                prob["test"], exec_timeout):
                    passed = True
                    break
            solved += int(passed)
        exec_pass_at_k = solved / len(exec_problems)

    # --- Headline score + metric kind --------------------------------------
    if exec_pass_at_k is not None:
        metric_kind = "exec"
        score = exec_pass_at_k
    else:
        metric_kind = "structural"
        score = completion_valid_rate

    return {
        "lang": lang,
        "metric_kind": metric_kind,
        "runtime": runtime or "none",
        "score": round(score, 4),
        "completion_valid_rate": round(completion_valid_rate, 4),
        "exec_pass_at_k": (round(exec_pass_at_k, 4) if exec_pass_at_k is not None else None),
        "exec_k": (exec_k if exec_pass_at_k is not None else None),
        "n_prompts": n,
        "ppl": round(ppl, 4),
        "bits_per_char": round(bpc, 4),
        "samples": samples,
    }


def score_stage(
    model: HamiltonianVortexNetwork,
    tokenizer,
    stage: str,
    eval_text: str,
    seq_len: int,
    device: torch.device,
    n_prompts: int = 8,
    allow_exec: bool = False,
) -> dict:
    """Orchestrator-facing wrapper (the #54 seam).

    Returns a compact metrics dict whose keys are a superset of the #54
    ``evaluate_stage`` contract (``ppl`` / ``bits_per_char`` / ``non_degenerate``
    / ``completion_sample``) plus the #55 coding score. ``allow_exec`` defaults to
    **False** so unattended curriculum runs stay fast and side-effect-free — the
    exec pass@k is an explicit ``python -m eval.coding`` pass.
    """
    metrics = evaluate_coding(
        model, tokenizer, stage, eval_text, seq_len, device,
        n_prompts=n_prompts, allow_exec=allow_exec,
    )
    sample = metrics["samples"][0]["completion"] if metrics["samples"] else ""
    return {
        # --- #54 evaluate_stage backward-compatible keys ---
        "ppl": metrics["ppl"],
        "bits_per_char": metrics["bits_per_char"],
        "non_degenerate": bool(metrics["completion_valid_rate"] > 0.0),
        "completion_sample": sample[:120],
        # --- #55 coding score ---
        "lang": metrics["lang"],
        "coding_score": metrics["score"],
        "metric_kind": metrics["metric_kind"],
        "runtime": metrics["runtime"],
        "completion_valid_rate": metrics["completion_valid_rate"],
        "exec_pass_at_k": metrics["exec_pass_at_k"],
        "n_prompts": metrics["n_prompts"],
    }


# ---------------------------------------------------------------------------
# Report formatting (consistent with the other eval/ harnesses)
# ---------------------------------------------------------------------------

def format_report(results: list[dict]) -> str:
    """Render a table of per-language results, matching eval/reasoning.py style."""
    lines = []
    lines.append("=" * 74)
    lines.append("  Coding eval — per-language held-out completion (Issue #55)")
    lines.append("=" * 74)
    header = f"  {'Lang':<8}{'Metric':<11}{'Runtime':<9}{'Score':>8}{'Valid%':>9}{'bits/char':>11}{'PPL':>10}"
    lines.append(header)
    lines.append("  " + "-" * 64)
    for r in results:
        exec_note = ""
        if r["exec_pass_at_k"] is not None:
            exec_note = ""  # score already reflects exec pass@k
        lines.append(
            f"  {r['lang']:<8}{r['metric_kind']:<11}{r['runtime']:<9}"
            f"{r['score']:>8.2%}{r['completion_valid_rate']:>9.2%}"
            f"{r['bits_per_char']:>11.3f}{r['ppl']:>10.2f}{exec_note}"
        )
    lines.append("  " + "-" * 64)
    lines.append("  Score = exec pass@k where a runtime exists, else structural "
                 "completion-validity rate.")
    lines.append("  bits/char is tokenization-invariant (lower is better); "
                 "the curriculum should lower it per stage.")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Checkpoint loading (mirrors auto_train.load_checkpoint; kept local to avoid an
# import cycle, since auto_train imports this module)
# ---------------------------------------------------------------------------

def _load_model_and_tokenizer(checkpoint: str, device: torch.device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    model = HamiltonianVortexNetwork(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=ckpt["num_layers"],
        n_heads=ckpt.get("n_heads", 4),
        bond_dimension=ckpt.get("bond_dimension", ckpt.get("bond_dim", 32)),
        chunk_size=ckpt.get("chunk_size", 64),
        n_threads=ckpt.get("n_threads", 1),
        use_deltanet=ckpt.get("use_deltanet", False),
        deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        weightless=ckpt.get("weightless", False),
        weightless_seed=ckpt.get("weightless_seed", 1234),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()

    tok_path = checkpoint.replace(".pt", "_tokenizer.json")
    try:
        from hvn.tokenizer import BPETokenizer

        tokenizer = BPETokenizer.load(tok_path)
    except Exception:
        from train import CharTokenizer

        tokenizer = CharTokenizer.load(tok_path)
    return model, tokenizer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="Per-language coding eval harness for HVN (Issue #55)")
    p.add_argument("--checkpoint", type=str, default="hvn_chat_checkpoint.pt",
                   help="Model checkpoint (tokenizer auto-detected from <stem>_tokenizer.json)")
    p.add_argument("--lang", type=str, default="all",
                   help=f"Language to score, or 'all' ({'/'.join(CODING_LANGS)})")
    p.add_argument("--local-dir", type=str, default=None,
                   help="Directory of pre-fetched held-out code for the stages (#53)")
    p.add_argument("--download", action="store_true",
                   help="Allow best-effort curated permissive downloads for held-out text (#53)")
    p.add_argument("--n-prompts", type=int, default=8,
                   help="Held-out completion prompts per language (structural rate)")
    p.add_argument("--seq-len", type=int, default=128, help="Chunk length for held-out bpc/PPL")
    p.add_argument("--no-exec", action="store_true",
                   help="Disable exec-based pass@k (structural/PPL only)")
    p.add_argument("--exec-k", type=int, default=1, help="pass@k samples for exec problems")
    p.add_argument("--json", type=str, default=None, help="Write the full results to this JSON path")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"✗ checkpoint not found: {args.checkpoint}")
        return 2

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    torch.manual_seed(args.seed)

    model, tokenizer = _load_model_and_tokenizer(args.checkpoint, device)
    print(f"Loaded {args.checkpoint} | vocab={model.vocab_size}")

    # Held-out text per language comes from the #53 loader (offline-safe).
    from eval.curriculum import load_curriculum_stage

    langs = CODING_LANGS if args.lang == "all" else [args.lang]
    for lang in langs:
        if lang not in CODING_LANGS:
            print(f"✗ unknown language {lang!r}; expected one of {CODING_LANGS}")
            return 2

    results = []
    for lang in langs:
        _train, eval_text = load_curriculum_stage(
            lang, local_dir=args.local_dir, allow_download=args.download, seed=args.seed,
        )
        res = evaluate_coding(
            model, tokenizer, lang, eval_text, args.seq_len, device,
            n_prompts=args.n_prompts, allow_exec=not args.no_exec, exec_k=args.exec_k,
        )
        results.append(res)
        print(f"  [{lang}] score={res['score']:.2%} ({res['metric_kind']}, "
              f"runtime={res['runtime']}) | valid={res['completion_valid_rate']:.2%} | "
              f"{res['bits_per_char']:.3f} bits/char | PPL {res['ppl']:.2f}")

    print("\n" + format_report(results))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)
        print(f"\n  ✓ results written to {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
