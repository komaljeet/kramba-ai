"""
Test suite for Issue #55 — per-language coding eval harness (eval/coding.py).

Validates:
  * the structural/compile validators reject gibberish and accept valid code;
  * ``evaluate_coding`` produces a stable, JSON-serialisable per-language score
    for a tiny checkpoint (held-out bpc/PPL + structural completion rate; exec
    pass@k where a runtime exists — Python always);
  * the exec path actually runs code (a hand-written correct completion passes,
    a wrong one fails), proving the pass@k metric measures execution;
  * the #54 orchestrator seam: ``auto_train.evaluate_stage`` delegates to it and
    still returns the backward-compatible keys the scheduler logs.

Runs fully offline on CPU (synthetic #53 fallback, no CUDA, no network).

Run from the repo root:
    python tests/test_issue55.py
"""

import os
import sys
import string

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from hvn.core import HamiltonianVortexNetwork
from train import CharTokenizer
from eval.curriculum import load_curriculum_stage
from eval import coding
from eval.coding import (
    evaluate_coding,
    score_stage,
    STRUCTURAL_VALIDATORS,
    _exec_passes,
    _runtime_for,
)

DEVICE = torch.device("cpu")


def _tiny_model_and_tokenizer(seed: int = 0):
    torch.manual_seed(seed)
    tokenizer = CharTokenizer(string.printable)
    model = HamiltonianVortexNetwork(
        vocab_size=tokenizer.vocab_size, d_model=32, num_layers=1,
        n_heads=2, chunk_size=16, use_deltanet=True, deltanet_conv_kernel=4,
    )
    model.eval()
    return model, tokenizer


def test_structural_validators():
    """Validators accept well-formed code and reject gibberish / imbalance."""
    v = STRUCTURAL_VALIDATORS
    assert v["python"]("def f(x):\n    return x + 1\n")
    assert not v["python"]("def f(x)\n    return")            # syntax error
    assert v["html"]("<div><p>hi</p></div>")
    assert not v["html"]("<div><p>hi</div>")                  # mismatched tags
    assert not v["html"]("just some text, no tags")
    assert v["css"](".a { color: red; }")
    assert not v["css"](".a { color: red; ")                  # unbalanced brace
    assert v["js"]("function add(a, b) { return a + b; }")
    assert not v["js"]("function add(a, b) { return a + b; ")  # unbalanced
    assert v["php"]("<?php function f() { return 1; }")
    assert not v["php"]("function f() { return 1; }")          # no php tag
    assert v["devops"]("FROM python:3.12-slim\nRUN pip install x\n")
    assert v["devops"]("name: ci\non:\n  push:\n    branches: [main]\n")
    assert not v["devops"]("lorem ipsum dolor sit amet")
    print("[1/5] structural validators accept valid, reject gibberish ...... OK")


def test_exec_passes_runs_code():
    """The exec harness distinguishes a correct completion from a wrong one."""
    assert _runtime_for("python") == "python"  # always available
    good = "def add(a, b):\n    return a + b\n"
    bad = "def add(a, b):\n    return a - b\n"
    test = "assert add(2, 3) == 5\n"
    assert _exec_passes("python", "python", good, test, timeout=10.0) is True
    assert _exec_passes("python", "python", bad, test, timeout=10.0) is False
    print("[2/5] exec harness passes correct code, fails wrong code ......... OK")


def test_evaluate_coding_stable_score():
    """evaluate_coding returns a JSON-serialisable, deterministic score dict."""
    model, tokenizer = _tiny_model_and_tokenizer()
    _train, eval_text = load_curriculum_stage("python", allow_download=False)

    kw = dict(n_prompts=4, max_new_tokens=32, allow_exec=False)
    torch.manual_seed(123)
    r1 = evaluate_coding(model, tokenizer, "python", eval_text, 64, DEVICE, **kw)
    torch.manual_seed(123)
    r2 = evaluate_coding(model, tokenizer, "python", eval_text, 64, DEVICE, **kw)

    # Required keys present and typed.
    for key in ("lang", "metric_kind", "runtime", "score", "completion_valid_rate",
                "bits_per_char", "ppl", "n_prompts"):
        assert key in r1, f"missing {key}"
    assert r1["lang"] == "python"
    assert 0.0 <= r1["score"] <= 1.0
    assert 0.0 <= r1["completion_valid_rate"] <= 1.0
    assert r1["bits_per_char"] > 0.0
    # Deterministic under a fixed seed (stable per-language score).
    assert r1["score"] == r2["score"]
    assert r1["bits_per_char"] == r2["bits_per_char"]
    # JSON round-trips.
    import json
    json.loads(json.dumps(r1))
    print("[3/5] evaluate_coding: stable, JSON-serialisable score dict ...... OK")


def test_exec_lang_uses_pass_at_k():
    """With allow_exec, Python reports an exec pass@k and uses it as the headline."""
    model, tokenizer = _tiny_model_and_tokenizer()
    _train, eval_text = load_curriculum_stage("python", allow_download=False)
    r = evaluate_coding(
        model, tokenizer, "python", eval_text, 64, DEVICE,
        n_prompts=2, max_new_tokens=24, allow_exec=True, exec_timeout=10.0,
    )
    assert r["metric_kind"] == "exec"
    assert r["runtime"] == "python"
    assert r["exec_pass_at_k"] is not None
    assert 0.0 <= r["exec_pass_at_k"] <= 1.0
    assert r["score"] == r["exec_pass_at_k"]

    # A non-exec language (no runtime here) falls back to structural.
    _t, css_eval = load_curriculum_stage("css", allow_download=False)
    rc = evaluate_coding(model, tokenizer, "css", css_eval, 64, DEVICE,
                         n_prompts=2, max_new_tokens=24, allow_exec=True)
    assert rc["metric_kind"] == "structural"
    assert rc["exec_pass_at_k"] is None
    assert rc["score"] == rc["completion_valid_rate"]
    print("[4/5] exec lang -> pass@k headline; non-exec -> structural ....... OK")


def test_orchestrator_seam_backward_compatible():
    """auto_train.evaluate_stage delegates here and keeps the #54 contract keys."""
    import auto_train
    model, tokenizer = _tiny_model_and_tokenizer()
    _train, eval_text = load_curriculum_stage("html", allow_download=False)

    metrics = auto_train.evaluate_stage(model, tokenizer, "html", eval_text, 64, DEVICE)
    # #54 backward-compatible keys the scheduler logs.
    for key in ("ppl", "bits_per_char", "non_degenerate", "completion_sample"):
        assert key in metrics, f"evaluate_stage dropped {key}"
    # #55 enrichment.
    assert metrics["lang"] == "html"
    assert "coding_score" in metrics and 0.0 <= metrics["coding_score"] <= 1.0
    assert metrics["metric_kind"] == "structural"  # no HTML runtime

    # score_stage direct call matches the same contract.
    s = score_stage(model, tokenizer, "css", eval_text, 64, DEVICE)
    assert "bits_per_char" in s and "coding_score" in s
    print("[5/5] evaluate_stage seam keeps #54 keys + adds coding_score ..... OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Issue #55 — per-language coding eval harness")
    print("=" * 60)
    test_structural_validators()
    test_exec_passes_runs_code()
    test_evaluate_coding_stable_score()
    test_exec_lang_uses_pass_at_k()
    test_orchestrator_seam_backward_compatible()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
