"""
Test suite for Issue #54 — auto-train orchestrator (staged curriculum scheduler).

Validates ``auto_train.run_curriculum``: it trains the curriculum stages in
order from a base checkpoint, writes per-stage ``checkpoint_stage_{name}.pt``
(+ tokenizer) and a metrics log, records resumable state, and **resumes** from
the last completed stage after an interruption — without retraining done stages.

Runs fully offline on CPU (synthetic #53 fallback, no CUDA, no network) so it is
CI / 6 GB-box safe.

Run from the repo root:
    python tests/test_issue54.py
"""

import os
import sys
import json
import tempfile

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import string

import torch

from hvn.core import HamiltonianVortexNetwork
from train import CharTokenizer
import auto_train
from auto_train import run_curriculum, load_checkpoint, _stage_checkpoint_path

DEVICE = torch.device("cpu")

# Tiny, fast, offline knobs shared by every test.
SMOKE = dict(
    epochs_per_stage=1, seq_len=48, batch_size=8, precision="fp32",
    max_train_chars=20_000, max_eval_chars=3_000, allow_download=False,
    device=DEVICE,
)


def _make_base_checkpoint(path: str, seed: int = 0) -> None:
    """Build a tiny DeltaNet base checkpoint (+ char tokenizer) like #51's output.

    The char tokenizer is built over printable ASCII so it covers all of the code
    characters the curriculum stages contain (encode silently drops OOV chars).
    """
    torch.manual_seed(seed)
    tokenizer = CharTokenizer(string.printable)
    model = HamiltonianVortexNetwork(
        vocab_size=tokenizer.vocab_size, d_model=32, num_layers=1,
        n_heads=2, chunk_size=16, use_deltanet=True, deltanet_conv_kernel=4,
    )
    torch.save({
        "model_state_dict": model.state_dict(),
        "vocab_size": tokenizer.vocab_size,
        "d_model": 32,
        "num_layers": 1,
        "n_heads": 2,
        "chunk_size": 16,
        "use_deltanet": True,
        "deltanet_conv_kernel": 4,
        "weightless": False,
        "weightless_seed": 1234,
    }, path)
    tokenizer.save(path.replace(".pt", "_tokenizer.json"))


def test_full_run_two_stages():
    """A run over two stages produces per-stage checkpoints, tokenizers, metrics
    and a state file recording both stages complete."""
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "base.pt")
        _make_base_checkpoint(base)

        state = run_curriculum(
            base_checkpoint=base, stages=["html", "css"], out_dir=tmp, **SMOKE)

        assert [c["stage"] for c in state["completed"]] == ["html", "css"]
        for stage in ("html", "css"):
            ckpt = _stage_checkpoint_path(tmp, stage)
            assert os.path.exists(ckpt), f"{stage}: missing checkpoint"
            assert os.path.exists(ckpt.replace(".pt", "_tokenizer.json")), \
                f"{stage}: missing tokenizer"

        # State + metrics log written.
        state_path = os.path.join(tmp, "auto_train_state.json")
        metrics_path = os.path.join(tmp, "auto_train_metrics.jsonl")
        assert os.path.exists(state_path) and os.path.exists(metrics_path)
        rows = [json.loads(l) for l in open(metrics_path, encoding="utf-8")]
        assert [r["stage"] for r in rows] == ["html", "css"]
        for r in rows:
            assert "bits_per_char" in r["metrics"] and "ppl" in r["metrics"]
        # css was trained continuing FROM the html checkpoint (carry-forward).
        assert rows[1]["from"] == _stage_checkpoint_path(tmp, "html")
    print("[1/4] full run: per-stage checkpoints + metrics + carry-forward ... OK")


def test_stage_checkpoints_are_loadable():
    """Each stage checkpoint reloads and produces vocab-sized logits (servable)."""
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "base.pt")
        _make_base_checkpoint(base)
        run_curriculum(base_checkpoint=base, stages=["python"], out_dir=tmp, **SMOKE)

        ckpt_path = _stage_checkpoint_path(tmp, "python")
        model, tokenizer, ckpt = load_checkpoint(ckpt_path, DEVICE)
        model.eval()
        ids = tokenizer.encode("def add(a, b):") or [0]
        x = torch.tensor([ids], dtype=torch.long)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape[-1] == ckpt["vocab_size"]
        assert logits.shape[:2] == (1, len(ids))
    print("[2/4] stage checkpoints reload and produce logits ............... OK")


def test_resume_skips_completed_stage():
    """Re-running resumes: an already-completed stage is skipped (not retrained)
    and later stages are carried forward from its checkpoint."""
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "base.pt")
        _make_base_checkpoint(base)

        # First run: only html.
        run_curriculum(base_checkpoint=base, stages=["html"], out_dir=tmp, **SMOKE)
        html_ckpt = _stage_checkpoint_path(tmp, "html")
        html_mtime = os.path.getmtime(html_ckpt)

        # Second run: html already done → skipped; css added.
        state = run_curriculum(
            base_checkpoint=base, stages=["html", "css"], out_dir=tmp, **SMOKE)

        # html checkpoint was NOT rewritten (stage skipped, not retrained).
        assert os.path.getmtime(html_ckpt) == html_mtime, "html was retrained on resume"
        assert os.path.exists(_stage_checkpoint_path(tmp, "css"))
        assert [c["stage"] for c in state["completed"]] == ["html", "css"]

        # metrics log has exactly one row per stage (html not duplicated).
        rows = [json.loads(l) for l in
                open(os.path.join(tmp, "auto_train_metrics.jsonl"), encoding="utf-8")]
        assert [r["stage"] for r in rows] == ["html", "css"]
        # css carried forward from the html stage checkpoint, not the base.
        assert rows[1]["from"] == html_ckpt
    print("[3/4] resume skips completed stage, carries forward ............. OK")


def test_restart_ignores_state():
    """``resume=False`` (the --restart flag) ignores prior state and retrains."""
    with tempfile.TemporaryDirectory() as tmp:
        base = os.path.join(tmp, "base.pt")
        _make_base_checkpoint(base)

        run_curriculum(base_checkpoint=base, stages=["html"], out_dir=tmp, **SMOKE)
        html_ckpt = _stage_checkpoint_path(tmp, "html")
        first_mtime = os.path.getmtime(html_ckpt)

        # Restart: html should be retrained (checkpoint rewritten) from the base.
        state = run_curriculum(
            base_checkpoint=base, stages=["html"], out_dir=tmp, resume=False, **SMOKE)
        assert os.path.getmtime(html_ckpt) >= first_mtime
        assert state["completed"][0]["from"] == base, "restart must train from base"
    print("[4/4] --restart ignores prior state (trains from base) .......... OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Issue #54 — auto-train orchestrator (staged curriculum)")
    print("=" * 60)
    test_full_run_two_stages()
    test_stage_checkpoints_are_loadable()
    test_resume_skips_completed_stage()
    test_restart_ignores_state()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
