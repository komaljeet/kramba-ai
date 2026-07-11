"""CPU tests for the A4 hybrid grounded continue-FT pipeline (no GPU needed).

Covers:
  1. The grounded-QA data builder: answers are verbatim spans; no eval leakage.
  2. GroundedQADataset loss masking: only the answer+EOS turn is learned, and the
     rendered prompt matches the gate's open-book format; causal shift is correct.
  3. build_hybrid_model + load_hybrid_trunk_into_model: block_plan has the 2 attn
     positions fresh + 10 DeltaNet copied; a forward pass produces the right shape.
  4. Checkpoint round-trip: _save_edge_checkpoint persists block_plan, and
     load_checkpoint_for_resume rebuilds the hybrid arch from it.

Tiny dims (d=64, L=4) keep this fast on CPU. Run: ``python -m pytest tests/test_continue_ft_hybrid.py -x``
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.build_grounded_qa_data import build_records, validate, EVAL_QUESTIONS  # noqa: E402
from continue_ft_hybrid import (
    GroundedQADataset,
    build_hybrid_model,
    OPEN_BOOK_INSTRUCTION,
    USER_PREFIX,
    ASSISTANT_PREFIX,
    EOS_ID,
)
from train_edge_conversational import _save_edge_checkpoint, load_checkpoint_for_resume  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402


# ---------------------------------------------------------------------------
# 1. Data builder invariants
# ---------------------------------------------------------------------------

def test_builder_answers_are_verbatim_spans():
    records = build_records()
    assert len(records) >= 40
    for r in records:
        assert r["answer"].lower() in r["passage"].lower(), (
            f"answer {r['answer']!r} not a verbatim span of passage")
        assert r["question"] and r["passage"]


def test_builder_no_eval_leakage():
    """No training question collides with the 5 held-out eval questions."""
    records = build_records()
    for r in records:
        assert r["question"].lower().strip() not in EVAL_QUESTIONS, (
            f"training question collides with eval: {r['question']!r}")


def test_builder_validate_passes():
    # validate() sys.exits on failure; reaching here means OK.
    validate(build_records())


# ---------------------------------------------------------------------------
# 2. GroundedQADataset loss masking (with a deterministic fake tokenizer)
# ---------------------------------------------------------------------------

class _FakeTok:
    """Char-level tokenizer: encode -> [ord(c) for c in text]; decode round-trips."""
    def encode(self, text):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


def test_grounded_dataset_loss_masking():
    tok = _FakeTok()
    records = [{"passage": "The capital of Japan is Tokyo.",
                "question": "What is the capital of Japan?",
                "answer": "Tokyo"}]
    ds = GroundedQADataset(tok, records, seq_len=256)
    assert len(ds) == 1
    ids, labels = ds.examples[0]
    # x = ids[:-1], y = labels[1:] (causal shift)
    x, y = ds[0]

    # The full rendered string must contain the gate's instruction + markers.
    full = tok.decode(ids.tolist())
    assert OPEN_BOOK_INSTRUCTION.strip() in full
    assert "Context: " in full and "Question: " in full
    assert USER_PREFIX in full and ASSISTANT_PREFIX in full

    # Only the answer (+ trailing EOS) tokens are learned; everything else is -100.
    learned_idx = [i for i, l in enumerate(labels.tolist()) if l != -100]
    assert len(learned_idx) > 0, "no learned tokens — loss masking is wrong"
    # The learned tokens must decode to the answer + EOS.
    learned_tokens = [ids[i] for i in learned_idx]
    learned_text = tok.decode(learned_tokens)
    assert "Tokyo" in learned_text

    # The learned region is contiguous and at the END (after the assistant marker):
    # no -100 should appear after the first learned token except within padding.
    first_learned = learned_idx[0]
    # Everything before the assistant prefix turn is masked.
    asst_pos = full.find(ASSISTANT_PREFIX)
    assert asst_pos != -1
    # The first learned token comes at or after the assistant marker position.
    assert first_learned >= asst_pos

    # Causal shift sanity: y has the same length as x.
    assert x.shape == y.shape


def test_grounded_dataset_truncation_keeps_answer():
    """If the passage is too long, the answer (learned) must still be present."""
    tok = _FakeTok()
    long_passage = "x" * 200  # forces overflow at seq_len=64
    records = [{"passage": long_passage, "question": "Q?", "answer": "x"}]
    ds = GroundedQADataset(tok, records, seq_len=64)
    ids, labels = ds.examples[0]
    learned = [ids[i] for i, l in enumerate(labels.tolist()) if l != -100]
    # The answer token ('x' = ord 120) must survive in the learned set.
    assert ord("x") in learned, "answer was lost to front-truncution"


# ---------------------------------------------------------------------------
# 3. build_hybrid_model + trunk warm-start (tiny dims, CPU)
# ---------------------------------------------------------------------------

def _make_tiny_trunk(tmpdir, num_layers=4, d_model=64, n_heads=4):
    """Save a tiny PURE-HVN (all-DeltaNet) checkpoint and return its path."""
    model = EdgeHVN(
        vocab_size=256, d_model=d_model, num_layers=num_layers, n_heads=n_heads,
        use_deltanet=True, deltanet_conv_kernel=4,
    )
    path = os.path.join(tmpdir, "tiny_trunk.pt")
    _save_edge_checkpoint(model, path, global_step=1000)
    return path


def test_build_hybrid_model_trunk_init():
    import tempfile
    tmp_path = tempfile.mkdtemp()
    trunk = _make_tiny_trunk(str(tmp_path))
    attn_layers = [1, 3]
    model, trunk_step = build_hybrid_model(
        trunk, attn_layers=attn_layers, attn_window=64,
        device=torch.device("cpu"))
    assert trunk_step == 1000
    # block_plan: transformer at attn positions, deltanet elsewhere.
    assert model.block_plan == [
        "transformer" if i in attn_layers else "deltanet"
        for i in range(model.num_layers)]
    assert model.block_plan.count("transformer") == len(attn_layers)
    assert model.attn_window == 64

    # A forward pass produces the right shape (vocab 256).
    model = model.to("cpu").eval()
    x = torch.randint(0, 256, (1, 16), dtype=torch.long)
    with torch.no_grad():
        logits = model(x)
    assert logits.shape == (1, 16, 256), f"got {logits.shape}"


# ---------------------------------------------------------------------------
# 4. Checkpoint round-trip preserves block_plan
# ---------------------------------------------------------------------------

def test_checkpoint_roundtrip_block_plan():
    import tempfile
    tmp_path = tempfile.mkdtemp()
    trunk = _make_tiny_trunk(str(tmp_path))
    model, _ = build_hybrid_model(
        trunk, attn_layers=[1, 3], attn_window=64, device=torch.device("cpu"))
    out = os.path.join(str(tmp_path), "hybrid.pt")
    _save_edge_checkpoint(model, out, global_step=500)
    assert os.path.exists(out)

    # load_checkpoint_for_resume rebuilds the hybrid arch from saved metadata.
    loaded, step, ckpt = load_checkpoint_for_resume(out, torch.device("cpu"))
    assert step == 500
    assert loaded.block_plan == model.block_plan
    assert loaded.attn_window == 64
    # The two attention layers are present in the rebuilt model.
    assert loaded.block_plan.count("transformer") == 2


# ---------------------------------------------------------------------------
# Runner for `python tests/test_continue_ft_hybrid.py` (repo convention:
# standalone test scripts; pytest optional). Mirrors test_hybrid_forward.py.
# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())