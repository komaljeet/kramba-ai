"""
Test suite for Issue #51 — Chat/base checkpoint pipeline.

Validates the wiring in ``build_chat_checkpoint.py`` without paying for a full
WikiText-103 pretrain: it exercises the fine-tune corpus builder, the smoke-gate
helpers, and a tiny end-to-end train → save → reload → generate loop on a
synthetic corpus. Proves the training→serving loop is intact (Milestone 1).

Run from the repo root:
    python tests/test_issue51.py
"""

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader

from hvn.core import HamiltonianVortexNetwork
from hvn.tokenizer import BPETokenizer
from hvn.chat import CHAT_TOKENS
from train import TextDataset, train_model
from build_chat_checkpoint import (
    build_finetune_text,
    is_non_degenerate,
    heldout_bpc,
    complete,
)


def test_finetune_corpus():
    """Fine-tune corpus contains chat + instruction structure, with a held-out split."""
    train_text, heldout = build_finetune_text(instruction_examples=40, chat_examples=40)
    assert len(train_text) > 0 and len(heldout) > 0, "empty fine-tune corpus"
    # Chat protocol tokens present (so the model can learn the serving format).
    assert CHAT_TOKENS["user_start"] in train_text, "missing [USER] chat tokens"
    assert CHAT_TOKENS["assistant_start"] in train_text, "missing [ASSISTANT] chat tokens"
    # Alpaca instruction scaffolding present.
    assert "### Instruction:" in train_text, "missing instruction format"
    # Held-out is disjoint-ish (different examples) and non-trivial.
    assert train_text != heldout
    print("[1/4] fine-tune corpus (chat + instruction + held-out split) ..... OK")


def test_non_degenerate_gate():
    """The smoke gate rejects empty/echo/single-char output and accepts real text."""
    assert not is_non_degenerate("", "p")
    assert not is_non_degenerate("   ", "p")
    assert not is_non_degenerate("aaaaaa", "p")          # single repeated char
    assert not is_non_degenerate("hello", "hello")        # pure echo
    assert is_non_degenerate("def add(a, b): return a + b", "Create a function")
    print("[2/4] non-degenerate smoke gate helper ......................... OK")


def _tiny_train(tmpdir, device):
    """Train a tiny BPE deltanet model on a synthetic corpus, mirroring the
    pipeline's pretrain→fine-tune→save wiring."""
    base = ("The Hamiltonian Vortex Network is efficient. " * 60)
    ft_train, ft_heldout = build_finetune_text(instruction_examples=60, chat_examples=60)
    tok = BPETokenizer.train(base + "\n" + ft_train, vocab_size=1000)

    model = HamiltonianVortexNetwork(
        vocab_size=tok.vocab_size, d_model=64, num_layers=2,
        n_heads=2, bond_dimension=16, chunk_size=32, dropout=0.0,
        use_deltanet=True,
    )
    out = os.path.join(tmpdir, "ckpt.pt")

    # Pretrain
    ids = tok.encode(base)
    loader = DataLoader(TextDataset(ids, seq_len=48), batch_size=8, shuffle=True)
    train_model(model, loader, epochs=2, lr=3e-4, device=device, save_path=out,
                precision="fp32", defer_loss_sync=True)
    # Fine-tune (carry weights forward)
    ft_ids = tok.encode(ft_train)
    ft_loader = DataLoader(TextDataset(ft_ids, seq_len=48), batch_size=8, shuffle=True)
    train_model(model, ft_loader, epochs=3, lr=1e-4, device=device, save_path=out,
                precision="fp32", defer_loss_sync=True)
    tok.save(out.replace(".pt", "_tokenizer.json"))
    return out, tok, model, ft_heldout


def test_end_to_end_save_reload(device):
    """Tiny pretrain→fine-tune→save→reload→generate loop works and round-trips."""
    with tempfile.TemporaryDirectory() as tmp:
        out, tok, model, ft_heldout = _tiny_train(tmp, device)
        assert os.path.exists(out), "checkpoint not saved"
        assert os.path.exists(out.replace(".pt", "_tokenizer.json")), "tokenizer not saved"

        # Reload from disk exactly as generate.py / the worker would.
        ck = torch.load(out, map_location=device, weights_only=False)
        assert ck["use_deltanet"] is True, "deltanet flag not persisted"
        reloaded = HamiltonianVortexNetwork(
            vocab_size=ck["vocab_size"], d_model=ck["d_model"],
            num_layers=ck["num_layers"], n_heads=ck["n_heads"],
            chunk_size=ck.get("chunk_size", 64),
            use_deltanet=ck["use_deltanet"],
            deltanet_conv_kernel=ck.get("deltanet_conv_kernel", 4),
        )
        reloaded.load_state_dict(ck["model_state_dict"])
        reloaded.to(device).eval()
        tok2 = BPETokenizer.load(out.replace(".pt", "_tokenizer.json"))

        # Generation produces something (non-crashing, decodable).
        text = complete(reloaded, tok2, "The Hamiltonian ", max_new_tokens=20,
                        temperature=0.7, top_k=5, device=device)
        assert isinstance(text, str)
        print("[3/4] end-to-end pretrain→fine-tune→save→reload→generate ...... OK")


def test_heldout_bpc_finite(device):
    """Held-out bits-per-char is finite (the acceptance comparison metric)."""
    with tempfile.TemporaryDirectory() as tmp:
        _, tok, model, ft_heldout = _tiny_train(tmp, device)
        ppl, bpc = heldout_bpc(model, tok, ft_heldout, seq_len=48, device=device)
        assert ppl == ppl and bpc == bpc, "NaN eval"          # not NaN
        assert 0 < bpc < 100, f"implausible bits/char {bpc}"
        print(f"[4/4] held-out bits/char finite ({bpc:.3f} bpc) ............... OK")


if __name__ == "__main__":
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print(f"Issue #51 — chat/base checkpoint pipeline  (device={dev})")
    print("=" * 60)
    test_finetune_corpus()
    test_non_degenerate_gate()
    test_end_to_end_save_reload(dev)
    test_heldout_bpc_finite(dev)
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
