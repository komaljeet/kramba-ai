"""
Chat/Base Checkpoint Pipeline (Issue #51)
=========================================

One command that produces the first *real* HVN checkpoint the Django app
(``web/``) can serve, by wiring together pieces that already exist:

  1. Train a real **BPE tokenizer** (``hvn/tokenizer.py``) over the full
     corpus (pretrain sample + instruction/chat fine-tune text) so subwords
     *and* the ``[SYSTEM]/[USER]/[ASSISTANT]`` chat protocol tokens are covered.
  2. **Pretrain** on WikiText-103 (``eval/datasets.py`` streaming loader) with
     the recommended stack — Gated-DeltaNet blocks (#34), bf16 (#44), and the
     Category-A exact speed levers from ``--full`` (#50): fused AdamW, per-epoch
     loss sync, pinned/async H2D transfers.
  3. **Instruction / chat fine-tune** on synthetic Alpaca-style instruction
     data (``hvn/lora.py``) + multi-turn chat data (``hvn/chat.py``), carrying
     the pretrained weights forward.
  4. Emit ``<out>`` + ``<out_stem>_tokenizer.json`` at a documented path.
  5. **Smoke-test gate:** the model must give a *non-degenerate*, on-topic
     completion to a trivial coding prompt and a short greeting — a cheap
     loop-validation, not a product goal.

This is Milestone 1 (Issue A of 5). It proves the training→serving loop before
the coding-curriculum work (#53/#54/#55). It reuses ``train.train_model`` /
``TextDataset`` rather than duplicating a training loop.

Usage
-----
    # Full (recommended) run — d=256, 3 layers, 30M-char WikiText-103 budget:
    python build_chat_checkpoint.py --out hvn_chat_checkpoint.pt

    # Fast loop-validation (tiny budget, a few epochs) — CI / smoke:
    python build_chat_checkpoint.py --smoke --out hvn_chat_smoke.pt

The produced ``hvn_chat_checkpoint.pt`` + ``hvn_chat_checkpoint_tokenizer.json``
are what Issue #52 points the Django worker at.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time

# Force UTF-8 stdout/stderr so the ✓/→ characters printed here (and by the
# reused train.py loop) don't crash on a legacy cp1252 Windows console.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):  # pragma: no cover - non-reconfigurable stream
        pass

import torch
from torch.utils.data import DataLoader

from hvn.core import HamiltonianVortexNetwork
from hvn.tokenizer import BPETokenizer
from hvn.chat import ConversationFormatter, generate_chat_training_data
from hvn.lora import generate_instruction_training_data, format_instruction_example
from eval.datasets import load_wikitext103
from train import TextDataset, train_model


# ---------------------------------------------------------------------------
# Fine-tune corpus construction
# ---------------------------------------------------------------------------

def build_finetune_text(
    instruction_examples: int,
    chat_examples: int,
    seed: int = 42,
) -> tuple[str, str]:
    """Build the instruction + chat fine-tune corpus.

    Returns ``(train_text, heldout_text)`` where the last 10% of the examples
    are held out for evaluation.  Each example is an Alpaca-formatted
    instruction string (``hvn/lora.py``) or a multi-turn chat string with
    ``[SYSTEM]/[USER]/[ASSISTANT]`` tags (``hvn/chat.py``).
    """
    import random
    rng = random.Random(seed)

    segments: list[str] = []

    instr = generate_instruction_training_data(num_examples=instruction_examples, seed=seed)
    for ex in instr:
        segments.append(format_instruction_example(ex))

    # generate_chat_training_data returns already-formatted conversation strings.
    segments.extend(generate_chat_training_data(num_examples=chat_examples))

    rng.shuffle(segments)
    split = max(1, int(len(segments) * 0.9))
    train_text = "\n\n".join(segments[:split])
    heldout_text = "\n\n".join(segments[split:])
    return train_text, heldout_text


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def heldout_bpc(
    model: HamiltonianVortexNetwork,
    tokenizer,
    text: str,
    seq_len: int,
    device: torch.device,
    max_chunks: int = 200,
) -> tuple[float, float]:
    """Return (perplexity, bits-per-character) on held-out ``text``.

    Bits-per-character is tokenization-invariant, so it lets a BPE model be
    compared fairly against a char-level baseline (the #51 acceptance bar).
    """
    model.eval()
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return float("inf"), float("inf")

    total_nll = 0.0        # natural-log loss summed over predicted tokens
    n_tokens = 0
    last_id_idx = 0        # highest token index consumed (for char coverage)
    criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    n_chunks = 0
    for start in range(0, len(ids) - 1, seq_len):
        chunk = ids[start:start + seq_len + 1]
        if len(chunk) < 2:
            break
        x = torch.tensor([chunk[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([chunk[1:]], dtype=torch.long, device=device)
        logits = model(x)
        loss = criterion(logits.reshape(-1, model.vocab_size), y.reshape(-1))
        total_nll += loss.item()
        n_tokens += y.numel()
        last_id_idx = start + len(chunk)
        n_chunks += 1
        if n_chunks >= max_chunks:
            break

    if n_tokens == 0:
        return float("inf"), float("inf")
    ppl = math.exp(min(total_nll / n_tokens, 20))
    # bits-per-char over the portion actually scored: NLL (nats) → bits, divided
    # by the number of characters those tokens decode to. Tokenization-invariant,
    # so a BPE model compares fairly against a char-level baseline.
    covered_chars = max(1, len(tokenizer.decode(ids[:last_id_idx])))
    bpc = (total_nll / math.log(2)) / covered_chars
    return ppl, bpc


@torch.no_grad()
def complete(
    model: HamiltonianVortexNetwork,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 60,
    temperature: float = 0.7,
    top_k: int = 10,
    device: torch.device | None = None,
    stop_tokens: tuple[str, ...] = ("[/ASSISTANT]", "[/USER]"),
) -> str:
    """Greedy/top-k completion that returns ONLY the newly generated text."""
    if device is None:
        device = next(model.parameters()).device
    model.eval()

    ids = tokenizer.encode(prompt) or [0]
    generated: list[int] = []

    for _ in range(max_new_tokens):
        context = (ids + generated)[-128:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)
        next_logits = logits[0, -1, :] / max(1e-6, temperature)
        if top_k > 0:
            k = min(top_k, next_logits.size(-1))
            values, _ = torch.topk(next_logits, k)
            next_logits[next_logits < values[-1]] = float("-inf")
        probs = torch.softmax(next_logits, dim=-1)
        next_id = torch.multinomial(probs, num_samples=1).item()
        generated.append(next_id)
        # Early stop if a stop token has been emitted.
        text_so_far = tokenizer.decode(generated)
        if any(stok in text_so_far for stok in stop_tokens):
            break

    text = tokenizer.decode(generated)
    for stok in stop_tokens:
        if stok in text:
            text = text[:text.index(stok)]
    return text.strip()


def is_non_degenerate(reply: str, prompt: str) -> bool:
    """A reply passes the smoke gate if it is non-empty, not an echo of the
    prompt, and not a single repeated character/token."""
    r = reply.strip()
    if len(r) < 2:
        return False
    if r == prompt.strip():
        return False
    # Reject "aaaa" / "    " style degenerate output.
    if len(set(r.replace(" ", ""))) <= 1:
        return False
    return True


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Build a real chat/base HVN checkpoint (Issue #51)")
    p.add_argument("--out", type=str, default="hvn_chat_checkpoint.pt",
                   help="Output checkpoint path (tokenizer saved alongside as <stem>_tokenizer.json)")
    p.add_argument("--wikitext-path", type=str, default="wikitext-103-v1",
                   help="Local directory with WikiText-103 (wiki.train.tokens / wiki.valid.tokens)")
    # Model size (6 GB RTX 4050 target: d=256-512).
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=3)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--bond_dim", type=int, default=64)
    p.add_argument("--chunk_size", type=int, default=64)
    p.add_argument("--seq_len", type=int, default=128)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--bpe_vocab", type=int, default=8000)
    # Pretrain budget (streaming char cap over the 539 MB WikiText-103 train split).
    p.add_argument("--pretrain_chars", type=int, default=30_000_000,
                   help="Cap of WikiText-103 train characters to stream (bounds memory)")
    p.add_argument("--pretrain_epochs", type=int, default=1)
    p.add_argument("--finetune_epochs", type=int, default=10)
    p.add_argument("--instruction_examples", type=int, default=400)
    p.add_argument("--chat_examples", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--finetune_lr", type=float, default=1e-4)
    p.add_argument("--precision", type=str, default="bf16", choices=["fp32", "fp16", "bf16"])
    p.add_argument("--no_deltanet", action="store_true", help="Disable Gated-DeltaNet (not recommended)")
    p.add_argument("--compile", action="store_true", help="torch.compile (opt-in; warmup cost)")
    p.add_argument("--smoke", action="store_true",
                   help="Fast loop-validation: tiny budget/model/epochs (CI). Proves the loop, not quality.")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)

    if args.smoke:
        # Tiny everything — just enough to exercise every stage end-to-end.
        args.d_model = 128
        args.num_layers = 2
        args.bpe_vocab = 2000
        args.pretrain_chars = 400_000
        args.pretrain_epochs = 1
        args.finetune_epochs = 4
        args.instruction_examples = 120
        args.chat_examples = 120
        args.seq_len = 96
        print("[smoke] fast loop-validation config (not a quality run)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_cuda = device.type == "cuda"
    use_deltanet = not args.no_deltanet
    print(f"Device: {device} | precision={args.precision} | deltanet={use_deltanet}")

    # --- 1. Load pretrain text + build fine-tune corpus --------------------
    print("\n[1/5] Loading corpora ...")
    pre_train_text, pre_eval_text = load_wikitext103(
        local_dir=args.wikitext_path,
        max_train_chars=args.pretrain_chars,
        max_eval_chars=200_000,
    )
    print(f"  WikiText-103 pretrain: {len(pre_train_text):,} chars | eval: {len(pre_eval_text):,} chars")

    ft_train_text, ft_heldout_text = build_finetune_text(
        args.instruction_examples, args.chat_examples, seed=args.seed,
    )
    print(f"  Fine-tune: {len(ft_train_text):,} train chars | {len(ft_heldout_text):,} held-out chars")

    # --- 2. Train a real BPE tokenizer over the full corpus ----------------
    print("\n[2/5] Training BPE tokenizer ...")
    # Cap the tokenizer-training sample so BPE training stays fast; include ALL
    # fine-tune text so chat/instruction subwords + special tokens are learned.
    bpe_sample = pre_train_text[: min(len(pre_train_text), 20_000_000)] + "\n" + ft_train_text
    tokenizer = BPETokenizer.train(bpe_sample, vocab_size=args.bpe_vocab)
    print(f"  Vocab size: {tokenizer.vocab_size} BPE tokens")

    # --- 3. Pretrain on WikiText-103 ---------------------------------------
    print("\n[3/5] Pretraining on WikiText-103 ...")
    pre_ids = tokenizer.encode(pre_train_text)
    print(f"  Pretrain tokens: {len(pre_ids):,}")
    pre_ds = TextDataset(pre_ids, seq_len=args.seq_len)
    pin = use_cuda
    pre_loader = DataLoader(pre_ds, batch_size=args.batch_size, shuffle=True,
                            drop_last=False, pin_memory=pin)

    model = HamiltonianVortexNetwork(
        vocab_size=tokenizer.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        bond_dimension=args.bond_dim,
        chunk_size=args.chunk_size,
        dropout=0.1,
        use_deltanet=use_deltanet,
    )
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    t0 = time.time()
    train_model(
        model, pre_loader, epochs=args.pretrain_epochs, lr=args.lr, device=device,
        save_path=args.out, compile=args.compile, precision=args.precision,
        fused_optimizer=use_cuda, defer_loss_sync=True, non_blocking=pin,
    )
    print(f"  Pretrain done in {time.time() - t0:.1f}s")

    # --- 4. Instruction / chat fine-tune -----------------------------------
    print("\n[4/5] Instruction/chat fine-tuning ...")
    ft_ids = tokenizer.encode(ft_train_text)
    print(f"  Fine-tune tokens: {len(ft_ids):,}")
    ft_ds = TextDataset(ft_ids, seq_len=args.seq_len)
    ft_loader = DataLoader(ft_ds, batch_size=args.batch_size, shuffle=True,
                           drop_last=False, pin_memory=pin)

    t0 = time.time()
    train_model(
        model, ft_loader, epochs=args.finetune_epochs, lr=args.finetune_lr, device=device,
        save_path=args.out, compile=args.compile, precision=args.precision,
        fused_optimizer=use_cuda, defer_loss_sync=True, non_blocking=pin,
    )
    print(f"  Fine-tune done in {time.time() - t0:.1f}s")

    # Save tokenizer alongside the checkpoint (train_model already saved <out>).
    tok_path = args.out.replace(".pt", "_tokenizer.json")
    tokenizer.save(tok_path)
    print(f"  ✓ Checkpoint: {args.out}\n  ✓ Tokenizer:  {tok_path}")

    # --- 5. Evaluation + smoke-test gate -----------------------------------
    print("\n[5/5] Held-out evaluation + smoke gate ...")
    wt_ppl, wt_bpc = heldout_bpc(model, tokenizer, pre_eval_text, args.seq_len, device)
    ft_ppl, ft_bpc = heldout_bpc(model, tokenizer, ft_heldout_text, args.seq_len, device)
    print(f"  WikiText-103 held-out : PPL {wt_ppl:.2f} | {wt_bpc:.3f} bits/char")
    print(f"  Fine-tune  held-out   : PPL {ft_ppl:.2f} | {ft_bpc:.3f} bits/char")

    # Best-effort char-baseline comparison (the existing tiny char checkpoint).
    _compare_char_baseline(ft_heldout_text, args.seq_len, device)

    print("\n  --- Smoke prompts ---")
    formatter = ConversationFormatter(system_prompt="You are a helpful assistant.")

    coding_prompt = format_instruction_example({
        "instruction": "Generate Python code for the following task.",
        "input": "Create a function that adds two numbers.",
        "output": "",
    }).rstrip()  # ends at "### Response:\n"
    coding_reply = complete(model, tokenizer, coding_prompt, max_new_tokens=60,
                            temperature=0.3, top_k=5, device=device)
    print(f"  [coding]   prompt='Create a function that adds two numbers'")
    print(f"             reply={coding_reply!r}")

    greeting_prompt = formatter.format_for_generation([{"role": "user", "content": "Hello!"}])
    greeting_reply = complete(model, tokenizer, greeting_prompt, max_new_tokens=40,
                              temperature=0.5, top_k=10, device=device)
    print(f"  [greeting] prompt='Hello!'")
    print(f"             reply={greeting_reply!r}")

    ok_coding = is_non_degenerate(coding_reply, coding_prompt)
    ok_greeting = is_non_degenerate(greeting_reply, greeting_prompt)
    passed = ok_coding and ok_greeting
    print(f"\n  Smoke gate: coding={'PASS' if ok_coding else 'FAIL'} | "
          f"greeting={'PASS' if ok_greeting else 'FAIL'}")

    if passed:
        print("\n✓ Pipeline complete — checkpoint is ready for the Django app (Issue #52).")
    else:
        print("\n✗ Smoke gate FAILED — reply(ies) degenerate. Train longer / larger "
              "(remove --smoke, raise --finetune_epochs / --d_model).")
    return 0 if passed else 1


def _compare_char_baseline(heldout_text: str, seq_len: int, device: torch.device):
    """If the bundled char-level checkpoint exists, report its bits-per-char on
    the same held-out text so the #51 'beats the char baseline' bar is concrete.
    Best-effort: silently skips if the baseline isn't present/loadable."""
    ckpt_path, tok_path = "hvn_checkpoint.pt", "hvn_checkpoint_tokenizer.json"
    if not (os.path.exists(ckpt_path) and os.path.exists(tok_path)):
        return
    try:
        from train import CharTokenizer
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        base = HamiltonianVortexNetwork(
            vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"],
            num_layers=ckpt["num_layers"], n_heads=ckpt.get("n_heads", 4),
            chunk_size=ckpt.get("chunk_size", 64),
            use_deltanet=ckpt.get("use_deltanet", False),
            deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        )
        base.load_state_dict(ckpt["model_state_dict"])
        base.to(device).eval()
        base_tok = CharTokenizer.load(tok_path)
        _, base_bpc = heldout_bpc(base, base_tok, heldout_text, seq_len, device)
        print(f"  Char baseline (hvn_checkpoint.pt): {base_bpc:.3f} bits/char "
              f"(lower is better — the BPE chat model should beat this)")
    except Exception as exc:  # pragma: no cover - best effort
        print(f"  (char-baseline comparison skipped: {type(exc).__name__}: {exc})")


if __name__ == "__main__":
    raise SystemExit(main())
