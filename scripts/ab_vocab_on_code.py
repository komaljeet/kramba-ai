"""Does the 32K vocab win on CODE? (Issue #139 follow-up)

Written by: Rambarun Komaljeet

#139 measured the 32K mixed vocabulary as **1.9% worse** than GPT-2 on held-out
bits/byte — but on FineWeb-Edu, pure English prose, with no code in either arm's
training mix. That is the one domain where a code-and-chat vocabulary cannot win,
so the result was close to tautological: *a code vocabulary loses on an
English-only corpus*.

Its measured advantages are code (0.701 tokens/line) and chat (0.759). This runs
the test it never got: **code in the training mix, code in the held-out set, the
tokenizer as the only difference, graded on bits per byte.**

Falsifiability, asked before running: if the 32K vocab were NOT better on code,
would this catch it? Yes — bits/byte normalises against source bytes, so it is
comparable across vocabularies, and both arms see identical held-out bytes. A
tokenizer that merely produced fewer tokens without modelling code better would
show flat or worse bits/byte.

Held-out is the TAIL of the corpus file, which is different repositories and
languages from the head (the collector walks repos in sorted order), so it is not
a memorisation check. Stated because it is the weakest part of the design.

Usage
-----
    python scripts/ab_vocab_on_code.py --steps 1500
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from eval.bits_per_byte import bits_per_byte  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402
from hvn.optimizers import build_optimizer  # noqa: E402
from train_edge_conversational import load_tokenizer  # noqa: E402

CODE_TXT = "data_cache/code_corpus/code_all.txt"


def split_corpus(path: str, heldout_bytes: int):
    """Train = head, held-out = tail. Returns (train_text, heldout_text)."""
    # Slice by CHARACTERS after reading, not by a byte count passed to read():
    # read(n) consumes n characters, and this corpus has fewer characters than
    # bytes, so `read(total_bytes - heldout)` hit EOF and left the held-out set
    # EMPTY -- which bits_per_byte would have happily reported as nan.
    with open(path, encoding="utf-8", errors="replace") as fh:
        text = fh.read()
    n = max(1, min(heldout_bytes, len(text) // 4))
    return text[:-n], text[-n:]


def tokenize(text: str, tok, workers: int) -> np.ndarray:
    from train_edge_conversational import _encode_parallel
    out = []
    step = 4_000_000
    for i in range(0, len(text), step):
        out.extend(_encode_parallel(tok, text[i:i + step], workers))
    return np.asarray(out, dtype=np.uint16)


def train_arm(name, tok_spec, ids, args, heldout_text, device) -> dict:
    tok = load_tokenizer(tok_spec)
    V = tok.vocab_size
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    plan = ["deltanet"] * args.layers
    for i in args.attn_layers:
        plan[i] = "transformer"
    model = EdgeHVN(vocab_size=V, d_model=args.d_model, num_layers=args.layers,
                    n_heads=args.n_heads, block_plan=plan,
                    attn_window=args.attn_window,
                    tie_embeddings=True,          # #139: default, and free
                    eos_token_id=tok.eos_token_id or 0).to(device)
    # NO ternary in this A/B, in BOTH arms. apply_ternary_to_model without an
    # exclusion list converts the decoder as well, which silently BREAKS tied
    # embeddings: measured 47.9M tied -> 73.6M, exactly one embedding table back.
    # build_model_and_config excludes the decoder, which is why #139's tied A/B
    # measured the saving correctly. Rather than replicate that exclusion list
    # here, this comparison drops ternary entirely -- it is identical in both
    # arms, so it cannot affect which tokenizer wins.
    for blk in model.blocks:
        inner = getattr(blk, "block", blk)
        if hasattr(inner, "use_triton_scan"):
            inner.use_triton_scan = True
            inner.use_megakernel_scan = True

    opt, desc = build_optimizer(model, "muon", lr=3e-4, weight_decay=0.01,
                               muon_lr=0.02)
    base_lrs = [pg["lr"] for pg in opt.param_groups]
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)

    S, B = args.seq_len, args.batch_size
    n_win = (len(ids) - 1) // S
    print(f"[code-ab] {name}: vocab {V}, {len(ids):,} train tokens, "
          f"{n_win:,} windows | {desc}")

    rng = np.random.RandomState(args.seed)
    model.train()
    t0 = time.time()
    for step in range(1, args.steps + 1):
        sel = rng.randint(0, n_win, size=B)
        xb = np.stack([ids[i * S:(i + 1) * S] for i in sel]).astype(np.int64)
        yb = np.stack([ids[i * S + 1:(i + 1) * S + 1] for i in sel]).astype(np.int64)
        x = torch.from_numpy(xb).to(device)
        y = torch.from_numpy(yb).to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(),
                                   y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if step <= args.warmup:
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = b * step / args.warmup
        opt.step()
        opt.zero_grad(set_to_none=True)
        if step > args.warmup:
            cos.step()
        if step % 250 == 0:
            print(f"[code-ab]   {name} step {step}/{args.steps} "
                  f"loss {float(loss):.4f} ({(time.time()-t0)/60:.1f} min)",
                  flush=True)

    model.eval()
    bpb = bits_per_byte(model, tok, heldout_text, seq_len=S, device=device)
    wall = time.time() - t0
    print(f"[code-ab] {name}: bits/byte {bpb['bits_per_byte']:.4f} "
          f"| tokens/byte {bpb['tokens_per_byte']:.4f} | {wall/60:.1f} min")
    del model, opt
    torch.cuda.empty_cache()
    return {"vocab_size": V, "bits_per_byte": bpb, "wall_sec": wall,
            "train_tokens": int(len(ids))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--attn-layers", type=int, nargs="+", default=[3, 10])
    ap.add_argument("--attn-window", type=int, default=4096)
    ap.add_argument("--heldout-bytes", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/vocab/ab_code.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md).")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device("cuda")

    train_text, heldout_text = split_corpus(CODE_TXT, args.heldout_bytes)
    print(f"[code-ab] train {len(train_text)/1024**2:.0f} MB | "
          f"held-out {len(heldout_text)/1024**2:.2f} MB (corpus tail)")

    workers = max(1, os.cpu_count() or 4)
    arms = {}
    for name, spec in (("gpt2", "gpt2"), ("v32k", "data_cache/tokenizer_32k")):
        tok = load_tokenizer(spec)
        t0 = time.time()
        ids = tokenize(train_text, tok, workers)
        print(f"[code-ab] {name}: tokenized in {time.time()-t0:.0f}s -> "
              f"{len(ids):,} tokens")
        arms[name] = train_arm(name, spec, ids, args, heldout_text, device)
        del ids

    g, k = arms["gpt2"]["bits_per_byte"], arms["v32k"]["bits_per_byte"]
    dq = (k["bits_per_byte"] / g["bits_per_byte"] - 1) * 100
    dt = (k["tokens_per_byte"] / g["tokens_per_byte"] - 1) * 100
    print("\n[code-ab] ================ CODE VERDICT ================")
    print(f"[code-ab] gpt2  bits/byte {g['bits_per_byte']:.4f}  "
          f"tokens/byte {g['tokens_per_byte']:.4f}")
    print(f"[code-ab] v32k  bits/byte {k['bits_per_byte']:.4f}  "
          f"tokens/byte {k['tokens_per_byte']:.4f}")
    print(f"[code-ab] 32K is {dq:+.2f}% bits/byte (negative = BETTER) and uses "
          f"{dt:+.2f}% tokens per byte")
    print(f"[code-ab] -> {'32K WINS on code' if dq < -0.5 else 'no clear win on code' if dq < 0.5 else '32K LOSES on code too'}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "arms": arms,
                   "bits_per_byte_pct_change": dq,
                   "tokens_per_byte_pct_change": dt}, fh, indent=2)
    print(f"[code-ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
