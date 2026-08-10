"""Does the 32K vocab win on a MIXED corpus? (the #139 one-way-door question)

Written by: Rambarun Komaljeet

#139 graded the 32K mixed vocabulary on English only and measured it 1.9% WORSE
per byte. The code follow-up graded it on code only and measured it 6.26% BETTER.
Both were single-domain, and they disagree in sign -- so neither answers the
question that actually decides adoption: **what happens on a corpus containing
both?**

That question matters because the tokenizer is a ONE-WAY DOOR. A model cannot be
fine-tuned onto a different vocabulary, so the choice is made once, before the
pretrain, and lives for the life of the model.

What this measures
------------------
Both arms train on the SAME mixed corpus (50% code / 40% English / 10% chat by
bytes -- the mix the 32K tokenizer was itself trained on) and are then scored on
THREE separate held-out sets, one per domain, on identical bytes per domain.

**Per-domain, not blended-only, and that is the whole design.** The two domains
pull in opposite directions. A single blended number would average a win against
a loss and report something in the middle that is true of no corpus anyone would
actually train on. With per-domain bits/byte you can re-blend for any mix
afterwards -- including mixes nobody has proposed yet -- without spending another
GPU hour.

Falsifiability, asked before running
------------------------------------
If the 32K vocab were NOT better on a mixed corpus, would this catch it? Yes.
bits/byte normalises against source bytes, so it is comparable across
vocabularies, and each domain's two arms see byte-identical held-out text. The
English arm is where the 32K vocab is already known to lose; if that loss
dominates, the blended figure goes positive and the per-domain table shows
exactly which domain did it. A result where 32K wins everywhere and a result
where it loses everywhere look completely different here.

The guard against a vacuous test: #139 shipped a held-out set that was silently
EMPTY (read(n) consumes characters, not bytes, so the slice hit EOF) and
bits_per_byte would have reported nan without complaint. `_check_heldout` below
refuses to run unless every domain has real bytes and enough windows to be worth
believing. It exists only to fail.

Known limit, stated up front
----------------------------
The arms are matched on TOKENS, not bytes -- both consume the same number of
tokens, so the 32K arm sees more source text. That is what a tokenizer does and
it is the intended effect, not a confound, but it means part of any 32K win is
extra text rather than better modelling. The per-domain `tokens_per_byte` column
reports the size of that effect for each domain.

Usage
-----
    python scripts/ab_vocab_mixed.py --steps 1500
"""

from __future__ import annotations

import argparse
import json
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
ENGLISH_TXT = "data_cache/slimpajama_en.txt"
CHAT_TXT = "data_cache/oasst1.txt"

# Held-out is carved from each source BEFORE any training text is taken, so the
# two can never overlap by construction rather than by a length calculation that
# could silently be off by the size of the held-out set.
HELDOUT_CHARS = 2_000_000

# Discarded between train and held-out. Without it the two slices touch, and a
# single repository (or chat thread) straddling the boundary puts its own text on
# both sides -- held-out contamination that no amount of held-out size fixes.
GAP_CHARS = 2_000_000


def _read_head(path: str, n_chars: int) -> str:
    """Read the first n_chars CHARACTERS. Not bytes -- see the #139 bug above."""
    with open(path, encoding="utf-8", errors="replace") as fh:
        return fh.read(n_chars)


def load_domains(code_chars: int, en_chars: int, chat_chars: int):
    """Returns {domain: (train_text, heldout_text)} with disjoint slices.

    English comes from a 5.4 GB file, so it is read as a single head slice of
    train+held-out and split, never loaded whole. Code and chat are small enough
    to read fully and split head/tail, which also makes their held-out sets a
    genuinely different part of the corpus (different repos, later threads).
    """
    def _split(path: str, n_train: int):
        raw = _read_head(path, n_train + GAP_CHARS + HELDOUT_CHARS)
        train = raw[:n_train]
        held = raw[n_train + GAP_CHARS:]
        return train, held

    return {
        "code": _split(CODE_TXT, code_chars),
        # English comes from a 5.4 GB file, so it is read as one head slice and
        # split, never loaded whole.
        "english": _split(ENGLISH_TXT, en_chars),
        "chat": _split(CHAT_TXT, chat_chars),
    }


def _check_heldout(domains: dict, min_chars: int = 500_000) -> None:
    """Refuse to run on a held-out set too small to mean anything.

    This function's only job is to fail. #139's held-out set came out EMPTY and
    the metric reported it without complaint; a silent nan is how a broken
    measurement becomes a published number.
    """
    for name, (train, held) in domains.items():
        if len(held) < min_chars:
            raise SystemExit(
                f"held-out '{name}' is {len(held):,} chars, below the "
                f"{min_chars:,} floor -- refusing to publish a number from it")
        if not train.strip():
            raise SystemExit(f"train split '{name}' is empty")


def build_mixed_corpus(domains: dict, weights: dict, block_chars: int = 1_000_000) -> str:
    """Interleave the domains in ~block_chars blocks at the requested byte mix.

    Interleaved rather than concatenated: training windows are sampled at random
    offsets, and three contiguous slabs would make many windows single-domain
    while a few straddle a hard seam. Blocks put every domain within reach of the
    sampler throughout the corpus.
    """
    streams = {k: domains[k][0] for k in domains}
    # Blocks per round, proportional to the weights (x10 so 0.1 -> 1 block).
    per_round = {k: max(1, int(round(weights[k] * 10))) for k in weights}
    pos = {k: 0 for k in streams}
    out = []
    exhausted = set()
    while len(exhausted) < len(streams):
        for name in ("code", "english", "chat"):
            if name in exhausted:
                continue
            for _ in range(per_round[name]):
                p = pos[name]
                chunk = streams[name][p:p + block_chars]
                if not chunk:
                    exhausted.add(name)
                    break
                out.append(chunk)
                pos[name] = p + block_chars
    return "".join(out)


def tokenize(text: str, tok, workers: int) -> np.ndarray:
    from train_edge_conversational import _encode_parallel
    out = []
    step = 4_000_000
    for i in range(0, len(text), step):
        out.extend(_encode_parallel(tok, text[i:i + step], workers))
    return np.asarray(out, dtype=np.uint16)


def train_arm(name, tok_spec, ids, args, domains, device) -> dict:
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
                    tie_embeddings=True,
                    eos_token_id=tok.eos_token_id or 0).to(device)
    # No ternary in either arm -- identical in both, so it cannot decide the
    # winner, and applying it here would silently break tied embeddings (see
    # scripts/ab_vocab_on_code.py for the measured 47.9M -> 73.6M regression).
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
    print(f"[mix-ab] {name}: vocab {V}, {len(ids):,} train tokens, "
          f"{n_win:,} windows | {desc}", flush=True)

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
            print(f"[mix-ab]   {name} step {step}/{args.steps} "
                  f"loss {loss.detach().item():.4f} "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)

    model.eval()
    per_domain = {}
    for dom in ("code", "english", "chat"):
        held = domains[dom][1]
        res = bits_per_byte(model, tok, held, seq_len=S, device=device)
        if res["n_windows"] < args.min_windows:
            raise SystemExit(
                f"{name}/{dom}: only {res['n_windows']} scored windows, below "
                f"the {args.min_windows} floor -- too few to publish")
        per_domain[dom] = res
        print(f"[mix-ab] {name}/{dom}: bits/byte {res['bits_per_byte']:.4f} "
              f"| tokens/byte {res['tokens_per_byte']:.4f} "
              f"| {res['n_windows']} windows", flush=True)

    wall = time.time() - t0
    print(f"[mix-ab] {name}: done in {wall/60:.1f} min", flush=True)
    del model, opt
    torch.cuda.empty_cache()
    return {"vocab_size": V, "per_domain": per_domain, "wall_sec": wall,
            "train_tokens": int(len(ids))}


def blended(per_domain: dict, weights: dict) -> float:
    """bits/byte if the corpus had exactly `weights` byte-shares per domain.

    Weighted on bits/byte directly, since each domain's figure is already
    normalised per byte -- so this is what the model would score on a corpus
    mixed at those shares, not an average of unrelated quantities.
    """
    return sum(weights[d] * per_domain[d]["bits_per_byte"] for d in weights)


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
    ap.add_argument("--code-chars", type=int, default=100_000_000)
    ap.add_argument("--english-chars", type=int, default=80_000_000)
    ap.add_argument("--chat-chars", type=int, default=20_000_000)
    ap.add_argument("--min-windows", type=int, default=200,
                    help="refuse to report a domain scored on fewer windows")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="results/vocab/ab_mixed.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md).")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.use_deterministic_algorithms(True, warn_only=False)
    device = torch.device("cuda")

    t0 = time.time()
    domains = load_domains(args.code_chars, args.english_chars, args.chat_chars)
    _check_heldout(domains)

    # Actual byte shares, measured -- not the requested fractions. The mix that
    # lands is what the write-up must quote.
    train_bytes = {k: len(v[0].encode("utf-8")) for k, v in domains.items()}
    total_tb = sum(train_bytes.values())
    weights = {k: train_bytes[k] / total_tb for k in train_bytes}
    print(f"[mix-ab] loaded in {time.time()-t0:.0f}s | train mix by bytes: " +
          ", ".join(f"{k} {100*weights[k]:.1f}%" for k in
                    ("code", "english", "chat")), flush=True)
    for k in ("code", "english", "chat"):
        print(f"[mix-ab]   {k}: train {train_bytes[k]/1024**2:.0f} MB | "
              f"held-out {len(domains[k][1].encode('utf-8'))/1024**2:.2f} MB",
              flush=True)

    mixed = build_mixed_corpus(domains, weights)
    print(f"[mix-ab] mixed corpus {len(mixed.encode('utf-8'))/1024**2:.0f} MB",
          flush=True)

    workers = max(1, os.cpu_count() or 4)
    arms = {}
    for name, spec in (("gpt2", "gpt2"), ("v32k", "data_cache/tokenizer_32k")):
        tok = load_tokenizer(spec)
        ts = time.time()
        ids = tokenize(mixed, tok, workers)
        print(f"[mix-ab] {name}: tokenized in {time.time()-ts:.0f}s -> "
              f"{len(ids):,} tokens", flush=True)
        arms[name] = train_arm(name, spec, ids, args, domains, device)
        del ids

    g, k = arms["gpt2"]["per_domain"], arms["v32k"]["per_domain"]
    print("\n[mix-ab] ============== MIXED-CORPUS VERDICT ==============")
    print("[mix-ab] domain     gpt2 b/B   32K b/B    change   (neg = 32K better)")
    deltas = {}
    for dom in ("code", "english", "chat"):
        d = (k[dom]["bits_per_byte"] / g[dom]["bits_per_byte"] - 1) * 100
        deltas[dom] = d
        print(f"[mix-ab] {dom:<10} {g[dom]['bits_per_byte']:>8.4f}  "
              f"{k[dom]['bits_per_byte']:>8.4f}   {d:>+7.2f}%")

    gb, kb = blended(g, weights), blended(k, weights)
    db = (kb / gb - 1) * 100
    print(f"[mix-ab] blended at the trained mix: gpt2 {gb:.4f} -> 32K {kb:.4f} "
          f"({db:+.2f}%)")
    print(f"[mix-ab] -> {'32K WINS on the mixed corpus' if db < -0.5 else 'no clear win' if db < 0.5 else '32K LOSES on the mixed corpus'}")
    print("[mix-ab] per-domain numbers above let this be re-blended for any "
          "other mix without re-running.")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"config": vars(args), "train_bytes": train_bytes,
                   "mix_weights": weights, "arms": arms,
                   "per_domain_pct_change": deltas,
                   "blended_gpt2": gb, "blended_v32k": kb,
                   "blended_pct_change": db}, fh, indent=2)
    print(f"[mix-ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
