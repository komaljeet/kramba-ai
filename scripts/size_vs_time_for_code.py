"""Which model size gives a CONVERGED code model soonest? (#163)

Author: Rambarun Komaljeet

The 157M run sees 328M tokens in ~20 h -- **10% of its Chinchilla requirement**.
An undertrained large model is worse than a converged small one at equal compute
(#112: "a 5-hour from-scratch run at any size lands at 0.7-6.6% of its required
tokens, i.e. strictly worse than the model already on disk").

So the question is not "how big can we train" but **"what is the largest model
that CONVERGES inside the time we have"**, given a fixed 3.83B-token corpus.

Two things this measures rather than extrapolates:

* **Real throughput per config.** Per-token cost scales as `layers x d^2`, not as
  parameter count, so scaling from the 157M number would be wrong.
* **Body vs vocabulary split.** #139 found 62% of the "83M" was two embedding
  tables. Vocabulary parameters do not add capability the way body parameters
  do, and their share GROWS as the model shrinks -- at 30M a 32K vocab could be
  most of the model. Tied embeddings (#139, now default) halve that, and the
  table below reports body separately because that is what actually learns.

Reports, per config: params, body params, measured s/step, Chinchilla tokens
(20/param on the BODY), hours to reach that, and how many times over the corpus
could train it (the "overtraining headroom" that small models exploit).

Usage::

    python scripts/size_vs_time_for_code.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN                    # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model   # noqa: E402
from hvn.optimizers import build_optimizer            # noqa: E402

#: (d_model, layers, heads) -- heads chosen so d_head is a power of 2 (Triton
#: requires it) and >= 32, the width #159 measured as the useful floor.
CONFIGS = [
    ("896x16", 896, 16, 28),      # the current run
    ("768x14", 768, 14, 24),
    ("640x12", 640, 12, 20),
    ("512x12", 512, 12, 16),
    ("512x8",  512,  8, 16),
    ("384x8",  384,  8, 12),
]


def build(d_model, layers, heads, vocab, device):
    attn = (max(1, layers // 4), max(2, (3 * layers) // 4))
    plan = ["transformer" if i in attn else "deltanet" for i in range(layers)]
    m = EdgeHVN(vocab_size=vocab, d_model=d_model, num_layers=layers,
                n_heads=heads, block_plan=plan, attn_window=4096,
                tie_embeddings=True, grad_checkpoint=True).to(device)
    lin = {n for n, x in m.named_modules() if isinstance(x, nn.Linear)}
    ex = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(m, target_names=[n for n in lin
                                            if not any(p in n for p in ex)])
    for b in m.blocks:
        if hasattr(b, "use_triton_scan"):
            b.use_triton_scan = True
            b.use_megakernel_scan = True
    return m


def measure(name, d_model, layers, heads, args, device):
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    m = build(d_model, layers, heads, args.vocab_size, device)

    total = sum(p.numel() for p in m.parameters())
    # Body = everything that is not the (tied) embedding table. That is what
    # learns; the vocabulary is a lookup and adds no per-token FLOPs.
    embed = m.token_embedding.weight.numel()
    body = total - embed

    opt, _ = build_optimizer(m, "muon", lr=3e-4, weight_decay=0.01, muon_lr=0.02)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(1)

    def one_step():
        for _ in range(args.grad_accum):
            x = torch.randint(0, args.vocab_size, (1, args.seq_len),
                              generator=g).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(m(x).reshape(-1, args.vocab_size),
                            torch.roll(x, -1, 1).reshape(-1)) / args.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    one_step()
    per = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        one_step()
        per.append(time.perf_counter() - t0)
    s_step = sorted(per)[len(per) // 2]
    vram = torch.cuda.max_memory_reserved() / 2 ** 20
    del m, opt
    torch.cuda.empty_cache()

    toks_step = args.seq_len * args.grad_accum
    tok_s = toks_step / s_step
    chinchilla = body * 20                      # 20 tokens per BODY param
    hours = chinchilla / tok_s / 3600
    headroom = args.corpus_tokens / chinchilla  # how many passes the corpus allows
    return {"name": name, "d_model": d_model, "layers": layers, "heads": heads,
            "params": total, "body": body, "vocab_share": embed / total * 100,
            "s_step": s_step, "tok_s": tok_s, "vram_mib": vram,
            "chinchilla_tokens": chinchilla, "hours_to_chinchilla": hours,
            "corpus_headroom": headroom}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=32003)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--corpus-tokens", type=float, default=3.83e9)
    p.add_argument("--out", default="results/size_vs_time/results.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    print(f"corpus {args.corpus_tokens/1e9:.2f}B tokens | "
          f"{args.seq_len * args.grad_accum:,} tokens/step | tied embeddings\n")
    print(f"{'config':<9}{'params':>10}{'body':>10}{'vocab%':>8}{'s/step':>8}"
          f"{'tok/s':>8}{'Chin.tok':>10}{'HOURS':>8}{'passes':>8}{'VRAM':>8}")

    out = []
    for name, d, l, h in CONFIGS:
        if d % h:
            continue
        try:
            r = measure(name, d, l, h, args, device)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"{name:<9}{'OOM':>10}")
            continue
        out.append(r)
        print(f"{name:<9}{r['params']/1e6:>9.0f}M{r['body']/1e6:>9.0f}M"
              f"{r['vocab_share']:>7.0f}%{r['s_step']:>8.2f}{r['tok_s']:>8.0f}"
              f"{r['chinchilla_tokens']/1e9:>9.2f}B{r['hours_to_chinchilla']:>8.1f}"
              f"{r['corpus_headroom']:>8.1f}{r['vram_mib']:>7.0f}M")

    if out:
        print("\nHOURS = time to reach Chinchilla-optimal (20 tokens per BODY param).")
        print("passes = how many times the corpus covers that budget; >1 means "
              "there is room to OVERTRAIN, which is what makes small models "
              "punch above their size.")
        fits = [r for r in out if r["hours_to_chinchilla"] <= 24]
        if fits:
            best = max(fits, key=lambda r: r["body"])
            print(f"\nLargest model that CONVERGES inside 24 h: **{best['name']}** "
                  f"-- {best['body']/1e6:.0f}M body, "
                  f"{best['hours_to_chinchilla']:.1f} h, "
                  f"{best['corpus_headroom']:.1f}x corpus headroom.")
        else:
            print("\nNothing converges inside 24 h at these sizes.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 163, "config": vars(args), "results": out}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
