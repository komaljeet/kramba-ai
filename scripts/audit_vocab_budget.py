"""Where do the parameters actually go? (Issue #139)

Written by: Rambarun Komaljeet

Reports the {50257, 16K} x {tied, untied} 2x2: parameter counts, forward FLOPs
per token, and peak training VRAM — through the REAL model-construction path
(``build_model_and_config``, with ternary and the Triton scan), never a
standalone ``EdgeHVN``.

That last point is not pedantry. Every VRAM envelope in ``docs/training.md`` was
wrong for two years because the benches hardcoded ``VOCAB = 8000`` against the
real pretrain's 50,257 and skipped ternary and the scan — so a config documented
as "4.3 GiB, the recommended ceiling" actually needed 6.34 GiB and spilled.
Repeating that here would produce another wrong envelope about vocabulary
specifically, which is the one thing this issue must get right.

FLOPs are counted analytically rather than profiled, because the quantity that
matters is the *ratio* between the head and the blocks, and a profiler on this
card mixes in launch overhead that would obscure it.

Usage
-----
    python scripts/audit_vocab_budget.py
    python scripts/audit_vocab_budget.py --vram     # adds a real VRAM pass (GPU)
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def param_breakdown(model) -> dict:
    groups = {"token_embedding": 0, "output_head": 0, "blocks_and_other": 0}
    for n, p in model.named_parameters():
        ln = n.lower()
        if "token_embedding" in ln or "embed" in ln:
            groups["token_embedding"] += p.numel()
        elif "decoder" in ln or "lm_head" in ln or "output" in ln.split(".")[0]:
            groups["output_head"] += p.numel()
        else:
            groups["blocks_and_other"] += p.numel()
    groups["total"] = sum(p.numel() for p in model.parameters())
    return groups


def forward_flops_per_token(d_model: int, n_layers: int, vocab: int,
                            ffn_expansion: int = 3) -> dict:
    """Analytic forward FLOPs per token — the head-vs-blocks ratio is the point.

    Counts 2 FLOPs per multiply-accumulate. The embedding LOOKUP is free (it is
    a gather, not a matmul) — which is exactly why the input table costs memory
    but no compute, while the output head costs both. That asymmetry is what
    makes tying and resizing different levers rather than the same one.
    """
    # Per block: DeltaNet mixer ~6 d^2, FFN 2 * expansion * d^2.
    per_block = 2 * (6 * d_model ** 2 + 2 * ffn_expansion * d_model ** 2)
    blocks = per_block * n_layers
    head = 2 * d_model * vocab
    return {
        "blocks": blocks,
        "output_head": head,
        "total": blocks + head,
        "head_share_pct": 100.0 * head / (blocks + head),
    }


_SETTINGS = {"batch_size": 4, "grad_accum": 8, "seq_len": 512,
             "d_model": 512, "num_layers": 12, "n_heads": 8}


def build(vocab: int, tie: bool, device, tokenizer=None):
    """The real construction path — ternary applied, hybrid block plan, all of it."""
    import train_edge_conversational as tec
    model, _config = tec.build_model_and_config(
        _SETTINGS, tokenizer, device,
        checkpoint_dir="checkpoints/_audit", arch="hvn-hybrid",
        apply_ternary=True, attn_layers=[3, 10], attn_window=4096,
        grad_checkpoint=False, tie_embeddings=tie,
        optimizer="muon", precision="bf16", lr=3e-4, vocab_size=vocab,
    )
    return model


def measure_vram(vocab: int, tie: bool) -> dict:
    """Peak allocated/reserved for one real forward+backward+optimizer step."""
    import train_edge_conversational as tec  # noqa: F401
    from hvn.optimizers import build_optimizer

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build(vocab, tie, device)
    opt, _ = build_optimizer(model, "muon", lr=3e-4, weight_decay=0.01, muon_lr=0.02)
    x = torch.randint(0, vocab, (4, 512), device=device)
    y = torch.randint(0, vocab, (4, 512), device=device)
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, vocab).float(), y.reshape(-1))
    loss.backward()
    opt.step()
    out = {
        "peak_alloc_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 1024 ** 3,
    }
    del model, opt, logits, loss, x, y
    torch.cuda.empty_cache()
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocabs", type=int, nargs="+", default=[50257, 16000])
    ap.add_argument("--vram", action="store_true",
                    help="also measure real peak VRAM (needs GPU)")
    ap.add_argument("--out", default="results/vocab/audit.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2TokenizerFast
    tok = GPT2TokenizerFast.from_pretrained("gpt2")
    rows = []
    print(f"{'vocab':>7} {'tied':>5} {'total':>12} {'embed':>11} {'head':>11} "
          f"{'blocks':>11} {'blk share':>9} {'MFLOP/tok':>10} {'head FLOP%':>10}")
    for vocab in args.vocabs:
        for tie in (False, True):
            model = build(vocab, tie, device, tok)
            pb = param_breakdown(model)
            fl = forward_flops_per_token(512, 12, vocab)
            row = {"vocab": vocab, "tied": tie, **pb, "flops": fl}
            if args.vram and device.type == "cuda":
                del model
                row["vram"] = measure_vram(vocab, tie)
            else:
                del model
            torch.cuda.empty_cache() if device.type == "cuda" else None
            rows.append(row)
            print(f"{vocab:>7} {str(tie):>5} {pb['total']:>12,} "
                  f"{pb['token_embedding']:>11,} {pb['output_head']:>11,} "
                  f"{pb['blocks_and_other']:>11,} "
                  f"{100*pb['blocks_and_other']/pb['total']:>8.1f}% "
                  f"{fl['total']/1e6:>10.1f} {fl['head_share_pct']:>9.1f}%")

    base = rows[0]
    print("\n[audit] vs the current config (50257, untied):")
    for r in rows[1:]:
        # SAVED, not changed: a "+31%" here means 31% FEWER parameters. The
        # first version of this line printed a bare signed percentage, which
        # reads as an increase and is exactly backwards.
        dp = 100 * (1 - r["total"] / base["total"])
        df = 100 * (1 - r["flops"]["total"] / base["flops"]["total"])
        print(f"[audit]   vocab {r['vocab']}, tied={r['tied']}: "
              f"{dp:.1f}% FEWER params, {df:.1f}% FEWER forward FLOPs, "
              f"real-model share {100*r['blocks_and_other']/r['total']:.1f}% "
              f"(was {100*base['blocks_and_other']/base['total']:.1f}%)")
    if args.vram:
        for r in rows:
            if "vram" in r:
                print(f"[audit]   vocab {r['vocab']} tied={r['tied']}: "
                      f"{r['vram']['peak_alloc_gib']:.2f} GiB alloc / "
                      f"{r['vram']['peak_reserved_gib']:.2f} GiB reserved")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"config": "d_model=512, 12 layers, 8 heads, ternary, "
                             "attn_layers=[3,10]", "rows": rows}, fh, indent=2)
    print(f"\n[audit] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
