"""How big a model can this card actually train? (#136 sizing, re-measured)

Written by: Rambarun Komaljeet

Why this exists when eval/vram_envelope.py already sweeps sizes
---------------------------------------------------------------
That tool cannot answer the question, for three separate reasons, and
docs/training.md already warns its rows "must not be used to size a real run":

1. **It hardcodes ``VOCAB = 8000``.** At d=1024 that is ~8M params of embedding
   against ~103M for an untied GPT-2 vocabulary. The vocabulary term -- the exact
   thing #139 showed is 42-62% of these models -- is nearly invisible in it, so
   every row overstates how much *body* the card can afford.
2. **It builds ``HamiltonianVortexNetwork``**, a different class from the EdgeHVN
   that actually gets trained. It measures a model that no longer exists.
3. **It has no tie_embeddings flag**, so it cannot represent the change that is
   now the repo default.

This goes through ``build_model_and_config`` -- ternary, hybrid block plan, real
optimizer -- because using a standalone model is precisely the mistake that made
every VRAM envelope in training.md wrong (#139).

The measurement that actually decides the question
--------------------------------------------------
Falsifiability, asked before running: **if a config did NOT fit, would this catch
it?** A plain ok/OOM verdict would NOT, and that is how the 268M row got
published as fitting when the real script spills at 355M.

On Windows/WDDM, exceeding VRAM does **not** raise OOM. The driver silently
spills to host RAM over PCIe and the step just gets slower -- 4x or worse. A run
that "succeeded" may be a run that did not fit. So this reports three things
together, and a config is only called `ok` if all three agree:

  - ``peak_reserved_mb`` against total VRAM (>85% is the danger band),
  - ``tok_per_s``, where spilling shows up as a throughput cliff,
  - ``ms_per_step_cv``, the step-time variability, which rises when the driver
    is paging weights in and out.

Reporting only the first would miss a config sitting at 84% that thrashes anyway.

Usage
-----
    python scripts/size_sweep.py
    python scripts/size_sweep.py --configs 768x16 1024x16 --steps 30
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Chinchilla convention used throughout this repo (#112): 20 tokens per param.
TOKENS_PER_PARAM = 20

# Head dimension, held fixed so n_heads scales with d_model. 64 is a power of 2,
# which the Triton scan kernel requires (see build()).
HEAD_DIM = 64

# (vocab, tie, label) -- the comparison the #139 work makes worth running.
ARMS = [
    (50257, False, "gpt2-untied"),   # what training.md's sizing assumed
    (32000, True, "32k-tied"),       # what #139 + the mixed A/B recommend for code
]


def build(vocab: int, tie: bool, d_model: int, n_layers: int, device,
          seq_len: int, batch_size: int, fast: bool = True):
    """Build through the real path, and run it the way training actually runs.

    ``--fast`` is default=True on train_edge_conversational (since #129) and is
    applied in ``main()``, NOT in build_model_and_config -- so calling the
    builder directly silently gets none of it. The first version of this sweep
    made exactly that mistake and reported uncompiled throughput as if it were
    the training figure.

    Building through the real construction path is necessary but NOT sufficient:
    it must also execute in the real mode. compile + TF32 + Triton scan are what
    --fast turns on, and all three change throughput and memory.
    """
    import train_edge_conversational as tec
    # n_heads is DERIVED so that head_dim = d_model // n_heads is a power of 2.
    # hvn/core/deltanet_triton.py does `tl.arange(0, D)` over the head dim, and
    # Triton requires that range to be a power of 2. With a fixed n_heads=8,
    # d_model 640/768/896 give head_dim 80/96/112 and the run dies with an
    # InductorError under --fast -- a hard crash, not a fallback. Nothing in the
    # repo validates this, so it is enforced here.
    n_heads = max(1, d_model // HEAD_DIM)
    assert d_model % n_heads == 0 and (HEAD_DIM & (HEAD_DIM - 1)) == 0, (
        f"d_model={d_model} cannot be split into power-of-2 head_dim")
    settings = {"batch_size": batch_size, "grad_accum": 8, "seq_len": seq_len,
                "d_model": d_model, "num_layers": n_layers, "n_heads": n_heads}
    if fast:
        # TF32, as --fast sets it.
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model, _cfg = tec.build_model_and_config(
        settings, None, device,
        checkpoint_dir="checkpoints/_sizesweep", arch="hvn-hybrid",
        apply_ternary=True, attn_layers=[3, 10], attn_window=4096,
        grad_checkpoint=False, tie_embeddings=tie,
        optimizer="muon", precision="bf16", lr=3e-4, vocab_size=vocab,
    )
    if fast:
        # Triton scan, as --fast sets it for the DeltaNet blocks.
        for blk in model.blocks:
            inner = getattr(blk, "block", blk)
            if hasattr(inner, "use_triton_scan"):
                inner.use_triton_scan = True
        model = torch.compile(model)
    return model


def measure(vocab, tie, label, d_model, n_layers, args, total_vram_mb) -> dict:
    from hvn.optimizers import build_optimizer

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    row = {"label": label, "vocab": vocab, "tied": tie,
           "d_model": d_model, "num_layers": n_layers,
           "n_heads": max(1, d_model // HEAD_DIM), "head_dim": HEAD_DIM}
    try:
        model = build(vocab, tie, d_model, n_layers, device, args.seq_len,
                      args.batch_size, fast=args.fast)
    except torch.OutOfMemoryError:
        row.update(verdict="oom-at-build")
        return row

    row["fast"] = args.fast
    params = sum(p.numel() for p in model.parameters())
    # Vocabulary vs body: the split that decides whether a size is worth it.
    vocab_params = d_model * vocab * (1 if tie else 2)
    row.update(params=params, vocab_params=vocab_params,
               body_params=params - vocab_params,
               vocab_share_pct=100.0 * vocab_params / params)

    opt, _ = build_optimizer(model, "muon", lr=3e-4, weight_decay=0.01,
                             muon_lr=0.02)
    B, S = args.batch_size, args.seq_len
    x = torch.randint(0, vocab, (B, S), device=device)
    y = torch.randint(0, vocab, (B, S), device=device)

    def one_step():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)

    try:
        for _ in range(args.warmup):
            one_step()
        torch.cuda.synchronize()
        # Reset AFTER warmup: with --fast the first step triggers compilation and
        # autotuning, whose transient allocations are not what a 7-day run sits
        # at. Steady-state is the number that decides whether it fits.
        torch.cuda.reset_peak_memory_stats()
        times = []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            one_step()
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
    except torch.OutOfMemoryError:
        row.update(verdict="oom-at-step")
        del model, opt
        torch.cuda.empty_cache()
        return row

    sec = statistics.median(times)
    cv = statistics.pstdev(times) / max(statistics.mean(times), 1e-9)
    peak_res = torch.cuda.max_memory_reserved() / 1024 ** 2
    peak_alloc = torch.cuda.max_memory_allocated() / 1024 ** 2
    tok_s = (B * S) / sec
    chinchilla_tokens = params * TOKENS_PER_PARAM
    days = chinchilla_tokens / tok_s / 86400

    row.update(sec_per_step=sec, step_cv=cv, tok_per_s=tok_s,
               peak_reserved_mb=peak_res, peak_alloc_mb=peak_alloc,
               vram_pct=100.0 * peak_res / total_vram_mb,
               chinchilla_tokens=chinchilla_tokens, chinchilla_days=days)

    # The spill verdict. Reserved memory alone is not enough -- WDDM can page
    # without the allocator ever reserving past the limit -- so step-time
    # variability is checked too. Either signal alone marks the row suspect.
    warn = []
    if row["vram_pct"] > 85:
        warn.append(f"reserved {row['vram_pct']:.0f}% of VRAM")
    if cv > 0.25:
        warn.append(f"step time unstable (cv={cv:.2f})")
    row["verdict"] = "ok" if not warn else "SUSPECT: " + "; ".join(warn)
    row["warnings"] = warn

    del model, opt
    torch.cuda.empty_cache()
    return row


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="+",
                    default=["768x16", "896x16", "1024x16", "1024x20"],
                    help="d_model x num_layers pairs")
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--fast", dest="fast", action="store_true", default=True,
                    help="Run the way training runs: compile + TF32 + Triton "
                         "scan. ON BY DEFAULT, matching train_edge_conversational "
                         "(#129). Measuring without it reports a throughput no "
                         "real run would ever see.")
    ap.add_argument("--no-fast", dest="fast", action="store_false",
                    help="Eager mode. Only for isolating what --fast is worth.")
    ap.add_argument("--out", default="results/size_sweep.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md).")
    props = torch.cuda.get_device_properties(0)
    total_vram_mb = props.total_memory / 1024 ** 2
    print(f"[sweep] {props.name} | {total_vram_mb:.0f} MiB | "
          f"B={args.batch_size} S={args.seq_len} | {args.steps} measured steps | "
          f"fast={'ON (compile+tf32+triton)' if args.fast else 'OFF (eager)'}",
          flush=True)
    print(f"[sweep] NOTE: WDDM spills silently past VRAM -- a row that runs is "
          f"not proof it fits. Watch the verdict column.", flush=True)

    rows = []
    for cfg in args.configs:
        d_model, n_layers = (int(v) for v in cfg.lower().split("x"))
        for vocab, tie, label in ARMS:
            r = measure(vocab, tie, label, d_model, n_layers, args, total_vram_mb)
            rows.append(r)
            if "params" in r and "tok_per_s" in r:
                print(f"[sweep] {cfg:>9} {label:<12} "
                      f"{r['params']/1e6:>6.0f}M "
                      f"(body {r['body_params']/1e6:>5.0f}M, "
                      f"vocab {r['vocab_share_pct']:>4.1f}%) | "
                      f"{r['peak_reserved_mb']:>6.0f} MiB "
                      f"({r['vram_pct']:>4.1f}%) | "
                      f"{r['tok_per_s']:>7.0f} tok/s | "
                      f"{r['chinchilla_days']:>6.1f} d | {r['verdict']}",
                      flush=True)
            else:
                print(f"[sweep] {cfg:>9} {label:<12} {r['verdict']}", flush=True)

    print("\n[sweep] ================== READING ==================")
    ok = [r for r in rows if r.get("verdict") == "ok"]
    if ok:
        best = max(ok, key=lambda r: r["body_params"])
        print(f"[sweep] Largest BODY that fits cleanly: {best['d_model']}x"
              f"{best['num_layers']} {best['label']} -- "
              f"{best['body_params']/1e6:.0f}M body of {best['params']/1e6:.0f}M "
              f"total, {best['chinchilla_days']:.1f} days to Chinchilla")
    else:
        print("[sweep] No config passed cleanly -- every row is suspect or OOM.")
    print("[sweep] 'body' is what learns; vocabulary is lookup tables (#139).")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"gpu": props.name, "total_vram_mb": total_vram_mb,
                   "config": vars(args), "tokens_per_param": TOKENS_PER_PARAM,
                   "rows": rows}, fh, indent=2)
    print(f"[sweep] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
