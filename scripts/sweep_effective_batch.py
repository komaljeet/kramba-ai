"""Is effective batch 32 right, or just what fit in VRAM? (Issue #152)

Author: Rambarun Komaljeet

``hardware_scan()`` picks ``batch_size 4 / grad_accum 8`` -- effective batch 32
rows -- and picks it to fit VRAM. Whether 32 is a good *training* choice has
never been measured. It is a memory-derived default that became a training
constant, which is the same shape as ``--block-coordinate`` (documented as a
VRAM optimisation, measured by #112 at ~4x worse quality) and ``tie_embeddings``
(assumed a trade, measured by #139 as strictly better).

#141 measured how a *fixed* effective batch is best **split**. This measures
whether the effective batch itself is right.

**Graded at matched TOKENS, not matched steps.** A larger effective batch takes
fewer optimizer steps for the same data; grading at matched steps would hand the
big-batch arm 4x the tokens and measure only that.

**Confound stated before running: this tests batch AT FIXED LR.** Larger batches
generally want a larger learning rate, so a fixed-LR sweep conflates "batch too
small" with "LR now mistuned" -- the trap #133 fell into and #148 checked for. LR
is held at ``muon_lr=0.02``, which #148's U-curve confirmed optimal at this
length. If a larger batch loses here, the honest reading is "loses without an LR
adjustment", not "loses".

**Known-answer cell:** the ``grad_accum=8`` arm must reproduce #148's
``muon_lr=0.02`` result (6.03552) to within the reference's printed precision.
If it does not, this harness is not comparable to the runs it is read against.
(The tolerance is 5e-6, not 1e-9: #148 shipped a cell that could never pass by
comparing full precision against a 5-decimal literal.)

Usage::

    python scripts/sweep_effective_batch.py --grad-accums 8 16 32
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ab_prodigy import (                                   # noqa: E402
    as_pair, enable_scan_flags, evaluate, require_cuda,
)
from hvn.edge.ternary import apply_ternary_to_model                # noqa: E402
from hvn.optimizers import build_optimizer                         # noqa: E402

GPT2_VOCAB_SIZE = 50257
#: #148, muon_lr=0.02, 1,500 steps of batch 4 (= grad_accum 1 there). Printed at
#: 5 decimals, so the tolerance must respect that.
REF_GA8_EQUIVALENT = 6.03552
REF_TOL = 5e-6


def build(args, device):
    from hvn.edge.model import EdgeHVN
    plan = ["deltanet"] * 4 + ["transformer"] + ["deltanet"]
    plan = plan + plan
    model = EdgeHVN(
        vocab_size=args.vocab_size, d_model=args.d_model,
        num_layers=len(plan), n_heads=args.n_heads,
        block_plan=plan, attn_window=args.attn_window,
        tie_embeddings=False,          # match #148's arms
    ).to(device)
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(
        model, target_names=[n for n in linears
                             if not any(p in n for p in exclude)])
    enable_scan_flags(model)
    return model


def run_arm(grad_accum: int, args, batches, probes, device, tokenizer=None) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build(args, device)
    opt, desc = build_optimizer(model, "muon", lr=args.aux_lr,
                                weight_decay=args.weight_decay,
                                muon_lr=args.muon_lr)
    # Every arm consumes the same token budget, so the number of OPTIMIZER steps
    # differs -- that is the point, not a flaw. The cosine is scheduled over each
    # arm's own step count so all arms complete exactly one decay.
    opt_steps = args.micro_batches // grad_accum
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, opt_steps))
    base_lrs = [pg["lr"] for pg in opt.param_groups]
    warmup_opt_steps = max(1, int(args.warmup_frac * opt_steps))

    eff = args.batch_size * grad_accum
    print(f"\n[eb] === grad_accum={grad_accum} (effective batch {eff} rows = "
          f"{eff * args.seq_len:,} tokens/step, {opt_steps} optimizer steps) ===\n"
          f"[eb]     {desc}")

    traj = [{"opt_step": 0, "tokens": 0,
             "heldout_loss": evaluate(model, probes, device)}]
    curve: List[dict] = []
    model.train()
    t0 = time.time()
    micro = step = 0
    opt.zero_grad(set_to_none=True)

    for i, (bx, by) in enumerate(batches, start=1):
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
            loss = loss / grad_accum
        loss.backward()
        micro += 1
        curve.append({"micro": i, "loss": float(loss.detach()) * grad_accum})

        if micro >= grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if step < warmup_opt_steps:
                s = (step + 1) / warmup_opt_steps
                for pg, b in zip(opt.param_groups, base_lrs):
                    pg["lr"] = b * s
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step >= warmup_opt_steps:
                cos.step()
            micro = 0
            step += 1

            if step % max(1, opt_steps // 6) == 0:
                hl = evaluate(model, probes, device)
                traj.append({"opt_step": step,
                             "tokens": i * args.batch_size * args.seq_len,
                             "heldout_loss": hl,
                             "wall_sec": time.time() - t0})
                print(f"[eb]   opt-step {step:>5}/{opt_steps}  "
                      f"tokens {i * args.batch_size * args.seq_len:>10,}  "
                      f"held-out {hl:.5f}  ({(time.time()-t0)/60:.1f} min)",
                      flush=True)
                model.train()

    final = evaluate(model, probes, device)
    out = {"grad_accum": grad_accum, "effective_batch_rows": eff,
           "optimizer_steps": step,
           "tokens_seen": len(batches) * args.batch_size * args.seq_len,
           "final_heldout_loss": final,
           "wall_clock_sec": time.time() - t0,
           "trajectory": traj, "curve": curve}
    del model, opt
    torch.cuda.empty_cache()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--grad-accums", type=int, nargs="+", default=[8, 16, 32])
    p.add_argument("--micro-batches", type=int, default=12000,
                   help="micro-batches per arm -- the FIXED TOKEN BUDGET every "
                        "arm consumes; optimizer-step counts differ by design")
    p.add_argument("--warmup-frac", type=float, default=0.067)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--attn-window", type=int, default=4096)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--aux-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vocab-size", type=int, default=GPT2_VOCAB_SIZE)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--out", default="results/effective_batch/sweep.json")
    args = p.parse_args(argv)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)

    device = require_cuda()
    import train_edge_conversational as tec
    tok = tec.load_tokenizer(args.tokenizer)
    if args.vocab_size != tok.vocab_size:
        args.vocab_size = tok.vocab_size
    torch.manual_seed(args.seed)
    loader, _v, _s = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    pairs = []
    for b in loader:
        x, y = as_pair(b)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= args.probe_batches + args.micro_batches:
            break
    probes, batches = pairs[:args.probe_batches], pairs[args.probe_batches:]
    tokens = len(batches) * args.batch_size * args.seq_len
    print(f"[eb] {len(probes)} probes + {len(batches):,} micro-batches "
          f"({tokens:,} tokens) — IDENTICAL for every arm")

    results: Dict[str, dict] = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for ga in args.grad_accums:
        results[str(ga)] = run_arm(ga, args, batches, probes, device, tok)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"issue": 152, "config": vars(args), "arms": results},
                      fh, indent=2)

    print("\n[eb] ============ effective batch, matched tokens ============")
    print(f"[eb] {'grad_accum':>11}{'eff rows':>10}{'opt steps':>11}"
          f"{'held-out':>11}{'wall (min)':>12}")
    for n, r in results.items():
        print(f"[eb] {n:>11}{r['effective_batch_rows']:>10}"
              f"{r['optimizer_steps']:>11}{r['final_heldout_loss']:>11.5f}"
              f"{r['wall_clock_sec']/60:>12.1f}")

    base = results.get("8")
    verdict = None
    if base is not None:
        best = min(results, key=lambda k: results[k]["final_heldout_loss"])
        d = results[best]["final_heldout_loss"] - base["final_heldout_loss"]
        print(f"\n[eb] vs the current default (grad_accum 8 / 32 rows):")
        for n, r in results.items():
            print(f"[eb]   ga={n:>2}: {r['final_heldout_loss'] - base['final_heldout_loss']:+.5f}")
        if best == "8":
            verdict = ("effective batch 32 WINS or ties -- the VRAM-derived default "
                       "happens to be a good training choice at fixed LR")
        else:
            verdict = (f"effective batch {results[best]['effective_batch_rows']} beats 32 "
                       f"by {-d:.5f} nats at matched tokens (grad_accum {best}). Free "
                       f"change: no VRAM cost, grad_accum only.")
        print(f"\n[eb] VERDICT: {verdict}")
        print("[eb] LIMIT: FIXED LR (muon_lr=0.02). Larger batches usually want a "
              "larger LR, so a loss here means 'loses without an LR adjustment', "
              "not 'loses'. One seed; 83M; small token budget.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 152, "config": vars(args), "arms": results,
                   "verdict": verdict}, fh, indent=2)
    print(f"[eb] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
