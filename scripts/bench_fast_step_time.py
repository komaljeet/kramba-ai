"""What does --fast actually cost and buy, per step? (Issue #155)

Author: Rambarun Komaljeet

Every attempt to measure this through `train_edge_conversational.py` produced
SILENCE rather than a number, and the silence was misread three different ways
in one evening: an output pipeline that buffered until process exit, orphaned
processes competing for the card, and -- the likely one -- `log_every_steps`
being hardcoded to 50 while every smoke asked for 3 to 20 steps, so the script
legitimately printed nothing until it finished.

This bypasses all of that. It builds the same model, compiles it the way
`--fast` does, and prints after EVERY step with no buffering and no hardcoded
log interval. The numbers wanted:

* compile cost -- step 1 wall time minus steady state
* steady-state step time and tokens/sec, which sets what 24 hours buys
* the same for --no-fast, so the comparison is measured rather than assumed

Runs both arms in ONE process, fast first, so a torch.compile cache warmed by
the fast arm cannot flatter the eager arm.

Usage::

    python scripts/bench_fast_step_time.py --seq-len 4096 --steps 5
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

from hvn.edge.model import EdgeHVN                        # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model       # noqa: E402
from hvn.optimizers import build_optimizer                # noqa: E402


def build(args, device):
    plan = ["transformer" if i in (4, 12) else "deltanet"
            for i in range(args.num_layers)]
    m = EdgeHVN(vocab_size=args.vocab_size, d_model=args.d_model,
                num_layers=args.num_layers, n_heads=args.n_heads,
                block_plan=plan, attn_window=4096, tie_embeddings=True,
                grad_checkpoint=args.grad_checkpoint).to(device)
    lin = {n for n, x in m.named_modules() if isinstance(x, nn.Linear)}
    ex = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(m, target_names=[n for n in lin
                                            if not any(p in n for p in ex)])
    for b in m.blocks:
        if hasattr(b, "use_triton_scan"):
            b.use_triton_scan = True
            b.use_megakernel_scan = True
    return m


def run(arm: str, compiled: bool, args, device) -> dict:
    torch.manual_seed(0)
    torch.backends.cuda.matmul.allow_tf32 = compiled
    torch.backends.cudnn.allow_tf32 = compiled
    model = build(args, device)
    train_model = torch.compile(model) if compiled else model
    opt, desc = build_optimizer(model, "muon", lr=3e-4, weight_decay=0.01,
                                muon_lr=args.muon_lr)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(1)
    print(f"\n[{arm}] {desc}")
    print(f"[{arm}] {'step':>5}{'wall (s)':>11}{'tok/s':>10}")

    per_step = []
    for s in range(1, args.steps + 1):
        t0 = time.perf_counter()
        for _ in range(args.grad_accum):
            x = torch.randint(0, args.vocab_size,
                              (args.batch_size, args.seq_len), generator=g).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=compiled):
                logits = train_model(x)
                loss = crit(logits.reshape(-1, logits.size(-1)),
                            torch.roll(x, -1, 1).reshape(-1)) / args.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        toks = args.batch_size * args.seq_len * args.grad_accum
        per_step.append(dt)
        print(f"[{arm}] {s:>5}{dt:>11.2f}{toks/dt:>10.0f}", flush=True)

    steady = sorted(per_step[1:])[len(per_step[1:]) // 2] if len(per_step) > 1 else per_step[0]
    toks = args.batch_size * args.seq_len * args.grad_accum
    out = {"arm": arm, "compiled": compiled, "per_step_s": per_step,
           "first_step_s": per_step[0], "steady_step_s": steady,
           "startup_cost_s": per_step[0] - steady,
           "tokens_per_step": toks, "steady_tok_s": toks / steady,
           "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20}
    print(f"[{arm}] first step {out['first_step_s']:.1f}s | steady {steady:.2f}s "
          f"| {out['steady_tok_s']:.0f} tok/s | startup "
          f"{out['startup_cost_s']:.1f}s | peak {out['peak_reserved_mib']:.0f} MiB")
    del model, train_model, opt
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--d-model", type=int, default=896)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=14)
    p.add_argument("--vocab-size", type=int, default=32003)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--out", default="results/fast_step_time/bench.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    print(f"157M hybrid | seq {args.seq_len} x batch {args.batch_size} x "
          f"accum {args.grad_accum} = {args.batch_size*args.seq_len*args.grad_accum:,} "
          f"tokens/step | grad_checkpoint={args.grad_checkpoint}")

    # fast FIRST: a compile cache warmed here cannot flatter the eager arm.
    res = {"fast": run("fast", True, args, device),
           "no-fast": run("no-fast", False, args, device)}

    f, n = res["fast"], res["no-fast"]
    speedup = n["steady_step_s"] / f["steady_step_s"]
    print(f"\n{'':>10}{'steady s/step':>15}{'tok/s':>10}{'first step':>12}")
    for k, r in res.items():
        print(f"{k:>10}{r['steady_step_s']:>15.2f}{r['steady_tok_s']:>10.0f}"
              f"{r['first_step_s']:>12.1f}")
    print(f"\n--fast is {speedup:.2f}x faster in steady state; its first step "
          f"costs {f['startup_cost_s']:.1f}s extra (one-time compile).")
    hrs24 = f["steady_tok_s"] * 86400
    print(f"24 h at --fast ~= {hrs24/1e6:.0f}M tokens "
          f"({hrs24/3.14e9*100:.1f}% of the 3.14B corpus plan)")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 155, "config": vars(args), "results": res,
                   "fast_speedup": speedup,
                   "tokens_in_24h_fast": hrs24}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
