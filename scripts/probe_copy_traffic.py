"""What is `aten::copy_` doing in 22% of the 1B step, and is any of it waste?

Author: Rambarun Komaljeet

The finding this chases
-----------------------
`scripts/attribute_1b_step.py` found **`aten::copy_` is the single largest
operation in the 1B training step** -- 0.822 s of 3.68 s, larger than any single
GEMM. It is not model arithmetic. It is bytes being moved from one place to
another.

Crucially it is **not** the PCIe offload traffic: `Memcpy HtoD` is counted
separately at 0.40 s/step. So this 0.82 s is device-to-device copying that
happens entirely on the card, and nobody has ever looked at what it is.

The three candidates, which the SHAPE distinguishes
----------------------------------------------------
* **autocast casts** -- fp32 weights cast to bf16 on every use. WEIGHT-shaped
  (2-D, `[out, in]`, no sequence dimension).
* **the BitNet STE** -- `ternary_quantize_ste` builds a new tensor per weight per
  forward (#131 measured this chain at 395 ms of decode against a 66 ms GEMM).
  Also WEIGHT-shaped.
* **activation handling** -- grad-checkpoint saves, contiguity, gradient
  accumulation. ACTIVATION-shaped (carries the 4096 sequence dimension).

Weight-shaped copies scale with *parameter count* and are paid once per forward
regardless of batch, so at batch 1 they amortise over the fewest possible tokens.
Activation-shaped copies scale with batch x seq. Which one dominates decides
whether there is anything to reclaim, and how.

Falsifiability check, asked before running
--------------------------------------------
*"If none of this were waste, would this show it?"* Yes. If the copies are
activation-shaped and roughly one per layer, they are the price of gradient
checkpointing and there is nothing to remove -- that is a real possible outcome
and it ends the line. The reading only claims waste if copies are weight-shaped
AND repeated more often than the weights change.

Shapes come from the profiler's own records, not inferred. The sanity check is
that the summed `copy_` device time here matches what the attribution run
measured; if it does not, the two harnesses are not looking at the same thing.

Usage::

    python scripts/probe_copy_traffic.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN                              # noqa: E402
from hvn.edge.chunked_loss import chunked_cross_entropy         # noqa: E402
from hvn.optimizers import build_optimizer                      # noqa: E402
from scripts.ab_prodigy import enable_scan_flags                # noqa: E402


def plan(n):
    attn = (max(1, n // 4), max(2, (3 * n) // 4))
    return ["transformer" if i in attn else "deltanet" for i in range(n)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=2048)
    p.add_argument("--n-heads", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--steps", type=int, default=1)
    p.add_argument("--out", default="results/roofline/copy_traffic.json")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md)")

    model = EdgeHVN(vocab_size=args.vocab_size, d_model=args.d_model,
                    num_layers=args.num_layers, n_heads=args.n_heads,
                    block_plan=plan(args.num_layers), attn_window=4096,
                    tie_embeddings=False, grad_checkpoint=True).to("cpu")
    enable_scan_flags(model)
    opt, _ = build_optimizer(model, "muon", lr=2e-5, weight_decay=0.01,
                             muon_lr=0.02)
    from hvn.core.block_coordinate import BlockCoordinateScheduler
    sched = BlockCoordinateScheduler(
        model, opt, steps_per_block=50, device="cuda",
        offload_inactive_weights=True, offload_park_dtype=torch.bfloat16,
        offload_prefetch=True)
    params = [q for q in model.parameters() if q.requires_grad]
    x = torch.randint(0, args.vocab_size, (args.batch, args.seq), device="cuda")
    y = torch.randint(0, args.vocab_size, (args.batch, args.seq), device="cuda")

    def step():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            h = model(x, return_hidden=True)
            h = h[0] if isinstance(h, tuple) else h
            loss = chunked_cross_entropy(h, y, model.decoder, chunk=512).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); opt.zero_grad(set_to_none=True); sched.step()

    for _ in range(2):
        step()
    torch.cuda.synchronize()

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=True) as prof:
        for _ in range(args.steps):
            step()
        torch.cuda.synchronize()

    seq = args.seq
    buckets = {"weight-shaped": 0.0, "activation-shaped": 0.0, "small/other": 0.0}
    rows = []
    total = 0.0
    for e in prof.key_averages(group_by_input_shape=True):
        if e.key != "aten::copy_":
            continue
        dev = getattr(e, "self_device_time_total", 0) or 0
        if dev <= 0:
            continue
        shp = e.input_shapes[0] if e.input_shapes else []
        numel = 1
        for s_ in shp:
            numel *= s_
        if seq in shp or (args.batch * seq) in shp:
            cat = "activation-shaped"
        elif len(shp) == 2 and numel >= 1 << 16:
            cat = "weight-shaped"
        else:
            cat = "small/other"
        buckets[cat] += dev
        total += dev
        rows.append({"shape": shp, "count": int(e.count),
                     "ms": dev / args.steps / 1e3, "category": cat})

    rows.sort(key=lambda r: -r["ms"])
    print(f"\n[copy] aten::copy_ total {total/args.steps/1e6:.3f} s/step "
          f"(attribution run measured 0.822 s -- these should agree)\n")
    print(f"[copy] {'shape':<34}{'calls':>8}{'ms/step':>10}{'share':>8}  bucket")
    for r in rows[:16]:
        print(f"[copy] {str(r['shape'])[:33]:<34}{r['count']:>8}"
              f"{r['ms']:>10.1f}{100*r['ms']*1e3/(total/args.steps):>7.1f}%"
              f"  {r['category']}")

    print(f"\n[copy] {'bucket':<22}{'s/step':>10}{'share':>9}")
    for b, v in sorted(buckets.items(), key=lambda kv: -kv[1]):
        print(f"[copy] {b:<22}{v/args.steps/1e6:>10.3f}"
              f"{100*v/max(total,1):>8.1f}%")

    print(f"\n[copy] ================== READING ==================")
    w = buckets["weight-shaped"] / max(total, 1)
    a = buckets["activation-shaped"] / max(total, 1)
    if w > 0.5:
        print(f"[copy] {w*100:.0f}% is WEIGHT-shaped -- casts and/or the BitNet "
              f"STE rebuilding tensors per forward. This cost scales with "
              f"PARAMETER count, is paid once per forward regardless of batch, "
              f"and so amortises worst at batch 1. Candidates: cache the bf16 "
              f"cast, or the ternary result, across the forward+recompute pair "
              f"(the grad-checkpoint recompute redoes it a second time).")
        print(f"[copy] CAVEAT: #131 cached exactly this for INFERENCE and found "
              f"training cannot reuse it -- the STE needs the live shadow weight "
              f"in the autograd graph and the weight moves every step. The "
              f"forward/recompute pair WITHIN one step is the untested case.")
    elif a > 0.5:
        print(f"[copy] {a*100:.0f}% is ACTIVATION-shaped -- gradient "
              f"checkpointing saves and contiguity. That is the price of fitting "
              f"1B on the card, not waste; removing checkpointing was measured "
              f"at 178% of the card and 5.6x slower.")
    else:
        print(f"[copy] no single bucket dominates "
              f"(weight {w*100:.0f}%, activation {a*100:.0f}%) -- spread across "
              f"many small copies, so there is no single fix.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"question": "what is aten::copy_ doing",
                   "config": vars(args),
                   "copy_total_s_per_step": total / args.steps / 1e6,
                   "buckets": {k: v / args.steps / 1e6 for k, v in buckets.items()},
                   "rows": rows[:40]}, fh, indent=2)
    print(f"[copy] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
