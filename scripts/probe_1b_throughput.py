"""What does a 1B-class model actually cost per step on this 6 GB card?

Answers two questions the 1B-on-6GB goal depends on, at the shape real training
uses (seq 4096, grad-checkpoint, #158 chunked loss):

1. **Does it fit, and how fast?** Reported against the card's real 5.9966 GB.
2. **What does going smaller buy?** The size/throughput trade, measured rather
   than extrapolated from a scaling rule.

Why seq 4096 and not something quicker
--------------------------------------
Per-step overhead amortises over tokens, so a short-sequence measurement is not
a throughput measurement. #157 recorded the same config at 313 tok/s (seq 128)
and 875 tok/s (seq 4096). The chunked loss compounds this: the logits tensor is
``[B, S, vocab]``, linear in sequence length -- 98 MB at seq 512 against 785 MB
at seq 4096 -- which is exactly how #158's "chunking does not help at 1B" line
came to be wrong (retracted 2026-08-27, see docs/state.md).

The spill rule
--------------
On Windows/WDDM, exceeding VRAM does not raise OOM -- the driver spills to host
RAM and steps merely get slower, so "it ran" is not "it fits". Anything above
85% of the card is reported SUSPECT rather than passing, per the #136 rule.

Measured 2026-08-27 on an idle RTX 4050:

  arm            params        s/step   tok/s      GB   %card
  dense_1b    1,214,543,633     4.04     1014    4.38     73%  FITS
  dense_478m    477,997,043     2.14     1911    2.64     44%  FITS

i.e. a 478M model is 1.88x the throughput of 1.21B at 60% of the memory. At
1,014 tok/s a 21.4B-token budget is ~244 days; at 1,911 tok/s a 9.6B-token
budget (30 tok/param, #159) is ~58 days. Size, not compression, is the lever.

Usage::

    python scripts/probe_1b_throughput.py
    python scripts/probe_1b_throughput.py --arms dense_1b
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN                       # noqa: E402
from hvn.edge.chunked_loss import chunked_cross_entropy  # noqa: E402
from hvn.optimizers import build_optimizer               # noqa: E402
from scripts.ab_prodigy import enable_scan_flags         # noqa: E402

CARD_GB = 5.9966

#: name -> (d_model, n_heads, offload). d_head is 64 everywhere: the Triton
#: scan requires d_model/n_heads to be a power of 2 or it crashes (#136).
ARMS = {
    "dense_1b":   (2048, 32, True),
    "dense_478m": (1216, 19, True),
    "dense_355m": (1024, 16, True),
}


def plan(num_layers: int):
    p = ["deltanet"] * num_layers
    p[3] = "transformer"
    p[10] = "transformer"
    return p


def run(name: str, args) -> dict:
    d, h, offload = ARMS[name]
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    print(f"\n=== {name} (d={d}, {args.num_layers}L, h={h}, "
          f"offload={'ON' if offload else 'OFF'}) ===", flush=True)

    model = EdgeHVN(
        vocab_size=args.vocab_size, d_model=d, num_layers=args.num_layers,
        n_heads=h, block_plan=plan(args.num_layers), attn_window=4096,
        tie_embeddings=not offload,  # #151: offload is incompatible with tying
        grad_checkpoint=True,
    )
    n = sum(p.numel() for p in model.parameters())
    print(f"  params: {n:,}")

    sched = None
    if offload:
        # #102: do NOT blanket-.to(device) under offload -- nn.Module.to()
        # needs the whole model transiently resident, which is what OOM'd at
        # 3.39B. Leave it CPU-side and let the scheduler own placement.
        model = model.to("cpu")
    else:
        model = model.cuda()
    enable_scan_flags(model)

    opt, desc = build_optimizer(model, "muon", lr=2e-5,
                                weight_decay=0.01, muon_lr=0.02)
    if offload:
        from hvn.core.block_coordinate import BlockCoordinateScheduler
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=args.bcd_steps, device="cuda",
            offload_inactive_weights=True, offload_park_dtype=torch.bfloat16,
            offload_prefetch=True,
        )

    params = [p for p in model.parameters() if p.requires_grad]
    x = torch.randint(0, args.vocab_size, (args.batch, args.seq), device="cuda")
    y = torch.randint(0, args.vocab_size, (args.batch, args.seq), device="cuda")

    times = []
    for i in range(args.warmup + args.iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(x, return_hidden=True)
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
            # Returns [B*S] per-token losses (matches reduction="none").
            loss = chunked_cross_entropy(hidden, y, model.decoder,
                                         chunk=args.loss_chunk).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        if sched is not None:
            sched.step()
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        if i >= args.warmup:
            times.append(dt)
        print(f"  iter {i}: {dt:.2f}s  peak "
              f"{torch.cuda.max_memory_reserved()/2**30:.2f} GB", flush=True)

    times.sort()
    med = times[len(times) // 2]
    reserved = torch.cuda.max_memory_reserved() / 2**30
    pct = 100.0 * reserved / CARD_GB
    verdict = "FITS" if pct <= 85 else ("SUSPECT" if pct <= 100 else "SPILLS")
    tok_s = args.batch * args.seq / med
    print(f"  --> {med:.2f} s/step, {tok_s:.0f} tok/s, "
          f"{reserved:.2f} GB ({pct:.0f}% of card) [{verdict}]")

    out = dict(name=name, d_model=d, n_heads=h, params=n, offload=offload,
               sec_per_step=med, tok_per_sec=tok_s,
               peak_alloc_gb=torch.cuda.max_memory_allocated() / 2**30,
               peak_reserved_gb=reserved, pct_of_card=pct, verdict=verdict,
               optimizer=desc)
    del model, opt, sched
    torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["dense_1b", "dense_478m"])
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--loss-chunk", type=int, default=512)
    p.add_argument("--bcd-steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=3)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--tokens-per-param", type=float, default=30.0,
                   help="#159 measured ~30 for this architecture, not Chinchilla's 20")
    p.add_argument("--out", default="results/throughput_1b.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required -- this measures GPU throughput.")
        return 2
    print(f"card: {torch.cuda.get_device_name(0)} "
          f"({torch.cuda.get_device_properties(0).total_memory/2**30:.4f} GB)")
    print(f"config: {args.num_layers}L, seq {args.seq}, batch {args.batch}, "
          f"grad-checkpoint, chunked loss (chunk {args.loss_chunk})")

    results = {}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    for arm in args.arms:
        try:
            results[arm] = run(arm, args)
        except Exception as e:                       # noqa: BLE001
            print(f"  !! {arm} FAILED: {type(e).__name__}: {str(e)[:200]}")
            results[arm] = dict(name=arm, failed=f"{type(e).__name__}: {e}"[:400])
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    print("\n" + "=" * 76)
    print(f"{'arm':<14}{'params':>15}{'s/step':>9}{'tok/s':>9}{'GB':>7}"
          f"{'%card':>7}  {'verdict':<9}{'days':>7}")
    for r in results.values():
        if r.get("failed"):
            print(f"{r['name']:<14}{'FAILED':>15}  {r['failed'][:40]}")
            continue
        budget = r["params"] * args.tokens_per_param
        days = budget / r["tok_per_sec"] / 86400
        print(f"{r['name']:<14}{r['params']:>15,}{r['sec_per_step']:>9.2f}"
              f"{r['tok_per_sec']:>9.0f}{r['peak_reserved_gb']:>7.2f}"
              f"{r['pct_of_card']:>6.0f}%  {r['verdict']:<9}{days:>7.0f}")
    print(f"\n'days' = params x {args.tokens_per_param:g} tokens/param (#159) at the "
          f"measured throughput. It is a floor: no dataloader in this loop, and a "
          f"laptop that throttles.")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
