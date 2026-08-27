"""Does TT let a 1B-shaped model train WITHOUT CPU offload, and is that faster? (#161)

The claim under test, and it is a throughput claim, not a size one
-------------------------------------------------------------------
A dense 1B model does not fit on this 6 GB card. It trains only with
block-coordinate descent + CPU weight offload, which streams every block over
PCIe on every step -- docs/training.md prices that inefficiency at ~2x, and
#113 measured a 25% speedup purely from halving the bytes moved.

If TT shrinks a d=2048/24L model enough to sit in VRAM outright, the entire
PCIe tax disappears. That is a different and much stronger argument than
"TT makes the model smaller", and it attacks the term that actually blocks a
1B run: TIME (267 days at the offload path's 875 tok/s).

Arms
----
  dense_offload : d=2048 24L dense, BCD + CPU offload   (the incumbent)
  tt_native     : d=2048 24L TT r32, NO offload         (the claim)

Both at seq 4096 -- the length real training uses. Measuring at seq 512 would
be meaningless here: per-step offload overhead amortises over tokens, and
#157 recorded the same config reading 313 tok/s at seq 128 and 875 tok/s at
seq 4096. A short-sequence number is not a throughput number.

Both use gradient checkpointing and the #158 chunked loss, because at seq 4096
with a 50257 vocabulary the logits alone are ~823 MB and would dominate either
arm's peak.

Falsifiability, asked before running
------------------------------------
If TT did NOT remove the need for offload, would this catch it? Yes -- the
tt_native arm would spill or OOM, and peak reserved is reported against the
card's real 5.9966 GB. The repo's own rule is that anything above 85% of the
card is SUSPECT rather than "fits", because Windows/WDDM silently spills to
host RAM instead of erroring, so the verdict here checks the ratio and says
SUSPECT rather than reporting a clean pass.

These are DIFFERENT MODELS -- 1.11B dense against 396M TT. This measures
throughput and fit only. It says nothing about which is the better model, and
the #161 A/B found TT at matched parameters to be no better than dense.

Usage: python scripts/probe_tt_1b_no_offload.py
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

from hvn.edge.model import EdgeHVN                       # noqa: E402
from hvn.edge.chunked_loss import chunked_cross_entropy  # noqa: E402
from hvn.optimizers import build_optimizer               # noqa: E402
from scripts.ab_prodigy import enable_scan_flags         # noqa: E402

CARD_GB = 5.9966


def plan24(num_layers: int = 24):
    p = ["deltanet"] * num_layers
    p[3] = "transformer"
    p[10] = "transformer"
    return p


def build(tt: bool, args, offload: bool, d: int | None = None, h: int | None = None):
    kw = {}
    if tt:
        kw = dict(tt_targets=("qkv", "out", "ffn"), tt_rank=args.tt_rank,
                  tt_contract="fused")
    model = EdgeHVN(
        vocab_size=args.vocab_size,
        d_model=d or args.d_model,
        num_layers=args.num_layers,
        n_heads=h or args.n_heads,
        block_plan=plan24(args.num_layers), attn_window=4096,
        tie_embeddings=not offload,  # #151: offload is incompatible with tying
        grad_checkpoint=True, **kw,
    )
    return model


def run(name: str, tt: bool, offload: bool, args,
        d: int | None = None, h: int | None = None) -> dict:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    dd, hh = d or args.d_model, h or args.n_heads
    print(f"\n=== {name} (d={dd}, {args.num_layers}L, "
          f"{'TT r%d' % args.tt_rank if tt else 'dense'}, "
          f"offload={'ON' if offload else 'OFF'}) ===", flush=True)

    model = build(tt, args, offload, dd, hh)
    n = sum(p.numel() for p in model.parameters())
    print(f"  params: {n:,}")

    sched = None
    if not offload:
        model = model.cuda()
    else:
        # #102's extended finding: with offload, do NOT blanket-.to(device) --
        # nn.Module.to() copies params one at a time and needs the WHOLE model
        # transiently resident, which is what OOM'd at 3.39B. Leave it CPU-side
        # and let the scheduler own all placement.
        model = model.to("cpu")
    enable_scan_flags(model)

    # The scheduler takes the optimizer, so it has to be built first -- it
    # clears the outgoing block's optimizer state on every switch.
    opt, desc = build_optimizer(model, "muon", lr=2e-5,
                                weight_decay=0.01, muon_lr=0.02)
    if offload:
        from hvn.core.block_coordinate import BlockCoordinateScheduler
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=args.bcd_steps, device="cuda",
            offload_inactive_weights=True, offload_park_dtype=torch.bfloat16,
            offload_prefetch=True,
        )
    print(f"  after placement: {torch.cuda.max_memory_reserved()/2**30:.2f} GB reserved",
          flush=True)

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
            # chunked_cross_entropy returns [B*S] per-token losses, matching
            # F.cross_entropy(reduction="none") -- reduce to a scalar here.
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
    alloc = torch.cuda.max_memory_allocated() / 2**30
    reserved = torch.cuda.max_memory_reserved() / 2**30
    tok_s = args.batch * args.seq / med
    pct = 100.0 * reserved / CARD_GB
    # The repo's own spill rule: >85% of the card is SUSPECT, not "fits",
    # because WDDM backs the overflow with host RAM instead of raising OOM.
    verdict = "FITS" if pct <= 85 else ("SUSPECT" if pct <= 100 else "SPILLS")

    print(f"  --> {med:.2f} s/step, {tok_s:.0f} tok/s, "
          f"{reserved:.2f} GB ({pct:.0f}% of card) [{verdict}]")

    out = dict(name=name, params=n, tt=tt, offload=offload, d_model=dd,
               sec_per_step=med, tok_per_sec=tok_s,
               peak_alloc_gb=alloc, peak_reserved_gb=reserved,
               pct_of_card=pct, verdict=verdict, optimizer=desc)
    del model, opt, sched
    torch.cuda.empty_cache()
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=2048)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--n-heads", type=int, default=32)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--tt-rank", type=int, default=32)
    p.add_argument("--loss-chunk", type=int, default=512)
    p.add_argument("--bcd-steps", type=int, default=50)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--iters", type=int, default=5)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--arms", nargs="+", default=["tt_native", "dense_offload"])
    p.add_argument("--out", default="results/tt/tt_1b_no_offload.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("CUDA required."); return 2
    print(f"card: {torch.cuda.get_device_name(0)}  "
          f"({torch.cuda.get_device_properties(0).total_memory/2**30:.4f} GB)")
    print(f"config: d={args.d_model}, {args.num_layers}L, seq {args.seq}, "
          f"batch {args.batch}, grad-checkpoint, chunked loss")

    # d/h override per arm, so the matched-parameter comparisons are explicit
    # rather than implied. d_head stays 64 everywhere (power of 2, or the
    # Triton scan crashes).
    specs = {
        # the incumbent: dense 1.21B, streamed from CPU
        "dense_offload":   dict(tt=False, offload=True),
        # TT resident on the card -- measured, spills at seq 4096
        "tt_native":       dict(tt=True,  offload=False),
        # TT + offload: 3.4x fewer params => 3.4x fewer bytes over PCIe.
        # This is the configuration the "avoid offload" argument actually
        # points at once you accept that PCIe traffic scales with PARAMETERS.
        "tt_offload":      dict(tt=True,  offload=True),
        # The control that decides whether TT adds anything: a DENSE model of
        # the same parameter count, also offloaded. It streams the same bytes
        # while being far narrower, so it should be cheaper per step. If TT
        # cannot beat this, TT is not the lever -- fewer parameters is.
        "dense_matched":   dict(tt=False, offload=True, d=1216, h=19),
        # A GENUINE ~1B-parameter TT model. Same bytes over PCIe as dense 1B,
        # but ~1.5x the width, so predicted slower with no transfer saving.
        "tt_1b":           dict(tt=True,  offload=True, d=3072, h=48),
    }
    results = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for arm in args.arms:
        try:
            sp = specs[arm]
            results[arm] = run(arm, sp["tt"], sp["offload"], args,
                               sp.get("d"), sp.get("h"))
        except Exception as e:                       # noqa: BLE001
            print(f"  !! {arm} FAILED: {type(e).__name__}: {str(e)[:200]}")
            results[arm] = dict(name=arm, failed=f"{type(e).__name__}: {e}"[:400])
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)

    print("\n" + "=" * 72)
    print(f"{'arm':<16}{'params':>15}{'s/step':>9}{'tok/s':>9}{'GB':>7}{'% card':>8}  verdict")
    for r in results.values():
        if r.get("failed"):
            print(f"{r['name']:<16}{'FAILED':>15}  {r['failed'][:40]}")
            continue
        print(f"{r['name']:<16}{r['params']:>15,}{r['sec_per_step']:>9.2f}"
              f"{r['tok_per_sec']:>9.0f}{r['peak_reserved_gb']:>7.2f}"
              f"{r['pct_of_card']:>7.0f}%  {r['verdict']}")

    a, b = results.get("tt_native"), results.get("dense_offload")
    if a and b and not a.get("failed") and not b.get("failed"):
        speedup = a["tok_per_sec"] / b["tok_per_sec"]
        print(f"\nTT-native vs dense-offload throughput: {speedup:.2f}x")
        days_a = 21.4e9 / a["tok_per_sec"] / 86400
        days_b = 21.4e9 / b["tok_per_sec"] / 86400
        print(f"days for a 21.4B-token budget:  tt_native {days_a:.0f}   "
              f"dense_offload {days_b:.0f}")
        if a["verdict"] != "FITS":
            print(f"CAVEAT: tt_native is {a['pct_of_card']:.0f}% of the card "
                  f"({a['verdict']}). Above 85% this repo treats a pass as "
                  f"unreliable -- WDDM spills silently rather than erroring, "
                  f"and a real run adds a dataloader on top of this.")
        print("\nNOTE: these are DIFFERENT MODELS (dense 1.11B vs TT 396M). "
              "This is a throughput-and-fit measurement, not a quality one. "
              "The #161 A/B found TT at matched parameters no better than dense.")

    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
