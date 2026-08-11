"""Does BCD's optimizer-state amnesia cause its 4x quality cost? (Issue #151)

Author: Rambarun Komaljeet

#112 measured block-coordinate descent at ~3.9x worse perplexity (4,200 steps:
full 4.83 vs BCD 6.19) and named a mechanism it never tested: *"only 1 of 14
blocks trains at a time with the outgoing block's optimizer state deleted on
every switch"*. Each block therefore restarts cold every rotation.

**Why the obvious fix is dead on arrival.** Heavy-ball momentum over k steps of
zero gradient decays to ``beta^k * m``. Muon runs beta=0.95 and a block is away
~1,300 steps, so ``0.95^1300 ~= 1e-29``: a *correctly decayed* restore is
arithmetically identical to deletion. Today's behaviour is already right for
heavy-ball momentum. The only variant with content treats parked time as
**paused** rather than as zero-gradient time -- that is what
``retain_optimizer_state=True`` does, and it is the hypothesis here.

Arms: ``full`` (no BCD) / ``bcd_delete`` (today) / ``bcd_retain``.

``bcd_delete`` reproducing #112's gap is the **known-answer cell** -- if it does
not, this harness is not measuring the thing #112 measured and the comparison is
void. 4,200 steps is not arbitrary: #112 established the first rotation (to step
1,400) is pure warmup, so anything shorter grades warmup instead of steady state.

Falsifiable both ways:
  bcd_retain ~= full         -> amnesia was the whole story; BCD becomes usable
  bcd_retain ~= bcd_delete   -> the 4x is UNDER-TRAINING (each block trains 1/14
                                of the time) and no state trick fixes it

Usage::

    python scripts/ab_bcd_retain_state.py --steps 4200
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
from hvn.core.block_coordinate import BlockCoordinateScheduler     # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model                # noqa: E402
from hvn.optimizers import build_optimizer                         # noqa: E402

GPT2_VOCAB_SIZE = 50257
#: #112, same step budget and rotation count. Cross-run reference only -- the
#: in-run known-answer is bcd_delete reproducing the GAP, not the absolute value.
REF_FULL_112, REF_BCD_112 = 4.83, 6.19

ARMS = {"full": None, "bcd_delete": False, "bcd_retain": True}


def build(args, device):
    from hvn.edge.model import EdgeHVN
    plan = ["deltanet"] * 4 + ["transformer"] + ["deltanet"]
    plan = plan + plan
    model = EdgeHVN(
        vocab_size=args.vocab_size, d_model=args.d_model,
        num_layers=len(plan), n_heads=args.n_heads,
        block_plan=plan, attn_window=args.attn_window,
        # #112's A/B ran untied (it predates #139). Held untied here so the BCD
        # comparison is not entangled with the tying change.
        tie_embeddings=False,
    ).to(device)
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(
        model, target_names=[n for n in linears
                             if not any(p in n for p in exclude)])
    enable_scan_flags(model)
    return model


def run_arm(arm: str, args, batches, probes, device, tokenizer=None) -> dict:
    retain = ARMS[arm]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = build(args, device)
    opt, desc = build_optimizer(model, "muon", lr=args.aux_lr,
                                weight_decay=args.weight_decay,
                                muon_lr=args.muon_lr)
    sched = None
    if retain is not None:
        sched = BlockCoordinateScheduler(
            model, opt, steps_per_block=args.steps_per_block,
            retain_optimizer_state=retain)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    n_blocks = len(sched.blocks) if sched else 0
    print(f"\n[ab] === arm '{arm}' "
          f"({'no BCD' if sched is None else f'BCD {n_blocks} blocks x '
             f'{args.steps_per_block} steps, retain={retain}'}) ===\n"
          f"[ab]     {desc}")
    if sched is not None:
        rot = n_blocks * args.steps_per_block
        print(f"[ab]     rotation = {rot} steps; a block is away {rot - args.steps_per_block} "
              f"steps -> honest decay 0.95^{rot - args.steps_per_block} ~ "
              f"{0.95 ** (rot - args.steps_per_block):.2e} (i.e. deletion)")

    traj = [{"step": 0, "heldout_loss": evaluate(model, probes, device)}]
    curve: List[dict] = []
    model.train()
    t0 = time.time()
    warmup_done = args.warmup_steps <= 0

    for i, (bx, by) in enumerate(batches, start=1):
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], 1.0)
        if not warmup_done:
            s = (i - 1) / max(1, args.warmup_steps)
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = b * s
            if i >= args.warmup_steps:
                warmup_done = True
        opt.step()
        opt.zero_grad(set_to_none=True)
        if warmup_done:
            cos.step()
        if sched is not None:
            sched.step()
        curve.append({"step": i, "loss": float(loss.detach())})

        if i % args.eval_every == 0:
            hl = evaluate(model, probes, device)
            traj.append({"step": i, "heldout_loss": hl,
                         "wall_sec": time.time() - t0})
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            model.train()

    out = {"arm": arm, "retain": retain,
           "final_heldout_loss": traj[-1]["heldout_loss"],
           "final_ppl": float(torch.exp(torch.tensor(traj[-1]["heldout_loss"]))),
           "wall_clock_sec": time.time() - t0,
           "parked_tensors": len(getattr(sched, "_parked_state", {}) or {}),
           "trajectory": traj, "curve": curve}
    del model, opt, sched
    torch.cuda.empty_cache()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--steps", type=int, default=4200)
    p.add_argument("--steps-per-block", type=int, default=100)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=350)
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
    p.add_argument("--out", default="results/bcd_retain/ab.json")
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
    loader, _v, _stats = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    pairs = []
    for b in loader:
        x, y = as_pair(b)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= args.probe_batches + args.steps:
            break
    probes, batches = pairs[:args.probe_batches], pairs[args.probe_batches:]
    print(f"[ab] {len(probes)} probes + {len(batches)} identical batches per arm")

    results: Dict[str, dict] = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for arm in args.arms:
        results[arm] = run_arm(arm, args, batches, probes, device, tok)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"issue": 151, "config": vars(args), "arms": results},
                      fh, indent=2)

    print("\n[ab] ============== BCD state retention ==============")
    print(f"[ab] {'arm':<14}{'held-out':>11}{'ppl':>10}{'wall (min)':>12}")
    for n, r in results.items():
        print(f"[ab] {n:<14}{r['final_heldout_loss']:>11.5f}"
              f"{r['final_ppl']:>10.1f}{r['wall_clock_sec']/60:>12.1f}")

    verdict = None
    if {"full", "bcd_delete"} <= set(results):
        f, d = (results["full"]["final_heldout_loss"],
                results["bcd_delete"]["final_heldout_loss"])
        gap = d - f
        ratio = results["bcd_delete"]["final_ppl"] / results["full"]["final_ppl"]
        print(f"\n[ab] KNOWN-ANSWER — BCD's cost reproduced: {gap:+.3f} nats "
              f"({ratio:.2f}x ppl). #112 measured +1.36 nats (3.9x).")
        if gap < 0.5:
            verdict = (f"VOID: this harness did not reproduce BCD's known cost "
                       f"(+{gap:.3f} nats vs #112's +1.36). It is not measuring "
                       f"what #112 measured; do NOT report a retention verdict.")
            print(f"[ab] *** {verdict}")

    if verdict is None and {"bcd_retain", "bcd_delete"} <= set(results) \
            and "full" not in results:
        d = results["bcd_delete"]["final_heldout_loss"]
        r = results["bcd_retain"]["final_heldout_loss"]
        print(f"\n[ab] retain vs delete: {r - d:+.5f} nats")
        print("[ab] (no `full` arm -> cannot say what FRACTION of BCD's gap this "
              "closes, which is the actual question. Partial run only.)")
        verdict = f"partial: retain vs delete {r - d:+.5f} nats, no full-model baseline"

    if verdict is None and "bcd_retain" in results:
        f, d, r = (results["full"]["final_heldout_loss"],
                   results["bcd_delete"]["final_heldout_loss"],
                   results["bcd_retain"]["final_heldout_loss"])
        closed = (d - r) / (d - f) * 100 if (d - f) else float("nan")
        print(f"\n[ab] retain vs delete: {r - d:+.5f} nats")
        print(f"[ab] retain vs full:   {r - f:+.5f} nats")
        print(f"[ab] fraction of BCD's gap closed by retention: {closed:.1f}%")
        if closed > 50:
            verdict = (f"AMNESIA WAS THE CAUSE: retention closes {closed:.0f}% of "
                       f"BCD's quality gap. BCD becomes usable; the CPU-paging "
                       f"work (and two-resident prefetch) is justified.")
        elif closed > 10:
            verdict = (f"PARTIAL: retention closes {closed:.0f}% of the gap. Real "
                       f"but not the main cause -- most of the 4x is under-training.")
        else:
            verdict = (f"UNDER-TRAINING, NOT AMNESIA: retention closes only "
                       f"{closed:.0f}% of the gap. Each block training 1/14 of the "
                       f"time is the cost, and no optimizer-state trick fixes it. "
                       f"The lever closes.")
        print(f"\n[ab] VERDICT: {verdict}")
        print("[ab] LIMIT: one seed; 12 blocks at this size vs #112's 14; "
              "undecayed restore only -- a rotation-matched beta is untested.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 151, "config": vars(args), "arms": results,
                   "reference_112": {"full": REF_FULL_112, "bcd": REF_BCD_112},
                   "verdict": verdict}, fh, indent=2)
    print(f"[ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
