"""Fixed-LR sweep: was #147's restart win a schedule effect, or a mis-set LR? (Issue #148)

Author: Rambarun Komaljeet

#147 measured that restarting the cosine mid-run is worth **-0.091 nats** at 1,500
steps, free. Two explanations fit that equally well:

1. the schedule *shape* matters -- restarts genuinely help;
2. ``muon_lr=0.02`` is simply wrong for a run this short, and the restart
   compensated by keeping the LR high for longer.

#133 already ran this play: a dynamic phase-aware LR "won by 0.30 nats", and the
win turned out to be that the baseline LR was 10x too high -- the schedule was
compensating for a bad number. Same shape, so it gets checked the same way.

Every arm here uses a **plain single cosine, no restarts**; only ``--muon-lr``
differs. Reference lines from #147 at the same seed, batches and length:

    plain cosine @ 0.02          6.03552
    cosine WITH restart @ 0.02   5.94410   <- the number a fixed LR must reach

**Known-answer cell:** the ``0.02`` arm must reproduce 6.03552 exactly. Same seed,
same batches, same code path, only the flag re-specified. If it does not, this
sweep is not comparable to the result it exists to explain, and the run is void.

**Limit, up front:** the optimal LR depends on run length. This characterises
1,500-step runs and is *not* a recommendation for the 100K-step pretrain. What it
can settle is whether #147's restart finding is real or an artifact.

Usage::

    python scripts/sweep_muon_lr.py --lrs 0.005 0.01 0.02 0.04 0.08
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

#: #147, identical seed / batches / length. The bar a fixed LR has to clear.
#: These are transcribed from #147's printed output at FIVE decimals, so they
#: carry ~5e-6 of quantisation. The known-answer tolerance below must respect
#: that: comparing a full-precision measurement against a rounded literal with a
#: 1e-9 threshold is a test that can never pass -- the mirror of #140's test that
#: could never fail, and just as useless. (The first run of this sweep declared
#: itself VOID at a drift of 2.2e-7, i.e. an exact reproduction.)
REF_PLAIN_002 = 6.03552
REF_RESTART_002 = 5.94410
#: Half a unit in the last printed place, which is all the reference can resolve.
REF_TOL = 5e-6


def build(args, device):
    from hvn.edge.model import EdgeHVN
    plan = ["deltanet"] * 4 + ["transformer"] + ["deltanet"]
    plan = plan + plan                      # #147's 12-layer plan: attn at 4, 10
    model = EdgeHVN(
        vocab_size=args.vocab_size, d_model=args.d_model,
        num_layers=len(plan), n_heads=args.n_heads,
        block_plan=plan, attn_window=args.attn_window,
    ).to(device)
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(
        model, target_names=[n for n in linears
                             if not any(p in n for p in exclude)])
    enable_scan_flags(model)
    return model


def run_lr(lr: float, args, batches, probes, device, tokenizer=None) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model = build(args, device)
    opt, desc = build_optimizer(model, "muon", lr=args.aux_lr,
                                weight_decay=args.weight_decay, muon_lr=lr)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    print(f"\n[lr] === muon_lr={lr:g} ===\n[lr]     {desc}")
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
        curve.append({"step": i, "loss": float(loss.detach())})
        if i % args.eval_every == 0:
            hl = evaluate(model, probes, device)
            traj.append({"step": i, "heldout_loss": hl,
                         "wall_sec": time.time() - t0})
            recent = [c["loss"] for c in curve[-args.eval_every:]]
            print(f"[lr]   step {i:>5}  held-out {hl:.5f}  "
                  f"train {sum(recent)/len(recent):.5f}  "
                  f"({(time.time()-t0)/60:.1f} min)", flush=True)
            model.train()

    bpb = None
    if args.bpb_text and tokenizer is not None and os.path.exists(args.bpb_text):
        from eval.bits_per_byte import bits_per_byte
        with open(args.bpb_text, encoding="utf-8", errors="replace") as fh:
            text = fh.read(args.bpb_max_bytes)
        bpb = bits_per_byte(model, tokenizer, text, seq_len=args.seq_len,
                            device=device)["bits_per_byte"]
        print(f"[lr]   bits/byte {bpb:.4f}")

    # A diverged arm must be reported as diverged, not ranked as merely bad --
    # at high LR this is the expected failure and it should not be mistaken for
    # a smooth trend.
    final = traj[-1]["heldout_loss"]
    diverged = (not torch.isfinite(torch.tensor(final))) or final > traj[0]["heldout_loss"]

    out = {"muon_lr": lr, "final_heldout_loss": final, "bits_per_byte": bpb,
           "diverged": bool(diverged), "wall_clock_sec": time.time() - t0,
           "trajectory": traj, "curve": curve}
    del model, opt
    torch.cuda.empty_cache()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--lrs", type=float, nargs="+",
                   default=[0.005, 0.01, 0.02, 0.04, 0.08])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--n-heads", type=int, default=8)
    p.add_argument("--attn-window", type=int, default=4096)
    p.add_argument("--aux-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vocab-size", type=int, default=GPT2_VOCAB_SIZE)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--bpb-text", default="data_cache/heldout_fineweb.txt")
    p.add_argument("--bpb-max-bytes", type=int, default=2_000_000)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--out", default="results/lr_sweep/sweep.json")
    args = p.parse_args(argv)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        print("[lr] deterministic ON — the 0.02 arm must reproduce #147 exactly")

    device = require_cuda()
    import train_edge_conversational as tec
    tok = tec.load_tokenizer(args.tokenizer)
    if args.vocab_size != tok.vocab_size:
        args.vocab_size = tok.vocab_size
    torch.manual_seed(args.seed)
    loader, _v, stats = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    pairs = []
    for batch in loader:
        x, y = as_pair(batch)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= args.probe_batches + args.steps:
            break
    probes, batches = pairs[:args.probe_batches], pairs[args.probe_batches:]
    print(f"[lr] {len(probes)} probes + {len(batches)} identical batches per arm")

    results: Dict[str, dict] = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for lr in args.lrs:
        results[f"{lr:g}"] = run_lr(lr, args, batches, probes, device, tok)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"issue": 148, "config": vars(args), "arms": results},
                      fh, indent=2)

    print("\n[lr] ================= fixed-LR sweep =================")
    print(f"[lr] {'muon_lr':>10}{'held-out':>11}{'bits/byte':>11}{'vs restart':>13}")
    for name, r in results.items():
        b = r["bits_per_byte"]
        flag = "  DIVERGED" if r["diverged"] else ""
        print(f"[lr] {name:>10}{r['final_heldout_loss']:>11.5f}"
              f"{'nan' if b is None else f'{b:>11.4f}'}"
              f"{r['final_heldout_loss'] - REF_RESTART_002:>+13.5f}{flag}")
    print(f"[lr] {'#147 plain 0.02':>10} {REF_PLAIN_002:.5f}   (known-answer target)")
    print(f"[lr] {'#147 restart':>10} {REF_RESTART_002:.5f}   (the bar to clear)")

    verdict = None
    if "0.02" in results:
        drift = abs(results["0.02"]["final_heldout_loss"] - REF_PLAIN_002)
        print(f"\n[lr] KNOWN-ANSWER CELL — 0.02 arm vs #147's control: {drift:.3e}")
        if drift > REF_TOL:
            verdict = f"VOID (0.02 arm does not reproduce #147: drift {drift:.3e})"
            print("[lr] *** VOID: this sweep is not comparable to the result it "
                  "exists to explain. Do NOT report it.")

    if verdict is None:
        ok = {k: v for k, v in results.items() if not v["diverged"]}
        if ok:
            best = min(ok, key=lambda k: ok[k]["final_heldout_loss"])
            bl = ok[best]["final_heldout_loss"]
            if bl <= REF_RESTART_002:
                verdict = (f"MIS-SET LR, not a schedule effect: a plain cosine at "
                           f"muon_lr={best} reaches {bl:.5f}, matching or beating the "
                           f"restart's {REF_RESTART_002:.5f}. #147's restart win is "
                           f"explained by LR level. Change one number, not the schedule.")
            else:
                verdict = (f"SCHEDULE EFFECT SURVIVES: the best fixed LR ({best}) reaches "
                           f"{bl:.5f}, still short of the restart's "
                           f"{REF_RESTART_002:.5f} by {bl - REF_RESTART_002:+.5f}. The "
                           f"restart is doing something a constant LR cannot.")
        print(f"\n[lr] VERDICT: {verdict}")
        print("[lr] LIMIT: optimal LR depends on run length. This characterises "
              "1,500-step runs and is NOT a recommendation for the 100K pretrain.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 148, "config": vars(args), "arms": results,
                   "reference": {"plain_002": REF_PLAIN_002,
                                 "restart_002": REF_RESTART_002},
                   "verdict": verdict}, fh, indent=2)
    print(f"[lr] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
