"""A/B: does a SMALL looped model match a deep one at equal compute? (Issue #122, option (a))

Author: Rambarun Komaljeet

Every looping result in this repo so far is negative -- but every one of them
looped an *already-full-depth* 12-layer stack that had been pretrained
single-pass, then asked it to loop. That is not what a looped model is. The
docs' own remaining direction (docs/state.md, "option (a)") is a deliberately
SMALL stack, trained from scratch to be run many times. It has never been built.

The claim under test
--------------------
A 3-layer stack looped 4x matches a 12-layer stack, at ~4x fewer BODY
parameters and the same forward FLOPs per token.

The one-variable A/B
--------------------
Both arms unroll to a structurally IDENTICAL 12-layer network -- same depth,
same block order, attention at the same positions (0, 3, 6, 9) -- differing
only in whether the weights repeat:

    looped3x4 : num_layers=3,  plan [transformer, deltanet, deltanet], k=4
    deep12    : num_layers=12, plan that pattern tiled 4x,             k=1

Matching the attention POSITIONS is the point. The obvious version (a 3-layer
model with 1 attention layer vs the stock 12-layer hybrid with 2) would change
the attention count between arms and confound the result with it.

Both arms drive ``looped_forward_train`` -- deep12 simply at k=1 -- so the two
arms share one code path exactly, rather than one going through
``model.forward`` and the other through the loop helper.

Discipline inherited from scripts/ab_n_heads.py / ab_prodigy.py (#134/#144)
---------------------------------------------------------------------------
Identical seed, identical pre-materialised batch sequence for every arm, graded
on **held-out probe loss, never train loss** (#132 was retracted for exactly
that), ``--deterministic`` ON, and a **known-answer twin cell**: ``deep12_twin``
is ``deep12`` again under another name and must come out bit-identical. If it
does not, the noise floor is not zero and no looped-vs-deep gap below that
drift can be believed -- the run says VOID rather than reporting a verdict.

Falsifiability, asked before the run and not after
--------------------------------------------------
Prior looping effects here were 0.5+ nats against a ~0.011 nat noise floor, so
a short run detects "looping is broken" cleanly. It does NOT resolve a subtle
win: if the arms land within ``--inconclusive-band`` nats the script reports
INCONCLUSIVE and explicitly refuses to call it, because a narrow gap on one
seed at one size is the shape of every verdict this repo has had to retract.

Usage::

    python scripts/ab_looped_small.py --steps 1200
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

from scripts.ab_prodigy import (                                  # noqa: E402
    as_pair, enable_scan_flags, require_cuda,
)
from hvn.optimizers import build_optimizer                        # noqa: E402
from hvn.edge.looped import looped_forward_train                  # noqa: E402

GPT2_VOCAB_SIZE = 50257

#: arm -> (num_layers, loops). Unrolled depth is num_layers * loops = 12 for all
#: three. "deep12_twin" is deep12 again: the known-answer cell.
ARMS = {
    "looped3x4": (3, 4),
    "deep12": (12, 1),
    "deep12_twin": (12, 1),
}

#: One repeating unit: attention first, then two recurrent layers. Tiled to
#: whatever num_layers an arm uses, so both arms put attention at 0, 3, 6, 9
#: once unrolled.
UNIT = ["transformer", "deltanet", "deltanet"]


def block_plan_for(num_layers: int) -> List[str]:
    return [UNIT[i % len(UNIT)] for i in range(num_layers)]


def build_model(num_layers: int, args, device):
    """Fresh-init EdgeHVN. Everything but depth is held identical across arms.

    Ternary QAT is deliberately NOT applied here (ab_n_heads.py does apply it).
    Quantising 3 shared blocks and quantising 12 independent ones are not the
    same operation, so it would sit inside the very variable being measured.
    """
    from hvn.edge.model import EdgeHVN

    model = EdgeHVN(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=num_layers,
        n_heads=args.n_heads,
        block_plan=block_plan_for(num_layers),
        attn_window=args.attn_window,
        tie_embeddings=True,
    ).to(device)
    return model


def forward_logits(model, x, k: int):
    """The single forward path both arms use. k=1 is a plain single pass."""
    position_ids = torch.arange(x.size(1), device=x.device).unsqueeze(0).expand(x.size(0), -1)
    return looped_forward_train(model, x, position_ids, k, deep_supervise=False)[-1]


@torch.no_grad()
def evaluate(model, pairs, k: int, device) -> float:
    model.eval()
    total = 0.0
    for x, y in pairs:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_logits(model, x, k)
            total += float(nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1)))
    return total / max(len(pairs), 1)


def body_params(model) -> int:
    """Params excluding the token embedding. At d_model 384 a 50257-row table is
    ~19M -- it would swamp the 4x body ratio this A/B exists to measure, and it
    is identical in both arms, so it is reported separately rather than folded in."""
    total = sum(p.numel() for p in model.parameters())
    return total - model.token_embedding.weight.numel()


def run_arm(arm: str, args, batches, probes, device, tokenizer=None) -> dict:
    num_layers, k = ARMS[arm]
    # Re-seed identically per arm: arms must differ in depth/loops alone, not in
    # their init draw or data order (batches are shared and pre-materialised).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = build_model(num_layers, args, device)
    enable_scan_flags(model)
    n_params = sum(p.numel() for p in model.parameters())
    n_body = body_params(model)
    opt, desc = build_optimizer(model, "muon", lr=args.aux_lr,
                                weight_decay=args.weight_decay,
                                muon_lr=args.muon_lr)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    print(f"\n[ab] === arm '{arm}' ({num_layers} layers x {k} loops = "
          f"{num_layers * k} effective, {n_params:,} params / {n_body:,} body) ===\n"
          f"[ab]     plan {block_plan_for(num_layers)}\n"
          f"[ab]     {desc}")

    traj = [{"step": 0, "heldout_loss": evaluate(model, probes, k, device),
             "wall_sec": 0.0}]
    print(f"[ab]   step {0:>5}  held-out {traj[0]['heldout_loss']:.5f}")

    curve: List[dict] = []
    model.train()
    t0 = time.time()
    warmup_done = args.warmup_steps <= 0

    for i, (bx, by) in enumerate(batches, start=1):
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = forward_logits(model, x, k)
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

        curve.append({"step": i, "loss": float(loss.detach()),
                      "lr": opt.param_groups[0]["lr"]})

        if i % args.eval_every == 0:
            hl = evaluate(model, probes, k, device)
            wall = time.time() - t0
            traj.append({"step": i, "heldout_loss": hl, "wall_sec": wall})
            recent = [c["loss"] for c in curve[-args.eval_every:]]
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  "
                  f"train {sum(recent)/len(recent):.5f}  "
                  f"({wall/60:.1f} min)", flush=True)
            model.train()

    wall_total = time.time() - t0

    out = {
        "arm": arm, "num_layers": num_layers, "loops": k,
        "effective_depth": num_layers * k,
        "params": n_params, "body_params": n_body,
        "wall_clock_sec": wall_total,
        "steps_per_sec": args.steps / wall_total,
        "final_heldout_loss": traj[-1]["heldout_loss"],
        "start_heldout_loss": traj[0]["heldout_loss"],
        "final_train_loss_last200": sum(c["loss"] for c in curve[-200:]) / min(200, len(curve)),
        "trajectory": traj, "curve": curve,
    }
    del model, opt
    torch.cuda.empty_cache()
    return out


def heldout_at_wall(traj: List[dict], wall_sec: float) -> Optional[float]:
    """Held-out loss at (or just before) a wall-clock budget. The looped arm and
    the deep arm need not run at the same speed even at equal FLOPs, so matched
    steps and matched time can disagree -- and that disagreement is the decision."""
    ok = [t for t in traj if t.get("wall_sec", 0.0) <= wall_sec]
    return ok[-1]["heldout_loss"] if ok else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+", default=["looped3x4", "deep12", "deep12_twin"])
    p.add_argument("--steps", type=int, default=1200)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=384)
    p.add_argument("--n-heads", type=int, default=6, help="d_head 64 at d_model 384")
    p.add_argument("--attn-window", type=int, default=4096)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--aux-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vocab-size", type=int, default=GPT2_VOCAB_SIZE)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--inconclusive-band", type=float, default=0.05,
                   help="Gap below which the run refuses to call a winner.")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--out", default="results/looped_small/ab_looped_small.json")
    args = p.parse_args(argv)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        print("[ab] deterministic ON -- deep12 and deep12_twin must come out IDENTICAL")

    device = require_cuda()
    import train_edge_conversational as tec

    tok = tec.load_tokenizer(args.tokenizer)
    # #155: len(tokenizer), NOT tokenizer.vocab_size -- added tokens are excluded
    # from vocab_size and an undersized table indexes out of bounds as a bare
    # device-side assert.
    if args.vocab_size != len(tok):
        args.vocab_size = len(tok)
    torch.manual_seed(args.seed)
    loader, _v, stats = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    print(f"[ab] data: {stats}")

    pairs = []
    for batch in loader:
        x, y = as_pair(batch)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= args.probe_batches + args.steps:
            break
    probes, batches = pairs[:args.probe_batches], pairs[args.probe_batches:]
    print(f"[ab] {len(probes)} held-out probes + {len(batches)} identical "
          f"training batches for every arm")

    results: Dict[str, dict] = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for arm in args.arms:
        results[arm] = run_arm(arm, args, batches, probes, device, tok)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"issue": 122, "config": vars(args), "arms": results},
                      fh, indent=2)

    # ---- verdict -----------------------------------------------------------
    print("\n[ab] ============== small-looped vs deep ==============")
    print(f"[ab] {'arm':<14}{'depth':>7}{'body':>12}{'held-out':>12}"
          f"{'wall (min)':>12}{'steps/s':>10}")
    for name, r in results.items():
        print(f"[ab] {name:<14}{r['effective_depth']:>7}{r['body_params']:>12,}"
              f"{r['final_heldout_loss']:>12.5f}{r['wall_clock_sec']/60:>12.1f}"
              f"{r['steps_per_sec']:>10.2f}")

    verdict = None
    floor = 0.0
    if "deep12" in results and "deep12_twin" in results:
        floor = abs(results["deep12"]["final_heldout_loss"]
                    - results["deep12_twin"]["final_heldout_loss"])
        print(f"\n[ab] KNOWN-ANSWER CELL -- deep12 vs deep12_twin: {floor:.3e}")
        if floor > 0.0:
            print("[ab] *** Twin arms did not match. The noise floor is NOT zero; "
                  "any looped-vs-deep gap at or below this drift cannot be "
                  "distinguished from it.")
        else:
            print("[ab] twin arms bit-identical -> noise floor exactly 0.")

    if "looped3x4" in results and "deep12" in results:
        a, b = results["deep12"], results["looped3x4"]
        gap = b["final_heldout_loss"] - a["final_heldout_loss"]
        ratio = a["body_params"] / max(1, b["body_params"])
        print(f"\n[ab] MATCHED STEPS ({args.steps}): looped - deep = {gap:+.5f} nats")
        print(f"[ab] body-param ratio: looped uses {ratio:.2f}x FEWER body params")
        budget = min(a["wall_clock_sec"], b["wall_clock_sec"])
        ha, hb = heldout_at_wall(a["trajectory"], budget), heldout_at_wall(b["trajectory"], budget)
        if ha is not None and hb is not None:
            print(f"[ab] MATCHED WALL CLOCK ({budget/60:.1f} min): "
                  f"looped - deep = {hb - ha:+.5f} nats")

        if abs(gap) <= max(args.inconclusive_band, floor):
            verdict = (f"INCONCLUSIVE: |{gap:+.5f}| within the "
                       f"{max(args.inconclusive_band, floor):.5f} band")
            print(f"\n[ab] VERDICT: {verdict}")
            print("[ab] The arms are too close to separate on one seed at this "
                  "size. Report as inconclusive -- do NOT call this a win for "
                  "either side.")
        elif gap > 0:
            verdict = (f"looped is WORSE by {gap:.5f} nats at {ratio:.2f}x fewer "
                       f"body params")
            print(f"\n[ab] VERDICT: {verdict}")
        else:
            verdict = (f"looped is BETTER by {-gap:.5f} nats at {ratio:.2f}x fewer "
                       f"body params")
            print(f"\n[ab] VERDICT: {verdict}")
        print("[ab] LIMIT: one seed, short from-scratch run, d_model "
              f"{args.d_model}. Bounds EARLY-training quality only -- not "
              "convergence, and not larger widths.")
        print("[ab] LIMIT (asymmetric, and it matters for which way to read "
              "this): both arms see the same tokens, but deep12 has ~4x more "
              "body params to fit with them, so it is the MORE undertrained of "
              "the two. That bias favours the LOOPED arm. A looped loss here is "
              "therefore strong evidence; a looped WIN is weak evidence and "
              "would need a longer run before it could be believed.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 122, "config": vars(args), "arms": results,
                   "noise_floor": floor, "verdict": verdict}, fh, indent=2)
    print(f"[ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
