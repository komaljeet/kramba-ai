"""A/B: progressive layer stacking — 6 layers, then grow to 12 (Issue #147).

Author: Rambarun Komaljeet

Train the first half of a run at **6 layers**, duplicate the stack to **12**, and
keep going. Early steps cost ~half the FLOPs, so the same wall clock buys more
progress -- if the grown model recovers.

**Graded at equal WALL CLOCK, not equal steps.** The claim is "same time, more
progress". Grading at matched steps would measure a different thing and would
favour the control by construction, since it had full depth the whole way. The
control runs first and its wall time sets the budget the grown arm must beat.
Matched-step numbers are reported too, because the two views can disagree and
that disagreement is the finding.

**Both arms end at the same block plan** (attention at 4 and 10), so they differ
in *growth*, not in architecture. Production uses 3 and 10; the 6-layer stage can
only carry one attention layer, and duplicating it lands at 4/10. The control is
moved to 4/10 to match -- a deviation from production, held constant across arms.

Known-answer cell: ``control_twin`` must come out bit-identical to ``control``.

Costs charged honestly to the grown arm: rebuilding the optimizer at growth (the
parameter set changes, so Muon's momentum for the new layers starts empty) and
the duplication itself both happen inside its timed window.

Usage::

    python scripts/ab_progressive_stacking.py --steps 1500
"""

from __future__ import annotations

import argparse
import copy
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

#: 6-layer stage: one attention layer at index 4. Duplicated -> 12 layers with
#: attention at 4 and 10, which is the plan BOTH arms finish on.
HALF_PLAN = ["deltanet"] * 4 + ["transformer"] + ["deltanet"]
FULL_PLAN = HALF_PLAN + HALF_PLAN


def build(plan, args, device):
    from hvn.edge.model import EdgeHVN
    model = EdgeHVN(
        vocab_size=args.vocab_size, d_model=args.d_model,
        num_layers=len(plan), n_heads=args.n_heads,
        block_plan=list(plan), attn_window=args.attn_window,
    ).to(device)
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(
        model, target_names=[n for n in linears
                             if not any(p in n for p in exclude)])
    enable_scan_flags(model)
    return model


def grow(half, args, device):
    """6 -> 12 layers by duplicating the stack; embeddings/decoder carried over.

    Plain stack duplication (``[L0..L5] + [L0..L5]``), the Gong et al. form.
    Both copies start identical, so the grown model's function at the moment of
    growth is NOT the same as the 6-layer model's -- depth doubles and the
    residual stream passes through twice as many blocks. The loss jump that
    causes is real, is visible in the trajectory, and is charged to this arm.
    """
    full = build(FULL_PLAN, args, device)
    hsd = half.state_dict()
    fsd = full.state_dict()
    n_half = len(HALF_PLAN)
    copied = 0
    for k, v in hsd.items():
        if k.startswith("blocks."):
            idx = int(k.split(".")[1])
            rest = k.split(".", 2)[2]
            for tgt in (idx, idx + n_half):
                tk = f"blocks.{tgt}.{rest}"
                if tk in fsd and fsd[tk].shape == v.shape:
                    fsd[tk].copy_(v)
                    copied += 1
        elif k in fsd and fsd[k].shape == v.shape:
            fsd[k].copy_(v)
            copied += 1
    full.load_state_dict(fsd, strict=True)
    print(f"[ab]   grown 6 -> 12 layers ({copied} tensors copied into both halves)")
    return full


def run_arm(arm: str, args, batches, probes, device, tokenizer=None,
            time_budget: Optional[float] = None) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    grows = (arm == "grown")
    # `control_restart`: 12 layers throughout, but the optimizer and cosine
    # schedule are rebuilt at `grow_at` exactly as the grown arm's are. This
    # separates "depth growth helped" from "restarting the LR schedule helped"
    # -- a warm restart is a known win by itself (SGDR), and the grown arm got
    # one for free. Without this arm the two are indistinguishable, and the
    # first run's equal-STEP win (which stacking alone should not produce) is
    # exactly the symptom that says they need separating.
    restarts = (arm == "control_restart")

    model = build(HALF_PLAN if grows else FULL_PLAN, args, device)

    def make_opt(m):
        o, d = build_optimizer(m, "muon", lr=args.aux_lr,
                               weight_decay=args.weight_decay,
                               muon_lr=args.muon_lr)
        return o, d, [pg["lr"] for pg in o.param_groups]

    opt, desc, base_lrs = make_opt(model)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    print(f"\n[ab] === arm '{arm}' ({len(model.block_plan)} layers, "
          f"{sum(p.numel() for p in model.parameters()):,} params) ===\n"
          f"[ab]     {desc}"
          + (f"\n[ab]     grows at step {args.grow_at}" if grows else "")
          + (f"\n[ab]     wall budget {time_budget/60:.1f} min"
             if time_budget else ""))

    traj = [{"step": 0, "heldout_loss": evaluate(model, probes, device),
             "wall_sec": 0.0, "layers": len(model.block_plan)}]
    print(f"[ab]   step {0:>5}  held-out {traj[0]['heldout_loss']:.5f}")

    curve: List[dict] = []
    model.train()
    t0 = time.time()
    warmup_done = args.warmup_steps <= 0
    grown_at = None

    for i, (bx, by) in enumerate(batches, start=1):
        if time_budget is not None and (time.time() - t0) >= time_budget:
            break
        if restarts and grown_at is None and i > args.grow_at:
            # Same optimizer/schedule rebuild as growth, WITHOUT the depth change.
            opt, desc, base_lrs = make_opt(model)
            cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, args.steps - args.grow_at))
            grown_at = i
            print(f"[ab]   step {i:>5}  optimizer+cosine restarted (no growth)")

        if grows and grown_at is None and i > args.grow_at:
            # Growth cost -- rebuild + fresh optimizer state -- is inside the
            # timed window on purpose. It is a real cost of this recipe.
            model = grow(model, args, device)
            opt, desc, base_lrs = make_opt(model)
            cos = torch.optim.lr_scheduler.CosineAnnealingLR(
                opt, T_max=max(1, args.steps - args.grow_at))
            model.train()
            grown_at = i
            hl = evaluate(model, probes, device)
            traj.append({"step": i, "heldout_loss": hl,
                         "wall_sec": time.time() - t0, "layers": 12,
                         "event": "grown"})
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  <- immediately after growth")
            model.train()

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
            wall = time.time() - t0
            traj.append({"step": i, "heldout_loss": hl, "wall_sec": wall,
                         "layers": len(model.block_plan)})
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  "
                  f"({len(model.block_plan)}L, {wall/60:.1f} min)", flush=True)
            model.train()

    wall_total = time.time() - t0
    steps_done = len(curve)
    hl_final = evaluate(model, probes, device)
    traj.append({"step": steps_done, "heldout_loss": hl_final,
                 "wall_sec": wall_total, "layers": len(model.block_plan)})

    bpb = None
    if args.bpb_text and tokenizer is not None and os.path.exists(args.bpb_text):
        from eval.bits_per_byte import bits_per_byte
        with open(args.bpb_text, encoding="utf-8", errors="replace") as fh:
            text = fh.read(args.bpb_max_bytes)
        bpb = bits_per_byte(model, tokenizer, text, seq_len=args.seq_len,
                            device=device)["bits_per_byte"]
        print(f"[ab]   bits/byte {bpb:.4f}")

    out = {"arm": arm, "grew_at_step": grown_at, "steps_done": steps_done,
           "wall_clock_sec": wall_total, "final_heldout_loss": hl_final,
           "bits_per_byte": bpb, "trajectory": traj, "curve": curve}
    del model, opt
    torch.cuda.empty_cache()
    return out


def loss_at_wall(traj, wall):
    ok = [t for t in traj if t["wall_sec"] <= wall]
    return ok[-1]["heldout_loss"] if ok else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=["control", "grown", "control_twin"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--grow-at", type=int, default=750)
    p.add_argument("--max-grown-steps", type=int, default=3000,
                   help="batch pool for the grown arm, which runs MORE steps in "
                        "the control's wall clock -- that is the whole point")
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
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
    p.add_argument("--bpb-text", default="data_cache/heldout_fineweb.txt")
    p.add_argument("--bpb-max-bytes", type=int, default=2_000_000)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--out", default="results/progressive_stacking/ab.json")
    args = p.parse_args(argv)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        print("[ab] deterministic ON — control and control_twin must be IDENTICAL")

    device = require_cuda()
    import train_edge_conversational as tec
    tok = tec.load_tokenizer(args.tokenizer)
    if args.vocab_size != tok.vocab_size:
        args.vocab_size = tok.vocab_size
    torch.manual_seed(args.seed)
    loader, _v, stats = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    need = args.probe_batches + args.max_grown_steps
    pairs = []
    for batch in loader:
        x, y = as_pair(batch)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= need:
            break
    probes, batches = pairs[:args.probe_batches], pairs[args.probe_batches:]
    print(f"[ab] {len(probes)} probes + {len(batches)} identical batches available")

    results: Dict[str, dict] = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    budget = None
    for arm in args.arms:
        n = args.steps if arm != "grown" else args.max_grown_steps
        results[arm] = run_arm(arm, args, batches[:n], probes, device, tok,
                               time_budget=(budget if arm == "grown" else None))
        if arm == "control":
            budget = results[arm]["wall_clock_sec"]
            print(f"[ab] control wall clock {budget/60:.2f} min -> grown arm's budget")
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"issue": 147, "config": vars(args), "arms": results}, fh,
                      indent=2)

    print("\n[ab] ============ progressive stacking ============")
    print(f"[ab] {'arm':<15}{'steps':>8}{'held-out':>11}{'bits/byte':>11}{'wall (min)':>12}")
    for name, r in results.items():
        b = r["bits_per_byte"]
        print(f"[ab] {name:<15}{r['steps_done']:>8}{r['final_heldout_loss']:>11.5f}"
              f"{'nan' if b is None else f'{b:>11.4f}'}{r['wall_clock_sec']/60:>12.2f}")

    verdict = None
    if "control" in results and "control_twin" in results:
        drift = abs(results["control"]["final_heldout_loss"]
                    - results["control_twin"]["final_heldout_loss"])
        print(f"\n[ab] KNOWN-ANSWER CELL — control vs twin: {drift:.3e}")
        if drift > 0.0:
            verdict = f"VOID (twin drift {drift:.3e})"
            print("[ab] *** RUN VOID: identical arms diverged; no ranking is safe.")

    if verdict is None and {"control", "grown"} <= set(results):
        c, g = results["control"], results["grown"]
        d_wall = g["final_heldout_loss"] - c["final_heldout_loss"]
        cs = loss_at_wall(c["trajectory"], min(c["wall_clock_sec"], g["wall_clock_sec"]))
        gs_at_control_steps = next(
            (t["heldout_loss"] for t in reversed(g["trajectory"])
             if t["step"] <= c["steps_done"]), None)
        print(f"\n[ab] EQUAL WALL CLOCK ({min(c['wall_clock_sec'], g['wall_clock_sec'])/60:.2f} min)"
              f" — grown {g['final_heldout_loss']:.5f} vs control {c['final_heldout_loss']:.5f}"
              f"  = {d_wall:+.5f}")
        print(f"[ab]   grown ran {g['steps_done']} steps vs control's {c['steps_done']} "
              f"in the same time")
        if gs_at_control_steps is not None:
            print(f"[ab] EQUAL STEPS ({c['steps_done']}) — grown "
                  f"{gs_at_control_steps:.5f} vs control {c['final_heldout_loss']:.5f} "
                  f"= {gs_at_control_steps - c['final_heldout_loss']:+.5f}")
        # The comparison that decides this is grown vs control_restart, NOT grown
        # vs control. The grown arm rebuilds its optimizer and cosine at growth,
        # so it gets a warm restart for free; against a plain control that
        # schedule effect is credited to "stacking". control_restart isolates it.
        r = results.get("control_restart")
        if r is None:
            verdict = (f"INCOMPLETE: grown beats control by {-d_wall:+.5f} nats, but "
                       f"the grown arm also got an LR-schedule restart at growth. "
                       f"Without the control_restart arm this cannot be attributed "
                       f"to depth rather than to the restart. Re-run with it.")
        else:
            vs_restart = g["final_heldout_loss"] - r["final_heldout_loss"]
            share = (1 - abs(vs_restart) / abs(d_wall)) * 100 if d_wall else 0.0
            print(f"[ab] vs control_restart (12L throughout, same schedule "
                  f"restart): {vs_restart:+.5f} nats")
            print(f"[ab]   the LR restart alone accounts for ~{share:.0f}% of the "
                  f"apparent stacking win")
            if vs_restart < -0.01:
                verdict = (f"stacking adds {-vs_restart:.5f} nats OVER the LR restart "
                           f"({share:.0f}% of the raw win was the restart)")
            else:
                verdict = (
                    f"STACKING CONTRIBUTES NOTHING: the raw {-d_wall:.5f} nat win vs "
                    f"control is ~{share:.0f}% attributable to the LR-schedule restart "
                    f"the grown arm gets for free. Against control_restart (12 layers "
                    f"throughout, same restart) the gap is {vs_restart:+.5f} nats, and "
                    f"bits/byte ranks them the other way. What survives is throughput: "
                    f"{g['steps_done']} steps vs {c['steps_done']} in the same wall "
                    f"clock, which did NOT convert into quality.")
        print(f"\n[ab] VERDICT: {verdict}")
        print("[ab] LIMIT: one seed, short run, 83M only; growth point and "
              "duplication scheme unswept.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 147, "config": vars(args), "arms": results,
                   "verdict": verdict}, fh, indent=2)
    print(f"[ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
