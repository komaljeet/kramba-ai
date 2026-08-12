"""A/B: does `n_heads` 8 vs 16 cost quality? (Issue #144)

Author: Rambarun Komaljeet

`docs/training.md` records `--n-heads 8 -> 16` as a **2.4x speedup at 355M**
(#112), and #141 measured **1.12-1.17x at 83M**.  It is the last unclaimed
pretrain throughput lever -- and its **quality effect has never been measured**.
It is documented purely as a speed knob, which is precisely how
``--block-coordinate`` was documented before #112 discovered it costs ~4x
perplexity.

**Why this cannot be a warm-started A/B.** `n_heads` changes the architecture --
``read_norm_weight`` is ``[n_heads, d_head]`` and the head split changes what
q/k/v mean -- so there is no way to swap heads on an existing checkpoint. Both
arms train from scratch.

**Discipline inherited from `scripts/ab_prodigy.py` (#134):** identical seed,
identical pre-materialised batch sequence for every arm, graded on **held-out
probe loss and bits/byte, never train loss** (#132 was retracted for grading on
train loss), and ``--deterministic`` ON, because without it two identical arms
land ~0.0046 nats apart from non-deterministic SDPA-backward atomics -- noise
sitting inside the effect being measured.

**The known-answer cell: `h8_twin`.**  A third arm that is `h8` run again under a
different name.  Under ``--deterministic`` it must come out **bit-identical** to
`h8`.  If it does not, determinism is not holding and no h8-vs-h16 difference can
be trusted -- a small gap would be indistinguishable from drift.  This is asked
before the numbers land, not after, and the script says the run is VOID if the
twin diverges.

**Two numbers are reported, deliberately:** held-out loss at matched **steps**
(the quality question) and at matched **wall-clock** (the practical question,
since h16 also trains faster and gets more steps per minute).  They can disagree,
and the disagreement is the actual decision.

Usage::

    python scripts/ab_n_heads.py --steps 1500
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
    as_pair, enable_scan_flags, evaluate, require_cuda,
)
from hvn.optimizers import build_optimizer                        # noqa: E402

GPT2_VOCAB_SIZE = 50257

#: arm name -> n_heads. "h8_twin" is h8 again: the known-answer cell.
#: Issue #159: the 157M code config is d_model=896, so the question is
#: 14 heads (d_head 64, production) vs 28 (d_head 32, measured 20.4% faster).
#: The twin is 14 run again -- under --deterministic it must come out
#: bit-identical, giving a zero noise floor.
ARMS = {"h14": 14, "h28": 28, "h14_twin": 14}


def build_model(n_heads: int, args, device):
    """Fresh-init EdgeHVN at the production shape, with n_heads overridden.

    Everything except `n_heads` is held at the 83M hybrid's config so the arms
    differ in exactly one thing.  d_model is unchanged, so d_head moves 64 -> 32
    -- that is the lever, not a confound.
    """
    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model

    model = EdgeHVN(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=n_heads,
        block_plan=(["transformer" if i in (4, 12) else "deltanet"
                     for i in range(args.num_layers)]),
        attn_window=args.attn_window,
    ).to(device)
    apply_ternary_to_model(model)
    return model


def run_arm(arm: str, args, batches, probes, device, tokenizer=None) -> dict:
    n_heads = ARMS[arm]
    # Re-seed identically per arm: the arms must differ in n_heads alone, not in
    # their random init draw or their data order (the batches are shared).
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = build_model(n_heads, args, device)
    enable_scan_flags(model)
    n_params = sum(p.numel() for p in model.parameters())
    opt, desc = build_optimizer(model, "muon", lr=args.aux_lr,
                                weight_decay=args.weight_decay,
                                muon_lr=args.muon_lr)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    print(f"\n[ab] === arm '{arm}' (n_heads={n_heads}, "
          f"d_head={args.d_model // n_heads}, {n_params:,} params) ===\n"
          f"[ab]     {desc}")

    traj = [{"step": 0, "heldout_loss": evaluate(model, probes, device),
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

        curve.append({"step": i, "loss": float(loss.detach()),
                      "lr": opt.param_groups[0]["lr"]})

        if i % args.eval_every == 0:
            hl = evaluate(model, probes, device)
            wall = time.time() - t0
            traj.append({"step": i, "heldout_loss": hl, "wall_sec": wall})
            recent = [c["loss"] for c in curve[-args.eval_every:]]
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  "
                  f"train {sum(recent)/len(recent):.5f}  "
                  f"({wall/60:.1f} min)", flush=True)
            model.train()

    wall_total = time.time() - t0

    # bits/byte: normalises against source TEXT rather than tokens. Both arms
    # share a vocabulary here, so held-out loss is already comparable -- this is
    # reported as a second, independent view rather than as the fix it was in #139.
    bpb = None
    if args.bpb_text and tokenizer is not None and os.path.exists(args.bpb_text):
        from eval.bits_per_byte import bits_per_byte
        with open(args.bpb_text, encoding="utf-8", errors="replace") as fh:
            heldout = fh.read(args.bpb_max_bytes)
        bpb = bits_per_byte(model, tokenizer, heldout, seq_len=args.seq_len,
                            device=device)
        print(f"[ab]   bits/byte {bpb['bits_per_byte']:.4f} on "
              f"{bpb['n_bytes']:,} held-out bytes")

    out = {
        "arm": arm, "n_heads": n_heads, "d_head": args.d_model // n_heads,
        "params": n_params,
        "bits_per_byte": bpb,
        "wall_clock_sec": wall_total,
        "steps_per_sec": args.steps / wall_total,
        "final_heldout_loss": traj[-1]["heldout_loss"],
        "start_heldout_loss": traj[0]["heldout_loss"],
        "final_train_loss_last200": sum(c["loss"] for c in curve[-200:]) / 200,
        "trajectory": traj, "curve": curve,
    }
    del model, opt
    torch.cuda.empty_cache()
    return out


def heldout_at_wall(traj: List[dict], wall_sec: float) -> Optional[float]:
    """Held-out loss at (or just before) a given wall-clock budget.

    The matched-wall-clock view: h16 trains faster, so within a fixed number of
    MINUTES it completes more steps. Grading only at matched steps would hide
    that, and grading only at matched time would hide a per-step quality cost --
    hence both.
    """
    ok = [t for t in traj if t.get("wall_sec", 0.0) <= wall_sec]
    return ok[-1]["heldout_loss"] if ok else None


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=["h14", "h28", "h14_twin"])
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=896)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--attn-window", type=int, default=4096)
    p.add_argument("--muon-lr", type=float, default=0.02,
                   help="pretrain-regime value per #129 (0.002 is the "
                        "continued-training value and would under-train here)")
    p.add_argument("--aux-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--code-corpus", default=None,
                   help="train the arms on a code corpus (#136), not the "
                        "four-source mix -- required when the A/B is deciding "
                        "a code pretrain")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vocab-size", type=int, default=GPT2_VOCAB_SIZE)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--bpb-text", default="data_cache/heldout_fineweb.txt")
    p.add_argument("--bpb-max-bytes", type=int, default=2_000_000)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--out", default="results/n_heads/ab_n_heads.json")
    args = p.parse_args(argv)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        print("[ab] deterministic ON — h8 and h8_twin must come out IDENTICAL")

    device = require_cuda()
    import train_edge_conversational as tec

    tok = tec.load_tokenizer(args.tokenizer)
    # Issue #155: len(tokenizer), NOT tokenizer.vocab_size. HuggingFace excludes
    # ADDED tokens from vocab_size, so a FIM tokenizer reports 32000 while its
    # sentinels occupy 32000-32002. Sizing the embedding table from vocab_size
    # builds it 3 rows short and the first sentinel indexes out of bounds --
    # surfacing as a bare "CUDA error: device-side assert triggered" with no hint
    # of the cause. Same bug as the training script had; this harness had its own
    # copy of it.
    if args.vocab_size != len(tok):
        args.vocab_size = len(tok)
    torch.manual_seed(args.seed)
    # Issue #159: this A/B decides a CODE pretrain, so it must train on the code
    # corpus -- the default four-source mix would answer the question for the
    # wrong data. `code_corpus` REPLACES the mix entirely (#136), and `fim` must
    # match the run being decided or the cache signature (correctly) refuses.
    loader, _v, stats = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0,
        code_corpus=args.code_corpus,
        fim=({"rate": 0.5, "spm_rate": 0.5, "seed": 0} if args.code_corpus else None))
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
            json.dump({"issue": 144, "config": vars(args), "arms": results},
                      fh, indent=2)

    # ---- verdict -----------------------------------------------------------
    print(f"\n[ab] ================= n_heads A/B =================")
    print(f"[ab] {'arm':<10}{'d_head':>8}{'held-out':>12}{'bits/byte':>12}"
          f"{'wall (min)':>12}{'steps/s':>10}")
    for name, r in results.items():
        bpb = r["bits_per_byte"]["bits_per_byte"] if r.get("bits_per_byte") else float("nan")
        print(f"[ab] {name:<10}{r['d_head']:>8}{r['final_heldout_loss']:>12.5f}"
              f"{bpb:>12.4f}{r['wall_clock_sec']/60:>12.1f}{r['steps_per_sec']:>10.2f}")

    verdict = None
    if "h8" in results and "h8_twin" in results:
        drift = abs(results["h8"]["final_heldout_loss"]
                    - results["h8_twin"]["final_heldout_loss"])
        print(f"\n[ab] KNOWN-ANSWER CELL — h8 vs h8_twin: {drift:.3e}")
        if drift > 0.0:
            print("[ab] *** RUN VOID: two identical arms did not match under "
                  "--deterministic. The floor is not zero, so any h8-vs-h16 gap "
                  "below this cannot be distinguished from drift. Do NOT report "
                  "these numbers as a quality verdict.")
            verdict = f"VOID (twin drift {drift:.3e})"
        else:
            print("[ab] twin arms bit-identical -> noise floor is exactly 0, "
                  "so ANY h8-vs-h16 difference below is real signal.")

    if verdict is None and "h8" in results and "h16" in results:
        a, b = results["h8"], results["h16"]
        gap = b["final_heldout_loss"] - a["final_heldout_loss"]
        print(f"\n[ab] MATCHED STEPS ({args.steps}): h16 - h8 = {gap:+.5f} nats "
              f"({'h16 worse' if gap > 0 else 'h16 better'})")
        # Matched wall clock: grade both at the FASTER arm's... no -- at the
        # SLOWER arm's budget, which is the one both actually reached.
        budget = min(a["wall_clock_sec"], b["wall_clock_sec"])
        ha, hb = heldout_at_wall(a["trajectory"], budget), heldout_at_wall(b["trajectory"], budget)
        if ha is not None and hb is not None:
            wgap = hb - ha
            print(f"[ab] MATCHED WALL CLOCK ({budget/60:.1f} min): "
                  f"h16 - h8 = {wgap:+.5f} nats "
                  f"({'h16 worse' if wgap > 0 else 'h16 better'})")
        speed = b["steps_per_sec"] / a["steps_per_sec"]
        print(f"[ab] h16 throughput: {speed:.3f}x of h8")
        verdict = (f"h16 {'costs' if gap > 0 else 'gains'} "
                   f"{abs(gap):.5f} nats at matched steps, at {speed:.2f}x speed")
        print(f"\n[ab] VERDICT: {verdict}")
        print("[ab] LIMIT: one seed, short from-scratch run -> this bounds "
              "EARLY-training quality at 83M only, not convergence or larger d.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 144, "config": vars(args), "arms": results,
                   "verdict": verdict}, fh, indent=2)
    print(f"[ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
