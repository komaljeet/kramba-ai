"""A/B: hand-tuned Muon vs parameter-free Prodigy, in BOTH regimes (Issue #134).

Written by: Rambarun Komaljeet

The claim under test is that ONE invocation with no learning rate covers both a
from-scratch pretrain and a continue-from-checkpoint fine-tune. Testing only one
regime would prove nothing about that claim, so ``--regime`` selects:

``finetune``  continue from a trained checkpoint (the #133 setup)
``pretrain``  a freshly initialised model of the same shape

Arms differ ONLY in the optimizer. Every arm sees a byte-identical
pre-materialised batch sequence, is re-seeded identically, and is graded on a
fixed held-out probe set — never on train loss (#132's retraction).

``--deterministic`` is ON by default. Two identical-code arms otherwise land
~0.0046 nats apart with the sign flipping run to run, because this torch build
has no flash-attention and the hybrid's 2 attention layers fall back to a
mem-efficient SDPA kernel whose backward accumulates grad_q with atomics (#133).
That floor sits inside the effect being measured.

**The Prodigy arms are never given a learning rate.** If one needs tuning to
compete, it has failed its own premise and that is the reportable result.

Usage
-----
    python scripts/ab_prodigy.py --regime finetune --steps 5000
    python scripts/ab_prodigy.py --regime pretrain --steps 1500
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

from hvn.optimizers import Prodigy, build_optimizer  # noqa: E402

GPT2_VOCAB_SIZE = 50257
DEFAULT_BASE = "checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt"


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA required — this project never falls back to CPU "
                         "(docs/ai_rules.md).")
    return torch.device("cuda")


def enable_scan_flags(model) -> None:
    """Triton + megakernel scan on every DeltaNet block, in EVERY arm.

    Mirrors continue_ft_hybrid.py. Identical across arms so it cannot confound;
    leaving it off runs DeltaNet's eager scan at ~42x the cost (#122).
    """
    n = 0
    for blk in getattr(model, "blocks", []):
        inner = getattr(blk, "block", blk)
        if hasattr(inner, "use_triton_scan"):
            inner.use_triton_scan = True
            inner.use_megakernel_scan = True
            n += 1
    print(f"[ab] triton+megakernel scan on {n} DeltaNet blocks")


def as_pair(batch):
    """The loader yields (x, y, src), both seq_len wide.

    Re-slicing x into x[:, :-1] / x[:, 1:] drops a token and leaves an odd width
    that the megakernel backward rejects ("S=511 must be divisible by block B=4").
    """
    return batch[0], batch[1]


@torch.no_grad()
def evaluate(model, pairs, device) -> float:
    model.eval()
    total = 0.0
    for x, y in pairs:
        x, y = x.to(device), y.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            total += float(nn.functional.cross_entropy(
                logits.reshape(-1, GPT2_VOCAB_SIZE).float(), y.reshape(-1)))
    return total / max(len(pairs), 1)


def build_model(args, device):
    """Warm (from checkpoint) or cold (fresh init) — the two regimes."""
    import train_edge_conversational as tec

    if args.regime == "finetune":
        model, base_step, _ck = tec.load_checkpoint_for_resume(args.base, device)
        return model, base_step

    ckpt = torch.load(args.base, map_location="cpu", weights_only=False)
    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model
    # Same architecture as the checkpoint, random weights — so the pretrain and
    # finetune regimes differ ONLY in the starting point, not in model shape.
    # Issue #139: vocab_size and tie_embeddings are size knobs, not constants.
    # At d_model=512 the two vocab tables are 62% of an "83M" model, so an A/B
    # that cannot vary them cannot see the largest lever in the config.
    model = EdgeHVN(
        vocab_size=args.vocab_size,
        tie_embeddings=args.tie_embeddings,
        d_model=ckpt.get("d_model", 512),
        num_layers=ckpt.get("num_layers", 12),
        n_heads=ckpt.get("n_heads", 8),
        block_plan=ckpt.get("block_plan"),
        attn_window=ckpt.get("attn_window", 4096),
    ).to(device)
    if ckpt.get("ternary", True):
        apply_ternary_to_model(model)
    return model, 0


def make_optimizer(arm: str, model, args):
    if arm == "muon":
        return build_optimizer(model, "muon", lr=args.aux_lr,
                               weight_decay=args.weight_decay,
                               muon_lr=args.muon_lr)
    if arm in ("prodigy", "muon-prodigy"):
        # No LR passed. That is the entire point.
        return build_optimizer(model, arm, lr=1.0,
                               weight_decay=args.weight_decay)
    raise ValueError(f"unknown arm {arm!r}")


def run_arm(arm, args, batches, probes, device) -> dict:
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model, base_step = build_model(args, device)
    enable_scan_flags(model)
    opt, desc = make_optimizer(arm, model, args)
    is_prodigy = isinstance(opt, Prodigy)

    # Hand-tuned arms get warmup + cosine, the schedule they were tuned with.
    # Prodigy gets NEITHER: its d-ramp is the warmup, and a decay schedule
    # would fight an estimator that is monotone by construction. Giving it a
    # schedule would also quietly re-introduce a tuned knob.
    cos = None if is_prodigy else torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=args.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    print(f"\n[ab] === arm '{arm}' ({args.regime}, base step {base_step}) ===\n"
          f"[ab]     {desc}\n"
          f"[ab]     schedule: {'none (d-ramp is the warmup)' if is_prodigy else 'warmup+cosine'}")

    traj = [{"step": 0, "heldout_loss": evaluate(model, probes, device)}]
    print(f"[ab]   step {0:>5}  held-out {traj[0]['heldout_loss']:.5f}")

    curve: List[dict] = []
    model.train()
    t0 = time.time()
    warmup_done = is_prodigy or args.warmup_steps <= 0

    for i, (bx, by) in enumerate(batches, start=1):
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, GPT2_VOCAB_SIZE).float(), y.reshape(-1))
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
        if cos is not None and warmup_done:
            cos.step()

        curve.append({
            "step": i, "loss": float(loss.detach()),
            "lr": opt.effective_lr if is_prodigy else opt.param_groups[0]["lr"],
            "d": opt.d if is_prodigy else None,
        })

        if i % args.eval_every == 0:
            hl = evaluate(model, probes, device)
            traj.append({"step": i, "heldout_loss": hl})
            recent = [c["loss"] for c in curve[-args.eval_every:]]
            extra = f"  d {opt.d:.3e}" if is_prodigy else ""
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  "
                  f"train {sum(recent)/len(recent):.5f}  "
                  f"lr {curve[-1]['lr']:.3e}{extra}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)
            model.train()

    out = {
        "arm": arm, "regime": args.regime,
        "optimizer_desc": desc,
        "wall_clock_sec": time.time() - t0,
        "final_heldout_loss": traj[-1]["heldout_loss"],
        "start_heldout_loss": traj[0]["heldout_loss"],
        "final_train_loss_last200": sum(c["loss"] for c in curve[-200:]) / 200,
        "hand_tuned_lr": None if is_prodigy else args.muon_lr,
        "estimated_lr": opt.effective_lr if is_prodigy else None,
        "final_d": opt.d if is_prodigy else None,
        "trajectory": traj, "curve": curve,
    }
    if is_prodigy:
        print(f"[ab]   ESTIMATED step size: {opt.effective_lr:.4e} "
              f"(hand-tuned reference: {args.muon_lr:g})")
    del model, opt
    torch.cuda.empty_cache()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--regime", choices=["finetune", "pretrain"], required=True)
    p.add_argument("--arms", nargs="+", default=["muon", "muon-prodigy", "prodigy"])
    p.add_argument("--base", default=DEFAULT_BASE)
    p.add_argument("--steps", type=int, default=5000)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--muon-lr", type=float, default=0.002,
                   help="hand-tuned reference for the 'muon' arm only. Default "
                        "0.002 = the sane value #133 landed on for continued "
                        "training; use 0.02 for the pretrain regime (#129).")
    p.add_argument("--aux-lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default=None)
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--tie-embeddings", action="store_true",
                   help="Issue #139: share the input embedding and output head. "
                        "Removes 31 pct of parameters at ZERO FLOP cost (the "
                        "input table is a gather, not a matmul). Pretrain regime "
                        "only — tying changes the model, so it cannot be applied "
                        "to an existing untied checkpoint.")
    p.add_argument("--vocab-size", type=int, default=GPT2_VOCAB_SIZE)
    args = p.parse_args(argv)

    if args.tie_embeddings and args.regime == "finetune":
        p.error("--tie-embeddings only applies to --regime pretrain: tying "
                "changes the parameter set, so it cannot be applied to an "
                "existing untied checkpoint (it would silently drop the head).")

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        print("[ab] deterministic ON (~9 pct slower, arms bitwise comparable)")

    device = require_cuda()
    out_dir = "results/prodigy"
    os.makedirs(out_dir, exist_ok=True)
    out_path = args.out or os.path.join(out_dir, f"ab_{args.regime}.json")

    import train_edge_conversational as tec
    from transformers import GPT2TokenizerFast

    tok = GPT2TokenizerFast.from_pretrained("gpt2")
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
    if os.path.exists(out_path):
        with open(out_path, encoding="utf-8") as fh:
            results = json.load(fh).get("arms", {})

    for arm in args.arms:
        results[arm] = run_arm(arm, args, batches, probes, device)
        _write(out_path, args, results)

    print(f"\n[ab] ============ {args.regime.upper()} ============")
    print(f"[ab] {'arm':<14} {'final held-out':>15} {'step size used':>16} {'wall (min)':>11}")
    for name, r in results.items():
        lr = (f"{r['estimated_lr']:.3e} (auto)" if r["estimated_lr"] is not None
              else f"{r['hand_tuned_lr']:g} (tuned)")
        print(f"[ab] {name:<14} {r['final_heldout_loss']:>15.5f} {lr:>16} "
              f"{r['wall_clock_sec'] / 60:>11.1f}")
    if "muon" in results:
        base = results["muon"]["final_heldout_loss"]
        for name, r in results.items():
            if name == "muon":
                continue
            gap = base - r["final_heldout_loss"]
            print(f"[ab] {name} vs hand-tuned muon: {-gap:+.5f} "
                  f"({'auto WINS' if gap > 0 else 'hand-tuned wins'}); "
                  f"noise floor sd 0.0046, so this is {abs(gap) / 0.0046:.1f} sigma")
    _write(out_path, args, results)
    print(f"[ab] wrote {out_path}")
    return 0


def _write(path, args, results):
    payload = {
        "regime": args.regime, "steps": args.steps, "seed": args.seed,
        "batch_size": args.batch_size, "seq_len": args.seq_len,
        "base_checkpoint": args.base.replace("\\", "/"),
        "hand_tuned_muon_lr": args.muon_lr,
        "graded_on": "held-out probe loss (NOT train loss — see #132)",
        "noise_floor_sd": 0.0046,
        "arms": results,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
    os.replace(tmp, path)


if __name__ == "__main__":
    raise SystemExit(main())
