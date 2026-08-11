"""A/B: does TRAINING with group-wise ternary scales beat per-tensor? (Issue #146)

Author: Rambarun Komaljeet

#145 killed the *retrofit*: re-grouping an already-trained checkpoint's scales
makes it worse (+0.25 to +0.33 nats) even though reconstruction error improves,
because the weights sit where per-tensor rounding put them.

**That mechanism is the reason to test training.** A model trained under finer
scales adapts to them from step 0, so the failure mode that sinks the retrofit
does not apply. #145 could not test it and said so; this does.

Headroom: ternary costs **+20% PPL** today (dense fp 119.4 vs dense ternary
143.2, docs/state.md 2026-08-02).

Method follows ``scripts/ab_n_heads.py`` (#144): from scratch, identical seed,
identical pre-materialised batch sequence for every arm, ``--deterministic``,
graded on **held-out probe loss and bits/byte, never train loss** (#132), with a
**twin arm** (``per_tensor`` run twice) that must come out bit-identical or the
run is void.

**Storage is reported alongside quality, and that is not decoration.** Ternary's
entire point is bytes. At d=512 a ``group_size=32`` weight carries 8,192 scales
against ~51 KiB of codes -- **+63% storage**. A scheme that wins on loss while
costing 63% more has not obviously won; that storage might buy more elsewhere.
Reporting quality without price is how a lever gets adopted that should not be.

**Ternary target set matches production**, i.e. excluding ``beta_proj`` /
``alpha_proj`` / ``decoder`` as ``continue_ft_hybrid.py`` does -- *not*
``ab_n_heads``'s convert-everything. The decoder alone is 50257x512, the largest
matrix in the model, so whether it is quantized would dominate any storage claim.

Usage::

    python scripts/ab_ternary_grouping.py --steps 1500
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ab_prodigy import (                                   # noqa: E402
    as_pair, enable_scan_flags, evaluate, require_cuda,
)
from hvn.edge.ternary import (                                     # noqa: E402
    GROUP_PER_ROW, GROUP_PER_TENSOR, TernaryLinear, apply_ternary_to_model,
)
from hvn.optimizers import build_optimizer                         # noqa: E402

GPT2_VOCAB_SIZE = 50257

#: arm -> group_size. "per_tensor_twin" is per_tensor again: the known-answer cell.
ARMS = {
    "per_tensor": GROUP_PER_TENSOR,
    "per_row": GROUP_PER_ROW,
    "group_128": 128,
    "group_32": 32,
    "per_tensor_twin": GROUP_PER_TENSOR,
}


def build_model(group_size: int, args, device):
    from hvn.edge.model import EdgeHVN

    model = EdgeHVN(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        num_layers=args.num_layers,
        n_heads=args.n_heads,
        block_plan=["transformer" if i in (3, 10) else "deltanet"
                    for i in range(args.num_layers)],
        attn_window=args.attn_window,
    ).to(device)

    # Production's exclusion list (continue_ft_hybrid.py), not "every Linear".
    linears = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    targets = [n for n in linears if not any(p in n for p in exclude)]
    apply_ternary_to_model(model, target_names=targets, group_size=group_size)
    return model


def storage_report(model) -> dict:
    """Bytes and effective bits/weight over the ternary layers -- the price tag."""
    tern = [m for m in model.modules() if isinstance(m, TernaryLinear)]
    weights = sum(m.weight.numel() for m in tern)
    total = sum(m.memory_bytes() for m in tern)
    return {
        "ternary_layers": len(tern),
        "ternary_weights": weights,
        "ternary_bytes": total,
        "effective_bits_per_weight": (total * 8 / weights) if weights else float("nan"),
    }


def run_arm(arm: str, args, batches, probes, device, tokenizer=None) -> dict:
    group_size = ARMS[arm]
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    model = build_model(group_size, args, device)
    enable_scan_flags(model)
    store = storage_report(model)
    opt, desc = build_optimizer(model, "muon", lr=args.aux_lr,
                                weight_decay=args.weight_decay,
                                muon_lr=args.muon_lr)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    print(f"\n[ab] === arm '{arm}' (group_size={group_size}, "
          f"{store['effective_bits_per_weight']:.3f} bits/weight over "
          f"{store['ternary_layers']} layers) ===\n[ab]     {desc}")

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
        curve.append({"step": i, "loss": float(loss.detach())})

        if i % args.eval_every == 0:
            hl = evaluate(model, probes, device)
            wall = time.time() - t0
            traj.append({"step": i, "heldout_loss": hl, "wall_sec": wall})
            recent = [c["loss"] for c in curve[-args.eval_every:]]
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  "
                  f"train {sum(recent)/len(recent):.5f}  ({wall/60:.1f} min)",
                  flush=True)
            model.train()

    wall_total = time.time() - t0
    bpb = None
    if args.bpb_text and tokenizer is not None and os.path.exists(args.bpb_text):
        from eval.bits_per_byte import bits_per_byte
        with open(args.bpb_text, encoding="utf-8", errors="replace") as fh:
            text = fh.read(args.bpb_max_bytes)
        bpb = bits_per_byte(model, tokenizer, text, seq_len=args.seq_len,
                            device=device)["bits_per_byte"]
        print(f"[ab]   bits/byte {bpb:.4f}")

    out = {"arm": arm, "group_size": group_size, **store,
           "wall_clock_sec": wall_total,
           "final_heldout_loss": traj[-1]["heldout_loss"],
           "bits_per_byte": bpb,
           "final_train_loss_last200": sum(c["loss"] for c in curve[-200:]) / 200,
           "trajectory": traj, "curve": curve}
    del model, opt
    torch.cuda.empty_cache()
    return out


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=list(ARMS))
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)
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
    p.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=False,
                   help="Enable TF32 matmuls, as --fast does. OFF by default here "
                        "and in every other A/B harness in this repo (#134/#144), "
                        "which is a real gap from production: under bf16 autocast "
                        "the fwd/bwd matmuls match either way, but Muon's "
                        "Newton-Schulz orthogonalisation runs fp32 matmuls on 102 "
                        "matrices every step and IS affected. Turn on to check a "
                        "verdict transfers to a --fast run.")
    p.add_argument("--out", default="results/ternary_grouping/ab_qat.json")
    args = p.parse_args(argv)

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        print("[ab] deterministic ON — the two per_tensor arms must be IDENTICAL")
    torch.backends.cuda.matmul.allow_tf32 = args.tf32
    torch.backends.cudnn.allow_tf32 = args.tf32
    print(f"[ab] TF32 {'ON (matches --fast)' if args.tf32 else 'OFF (differs from --fast)'}")

    device = require_cuda()
    import train_edge_conversational as tec

    tok = tec.load_tokenizer(args.tokenizer)
    if args.vocab_size != tok.vocab_size:
        args.vocab_size = tok.vocab_size
    torch.manual_seed(args.seed)
    loader, _v, stats = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    print(f"[ab] data: {stats['source']}, {stats['total_tokens']:,} tokens")

    pairs = []
    for batch in loader:
        x, y = as_pair(batch)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= args.probe_batches + args.steps:
            break
    probes, batches = pairs[:args.probe_batches], pairs[args.probe_batches:]
    print(f"[ab] {len(probes)} held-out probes + {len(batches)} identical "
          f"training batches per arm")

    results: Dict[str, dict] = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for arm in args.arms:
        results[arm] = run_arm(arm, args, batches, probes, device, tok)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"issue": 146, "config": vars(args), "arms": results},
                      fh, indent=2)

    # ---- verdict ----------------------------------------------------------
    print("\n[ab] ============== group-wise ternary QAT ==============")
    print(f"[ab] {'arm':<16}{'group':>7}{'held-out':>11}{'bits/byte':>11}"
          f"{'bits/weight':>13}{'wall (min)':>11}")
    for name, r in results.items():
        b = r["bits_per_byte"]
        print(f"[ab] {name:<16}{r['group_size']:>7}{r['final_heldout_loss']:>11.5f}"
              f"{'nan' if b is None else f'{b:>11.4f}'}"
              f"{r['effective_bits_per_weight']:>13.3f}"
              f"{r['wall_clock_sec']/60:>11.1f}")

    verdict = None
    if "per_tensor" in results and "per_tensor_twin" in results:
        drift = abs(results["per_tensor"]["final_heldout_loss"]
                    - results["per_tensor_twin"]["final_heldout_loss"])
        print(f"\n[ab] KNOWN-ANSWER CELL — per_tensor vs its twin: {drift:.3e}")
        if drift > 0.0:
            print("[ab] *** RUN VOID: two identical arms diverged under "
                  "--deterministic, so the floor is not zero and no ranking "
                  "below can be separated from drift. Do NOT report these.")
            verdict = f"VOID (twin drift {drift:.3e})"

    if verdict is None and "per_tensor" in results:
        base = results["per_tensor"]
        print(f"\n[ab] {'arm':<16}{'vs per_tensor':>15}{'storage':>12}")
        wins = []
        for name in ("per_row", "group_128", "group_32"):
            if name not in results:
                continue
            r = results[name]
            d = r["final_heldout_loss"] - base["final_heldout_loss"]
            sx = r["effective_bits_per_weight"] / base["effective_bits_per_weight"]
            print(f"[ab] {name:<16}{d:>+15.5f}{sx:>11.3f}x")
            if d < 0:
                wins.append((d, name, sx))
        if wins:
            d, name, sx = min(wins)
            verdict = (f"{name} beats per-tensor by {-d:.5f} nats at {sx:.2f}x "
                       f"storage — judge the gain against that price")
        else:
            verdict = ("per-tensor WINS or ties: no grouped scheme improved "
                       "held-out loss, and every one of them costs more storage. "
                       "The lever is closed for training too, not just retrofit")
        print(f"\n[ab] VERDICT: {verdict}")
        print("[ab] LIMIT: one seed; 1,500 steps x 2,048 tokens = 3.07M, ~0.19% "
              "of a real pretrain. Bounds EARLY training at 83M only.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 146, "config": vars(args), "arms": results,
                   "verdict": verdict}, fh, indent=2)
    print(f"[ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
