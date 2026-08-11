"""How much does starting the recurrent state at zero actually cost? (Issue #150 gate)

Author: Rambarun Komaljeet

Warm-starting the DeltaNet state would mean giving the Triton scan an initial
state it does not currently accept (``_triton_gated_delta_scan(q,k,v,alpha,beta)``
has no ``S0``; the eager paths hardcode ``S_state = torch.zeros`` at
``deltanet.py:396,439,534``). That is a forward-and-backward kernel change to the
most correctness-critical code in this repo -- hours of work, and the docs record
this scan going silently wrong before.

**So measure the premise first.** The whole idea rests on "every sequence pays to
escape ``S = 0``". If it does not -- if the state becomes useful within a handful
of tokens -- there is nothing to recover and no kernel needs touching.

This measures **loss as a function of position in the sequence** on the trained
model. Position 0 has an empty state by construction; position 400 has 400 tokens
of accumulated state. The gap between them is an upper bound on what any
warm-start could buy.

**Why it is an upper bound, not an estimate:** early positions are also harder for
reasons that have nothing to do with the recurrence -- less context is genuinely
less information, and a perfect oracle would still do worse at position 0 than at
position 400. So the measured gap *over-states* the warm-start opportunity. If
even this optimistic number is small, the idea is dead; if it is large, it is
suggestive but confounded, and only the real experiment settles it.

Data is contiguous pretrain windows (not packed QA), so position-in-row is
position-since-state-reset.

Usage::

    python scripts/probe_state_warmup_cost.py --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ab_prodigy import as_pair, enable_scan_flags, require_cuda  # noqa: E402


@torch.no_grad()
def per_position_loss(model, batches, device, seq_len: int) -> torch.Tensor:
    """Mean loss at each position index, averaged over batches."""
    tot = torch.zeros(seq_len, device=device, dtype=torch.float64)
    n = 0
    model.eval()
    for bx, by in batches:
        x, y = bx.to(device), by.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
        ls = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
            reduction="none").view(y.shape)          # [B, S]
        tot += ls.double().mean(dim=0)[:seq_len]
        n += 1
    return tot / max(1, n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint",
                   default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt")
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--batches", type=int, default=48)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results/state_warmstart/position_cost.json")
    args = p.parse_args()

    device = require_cuda()
    import train_edge_conversational as tec
    tok = tec.load_tokenizer(args.tokenizer)
    torch.manual_seed(args.seed)
    loader, _v, stats = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    batches = []
    for b in loader:
        x, y = as_pair(b)
        batches.append((x.clone(), y.clone()))
        if len(batches) >= args.batches:
            break

    model, step, _ = tec.load_checkpoint_for_resume(args.checkpoint, device)
    enable_scan_flags(model)
    print(f"[probe] {args.checkpoint} (step {step}), {len(batches)} batches, "
          f"seq_len {args.seq_len}")

    pos = per_position_loss(model, batches, device, args.seq_len)

    buckets = [(0, 1), (1, 4), (4, 16), (16, 64), (64, 128), (128, 256),
               (256, 384), (384, 512)]
    print(f"\n[probe] {'positions':>12}{'mean loss':>12}{'vs tail':>10}")
    tail = float(pos[384:512].mean())
    rows = {}
    for a, b in buckets:
        m = float(pos[a:b].mean())
        rows[f"{a}-{b}"] = m
        print(f"[probe] {f'{a}-{b}':>12}{m:>12.4f}{m - tail:>+10.4f}")

    first16 = float(pos[:16].mean())
    gap = first16 - tail
    # Cost per 512-token sequence of the first 16 positions being worse than the
    # tail, expressed as a fraction of total sequence loss -- the honest units.
    total = float(pos.mean())
    share = (gap * 16 / args.seq_len) / total * 100

    print(f"\n[probe] first 16 positions: {first16:.4f}   tail (384-512): {tail:.4f}")
    print(f"[probe] gap {gap:+.4f} nats, which is {share:.2f}% of total sequence loss")
    if gap <= 0:
        verdict = ("DEAD: early positions are not worse than late ones, so there is "
                   "no zero-state penalty to recover. Do not touch the kernel.")
    elif share < 0.5:
        verdict = (f"DEAD: the entire early-position penalty is {share:.2f}% of "
                   f"sequence loss, and that is an UPPER BOUND (early tokens are "
                   f"genuinely harder regardless of state). A warm start cannot "
                   f"recover enough to justify a Triton forward+backward change.")
    else:
        verdict = (f"WORTH BUILDING: early positions cost {share:.2f}% of sequence "
                   f"loss. Still an upper bound -- less context is genuinely less "
                   f"information -- but large enough that the real A/B is warranted.")
    print(f"\n[probe] VERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 150, "checkpoint": args.checkpoint, "step": step,
                   "buckets": rows, "first16": first16, "tail": tail,
                   "gap_nats": gap, "share_of_sequence_loss_pct": share,
                   "per_position": [float(v) for v in pos],
                   "verdict": verdict}, fh, indent=2)
    print(f"[probe] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
