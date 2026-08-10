"""Selective Language Modeling — train only on the tokens the model gets WRONG.

Rho-1 (arXiv:2404.07965) scores every token with a reference model trained on
high-quality data, then applies the training loss only to the top fraction by
*excess loss* (target loss minus reference loss) -- the tokens where the model is
much worse than the reference, i.e. the ones still worth learning. Reported: up to
10x faster to baseline accuracy on math, and +6.8% average across 15 benchmarks for
general-domain continual pretraining.

**Why this repo can test it honestly.** The method needs a reference model that is
the same base trained further on curated data. That is exactly what we have:
`hybrid_chat_v1.pt` IS `hybrid_a4_v2/hybrid_a4_step8000.pt` fine-tuned on the
curated OASST + grounded-QA corpora (docs/state.md, Issue #108). So target-init and
reference are the paper's relationship, not a stand-in for it.

**Two things this experiment must not get wrong.**

1. *Validation is scored on ALL tokens, for both arms.* Scoring only the selected
   tokens would rig the comparison toward the arm that chose them. The selection is
   a training-time intervention; the metric has to stay neutral.

2. *Validation loss may not be the metric that moves.* The paper's general-domain
   result is +6.8% on downstream benchmarks, NOT a perplexity win -- and SLM trains
   on a deliberately hard subset, so its loss over all tokens can legitimately end up
   flat or worse while downstream quality improves. Judging on val loss alone could
   produce a false negative. `eval/synthetic_recall_probe.py` is therefore run on
   both final checkpoints as the downstream reading, and both numbers are reported.

**Scale caveat, stated up front:** Rho-1 was evaluated at 1.1B and 7B. This is 83M.
Nothing establishes the effect survives an order of magnitude down, and this project
has already been burned once today by a technique whose benefit was scale-dependent
(multi-token prediction, worse than baseline below ~1B). A null result here is
therefore weak evidence about the method and strong evidence about this scale.

Arms are matched on STEPS and see identical batches in identical order:
    baseline   loss on every token (standard next-token prediction)
    selective  loss on the top --select-frac of tokens by excess loss

Cost: reference forward adds ~30% per step (forward only, no backward).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train_edge_conversational as tec  # noqa: E402
from train_edge_conversational import (  # noqa: E402
    GPT2_VOCAB_SIZE,
    load_checkpoint_for_resume,
    _save_edge_checkpoint,
)


def per_token_loss(model, x, y, autocast: bool = True):
    """[B,S] cross-entropy, one value per predicted token."""
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast):
        logits = model(x)
    return F.cross_entropy(
        logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1),
        reduction="none",
    ).view(y.shape)


@torch.no_grad()
def val_loss_all_tokens(model, batches, device) -> float:
    """Held-out loss over EVERY token — the neutral metric (see module docstring)."""
    model.eval()
    total, n = 0.0, 0
    for xb in batches:
        xb = xb.to(device)
        x, y = xb[:, :-1], xb[:, 1:]
        pt = per_token_loss(model, x, y)
        total += float(pt.sum().item()); n += pt.numel()
    model.train()
    return total / max(1, n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reference", default="checkpoints/hybrid_chat_v1.pt",
                    help="model trained on CURATED data; defines which tokens matter")
    ap.add_argument("--target-init", default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt",
                    help="the base the reference was fine-tuned FROM (paper's setup)")
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--select-frac", type=float, default=0.6,
                    help="Rho-1's default for ~1B models")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--val-batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-dir", default="data_cache",
                    help="tokenised corpus cache; point at the main tree from a worktree")
    ap.add_argument("--out", default="results/slm/ab_selective_tokens.json")
    args = ap.parse_args()

    device = torch.device("cuda")
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    # Identical batch sequence for both arms — materialised once. Without this the
    # arms differ by data order as well as by the intervention.
    torch.manual_seed(args.seed)
    loader, _v, _s = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    train_batches, val_batches = [], []
    need = args.steps + args.val_batches
    for b in loader:
        b = (b[0] if isinstance(b, (tuple, list)) else b).clone()
        (val_batches if len(val_batches) < args.val_batches else train_batches).append(b)
        if len(train_batches) >= args.steps:
            break
    print(f"[slm] {len(train_batches)} train batches + {len(val_batches)} held-out, "
          f"identical for both arms", flush=True)

    # Reference: frozen, eval, no grad. Never updated.
    ref, ref_step, _ = load_checkpoint_for_resume(args.reference, device)
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    print(f"[slm] reference {args.reference} (step {ref_step}) frozen", flush=True)

    results = {}
    for arm in ("baseline", "selective"):
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        model, init_step, _ = load_checkpoint_for_resume(args.target_init, device)
        model.train()
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        traj = [{"step": 0, "val_loss": val_loss_all_tokens(model, val_batches, device)}]
        print(f"\n[slm] === arm '{arm}' (init step {init_step}) "
              f"start val {traj[0]['val_loss']:.5f} ===", flush=True)

        kept_frac_log, t0 = [], time.time()
        for i, xb in enumerate(train_batches, start=1):
            xb = xb.to(device)
            x, y = xb[:, :-1], xb[:, 1:]

            tgt_pt = per_token_loss(model, x, y)          # [B,S], carries grad

            if arm == "selective":
                with torch.no_grad():
                    ref_pt = per_token_loss(ref, x, y)
                # Excess loss: how much worse the target is than the reference on
                # this token. High = still worth learning. Low/negative = already
                # matched (or the reference itself is bad here).
                excess = (tgt_pt - ref_pt).detach()
                k = max(1, int(args.select_frac * excess.numel()))
                thresh = torch.topk(excess.flatten(), k, largest=True).values[-1]
                mask = (excess >= thresh).float()
                loss = (tgt_pt * mask).sum() / mask.sum().clamp_min(1.0)
                kept_frac_log.append(float(mask.mean().item()))
            else:
                loss = tgt_pt.mean()

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if i % args.val_every == 0 or i == len(train_batches):
                v = val_loss_all_tokens(model, val_batches, device)
                traj.append({"step": i, "val_loss": v})
                print(f"[slm]   {arm:9s} step {i:5d}  val(all tokens) {v:.5f}  "
                      f"({(time.time()-t0)/60:.1f} min)", flush=True)

        ck = f"checkpoints/slm_{arm}/final.pt"
        os.makedirs(os.path.dirname(ck), exist_ok=True)
        _save_edge_checkpoint(model, ck, global_step=len(train_batches))
        results[arm] = {
            "trajectory": traj,
            "final_val_loss": traj[-1]["val_loss"],
            "checkpoint": ck,
            "mean_kept_frac": (sum(kept_frac_log) / len(kept_frac_log)) if kept_frac_log else 1.0,
            "minutes": round((time.time() - t0) / 60, 1),
        }
        del model, opt
        torch.cuda.empty_cache()

    b, s = results["baseline"]["final_val_loss"], results["selective"]["final_val_loss"]
    results["summary"] = {
        "baseline_final": b, "selective_final": s, "delta_nats": s - b,
        "select_frac": args.select_frac,
        "note": "val loss is over ALL tokens for both arms; the paper's general-domain "
                "gain was on downstream benchmarks, not perplexity -- run the recall "
                "probe on both checkpoints before concluding",
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}\n  RESULT (val loss over ALL tokens, lower is better)\n{'='*60}")
    print(f"  baseline   {b:.5f}")
    print(f"  selective  {s:.5f}   ({s-b:+.5f})")
    print(f"  selective kept {results['selective']['mean_kept_frac']:.1%} of tokens")
    print(f"\n  Next: eval/synthetic_recall_probe.py on both checkpoints — the paper's")
    print(f"  general-domain win was downstream, not perplexity.")
    print(f"  Wrote {args.out}")


if __name__ == "__main__":
    main()
