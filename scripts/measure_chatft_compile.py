"""Is torch.compile worth it in the chat-FT loop specifically? (Issue #143)

Author: Rambarun Komaljeet

#141 measured `torch.compile` at ~+39% -- but in `continue_ft_hybrid`'s loop, on
grounded-QA data.  #142 declined to turn it on in `finetune_edge_conversational`
on the grounds that a number measured in one loop is not a number measured in
another.  This script supplies the missing measurement.

**What is actually different here**, and therefore what could make the transfer
fail:

* different corpus (OpenAssistant conversations, not grounded-QA triples), so a
  different distribution of packed rows;
* different loss-mask density -- chat masks everything but assistant turns;
* a different `main()` that never previously touched `torch.compile`, so nothing
  guaranteed the traced path was compile-friendly at all.

Shapes are fixed at `seq_len` under packing (#132, on by default), so the
recompile-churn failure mode is not expected -- but "not expected" is the thing
being checked.  The run reports the compile count directly
(`torch._dynamo.utils.counters`), so churn shows up as a number rather than as a
mysteriously bad result.

**Falsifiability, asked before running:** if compile did NOT help here, would
this detect it?  Yes -- the arms are the same loop, the same data and the same
step count, differing only in the wrapper, and the eager-vs-eager noise floor
bounds what counts as a difference.  A result inside that floor means "no gain
in this loop", which is a publishable answer and the reason #142 refused to
guess.

Arms are interleaved and eager is entered twice under two names (the noise
floor).  Compile warmup is excluded from the timed window by construction --
timing starts after the warmup steps, by which point compilation has happened;
the compile counter confirms no further compiles occur inside the window.

Usage::

    python scripts/measure_chatft_compile.py --base-checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, ConcatDataset, WeightedRandomSampler

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune_edge_conversational import (                              # noqa: E402
    InstructionDataset, GroundedQADataset, build_oasst_conversations, EOS_ID,
)
from hvn.core.packing import PackedDataset                              # noqa: E402
from train_edge_conversational import load_checkpoint_for_resume        # noqa: E402

MIB = 2 ** 20


def compile_count() -> int:
    """How many times dynamo compiled a frame -- churn detector."""
    try:
        from torch._dynamo.utils import counters
        return int(sum(counters["frames"].values())) if "frames" in counters else 0
    except Exception:
        return -1


def time_arm(model, dataset, *, device, batch_size, grad_accum, opt_steps,
             warmup_opt_steps, compiled, seed):
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, pin_memory=True)
    train_model = torch.compile(model) if compiled else model
    opt = torch.optim.AdamW(train_model.parameters(), lr=1e-9, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    train_model.train()

    it = iter(loader)
    micro = step = 0
    t_start = None
    compiles_at_start = None
    opt.zero_grad(set_to_none=True)

    while step < warmup_opt_steps + opt_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if len(batch) == 3:
            x, y, doc_ids = batch
            doc_ids = doc_ids.to(device, non_blocking=True)
        else:
            x, y = batch
            doc_ids = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = (train_model(x) if doc_ids is None
                      else train_model(x, doc_ids=doc_ids))
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss = loss / grad_accum
        loss.backward()
        micro += 1

        if micro >= grad_accum:
            torch.nn.utils.clip_grad_norm_(train_model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            micro = 0
            step += 1
            if step == warmup_opt_steps:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                compiles_at_start = compile_count()
                t_start = time.perf_counter()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    tokens = opt_steps * batch_size * grad_accum * x.size(1)
    return {
        "seconds": elapsed,
        "tokens_per_sec": tokens / elapsed,
        "compiles_during_timed_window": compile_count() - (compiles_at_start or 0),
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-checkpoint", required=True)
    ap.add_argument("--grounded-replay", type=str, default=None)
    ap.add_argument("--grounded-replay-frac", type=float, default=0.15)
    ap.add_argument("--max-conversations", type=int, default=3000)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--opt-steps", type=int, default=30)
    ap.add_argument("--warmup-opt-steps", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--out", type=str,
                    default="results/throughput_sweep/chatft_compile.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable -- docs/ai_rules.md requires GPU.")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    props = torch.cuda.get_device_properties(0)
    print(f"GPU: {props.name} | card {props.total_memory/MIB:.0f} MiB")

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token

    convos = build_oasst_conversations(max_conversations=args.max_conversations)
    chat = InstructionDataset(tok, convos, seq_len=args.seq_len, pad=False)
    chat = PackedDataset(chat.examples, seq_len=args.seq_len, pad_id=EOS_ID,
                         seed=0, strategy="shuffled_first_fit")
    dataset = chat
    if args.grounded_replay:
        recs = [json.loads(l) for l in open(args.grounded_replay, encoding="utf-8")]
        g = GroundedQADataset(tok, recs, seq_len=args.seq_len, pad=False)
        g = PackedDataset(g.examples, seq_len=args.seq_len, pad_id=EOS_ID,
                          seed=0, strategy="shuffled_first_fit")
        dataset = ConcatDataset([chat, g])
        print(f"  mixed: {len(chat)} chat rows + {len(g)} grounded rows")

    # eager entered twice under two names -> the noise floor.
    arms = [("eager_a", False), ("compiled", True), ("eager_b", False)]
    samples: dict[str, list[dict]] = {}

    print(f"\n{args.opt_steps} timed optimizer steps/arm, batch {args.batch_size} "
          f"x grad_accum {args.grad_accum}, interleaved, {args.repeats} rounds.\n")

    for rnd in range(args.repeats):
        for name, compiled in arms:
            model, _, _ = load_checkpoint_for_resume(args.base_checkpoint, device)
            for blk in (model.blocks or []):
                if hasattr(blk, "use_triton_scan"):
                    blk.use_triton_scan = True
                    blk.use_megakernel_scan = True
            r = time_arm(model, dataset, device=device,
                         batch_size=args.batch_size, grad_accum=args.grad_accum,
                         opt_steps=args.opt_steps,
                         warmup_opt_steps=args.warmup_opt_steps,
                         compiled=compiled, seed=rnd)
            samples.setdefault(name, []).append(r)
            print(f"  round {rnd+1}/{args.repeats}  {name:<9} "
                  f"{r['tokens_per_sec']:8.1f} tok/s  "
                  f"reserved {r['peak_reserved_mib']:7.1f} MiB  "
                  f"compiles in window: {r['compiles_during_timed_window']}")
            del model
            torch.cuda.empty_cache()
        print()

    def med(n):
        return statistics.median(s["tokens_per_sec"] for s in samples[n])

    base = med("eager_a")
    noise = abs(med("eager_b") - base) / base * 100.0
    delta = (med("compiled") - base) / base * 100.0
    churn = max(s["compiles_during_timed_window"] for s in samples["compiled"])

    print("=" * 70)
    print(f"eager      {base:8.1f} tok/s")
    print(f"compiled   {med('compiled'):8.1f} tok/s   = {delta:+.1f}%")
    print(f"noise floor (eager vs itself): {noise:.2f}%")
    print(f"recompiles inside the timed window: {churn}")
    print("=" * 70)

    if churn > 0:
        verdict = (f"RECOMPILE CHURN ({churn} in-window) -- the timed number is "
                   f"contaminated; investigate before trusting it")
    elif abs(delta) <= noise:
        verdict = (f"NO GAIN in this loop ({delta:+.1f}% is inside the "
                   f"{noise:.2f}% noise floor) -- do NOT default it on")
    elif delta > 0:
        verdict = f"REAL GAIN of {delta:.1f}% (noise floor {noise:.2f}%) -- default it on"
    else:
        verdict = f"SLOWER by {abs(delta):.1f}% -- leave it off"
    print(f"\nVerdict: {verdict}")
    print(f"For reference, #141 measured +39% in continue_ft_hybrid's loop.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"issue": 143, "config": vars(args), "gpu": props.name,
                   "samples": samples, "eager_tok_s": base,
                   "compiled_tok_s": med("compiled"), "delta_pct": delta,
                   "noise_floor_pct": noise,
                   "recompiles_in_window": churn,
                   "verdict": verdict,
                   "reference_141_pct": 39.0}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
