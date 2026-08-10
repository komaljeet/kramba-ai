"""End-to-end check of the +37.7% batch-size win, through the real pipeline (Issue #141).

Author: Rambarun Komaljeet

`scripts/throughput_sweep.py` measured `batch 2 -> 4` at +37.7% using **synthetic
tokens and no dataloader**.  That is the compute-side ceiling.  The obvious way
for it to be wrong in practice is that the real loader cannot keep a doubled
batch fed, in which case the end-to-end gain is smaller -- or absent.

This script rebuilds the *real* continued-FT path to find out:

* the real corpus (`data/grounded_qa_expanded.jsonl`, 5,240 records),
* the real `GroundedQADataset` + `PackedDataset` packing (#132, on by default),
* the real training loop copied from `continue_ft_hybrid.continue_ft` --
  `doc_ids` plumbing, bf16 autocast, `clip_grad_norm_`, warmup, the same
  optimizer and accumulation structure.

**Both arms do identical work.**  Effective batch is 32 rows either way
(`b2 x ga16` vs `b4 x ga8`), so a fixed number of *optimizer* steps is a fixed
number of tokens: 32 x 512 = 16,384 per step, same gradient, same maths.  Wall
clock is therefore directly comparable.

**Falsifiability, asked before running:** if the loader *were* the bottleneck,
would this catch it?  Yes -- that is precisely the difference between this
harness and the synthetic one, and it is the only thing changed.  A result at or
near +37.7% means the loader keeps up; a result near 0% means it does not.  The
run also reports the arms' *loader wait* directly (time blocked on `next()`), so
a shortfall can be attributed rather than guessed at.

Arms are interleaved (A,B,A,B) so drift hits both equally, per the contention
finding in docs/state.md.

Usage::

    python scripts/confirm_batch_speedup.py --checkpoint <ckpt> --data <jsonl>
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
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from finetune_edge_conversational import GroundedQADataset, EOS_ID   # noqa: E402
from hvn.core.packing import PackedDataset                           # noqa: E402
from hvn.edge.model import EdgeHVN                                   # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model                  # noqa: E402

MIB = 2 ** 20


def build_model(ckpt: dict, device, attn_layers=(3, 10), attn_window=4096):
    """Real architecture at the checkpoint's config; weights fresh (timing only)."""
    num_layers = ckpt["num_layers"]
    plan = ["transformer" if i in set(attn_layers) else "deltanet"
            for i in range(num_layers)]
    model = EdgeHVN(
        vocab_size=ckpt["vocab_size"], d_model=ckpt["d_model"],
        num_layers=num_layers, n_heads=ckpt.get("n_heads", 8),
        use_deltanet=False,
        deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        block_plan=plan, attn_window=attn_window,
    )
    linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(
        model, target_names=[n for n in linears
                             if not any(p in n for p in exclude)])
    model = model.to(device)
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
            blk.use_megakernel_scan = True
    return model


def run_arm(model, dataset, *, device, batch_size, grad_accum, opt_steps,
            warmup_opt_steps, num_workers, seed):
    """Time `opt_steps` optimizer steps through the real loop.

    Mirrors continue_ft_hybrid.continue_ft: packed doc_ids, bf16 autocast,
    grad clipping, AdamW. Returns wall time excluding warmup, plus the share of
    it spent blocked waiting on the dataloader.
    """
    torch.manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        drop_last=True, pin_memory=True, num_workers=num_workers,
                        persistent_workers=bool(num_workers))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    data_iter = iter(loader)
    micro = 0
    step = 0
    loader_wait = 0.0
    t_start = None
    opt.zero_grad(set_to_none=True)

    while step < warmup_opt_steps + opt_steps:
        t_w = time.perf_counter()
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)
        if t_start is not None:
            loader_wait += time.perf_counter() - t_w

        if len(batch) == 3:
            x, y, doc_ids = batch
            doc_ids = doc_ids.to(device, non_blocking=True)
        else:
            x, y = batch
            doc_ids = None
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x) if doc_ids is None else model(x, doc_ids=doc_ids)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss = loss / grad_accum
        loss.backward()
        micro += 1

        if micro >= grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            micro = 0
            step += 1
            if step == warmup_opt_steps:
                # Start the clock only once compile/allocator/loader are warm.
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                loader_wait = 0.0
                t_start = time.perf_counter()

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t_start
    tokens = opt_steps * batch_size * grad_accum * x.size(1)
    del loader, data_iter
    return {
        "seconds": elapsed,
        "tokens": tokens,
        "tokens_per_sec": tokens / elapsed,
        "loader_wait_s": loader_wait,
        "loader_wait_pct": loader_wait / elapsed * 100.0,
        "peak_alloc_mib": torch.cuda.max_memory_allocated() / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / MIB,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--opt-steps", type=int, default=40,
                    help="timed optimizer steps per arm (identical tokens per arm)")
    ap.add_argument("--warmup-opt-steps", type=int, default=3)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--compile", action="store_true",
                    help="also run each arm under torch.compile (--fast)")
    ap.add_argument("--out", type=str,
                    default="results/throughput_sweep/real_run_confirm.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable -- docs/ai_rules.md requires GPU.")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    props = torch.cuda.get_device_properties(0)
    card_mib = props.total_memory / MIB
    print(f"GPU: {props.name} | {props.multi_processor_count} SMs | "
          f"{card_mib/1024:.4f} GiB")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    records = [json.loads(l) for l in open(args.data, encoding="utf-8")]
    print(f"corpus: {len(records)} records from {args.data}")
    ds = GroundedQADataset(tok, records, seq_len=args.seq_len, pad=False)
    ds = PackedDataset(ds.examples, seq_len=args.seq_len, pad_id=EOS_ID,
                       seed=0, strategy="shuffled_first_fit")

    # Both arms: effective batch 32 rows -> 16,384 tokens per optimizer step.
    arms = {"b2_ga16": (2, 16), "b4_ga8": (4, 8)}
    modes = [False] + ([True] if args.compile else [])
    samples: dict[str, list[dict]] = {}

    print(f"\nEffective batch 32 rows (16,384 tokens) per optimizer step in BOTH arms.")
    print(f"{args.opt_steps} timed optimizer steps each => identical token counts.\n")

    for compiled in modes:
        tag = "compiled" if compiled else "eager"
        for rnd in range(args.repeats):
            for name, (bs, ga) in arms.items():
                key = f"{name}_{tag}"
                model = build_model(ckpt, device)
                if compiled:
                    model = torch.compile(model)
                r = run_arm(model, ds, device=device, batch_size=bs, grad_accum=ga,
                            opt_steps=args.opt_steps,
                            warmup_opt_steps=args.warmup_opt_steps,
                            num_workers=args.num_workers, seed=rnd)
                r["spilled"] = r["peak_reserved_mib"] > card_mib
                samples.setdefault(key, []).append(r)
                print(f"  [{tag}] round {rnd+1}/{args.repeats}  {name:<9} "
                      f"{r['tokens_per_sec']:8.1f} tok/s  "
                      f"loader wait {r['loader_wait_pct']:5.1f}%  "
                      f"reserved {r['peak_reserved_mib']:7.1f} MiB"
                      f"{'  <-- SPILLED' if r['spilled'] else ''}")
                del model
                torch.cuda.empty_cache()
            print()

    def med(k, key="tokens_per_sec"):
        return statistics.median(s[key] for s in samples[k])

    print("=" * 74)
    results = {}
    for compiled in modes:
        tag = "compiled" if compiled else "eager"
        a, b = f"b2_ga16_{tag}", f"b4_ga8_{tag}"
        delta = (med(b) - med(a)) / med(a) * 100.0
        results[tag] = {
            "b2_ga16_tok_s": med(a), "b4_ga8_tok_s": med(b),
            "delta_pct": delta,
            "b2_loader_wait_pct": med(a, "loader_wait_pct"),
            "b4_loader_wait_pct": med(b, "loader_wait_pct"),
        }
        print(f"[{tag}] b2_ga16 {med(a):8.1f} tok/s  ->  b4_ga8 {med(b):8.1f} tok/s"
              f"   = {delta:+.1f}%")
        print(f"         loader wait: {med(a,'loader_wait_pct'):.1f}% (b2) vs "
              f"{med(b,'loader_wait_pct'):.1f}% (b4)")
    print("=" * 74)
    print("\nSynthetic-token claim to confirm: +37.7% (scripts/throughput_sweep.py).")
    print("A large shortfall with high loader wait = the dataloader is the limit.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({"issue": 141, "config": vars(args), "gpu": props.name,
                   "card_mib": card_mib, "samples": samples,
                   "results": results,
                   "synthetic_claim_pct": 37.7}, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
