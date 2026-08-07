"""A/B the real wall-clock cost of padded vs packed fine-tuning (Issue #132).

Reports time per *real* (non-padding) token, which is the only number that
answers "does packing actually make training cheaper". Rows/sec would flatter
packing (each packed row does ~5x more real work); tokens/sec would flatter
padding (padding tokens are cheap but worthless).

Uses the real 83M hybrid configuration (12 layers, 2 windowed-attention layers,
d=512, ternary linears, triton + megakernel scan) and the real grounded-QA
corpus, at fresh init -- throughput does not depend on trained weights, and this
avoids needing the LFS checkpoint.

Usage::

    python scripts/ab_packing.py --steps 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from finetune_edge_conversational import EOS_ID, GroundedQADataset
from hvn.core.packing import PackedDataset
from hvn.edge.model import EdgeHVN
from hvn.edge.ternary import apply_ternary_to_model

GPT2_VOCAB_SIZE = 50257


def build_hybrid(device, num_layers=12, d_model=512, n_heads=4, attn_window=4096):
    """The hybrid FT config: 10 DeltaNet + 2 windowed-attention layers."""
    attn_layers = [num_layers // 2 - 1, num_layers - 1]
    plan = ["transformer" if i in attn_layers else "deltanet" for i in range(num_layers)]
    model = EdgeHVN(
        vocab_size=GPT2_VOCAB_SIZE, d_model=d_model, num_layers=num_layers,
        n_heads=n_heads, use_deltanet=False, deltanet_conv_kernel=4,
        block_plan=plan, attn_window=attn_window,
    )
    all_linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(
        model, target_names=[n for n in all_linears if not any(p in n for p in exclude)]
    )
    model = model.to(device)
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
            blk.use_megakernel_scan = True
    return model


def run(model, loader, device, steps, packed):
    crit = nn.CrossEntropyLoss(ignore_index=-100)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-5)
    model.train()

    it = iter(loader)
    real_tokens = 0
    n = 0
    t0 = None
    while n < steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        if packed:
            x, y, doc = batch
            doc = doc.to(device)
        else:
            x, y = batch
            doc = None
        x, y = x.to(device), y.to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x) if doc is None else model(x, doc_ids=doc)
            loss = crit(logits.reshape(-1, GPT2_VOCAB_SIZE), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad()

        n += 1
        if n == 5:                      # discard warmup/compile
            torch.cuda.synchronize()
            t0 = time.time()
            real_tokens = 0
        if n > 5:
            # A "real" token is one the loss can actually learn from.
            real_tokens += int((y != -100).sum())

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    measured = steps - 5
    return {
        "steps_measured": measured,
        "seconds": elapsed,
        "s_per_step": elapsed / measured,
        "real_tokens": real_tokens,
        "real_tokens_per_step": real_tokens / measured,
        "us_per_real_token": elapsed / max(real_tokens, 1) * 1e6,
        "peak_gib": torch.cuda.max_memory_allocated() / 1024 ** 3,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grounded-data", default="data/grounded_qa_expanded.jsonl")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--out", default="results/packing/ab_packing.json")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "this benchmark needs the GPU"
    device = torch.device("cuda")

    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")
    records = [json.loads(l) for l in open(args.grounded_data, encoding="utf-8")]

    padded = GroundedQADataset(tok, records, seq_len=args.seq_len)
    unpadded = GroundedQADataset(tok, records, seq_len=args.seq_len, pad=False)
    packed = PackedDataset(unpadded.examples, seq_len=args.seq_len, pad_id=EOS_ID)

    results = {"packing_stats": packed.stats}
    for label, ds, is_packed in (("padded", padded, False), ("packed", packed, True)):
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=True)
        torch.manual_seed(0)
        model = build_hybrid(device)
        torch.cuda.reset_peak_memory_stats(device)
        r = run(model, loader, device, args.steps, is_packed)
        results[label] = r
        print(f"\n{label}: {r['s_per_step']:.3f}s/step | "
              f"{r['real_tokens_per_step']:.0f} real tokens/step | "
              f"{r['us_per_real_token']:.2f} us/real token | "
              f"peak {r['peak_gib']:.2f} GiB")
        del model
        torch.cuda.empty_cache()

    speedup = results["padded"]["us_per_real_token"] / results["packed"]["us_per_real_token"]
    results["speedup_per_real_token"] = speedup
    print(f"\n==> packing is {speedup:.2f}x cheaper per real token")
    print(f"    (row reduction was {packed.stats['row_reduction']:.2f}x -- the gap "
          f"is the explicit attention mask + longer effective sequences)")

    import os
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"    wrote {args.out}")


if __name__ == "__main__":
    main()
