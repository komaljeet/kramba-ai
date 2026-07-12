"""Matched training-step-time benchmark: original Hamiltonian HVN vs DeltaNet vs
DeltaNet+fast vs the hybrid vs a real Transformer — all at the SAME d/L/seq/batch.

Answers "how much slower does HVN actually train, and how much do the DeltaNet
Triton/megakernel optimizations buy?" with fresh numbers on THIS GPU, replacing
stale recollections. Each config is embedding -> N stacked real blocks -> RMSNorm
-> tied head; we time forward + loss + backward + optimizer.step (a full training
step), median over `--steps` after `--warmup` (so torch.compile / cudnn autotune
warmup is excluded). All configs run under bf16 autocast + TF32 (what you'd
actually train with), so the only variable is the block.

    python scripts/bench_train_steptime.py --d-model 512 --n-layers 12 \
        --n-heads 8 --seq-len 512 --batch 4 --steps 30 --warmup 12
"""

from __future__ import annotations

import argparse
import os
import statistics
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from hvn.core.hvn_block import HVNBlock, RMSNorm            # noqa: E402
from hvn.core.deltanet import DeltaNetBlock                 # noqa: E402
from hvn.core.transformer_block import TransformerBlock     # noqa: E402


def build_blocks(cfg: str, d_model, n_layers, n_heads):
    """Return a ModuleList of `n_layers` blocks for the named config."""
    blocks = []
    for i in range(n_layers):
        if cfg == "transformer":
            blocks.append(TransformerBlock(d_model, n_heads=n_heads))
        elif cfg == "hvn":  # original Hamiltonian block (pre-#34)
            blocks.append(HVNBlock(d_model, n_heads=n_heads, chunk_size=64))
        elif cfg == "deltanet":
            blocks.append(DeltaNetBlock(d_model, n_heads=n_heads))
        elif cfg == "deltanet_fast":
            blocks.append(DeltaNetBlock(d_model, n_heads=n_heads,
                                        use_triton_scan=True, use_megakernel_scan=True))
        elif cfg == "hybrid":  # what we actually train: 2 attn @ 3,10 + DeltaNet-fast
            if i in (3, 10):
                blocks.append(TransformerBlock(d_model, n_heads=n_heads))
            else:
                blocks.append(DeltaNetBlock(d_model, n_heads=n_heads,
                                            use_triton_scan=True, use_megakernel_scan=True))
        else:
            raise ValueError(cfg)
    return nn.ModuleList(blocks)


class StackModel(nn.Module):
    def __init__(self, cfg, vocab, d_model, n_layers, n_heads):
        super().__init__()
        self.embed = nn.Embedding(vocab, d_model)
        self.blocks = build_blocks(cfg, d_model, n_layers, n_heads)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, vocab, bias=False)
        self.head.weight = self.embed.weight  # tie

    def forward(self, ids):
        x = self.embed(ids)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.norm(x))


def time_config(cfg, args, device):
    torch.manual_seed(0)
    model = StackModel(cfg, args.vocab, args.d_model, args.n_layers, args.n_heads).to(device)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    params = sum(p.numel() for p in model.parameters())
    ids = torch.randint(0, args.vocab, (args.batch, args.seq_len + 1), device=device)
    x, y = ids[:, :-1], ids[:, 1:]

    times = []
    for step in range(args.warmup + args.steps):
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, args.vocab).float(), y.reshape(-1))
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        if step >= args.warmup:
            times.append(dt)
    med = statistics.median(times)
    return {"cfg": cfg, "params": params, "s_step": med, "steps_s": 1.0 / med}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--n-layers", type=int, default=12)
    ap.add_argument("--n-heads", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--vocab", type=int, default=8000)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=12)
    ap.add_argument("--configs", default="transformer,hvn,deltanet,deltanet_fast,hybrid")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device("cuda")

    print(f"config: d={args.d_model} L={args.n_layers} H={args.n_heads} "
          f"seq={args.seq_len} batch={args.batch} vocab={args.vocab} | "
          f"bf16 autocast + TF32 | median of {args.steps} steps (warmup {args.warmup})\n")

    results = []
    for cfg in args.configs.split(","):
        try:
            r = time_config(cfg, args, device)
            results.append(r)
            print(f"  {cfg:15s} done: {r['s_step']*1000:7.1f} ms/step")
        except Exception as e:  # a config may OOM or lack triton — report, continue
            print(f"  {cfg:15s} FAILED: {type(e).__name__}: {e}")

    base = next((r for r in results if r["cfg"] == "transformer"), None)
    print("\n" + "=" * 68)
    print(f"{'config':16s} {'params':>12s} {'ms/step':>10s} {'steps/s':>9s} {'vs Tf':>8s}")
    print("-" * 68)
    for r in results:
        ratio = (r["s_step"] / base["s_step"]) if base else float("nan")
        print(f"{r['cfg']:16s} {r['params']:>12,d} {r['s_step']*1000:>10.1f} "
              f"{r['steps_s']:>9.2f} {ratio:>7.2f}x")
    print("=" * 68)
    if base:
        print("vs Tf = training-step time relative to the Transformer (1.00x = same; "
              "2.00x = twice as slow).")


if __name__ == "__main__":
    main()
