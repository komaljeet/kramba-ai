"""Does n_heads 14 -> 28 speed up the 157M code model? (#159)

Author: Rambarun Komaljeet

#112 measured `--n-heads 8 -> 16` at **2.4x** on the 355M config, with peak VRAM
going DOWN (3.43 -> 3.36 GiB): head count is a reshape, not new tensors, so
q/k/v stay d_model-wide and only the Triton scan's program count changes
(`N = B x H` against this card's 20 SMs).

At batch 1 x 14 heads that is **14 programs for 20 SMs** -- structurally
underfilled, even though nvidia-smi reports 98% (which counts "any SM busy",
not "all SMs busy").

Two reasons this might not reproduce here, stated before running:

* #141 partly REFUTED the pure-occupancy story -- equal `B x H` gave 55%
  different throughput, so program count is not the whole picture.
* d_head would drop 64 -> 32. #144 measured head count as quality-neutral at
  d_head 64; nobody has tested 32, so a speed win here is not automatically a
  free win.

Runs at the REAL training shape (seq 4096, grad_accum 8, grad-checkpoint on)
because a 1-micro-batch microbenchmark is exactly what produced the wrong +18%
compile number earlier today. Samples GPU utilisation during the timed steps.

Usage::

    python scripts/bench_head_count.py --heads 14 28
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import threading
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN                    # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model   # noqa: E402
from hvn.optimizers import build_optimizer            # noqa: E402


def bench(heads: int, args, device) -> dict:
    torch.manual_seed(0)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    plan = ["transformer" if i in (4, 12) else "deltanet"
            for i in range(args.num_layers)]
    m = EdgeHVN(vocab_size=args.vocab_size, d_model=args.d_model,
                num_layers=args.num_layers, n_heads=heads, block_plan=plan,
                attn_window=4096, tie_embeddings=True,
                grad_checkpoint=True).to(device)
    lin = {n for n, x in m.named_modules() if isinstance(x, nn.Linear)}
    ex = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(m, target_names=[n for n in lin
                                            if not any(p in n for p in ex)])
    for b in m.blocks:
        if hasattr(b, "use_triton_scan"):
            b.use_triton_scan = True
            b.use_megakernel_scan = True

    opt, _ = build_optimizer(m, "muon", lr=3e-4, weight_decay=0.01,
                             muon_lr=args.muon_lr)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(1)

    utils, running = [], [True]

    def sample():
        while running[0]:
            try:
                o = subprocess.run(
                    ["nvidia-smi", "--query-gpu=utilization.gpu",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5)
                utils.append(int(o.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.4)

    def one_step():
        for _ in range(args.grad_accum):
            x = torch.randint(0, args.vocab_size, (1, args.seq_len),
                              generator=g).to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = m(x)
                loss = crit(logits.reshape(-1, args.vocab_size),
                            torch.roll(x, -1, 1).reshape(-1)) / args.grad_accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    one_step()                                   # warm
    threading.Thread(target=sample, daemon=True).start()
    per = []
    for _ in range(args.steps):
        t0 = time.perf_counter()
        one_step()
        per.append(time.perf_counter() - t0)
    running[0] = False
    time.sleep(0.5)

    r = {"heads": heads, "d_head": args.d_model // heads,
         "params": sum(p.numel() for p in m.parameters()),
         "s_step": sorted(per)[len(per) // 2],
         "vram_mib": torch.cuda.max_memory_reserved() / 2 ** 20,
         "gpu_pct": (sum(utils) / len(utils)) if utils else -1,
         "per_step": per}
    del m, opt
    torch.cuda.empty_cache()
    return r


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--heads", type=int, nargs="+", default=[14, 28])
    p.add_argument("--d-model", type=int, default=896)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--vocab-size", type=int, default=32003)
    p.add_argument("--muon-lr", type=float, default=0.02)
    p.add_argument("--out", default="results/head_count/bench.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    toks = args.seq_len * args.grad_accum
    print(f"157M hybrid, REAL shape: seq {args.seq_len} x accum {args.grad_accum} "
          f"= {toks:,} tokens/step, grad-checkpoint ON")
    print(f"production today: 14 heads at 8.66 s/step\n")

    print(f"{'heads':>6}{'d_head':>8}{'params':>13}{'s/step':>9}"
          f"{'tok/s':>9}{'VRAM':>9}{'GPU%':>7}")
    out = []
    for h in args.heads:
        if args.d_model % h:
            print(f"{h:>6}  skipped: d_model {args.d_model} not divisible")
            continue
        dh = args.d_model // h
        if dh & (dh - 1):
            # deltanet_triton.py does tl.arange(0, D) over the head dim, which
            # Triton requires to be a power of 2 (see check_head_dim).
            print(f"{h:>6}  skipped: d_head {dh} is not a power of 2")
            continue
        r = bench(h, args, device)
        out.append(r)
        print(f"{r['heads']:>6}{r['d_head']:>8}{r['params']:>13,}"
              f"{r['s_step']:>9.2f}{toks/r['s_step']:>9,.0f}"
              f"{r['vram_mib']:>8.0f}M{r['gpu_pct']:>7.0f}")

    if len(out) >= 2:
        a, b = out[0], out[-1]
        sp = a["s_step"] / b["s_step"]
        print(f"\n{b['heads']} heads vs {a['heads']}: {sp:.3f}x "
              f"({(sp-1)*100:+.1f}%)")
        print(f"VRAM delta {b['vram_mib']-a['vram_mib']:+.0f} MiB | "
              f"params delta {b['params']-a['params']:+,}")
        if sp > 1.05:
            print(f"\nWORTH IT on speed -- but d_head drops "
                  f"{a['d_head']} -> {b['d_head']}, and #144 only measured head "
                  f"count as quality-neutral at d_head 64. A fresh pretrain "
                  f"would need a quality A/B before adopting this.")
        else:
            print(f"\nNO GAIN: {sp:.3f}x. #112's 2.4x at 355M does not "
                  f"reproduce at this size -- consistent with #141 refuting the "
                  f"pure `B x H` occupancy model.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 159, "config": vars(args), "results": out}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
