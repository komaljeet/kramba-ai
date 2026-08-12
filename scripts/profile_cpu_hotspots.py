"""What CPU code actually runs during a training step, and is any of it worth rewriting? (#162)

Author: Rambarun Komaljeet

The claim "Python is not the bottleneck" has been asserted from a whole-run
number (#153: 97.2% GPU-busy). That is evidence, but it does not identify WHICH
host code runs, how long it takes, or whether any single function is worth
moving to a compiled language. This measures it directly.

Two profilers, because they answer different questions:

* ``cProfile`` -- pure-Python time by function. Anything expensive here is a
  genuine rewrite candidate.
* ``torch.profiler`` with CPU+CUDA -- separates *self CPU time* (host work) from
  *device time*. A PyTorch op with large self-CPU and small device time is a
  dispatch bottleneck; the reverse means the GPU is doing the work and the host
  is just issuing it.

**The decision rule, stated before running:** a function is worth rewriting in
Rust/C++ only if its self-CPU time is a meaningful fraction of step wall clock
AND it is not already dominated by a CUDA call it merely launches. Rewriting a
dispatcher that spends its time waiting on the GPU changes nothing -- which is
exactly what the whole-run number implies but does not prove.

Runs the REAL shape (seq 4096, grad_accum 8, grad-checkpoint on), because a
smaller shape would inflate Python's share and produce a flattering, wrong
answer -- the same error that produced a bogus +18% for torch.compile earlier.

Usage::

    python scripts/profile_cpu_hotspots.py --steps 2
"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN                    # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model   # noqa: E402
from hvn.optimizers import build_optimizer            # noqa: E402


def build(args, device):
    plan = ["transformer" if i in (4, 12) else "deltanet"
            for i in range(args.num_layers)]
    m = EdgeHVN(vocab_size=args.vocab_size, d_model=args.d_model,
                num_layers=args.num_layers, n_heads=args.n_heads,
                block_plan=plan, attn_window=4096, tie_embeddings=True,
                grad_checkpoint=True).to(device)
    lin = {n for n, x in m.named_modules() if isinstance(x, nn.Linear)}
    ex = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(m, target_names=[n for n in lin
                                            if not any(p in n for p in ex)])
    for b in m.blocks:
        if hasattr(b, "use_triton_scan"):
            b.use_triton_scan = True
            b.use_megakernel_scan = True
    return m


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--d-model", type=int, default=896)
    p.add_argument("--num-layers", type=int, default=16)
    p.add_argument("--n-heads", type=int, default=14)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--vocab-size", type=int, default=32003)
    p.add_argument("--steps", type=int, default=2)
    p.add_argument("--top", type=int, default=25)
    p.add_argument("--out", default="results/cpu_hotspots/profile.json")
    args = p.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA required")
    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    m = build(args, device)
    opt, _ = build_optimizer(m, "muon", lr=3e-4, weight_decay=0.01, muon_lr=0.02)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(0)

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

    print("warming up (kernels, allocator)...")
    one_step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.steps):
        one_step()
    wall = (time.perf_counter() - t0) / args.steps
    print(f"baseline: {wall:.2f} s/step (unprofiled)\n")

    # ---- 1. pure-Python time by function -------------------------------
    pr = cProfile.Profile()
    pr.enable()
    one_step()
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(args.top)
    text = s.getvalue()
    print("=== cProfile: top functions by SELF time (one step) ===")
    lines = [l for l in text.splitlines() if l.strip()]
    hdr = next((i for i, l in enumerate(lines) if l.strip().startswith("ncalls")), 0)
    for l in lines[hdr:hdr + args.top + 1]:
        print("  " + l)

    # ---- 2. host vs device time per op ---------------------------------
    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        one_step()
    ka = prof.key_averages()
    # self_cpu_time_total is host time NOT attributable to a child op -- the
    # quantity a rewrite could actually remove.
    rows = sorted(ka, key=lambda e: e.self_cpu_time_total, reverse=True)[:args.top]
    total_cpu = sum(e.self_cpu_time_total for e in ka) / 1e6
    print(f"\n=== torch.profiler: top ops by SELF CPU time "
          f"(host total {total_cpu:.2f}s vs {wall:.2f}s step) ===")
    print(f"  {'op':<42}{'self CPU s':>12}{'device s':>11}{'calls':>8}")
    out_rows = []
    for e in rows:
        scpu = e.self_cpu_time_total / 1e6
        dev = getattr(e, "self_device_time_total", 0) / 1e6
        out_rows.append({"op": e.key, "self_cpu_s": scpu,
                         "device_s": dev, "calls": e.count})
        print(f"  {e.key[:42]:<42}{scpu:>12.3f}{dev:>11.3f}{e.count:>8}")

    host_share = total_cpu / wall * 100
    print(f"\nhost CPU time is {host_share:.1f}% of step wall clock")
    if host_share < 15:
        verdict = (f"NOT WORTH REWRITING: host code accounts for {host_share:.1f}% "
                   f"of the step. Even eliminating ALL Python entirely would save "
                   f"less than that, because most of it is dispatch that overlaps "
                   f"with GPU execution. The step is GPU-bound.")
    elif host_share < 35:
        verdict = (f"MARGINAL: host code is {host_share:.1f}% of the step. Check the "
                   f"table above for a single dominant pure-Python function -- if "
                   f"the time is spread across dispatch, a rewrite buys nothing.")
    else:
        verdict = (f"WORTH INVESTIGATING: host code is {host_share:.1f}% of the "
                   f"step. Identify the top pure-Python entries above; those are "
                   f"real rewrite candidates.")
    print(f"\nVERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 162, "config": vars(args),
                   "step_wall_s": wall, "host_cpu_total_s": total_cpu,
                   "host_share_pct": host_share, "top_ops": out_rows,
                   "cprofile_text": text[:8000], "verdict": verdict}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
