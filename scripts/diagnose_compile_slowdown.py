"""WHY is per-block compile 2.4x slower in the real loop? (#158 follow-up)

Author: Rambarun Komaljeet

A 4-step, 1-micro-batch bench said per-block compile was **+18%**. The real
training loop measured **20.72 s/step against 8.66 uncompiled -- 2.4x SLOWER**,
with GPU utilisation collapsing from 98% to 35%. So the bench measured a
configuration training never runs, and the question "why" is still open.

The obvious suspect is an interaction with the hand-written kernels: this
architecture's speed comes from the fused Triton scan (#83) and the
blockwise-recompute megakernel backward (#96). If Dynamo traces *through* a
block's forward and replaces or fragments those with generated code, compile
would be replacing a good kernel with a worse one -- which fits a starving GPU.

**The maintainer's hypothesis is the interesting one:** maybe the megakernel is
not needed WITH compile. That combination has never been measured. The grid
below tests it directly.

Every arm runs the REAL shape -- grad_accum 8, seq 4096, grad-checkpoint on --
because that is precisely what the earlier bench got wrong. Each runs in a fresh
process so Inductor cache state cannot leak between arms.

Arms (compile x kernels):

    eager_full        no compile, triton scan + megakernel   <- production today
    eager_triton      no compile, triton scan only
    eager_none        no compile, no fused kernels           <- what compile would replace
    compiled_full     per-block compile, triton + megakernel <- the 2.4x-slower config
    compiled_triton   per-block compile, triton only
    compiled_none     per-block compile, NO fused kernels    <- the untested hypothesis

Reports s/step, tokens/s and GPU utilisation sampled DURING the step, since the
35%-vs-98% gap is the clearest signal of what is actually wrong.

Usage::

    python scripts/diagnose_compile_slowdown.py
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import textwrap
import time

CHILD = textwrap.dedent(r'''
    import os, sys, json, time, threading, subprocess
    arm, seq_len, repo, steps, accum = (sys.argv[1], int(sys.argv[2]), sys.argv[3],
                                        int(sys.argv[4]), int(sys.argv[5]))
    sys.path.insert(0, repo)
    import torch, torch.nn as nn
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    V, D, L, H = 32003, 896, 16, 14

    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model
    from hvn.optimizers import build_optimizer

    compiled = arm.startswith("compiled")
    kernels  = arm.split("_")[1]          # full | triton | none

    plan = ["transformer" if i in (4, 12) else "deltanet" for i in range(L)]
    m = EdgeHVN(vocab_size=V, d_model=D, num_layers=L, n_heads=H, block_plan=plan,
                attn_window=4096, tie_embeddings=True, grad_checkpoint=True).to(dev)
    lin = {n for n, x in m.named_modules() if isinstance(x, nn.Linear)}
    ex = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(m, target_names=[n for n in lin if not any(p in n for p in ex)])
    for b in m.blocks:
        if hasattr(b, "use_triton_scan"):
            b.use_triton_scan = kernels in ("full", "triton")
            b.use_megakernel_scan = kernels == "full"

    if compiled:
        for i in range(len(m.blocks)):
            m.blocks[i] = torch.compile(m.blocks[i])

    opt, _ = build_optimizer(m, "muon", lr=3e-4, weight_decay=0.01, muon_lr=0.02)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(0)

    # Sample GPU utilisation DURING the timed steps: the 35% vs 98% gap is the
    # clearest evidence of what is wrong, and an average over the whole process
    # would be diluted by compile time.
    utils, sampling = [], True
    def sampler():
        while sampling:
            try:
                out = subprocess.run(["nvidia-smi","--query-gpu=utilization.gpu",
                                      "--format=csv,noheader,nounits"],
                                     capture_output=True, text=True, timeout=5)
                utils.append(int(out.stdout.strip().splitlines()[0]))
            except Exception: pass
            time.sleep(0.5)

    def one_step():
        for _ in range(accum):
            x = torch.randint(0, V, (1, seq_len), generator=g).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = m(x)
                loss = crit(logits.reshape(-1, V), torch.roll(x, -1, 1).reshape(-1)) / accum
            loss.backward()
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    one_step()                                  # warm / compile
    warm = time.perf_counter() - t0

    th = threading.Thread(target=sampler, daemon=True); th.start()
    per = []
    for s in range(steps):
        t0 = time.perf_counter(); one_step(); per.append(time.perf_counter() - t0)
        print(f"  step {s+1}: {per[-1]:.2f}s", flush=True)
    sampling = False; time.sleep(0.6)

    steady = sorted(per)[len(per)//2]
    print("RESULT " + json.dumps({
        "arm": arm, "ok": True, "warmup_s": warm, "steady_s": steady,
        "tokens_per_s": seq_len * accum / steady,
        "gpu_util_mean": (sum(utils)/len(utils)) if utils else -1,
        "gpu_util_max": max(utils) if utils else -1,
        "peak_vram_mib": torch.cuda.max_memory_reserved()/2**20}))
''')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--arms", nargs="+",
                   default=["eager_full", "eager_triton", "eager_none",
                            "compiled_full", "compiled_triton", "compiled_none"])
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--accum", type=int, default=8)
    p.add_argument("--steps", type=int, default=3)
    p.add_argument("--timeout", type=int, default=5400)
    p.add_argument("--repo", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--out", default="results/compile_diagnosis/results.json")
    args = p.parse_args()

    print(f"REAL shape: seq {args.seq_len} x accum {args.accum} = "
          f"{args.seq_len*args.accum:,} tokens/step, grad-checkpoint ON")
    print("production today = eager_full at 8.66 s/step\n")

    results = {}
    for arm in args.arms:
        print(f"=== {arm} ===", flush=True)
        t0 = time.perf_counter()
        try:
            pr = subprocess.run([sys.executable, "-c", CHILD, arm, str(args.seq_len),
                                 args.repo, str(args.steps), str(args.accum)],
                                capture_output=True, text=True, timeout=args.timeout)
            line = next((l for l in pr.stdout.splitlines()
                         if l.startswith("RESULT ")), None)
            if line:
                r = json.loads(line[len("RESULT "):]); results[arm] = r
                print(f"  {r['steady_s']:.2f} s/step | {r['tokens_per_s']:,.0f} tok/s "
                      f"| GPU {r['gpu_util_mean']:.0f}% (max {r['gpu_util_max']}%) "
                      f"| warmup {r['warmup_s']:.0f}s | VRAM {r['peak_vram_mib']:.0f}M\n")
            else:
                err = [l for l in pr.stderr.splitlines() if l.strip()]
                results[arm] = {"arm": arm, "ok": False,
                                "error": (err[-1] if err else "?")[:300]}
                print(f"  FAILED: {(err[-1] if err else '?')[:150]}\n")
        except subprocess.TimeoutExpired:
            results[arm] = {"arm": arm, "ok": False, "error": "timeout"}
            print(f"  TIMEOUT\n")

    ok = {k: v for k, v in results.items() if v.get("ok")}
    if ok:
        print(f"{'arm':<18}{'s/step':>9}{'tok/s':>10}{'GPU%':>7}{'VRAM':>8}")
        for k, r in sorted(ok.items(), key=lambda kv: kv[1]["steady_s"]):
            print(f"{k:<18}{r['steady_s']:>9.2f}{r['tokens_per_s']:>10,.0f}"
                  f"{r['gpu_util_mean']:>7.0f}{r['peak_vram_mib']:>7.0f}M")
        best = min(ok, key=lambda k: ok[k]["steady_s"])
        base = ok.get("eager_full")
        if base:
            print(f"\nfastest: {best} "
                  f"({base['steady_s']/ok[best]['steady_s']:.2f}x vs eager_full)")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 158, "config": vars(args), "results": results}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
