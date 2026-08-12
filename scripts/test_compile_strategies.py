"""Make torch.compile work on the 157M model (#158).

Author: Rambarun Komaljeet

Whole-model `torch.compile` dies three times over with

    torch._inductor.exc.InductorError: MemoryError
      scheduler.py compute_ancestors: ancestors |= name_to_ancestors[dep_node_name]

on 15.3 GB of system RAM -- at seq 4096 AND 2048, with and without a browser
resident, on a fresh boot with 6.9 GB free. Not a memory-pressure problem: it
failed with ~24 GB of commit headroom left.

**The mechanism**, from the source: `compute_ancestors` keeps a COMPLETE
ancestor set for EVERY node alive at once (`name_to_ancestors`), so its memory
is O(N^2) in graph nodes. 16 layers with gradient-checkpointed recomputation
makes N large enough that N^2 strings do not fit.

So the fix is to reduce N, and the strategies below do that different ways:

``whole``            baseline -- expected to die, included so the comparison is real
``per_block``        compile each of the 16 blocks separately. 16 graphs of N/16
                     nodes cost 16*(N/16)^2 = N^2/16 -- a 16x memory reduction --
                     while keeping nearly all the fusion benefit, since fusion
                     opportunities are within blocks, not across them.
``low_threshold``    lower `realize_opcount_threshold` so buffers are realized
                     sooner, cutting nodes per fusion group.
``no_epilogue``      disable epilogue fusion: fewer fusion candidates, smaller graph.
``per_block_low``    per-block AND the lowered threshold.

Each strategy runs in a FRESH process, because a failed Inductor compile leaves
allocator and cache state behind that would flatter or poison the next attempt.
Each reports peak RSS, compile wall clock, and steady-state step time -- so a
strategy that "works" but is slower than the uncompiled 3,767 tok/s is visible as
such rather than counted as a win.

Usage::

    python scripts/test_compile_strategies.py --strategies per_block low_threshold
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
    import os, sys, json, time, gc
    strategy, seq_len, repo, steps = sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4])
    sys.path.insert(0, repo)
    import torch, torch.nn as nn
    from torch._inductor import config as ic

    if strategy in ("low_threshold", "per_block_low"):
        # Realize buffers far sooner: fewer ops accumulate into each fusion
        # group, so the node count -- and thus the N^2 ancestor memory -- drops.
        ic.realize_opcount_threshold = 8
        ic.max_fusion_size = 16
    if strategy == "no_epilogue":
        ic.epilogue_fusion = False
        ic.realize_opcount_threshold = 8

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    V, D, L, H = 32003, 896, 16, 14

    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model
    from hvn.optimizers import build_optimizer

    plan = ["transformer" if i in (4, 12) else "deltanet" for i in range(L)]
    m = EdgeHVN(vocab_size=V, d_model=D, num_layers=L, n_heads=H, block_plan=plan,
                attn_window=4096, tie_embeddings=True, grad_checkpoint=True).to(dev)
    lin = {n for n, x in m.named_modules() if isinstance(x, nn.Linear)}
    ex = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(m, target_names=[n for n in lin if not any(p in n for p in ex)])
    for b in m.blocks:
        if hasattr(b, "use_triton_scan"):
            b.use_triton_scan = True; b.use_megakernel_scan = True

    if strategy in ("per_block", "per_block_low"):
        # Compile each block IN PLACE. The model keeps its own forward (so
        # grad_checkpoint, block_plan and the doc_ids path are untouched) but
        # every block is an OptimizedModule -- 16 small graphs instead of one
        # enormous one.
        for i, b in enumerate(m.blocks):
            m.blocks[i] = torch.compile(b)
        train_model = m
    elif strategy == "eager":
        train_model = m
    else:
        train_model = torch.compile(m)

    opt, _desc = build_optimizer(m, "muon", lr=3e-4, weight_decay=0.01, muon_lr=0.02)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(0)
    peak_rss = 0.0
    try:
        import psutil; proc = psutil.Process()
    except Exception:
        proc = None

    per_step = []
    for s in range(steps):
        t0 = time.perf_counter()
        x = torch.randint(0, V, (1, seq_len), generator=g).to(dev)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = train_model(x)
            loss = crit(logits.reshape(-1, V), torch.roll(x, -1, 1).reshape(-1))
        loss.backward()
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        per_step.append(time.perf_counter() - t0)
        if proc is not None:
            peak_rss = max(peak_rss, proc.memory_info().rss / 2**20)
        print(f"  step {s+1}: {per_step[-1]:.2f}s", flush=True)

    steady = min(per_step[1:]) if len(per_step) > 1 else per_step[0]
    print("RESULT " + json.dumps({
        "strategy": strategy, "ok": True, "first_step_s": per_step[0],
        "steady_step_s": steady, "tokens_per_s": seq_len / steady,
        "peak_rss_mib": peak_rss,
        "peak_vram_mib": torch.cuda.max_memory_reserved() / 2**20}))
''')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--strategies", nargs="+",
                   default=["eager", "per_block", "low_threshold",
                            "per_block_low", "no_epilogue", "whole"])
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--steps", type=int, default=4)
    p.add_argument("--timeout", type=int, default=3600)
    p.add_argument("--repo", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--out", default="results/compile_strategies/results.json")
    args = p.parse_args()

    print(f"157M hybrid, seq {args.seq_len}, {args.steps} steps per strategy")
    print("whole-model compile is expected to MemoryError -- it is the baseline\n")

    results = {}
    for strat in args.strategies:
        print(f"=== {strat} ===", flush=True)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [sys.executable, "-c", CHILD, strat, str(args.seq_len),
                 args.repo, str(args.steps)],
                capture_output=True, text=True, timeout=args.timeout)
            wall = time.perf_counter() - t0
            line = next((l for l in proc.stdout.splitlines()
                         if l.startswith("RESULT ")), None)
            if line:
                r = json.loads(line[len("RESULT "):])
                r["process_wall_s"] = wall
                results[strat] = r
                print(f"  OK  steady {r['steady_step_s']:.2f}s/step  "
                      f"first {r['first_step_s']:.1f}s  "
                      f"peak RSS {r['peak_rss_mib']:.0f} MiB  "
                      f"({wall:.0f}s total)\n")
            else:
                err = [l for l in proc.stderr.splitlines() if l.strip()]
                why = next((l for l in reversed(err)
                            if "Error" in l or "error" in l), err[-1] if err else "?")
                results[strat] = {"strategy": strat, "ok": False,
                                  "error": why[:300], "process_wall_s": wall}
                print(f"  FAILED after {wall:.0f}s: {why[:160]}\n")
        except subprocess.TimeoutExpired:
            results[strat] = {"strategy": strat, "ok": False,
                              "error": f"timeout >{args.timeout}s"}
            print(f"  TIMEOUT (>{args.timeout}s)\n")

    ok = {k: v for k, v in results.items() if v.get("ok")}
    if ok:
        base = ok.get("eager")
        print(f"{'strategy':<16}{'s/step':>9}{'vs eager':>10}{'peak RSS':>11}")
        for k, r in sorted(ok.items(), key=lambda kv: kv[1]["steady_step_s"]):
            rel = (f"{base['steady_step_s']/r['steady_step_s']:.2f}x"
                   if base else "-")
            print(f"{k:<16}{r['steady_step_s']:>9.2f}{rel:>10}"
                  f"{r['peak_rss_mib']:>10.0f}M")
        best = min(ok, key=lambda k: ok[k]["steady_step_s"])
        if base and best != "eager":
            gain = base["steady_step_s"] / ok[best]["steady_step_s"]
            print(f"\nBEST: {best} at {gain:.2f}x eager "
                  f"({(gain-1)*100:+.1f}%)")
        elif base:
            print("\nNo compiled strategy beat eager.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 158, "config": vars(args), "results": results},
                  fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
