"""Does raising Inductor's compile_threads speed up torch.compile here? (#155)

Author: Rambarun Komaljeet

`torch._inductor.config.compile_threads` defaults to **1 on Windows** (the
`subprocess` worker start method is expensive there), so kernel codegen runs
single-threaded regardless of core count. Observed directly: a 157M model at
seq 4096 spent over an hour compiling with ~1 core of 16 busy and overall CPU
load at 7%.

That is a configuration default, not a hardware limit -- and it matters beyond
this laptop. On rented Linux boxes (RunPod) the compile is billed wall-clock
time on machines with 32+ cores; leaving 31 idle during every compile is a
direct cost. Linux already defaults this higher, but a repo that pins or
inherits `1` would carry the penalty over.

Measures compile wall-clock at several `compile_threads` values on the REAL
model shape, with the Inductor cache cleared between arms -- otherwise the
second arm reads a cached kernel and "wins" by doing nothing. That cache reset
is the one thing this benchmark cannot skip; without it the result is
guaranteed and meaningless.

Usage::

    python scripts/bench_compile_threads.py --threads 1 4 8 16
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import time

# Each arm runs in a FRESH interpreter: torch.compile caches aggressively in
# process, and `compile_threads` is read once when the worker pool is created,
# so setting it twice in one process would measure the first value twice.
CHILD = textwrap.dedent(r'''
    import os, sys, time, json
    os.environ["TORCHINDUCTOR_CACHE_DIR"] = sys.argv[2]
    import torch, torch.nn as nn
    from torch._inductor import config as ic
    ic.compile_threads = int(sys.argv[1])
    sys.path.insert(0, sys.argv[3])
    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model

    dev = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    V, D, L, H, S = 32003, 896, 16, 14, int(sys.argv[4])
    plan = ["transformer" if i in (4, 12) else "deltanet" for i in range(L)]
    m = EdgeHVN(vocab_size=V, d_model=D, num_layers=L, n_heads=H,
                block_plan=plan, attn_window=4096, tie_embeddings=True,
                grad_checkpoint=True).to(dev)
    lin = {n for n, x in m.named_modules() if isinstance(x, nn.Linear)}
    ex = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(m, target_names=[n for n in lin
                                            if not any(p in n for p in ex)])
    for b in m.blocks:
        if hasattr(b, "use_triton_scan"):
            b.use_triton_scan = True; b.use_megakernel_scan = True
    mc = torch.compile(m)
    x = torch.randint(0, V, (1, S), device=dev)
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        mc(x).float().pow(2).mean().backward()   # fwd AND bwd compile
    torch.cuda.synchronize()
    first = time.perf_counter() - t0
    t0 = time.perf_counter()
    with torch.amp.autocast("cuda", dtype=torch.bfloat16):
        mc(x).float().pow(2).mean().backward()
    torch.cuda.synchronize()
    second = time.perf_counter() - t0
    print("RESULT " + json.dumps({"compile_threads": ic.compile_threads,
                                  "first_call_s": first, "second_call_s": second,
                                  "compile_s": first - second}))
''')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4, 8, 16])
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--repo", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--out", default="results/compile_threads/bench.json")
    args = p.parse_args()

    import torch
    print(f"cores: {os.cpu_count()} logical | torch threads {torch.get_num_threads()}")
    print(f"model: 157M hybrid @ seq {args.seq_len}, fwd+bwd compile\n")

    results = {}
    for n in args.threads:
        cache = tempfile.mkdtemp(prefix=f"inductor_ct{n}_")
        try:
            t0 = time.perf_counter()
            proc = subprocess.run(
                [sys.executable, "-c", CHILD, str(n), cache, args.repo,
                 str(args.seq_len)],
                capture_output=True, text=True, timeout=5400)
            wall = time.perf_counter() - t0
            line = next((l for l in proc.stdout.splitlines()
                         if l.startswith("RESULT ")), None)
            if line is None:
                print(f"  compile_threads={n:<3} FAILED")
                print("   " + (proc.stderr.strip().splitlines() or ["(no stderr)"])[-1])
                continue
            r = json.loads(line[len("RESULT "):])
            r["process_wall_s"] = wall
            results[str(n)] = r
            print(f"  compile_threads={n:<3} compile {r['compile_s']:7.1f}s  "
                  f"(process {wall:6.1f}s, warm step {r['second_call_s']:.3f}s)")
        except subprocess.TimeoutExpired:
            print(f"  compile_threads={n:<3} TIMEOUT (>90 min)")
        finally:
            shutil.rmtree(cache, ignore_errors=True)

    if len(results) > 1:
        base = results.get("1")
        if base:
            print(f"\n{'threads':>9}{'compile s':>12}{'speedup':>10}")
            for n, r in results.items():
                print(f"{n:>9}{r['compile_s']:>12.1f}"
                      f"{base['compile_s'] / r['compile_s']:>9.2f}x")
            best = min(results, key=lambda k: results[k]["compile_s"])
            gain = base["compile_s"] / results[best]["compile_s"]
            verdict = (f"compile_threads={best} is {gain:.2f}x faster than the "
                       f"Windows default of 1 -- worth setting"
                       if gain > 1.2 else
                       f"no meaningful gain (best {gain:.2f}x). The compile is not "
                       f"parallelising here; raising the setting is not the fix.")
            print(f"\nVERDICT: {verdict}")
            results["verdict"] = verdict

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 155, "cores": os.cpu_count(),
                   "seq_len": args.seq_len, "results": results}, fh, indent=2)
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
