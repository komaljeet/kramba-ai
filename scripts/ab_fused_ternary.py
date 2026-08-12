"""Does the fused ternary quantizer speed up the REAL training loop? (#154)

Author: Rambarun Komaljeet

A microbenchmark says the fused kernel is **4.14x** on the quantizer alone, which
projects to ~12% of a step. cProfile attributed only **4.6%** to it. Those
disagree, and today a microbenchmark already claimed torch.compile was +18% when
the real loop measured it **2.4x SLOWER** (#158). So the projection is not
trusted and this measures the loop.

Both arms are one process each, identical seed, identical shape, identical data
-- only `HVN_FUSED_TERNARY` differs. The arms are INTERLEAVED (eager, fused,
eager, fused) so thermal drift or a background process cannot be mistaken for a
result, per the contention finding in docs/state.md.

Reports median s/step over the timed steps, and checks the LOSS matches between
arms: a quantizer change that alters the loss trajectory is a correctness
problem, not a speedup, however fast it runs.

Usage::

    python scripts/ab_fused_ternary.py --steps 6 --rounds 2
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import textwrap
import time

CHILD = textwrap.dedent(r'''
    import os, sys, json, time
    fused, repo, steps, accum, seq = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])
    os.environ["HVN_FUSED_TERNARY"] = fused          # BEFORE importing hvn
    sys.path.insert(0, repo)
    import torch, torch.nn as nn
    from hvn.edge.ternary import _USE_FUSED_TERNARY
    assert (_USE_FUSED_TERNARY is True) == (fused == "1"), "flag did not take"

    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = torch.device("cuda")
    V, D, L, H = 32003, 896, 16, 28
    torch.manual_seed(0)

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

    opt, _ = build_optimizer(m, "muon", lr=3e-4, weight_decay=0.01, muon_lr=0.02)
    crit = nn.CrossEntropyLoss()
    g = torch.Generator().manual_seed(1)

    def one_step():
        tot = 0.0
        for _ in range(accum):
            x = torch.randint(0, V, (1, seq), generator=g).to(dev)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                loss = crit(m(x).reshape(-1, V), torch.roll(x, -1, 1).reshape(-1)) / accum
            loss.backward(); tot += float(loss)
        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
        opt.step(); opt.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        return tot

    one_step()                                        # warm
    per, losses = [], []
    for _ in range(steps):
        t0 = time.perf_counter(); losses.append(one_step()); per.append(time.perf_counter() - t0)
    print("RESULT " + json.dumps({
        "fused": fused == "1", "steady_s": sorted(per)[len(per)//2],
        "per_step": per, "first_loss": losses[0], "last_loss": losses[-1],
        "vram_mib": torch.cuda.max_memory_reserved() / 2**20}))
''')


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--seq-len", type=int, default=4096)
    p.add_argument("--repo", default=os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))))
    p.add_argument("--out", default="results/fused_ternary/ab.json")
    args = p.parse_args()

    print(f"157M @ 28 heads, seq {args.seq_len} x accum {args.grad_accum} "
          f"= {args.seq_len*args.grad_accum:,} tokens/step")
    print(f"interleaved: {args.rounds} rounds x (eager, fused)\n")

    samples = {"eager": [], "fused": []}
    losses = {"eager": [], "fused": []}
    for r in range(args.rounds):
        for name, flag in (("eager", "0"), ("fused", "1")):
            pr = subprocess.run(
                [sys.executable, "-c", CHILD, flag, args.repo, str(args.steps),
                 str(args.grad_accum), str(args.seq_len)],
                capture_output=True, text=True, timeout=3600)
            line = next((l for l in pr.stdout.splitlines()
                         if l.startswith("RESULT ")), None)
            if not line:
                err = [l for l in pr.stderr.splitlines() if l.strip()]
                print(f"  round {r+1} {name}: FAILED -- {(err[-1] if err else '?')[:140]}")
                continue
            res = json.loads(line[len("RESULT "):])
            samples[name].append(res["steady_s"])
            losses[name].append(res["last_loss"])
            print(f"  round {r+1} {name:<6} {res['steady_s']:.3f} s/step  "
                  f"loss {res['last_loss']:.5f}  VRAM {res['vram_mib']:.0f}M")

    if not (samples["eager"] and samples["fused"]):
        print("\nincomplete -- cannot conclude")
        return 1

    e, f = statistics.median(samples["eager"]), statistics.median(samples["fused"])
    print(f"\n{'arm':<8}{'s/step':>10}{'speedup':>10}")
    print(f"{'eager':<8}{e:>10.3f}{1.0:>9.2f}x")
    print(f"{'fused':<8}{f:>10.3f}{e/f:>9.2f}x")
    gain = (e / f - 1) * 100

    # A quantizer change that moves the loss is a correctness problem, however
    # fast it is. Identical seed and data mean the losses should agree closely.
    le, lf = statistics.median(losses["eager"]), statistics.median(losses["fused"])
    ldiff = abs(le - lf)
    print(f"\nloss: eager {le:.5f} vs fused {lf:.5f}  (diff {ldiff:.2e})")

    if ldiff > 1e-3:
        verdict = (f"REJECT: the loss differs by {ldiff:.2e} between arms. A "
                   f"quantizer change must not move the trajectory -- investigate "
                   f"before considering the {gain:+.1f}% speed.")
    elif gain > 3:
        verdict = (f"ADOPT: {gain:+.1f}% in the real loop ({e:.2f} -> {f:.2f} "
                   f"s/step), loss unchanged. On a 9-day run that is "
                   f"~{gain/100*9*24:.1f} hours.")
    else:
        verdict = (f"NOT WORTH IT: {gain:+.1f}% in the real loop. The kernel is "
                   f"4.14x faster in isolation but the step is GPU-bound, so the "
                   f"host-side saving does not surface -- the same reason "
                   f"torch.compile's microbenchmark did not transfer (#158).")
    print(f"\nVERDICT: {verdict}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 154, "config": vars(args), "samples": samples,
                   "losses": losses, "eager_s": e, "fused_s": f,
                   "gain_pct": gain, "loss_diff": ldiff,
                   "verdict": verdict}, fh, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
