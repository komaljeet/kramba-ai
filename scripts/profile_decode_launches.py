"""Count kernel launches per decoded token (Issue #138).

Written by: Rambarun Komaljeet

#131 profiled decode and found it **launch-bound**, not compute-bound: ~2,815
kernel launches per token and a CPU/device self-time ratio of 0.91, meaning the
GPU spends about as long waiting to be told what to do as computing. It closed
with the recommendation to attack launch count, unactioned until now.

This script establishes the baseline that #138's acceptance criteria require:
launches/token **before and after**, reported as the primary number.

**Launch count is the metric, not tokens/sec.** #131 measured the same arm at
21.45 and 48.40 tok/s across runs — a ~1.5x spread on identical code, from GPU
clock ramp and contention on this machine. Launch count is deterministic. A
change that improves tok/s without reducing launches did not address the
diagnosis, whatever the stopwatch says.

Profiles ``forward_step``, never ``forward``: the latter dispatches to
``_scan_chunked``, a path serving never takes, and it diverges numerically from
``_scan_sequential`` on trained weights (#97).

Usage
-----
    python scripts/profile_decode_launches.py --checkpoint checkpoints/hybrid_chat_v1.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from torch.profiler import ProfilerActivity, profile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402


def _is_kernel(evt) -> bool:
    """A real device kernel launch, not a CPU-side op wrapper.

    ``key_averages()`` mixes op-level entries (``aten::mm``) with kernel-level
    ones (the mangled CUDA symbol) carrying the SAME device time — #131 was
    burned by summing across both and double-counting. Counting launches avoids
    that trap only if the filter is on device-side events, which is what
    ``device_type``/``is_gpu`` distinguishes.
    """
    dt = getattr(evt, "device_type", None)
    if dt is not None:
        return str(dt).endswith("CUDA") or str(dt).endswith("PrivateUse1")
    return bool(getattr(evt, "is_gpu", False))


@torch.no_grad()
def count_launches(model, n_tokens: int, warmup: int, device) -> dict:
    """Kernel launches over ``n_tokens`` of real incremental decode."""
    # forward_step, not chat_step: chat_step generates a whole TURN (a sampling
    # loop over max_new_tokens), so its launch count would fold in sampling and
    # be divided by a token count the caller does not control. forward_step is
    # the per-token primitive, and the thing a CUDA graph would capture.
    state = model.init_state(batch_size=1, device=device)
    tok = torch.randint(0, getattr(model, "vocab_size", 50257), (1, 1), device=device)

    # Warm up: first-call allocations and lazy init would otherwise be counted as
    # steady-state launches. This also fills the KV ring past its warm-up branch
    # (#138 approach 0), so the profile reflects the WRAPPED steady state — the
    # regime a graph capture would target, and the only one with fixed shapes.
    pos = 0
    for _ in range(warmup):
        _logits, state = model.forward_step(tok, state, position_idx=pos)
        pos += 1
    torch.cuda.synchronize()

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(n_tokens):
            _logits, state = model.forward_step(tok, state, position_idx=pos)
            pos += 1
        torch.cuda.synchronize()

    kernels = 0
    cpu_self_us = 0.0
    dev_self_us = 0.0
    for evt in prof.key_averages():
        if _is_kernel(evt):
            kernels += evt.count
            dev_self_us += float(getattr(evt, "self_device_time_total",
                                         getattr(evt, "self_cuda_time_total", 0)) or 0)
        else:
            cpu_self_us += float(getattr(evt, "self_cpu_time_total", 0) or 0)

    return {
        "n_tokens": n_tokens,
        "kernel_launches": kernels,
        "launches_per_token": kernels / max(n_tokens, 1),
        "cpu_self_ms": cpu_self_us / 1000.0,
        "device_self_ms": dev_self_us / 1000.0,
        # The #131 diagnostic. ~0.9 means the GPU waits as long as it computes;
        # if this stays ~0.9 after a change, the change did not address it.
        "cpu_over_device": (cpu_self_us / dev_self_us) if dev_self_us else float("nan"),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/hybrid_chat_v1.pt")
    ap.add_argument("--tokens", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--label", default="baseline")
    ap.add_argument("--out", default="results/decode_launches/launches.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md).")
    device = torch.device("cuda")

    model, step, _ck = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()

    res = count_launches(model, args.tokens, args.warmup, device)
    res.update(label=args.label, checkpoint=args.checkpoint.replace("\\", "/"),
               vocab_size=getattr(model, "vocab_size", None), step=step)

    print(f"\n[launch] {args.label} | {os.path.basename(args.checkpoint)}")
    print(f"[launch]   launches/token : {res['launches_per_token']:.1f}"
          f"   <- THE METRIC")
    print(f"[launch]   total launches : {res['kernel_launches']:,} "
          f"over {res['n_tokens']} tokens")
    print(f"[launch]   CPU self       : {res['cpu_self_ms']:.1f} ms")
    print(f"[launch]   device self    : {res['device_self_ms']:.1f} ms")
    print(f"[launch]   CPU/device     : {res['cpu_over_device']:.2f}"
          f"   <- ~0.9 means launch-bound (#131)")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    prev = []
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            prev = json.load(fh).get("runs", [])
    prev.append(res)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"runs": prev}, fh, indent=2)
    print(f"[launch] appended to {args.out}")

    if len(prev) > 1:
        base = prev[0]
        print(f"\n[launch] vs '{base['label']}': "
              f"{base['launches_per_token']:.1f} -> {res['launches_per_token']:.1f} "
              f"({100*(1-res['launches_per_token']/base['launches_per_token']):+.1f}% fewer)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
