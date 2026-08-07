"""Bound the MatMul-free (Scope A) win before building it -- issue #131.

Author: Rambarun Komaljeet

Measures what fraction of EdgeHVN's incremental-decode GPU time is actually spent
inside the PROJECTION layers (the only thing Scope A can make faster). Everything
else -- the DeltaNet recurrence, attention QK^T/PV, norms, sampling -- is out of
scope by construction, so it is Amdahl headroom this project cannot touch.

Why this runs BEFORE any kernel work: docs/state.md records the speed-ceiling
investigation finding the scan was 60% of the step but only ~7.5% of GPU
self-time, and issue #43 found decode is launch-bound (a GEMM reformulation
measured throughput-neutral-to-negative; the win came from collapsing launches).
A profile is an hour; a fused ternary kernel is days.

Decision rule, fixed before the numbers land:
    projections > 30% of decode device self-time -> real headroom, proceed
    projections < 15%                            -> Amdahl caps Scope A, stop

Deliberately profiles ``forward_step`` and NOT ``forward``: docs/state.md records
that ``forward`` dispatches to ``_scan_chunked`` while ``forward_step`` uses
``_scan_sequential``, so profiling ``forward`` would measure a path serving never
takes.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402

try:
    from hvn.edge.ternary import TernaryLinear
except Exception:  # pragma: no cover - ternary module is expected to exist
    TernaryLinear = ()  # type: ignore[assignment]

PROJ_LABEL = "SCOPE_A_PROJ"


def _dev_total(evt) -> float:
    """Total device time for a profiler event, across torch naming versions."""
    for attr in ("device_time_total", "cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def _dev_self(evt) -> float:
    """Self device time for a profiler event, across torch naming versions."""
    for attr in ("self_device_time_total", "self_cuda_time_total"):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def _cpu_self(evt) -> float:
    for attr in ("self_cpu_time_total",):
        v = getattr(evt, attr, None)
        if v is not None:
            return float(v)
    return 0.0


def collect_projections(model: nn.Module) -> List[str]:
    """Every Linear-like module inside the block stack.

    Excludes the token embedding and the decoder/unembedding -- the Scope A spec
    keeps those dense, so counting them would inflate the headroom estimate.
    """
    linear_types: tuple = (nn.Linear,) + (
        (TernaryLinear,) if isinstance(TernaryLinear, type) else ()
    )
    names: List[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, linear_types):
            continue
        head = name.split(".")[0]
        if head in ("decoder", "token_embedding", "lm_head", "embed_out"):
            continue
        names.append(name)
    return names


def wrap_projections(model: nn.Module, names: List[str]) -> None:
    """Tag each projection's forward with a record_function scope.

    The profiler attributes child kernels to the enclosing scope, so the summed
    device time of PROJ_LABEL is the projections' true cost including whatever
    quantization work TernaryLinear does on the way into the GEMM.
    """
    lookup = dict(model.named_modules())
    for name in names:
        mod = lookup[name]
        orig = mod.forward

        def make(fn):
            def fwd(*a, **kw):
                with torch.profiler.record_function(PROJ_LABEL):
                    return fn(*a, **kw)
            return fwd

        mod.forward = make(orig)


@torch.no_grad()
def decode_loop(model, n_tokens: int, device, vocab: int, batch: int = 1) -> None:
    states = model.init_state(batch, device=device,
                              dtype=next(model.parameters()).dtype)
    tok = torch.randint(0, vocab, (batch, 1), device=device)
    for i in range(n_tokens):
        logits, states = model.forward_step(tok, states, position_idx=i)
        tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)


@torch.no_grad()
def time_decode(model, n_tokens: int, device, vocab: int, repeats: int,
                batch: int = 1) -> Dict[str, float]:
    """Wall-clock tok/s, measured OUTSIDE the profiler (which distorts timing).

    Repeated because docs/state.md records a single-generation timing in the #122
    sweep producing a physically impossible 2x -- single runs here are noise.
    """
    samples: List[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decode_loop(model, n_tokens, device, vocab, batch)
        torch.cuda.synchronize()
        samples.append(n_tokens / (time.perf_counter() - t0))
    samples.sort()
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / max(1, len(samples) - 1)
    return {
        "tokens_per_sec_mean": mean,
        "tokens_per_sec_std": var ** 0.5,
        "tokens_per_sec_min": samples[0],
        "tokens_per_sec_max": samples[-1],
        "samples": samples,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--out", default="results/matmul_free/decode_profile.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU required (docs/ai_rules.md: never fall back to CPU)")
    device = torch.device("cuda")

    model, step, _ = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    vocab = model.decoder.out_features if hasattr(model.decoder, "out_features") else 50257
    dtype = str(next(model.parameters()).dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[#131] loaded {args.checkpoint} (step {step}) "
          f"params={n_params/1e6:.1f}M dtype={dtype}")

    proj_names = collect_projections(model)
    proj_params = 0
    lookup = dict(model.named_modules())
    for n in proj_names:
        proj_params += sum(p.numel() for p in lookup[n].parameters())
    print(f"[#131] {len(proj_names)} projection modules, "
          f"{proj_params/1e6:.1f}M params "
          f"({100.0*proj_params/n_params:.1f}% of model)")

    # --- wall-clock first, unwrapped and unprofiled -------------------------
    decode_loop(model, args.warmup, device, vocab, args.batch)
    timing = time_decode(model, args.tokens, device, vocab, args.repeats, args.batch)
    print(f"[#131] decode {timing['tokens_per_sec_mean']:.2f} +/- "
          f"{timing['tokens_per_sec_std']:.2f} tok/s "
          f"(min {timing['tokens_per_sec_min']:.2f}, "
          f"max {timing['tokens_per_sec_max']:.2f}, n={args.repeats})")

    # --- profiled pass ------------------------------------------------------
    wrap_projections(model, proj_names)
    decode_loop(model, args.warmup, device, vocab, args.batch)
    torch.cuda.synchronize()

    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        decode_loop(model, args.tokens, device, vocab, args.batch)
        torch.cuda.synchronize()

    events = prof.key_averages()
    total_dev_self = sum(_dev_self(e) for e in events)
    total_cpu_self = sum(_cpu_self(e) for e in events)
    proj_dev = sum(_dev_total(e) for e in events if e.key == PROJ_LABEL)
    proj_calls = sum(int(getattr(e, "count", 0)) for e in events if e.key == PROJ_LABEL)
    n_kernels = sum(int(getattr(e, "count", 0)) for e in events if _dev_self(e) > 0)

    share = (100.0 * proj_dev / total_dev_self) if total_dev_self > 0 else 0.0

    if share > 30.0:
        verdict = "PROCEED - projections dominate enough for a kernel to matter"
    elif share < 15.0:
        verdict = "STOP - Amdahl caps Scope A in the low single digits"
    else:
        verdict = "MARGINAL - between the pre-registered thresholds"

    top = []
    for e in sorted(events, key=_dev_self, reverse=True)[:15]:
        if _dev_self(e) <= 0:
            continue
        top.append({"op": e.key, "self_device_us": _dev_self(e),
                    "count": int(getattr(e, "count", 0)),
                    "pct_of_device": 100.0 * _dev_self(e) / total_dev_self})

    out = {
        "issue": 131,
        "checkpoint": args.checkpoint,
        "step": step,
        "dtype": dtype,
        "params_total": n_params,
        "params_projections": proj_params,
        "projection_modules": len(proj_names),
        "decode": {"tokens": args.tokens, "batch": args.batch, **timing},
        "profile": {
            "total_device_self_us": total_dev_self,
            "total_cpu_self_us": total_cpu_self,
            "projection_device_us": proj_dev,
            "projection_calls": proj_calls,
            "kernel_launches": n_kernels,
            "kernel_launches_per_token": n_kernels / float(args.tokens),
            "cpu_to_device_ratio": (total_cpu_self / total_dev_self
                                    if total_dev_self > 0 else None),
        },
        "projection_share_pct": share,
        "thresholds": {"proceed_above": 30.0, "stop_below": 15.0},
        "verdict": verdict,
        "top_ops_by_device_self_time": top,
    }

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"[#131] projection share of device self-time: {share:.1f}%")
    print(f"[#131] kernel launches/token: {n_kernels/float(args.tokens):.1f}")
    print(f"[#131] CPU/device self-time ratio: "
          f"{out['profile']['cpu_to_device_ratio']:.2f}"
          if out["profile"]["cpu_to_device_ratio"] else "")
    print(f"[#131] VERDICT: {verdict}")
    print(f"[#131] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
