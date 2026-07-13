"""Benchmark + identity-guardrail harness (Issue #79) -- the backbone epic #78
recommended landing FIRST and never has. Every training-efficiency optimization
in #78 (bf16, torch.compile, fused kernels, GaLore, offload, ...) is only a real
win if it (a) improves step-time/tokens-per-second/peak-VRAM and (b) does not
change what the model learns -- the same "HVN identity preserved" bar every
other issue in this epic has been individually, ad-hoc verifying (gradient
parity in test_block_coordinate_offload.py, logits parity for #101, etc.).
This gives that check one reusable, general home instead of re-deriving it
per issue.

Two configs (a documented "baseline" -- e.g. fp32, vanilla AdamW, no tricks --
and a "candidate" with one or more optimizations flipped on) are each run in
an isolated subprocess (mirrors scripts/bench_offload_vram.py's methodology;
CUDA memory stats and compiled-kernel state don't reset cleanly within one
process) on IDENTICAL fixed-seed synthetic batches, so any loss divergence
between them is attributable to the optimization, not to random data
differences. Reports speedup, VRAM delta, and a pass/fail parity verdict.

Usage:
    python scripts/bench_harness.py --baseline configs/baseline.json \\
        --candidate configs/candidate.json
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

VOCAB = 8000


def run_probe(cfg: dict) -> dict:
    """Train `model_kwargs` for `steps` steps on fixed-seed synthetic batches
    (seed defaults to 0, shared across configs being compared -- this is what
    makes cross-config loss comparison meaningful). Returns per-step losses,
    step times (first step excluded from timing as warmup/compile-cost), and
    peak VRAM stats."""
    import torch
    from hvn.edge.model import EdgeHVN

    result: dict = {"label": cfg.get("label", "unnamed")}
    if not torch.cuda.is_available() and cfg.get("require_cuda", True):
        result["verdict"] = "no-cuda"
        return result
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_seed = cfg.get("data_seed", 0)
    model_seed = cfg.get("model_seed", data_seed)
    steps = cfg.get("steps", 8)
    batch_size = cfg.get("batch_size", 4)
    seq_len = cfg.get("seq_len", 128)
    lr = cfg.get("lr", 1e-3)
    precision = cfg.get("precision", "fp32")  # "fp32" | "bf16"

    torch.manual_seed(model_seed)
    model_kwargs = dict(cfg.get("model_kwargs", {}))
    model_kwargs.setdefault("vocab_size", VOCAB)
    model = EdgeHVN(**model_kwargs).to(device)
    model.train()
    result["params"] = sum(p.numel() for p in model.parameters())

    opt_kwargs = dict(cfg.get("optimizer_kwargs", {}))
    opt_cls = getattr(torch.optim, cfg.get("optimizer", "AdamW"))
    opt = opt_cls(model.parameters(), lr=lr, **opt_kwargs)

    sched = None
    if cfg.get("block_coordinate"):
        from hvn.core.block_coordinate import BlockCoordinateScheduler
        sched = BlockCoordinateScheduler(model, opt, **cfg["block_coordinate"])

    # Fixed-seed synthetic batches, generated ONCE per step index up front so
    # both configs being compared see byte-identical data regardless of any
    # RNG consumption difference the optimization itself introduces (e.g.
    # dropout, if ever turned on here -- currently always 0 in model_kwargs
    # defaults, kept explicit rather than assumed).
    torch.manual_seed(data_seed)
    batches = [
        (
            torch.randint(0, VOCAB, (batch_size, seq_len), device=device),
            torch.randint(0, VOCAB, (batch_size, seq_len), device=device),
        )
        for _ in range(steps)
    ]

    crit = torch.nn.CrossEntropyLoss()
    autocast_dtype = torch.bfloat16 if precision == "bf16" else None
    losses = []
    step_times = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        for i, (x, y) in enumerate(batches):
            t0 = time.time()
            ctx = (
                torch.amp.autocast("cuda", dtype=autocast_dtype)
                if autocast_dtype is not None else torch.enable_grad()
            )
            with ctx:
                loss = crit(model(x).reshape(-1, VOCAB), y.reshape(-1))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched is not None:
                sched.step()
            if device.type == "cuda":
                torch.cuda.synchronize()
            if i > 0:  # exclude first step: compile/cudnn-autotune warmup cost
                step_times.append(time.time() - t0)
            losses.append(loss.item())
    except torch.OutOfMemoryError:
        result["verdict"] = "oom"
        return result

    result.update({
        "verdict": "ok",
        "losses": losses,
        "median_step_s": sorted(step_times)[len(step_times) // 2] if step_times else None,
        "tokens_per_s": (
            (batch_size * seq_len) / (sorted(step_times)[len(step_times) // 2])
            if step_times else None
        ),
    })
    if device.type == "cuda":
        result["peak_alloc_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        result["peak_reserved_mb"] = round(torch.cuda.max_memory_reserved() / 2**20)
    return result


def _probe_subprocess(cfg: dict, timeout: int = 300) -> dict:
    proc = subprocess.run(
        [sys.executable, __file__, "--probe", json.dumps(cfg)],
        capture_output=True, text=True, timeout=timeout, cwd=_REPO_ROOT,
    )
    line = None
    for ln in proc.stdout.strip().splitlines():
        if ln.startswith("{"):
            line = ln
    if line is None:
        return {**cfg, "verdict": "subprocess-failed", "stderr_tail": proc.stderr[-1500:]}
    return json.loads(line)


def compare(baseline: dict, candidate: dict, loss_atol: float = 1e-2, loss_rtol: float = 1e-2) -> dict:
    """Parity verdict between a baseline and candidate probe result. Default
    tolerance is loose enough to admit bf16's genuine numerical noise (an
    accepted, documented cost of that optimization) but tight enough to catch
    a real divergence (wrong device placement, a dropped block, mismatched
    data). Tighten via loss_atol/loss_rtol for fp32-only comparisons (e.g.
    #99/#100/#101/#102/#103, which claim BIT-level parity, not just "close")."""
    out = {
        "baseline_label": baseline.get("label"), "candidate_label": candidate.get("label"),
        "speedup": None, "vram_delta_mb": None, "loss_parity": None,
    }
    if baseline.get("verdict") != "ok" or candidate.get("verdict") != "ok":
        out["loss_parity"] = "skipped (non-ok verdict)"
        return out

    if baseline.get("median_step_s") and candidate.get("median_step_s"):
        out["speedup"] = round(baseline["median_step_s"] / candidate["median_step_s"], 3)
    if "peak_reserved_mb" in baseline and "peak_reserved_mb" in candidate:
        out["vram_delta_mb"] = candidate["peak_reserved_mb"] - baseline["peak_reserved_mb"]

    b_losses, c_losses = baseline["losses"], candidate["losses"]
    if len(b_losses) != len(c_losses):
        out["loss_parity"] = f"FAIL (length mismatch: {len(b_losses)} vs {len(c_losses)})"
        return out
    max_diff = max(abs(b - c) for b, c in zip(b_losses, c_losses))
    tol = [loss_atol + loss_rtol * abs(b) for b in b_losses]
    ok = all(abs(b - c) <= t for b, c, t in zip(b_losses, c_losses, tol))
    out["loss_parity"] = f"{'PASS' if ok else 'FAIL'} (max|diff|={max_diff:.4e})"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", type=str, default=None, help="(internal) run one probe, print JSON")
    ap.add_argument("--baseline", type=str, help="path to baseline config JSON")
    ap.add_argument("--candidate", type=str, help="path to candidate config JSON")
    ap.add_argument("--loss-atol", type=float, default=1e-2)
    ap.add_argument("--loss-rtol", type=float, default=1e-2)
    ap.add_argument("--out", type=str, default=None, help="optional: write full JSON result here")
    args = ap.parse_args()

    if args.probe is not None:
        print(json.dumps(run_probe(json.loads(args.probe))))
        return 0

    if not args.baseline or not args.candidate:
        ap.error("--baseline and --candidate are required (unless --probe)")

    with open(args.baseline) as f:
        baseline_cfg = json.load(f)
    with open(args.candidate) as f:
        candidate_cfg = json.load(f)

    print(f"running baseline: {baseline_cfg.get('label', args.baseline)} ...", flush=True)
    baseline = _probe_subprocess(baseline_cfg)
    print(f"  -> {baseline.get('verdict')}  median_step_s={baseline.get('median_step_s')}  "
          f"peak_reserved_mb={baseline.get('peak_reserved_mb')}")

    print(f"running candidate: {candidate_cfg.get('label', args.candidate)} ...", flush=True)
    candidate = _probe_subprocess(candidate_cfg)
    print(f"  -> {candidate.get('verdict')}  median_step_s={candidate.get('median_step_s')}  "
          f"peak_reserved_mb={candidate.get('peak_reserved_mb')}")

    verdict = compare(baseline, candidate, args.loss_atol, args.loss_rtol)
    print("\n" + "=" * 72)
    print(f"speedup:        {verdict['speedup']}x")
    print(f"VRAM delta:     {verdict['vram_delta_mb']:+d} MB" if verdict["vram_delta_mb"] is not None
          else "VRAM delta:     n/a")
    print(f"loss parity:    {verdict['loss_parity']}")
    print("=" * 72)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as f:
            json.dump({"baseline": baseline, "candidate": candidate, "verdict": verdict}, f, indent=2)
        print(f"wrote {args.out}")

    return 0 if verdict["loss_parity"] is None or verdict["loss_parity"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
