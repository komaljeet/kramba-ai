"""How much of the real 1B step is the scan? (Amdahl bound before any kernel work)

Author: Rambarun Komaljeet

Why this runs BEFORE anyone writes a kernel
---------------------------------------------
`scripts/probe_scan_levers.py` found the DeltaNet scan is latency-bound: a fully
serial 4,096-step chain at **1,590 ns/step**, against ~120 ns implied by the data
it actually needs -- a **13x** gap that a better kernel could in principle attack.

That number is worthless on its own. A 13x win on 5% of the step is 4.6%. This
measures the denominator first, because the alternative is days of Triton work
for a bounded ceiling nobody checked. #131's precedent is exact: its bounding
profile found the projection GEMMs were the *small* slice and retired the plan it
was meant to justify.

Method
------
`torch.profiler` over real steps of the real 1B config, attributing **CUDA
device time** (not CPU time) by kernel name:

* **scan**       -- the hand-written Triton kernels (`gated_delta`, `gd_fwd`,
                    `gd_bwd`, `mk_fwd`, `loggram`, and Triton JIT names)
* **gemm**       -- cuBLAS/CUTLASS matmuls: projections, FFN, the decoder
* **elementwise**-- pointwise, norms, copies, casts
* **optimizer**  -- Muon's Newton-Schulz and the AdamW aux group
* **other**      -- everything unattributed, reported rather than hidden

Falsifiability / sanity checks, asked before running
-------------------------------------------------------
Two ways this could report a confident wrong answer, both checked:

1. **Attribution that does not sum.** If the categories miss kernels, every share
   is wrong. `other` is printed, never folded away, and a large `other` voids the
   reading.
2. **Device time that does not match the wall clock.** The step is ~98% GPU-busy,
   so summed device time must land near measured step time. If it is far off, the
   profiler is missing work and the shares are not trustworthy. Asserted, not
   assumed.

*"If the scan were a tiny slice, would this catch it?"* Yes -- that is the
expected outcome, and it is the one that saves the most work.

Usage::

    python scripts/attribute_1b_step.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.model import EdgeHVN                              # noqa: E402
from hvn.optimizers import build_optimizer                      # noqa: E402
from scripts.ab_prodigy import enable_scan_flags                # noqa: E402

#: substring -> category. Checked in order; first match wins.
PATTERNS = [
    ("scan", ("gated_delta", "gd_fwd", "gd_bwd", "mk_fwd", "mk_bwd",
              "megascan", "mega_scan", "loggram", "log_gram", "delta_rule",
              "triton_")),
    ("gemm", ("gemm", "cutlass", "ampere", "sm80", "sm90", "dot_kernel",
              "s16816", "implicit", "cublas", "nn_128x", "tn_128x", "nt_128x")),
    ("optimizer", ("newton", "schulz", "adam", "muon", "orthogon", "foreach")),
    ("elementwise", ("elementwise", "vectorized", "pointwise", "reduce",
                     "norm", "copy", "cast", "fill", "index", "embedding",
                     "softmax", "cat_", "unrolled", "direct_copy")),
]


def categorize(name: str) -> str:
    low = name.lower()
    for cat, keys in PATTERNS:
        if any(k in low for k in keys):
            return cat
    return "other"


def plan(num_layers: int):
    attn = (max(1, num_layers // 4), max(2, (3 * num_layers) // 4))
    return ["transformer" if i in attn else "deltanet" for i in range(num_layers)]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--d-model", type=int, default=2048)
    p.add_argument("--n-heads", type=int, default=64)
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--seq", type=int, default=4096)
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--loss-chunk", type=int, default=512)
    p.add_argument("--vocab-size", type=int, default=50257)
    p.add_argument("--profile-steps", type=int, default=2)
    p.add_argument("--out", default="results/roofline/attribution_1b.json")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md)")

    from hvn.edge.chunked_loss import chunked_cross_entropy
    model = EdgeHVN(vocab_size=args.vocab_size, d_model=args.d_model,
                    num_layers=args.num_layers, n_heads=args.n_heads,
                    block_plan=plan(args.num_layers), attn_window=4096,
                    tie_embeddings=False, grad_checkpoint=True)
    n_params = sum(p_.numel() for p_ in model.parameters())
    model = model.to("cpu")
    enable_scan_flags(model)
    opt, desc = build_optimizer(model, "muon", lr=2e-5, weight_decay=0.01,
                                muon_lr=0.02)
    from hvn.core.block_coordinate import BlockCoordinateScheduler
    sched = BlockCoordinateScheduler(
        model, opt, steps_per_block=50, device="cuda",
        offload_inactive_weights=True, offload_park_dtype=torch.bfloat16,
        offload_prefetch=True)
    params = [p_ for p_ in model.parameters() if p_.requires_grad]
    print(f"[attr] {n_params:,} params, d={args.d_model}/{args.num_layers}L, "
          f"n_heads={args.n_heads}, seq={args.seq}")
    print(f"[attr] {desc}")

    x = torch.randint(0, args.vocab_size, (args.batch, args.seq), device="cuda")
    y = torch.randint(0, args.vocab_size, (args.batch, args.seq), device="cuda")

    def one_step():
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(x, return_hidden=True)
            hidden = hidden[0] if isinstance(hidden, tuple) else hidden
            loss = chunked_cross_entropy(hidden, y, model.decoder,
                                         chunk=args.loss_chunk).mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)
        sched.step()

    # warm up, and take an unprofiled reference step time
    for _ in range(2):
        one_step()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(args.profile_steps):
        one_step()
    torch.cuda.synchronize()
    ref_step = (time.perf_counter() - t0) / args.profile_steps
    print(f"[attr] unprofiled step: {ref_step:.3f} s "
          f"({args.batch*args.seq/ref_step:.0f} tok/s)\n")

    from torch.profiler import profile, ProfilerActivity
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                 record_shapes=False) as prof:
        for _ in range(args.profile_steps):
            one_step()
        torch.cuda.synchronize()

    # ONLY kernel-level entries. `key_averages()` also returns the op-level
    # rows (aten::mm, _MegaScanBackward, ...) that LAUNCHED those kernels, and
    # both carry the same device time -- summing across all of it double-counts.
    # That is precisely the flaw #131 disclosed in its own decode profile
    # ("they agree to the last decimal at 54,794.628 us"), and here it showed up
    # as a device/wall ratio of 1.99: almost exactly 2x.
    from torch.autograd import DeviceType
    totals, per_kernel, dropped = {}, {}, 0.0
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", 0) or 0
        if dev <= 0:
            continue
        if getattr(e, "device_type", None) != DeviceType.CUDA:
            dropped += dev          # op-level launcher; its kernel is counted
            continue
        cat = categorize(e.key)
        totals[cat] = totals.get(cat, 0.0) + dev
        per_kernel[e.key] = (cat, dev)

    grand = sum(totals.values())
    per_step_dev = grand / args.profile_steps / 1e6      # us -> s

    print(f"[attr] {'category':<14}{'s/step':>10}{'share':>9}")
    rows = {}
    for cat in sorted(totals, key=lambda c: -totals[c]):
        s = totals[cat] / args.profile_steps / 1e6
        print(f"[attr] {cat:<14}{s:>10.3f}{100*s/per_step_dev:>8.1f}%")
        rows[cat] = {"s_per_step": s, "share_pct": 100 * s / per_step_dev}
    print(f"[attr] {'TOTAL DEVICE':<14}{per_step_dev:>10.3f}{'100.0%':>9}")
    print(f"[attr] (excluded {dropped/args.profile_steps/1e6:.3f} s/step of "
          f"op-level launcher rows to avoid the #131 double-count)")

    print(f"\n[attr] top kernels by device time:")
    for k, (cat, dev) in sorted(per_kernel.items(), key=lambda kv: -kv[1][1])[:10]:
        print(f"[attr]   {dev/args.profile_steps/1e6:>7.3f}s  [{cat:<11}] "
              f"{k[:66]}")

    # ---- op-level view: WHAT the elementwise time actually is --------------
    #
    # Deliberately a SEPARATE view, never summed with the kernel table above.
    # Kernel rows say how much; their mangled C++ names cannot say what. The
    # aten:: rows name the operation, and their self-device-times also sum to
    # ~one step (they are the launchers), so they are a valid decomposition on
    # their own -- just not one that may be added to the other.
    #
    # The BitNet STE chain is the specific suspect: ternary_quantize_ste is
    # abs -> mean -> div -> round -> clamp -> mul -> sub -> add over every
    # weight, EVERY forward, and #131 measured it at 395 ms of decode against a
    # 66 ms GEMM. In training it amortises over batch x seq, but at batch 1 that
    # amortisation is at its weakest.
    STE_CHAIN = ("abs", "mean", "div", "round", "clamp", "mul", "sub", "add",
                 "rsqrt", "pow", "neg", "sign")
    op_rows, ste_total, op_total = [], 0.0, 0.0
    for e in prof.key_averages():
        dev = getattr(e, "self_device_time_total", 0) or 0
        if dev <= 0 or getattr(e, "device_type", None) == DeviceType.CUDA:
            continue
        if not e.key.startswith("aten::"):
            continue
        op = e.key[len("aten::"):]
        op_rows.append((op, dev))
        op_total += dev
        if any(op == c or op.startswith(c + "_") or op.startswith(c + ".")
               for c in STE_CHAIN):
            ste_total += dev

    op_rows.sort(key=lambda r: -r[1])
    print()
    print("[attr] ===== op-level view (names the work; "
          "do NOT add to the kernel table) =====")
    print(f"[attr] {'aten op':<24}{'s/step':>10}{'share':>9}")
    for op, dev in op_rows[:14]:
        s_ = dev / args.profile_steps / 1e6
        print(f"[attr] {op:<24}{s_:>10.3f}{100*dev/op_total:>8.1f}%")
    print(f"[attr] {'(op-level total)':<24}"
          f"{op_total/args.profile_steps/1e6:>10.3f}{'100.0%':>9}")
    print(f"[attr] pointwise/STE-chain ops together: "
          f"{ste_total/args.profile_steps/1e6:.3f} s/step "
          f"({100*ste_total/op_total:.1f}% of op-level device time)")

    # ---- sanity checks -----------------------------------------------------
    print(f"\n[attr] ================== CHECKS ==================")
    void = []
    other = rows.get("other", {}).get("share_pct", 0.0)
    print(f"[attr] unattributed 'other': {other:.1f}%")
    if other > 25:
        void.append(f"{other:.0f}% of device time is unattributed")
        print("[attr] *** too much is unclassified for the shares to be read")

    ratio = per_step_dev / ref_step
    print(f"[attr] device time / wall clock: {ratio:.2f} "
          f"({per_step_dev:.3f}s vs {ref_step:.3f}s)")
    if not (0.55 <= ratio <= 1.60):
        void.append(f"device/wall ratio {ratio:.2f} is implausible")
        print("[attr] *** summed device time does not match the step; the "
              "profiler is missing work or double-counting, so shares are "
              "not trustworthy")
    else:
        print("[attr] consistent with a ~98% GPU-busy step (overlap and "
              "profiler overhead keep this from being exactly 1.00)")

    scan_share = rows.get("scan", {}).get("share_pct", 0.0)
    print(f"\n[attr] ================== READING ==================")
    if void:
        print("[attr] *** VOID: " + "; ".join(void))
    else:
        print(f"[attr] the scan is {scan_share:.1f}% of device time.")
        for speedup in (2, 4, 13):
            frac = scan_share / 100
            gain = 1 / ((1 - frac) + frac / speedup)
            print(f"[attr]   a {speedup:>2}x scan kernel -> {gain:.2f}x on the "
                  f"whole step  ({ref_step:.2f}s -> {ref_step/gain:.2f}s, "
                  f"~{int(375/gain)} days)")
        print("[attr] Amdahl, not pessimism: the ceiling is set by the "
              "denominator, and 13x was the measured latency gap.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"question": "what fraction of the 1B step is the scan",
                   "config": vars(args), "params": n_params,
                   "ref_step_s": ref_step, "device_s_per_step": per_step_dev,
                   "categories": rows, "void_reasons": void}, fh, indent=2)
    print(f"[attr] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
