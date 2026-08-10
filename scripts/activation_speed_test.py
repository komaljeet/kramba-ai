"""Does swapping the FFN activation move training throughput? (Issue #140)

Author: Rambarun Komaljeet

The activation ablation's full form costs ~6 GPU-hours, and most of that buys a
*quality* answer.  The throughput question -- "is ReLU-squared or PowLU faster to
train than the incumbent GELU" -- needs no training at all, only timing, and this
script answers it in ~20 minutes.

Prior expectation, stated before running (the estimate this is meant to falsify):
**no measurable difference.**  The FFN is two ``d x 2d`` matmuls around one
pointwise op; at d=512 the matmuls are ~400x the arithmetic, and all three
activations read and write the same tensors.  If that reasoning is right, the
arms land inside the noise.

Three design choices make the result trustworthy rather than decorative:

1. **Interleaved, not sequential.**  Arms are timed round-robin (A,B,C,D,A,B,C,D,
   ...) so that thermal drift or another process claiming the GPU hits every arm
   equally.  Running arms back-to-back is how this repo previously measured a
   ~2x difference between a config and *itself* (docs/state.md, speed ceiling).

2. **A noise-floor arm.**  ``gelu`` is entered twice under different names.  The
   gap between those two identical configs is the harness's own noise; any arm
   difference smaller than it is not a result.  Without this there is no way to
   tell a real 2% win from drift, and the temptation is to report drift.

3. **A positive control.**  ``slow_control`` is a deliberately expensive
   activation (stacked transcendentals).  It exists to answer "if an activation
   *did* cost real time, would this harness see it?" -- asked before the numbers
   land, per CLAUDE.md.  If the slow control does not come out clearly slower,
   the harness is not sensitive enough to make any claim and the run is void.

Eager mode, deliberately: under ``torch.compile`` all three activations fuse into
the surrounding elementwise epilogue, which can only *shrink* the difference.
Eager is therefore the sensitive case -- a difference absent here is absent in
production too, while a difference found here would still need a compiled re-run
before being believed.

Synthetic token batches, deliberately: the real dataloader's variance is far
larger than the effect being measured and would swamp it.  Timing does not depend
on token content.

Usage::

    python scripts/activation_speed_test.py \
        --checkpoint ../kramba-ai/checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.core.activations import make_activation          # noqa: E402
from hvn.core.deltanet import DeltaNetBlock               # noqa: E402
from hvn.edge.model import EdgeHVN                        # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model       # noqa: E402


class _SlowControl(nn.Module):
    """Deliberately expensive pointwise activation -- the positive control.

    Not a candidate for anything; its only job is to be visibly slower so the
    harness can prove it is able to resolve an activation-level cost difference
    at all.  If this does not separate from GELU, no null result here means
    anything.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = torch.nn.functional.gelu(x)
        for _ in range(6):
            y = torch.tanh(y) + torch.sin(y) * torch.exp(-y.abs())
        return y


def build_model(ckpt: dict, activation: str, device: torch.device,
                attn_layers=(3, 10), attn_window: int = 4096) -> EdgeHVN:
    """Build the production hybrid at the checkpoint's own config.

    Weights are left at fresh init: timing is independent of weight *values*, and
    loading 317 MB per arm per repetition would dominate the run.  Shapes,
    dtypes, block plan, and the ternary wrapping all match production, which is
    what determines the kernels that execute.
    """
    num_layers = ckpt["num_layers"]
    block_plan = ["transformer" if i in set(attn_layers) else "deltanet"
                  for i in range(num_layers)]
    model = EdgeHVN(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=num_layers,
        n_heads=ckpt.get("n_heads", 4),
        use_deltanet=False,
        deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        block_plan=block_plan,
        attn_window=attn_window,
        ffn_activation=activation if activation in ("gelu", "relu2", "powlu") else "gelu",
    )
    # Match production: ternary BitNet b1.58 on everything except beta/alpha/decoder.
    all_linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    targets = [n for n in all_linears if not any(p in n for p in exclude)]
    apply_ternary_to_model(model, target_names=targets)

    if activation == "slow_control":
        for blk in model.blocks:
            if isinstance(blk, DeltaNetBlock):
                blk.ffn[1] = _SlowControl()

    model = model.to(device)
    # Mirror the FT recipe's scan flags (continue_ft_hybrid.enable_scan_flags).
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
            blk.use_megakernel_scan = True
    return model


def time_arm(model: EdgeHVN, *, device, batch_size: int, seq_len: int,
             grad_accum: int, warmup_steps: int, timed_steps: int,
             vocab_size: int, seed: int) -> dict:
    """One timing measurement: micro-batches/sec and tokens/sec, steady state."""
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-9)  # lr~0: weights stay put
    criterion = nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)

    def one_micro_batch(i: int):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
        x = x.to(device, non_blocking=True)
        y = torch.roll(x, -1, dims=1)
        with torch.autocast("cuda", dtype=torch.bfloat16,
                            enabled=(device.type == "cuda")):
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss = loss / grad_accum
        loss.backward()
        if (i + 1) % grad_accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

    for i in range(warmup_steps):
        one_micro_batch(i)
    if device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for i in range(timed_steps):
        one_micro_batch(i)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    peak = (torch.cuda.max_memory_allocated() / 2**20) if device.type == "cuda" else 0.0
    opt.zero_grad(set_to_none=True)
    return {
        "seconds": elapsed,
        "tokens_per_sec": timed_steps * batch_size * seq_len / elapsed,
        "peak_vram_mib": peak,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True,
                    help="only its config is read; weights are not loaded")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--warmup-steps", type=int, default=8)
    ap.add_argument("--timed-steps", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3,
                    help="interleaved rounds; each arm is measured once per round")
    ap.add_argument("--out", type=str,
                    default="results/activation_ablation/speed_test.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable -- docs/ai_rules.md requires GPU. Fix the "
                 "install rather than timing on CPU (the answer would be "
                 "meaningless: CPU kernel costs do not predict GPU ones).")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"config from {args.checkpoint}: d_model={ckpt['d_model']}, "
          f"layers={ckpt['num_layers']}, vocab={ckpt['vocab_size']}")

    # gelu appears twice under different names on purpose -- see module docstring.
    arms = ["gelu_a", "relu2", "powlu", "gelu_b", "slow_control"]
    impl = {"gelu_a": "gelu", "gelu_b": "gelu", "relu2": "relu2",
            "powlu": "powlu", "slow_control": "slow_control"}
    samples: dict[str, list[dict]] = {a: [] for a in arms}

    print(f"\nInterleaved timing: {args.repeats} rounds x {len(arms)} arms, "
          f"{args.timed_steps} timed micro-batches each "
          f"(batch {args.batch_size} x seq {args.seq_len})")
    print("Arms are round-robin so drift hits all of them equally.\n")

    for rnd in range(args.repeats):
        for arm in arms:
            model = build_model(ckpt, impl[arm], device)
            r = time_arm(model, device=device, batch_size=args.batch_size,
                         seq_len=args.seq_len, grad_accum=args.grad_accum,
                         warmup_steps=args.warmup_steps,
                         timed_steps=args.timed_steps,
                         vocab_size=ckpt["vocab_size"],
                         seed=1000 * rnd + hash(arm) % 1000)
            samples[arm].append(r)
            print(f"  round {rnd+1}/{args.repeats}  {arm:<13} "
                  f"{r['tokens_per_sec']:8.1f} tok/s   "
                  f"peak {r['peak_vram_mib']:7.1f} MiB")
            del model
            torch.cuda.empty_cache()
        print()

    def med(arm, key):
        return statistics.median(s[key] for s in samples[arm])

    base = med("gelu_a", "tokens_per_sec")
    noise_pct = abs(med("gelu_b", "tokens_per_sec") - base) / base * 100.0
    slow_pct = (med("slow_control", "tokens_per_sec") - base) / base * 100.0

    print("=" * 68)
    print(f"{'arm':<14}{'tok/s (median)':>16}{'vs gelu_a':>12}{'peak MiB':>12}")
    print("-" * 68)
    for arm in arms:
        t = med(arm, "tokens_per_sec")
        print(f"{arm:<14}{t:>16.1f}{(t - base) / base * 100:>11.2f}%"
              f"{med(arm, 'peak_vram_mib'):>12.1f}")
    print("=" * 68)
    print(f"\nNoise floor (gelu vs itself): {noise_pct:.2f}%")
    print(f"Positive control (slow_control vs gelu): {slow_pct:+.2f}%")

    # The harness must prove it can see a difference before its null means anything.
    control_ok = slow_pct < -3.0 * max(noise_pct, 0.5)
    if not control_ok:
        print("\n*** RUN VOID: the deliberately-expensive control did not come "
              "out clearly slower. The harness cannot resolve activation-level "
              "cost differences here, so it cannot support a 'no difference' "
              "claim either. Do not report these numbers.")

    verdicts = {}
    for arm in ("relu2", "powlu"):
        delta = (med(arm, "tokens_per_sec") - base) / base * 100.0
        if not control_ok:
            verdicts[arm] = "void (positive control failed)"
        elif abs(delta) <= noise_pct:
            verdicts[arm] = f"tie ({delta:+.2f}% is within the {noise_pct:.2f}% noise floor)"
        else:
            verdicts[arm] = f"{'faster' if delta > 0 else 'slower'} by {abs(delta):.2f}% (noise floor {noise_pct:.2f}%)"
        print(f"  {arm}: {verdicts[arm]}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "issue": 140,
            "what_this_measures": "training throughput vs FFN activation, eager mode",
            "config": vars(args),
            "gpu": torch.cuda.get_device_name(0),
            "model": {k: ckpt[k] for k in ("d_model", "num_layers", "vocab_size")},
            "samples": samples,
            "median_tokens_per_sec": {a: med(a, "tokens_per_sec") for a in arms},
            "median_peak_vram_mib": {a: med(a, "peak_vram_mib") for a in arms},
            "noise_floor_pct": noise_pct,
            "positive_control_delta_pct": slow_pct,
            "positive_control_passed": control_ok,
            "verdicts": verdicts,
        }, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
