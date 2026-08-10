"""Is the 83M starving the GPU? Throughput vs batch_size x n_heads (Issue #141).

Author: Rambarun Komaljeet

`docs/training.md:122` records head count as a **free 2.4x speedup** at 355M
(``--n-heads 8 -> 16``, 1.91 -> 0.79 s/micro-batch, same params, same memory).
The mechanism given there is occupancy: the Triton scan launches ``N = B x H``
programs (``hvn/core/deltanet_triton.py:218,222``) against this card's **20 SMs**.

If that is right, the 83M is underfilled and nobody has checked.  It runs
``n_heads=8`` and the fine-tune recipe uses ``batch_size=2``, so ``N = 16``
programs for 20 SMs -- the card cannot be filled even once.

**The prediction, written down before running:** tokens/sec should climb with
``B x H`` up to roughly 20 and then flatten.  If the curve is flat from batch 1
onward, the occupancy story is wrong and this sweep says so.

Two corrections over the #140 harness, both taken from docs/training.md:

* **Reports ``max_memory_reserved``, not just ``max_memory_allocated``.**  #112
  found configs that looked fine at 5.92 GiB allocated while spilling at 6.34 GiB
  reserved on a 5.9966 GiB card.  Allocated alone cannot see the failure this
  sweep is most likely to hit.
* **Detects the WDDM silent spill.**  Past the card size Windows backs the
  overflow with host RAM instead of OOM-ing; steps go 4-15x slower and the run
  looks healthy.  A spilled arm is reported as SPILLED, not as "slower" --
  otherwise this concludes "big batches are slow" when the truth is "that batch
  did not fit".

Batch 1 (8 programs, under half the SMs) is the positive control: if even that is
not visibly slower per token, the harness cannot resolve occupancy effects and
the run is void.

Timing only -- no training, no quality claim.  ``n_heads`` changes the
architecture (``read_norm_weight`` is ``[n_heads, d_head]``), so a head-count
result applies to a *new* pretrain, not to an existing checkpoint.

Usage::

    python scripts/throughput_sweep.py --checkpoint <ckpt>
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

from hvn.edge.model import EdgeHVN                        # noqa: E402
from hvn.edge.ternary import apply_ternary_to_model       # noqa: E402

MIB = 2 ** 20
GIB = 2 ** 30


def build_model(ckpt: dict, n_heads: int, device: torch.device,
                attn_layers=(3, 10), attn_window: int = 4096) -> EdgeHVN:
    """Production hybrid at the checkpoint's config, with n_heads overridden.

    Weights stay at fresh init: throughput does not depend on weight *values*,
    and n_heads != the checkpoint's would not load anyway.  Shapes, dtypes, block
    plan and ternary wrapping all match production, which is what decides the
    kernels that run.
    """
    num_layers = ckpt["num_layers"]
    block_plan = ["transformer" if i in set(attn_layers) else "deltanet"
                  for i in range(num_layers)]
    model = EdgeHVN(
        vocab_size=ckpt["vocab_size"],
        d_model=ckpt["d_model"],
        num_layers=num_layers,
        n_heads=n_heads,
        use_deltanet=False,
        deltanet_conv_kernel=ckpt.get("deltanet_conv_kernel", 4),
        block_plan=block_plan,
        attn_window=attn_window,
    )
    all_linears = {n: m for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    exclude = ("beta_proj", "alpha_proj", "decoder")
    targets = [n for n in all_linears if not any(p in n for p in exclude)]
    apply_ternary_to_model(model, target_names=targets)
    model = model.to(device)
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
            blk.use_megakernel_scan = True
    return model


def time_config(model, *, device, batch_size, seq_len, effective_batch,
                warmup_steps, timed_steps, vocab_size, seed) -> dict:
    """Time one (batch, heads) point at a FIXED effective batch.

    grad_accum is derived so every arm does the same optimizer maths -- otherwise
    a bigger micro-batch would win partly by taking fewer optimizer steps, which
    is a different change than the one being tested.
    """
    grad_accum = max(1, effective_batch // batch_size)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-9)
    criterion = nn.CrossEntropyLoss()
    g = torch.Generator(device="cpu").manual_seed(seed)

    def micro(i: int):
        x = torch.randint(0, vocab_size, (batch_size, seq_len), generator=g)
        x = x.to(device, non_blocking=True)
        y = torch.roll(x, -1, dims=1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = criterion(logits.reshape(-1, logits.size(-1)), y.reshape(-1))
            loss = loss / grad_accum
        loss.backward()
        if (i + 1) % grad_accum == 0:
            opt.step()
            opt.zero_grad(set_to_none=True)

    for i in range(warmup_steps):
        micro(i)
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    t0 = time.perf_counter()
    for i in range(timed_steps):
        micro(i)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    alloc = torch.cuda.max_memory_allocated() / MIB
    reserved = torch.cuda.max_memory_reserved() / MIB
    opt.zero_grad(set_to_none=True)
    return {
        "seconds": elapsed,
        "grad_accum": grad_accum,
        "tokens_per_sec": timed_steps * batch_size * seq_len / elapsed,
        "peak_alloc_mib": alloc,
        "peak_reserved_mib": reserved,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--heads", type=int, nargs="+", default=[8, 16])
    ap.add_argument("--effective-batch", type=int, default=32,
                    help="held constant across arms; grad_accum is derived")
    ap.add_argument("--warmup-steps", type=int, default=6)
    ap.add_argument("--timed-steps", type=int, default=32)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=str,
                    default="results/throughput_sweep/results.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        sys.exit("CUDA unavailable -- docs/ai_rules.md requires GPU.")

    device = torch.device("cuda")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    props = torch.cuda.get_device_properties(0)
    n_sms = props.multi_processor_count
    card_mib = props.total_memory / MIB
    print(f"GPU: {props.name} | {n_sms} SMs | {card_mib/1024:.4f} GiB")

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    print(f"config: d_model={ckpt['d_model']}, layers={ckpt['num_layers']}, "
          f"checkpoint n_heads={ckpt.get('n_heads')}\n")

    # Arms, plus the baseline entered a second time as the noise-floor arm.
    arms = [(b, h) for h in args.heads for b in args.batches]
    baseline = (2, ckpt.get("n_heads", 8))
    noise_twin = ("noise_twin", baseline)

    samples: dict[str, list[dict]] = {}
    spilled: dict[str, bool] = {}

    def key(b, h):
        return f"b{b}_h{h}"

    print(f"Prediction under test: tok/s climbs with B*H up to ~{n_sms}, then flattens.")
    print(f"Effective batch held at {args.effective_batch} (grad_accum derived).\n")

    for rnd in range(args.repeats):
        todo = [(key(b, h), b, h) for (b, h) in arms] + \
               [(noise_twin[0], baseline[0], baseline[1])]
        for name, b, h in todo:
            try:
                model = build_model(ckpt, h, device)
                r = time_config(model, device=device, batch_size=b,
                                seq_len=args.seq_len,
                                effective_batch=args.effective_batch,
                                warmup_steps=args.warmup_steps,
                                timed_steps=args.timed_steps,
                                vocab_size=ckpt["vocab_size"],
                                seed=1000 * rnd + b * 31 + h)
                del model
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  round {rnd+1}  {name:<12} OOM -- skipped")
                continue
            torch.cuda.empty_cache()

            # A spilled arm is a fit failure, not a slow config (docs/training.md).
            r["spilled"] = r["peak_reserved_mib"] > card_mib
            spilled[name] = spilled.get(name, False) or r["spilled"]
            samples.setdefault(name, []).append(r)
            flag = "  <-- SPILLED to host RAM" if r["spilled"] else ""
            print(f"  round {rnd+1}/{args.repeats}  {name:<12} "
                  f"B*H={b*h:>3}  {r['tokens_per_sec']:8.1f} tok/s  "
                  f"alloc {r['peak_alloc_mib']:7.1f}  "
                  f"reserved {r['peak_reserved_mib']:7.1f} MiB{flag}")
        print()

    def med(name, k):
        return statistics.median(s[k] for s in samples[name])

    base_key = key(*baseline)
    base = med(base_key, "tokens_per_sec")
    noise_pct = (abs(med(noise_twin[0], "tokens_per_sec") - base) / base * 100.0
                 if noise_twin[0] in samples else float("nan"))

    print("=" * 86)
    print(f"{'arm':<12}{'B*H':>6}{'tok/s':>11}{'vs b2':>10}{'alloc MiB':>12}"
          f"{'reserved':>11}{'fit':>10}")
    print("-" * 86)
    for name in list(samples):
        if name == noise_twin[0]:
            continue
        b, h = int(name.split("_")[0][1:]), int(name.split("_")[1][1:])
        t = med(name, "tokens_per_sec")
        print(f"{name:<12}{b*h:>6}{t:>11.1f}{(t-base)/base*100:>9.1f}%"
              f"{med(name,'peak_alloc_mib'):>12.1f}{med(name,'peak_reserved_mib'):>11.1f}"
              f"{'SPILLED' if spilled.get(name) else 'ok':>10}")
    print("=" * 86)
    print(f"\nNoise floor (b2 config vs itself): {noise_pct:.2f}%")

    # Positive control: the smallest batch must be visibly slower per token.
    ctrl_key = key(min(args.batches), baseline[1])
    control_ok = None
    if ctrl_key in samples and ctrl_key != base_key:
        ctrl_delta = (med(ctrl_key, "tokens_per_sec") - base) / base * 100.0
        control_ok = ctrl_delta < -max(noise_pct, 1.0)
        print(f"Positive control ({ctrl_key}, {min(args.batches)*baseline[1]} programs "
              f"for {n_sms} SMs): {ctrl_delta:+.1f}%")
        if not control_ok:
            print("\n*** RUN VOID: the smallest batch was not measurably slower per "
                  "token, so this harness cannot resolve occupancy effects. Its flat "
                  "regions mean nothing. Do not report these numbers.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump({
            "issue": 141,
            "what_this_measures": "training throughput vs batch_size x n_heads",
            "not_a_quality_result": "n_heads changes the architecture; quality unmeasured",
            "gpu": props.name, "sms": n_sms, "card_mib": card_mib,
            "config": vars(args),
            "model": {k: ckpt[k] for k in ("d_model", "num_layers", "vocab_size")},
            "checkpoint_n_heads": ckpt.get("n_heads"),
            "samples": samples,
            "spilled": spilled,
            "noise_floor_pct": noise_pct,
            "positive_control_passed": control_ok,
        }, f, indent=2)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
