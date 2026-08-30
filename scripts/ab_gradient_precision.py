"""A/B: how many mantissa bits do gradients actually need before training breaks?

Author: Rambarun Komaljeet

Why this exists, and why it is NOT the experiment that was originally proposed
-----------------------------------------------------------------------------
The proposal was to replace floating-point multiplication with logarithm-based
addition (Mitchell / "L-Mul") in forward and backward, on the theory that ternary
weights make the forward multiply trivial and gradients may not need full
precision.

Two facts killed the arithmetic half of that before any code was written, both
measured rather than argued (`scripts/bench_logmul_ceiling.py`):

1. **Multiplies run on tensor cores; adds do not.** A bf16 GEMM on this card
   measures 23.2 TFLOP/s. An add-only reduction of identical shape and identical
   memory traffic -- no exponent repack, no bias correction, no int/float
   conversion, i.e. strictly optimistic for log-mul -- measures 0.35 TFLOP/s.
   **66x slower** at the 1B projection shape, 54x at the 48M shape. Even a
   perfect kernel hitting the card's theoretical int32 ALU ceiling (~6 TOPS) is
   still ~4x slower than the GEMM it would replace.
2. **Ternary weights are not ternary during training.** `ternary_quantize_ste`
   (`hvn/edge/ternary.py:108`) returns a *float* tensor holding
   ``{-scale, 0, +scale}``; the hardware still issues a full bf16 GEMM. Ternary
   is a post-``freeze()`` storage format, not an arithmetic one. Issue #131
   established this while scoping the MatMul-free work and found the real decode
   cost was the quantization chain (395 ms), not the GEMM (66 ms).

What survives is the *question underneath*: *how much precision do gradients
need?* On a bandwidth-bound machine the payoff for answering it is *fewer bytes
moved* -- fp8 gradient traffic over PCIe for the CPU-offload path (#114 was
opened for parked-block storage and closed without a convergence check), smaller
accumulation buffers -- not fewer multiplies. That is what this harness measures.

**This measures tolerance, not speed.** No arm here is faster than baseline; the
emulator is strictly slower. A speed claim only becomes available if a bit-width
survives, and then it has to be measured separately in the real transfer path.

The measurement
---------------
Gradients are rounded to *n* mantissa bits (fp32 exponent kept), which is exactly
what a reduced-precision storage format does to them:

===========  ======================  ============================================
mantissa     nominal format          expectation
===========  ======================  ============================================
23           fp32                    known-answer cell: a no-op, must be IDENTICAL
7            bf16                    already the training dtype; must be harmless
3            fp8 e4m3                the interesting rung
2            fp8 e5m2                the aggressive rung
0            below any real format   POSITIVE CONTROL: must visibly break
===========  ======================  ============================================

Round-to-nearest-**even on magnitude**, not truncation. Truncation biases every
gradient toward zero, which would confound "not enough precision" with "shrunk
the learning rate" -- a different finding wearing the same clothes. Stochastic
rounding is offered as a second family (``*_sr`` arms) because SR makes the error
zero-mean and lets it average out across accumulation, and it is the standard
reason low-bit gradients work at all. If RN dies at 3 bits and SR survives, that
difference *is* the result.

Falsifiability check, asked and answered BEFORE running (docs/ai_rules.md)
--------------------------------------------------------------------------
*"If gradients were NOT tolerant of low precision, would this catch it?"*

Yes -- but only if the sweep reaches a width that genuinely breaks. That is what
``m0_control`` is for: with the mantissa entirely gone every gradient collapses
to a signed power of two. **If m0_control trains as well as baseline, the
injection is not reaching the gradients and the whole run is VOID** -- a
harness that cannot fail cannot publish a number. The verdict block asserts this
rather than leaving it to the reader, because "every arm looked fine" is exactly
how #138 shipped a 6.8x from a kernel that could take one correct step.

Two further guards, both from this repo's retraction history:

* **Known-answer cell** (``fp32_twin``): baseline run again under a different
  name. Under ``--deterministic`` it must come out bit-identical. A non-zero
  twin drift means the noise floor is not zero and no gap below it is real.
* **Identity self-test**, run before any GPU time is spent: the quantizer at 23
  bits must be an exact identity on random data, and stochastic rounding must be
  unbiased. Catches a broken injection path in milliseconds instead of after a
  six-hour sweep.

Where the injection happens, and why it is after the clip
---------------------------------------------------------
``--site param`` quantizes ``p.grad`` **after** ``clip_grad_norm_``, immediately
before ``opt.step()``. That is the realistic order for the thing being modelled:
backward produces gradients, they are clipped, and *then* they would be cast down
for storage or transfer. Quantizing before the clip would let the renormalisation
partially undo the perturbation and understate the damage.

``--site activation`` instead quantizes the gradient flowing backward *between
blocks* (identity in forward, quantize in backward). That answers the different
question the original proposal was really asking -- whether the backprop *signal*
tolerates low precision -- and it is strictly harsher, because the error
compounds through depth rather than landing once at the optimizer.

Grading
-------
Held-out loss on shared probe batches, never train loss (#132's verdict was
retracted for grading arms on train loss in a multi-pass regime). The practical
threshold is this repo's documented +/-0.011 nats held-out noise; the twin drift
is reported alongside so the floor is checkable rather than assumed.

Optimizer note: every real run here uses Muon (#86/#129), so Muon is the default.
Muon orthogonalises gradients through Newton-Schulz, which is a *nonlinear*
function of the gradient -- low-precision input may behave differently than it
would into AdamW. The result is optimizer-specific and should not be quoted for
a run that uses something else. ``--optimizer adamw`` runs the contrast.

Usage::

    # self-test only, no GPU time
    python scripts/ab_gradient_precision.py --selftest-only

    # the main sweep
    python scripts/ab_gradient_precision.py --steps 1500

    # harsher site, and the stochastic-rounding contrast
    python scripts/ab_gradient_precision.py --site activation \\
        --arms fp32 fp32_twin m3_fp8e4m3 m3_sr m0_control
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional, Sequence

# MUST come before torch initializes CUDA. Importing pyarrow's C extension
# after the CUDA driver is armed aborts the process with 0xC0000374
# (STATUS_HEAP_CORRUPTION) -- a native crash with no Python traceback, on
# torch 2.12.1+cu126 / pyarrow 24 / Windows. `train_edge_conversational`
# imports pandas LAZILY inside `_download_language_foundation`, so this only
# fires on a cache MISS, which is why the main trainer has never hit it.
# Verified boundary:
#   torch.cuda.is_available(); import pyarrow.lib   -> 0xC0000374
#   import pyarrow.lib; torch.randn(8, device=cuda) -> fine
import pandas as _pandas_preload  # noqa: F401

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.ab_prodigy import (                                  # noqa: E402
    as_pair, enable_scan_flags, evaluate, require_cuda,
)
from hvn.optimizers import build_optimizer                        # noqa: E402

GPT2_VOCAB_SIZE = 50257
FP32_MANTISSA_BITS = 23

#: arm name -> (mantissa bits kept, stochastic rounding)
#:
#: `fp32` is the baseline. `fp32_twin` is `fp32` again -- the known-answer cell.
#: `m0_control` is the positive control: it MUST break, or the harness is void.
#: sentinel: zero the gradient outright rather than round it.
ZERO_GRAD = -1

ARMS: Dict[str, dict] = {
    "fp32":       {"bits": 23, "stochastic": False, "format": "fp32 (baseline)"},
    "fp32_twin":  {"bits": 23, "stochastic": False, "format": "fp32 (twin / known-answer)"},
    "m7_bf16":    {"bits": 7,  "stochastic": False, "format": "bf16"},
    "m3_fp8e4m3": {"bits": 3,  "stochastic": False, "format": "fp8 e4m3"},
    "m2_fp8e5m2": {"bits": 2,  "stochastic": False, "format": "fp8 e5m2"},
    "m0_pow2":    {"bits": 0,  "stochastic": False, "format": "signed power-of-2"},
    "zero_control": {"bits": ZERO_GRAD, "stochastic": False,
                     "format": "SANITY: gradients zeroed"},
    # stochastic-rounding contrast
    "m3_sr":      {"bits": 3,  "stochastic": True,  "format": "fp8 e4m3 + SR"},
    "m2_sr":      {"bits": 2,  "stochastic": True,  "format": "fp8 e5m2 + SR"},
    "m0_sr":      {"bits": 0,  "stochastic": True,  "format": "signed power-of-2 + SR"},
}

DEFAULT_ARMS = ["fp32", "fp32_twin", "m7_bf16", "m3_fp8e4m3", "m2_fp8e5m2",
                "m0_pow2", "zero_control"]

# Why `zero_control` exists, and why `m0` is NOT the control it was first
# written as. A 20-step smoke run had `m0` (0 mantissa bits) land 0.00128 nats
# BETTER than baseline, which tripped the void check. Measuring the injection
# directly explained it rather than leaving it a mystery:
#
#     0 bits: max rel err 0.3333  mean 0.16912   (20 distinct magnitudes)
#     2 bits: max rel err 0.1111  mean 0.04491
#
# Zero mantissa bits keeps the full 8-bit EXPONENT, so a gradient is rounded to
# the nearest signed power of two -- at most 33% off, ~17% on average. That is a
# perfectly workable training signal (it is close to signSGD-with-magnitude), and
# Muon makes it milder still, since Newton-Schulz orthogonalisation normalises
# magnitude away before the update. So `m0` is a legitimate aggressive RUNG, and
# a harness that treats it as "must break" will void itself on a correct result.
#
# `zero_control` is the arm that genuinely cannot pass: gradients replaced with
# zero, so no learning is possible and held-out loss must stay at its init value.
# If it improves, the injection is not reaching the optimizer and every other
# number in the table is meaningless.


# --------------------------------------------------------------------------
# the quantizer
# --------------------------------------------------------------------------

def quantize_mantissa(t: torch.Tensor, bits: int, stochastic: bool = False,
                      generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Round ``t`` to ``bits`` mantissa bits, keeping the fp32 exponent.

    Operates on the raw IEEE-754 bit pattern. Adding ``1 << drop`` to a truncated
    pattern carries correctly into the exponent field, and because the sign bit
    is the *top* bit, the same arithmetic rounds magnitude symmetrically for
    negative values -- so this is round-half-to-even on |x|, which is unbiased.

    Non-finite values are passed through untouched: rounding an inf or a nan has
    no meaning here, and letting a carry manufacture an inf out of a large finite
    would be a silent corruption rather than a precision effect.
    """
    if bits >= FP32_MANTISSA_BITS:
        return t
    if bits < 0:
        raise ValueError(f"bits must be >= 0, got {bits}")

    orig_dtype = t.dtype
    x = t.detach().float().contiguous()
    i = x.view(torch.int32)

    drop = FP32_MANTISSA_BITS - bits
    mask = (1 << drop) - 1
    keep = ~mask

    if stochastic:
        # Uniform dither in [0, 2^drop) then truncate: the standard unbiased SR.
        r = torch.randint(0, 1 << drop, x.shape, device=x.device,
                          dtype=torch.int32, generator=generator)
        out_i = (i + r) & keep
    else:
        rem = i & mask
        half = 1 << (drop - 1)
        # tie-to-even on the surviving low bit
        odd = ((i >> drop) & 1).bool()
        up = (rem > half) | ((rem == half) & odd)
        out_i = (i & keep) + torch.where(up, torch.full_like(i, 1 << drop),
                                         torch.zeros_like(i))

    out = out_i.view(torch.float32)
    # Preserve the original wherever it was not finite, or where rounding
    # manufactured a non-finite from a finite input.
    finite = torch.isfinite(x)
    out = torch.where(finite & torch.isfinite(out), out, x)
    return out.to(orig_dtype)


class _QuantGrad(torch.autograd.Function):
    """Identity forward; quantize the gradient on the way back."""

    @staticmethod
    def forward(ctx, x, bits, stochastic):
        ctx.bits = bits
        ctx.stochastic = stochastic
        return x

    @staticmethod
    def backward(ctx, g):
        if ctx.bits == ZERO_GRAD:
            return torch.zeros_like(g), None, None
        return quantize_mantissa(g, ctx.bits, ctx.stochastic), None, None


def attach_activation_hooks(model, bits: int, stochastic: bool) -> int:
    """Quantize the gradient flowing backward out of every block.

    A forward hook returning a wrapped output is the least invasive way in: it
    does not touch any block's own code, so the arms cannot differ by anything
    except the quantization.
    """
    if bits >= FP32_MANTISSA_BITS:
        return 0

    def hook(_module, _inp, out):
        if isinstance(out, torch.Tensor):
            return _QuantGrad.apply(out, bits, stochastic)
        if isinstance(out, tuple) and out and isinstance(out[0], torch.Tensor):
            return (_QuantGrad.apply(out[0], bits, stochastic),) + tuple(out[1:])
        return out

    n = 0
    for blk in getattr(model, "blocks", []):
        blk.register_forward_hook(hook)
        n += 1
    return n


def quantize_param_grads(model, bits: int, stochastic: bool,
                         generator: Optional[torch.Generator] = None) -> None:
    """In-place quantize every ``p.grad``. Call AFTER clipping, before step()."""
    if bits >= FP32_MANTISSA_BITS:
        return
    for p in model.parameters():
        if p.grad is None:
            continue
        if bits == ZERO_GRAD:
            p.grad.zero_()
        else:
            p.grad.copy_(quantize_mantissa(p.grad, bits, stochastic, generator))


# --------------------------------------------------------------------------
# self-test -- runs before any GPU time is spent
# --------------------------------------------------------------------------

def selftest(device) -> None:
    """Assert the quantizer is what it claims. Cheap; catches a dead injection.

    A harness whose perturbation silently does nothing produces a table of arms
    that all match baseline and reads as "gradients are perfectly tolerant".
    That is the #138 failure mode, so these are assertions, not prints.
    """
    torch.manual_seed(0)
    x = torch.randn(4096, device=device) * torch.exp(
        torch.randn(4096, device=device) * 4)      # wide dynamic range

    # 1. 23 bits is an exact identity.
    idt = quantize_mantissa(x, 23)
    assert torch.equal(idt, x), "23-bit quantize is not an identity"

    # 2. 0 bits must clear the mantissa field entirely.
    z = quantize_mantissa(x, 0)
    mant = z.contiguous().view(torch.int32) & ((1 << 23) - 1)
    assert int(mant.abs().sum()) == 0, "0-bit quantize left mantissa bits set"

    # 3. It must actually change the data at the rungs being tested.
    for b in (2, 3, 7):
        q = quantize_mantissa(x, b)
        assert not torch.equal(q, x), f"{b}-bit quantize was a no-op"
        rel = ((q - x).abs() / x.abs().clamp(min=1e-30)).max().item()
        assert rel < 2.0 ** -(b), f"{b}-bit relative error {rel:.3e} too large"

    # 4. The RN-vs-SR distinction, tested where it is actually sharp.
    #
    # A first attempt compared bias on wide-dynamic-range random data against a
    # fixed 1e-3 threshold. It FAILED at 1.8e-3 -- not because SR is biased, but
    # because a heavy-tailed sum is dominated by a few large elements, so the
    # estimator's own Monte-Carlo error exceeded the threshold. That is exactly
    # the mistake docs/state.md records for the fused-quantizer A/B (rejected on
    # 3.08e-03 when the same arm's reproducibility was 2.58e-03): a threshold
    # tighter than the measurement's noise floor rejects correct code.
    #
    # So the test is built to be sharp instead. Values are placed at a KNOWN
    # fraction of a ULP above a representable point, where the two schemes have
    # exactly predictable behaviour:
    #   RN -- deterministic, must round every one of them DOWN (bias = -f ULP)
    #   SR -- must round up with probability f, so the mean converges to the
    #         true value, and "converges" is judged against the binomial
    #         standard error rather than a made-up constant.
    T = 512
    for b in (2, 3):
        drop = FP32_MANTISSA_BITS - b
        frac = 0.25                                  # a quarter of a ULP up
        # 0x40000000 is the bit pattern of 2.0; masking the low bits lands it
        # exactly on a representable point of the b-bit grid.
        base_i = torch.full((2048,), 0x40000000, device=device,
                            dtype=torch.int32) & ~((1 << drop) - 1)
        ulp_i = 1 << drop
        v_i = base_i + int(frac * ulp_i)
        v = v_i.view(torch.float32)
        grid = base_i.view(torch.float32)
        ulp = (( base_i + ulp_i).view(torch.float32) - grid)

        rn = quantize_mantissa(v, b)
        rn_up = ((rn - grid) / ulp).mean().item()
        assert abs(rn_up) < 1e-6, (
            f"RN at {b} bits rounded {rn_up:.4f} ULP up on a 0.25-ULP input; "
            "round-to-nearest must round these DOWN, every time")

        draws = torch.stack([((quantize_mantissa(v, b, stochastic=True) - grid)
                              / ulp).mean() for _ in range(T)])
        sr_up = draws.mean().item()
        # binomial SE of the mean of (2048 x T) Bernoulli(frac) draws
        se = (frac * (1 - frac) / (2048 * T)) ** 0.5
        assert abs(sr_up - frac) < 4 * se, (
            f"SR at {b} bits rounded up {sr_up:.5f} of the time, expected "
            f"{frac} +/- {4*se:.5f} — stochastic rounding is biased")
        print(f"[selftest] {b:>2} bits: RN rounds up {rn_up:.5f} (must be 0) | "
              f"SR rounds up {sr_up:.5f} vs expected {frac} +/-{4*se:.5f}")

    # 5. Non-finite passthrough.
    nf = torch.tensor([float("inf"), float("-inf"), float("nan"), 0.0],
                      device=device)
    q = quantize_mantissa(nf, 2)
    assert torch.isinf(q[0]) and torch.isinf(q[1]) and torch.isnan(q[2]) \
        and float(q[3]) == 0.0, "non-finite passthrough broken"

    print("[selftest] PASS — quantizer is an identity at 23 bits, "
          "destroys the mantissa at 0, and is unbiased under SR")


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def build_model(args, device):
    """Fresh-init EdgeHVN. Identical in every arm -- only the gradients differ."""
    from hvn.edge.model import EdgeHVN
    from hvn.edge.ternary import apply_ternary_to_model

    layers = args.num_layers
    attn = (max(1, layers // 4), max(2, (3 * layers) // 4))
    plan = ["transformer" if i in attn else "deltanet" for i in range(layers)]
    model = EdgeHVN(
        vocab_size=args.vocab_size, d_model=args.d_model, num_layers=layers,
        n_heads=args.n_heads, block_plan=plan, attn_window=args.attn_window,
        tie_embeddings=True, grad_checkpoint=args.grad_checkpoint,
    ).to(device)
    lin = {n for n, m in model.named_modules() if isinstance(m, nn.Linear)}
    ex = ("beta_proj", "alpha_proj", "decoder")
    apply_ternary_to_model(model, target_names=[n for n in lin
                                                if not any(p in n for p in ex)])
    return model


def run_arm(arm: str, args, batches, probes, device) -> dict:
    spec = ARMS[arm]
    bits, stochastic = spec["bits"], spec["stochastic"]

    # Identical init and identical data order in every arm: the arms differ in
    # gradient precision alone.
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    model = build_model(args, device)
    enable_scan_flags(model)
    n_hooks = 0
    if args.site == "activation":
        n_hooks = attach_activation_hooks(model, bits, stochastic)

    n_params = sum(p.numel() for p in model.parameters())
    opt, desc = build_optimizer(model, args.optimizer, lr=args.aux_lr,
                                weight_decay=args.weight_decay,
                                muon_lr=args.muon_lr)
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]

    print(f"\n[ab] === arm '{arm}' ({spec['format']}, {bits} mantissa bits, "
          f"SR={stochastic}, site={args.site}, {n_params:,} params) ===\n"
          f"[ab]     {desc}"
          + (f"  |  {n_hooks} activation hooks" if args.site == "activation" else ""))

    traj = [{"step": 0, "heldout_loss": evaluate(model, probes, device),
             "wall_sec": 0.0}]
    print(f"[ab]   step {0:>5}  held-out {traj[0]['heldout_loss']:.5f}")

    curve: List[dict] = []
    diverged_at = None
    model.train()
    t0 = time.time()
    warmup_done = args.warmup_steps <= 0

    for i, (bx, by) in enumerate(batches, start=1):
        x = bx.to(device, non_blocking=True)
        y = by.to(device, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        loss.backward()

        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        # AFTER the clip: the realistic order for a storage/transfer format, and
        # quantizing first would let the renormalisation hide the damage.
        if args.site == "param":
            quantize_param_grads(model, bits, stochastic, gen)

        if not warmup_done:
            s = (i - 1) / max(1, args.warmup_steps)
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = b * s
            if i >= args.warmup_steps:
                warmup_done = True

        opt.step()
        opt.zero_grad(set_to_none=True)
        if warmup_done:
            cos.step()

        lv = float(loss.detach())
        curve.append({"step": i, "loss": lv, "grad_norm": gnorm,
                      "lr": opt.param_groups[0]["lr"]})
        if diverged_at is None and (lv != lv or lv == float("inf")):
            diverged_at = i
            print(f"[ab]   *** non-finite loss at step {i} — arm diverged")
            break

        if i % args.eval_every == 0:
            hl = evaluate(model, probes, device)
            wall = time.time() - t0
            traj.append({"step": i, "heldout_loss": hl, "wall_sec": wall})
            recent = [c["loss"] for c in curve[-args.eval_every:]]
            print(f"[ab]   step {i:>5}  held-out {hl:.5f}  "
                  f"train {sum(recent)/len(recent):.5f}  "
                  f"|g| {gnorm:.3f}  ({wall/60:.1f} min)", flush=True)
            model.train()

    wall_total = time.time() - t0
    out = {
        "arm": arm, "mantissa_bits": bits, "stochastic": stochastic,
        "format": spec["format"], "site": args.site, "params": n_params,
        "diverged_at_step": diverged_at,
        "wall_clock_sec": wall_total,
        "final_heldout_loss": traj[-1]["heldout_loss"],
        "start_heldout_loss": traj[0]["heldout_loss"],
        "mean_grad_norm": (sum(c["grad_norm"] for c in curve) / len(curve)
                           if curve else None),
        "trajectory": traj, "curve": curve,
    }
    del model, opt
    torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--arms", nargs="+", default=DEFAULT_ARMS,
                   help=f"any of: {' '.join(ARMS)}")
    p.add_argument("--site", choices=("param", "activation"), default="param",
                   help="param: quantize p.grad after clipping (models fp8 "
                        "gradient storage/transfer). activation: quantize the "
                        "gradient between blocks (models a low-precision "
                        "backward signal; strictly harsher).")
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--eval-every", type=int, default=250)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--num-layers", type=int, default=12)
    p.add_argument("--n-heads", type=int, default=16)
    p.add_argument("--attn-window", type=int, default=4096)
    p.add_argument("--grad-checkpoint", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--optimizer", default="muon", choices=("muon", "adamw"))
    p.add_argument("--muon-lr", type=float, default=0.02,
                   help="pretrain-regime value per #129")
    p.add_argument("--aux-lr", type=float, default=None,
                   help="Resolved from --optimizer when not given. Under muon "
                        "this is the AUXILIARY group only (embeddings/norms) at "
                        "2e-5, with the real work done at --muon-lr. Under adamw "
                        "the SAME value drives every parameter "
                        "(hvn/optimizers.py:604), so inheriting 2e-5 would run "
                        "AdamW ~15x under this repo's standard 3e-4 and produce "
                        "a barely-trained arm that is insensitive to everything "
                        "-- the #148 'mis-set LR' confound, wearing the costume "
                        "of a precision result.")
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--probe-batches", type=int, default=16)
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--code-corpus", default=None)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--vocab-size", type=int, default=GPT2_VOCAB_SIZE)
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--noise-threshold", type=float, default=0.011,
                   help="documented held-out noise floor for this repo")
    p.add_argument("--deterministic", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--selftest-only", action="store_true")
    p.add_argument("--out", default="results/grad_precision/ab_grad_precision.json")
    args = p.parse_args(argv)

    # Resolve the LR against the optimizer rather than sharing one default.
    # 3e-4 is this repo's standard (train_edge_conversational.py:3231,
    # scripts/ab_dynamic_lr.py:264); 2e-5 is Muon's auxiliary-group value.
    if args.aux_lr is None:
        args.aux_lr = 2e-5 if args.optimizer == "muon" else 3e-4
        print(f"[ab] --aux-lr not given; using {args.aux_lr:g} for "
              f"--optimizer {args.optimizer}")
    if args.optimizer == "adamw" and args.aux_lr < 1e-4:
        print(f"[ab] WARNING: AdamW at lr={args.aux_lr:g} is far below this "
              f"repo's 3e-4. A barely-trained arm is insensitive to gradient "
              f"precision for reasons that have nothing to do with precision.")

    if args.deterministic:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=False)
        print("[ab] deterministic ON — fp32 and fp32_twin must come out IDENTICAL")

    device = require_cuda()
    selftest(device)
    if args.selftest_only:
        return 0

    import train_edge_conversational as tec

    tok = tec.load_tokenizer(args.tokenizer)
    # #155: len(tokenizer), never tokenizer.vocab_size (added tokens excluded).
    if args.vocab_size != len(tok):
        args.vocab_size = len(tok)
    torch.manual_seed(args.seed)
    loader, _v, stats = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0,
        code_corpus=args.code_corpus,
        fim=({"rate": 0.5, "spm_rate": 0.5, "seed": 0} if args.code_corpus else None))
    print(f"[ab] data: {stats}")

    pairs = []
    for batch in loader:
        x, y = as_pair(batch)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= args.probe_batches + args.steps:
            break
    probes, batches = pairs[:args.probe_batches], pairs[args.probe_batches:]
    print(f"[ab] {len(probes)} held-out probes + {len(batches)} identical "
          f"training batches for every arm")

    results: Dict[str, dict] = {}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for arm in args.arms:
        if arm not in ARMS:
            raise SystemExit(f"unknown arm {arm!r}; known: {' '.join(ARMS)}")
        results[arm] = run_arm(arm, args, batches, probes, device)
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump({"question": "gradient mantissa-bit tolerance",
                       "config": vars(args), "arms": results}, fh, indent=2)

    # ---- verdict -----------------------------------------------------------
    base = results.get("fp32")
    print("\n[ab] ============ gradient precision sweep "
          f"(site={args.site}) ============")
    print(f"[ab] {'arm':<14}{'bits':>6}{'format':>24}{'held-out':>12}"
          f"{'vs fp32':>12}{'diverged':>10}")
    for name, r in results.items():
        d = (r["final_heldout_loss"] - base["final_heldout_loss"]) if base else float("nan")
        dv = r["diverged_at_step"]
        print(f"[ab] {name:<14}{r['mantissa_bits']:>6}{r['format']:>24}"
              f"{r['final_heldout_loss']:>12.5f}{d:>+12.5f}"
              f"{(str(dv) if dv else '-'):>10}")

    verdict, void = None, []

    # known-answer cell
    if "fp32" in results and "fp32_twin" in results:
        drift = abs(results["fp32"]["final_heldout_loss"]
                    - results["fp32_twin"]["final_heldout_loss"])
        print(f"\n[ab] KNOWN-ANSWER CELL — fp32 vs fp32_twin: {drift:.3e}")
        if drift > 0.0:
            void.append(f"twin drift {drift:.3e} is not zero")
            print("[ab] *** two identical arms did not match under "
                  "--deterministic; any gap below this drift is not resolvable")
    else:
        print("\n[ab] NOTE: no fp32_twin arm — the noise floor is unmeasured, "
              "so every gap below the assumed threshold is unverified")

    # Positive control -- the arm whose whole job is to fail.
    #
    # The test is against the arm's OWN starting loss, not against baseline:
    # with gradients zeroed no learning is possible, so held-out loss must sit
    # where it started. Comparing to baseline instead would let a harness where
    # both arms silently failed to train look like agreement.
    ctrl = results.get("zero_control")
    if ctrl:
        moved = ctrl["start_heldout_loss"] - ctrl["final_heldout_loss"]
        learned = moved > 10 * args.noise_threshold
        print(f"[ab] SANITY CONTROL — zero_control moved "
              f"{ctrl['start_heldout_loss']:.5f} -> "
              f"{ctrl['final_heldout_loss']:.5f} ({moved:+.5f} nats)")
        if learned:
            void.append("zero_control learned despite zeroed gradients")
            print("[ab] *** zero_control IMPROVED with its gradients set to "
                  "zero. Learning happened that the injection never touched, so "
                  "the injection is not reaching the optimizer. This harness "
                  "CANNOT FAIL and its numbers are not evidence.")
        if base:
            print(f"[ab]   (for scale, fp32 moved "
                  f"{base['start_heldout_loss'] - base['final_heldout_loss']:+.5f})")
    else:
        void.append("no zero_control arm was run")
        print("[ab] *** no zero_control arm — run one, or the sweep cannot "
              "demonstrate it is able to detect damage at all")

    if void:
        verdict = "VOID: " + "; ".join(void)
        print(f"\n[ab] *** RUN VOID — {verdict}")
        print("[ab] Do NOT report these numbers as a precision threshold.")
    elif base:
        graded = {n: r for n, r in results.items()
                  if n not in ("fp32", "fp32_twin", "zero_control")}
        ok = [r for r in graded.values()
              if r["diverged_at_step"] is None
              and (r["final_heldout_loss"] - base["final_heldout_loss"])
              <= args.noise_threshold]
        if ok:
            best = min(ok, key=lambda r: r["mantissa_bits"])
            verdict = (f"gradients tolerate {best['mantissa_bits']} mantissa bits "
                       f"({best['format']}) within the {args.noise_threshold} "
                       f"nat noise floor")
        elif graded:
            verdict = (f"no reduced width stayed within {args.noise_threshold} "
                       f"nats — gradients need more than "
                       f"{max(r['mantissa_bits'] for r in graded.values())} bits here")
        else:
            verdict = "no graded arms were run"
        print(f"\n[ab] VERDICT: {verdict}")
        print("[ab] NOTE: this is a TOLERANCE result, not a speedup. Turning a "
              "surviving width into wall-clock requires measuring it in the "
              "real transfer path (#114), and no arm here is faster than fp32.")

    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"question": "gradient mantissa-bit tolerance",
                   "config": vars(args), "arms": results,
                   "verdict": verdict, "void_reasons": void}, fh, indent=2)
    print(f"[ab] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
