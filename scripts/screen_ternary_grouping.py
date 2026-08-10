"""Cheap screen: does group-wise ternary scaling reduce quantization damage? (Issue #145)

Author: Rambarun Komaljeet

``ternary_quantize_ste`` (``hvn/edge/ternary.py:46``) uses **one scalar scale per
weight matrix** -- BitNet b1.58 absmean, ``w.abs().mean()`` with no ``dim=``. At
d=512 that is 262,144 weights sharing a single normaliser: rows of small weights
round to all-zero, rows of large weights saturate.

Ternary's measured cost today is **+20% PPL** (dense fp 119.4 vs dense ternary
143.2, docs/state.md). Per-row / per-group scales are what every int4 scheme uses,
and ``hvn/compress.py`` already has group-wise code -- but only for post-training
compression, never for the training path.

**This is the screen, not the feature.** Real group-wise QAT changes how the model
trains and needs a from-scratch A/B (hours). This spends ~10 minutes first: take
the already-trained 83M, re-quantize its shadow weights under each scheme, and
measure the damage. If finer scaling cannot reduce damage on weights that already
exist, it will not help during training either.

**Known-answer cell.** The ``per_tensor`` arm re-implements what ``TernaryLinear``
already does, so it must reproduce the untouched ternary model's held-out loss
**exactly**. If it does not, this harness quantizes differently from production
and every other arm is measuring the wrong thing. The run reports that delta first
and calls itself VOID if it is non-zero.

**Two metrics, deliberately.** Held-out loss is what we care about but is
confounded -- this checkpoint was *trained* with per-tensor QAT, so its weights
are adapted to that scheme. Weight reconstruction error
``||W_q - W||_F / ||W||_F`` is adaptation-free and must improve by construction
with finer scaling; if it does not, the implementation is wrong. Reconstruction
improving while loss does not would mean the model has adapted to per-tensor
quantization -- informative, and an argument that only a QAT A/B can settle this.

Usage::

    python scripts/screen_ternary_grouping.py --checkpoint <ckpt>
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hvn.edge.ternary import TernaryLinear                        # noqa: E402

EPS = 1e-6


def quantize(w: torch.Tensor, mode: str, group: Optional[int] = None) -> torch.Tensor:
    """Dequantized ternary weight under a given scale granularity.

    ``per_tensor`` must match ``ternary_quantize_ste`` exactly -- it is the
    known-answer arm, and any drift here invalidates the whole screen.
    """
    if mode == "fp":
        return w.clone()

    if mode == "per_tensor":
        s = w.abs().mean().clamp(min=EPS)
        return torch.clamp(torch.round(w / s), -1, 1) * s

    if mode == "per_row":
        # One scale per OUTPUT channel. Storage cost is out_features extra
        # scalars against out*in codes -- ~0.4% at d=512.
        s = w.abs().mean(dim=1, keepdim=True).clamp(min=EPS)
        return torch.clamp(torch.round(w / s), -1, 1) * s

    if mode == "group":
        assert group and group > 0
        out, inn = w.shape
        pad = (-inn) % group
        wp = F.pad(w, (0, pad)) if pad else w
        g = wp.view(out, -1, group)
        s = g.abs().mean(dim=2, keepdim=True).clamp(min=EPS)
        q = torch.clamp(torch.round(g / s), -1, 1) * s
        return q.view(out, -1)[:, :inn].contiguous()

    raise ValueError(f"unknown mode {mode!r}")


def apply_scheme(model: nn.Module, mode: str, group: Optional[int]) -> dict:
    """Replace every TernaryLinear with a dense Linear holding its quantized weight.

    Swapping the module out (rather than poking TernaryLinear internals) means the
    forward path is a plain F.linear on exactly the weights a deployment with this
    scheme would use -- no re-quantization happening behind the measurement.
    """
    err_num, err_den, n_layers, n_weights = 0.0, 0.0, 0, 0
    targets = [(n, m) for n, m in model.named_modules() if isinstance(m, TernaryLinear)]
    for name, mod in targets:
        w = mod.weight.detach()
        wq = quantize(w.float(), mode, group).to(w.dtype)
        err_num += float((wq - w).pow(2).sum())
        err_den += float(w.pow(2).sum())
        n_layers += 1
        n_weights += w.numel()

        lin = nn.Linear(mod.in_features, mod.out_features,
                        bias=mod.bias is not None).to(w.device, w.dtype)
        lin.weight.data.copy_(wq)
        if mod.bias is not None:
            lin.bias.data.copy_(mod.bias.detach())

        parent = model
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], lin)

    return {
        "layers_quantized": n_layers,
        "weights_quantized": n_weights,
        "recon_rel_error": (err_num / err_den) ** 0.5 if err_den else float("nan"),
    }


@torch.no_grad()
def heldout_loss(model, probes, device) -> float:
    model.eval()
    tot, n = 0.0, 0
    for bx, by in probes:
        x, y = bx.to(device), by.to(device)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        tot += float(loss)
        n += 1
    return tot / max(1, n)


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--checkpoint",
                   default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt")
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--tokenizer", default="gpt2")
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--probe-batches", type=int, default=32)
    p.add_argument("--bpb-text", default="data_cache/heldout_fineweb.txt")
    p.add_argument("--bpb-max-bytes", type=int, default=2_000_000)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="results/ternary_grouping/screen.json")
    args = p.parse_args(argv)

    if not torch.cuda.is_available():
        sys.exit("CUDA required (docs/ai_rules.md).")
    device = torch.device("cuda")

    import train_edge_conversational as tec
    from scripts.ab_prodigy import as_pair

    tok = tec.load_tokenizer(args.tokenizer)
    torch.manual_seed(args.seed)
    loader, _v, stats = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=args.seq_len, batch_size=args.batch_size,
        data_dir=args.data_dir, val_fraction=0.0, num_workers=0)
    probes = []
    for batch in loader:
        x, y = as_pair(batch)
        probes.append((x.clone(), y.clone()))
        if len(probes) >= args.probe_batches:
            break
    print(f"[screen] {len(probes)} probe batches; data {stats['source']}")

    def fresh_model():
        m, _step, _ck = tec.load_checkpoint_for_resume(args.checkpoint, device)
        for blk in (m.blocks or []):
            if hasattr(blk, "use_triton_scan"):
                blk.use_triton_scan = True
                blk.use_megakernel_scan = True
        return m

    # Reference: the untouched ternary model, exactly as production runs it.
    native = fresh_model()
    native_loss = heldout_loss(native, probes, device)
    n_tern = sum(isinstance(m, TernaryLinear) for m in native.modules())
    print(f"[screen] native ternary model: held-out {native_loss:.5f} "
          f"({n_tern} TernaryLinear layers)")
    del native
    torch.cuda.empty_cache()

    schemes = [("fp", "fp", None), ("per_tensor", "per_tensor", None),
               ("per_row", "per_row", None), ("group_128", "group", 128),
               ("group_64", "group", 64), ("group_32", "group", 32)]

    results = {}
    for label, mode, group in schemes:
        model = fresh_model()
        info = apply_scheme(model, mode, group)
        hl = heldout_loss(model, probes, device)
        bpb = None
        if args.bpb_text and os.path.exists(args.bpb_text):
            from eval.bits_per_byte import bits_per_byte
            with open(args.bpb_text, encoding="utf-8", errors="replace") as fh:
                text = fh.read(args.bpb_max_bytes)
            bpb = bits_per_byte(model, tok, text, seq_len=args.seq_len,
                                device=device)["bits_per_byte"]
        results[label] = {"held_out_loss": hl, "bits_per_byte": bpb, **info}
        print(f"[screen] {label:<11} held-out {hl:.5f}  "
              f"bpb {bpb if bpb is None else f'{bpb:.4f}'}  "
              f"recon_err {info['recon_rel_error']:.5f}")
        del model
        torch.cuda.empty_cache()

    # ---- known-answer cell ------------------------------------------------
    drift = abs(results["per_tensor"]["held_out_loss"] - native_loss)
    print(f"\n[screen] KNOWN-ANSWER — my per_tensor vs native TernaryLinear: {drift:.3e}")
    void = drift > 1e-9
    if void:
        print("[screen] *** VOID: this harness does not reproduce production "
              "quantization, so every other arm is measuring something else. "
              "Do NOT report these numbers.")

    fp = results["fp"]["held_out_loss"]
    pt = results["per_tensor"]["held_out_loss"]

    # The `fp` arm is NOT an upper bound, and this run is what proved it. In a
    # QAT model the forward pass ALWAYS used quantized weights; the fp32 shadow
    # weight is an optimizer accumulator that was never itself the model's
    # function. Feeding it directly gives a different, untrained network -- here
    # it scores far WORSE than the quantized model. So there is no "damage to
    # recover" available from this checkpoint, and any "% recovered" computed
    # against it would be a fabricated denominator. The dense-fp reference in
    # docs/state.md (119.4 vs 143.2) came from a separately TRAINED fp arm,
    # which is the only way to get one.
    print(f"\n[screen] shadow-weight (fp) arm: {fp:.5f} vs per_tensor {pt:.5f} "
          f"-> {'fp WORSE (expected for QAT: shadow weights were never the '
               'forward function)' if fp > pt else 'fp better'}")
    print(f"[screen] {'scheme':<12}{'held-out':>11}{'vs per_tensor':>15}"
          f"{'recon err':>12}{'bits/byte':>11}")
    verdict = None
    alts = ("per_row", "group_128", "group_64", "group_32")
    for label in ("per_tensor",) + alts:
        r = results[label]
        d = r["held_out_loss"] - pt
        b = r["bits_per_byte"]
        print(f"[screen] {label:<12}{r['held_out_loss']:>11.5f}{d:>+15.5f}"
              f"{r['recon_rel_error']:>12.5f}"
              f"{'' if b is None else f'{b:>11.4f}'}")

    if not void:
        best = min(alts, key=lambda k: results[k]["held_out_loss"])
        d_loss = results[best]["held_out_loss"] - pt
        d_bpb = ((results[best]["bits_per_byte"] - results["per_tensor"]["bits_per_byte"])
                 if results[best]["bits_per_byte"] is not None else None)
        recon_improved = (results[best]["recon_rel_error"]
                          < results["per_tensor"]["recon_rel_error"])
        if not recon_improved:
            verdict = ("STOP: finer scaling did not even reduce reconstruction "
                       "error -- the implementation is wrong, not the idea")
        elif d_loss < 0:
            verdict = (f"PROCEED: {best} improves held-out by {-d_loss:.5f} nats "
                       f"post-hoc; a QAT A/B is worth the hours")
        else:
            verdict = (
                f"NOT RETROFITTABLE: finer scaling cuts reconstruction error "
                f"({results['per_tensor']['recon_rel_error']:.4f} -> "
                f"{results[best]['recon_rel_error']:.4f}) yet every scheme is WORSE "
                f"on held-out (best {best}: {d_loss:+.5f} nats"
                + (f", bits/byte {d_bpb:+.4f}" if d_bpb is not None else "") + "). "
                f"The model has ADAPTED to per-tensor quantization -- its weights sit "
                f"where per-tensor rounding puts them, so a 'better' approximation of "
                f"the shadow weights moves the function away from what was trained. "
                f"This kills the retrofit; it does NOT decide QAT-with-grouping.")
        print(f"\n[screen] VERDICT: {verdict}")
        print("[screen] LIMIT: this is post-hoc regrouping on a per-tensor-TRAINED "
              "checkpoint. It can only rule out the free version. Whether TRAINING "
              "with group scales beats training with per-tensor is untouched by this "
              "screen and needs a from-scratch A/B.")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump({"issue": 145, "config": vars(args),
                   "native_heldout_loss": native_loss,
                   "known_answer_drift": drift, "void": void,
                   # No "damage recovered" field: the fp arm is not an upper
                   # bound for a QAT model (see the note above), so any such
                   # ratio would have a fabricated denominator.
                   "shadow_fp_heldout_loss": fp,
                   "per_tensor_heldout_loss": pt,
                   "schemes": results, "verdict": verdict}, fh, indent=2)
    print(f"[screen] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
