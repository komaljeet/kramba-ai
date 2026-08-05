"""A/B the ternary re-quantization overhead found by the #131 decode profile.

Author: Rambarun Komaljeet

The profile's smoking gun: ``aten::mean`` fires 6,656 times over 64 decoded
tokens, against exactly 102 projection modules x 64 tokens = 6,528 projection
calls. That is one absmean reduction per projection per token -- i.e.
``TernaryLinear.forward`` (hvn/edge/ternary.py:82) is re-running
``ternary_quantize_ste`` over the ENTIRE weight matrix on every single decoded
token, at inference, with the weights frozen.

That work is O(d^2) per projection per token, the same order as the matvec it
feeds, and it is pure waste during decode: the weights do not change.

This measures the win directly rather than arguing it from a profiler
attribution, because the profile's own share number is not trustworthy --
key_averages() reports both op-level (``aten::mul``) and kernel-level
(``_ZN2at6native...``) entries carrying the SAME device time (``aten::abs`` and
its AbsFunctor kernel agree to the last decimal at 54794.628us), so summing
across all events double-counts. A wall-clock A/B has no such ambiguity.

KNOWN-ANSWER CELL (the thing docs/state.md requires after the #122 retraction):
caching a deterministic function of frozen weights is mathematically a no-op, so
the cached arm MUST produce bit-identical logits. If it does not, the cache is
wrong and the speed number is meaningless.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402
from hvn.edge.ternary import TernaryLinear, ternary_quantize_ste  # noqa: E402


def install_legacy(model):
    """Restore the PRE-#131 forward: re-quantize the whole weight every call.

    The cache now lives in ``TernaryLinear`` itself, so the arm that needs
    simulating is the old behaviour, not the new one. This reproduces exactly
    what ``hvn/edge/ternary.py`` did before this issue -- ``ternary_quantize_ste``
    over the full weight matrix on every forward, i.e. once per decoded token.

    Returns ``(n_patched, restore)`` so the arms can run in either order; the
    baseline running first every time would let GPU clock ramp inflate the
    second arm, which is the kind of order effect that turns a real measurement
    into an artifact.
    """
    n = 0
    originals = []
    for mod in model.modules():
        if not isinstance(mod, TernaryLinear):
            continue
        originals.append((mod, mod.forward))

        def make(m):
            def fwd(x):
                if m._frozen_codes is not None:
                    w = m._frozen_codes.to(x.dtype) * m._frozen_scale
                else:
                    w = ternary_quantize_ste(m.weight).to(x.dtype)
                return F.linear(x, w, m.bias)
            return fwd

        mod.forward = make(mod)
        n += 1

    def restore() -> None:
        for m, fn in originals:
            m.forward = fn
            m.invalidate_cache()

    return n, restore


@torch.no_grad()
def decode(model, n_tokens: int, device, vocab: int, capture: bool = False):
    """Greedy decode from a fixed start token -- deterministic, so two arms are
    directly comparable token-for-token."""
    states = model.init_state(1, device=device, dtype=next(model.parameters()).dtype)
    tok = torch.full((1, 1), 42, device=device, dtype=torch.long)
    logits_out = []
    for i in range(n_tokens):
        logits, states = model.forward_step(tok, states, position_idx=i)
        if capture:
            logits_out.append(logits[:, -1, :].detach().clone())
        tok = logits[:, -1, :].argmax(dim=-1, keepdim=True)
    return torch.cat(logits_out, 0) if capture else None


@torch.no_grad()
def timed(model, n_tokens: int, device, vocab: int, repeats: int) -> Dict[str, float]:
    samples: List[float] = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        decode(model, n_tokens, device, vocab)
        torch.cuda.synchronize()
        samples.append(n_tokens / (time.perf_counter() - t0))
    samples.sort()
    mean = sum(samples) / len(samples)
    var = sum((s - mean) ** 2 for s in samples) / max(1, len(samples) - 1)
    return {"mean": mean, "std": var ** 0.5,
            "min": samples[0], "max": samples[-1], "samples": samples}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint",
                    default="checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt")
    ap.add_argument("--tokens", type=int, default=64)
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--order", choices=["base-first", "cache-first"],
                    default="base-first",
                    help="arm order; run both to rule out GPU clock-ramp order effects")
    ap.add_argument("--out", default="results/matmul_free/ternary_cache_ab.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("GPU required (docs/ai_rules.md)")
    device = torch.device("cuda")

    model, step, _ = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    vocab = getattr(model.decoder, "out_features", 50257)
    n_tern = sum(1 for m in model.modules() if isinstance(m, TernaryLinear))
    print(f"[#131] {args.checkpoint} step={step} TernaryLinear modules={n_tern}")
    if n_tern == 0:
        raise SystemExit("no TernaryLinear modules -- this checkpoint is not ternary; "
                         "the re-quantization hypothesis does not apply")

    def run_cached():
        """The landed #131 behaviour -- TernaryLinear's own cache."""
        decode(model, args.warmup, device, vocab)
        t = timed(model, args.tokens, device, vocab, args.repeats)
        lg = decode(model, args.tokens, device, vocab, capture=True)
        print(f"[#131] cached (landed) {t['mean']:.2f} +/- {t['std']:.2f} tok/s")
        return t, lg

    def run_legacy():
        n, restore = install_legacy(model)
        decode(model, args.warmup, device, vocab)
        t = timed(model, args.tokens, device, vocab, args.repeats)
        lg = decode(model, args.tokens, device, vocab, capture=True)
        print(f"[#131] legacy (pre)    {t['mean']:.2f} +/- {t['std']:.2f} tok/s "
              f"({n} modules patched)")
        return t, lg, n, restore

    if args.order == "cache-first":
        cached, cached_logits = run_cached()
        base, base_logits, n_patched, restore = run_legacy()
        restore()
    else:
        base, base_logits, n_patched, restore = run_legacy()
        restore()
        cached, cached_logits = run_cached()

    max_abs = float((base_logits - cached_logits).abs().max())
    bit_identical = bool(torch.equal(base_logits, cached_logits))
    speedup = cached["mean"] / base["mean"]

    print(f"[#131] KNOWN-ANSWER CELL: max|dlogits| = {max_abs:.3e}, "
          f"bit-identical = {bit_identical}")
    print(f"[#131] speedup = {speedup:.2f}x")
    if not bit_identical:
        print("[#131] *** cache is NOT a no-op -- speed number is INVALID ***")

    out = {
        "issue": 131,
        "checkpoint": args.checkpoint,
        "ternary_modules": n_tern,
        "modules_patched": n_patched,
        "tokens": args.tokens,
        "repeats": args.repeats,
        "arm_order": args.order,
        "baseline_tok_s": base,
        "cached_tok_s": cached,
        "speedup": speedup,
        "known_answer_cell": {
            "claim": "caching a deterministic function of frozen weights is a no-op",
            "max_abs_logit_diff": max_abs,
            "bit_identical": bit_identical,
            "valid": bit_identical,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"[#131] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
