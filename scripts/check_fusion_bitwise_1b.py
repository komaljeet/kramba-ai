"""Bit-identity of the 2026-10-17 fusions ON THE REAL 1B MODEL, over a BCD cycle.

Author: Rambarun Komaljeet

Why not just compare the A/B runs' losses: the real step is not run-to-run
deterministic (the token-embedding backward scatters with atomics), so two runs
of the SAME code already drift after the first embedding step -- an end-to-end
loss diff cannot tell a bug from that noise. This compares in ONE process from
the SAME state instead: for every BCD step, run the step's micro-batches with
the levers OFF, keep loss + every trainable grad, then again with them ON, and
require torch.equal. The optimizer then steps (on the ON grads -- equal, if the
check passed), so state carries forward and later steps are checked from a
trained, not freshly initialised, model.

The token_embedding step is compared on loss only (forward is deterministic;
its grad is the atomic scatter). Every other block's grads are compared.

Falsifiability: a third arm turns on a lever that is known to be only
ulp-close (the fused q/k L2 norm OFF -> eager, which differs by ~1 ulp) and
must FAIL the same equality on the first compared step -- proving that this
check, at this scale and dtype, can see a single-ulp change. STATS counters
must show the fast paths ran in the ON arm and not in the OFF arm.

Usage::

    python -u scripts/check_fusion_bitwise_1b.py --out results/fusion_sweep_1b/bitwise.json
"""
import argparse
import json
import os
import sys

ap = argparse.ArgumentParser()
ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap.add_argument("--out", required=True)
ap.add_argument("--ga", type=int, default=2)
ap.add_argument("--steps", type=int, default=13)
args = ap.parse_args()

sys.path.insert(0, args.repo)
import pyarrow  # noqa: F401,E402
import torch  # noqa: E402

import train_edge_conversational as tec  # noqa: E402
import hvn.optimizers as hopt  # noqa: E402
import hvn.edge.chunked_loss as hcl  # noqa: E402
import hvn.core.fused_gates as fg  # noqa: E402
import hvn.core.fused_read_norm as frn  # noqa: E402
from hvn.core.block_coordinate import BlockCoordinateScheduler  # noqa: E402
from hvn.core.deltanet import DeltaNetBlock  # noqa: E402
from hvn.edge.ternary import TernaryLinear  # noqa: E402

GA, SEQ, VOCAB, CHUNK = args.ga, 4096, 50257, 512
dev = torch.device("cuda")
torch.manual_seed(1234)
torch.cuda.manual_seed_all(1234)
settings = {"batch_size": 1, "grad_accum": GA, "seq_len": SEQ, "d_model": 2048,
            "num_layers": 24, "n_heads": 64}
model, _ = tec.build_model_and_config(
    settings, None, dev, checkpoint_dir=os.path.splitext(args.out)[0] + "_ckpt",
    arch="hvn-hybrid", apply_ternary=True, attn_layers=[6, 18], attn_window=4096,
    grad_checkpoint=True, tie_embeddings=True, chunked_loss=CHUNK, optimizer="muon",
    precision="bf16", lr=3e-4, vocab_size=VOCAB, offload_weights=True, dropout=0.0)
model = tec.apply_fast_stack(model, triton_scan=True, megakernel_scan=True, tf32=True,
                             compile_model=False, verbose=False, fused_ternary=True,
                             ternary_step_cache=True, fused_rmsnorm=True)
model.train()
opt, _ = hopt.build_optimizer(model, "muon", lr=3e-4, weight_decay=0.01,
                              fused=False, muon_lr=0.02)
sched = BlockCoordinateScheduler(
    model, opt, steps_per_block=1, offload_inactive_weights=True, device=dev,
    offload_park_dtype=torch.bfloat16, offload_prefetch=True,
    offload_cache_blocks=1, partition="balanced")
g = torch.Generator().manual_seed(0)
XS = torch.randint(0, VOCAB, (GA, 1, SEQ), generator=g)
YS = torch.randint(0, VOCAB, (GA, 1, SEQ), generator=g)
DN = [m for m in model.modules() if isinstance(m, DeltaNetBlock)]
TL = [m for m in model.modules() if isinstance(m, TernaryLinear)]


def set_arm(on, l2_eager=False):
    for m in DN:
        m.use_fused_read_gate = on
        m.use_fused_sigmoid = on
        m.use_fused_residual = on
        m.use_fused_l2norm = False if l2_eager else None
    for m in TL:
        m.use_autocast_dtype = on


def fwd_bwd():
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    for _, p in named:
        p.grad = None
    loss_acc = torch.zeros((), device=dev)
    for mb in range(GA):
        x = XS[mb].to(dev, non_blocking=True)
        y = YS[mb].to(dev, non_blocking=True)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            hidden = model(x, return_hidden=True)
            loss = hcl.chunked_cross_entropy(hidden, y, model.decoder, chunk=CHUNK).mean() / GA
        loss.backward()
        loss_acc += loss.detach()
    return loss_acc.item()


def grads_cpu():
    return {n: p.grad.detach().cpu().clone() for n, p in model.named_parameters()
            if p.requires_grad and p.grad is not None}


def mismatches(ref):
    """Names whose CURRENT grad differs from ``ref`` (one param on host at a time:
    host RAM stays at one copy of the active block's grads, ~0.5 GB)."""
    cur = {n: p for n, p in model.named_parameters() if p.requires_grad and p.grad is not None}
    if cur.keys() != ref.keys():
        return ["<grad key sets differ>"]
    return [n for n, p in cur.items() if not torch.equal(ref[n], p.grad.detach().cpu())]


def stats():
    return (frn.STATS["read_gate"], fg.STATS["sigmoid"], fg.STATS["add3"])


out = {"config": vars(args), "steps": [], "falsifier": None}
ok = True
for s in range(args.steps):
    a = sched.active_block_name
    s0 = stats()
    set_arm(False)
    l_off = fwd_bwd()
    s1 = stats()
    emb = a == "token_embedding"
    g_off = None if emb else grads_cpu()
    set_arm(True)
    l_on = fwd_bwd()
    s2 = stats()
    fired_off = s1 != s0
    fired_on = all(b > a_ for a_, b in zip(s1, s2))
    eq_loss = float(l_off).hex() == float(l_on).hex()
    bad = [] if emb else mismatches(g_off)
    r = {"active": a, "loss_off": float(l_off).hex(), "loss_on": float(l_on).hex(),
         "loss_equal": eq_loss, "n_grads": 0 if emb else len(g_off), "grad_mismatch": bad,
         "fast_path_fired_off_arm": fired_off, "fast_path_fired_on_arm": fired_on}
    step_ok = eq_loss and not bad and not fired_off and fired_on
    ok &= step_ok
    print(f"[{s:2d}] {a:<38} loss {l_off:.6f}/{l_on:.6f} equal={eq_loss} "
          f"grads {'(loss only)' if emb else f'{len(g_off) - len(bad)}/{len(g_off)} equal'} "
          f"fired off={fired_off} on={fired_on} -> {'OK' if step_ok else 'FAIL'}", flush=True)
    if out["falsifier"] is None and not emb:
        # Same state, ON arm again but with the ulp-close eager L2 norm: must differ.
        g_on = grads_cpu()
        del g_off
        set_arm(True, l2_eager=True)
        l_f = fwd_bwd()
        diff = mismatches(g_on)
        out["falsifier"] = {"active": a, "loss_differs": float(l_f).hex() != float(l_on).hex(),
                            "grads_differing": len(diff), "of": len(g_on)}
        print(f"[falsifier] eager L2 norm arm: loss differs={out['falsifier']['loss_differs']}, "
              f"{len(diff)}/{len(g_on)} grads differ (must be > 0)", flush=True)
        ok &= len(diff) > 0
        set_arm(True)
        l_on2 = fwd_bwd()                       # restore the ON grads for the step
        ok &= float(l_on2).hex() == float(l_on).hex() and not mismatches(g_on)
        g_off = None
        del g_on
    out["steps"].append(r)
    del g_off
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    opt.step()
    opt.zero_grad()
    sched.step()
out["all_ok"] = bool(ok)
with open(args.out, "w", encoding="utf-8") as fh:
    json.dump(out, fh, indent=1)
print(f"[result] {'PASS' if ok else 'FAIL'}", flush=True)
sys.exit(0 if ok else 1)
