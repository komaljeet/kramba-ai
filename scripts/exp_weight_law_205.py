"""#205: a law of motion for the active BCD block, and an early-switch rule built from it.

STAND-IN, NOT 1B: d512, 24 layers (attention at 6 and 18), 16 heads, the #198 harness
model/data/optimizer (scripts/ab_local_loss.py: ternary linears, Muon, warmup 100 +
cosine over --steps, grad clip 1.0, batch 4 x 512), BCD with the embedding table alone
and 4 layers per block (7 blocks, the last one carries final_norm+decoder).

Modes
  fixed  every turn is --spb steps (arm A: 100; arm C: B's mean turn length)
  rule   the pre-registered early-switch rule (arm B), see ``rule_says_switch``

Logging (all modes): per-step train loss, active-block update norm ||w_t - w_{t-1}||,
grad norm (pre-clip), lr; per-turn val loss (8 fixed batches, NOT the held-out set) every
10 turn-steps when --val-every > 0; held-out loss (16 fixed batches, the #198 probes) at
every global multiple of --heldout-every and at the end, plus every 10 turn-steps with
--record. With --record the active block's weights are snapshotted (fp32, CPU RAM) every 5
turn-steps and at turn end; at turn end they are reduced to block-level distances
d_end(t)=||w_t - w_end||, d_start(t)=||w_t - w_0|| and the cosine between consecutive
5-step deltas. Raw snapshots are not written to disk (2.1 GB per embedding turn).

Data seed and init seed are separate so a second init seed keeps the same batches and the
same held-out set. Both sets and the training batches are sha256-hashed into the output.
Known-answer: with init seed 7 the step-0 held-out loss must equal the #198 run's
(10.914026200771332), which pins the same probes and the same init.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time

import torch
import torch.nn as nn

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from scripts.ab_prodigy import as_pair, evaluate, require_cuda       # noqa: E402
from scripts.ab_local_loss import build                               # noqa: E402
import hvn.core.block_coordinate as bc                                # noqa: E402
from hvn.optimizers import build_optimizer                            # noqa: E402

MAX_STEPS_DATA = 2800     # batches are always taken from the same fixed window
VAL_BATCHES = 8
PROBE_BATCHES = 16
REF_STEP0 = 10.914026200771332


def sha(pairs) -> str:
    h = hashlib.sha256()
    for x, y in pairs:
        h.update(x.numpy().tobytes()); h.update(y.numpy().tobytes())
    return h.hexdigest()


def make_sched(model, opt, spb):
    """Embedding alone, then 4-layer groups; the tail final_norm+decoder merged in (#193)."""
    layer_sizes = [sum(p.numel() for p in b.parameters()) for b in model.blocks]
    budget = 4 * max(layer_sizes)
    orig = bc._balance_groups

    def keep_emb(blocks, b=None):
        groups = orig(blocks[1:], b)
        if len(groups) > 1 and groups[-1][0] == ["final_norm+decoder"]:
            groups[-2] = (groups[-2][0] + groups[-1][0],
                          list(groups[-2][1]) + list(groups[-1][1]))
            groups.pop()
        return [([blocks[0][0]], blocks[0][1])] + groups
    bc._balance_groups = keep_emb
    try:
        s = bc.BlockCoordinateScheduler(model, opt, steps_per_block=spb,
                                        partition="balanced", partition_budget=budget)
    finally:
        bc._balance_groups = orig
    return s


# ---------------------------------------------------------------------------------
# Pre-registered rule (issue #205 comment, committed before any arm-B run).
# Law: val(t) = c + A exp(-t/tau), fitted (grid over tau, least squares for c, A) to this
# turn's val points t = 0, 10, ..., t. Checked at t = 30, 40, ..., 90 (a 3-parameter fit
# needs at least 4 points). Switch when the fitted loss is not falling (A <= 0) or when the
# fitted improvement rate at t has dropped below `frac` of the fitted rate at t = 0, i.e.
# A/tau * exp(-t/tau) < frac * A/tau  <=>  t > tau * ln(1/frac). Never past 100 steps.
# ---------------------------------------------------------------------------------
RULE = {"min_steps": 30, "max_steps": 100, "frac": 0.25}


def rule_says_switch(t, val_ts, val_ls, rule) -> bool:
    import numpy as np
    from scripts.analyze_weight_law_205 import fit
    T, V = np.asarray(val_ts, float), np.asarray(val_ls, float)
    c, A, tau = fit("exp", T, V)[0]
    return bool(A <= 0 or math.exp(-t / tau) < rule["frac"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fixed", "rule"], default="fixed")
    ap.add_argument("--spb", type=int, default=100)
    ap.add_argument("--steps", type=int, default=2800)
    ap.add_argument("--init-seed", type=int, default=7)
    ap.add_argument("--data-seed", type=int, default=7)
    ap.add_argument("--record", action="store_true")
    ap.add_argument("--val-every", type=int, default=0, help="turn-steps between val evals")
    ap.add_argument("--heldout-every", type=int, default=100)
    ap.add_argument("--rule-json", default=None, help="rule params (pre-registered)")
    ap.add_argument("--data-dir", default=r"C:\Users\komal\Documents\Codes\kramba-ai\data_cache")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    assert a.steps <= MAX_STEPS_DATA

    # model/optimizer settings identical to the #198 L24 run
    h = argparse.Namespace(d_model=512, n_heads=16, num_layers=24, attn_layers=[6, 18],
                           seq_len=512, batch_size=4, muon_lr=0.02, aux_lr=3e-4,
                           weight_decay=0.01, warmup_steps=100)
    import train_edge_conversational as tec
    device = require_cuda()
    tok = tec.load_tokenizer("gpt2")
    h.vocab_size = len(tok)
    torch.manual_seed(a.data_seed)
    loader, _v, _s = tec.build_dataset_pipeline(
        tokenizer=tok, seq_len=h.seq_len, batch_size=h.batch_size,
        data_dir=a.data_dir, val_fraction=0.0, num_workers=0)
    pairs = []
    need = PROBE_BATCHES + MAX_STEPS_DATA + VAL_BATCHES
    for b in loader:
        x, y = as_pair(b)
        pairs.append((x.clone(), y.clone()))
        if len(pairs) >= need:
            break
    assert len(pairs) == need
    probes = pairs[:PROBE_BATCHES]
    batches = pairs[PROBE_BATCHES:PROBE_BATCHES + a.steps]
    val = pairs[PROBE_BATCHES + MAX_STEPS_DATA:]
    hashes = {"heldout": sha(probes), "val": sha(val),
              "train_window": sha(pairs[PROBE_BATCHES:PROBE_BATCHES + MAX_STEPS_DATA])}
    print(f"[205] sha heldout {hashes['heldout'][:16]} val {hashes['val'][:16]} "
          f"train {hashes['train_window'][:16]}", flush=True)

    rule = None
    if a.mode == "rule":
        rule = dict(RULE); rule.update(json.load(open(a.rule_json)))

    torch.manual_seed(a.init_seed)
    torch.cuda.manual_seed_all(a.init_seed)
    model = build(h, device)
    opt, desc = build_optimizer(model, "muon", lr=h.aux_lr, weight_decay=h.weight_decay,
                                muon_lr=h.muon_lr)
    spb_sched = a.spb if a.mode == "fixed" else 10 ** 9
    sched = make_sched(model, opt, spb_sched)
    assert sched.num_blocks == 7, [n for n, _ in sched.blocks]
    cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.steps)
    base_lrs = [pg["lr"] for pg in opt.param_groups]
    for n, ps in sched.blocks:
        print(f"[205] block {n:<60} {sum(p.numel() for p in ps)/1e6:7.2f}M", flush=True)

    os.makedirs(a.out, exist_ok=True)
    f_steps = open(os.path.join(a.out, "steps.jsonl"), "w")
    f_turns = open(os.path.join(a.out, "turns.jsonl"), "w")
    heldout_traj = []

    def ev(pairs_):
        v = evaluate(model, pairs_, device); model.train(); return v

    h0 = ev(probes)
    heldout_traj.append({"step": 0, "heldout": h0})
    print(f"[205] step 0 held-out {h0:.9f} (ref {REF_STEP0:.9f}, init seed {a.init_seed})",
          flush=True)
    known_answer = abs(h0 - REF_STEP0) < 1e-6 if a.init_seed == 7 else None
    if a.init_seed == 7 and not known_answer:
        raise SystemExit("step-0 held-out differs from #198: not the same probes/init")

    def active():
        return [p for p in sched.blocks[sched._active_idx][1]]

    def snap():
        return [p.detach().float().cpu().clone() for p in active()]

    def new_turn(gstep):
        tr = {"turn": len(turns_done), "block": sched.active_block_name,
              "block_idx": sched._active_idx, "rotation": len(turns_done) // 7,
              "start_step": gstep, "lr_start": opt.param_groups[0]["lr"],
              "train_loss": [], "upd_norm": [], "grad_norm": [], "lr": [],
              "val_t": [], "val": [], "heldout_t": [], "heldout": [], "snap_t": []}
        if a.val_every:
            tr["val_t"].append(0); tr["val"].append(ev(val))
        if a.record:
            tr["heldout_t"].append(0); tr["heldout"].append(heldout_traj[-1]["heldout"]
                                                             if heldout_traj[-1]["step"] == gstep
                                                             else ev(probes))
            snaps.clear(); snaps.append(snap()); tr["snap_t"].append(0)
        return tr

    def close_turn(tr, t):
        tr["length"] = t
        if a.record:
            wend = snaps[-1] if tr["snap_t"][-1] == t else snap()
            if tr["snap_t"][-1] != t:
                snaps.append(wend); tr["snap_t"].append(t)
            w0 = snaps[0]
            d_end, d_start = [], []
            for s in snaps:
                d_end.append(math.sqrt(sum(float((u - e).pow(2).sum()) for u, e in zip(s, wend))))
                d_start.append(math.sqrt(sum(float((u - z).pow(2).sum()) for u, z in zip(s, w0))))
            cosv = []
            for k in range(1, len(snaps) - 1):
                num = sum(float(((b - a_) * (c - b)).sum())
                          for a_, b, c in zip(snaps[k - 1], snaps[k], snaps[k + 1]))
                n1 = math.sqrt(sum(float((b - a_).pow(2).sum())
                                   for a_, b in zip(snaps[k - 1], snaps[k])))
                n2 = math.sqrt(sum(float((c - b).pow(2).sum())
                                   for b, c in zip(snaps[k], snaps[k + 1])))
                cosv.append(num / max(n1 * n2, 1e-30))
            tr["d_end"], tr["d_start"], tr["cos5"] = d_end, d_start, cosv
            tr["w_norm_end"] = math.sqrt(sum(float(e.pow(2).sum()) for e in wend))
            snaps.clear()
        f_turns.write(json.dumps(tr) + "\n"); f_turns.flush()
        vs = f" val {tr['val'][0]:.4f}->{tr['val'][-1]:.4f}" if tr["val"] else ""
        print(f"[205] turn {tr['turn']:>3} {tr['block'][:28]:<28} len {t:>3} "
              f"steps {tr['start_step']}-{tr['start_step']+t}{vs}", flush=True)
        turns_done.append({k: tr[k] for k in ("turn", "block", "length", "start_step")})

    turns_done, snaps = [], []
    model.train()
    warmup_done = False
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tr = new_turn(0)
    t = 0
    for i, (bx, by) in enumerate(batches, start=1):
        x = bx.to(device, non_blocking=True); y = by.to(device, non_blocking=True)
        prev = [p.detach().clone() for p in active()]
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = nn.functional.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1))
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],
                                            1.0)
        if not warmup_done:
            s = (i - 1) / max(1, h.warmup_steps)
            for pg, b in zip(opt.param_groups, base_lrs):
                pg["lr"] = b * s
            if i >= h.warmup_steps:
                warmup_done = True
        lr_used = opt.param_groups[0]["lr"]
        opt.step()
        opt.zero_grad(set_to_none=True)
        if warmup_done:
            cos.step()
        with torch.no_grad():
            un = float(torch.stack([(p.detach().float() - q.float()).pow(2).sum()
                                    for p, q in zip(active(), prev)]).sum().sqrt())
        del prev
        t += 1
        tl = float(loss)
        tr["train_loss"].append(tl); tr["upd_norm"].append(un)
        tr["grad_norm"].append(float(gn)); tr["lr"].append(lr_used)
        f_steps.write(json.dumps({"step": i, "turn": tr["turn"], "t": t, "block": tr["block"],
                                  "train_loss": tl, "upd_norm": un, "grad_norm": float(gn),
                                  "lr": lr_used}) + "\n")
        if a.record and t % 5 == 0:
            snaps.append(snap()); tr["snap_t"].append(t)
        if a.val_every and t % a.val_every == 0:
            tr["val_t"].append(t); tr["val"].append(ev(val))
        if a.record and t % 10 == 0:
            hv = ev(probes)
            tr["heldout_t"].append(t); tr["heldout"].append(hv)
            if i % a.heldout_every == 0 or i == a.steps:
                heldout_traj.append({"step": i, "heldout": hv})
        elif i % a.heldout_every == 0 or i == a.steps:
            heldout_traj.append({"step": i, "heldout": ev(probes)})
        if heldout_traj[-1]["step"] == i:
            print(f"[205] step {i:>5} held-out {heldout_traj[-1]['heldout']:.5f} "
                  f"{(time.time()-t0)/60:.1f} min", flush=True)

        if a.mode == "fixed":
            end = t >= a.spb
        else:
            end = t >= rule["max_steps"] or (
                t >= rule["min_steps"] and tr["val_t"] and tr["val_t"][-1] == t
                and rule_says_switch(t, tr["val_t"], tr["val"], rule))
        if i == a.steps:
            close_turn(tr, t)
            break
        if end:
            close_turn(tr, t)
            sched._step_in_block = sched.steps_per_block - 1
            assert sched.step()
            t = 0
            tr = new_turn(i)
        else:
            sched._step_in_block = 0      # the scheduler never switches by itself
            sched.step()

    f_steps.close(); f_turns.close()
    lens = [d["length"] for d in turns_done]
    full_lens = lens[:-1] if lens else lens        # last turn is cut by the step budget
    out = {"args": vars(a), "hashes": hashes, "known_answer_step0": known_answer,
           "step0_heldout": h0, "heldout_traj": heldout_traj,
           "final_heldout": heldout_traj[-1]["heldout"], "turn_lengths": lens,
           "mean_turn_len_completed": (sum(full_lens) / len(full_lens)) if full_lens else None,
           "n_turns": len(lens), "rule": rule, "wall_min": (time.time() - t0) / 60,
           "peak_alloc_gib": torch.cuda.max_memory_allocated() / 2 ** 30}
    json.dump(out, open(os.path.join(a.out, "summary.json"), "w"), indent=1)
    print(f"[205] DONE final held-out {out['final_heldout']:.5f} turns {len(lens)} "
          f"mean len {out['mean_turn_len_completed']} wall {out['wall_min']:.1f} min",
          flush=True)


if __name__ == "__main__":
    main()
