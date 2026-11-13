"""What fills the GPU at 1B, per BCD block? (#192)

Runs the REAL training entrypoint (train_edge_conversational.main) with the
production 1B flags and a short BCD turn, and records at every optimizer step,
via a global pre-step hook, what is resident on the GPU:

  weights   parameter bytes on CUDA (the active block's fp32 shadow weights plus
            whatever offloaded layers happen to be loaded)
  grads     .grad bytes (right before the update, so the active block's full set)
  state     optimizer state bytes (Muon momentum, AdamW moments)
  other     allocated - the three above: activations, loss chunks, FP8/ternary
            caches, pinned tied table buffers, offload prefetch buffers
  peak      max allocated over the whole step (forward + backward + recompute)

Known-answer cell: for every block, grad bytes must equal active params x grad
element size. If that does not hold, the instrument is counting the wrong thing
and nothing here means anything.

One row per optimizer step is appended to --out as a JSON line from inside the
hook; code placed after main() returns is not reliable here.

Usage: python scripts/profile_bcd_memory.py --out results/bcd_memory/1b.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PROD_ARGV = (
    "--arch hvn-hybrid --d-model 2048 --num-layers 24 --n-heads 64 --attn-layers 6,18 "
    "--attn-window 4096 --batch-size 1 --seq-len 4096 --chunked-loss 512 --offload "
    "--bcd-partition balanced --optimizer muon --muon-lr 0.02 --lr 3e-4 --dropout 0 "
    "--tie-embeddings --pretrain-mix 0.0,0.0,0.0,1.0 --min-tokens-per-param 0 "
    "--fast --fp8-linear --log-every 1 --save-every-steps 100000"
).split()


def install_hooks(out_path: str, budget_layers: int = 0):
    import torch
    from torch.optim.optimizer import (register_optimizer_step_post_hook,
                                       register_optimizer_step_pre_hook)
    import hvn.core.block_coordinate as bc

    sched_ref = []
    orig_init = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *a, **k):
        if budget_layers:
            # Cap each balanced block at N x the largest layer instead of the
            # embedding table's size (the production default).
            sizes = [sum(p.numel() for p in b.parameters()) for b in model.blocks]
            k["partition_budget"] = budget_layers * max(sizes)
        orig_init(self, model, optimizer, *a, **k)
        sched_ref.append(self)

    bc.BlockCoordinateScheduler.__init__ = init
    if budget_layers:
        # Keep the embedding table in its own block, as at budget 1 layer. Otherwise a
        # budget above embedding+1 layer packs the heavy AdamW-state embedding with
        # layers, and the first block (not the block size being tested) hits the guard.
        orig_bg = bc._balance_groups

        def keep_emb(blocks, budget=None):
            return [([blocks[0][0]], blocks[0][1])] + orig_bg(blocks[1:], budget)

        bc._balance_groups = keep_emb
    state = {"n": 0, "t": None}
    f = open(out_path, "w", encoding="utf-8")

    def pre(opt, args, kwargs):
        if not sched_ref or not torch.cuda.is_available():
            return
        sched = sched_ref[0]
        name = sched.active_block_name
        params = [p for g in opt.param_groups for p in g["params"]]
        w = sum(p.numel() * p.element_size() for p in params if p.is_cuda)
        g = sum(p.grad.numel() * p.grad.element_size() for p in params
                if p.grad is not None and p.grad.is_cuda)
        active = {id(p) for p in dict(sched.blocks)[name]}
        active_n = sum(p.numel() for p in params if id(p) in active)
        active_grad_elem = {p.grad.element_size() for p in params
                            if id(p) in active and p.grad is not None}
        st = 0
        for s in opt.state.values():
            for v in s.values():
                if torch.is_tensor(v) and v.is_cuda:
                    st += v.numel() * v.element_size()
        alloc = torch.cuda.memory_allocated()
        ms = torch.cuda.memory_stats()
        now = time.perf_counter()
        step_sec = None if state["t"] is None else now - state["t"]
        state["t"] = now
        row = {
            "step_sec": step_sec,
            "reserved_now": ms["reserved_bytes.all.current"],
            "inactive_split": ms["inactive_split_bytes.all.current"],
            "segments": ms["segment.all.current"],
            "alloc_retries": ms["num_alloc_retries"],
            "opt_step": state["n"], "block": name, "step_in_block": sched._step_in_block,
            "weights": w, "grads": g, "state": st, "allocated": alloc,
            "other": alloc - w - g - st,
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "active_params": active_n, "grad_elem_sizes": sorted(active_grad_elem),
        }
        f.write(json.dumps(row) + "\n")
        f.flush()
        state["n"] += 1

    def post(opt, args, kwargs):
        torch.cuda.reset_peak_memory_stats()

    register_optimizer_step_pre_hook(pre)
    register_optimizer_step_post_hook(post)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/bcd_memory/1b.jsonl")
    ap.add_argument("--steps-per-block", type=int, default=3)
    ap.add_argument("--blocks", type=int, default=13)
    ap.add_argument("--ckpt-dir", default="checkpoints/bcd_memory_scratch")
    ap.add_argument("--data-dir", default="data_cache")
    ap.add_argument("--grad-accum", type=int, default=1,
                    help="micro-batches per optimizer step (production uses 8)")
    ap.add_argument("--budget-layers", type=int, default=0,
                    help="0 = production default; N = blocks of up to N layers")
    ap.add_argument("--layers-per-block", type=int, default=0,
                    help="pass --bcd-layers-per-block N to the real training script (#194)")
    ap.add_argument("--shape", default="",
                    help="d_model,layers,heads,attn_layers e.g. 2784,12,87,3,9 (default: 1B production 2048x24)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    install_hooks(a.out, a.budget_layers)
    import train_edge_conversational as T
    prod = list(PROD_ARGV)
    if a.shape:
        d, nl, nh, *attn = a.shape.split(",")
        for flag, val in (("--d-model", d), ("--num-layers", nl), ("--n-heads", nh),
                          ("--attn-layers", ",".join(attn))):
            prod[prod.index(flag) + 1] = val
    steps = a.steps_per_block * a.blocks + 2   # --blocks must equal the layout's block count
    sys.argv = (["train_edge_conversational.py"] + prod
                + ["--grad-accum", str(a.grad_accum), "--bcd-steps-per-block", str(a.steps_per_block),
                   "--max-steps", str(steps), "--checkpoint-dir", a.ckpt_dir,
                   "--data-dir", a.data_dir]
                + (["--bcd-layers-per-block", str(a.layers_per_block)] if a.layers_per_block else []))
    T.main()


if __name__ == "__main__":
    main()
