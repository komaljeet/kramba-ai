"""Do frozen blocks BEFORE the active BCD block run backward? (#199)

Runs the REAL training entrypoint (train_edge_conversational.main) with the
production 1B flags (scripts/profile_bcd_memory.PROD_ARGV) and counts, per
micro-batch, how often each segment's forward runs and in which phase:

  fwd  called during the forward pass
  bwd  called INSIDE backward (autograd graph task running): a checkpoint
       recompute (layer blocks) or the chunked loss's decoder recompute

A block that gets a backward recompute shows bwd=1; a forward-only block
shows bwd=0. It also records whether the block's input/output required grad
(an autograd graph was built through it), and CUDA-event timings: each
segment's forward time, the prefix forward span (embedding start -> first
active member start), and each block's backward span (grad of its output
arrives -> grad of its input arrives).

Positive control: blocks AFTER the active block must show bwd=1, and the
decoder must show bwd>0 (chunked recompute). If they do not, the phase
detector is broken and a prefix bwd=0 means nothing.
Falsifier arm (--force-prefix-grad): make the embedding output require grad,
which is exactly the bug being looked for. The counter must then show bwd=1
for the prefix blocks too, and the step gets slower by the cost of the bug.
Arm --prefix-no-grad: run frozen prefix blocks under torch.no_grad (the
candidate "fix"), to measure whether autograd bookkeeping in the prefix costs
anything.

One JSON row per optimizer step, written from the optimizer pre-step hook.

Usage: python scripts/profile_bcd_prefix.py --out results/bcd_prefix/base.jsonl
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))


def install(out_path, visit, force_prefix_grad=False, prefix_no_grad=False,
            grad_hash=False):
    import torch
    from torch.optim.optimizer import (register_optimizer_step_post_hook,
                                       register_optimizer_step_pre_hook)
    import hvn.core.block_coordinate as bc

    S = {"sched": None, "model": None, "mods": {}, "mb": [], "cur": None,
         "n": 0, "t": None, "acc_ev": None}
    f = open(out_path, "w", encoding="utf-8")

    def in_backward():
        return torch._C._current_graph_task_id() != -1

    def ev():
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        return e

    def first_tensor(args):
        for a in args:
            if torch.is_tensor(a):
                return a
        return None

    def make_hooks(name, mod):
        def pre(m, args):
            if m is not S["mods"].get(name):
                return  # e.g. the audit's deepcopy
            if name == "token_embedding" and not in_backward():
                S["cur"] = {"calls": {}, "ev": [], "req": {}}
                S["mb"].append(S["cur"])
                S["cur"]["ev"].append(("start", ev()))
            cur = S["cur"]
            if cur is None:
                return
            ph = "bwd" if in_backward() else "fwd"
            c = cur["calls"].setdefault(name, {"fwd": 0, "bwd": 0})
            c[ph] += 1
            x = first_tensor(args)
            if ph == "fwd" and x is not None:
                cur["req"].setdefault(name, {})["in_req"] = bool(
                    x.requires_grad and torch.is_floating_point(x))
                cur["req"][name]["grad_mode"] = torch.is_grad_enabled()
            cur["ev"].append((f"{ph}_pre:{name}", ev()))

        def post(m, args, out):
            if m is not S["mods"].get(name) or S["cur"] is None:
                return
            cur = S["cur"]
            ph = "bwd" if in_backward() else "fwd"
            cur["ev"].append((f"{ph}_post:{name}", ev()))
            o = out if torch.is_tensor(out) else first_tensor(out)
            if ph == "fwd" and o is not None:
                cur["req"].setdefault(name, {})["out_req"] = bool(o.requires_grad)
                if force_prefix_grad and name == "token_embedding" and not o.requires_grad:
                    out = o.detach().requires_grad_(True)
                    o = out
                if o.requires_grad and name != "decoder":
                    def gh(g, n=name, c=cur):
                        c["ev"].append((f"outgrad:{n}", ev()))
                        return None
                    o.register_hook(gh)
                return out if force_prefix_grad and name == "token_embedding" else None

        mod.register_forward_pre_hook(pre)
        mod.register_forward_hook(post)

    orig_init = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *a, **k):
        orig_init(self, model, optimizer, *a, **k)
        n = len(self.blocks)
        perm = list(visit) + [i for i in range(n) if i not in visit]
        assert perm[0] == self.order[0] == 0, "visit must start at block 0"
        self.order = perm
        S["sched"], S["model"] = self, model
        mods = {"token_embedding": model.token_embedding,
                "final_norm": model.final_norm, "decoder": model.decoder}
        for i, b in enumerate(model.blocks):
            mods[f"blocks.{i}"] = b
        S["mods"] = mods
        for name, m in mods.items():
            make_hooks(name, m)
        for p in model.parameters():
            def acc(p_, c=S):
                if c["cur"] is not None:
                    c["cur"]["ev"].append(("acc", ev()))
            was = p.requires_grad  # hooks can only be added to grad-requiring params
            p.requires_grad_(True)
            p.register_post_accumulate_grad_hook(acc)
            p.requires_grad_(was)
        if prefix_no_grad:
            for i, b in enumerate(model.blocks):
                orig_fwd = b.forward

                def fwd(x, *aa, _o=orig_fwd, _b=b, **kk):
                    if (torch.is_grad_enabled() and not in_backward()
                            and not x.requires_grad
                            and not any(p.requires_grad for p in _b.parameters())):
                        with torch.no_grad():
                            return _o(x, *aa, **kk)
                    return _o(x, *aa, **kk)
                b.forward = fwd
        print(f"[prefix-probe] {n} blocks, visit order {perm[:len(visit)]}; "
              f"force_prefix_grad={force_prefix_grad} prefix_no_grad={prefix_no_grad}",
              flush=True)

    bc.BlockCoordinateScheduler.__init__ = init

    def summarize(mb, members):
        t = {}
        start = mb["ev"][0][1]
        for lab, e in mb["ev"]:
            t.setdefault(lab, []).append(start.elapsed_time(e))
        out = {}
        # forward times per segment
        out["fwd_ms"] = {k.split(":", 1)[1]: round(v[0] - t[f"fwd_pre:{k.split(':', 1)[1]}"][0], 3)
                         for k, v in t.items() if k.startswith("fwd_post:")}
        first_active = [f"fwd_pre:{m}" for m in members if f"fwd_pre:{m}" in t]
        out["prefix_fwd_ms"] = round(min(t[k][0] for k in first_active), 3) if first_active else None
        out["fwd_end_ms"] = round(max(v[0] for k, v in t.items() if k == "fwd_post:final_norm"), 3)
        # backward spans: grad of block i's output arrives -> grad of its input
        # (= output of i-1 / embedding) arrives
        og = {k.split(":", 1)[1]: v[0] for k, v in t.items() if k.startswith("outgrad:")}
        nblk = len([k for k in S["mods"] if k.startswith("blocks.")])
        bwd = {}
        for i in range(nblk):
            n = f"blocks.{i}"
            if n not in og:
                continue
            prev = "token_embedding" if i == 0 else f"blocks.{i - 1}"
            if prev in og:
                bwd[n] = round(og[prev] - og[n], 3)
            elif "acc" in t:
                bwd[n] = round(max(t["acc"]) - og[n], 3)  # active block: until its last grad lands
        out["bwd_ms"] = bwd
        out["bwd_end_ms"] = round(max([max(t["acc"])] if "acc" in t else [0.0] + list(og.values())), 3)
        out["bwd_start_ms"] = round(og["final_norm"], 3) if "final_norm" in og else None
        return out

    def pre_step(opt, args, kwargs):
        sched = S["sched"]
        if sched is None:
            return
        torch.cuda.synchronize()
        now = time.perf_counter()
        step_sec = None if S["t"] is None else now - S["t"]
        S["t"] = now
        members = list(sched.active_block_members)
        mbs = S["mb"]
        calls = {}
        for mb in mbs:
            for k, v in mb["calls"].items():
                c = calls.setdefault(k, {"fwd": 0, "bwd": 0})
                c["fwd"] += v["fwd"]
                c["bwd"] += v["bwd"]
        summ = [summarize(mb, members) for mb in mbs]

        def med(key):
            vals = [s[key] for s in summ if s.get(key) is not None]
            return round(statistics.median(vals), 3) if vals else None

        def med_d(key):
            keys = sorted({k for s in summ for k in s[key]})
            return {k: round(statistics.median([s[key][k] for s in summ if k in s[key]]), 3)
                    for k in keys}
        row = {
            "opt_step": S["n"], "block": sched.active_block_name, "members": members,
            "step_in_block": sched._step_in_block, "step_sec": step_sec,
            "micro_batches": len(mbs), "calls": calls,
            "req": mbs[-1]["req"] if mbs else {},
            "prefix_fwd_ms": med("prefix_fwd_ms"), "fwd_end_ms": med("fwd_end_ms"),
            "bwd_start_ms": med("bwd_start_ms"), "bwd_end_ms": med("bwd_end_ms"),
            "fwd_ms": med_d("fwd_ms"), "bwd_ms": med_d("bwd_ms"),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "total_mem": torch.cuda.get_device_properties(0).total_memory,
        }
        if grad_hash:
            act = dict(sched.blocks)[sched.active_block_name]
            h = hashlib.sha256()
            for p in act:
                if p.grad is not None:
                    h.update(p.grad.detach().float().cpu().numpy().tobytes())
            row["grad_sha256"] = h.hexdigest()
        f.write(json.dumps(row) + "\n")
        f.flush()
        S["mb"] = []
        S["cur"] = None
        S["n"] += 1

    def post_step(opt, args, kwargs):
        torch.cuda.reset_peak_memory_stats()

    register_optimizer_step_pre_hook(pre_step)
    register_optimizer_step_post_hook(post_step)


def main():
    from profile_bcd_memory import PROD_ARGV
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--visit", default="0,1,6,12",
                    help="BCD block indices to visit, in order (must start at 0)")
    ap.add_argument("--steps-per-block", type=int, default=3)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--force-prefix-grad", action="store_true")
    ap.add_argument("--prefix-no-grad", action="store_true")
    ap.add_argument("--grad-hash", action="store_true")
    ap.add_argument("--deterministic", action="store_true")
    ap.add_argument("--lr-zero", action="store_true",
                    help="lr 0 so every step sees the same weights (for grad hashes)")
    ap.add_argument("--ckpt-dir", default="checkpoints/bcd_prefix_scratch")
    ap.add_argument("--data-dir", default="data_cache")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    visit = [int(v) for v in a.visit.split(",")]
    install(a.out, visit, a.force_prefix_grad, a.prefix_no_grad, a.grad_hash)
    os.chdir(REPO)
    import train_edge_conversational as T
    argv = list(PROD_ARGV)
    if a.lr_zero:
        argv[argv.index("--muon-lr") + 1] = "0"
        argv[argv.index("--lr") + 1] = "0"
    extra = ["--deterministic", "--seed", "0"] if a.deterministic else []
    sys.argv = (["train_edge_conversational.py"] + argv + extra
                + ["--grad-accum", str(a.grad_accum),
                   "--bcd-steps-per-block", str(a.steps_per_block),
                   "--max-steps", str(a.steps_per_block * len(visit)),
                   "--checkpoint-dir", a.ckpt_dir, "--data-dir", a.data_dir])
    T.main()


if __name__ == "__main__":
    main()
