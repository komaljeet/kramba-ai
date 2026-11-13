"""#194: at 1B, does 4 layers per BCD block beat 2 on held-out loss? One arm per process.

Author: Rambarun Komaljeet

Runs the real ``train_edge_conversational.main`` with ``PROD_ARGV`` (24L d2048
hybrid, seq 4096, batch 1, grad_accum 8, BCD offload, balanced partition, Muon,
fp8) plus ``--deterministic --seed 0``. Arms differ ONLY in
``--bcd-layers-per-block K`` (K=2 is passed explicitly so both arms take the same
code path): K=2 -> 13 training blocks, K=4 -> 7 (embedding alone, tail merged into
the last group). Data order, init, LR schedule and optimizer are identical.

Held-out loss is scored exactly as in ``scripts/exp_fp8_ab_197.py`` (same split,
same first ``--eval-windows`` windows, same offload-aware forward, fp8 off), so the
#197 numbers are directly comparable: the step-1 eval (lr 0, weights at init) must
reproduce #197's 11.23556 and its target sha.

Logged per arm in ``--out``:
  layout_<tag>.json   block names / count / params (asserted against --expect-blocks)
  steps_<tag>.jsonl   per optimizer step: seconds (eval excluded), wall clock, train
                      loss, active block, lr, running peak reserved within the turn
  turns_<tag>.jsonl   per block turn: peak reserved/allocated (stats reset at every
                      turn boundary: the trainer's #163 guard only checks step 1)
  eval_<tag>.jsonl    held-out loss, per-window losses, target sha, eval peak
  fingerprint_<tag>.jsonl  (--fingerprint only) which parameters changed per turn

Resume: the trainer auto-resumes from ``<out>/ckpt_<tag>`` (#112: weights,
optimizer, BCD position, LR schedule, data position, RNG). Re-running the SAME
command continues; this harness reads the resumed step from the newest checkpoint
and appends to its logs.
"""
import os, sys, json, time, argparse, hashlib

MAIN = r"C:\Users\komal\Documents\Codes\kramba-ai"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, required=True, help="--bcd-layers-per-block")
    ap.add_argument("--expect-blocks", type=int, required=True,
                    help="training-block count the layout must have (13 for k=2, 7 for k=4)")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, required=True, help="exit after this many steps")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="trainer --max-steps (sets the cosine). Default = --steps")
    ap.add_argument("--eval-every", type=int, default=100)
    ap.add_argument("--eval-at", default="1", help="extra comma list of eval steps")
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--val-fraction", type=float, default=0.001)
    ap.add_argument("--warmup", type=int, default=None, help="default: production 500")
    ap.add_argument("--spb", type=int, default=100, help="--bcd-steps-per-block")
    ap.add_argument("--save-every", type=int, default=100000)
    ap.add_argument("--keep", type=int, default=2, help="--keep-checkpoints")
    ap.add_argument("--fingerprint", action="store_true",
                    help="falsifier: fingerprint every parameter (bf16) at every turn "
                         "boundary; fail if a frozen parameter changes, and report "
                         "parameters never changed over the run")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
    os.chdir(REPO)
    import torch
    from torch.optim.optimizer import (register_optimizer_step_post_hook,
                                       register_optimizer_step_pre_hook)
    import hvn.core.block_coordinate as bc
    import hvn.edge.chunked_loss as CL
    from profile_bcd_memory import PROD_ARGV
    import train_edge_conversational as T
    assert os.path.abspath(bc.__file__).startswith(REPO), bc.__file__

    ckpt_dir = os.path.join(a.out, "ckpt_" + a.tag)
    latest = T.find_latest_checkpoint(ckpt_dir)
    start = 0
    if latest:
        start = int(os.path.basename(latest).split("_step")[-1].replace(".pt", ""))
        print(f"[h194] RESUME: {latest} -> continuing after step {start}", flush=True)
    eval_at = {int(s) for s in a.eval_at.split(",") if s}
    eval_at |= set(range(a.eval_every, a.steps + 1, a.eval_every)) if a.eval_every else set()
    total_mem = None

    VAL = []
    orig_bdp = T.build_dataset_pipeline

    def bdp(*x, **k):
        dl, vl, st_ = orig_bdp(*x, **k)
        assert vl is not None, "no held-out split: --val-fraction produced 0 windows"
        VAL.append(vl)
        return dl, vl, st_
    T.build_dataset_pipeline = bdp

    SCHED, MODEL = [], []
    orig_init = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *x, **k):
        orig_init(self, model, optimizer, *x, **k)
        SCHED.append(self); MODEL.append(model)
        names = [n for n, _ in self.blocks]
        lay = {"k": a.k, "n_blocks": len(names), "blocks": names,
               "params": [sum(p.numel() for p in ps) for _, ps in self.blocks],
               "partition": self.partition, "steps_per_block": self.steps_per_block}
        json.dump(lay, open(os.path.join(a.out, f"layout_{a.tag}.json"), "w"), indent=1)
        print(f"[h194] LAYOUT k={a.k}: {len(names)} blocks: {names}", flush=True)
        assert len(names) == a.expect_blocks, (len(names), a.expect_blocks)
        assert names[0] == "token_embedding" and names[-1].endswith("+final_norm+decoder"), names
        for n in names[1:]:
            got = sum(m.startswith("blocks.") for m in n.split("+"))
            assert got == a.k, (n, a.k)
        covered = sorted(id(p) for _, ps in self.blocks for p in ps)
        assert covered == sorted(id(p) for p in model.parameters()), "partition gap/overlap"
    bc.BlockCoordinateScheduler.__init__ = init

    losses = []
    orig_cce = CL.chunked_cross_entropy

    def cce(*x, **k):
        out = orig_cce(*x, **k)
        if torch.is_grad_enabled():
            losses.append(out.detach().float().mean())
        return out
    CL.chunked_cross_entropy = cce

    windows = []

    @torch.no_grad()
    def heldout():
        # Identical to scripts/exp_fp8_ab_197.py heldout().
        model = MODEL[0]
        if not windows:
            for i, (x, y, _src) in enumerate(VAL[0]):
                if len(windows) >= a.eval_windows:
                    break
                for b in range(x.shape[0]):
                    windows.append((x[b:b + 1].clone(), y[b:b + 1].clone()))
            windows[:] = windows[:a.eval_windows]
            assert len(windows) == a.eval_windows, len(windows)
        pool, act = model._offload_pool, model._offload_active_names
        was = model.training
        model.eval()

        def seg(name, mods, fn):
            params = [p for m in mods for p in m.parameters()]
            if name in act or not pool._offloadable(params):
                return fn()
            pool.load(name, params)
            out = fn()
            pool.evict(name)
            return out
        per, dev = [], torch.device("cuda")
        h = hashlib.sha256()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in windows:
                h.update(y.numpy().tobytes())
                x, y = x.to(dev), y.to(dev)
                pos = torch.arange(x.shape[1], device=dev).unsqueeze(0)
                z = seg("token_embedding", [model.token_embedding],
                        lambda: model.token_embedding(x))
                for i, blk in enumerate(model.blocks):
                    z = seg(f"blocks.{i}", [blk], lambda: blk(z, pos))
                tl = seg("final_norm+decoder", [model.final_norm, model.decoder],
                         lambda: orig_cce(model.final_norm(z), y, model.decoder,
                                          chunk=512, use_checkpoint=False))
                per.append(float(tl.float().mean()))
        if was:
            model.train()
        return per, h.hexdigest()

    # ---- falsifier: per-parameter bf16 fingerprints at turn boundaries ----
    FP = {"prev": None, "turn_ids": None, "turn_name": None, "ever": set()}
    pname = {}

    @torch.no_grad()
    def fingerprints():
        # Exact integer checksums of the bf16 bit patterns: order- and
        # device-independent. (A float64 sum was not: the same parked-on-CPU vs
        # active-on-GPU tensor summed differently and showed fake "changes" at lr 0.)
        # bf16 because parked weights are bf16; an fp32 master would differ from
        # its own parked copy.
        out = {}
        for n, p in MODEL[0].named_parameters():
            s1 = s2 = 0
            off = 0
            for c in p.detach().reshape(-1).split(1 << 22):
                b = c.to(torch.bfloat16).view(torch.int16).to(torch.int64)
                w = torch.arange(off, off + b.numel(), device=b.device) % 65521 + 1
                s1 += int(b.sum()); s2 += int((b * w).sum())
                off += b.numel()
            out[n] = (s1, s2)
        return out

    @torch.no_grad()
    def fingerprints_full(names):
        # Same idea at the stored precision (fp32 while active). bf16 cannot see
        # 3 small AdamW steps on a weight near 1.0 (norm weights: bf16 spacing
        # 0.0078 at 1.0), so "did the active block update" is checked here.
        out, ps = {}, dict(MODEL[0].named_parameters())
        for n in names:
            t = ps[n].detach().reshape(-1)
            iv = {torch.float32: torch.int32, torch.bfloat16: torch.int16}[t.dtype]
            s1 = s2 = 0
            off = 0
            for c in t.split(1 << 22):
                b = c.view(iv).to(torch.int64)
                w = torch.arange(off, off + b.numel(), device=b.device) % 65521 + 1
                s1 += int(b.sum()); s2 += int((b * w).sum())
                off += b.numel()
            out[n] = (str(t.dtype), s1, s2)
        return out

    def pre(opt, args, kw):
        if not a.fingerprint:
            return
        if st["n"] % a.spb == 0:   # first step of a turn
            if FP["prev"] is None:
                for n, p in MODEL[0].named_parameters():
                    pname[id(p)] = n
                FP["prev"] = fingerprints()
            s = SCHED[0]
            FP["turn_name"] = s.active_block_name
            FP["turn_ids"] = {pname[id(p)] for p in s.blocks[s._active_idx][1]}
            FP["start_full"] = fingerprints_full(sorted(FP["turn_ids"]))
    register_optimizer_step_pre_hook(pre)

    def fingerprint_turn(n):
        cur = fingerprints()
        changed = {k for k in cur if cur[k] != FP["prev"][k]}
        illegal = sorted(changed - FP["turn_ids"])
        unchanged_active = sorted(FP["turn_ids"] - changed)
        FP["ever"] |= changed
        never = sorted(set(cur) - FP["ever"])
        end_full = fingerprints_full(sorted(FP["turn_ids"]))
        ch_full = {k for k in end_full if end_full[k] != FP["start_full"][k]}
        unch_full = sorted(FP["turn_ids"] - ch_full)
        FP["ever_full"] = FP.get("ever_full", set()) | ch_full
        dtypes = sorted({v[0] for v in end_full.values()})
        row = {"step": n, "block": FP["turn_name"], "n_active": len(FP["turn_ids"]),
               "n_changed": len(changed), "illegal_changed": illegal,
               "active_unchanged": unchanged_active, "never_changed_so_far": len(never),
               "never_changed_names": never if len(never) <= 40 else never[:40],
               "n_changed_full": len(ch_full), "active_unchanged_full": unch_full,
               "active_dtypes": dtypes,
               "never_changed_full_so_far": len(set(cur) - FP["ever_full"])}
        fpf.write(json.dumps(row) + "\n"); fpf.flush()
        print(f"[h194] FINGERPRINT turn {FP['turn_name']} @step {n}: changed "
              f"{len(changed)}/{len(FP['turn_ids'])} active (bf16), "
              f"{len(ch_full)}/{len(FP['turn_ids'])} at stored precision {dtypes}, "
              f"illegal {len(illegal)}, never-changed so far {len(never)} (bf16) / "
              f"{row['never_changed_full_so_far']} (stored)", flush=True)
        assert not illegal, f"frozen params changed during {FP['turn_name']}: {illegal[:10]}"
        FP["prev"] = cur

    mode = "a" if start else "w"
    rows = open(os.path.join(a.out, f"steps_{a.tag}.jsonl"), mode)
    evf = open(os.path.join(a.out, f"eval_{a.tag}.jsonl"), mode)
    tf = open(os.path.join(a.out, f"turns_{a.tag}.jsonl"), mode)
    fpf = open(os.path.join(a.out, f"fingerprint_{a.tag}.jsonl"), mode) if a.fingerprint else None
    st = {"n": start, "t": time.perf_counter(), "turn_t0": time.time()}

    def post(opt, args, kw):
        nonlocal total_mem
        torch.cuda.synchronize()
        now = time.perf_counter()
        if total_mem is None:
            total_mem = torch.cuda.get_device_properties(0).total_memory
        n = st["n"] + 1          # optimizer steps completed
        tl = float(torch.stack(losses).mean()) if losses else float("nan")
        losses.clear()
        res = torch.cuda.max_memory_reserved()
        rows.write(json.dumps({"step": n, "sec": now - st["t"], "wall": time.time(),
                               "train_loss": tl, "block": SCHED[0].active_block_name,
                               "lr": [g["lr"] for g in opt.param_groups],
                               "peak_res_turn": res}) + "\n")
        rows.flush()
        boundary = n % a.spb == 0
        if boundary or n >= a.steps:
            pct = 100.0 * res / total_mem
            tf.write(json.dumps({"step": n, "block": SCHED[0].active_block_name,
                                 "peak_res": res, "peak_alloc": torch.cuda.max_memory_allocated(),
                                 "peak_res_pct": pct, "over_85": pct > 85.0,
                                 "turn_wall_sec": time.time() - st["turn_t0"]}) + "\n")
            tf.flush()
            print(f"[h194] TURN {SCHED[0].active_block_name} ends @step {n}: peak reserved "
                  f"{res / 2**20:.0f} MiB ({pct:.1f}%)" + ("  WARNING >85% SPILL RISK" if pct > 85 else ""),
                  flush=True)
            if a.fingerprint:
                fingerprint_turn(n)
        if n in eval_at:
            # Smoke: with a 4-layer block active, eval reached 94% reserved
            # (spill zone, eval 68 s vs 51 s), mostly the step's still-live
            # gradients. The trainer calls optimizer.zero_grad() (set_to_none)
            # right after optimizer.step() returns, so dropping them here is
            # what happens next anyway. Then release the cache before and after
            # so eval neither spills nor leaves training a bloated pool.
            for p in MODEL[0].parameters():
                p.grad = None
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            per, hs = heldout()
            torch.cuda.synchronize()
            row = {"step": n, "heldout_nll": sum(per) / len(per), "windows": len(per),
                   "tokens": len(per) * windows[0][1].numel(), "target_sha": hs,
                   "per_window": per, "eval_sec": time.perf_counter() - t0,
                   "eval_peak_res": torch.cuda.max_memory_reserved(),
                   "block": SCHED[0].active_block_name, "wall": time.time()}
            torch.cuda.empty_cache()
            evf.write(json.dumps(row) + "\n"); evf.flush()
            print(f"[h194] EVAL step {n}: held-out {row['heldout_nll']:.5f} "
                  f"({row['eval_sec']:.0f}s)", flush=True)
        if boundary:
            torch.cuda.reset_peak_memory_stats()
            st["turn_t0"] = time.time()
        st["n"] = n
        if n >= a.steps:
            for f in (rows, evf, tf, fpf):
                if f:
                    f.close()
            sys.stdout.flush(); os._exit(0)
        st["t"] = time.perf_counter()
    register_optimizer_step_post_hook(post)

    extra = ["--warmup-steps", str(a.warmup)] if a.warmup is not None else []
    sys.argv = ["train_edge_conversational.py"] + list(PROD_ARGV) + [
        "--grad-accum", "8", "--bcd-steps-per-block", str(a.spb),
        "--bcd-layers-per-block", str(a.k),
        "--max-steps", str(a.max_steps or a.steps), "--val-fraction", str(a.val_fraction),
        "--val-every", "100000000",
        "--save-every-steps", str(a.save_every), "--keep-checkpoints", str(a.keep),
        "--checkpoint-dir", ckpt_dir,
        "--data-dir", os.path.join(MAIN, "data_cache"), "--seed", "0",
        "--deterministic"] + extra
    T.main()


if __name__ == "__main__":
    main()
