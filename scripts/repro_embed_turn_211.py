"""#211: reproduce the slow embedding turns of the #209 long A/B.

Copy of scripts/ab_heldout_209.py (exp/scan-t-long-ab-209, the harness that produced the
slow turns) with per-step memory fields added -- res_now / alloc_now (caching allocator,
current, not peak) and drv_free (cudaMemGetInfo, the driver's view) -- and
``--no-empty-cache`` to skip the empty_cache() after each eval.

#209 docstring follows.

#209: LONG 1B k4 held-out loss A/B (--scan-kernel t vs default). One arm per process.

Author: Rambarun Komaljeet

Copy of scripts/ab_heldout_208.py with three changes for multi-hour runs:
  resume     the trainer's own checkpoints (--save-every, keep 2) go to
             $AB209_CKPT/ckpt_<tag>; re-running the same command auto-resumes from
             the newest one. The step/eval logs are truncated back to the resumed
             step and appended to, and a RESUMED line is printed and logged.
  --stop-at  exit after this optimizer step while the LR schedule still runs to
             --steps (the falsifier runs 300 steps of the 1800-step schedule).
  wall       every step row carries the wall-clock time, so sleeps/stalls show.

#208 docstring follows.

Built on scripts/exp_fp8_ab_197.py (same held-out eval walk through the offload
pool, fp8 off in eval). Runs the real ``train_edge_conversational.main`` with
``PROD_ARGV`` + the post-#207 k4 production settings (4 layers per BCD block,
grad_accum 8, --deterministic) and short turns (``--spb``) so the run passes through
all 7 BCD blocks more than once.

What is held fixed across arms and seeds:
  data order   the train sampler's shuffle seed is pinned (``--shuffle-seed``,
               default 0) -- every arm and seed sees the same batches
  held-out     the first ``--eval-windows`` windows of the trainer's token-disjoint
               FineWeb-Edu val split (deterministic; target sha logged per eval)
What varies:
  ``--seed``   model init (and nothing else: the sampler seed is pinned)
  ``--extra``  the trainer flags under test, e.g. "--scan-kernel t"

Falsifier arm (``--break-scan``): the DeltaNet scan's backward returns ZERO input
gradients (a kernel that never writes its gradient outputs). The scan still runs
forward correctly, so the run looks healthy (loss falls through the MLPs, attention,
embeddings). The A/B must flag this arm; if it cannot, it cannot see a scan bug.
A per-step counter proves the broken backward actually ran.
"""
import os, sys, json, time, argparse, hashlib

MAIN = r"C:\Users\komal\Documents\Codes\kramba-ai"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--spb", type=int, required=True, help="--bcd-steps-per-block")
    ap.add_argument("--eval-at", default="", help="comma list of optimizer-step counts")
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--val-fraction", type=float, default=0.001)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="model init seed")
    ap.add_argument("--shuffle-seed", type=int, default=0, help="data order (pinned)")
    ap.add_argument("--extra", default="", help="trainer flags under test")
    ap.add_argument("--break-scan", action="store_true",
                    help="falsifier: scan backward returns zero input gradients")
    ap.add_argument("--stop-at", type=int, default=None,
                    help="exit after this step (schedule still spans --steps)")
    ap.add_argument("--save-every", type=int, default=200)
    ap.add_argument("--no-empty-cache", action="store_true",
                    help="#211: do not empty the CUDA cache after each eval")
    a = ap.parse_args()
    stop_at = a.stop_at or a.steps
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
    os.chdir(REPO)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    import hvn.core.block_coordinate as bc
    import hvn.edge.chunked_loss as CL
    import hvn.core.deltanet_triton as DT
    assert os.path.abspath(DT.__file__).startswith(REPO), DT.__file__
    from profile_bcd_memory import PROD_ARGV
    eval_at = {int(s) for s in a.eval_at.split(",") if s}

    BROKE = {"n": 0}
    if a.break_scan:
        for cls in (DT._MegaScanBSHD, DT._MegaScanBSHD_T):
            orig_bwd = cls.backward

            def bwd(ctx, do, orig_bwd=orig_bwd):
                grads = orig_bwd(ctx, do)
                BROKE["n"] += 1
                return tuple(None if g is None else torch.zeros_like(g) for g in grads)
            cls.backward = staticmethod(bwd)
        print("[ab208] FALSIFIER: scan backward returns zero gradients", flush=True)

    import train_edge_conversational as T
    VAL = []
    orig_bdp = T.build_dataset_pipeline

    def bdp(*x, **k):
        k["shuffle_seed"] = a.shuffle_seed
        dl, vl, st = orig_bdp(*x, **k)
        assert vl is not None, "no held-out split: --val-fraction produced 0 windows"
        VAL.append(vl)
        return dl, vl, st
    T.build_dataset_pipeline = bdp

    SCHED, MODEL = [], []
    orig_init = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *x, **k):
        orig_init(self, model, optimizer, *x, **k)
        SCHED.append(self); MODEL.append(model)
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

    ckpt_dir = os.path.join(os.environ.get("AB209_CKPT", a.out), "ckpt_" + a.tag)
    latest = T.find_latest_checkpoint(ckpt_dir)
    resume = int(latest.split("_step")[-1][:-3]) if latest and "_step" in latest else 0
    paths = {k: os.path.join(a.out, f"{k}_{a.tag}.jsonl") for k in ("steps", "eval")}
    for k, p in paths.items():
        keep = []
        if resume and os.path.exists(p):
            keep = [l for l in open(p) if l.strip() and json.loads(l)["step"] <= resume]
        elif os.path.exists(p) and os.path.getsize(p):
            print(f"[ab209] {p} has rows but no checkpoint: died before the first save, "
                  "starting over from step 0", flush=True)
        with open(p, "w") as f:
            f.writelines(keep)
    if resume:
        print(f"[ab209] RESUMED from {latest} at step {resume}", flush=True)
        with open(os.path.join(a.out, f"resumes_{a.tag}.jsonl"), "a") as f:
            f.write(json.dumps({"step": resume, "ckpt": latest, "wall": time.time()}) + "\n")
    rows = open(paths["steps"], "a")
    evf = open(paths["eval"], "a")
    st = {"n": resume, "t": time.perf_counter(), "first": resume + 1}

    def post(opt, args, kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        n = st["n"] + 1
        tl = float(torch.stack(losses).mean()) if losses else float("nan")
        losses.clear()
        rows.write(json.dumps({"step": n, "sec": now - st["t"], "wall": time.time(),
                               "train_loss": tl,
                               "block": SCHED[0].active_block_name,
                               "lr": [g["lr"] for g in opt.param_groups],
                               "peak_res": torch.cuda.max_memory_reserved(),
                               "total_mem": torch.cuda.get_device_properties(0).total_memory,
                               "res_now": torch.cuda.memory_reserved(),
                               "alloc_now": torch.cuda.memory_allocated(),
                               "drv_free": torch.cuda.mem_get_info()[0],
                               "scan_kernel": DT.SCAN_KERNEL[0],
                               "t_stats": dict(DT.SCAN_T_STATS),
                               "broken_bwd_calls": BROKE["n"],
                               "fnr": dict(sys.modules["hvn.core.fused_norm_residual"].STATS)
                               if "hvn.core.fused_norm_residual" in sys.modules else None}) + "\n")
        rows.flush()
        if n in eval_at:
            for q in MODEL[0].parameters():
                q.grad = None
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            per, hs = heldout()
            torch.cuda.synchronize()
            row = {"step": n, "heldout_nll": sum(per) / len(per), "windows": len(per),
                   "tokens": len(per) * windows[0][1].numel(), "target_sha": hs,
                   "per_window": per, "eval_sec": time.perf_counter() - t0,
                   "eval_peak_res": torch.cuda.max_memory_reserved(),
                   "block": SCHED[0].active_block_name}
            evf.write(json.dumps(row) + "\n"); evf.flush()
            # A resumed process's eval reserved 5.10 GB (vs 4.96 fresh) and tripped the
            # trainer's 85% first-step spill guard right after it. Hand the eval's
            # cached blocks back so the next training step starts from training memory.
            if not a.no_empty_cache:
                torch.cuda.empty_cache()
            print(f"[ab209] EVAL step {n}: held-out {row['heldout_nll']:.5f} "
                  f"({row['eval_sec']:.0f}s, peak {row['eval_peak_res'] / 2**30:.2f} GiB)", flush=True)
        st["n"] = n
        if a.break_scan and n == st["first"]:
            assert BROKE["n"] > 0, "falsifier never fired"
        if n >= stop_at:
            rows.close(); evf.close(); sys.stdout.flush(); os._exit(0)
        torch.cuda.reset_peak_memory_stats()
        st["t"] = time.perf_counter()
    register_optimizer_step_post_hook(post)

    extra = ["--warmup-steps", str(a.warmup)] if a.warmup is not None else []
    sys.argv = ["train_edge_conversational.py"] + list(PROD_ARGV) + [
        "--grad-accum", "8", "--bcd-steps-per-block", str(a.spb),
        "--bcd-layers-per-block", "4",
        "--max-steps", str(a.steps), "--val-fraction", str(a.val_fraction),
        "--val-every", "100000000",
        "--save-every-steps", str(a.save_every), "--keep-checkpoints", "2",
        "--checkpoint-dir", ckpt_dir,
        "--data-dir", os.path.join(MAIN, "data_cache"), "--seed", str(a.seed),
        "--deterministic"] + extra + (a.extra.split() if a.extra else [])
    T.main()


if __name__ == "__main__":
    main()
