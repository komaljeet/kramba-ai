"""#214: BCD quality penalty vs full backprop, held-out loss. One arm per process.

Author: Rambarun Komaljeet

Copy of scripts/ab_heldout_209.py (the #209 long held-out A/B harness: same held-out
walk, same pinned data order, same resume logic) with these changes:

  --mode full   all layers trainable every step: PROD_ARGV minus --offload, so no BCD and
                no host-RAM parking; --grad-checkpoint added (exact recompute, needed to fit).
  --mode bcd    PROD_ARGV as is (BCD + offload, 4 layers per block = 7 training blocks at
                24 layers), --bcd-steps-per-block from --spb.
  --d-model / --n-heads   shrink the 1B family (d2048 x 24 layers, 64 heads = head dim 32)
                to a size where full backprop fits on the 6 GB card. Layer count, attention
                layers, block layout and every other flag stay the 1B production ones.
  model capture via EdgeHVN.__init__ (the full arm has no BCD scheduler).
  held-out walk: plain forward when the model has no offload pool (full arm).
  grad_numel    per step, the number of parameter elements that received a gradient
                (read right after optimizer.step, before zero_grad). Full must be every
                trainable element every step; BCD only the active block's. This is the
                check that the "full" arm really trains everything and BCD really freezes.

Held fixed across arms and seeds: data order (shuffle seed pinned), the held-out windows
(first --eval-windows of the token-disjoint FineWeb-Edu val split, sha logged per eval),
LR schedule (warmup + cosine over --steps), grad_accum, tokens per step.
Varies: --seed = model init only; --mode / --spb = the thing under test.
"""
import os, sys, json, time, argparse, hashlib

MAIN = r"C:\Users\komal\Documents\Codes\kramba-ai"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def set_flag(argv, flag, value):
    if flag in argv:
        argv[argv.index(flag) + 1] = value
    else:
        argv += [flag, value]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["full", "bcd"], required=True)
    ap.add_argument("--d-model", type=int, required=True)
    ap.add_argument("--n-heads", type=int, required=True)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--spb", type=int, default=100, help="--bcd-steps-per-block")
    ap.add_argument("--eval-at", default="", help="comma list of optimizer-step counts")
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--val-fraction", type=float, default=0.001)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0, help="model init seed")
    ap.add_argument("--shuffle-seed", type=int, default=0, help="data order (pinned)")
    ap.add_argument("--extra", default="", help="extra trainer flags")
    ap.add_argument("--stop-at", type=int, default=None,
                    help="exit after this step (schedule still spans --steps)")
    ap.add_argument("--save-every", type=int, default=500)
    a = ap.parse_args()
    stop_at = a.stop_at or a.steps
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
    os.chdir(REPO)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    import hvn.edge.model as EM
    import hvn.edge.chunked_loss as CL
    import hvn.core.deltanet_triton as DT
    assert os.path.abspath(DT.__file__).startswith(REPO), DT.__file__
    import hvn.core.block_coordinate as bc
    from profile_bcd_memory import PROD_ARGV
    eval_at = {int(s) for s in a.eval_at.split(",") if s}

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

    MODEL, SCHED = [], []
    orig_minit = EM.EdgeHVN.__init__

    def minit(self, *x, **k):
        orig_minit(self, *x, **k)
        MODEL.append(self)
    EM.EdgeHVN.__init__ = minit
    orig_sinit = bc.BlockCoordinateScheduler.__init__

    def sinit(self, *x, **k):
        orig_sinit(self, *x, **k)
        SCHED.append(self)
    bc.BlockCoordinateScheduler.__init__ = sinit

    losses = []
    orig_cce = CL.chunked_cross_entropy

    def cce(*x, **k):
        out = orig_cce(*x, **k)
        if torch.is_grad_enabled():
            losses.append(out.detach().float().mean())
        return out
    CL.chunked_cross_entropy = cce

    windows = []
    TRAINED = []

    def the_model(opt=None):
        # The model whose parameters the optimizer steps (the audit builds copies).
        if not TRAINED and opt is not None:
            ids = {id(p) for g in opt.param_groups for p in g["params"]}
            for m in reversed(MODEL):
                if any(id(p) in ids for p in m.parameters()):
                    TRAINED.append(m)
                    break
        return TRAINED[0]

    @torch.no_grad()
    def heldout():
        model = the_model()
        if not windows:
            for i, (x, y, _src) in enumerate(VAL[0]):
                if len(windows) >= a.eval_windows:
                    break
                for b in range(x.shape[0]):
                    windows.append((x[b:b + 1].clone(), y[b:b + 1].clone()))
            windows[:] = windows[:a.eval_windows]
            assert len(windows) == a.eval_windows, len(windows)
        pool = getattr(model, "_offload_pool", None)
        act = model._offload_active_names
        was = model.training
        model.eval()

        def seg(name, mods, fn):
            if pool is None or act is None:
                return fn()
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

    ckpt_dir = os.path.join(os.environ.get("AB214_CKPT", a.out), "ckpt_" + a.tag)
    latest = T.find_latest_checkpoint(ckpt_dir)
    resume = int(latest.split("_step")[-1][:-3]) if latest and "_step" in latest else 0
    paths = {k: os.path.join(a.out, f"{k}_{a.tag}.jsonl") for k in ("steps", "eval")}
    for k, p in paths.items():
        keep = []
        if resume and os.path.exists(p):
            keep = [l for l in open(p) if l.strip() and json.loads(l)["step"] <= resume]
        elif os.path.exists(p) and os.path.getsize(p):
            print(f"[ab214] {p} has rows but no checkpoint: died before the first save, "
                  "starting over from step 0", flush=True)
        with open(p, "w") as f:
            f.writelines(keep)
    if resume:
        print(f"[ab214] RESUMED from {latest} at step {resume}", flush=True)
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
        model = the_model(opt)
        grad_numel = sum(p.numel() for p in model.parameters() if p.grad is not None)
        rows.write(json.dumps({"step": n, "sec": now - st["t"], "wall": time.time(),
                               "train_loss": tl,
                               "block": SCHED[0].active_block_name if SCHED else "all",
                               "grad_numel": grad_numel,
                               "lr": [g["lr"] for g in opt.param_groups],
                               "peak_res": torch.cuda.max_memory_reserved(),
                               "peak_alloc": torch.cuda.max_memory_allocated(),
                               "total_mem": torch.cuda.get_device_properties(0).total_memory,
                               "scan_kernel": DT.SCAN_KERNEL[0]}) + "\n")
        rows.flush()
        if n == st["first"]:
            total = sum(p.numel() for p in model.parameters())
            print(f"[ab214] step {n}: params {total:,}  grad_numel {grad_numel:,}  "
                  f"mode {a.mode}  scheduler {'yes' if SCHED else 'no'}", flush=True)
            if a.mode == "full":
                assert not SCHED, "full arm built a BCD scheduler"
                assert grad_numel == total, (grad_numel, total)
            else:
                assert SCHED and 0 < grad_numel < total, (grad_numel, total)
        if n in eval_at:
            for q in model.parameters():
                q.grad = None
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            per, hs = heldout()
            torch.cuda.synchronize()
            row = {"step": n, "heldout_nll": sum(per) / len(per), "windows": len(per),
                   "tokens": len(per) * windows[0][1].numel(), "target_sha": hs,
                   "per_window": per, "eval_sec": time.perf_counter() - t0,
                   "eval_peak_res": torch.cuda.max_memory_reserved(),
                   "block": SCHED[0].active_block_name if SCHED else "all"}
            evf.write(json.dumps(row) + "\n"); evf.flush()
            torch.cuda.empty_cache()
            print(f"[ab214] EVAL step {n}: held-out {row['heldout_nll']:.5f} "
                  f"({row['eval_sec']:.0f}s, peak {row['eval_peak_res'] / 2**30:.2f} GiB)", flush=True)
        st["n"] = n
        if n >= stop_at:
            rows.close(); evf.close(); sys.stdout.flush(); os._exit(0)
        torch.cuda.reset_peak_memory_stats()
        st["t"] = time.perf_counter()
    register_optimizer_step_post_hook(post)

    argv = list(PROD_ARGV)
    set_flag(argv, "--d-model", str(a.d_model))
    set_flag(argv, "--n-heads", str(a.n_heads))
    if a.mode == "full":
        argv.remove("--offload")
        argv += ["--grad-checkpoint"]
    else:
        argv += ["--bcd-steps-per-block", str(a.spb), "--bcd-layers-per-block", "4"]
    extra = ["--warmup-steps", str(a.warmup)] if a.warmup is not None else []
    sys.argv = ["train_edge_conversational.py"] + argv + [
        "--grad-accum", str(a.grad_accum),
        "--max-steps", str(a.steps), "--val-fraction", str(a.val_fraction),
        "--val-every", "100000000",
        "--save-every-steps", str(a.save_every), "--keep-checkpoints", "2",
        "--checkpoint-dir", ckpt_dir,
        "--data-dir", os.path.join(MAIN, "data_cache"), "--seed", str(a.seed),
        "--deterministic"] + extra + (a.extra.split() if a.extra else [])
    print("[ab214] argv:", " ".join(sys.argv[1:]), flush=True)
    T.main()


if __name__ == "__main__":
    main()
