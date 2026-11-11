"""#197 Stage 2: does fp8 gradient noise cost held-out loss at 1B? One arm per process.

Author: Rambarun Komaljeet

Runs the real ``train_edge_conversational.main`` with ``PROD_ARGV`` (24L d2048
hybrid, seq 4096, batch 1, grad_accum 8, BCD offload, Muon) and
``--deterministic``. Arms differ ONLY in the Linear arithmetic:

  fp8   production ``--fp8-linear``
  bf16  ``--fp8-linear`` dropped (bf16 autocast Linears)

Everything else is held fixed: ``--seed 0`` (init), data order (the train
sampler's shuffle seed, overridable with ``--shuffle-seed`` to measure the
data-order noise floor), BCD schedule, optimizer, LR schedule.

Held-out loss, not train loss. ``--val-fraction`` makes the trainer hold out a
token-disjoint FineWeb-Edu split (never in the train index); this harness scores
the first ``--eval-windows`` of it (deterministic order) after optimizer steps
listed in ``--eval-at``. The trainer's own ``evaluate`` cannot run with
``--offload`` (parked weights are on the CPU; docs/state.md "Latent, not
fixed"), so the eval here walks the stack itself, loading each parked segment
through the production offload pool exactly as ``EdgeHVN._offload_wrap_plain``
does, under ``model.eval()`` + no_grad + bf16 autocast (so fp8 is OFF in eval:
both arms' weights are scored with the same arithmetic). Per-window losses are
saved so a paired difference and its sampling error can be computed.

Eval after optimizer step 0 (lr 0: weights still at init) must be bit-identical
across arms -- the check that both arms start from the same model and score the
same tokens.
"""
import os, sys, json, time, math, argparse, hashlib

MAIN = r"C:\Users\komal\Documents\Codes\kramba-ai"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["fp8", "bf16"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--eval-at", default="", help="comma list of optimizer-step counts")
    ap.add_argument("--eval-windows", type=int, default=64)
    ap.add_argument("--val-fraction", type=float, default=0.001)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--spb", type=int, default=100, help="--bcd-steps-per-block")
    ap.add_argument("--shuffle-seed", type=int, default=None)
    ap.add_argument("--pert-scale", action="store_true",
                    help="noise-floor arm: multiply the scale of the FIRST e5m2 "
                         "gradient quantization of optimizer step 2 (lr > 0) by "
                         "(1 + 2^-23), one fp32 ulp. Quality-neutral by construction; "
                         "how far the run then drifts from the unperturbed arm is the "
                         "chaos floor an fp8-vs-bf16 difference must clear.")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
    os.chdir(REPO)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    import hvn.core.block_coordinate as bc
    import hvn.edge.chunked_loss as CL
    import hvn.edge.fp8_fused as F8
    assert os.path.abspath(F8.__file__).startswith(REPO), F8.__file__
    from profile_bcd_memory import PROD_ARGV
    argv = list(PROD_ARGV)
    if a.mode == "bf16":
        argv.remove("--fp8-linear")
    eval_at = {int(s) for s in a.eval_at.split(",") if s}

    PERT = {"armed": False, "done": False}
    if a.pert_scale:
        assert a.mode == "fp8", "--pert-scale perturbs an e5m2 scale: fp8 arm only"
        orig_q = F8.fp8_quantize

        def q(t, dtype, transpose=False):
            out = orig_q(t, dtype, transpose)
            if dtype == F8.E5M2 and PERT["armed"] and not PERT["done"]:
                PERT["done"] = True
                inv = out[1] * (1 + 2 ** -23)
                assert inv.item() != out[1].item(), "perturbation is a no-op"
                print("[h197ab] PERT: e5m2 scale x (1+2^-23) at optimizer step 2", flush=True)
                return out[0], inv, out[2]
            return out
        F8.fp8_quantize = q

    import train_edge_conversational as T
    VAL = []
    orig_bdp = T.build_dataset_pipeline

    def bdp(*x, **k):
        if a.shuffle_seed is not None:
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
                # Same key and parameter list as the trainer's tail prefetch
                # (final_norm + decoder: the decoder bias is parked; its tied
                # weight is pinned and skipped by _offloadable).
                tl = seg("final_norm+decoder", [model.final_norm, model.decoder],
                         lambda: orig_cce(model.final_norm(z), y, model.decoder,
                                          chunk=512, use_checkpoint=False))
                per.append(float(tl.float().mean()))
        if was:
            model.train()
        return per, h.hexdigest()

    rows = open(os.path.join(a.out, f"steps_{a.tag}.jsonl"), "w")
    evf = open(os.path.join(a.out, f"eval_{a.tag}.jsonl"), "w")
    st = {"n": 0, "t": time.perf_counter()}

    def post(opt, args, kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        n = st["n"] + 1          # optimizer steps completed
        tl = float(torch.stack(losses).mean()) if losses else float("nan")
        losses.clear()
        rows.write(json.dumps({"step": n, "sec": now - st["t"], "train_loss": tl,
                               "block": SCHED[0].active_block_name,
                               "lr": [g["lr"] for g in opt.param_groups],
                               "peak_res": torch.cuda.max_memory_reserved(),
                               "total_mem": torch.cuda.get_device_properties(0).total_memory,
                               "fp8_fwd": F8.FP8_STATS["fwd"]}) + "\n")
        rows.flush()
        if n in eval_at:
            t0 = time.perf_counter()
            per, hs = heldout()
            row = {"step": n, "heldout_nll": sum(per) / len(per), "windows": len(per),
                   "tokens": len(per) * windows[0][1].numel(), "target_sha": hs,
                   "per_window": per, "eval_sec": time.perf_counter() - t0,
                   "block": SCHED[0].active_block_name}
            evf.write(json.dumps(row) + "\n"); evf.flush()
            print(f"[h197ab] EVAL step {n}: held-out {row['heldout_nll']:.5f} "
                  f"({row['eval_sec']:.0f}s)", flush=True)
        st["n"] = n
        if a.pert_scale and n == 1:
            PERT["armed"] = True
        if a.pert_scale and n == 2:
            assert PERT["done"], "pert never fired: the floor arm would equal arm A"
        if n >= a.steps:
            rows.close(); evf.close(); sys.stdout.flush(); os._exit(0)
        st["t"] = time.perf_counter()
    register_optimizer_step_post_hook(post)

    extra = ["--warmup-steps", str(a.warmup)] if a.warmup is not None else []
    sys.argv = ["train_edge_conversational.py"] + argv + [
        "--grad-accum", "8", "--bcd-steps-per-block", str(a.spb),
        "--max-steps", str(a.steps), "--val-fraction", str(a.val_fraction),
        "--val-every", "100000000",
        "--checkpoint-dir", os.path.join(a.out, "ckpt_" + a.tag),
        "--data-dir", os.path.join(MAIN, "data_cache"), "--seed", "0",
        "--deterministic"] + extra
    T.main()


if __name__ == "__main__":
    main()
