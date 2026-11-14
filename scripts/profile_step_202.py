"""#202: where does the 1B k4 pretrain step go, and is a speed change bit-identical?

Author: Rambarun Komaljeet

Runs the real ``train_edge_conversational.main`` with the #194/#200 k4 production
settings (``PROD_ARGV``, grad_accum 8, fp8, balanced partition, 4 layers per block,
seed 0, deterministic, LR schedule over 2800 steps) with short turns (``--spb``
steps per block) so every block position is visited in a few minutes.

Per optimizer step it writes one JSON row with:
  sec          host wall time between optimizer-step ends (after a sync)
  phases       GPU time per phase, from CUDA events recorded on the compute
               stream at phase boundaries (model / block / loss-slice hooks,
               backward, clip, optimizer). Consecutive-event differences, so
               the phases sum to the step's GPU timeline.
  peak_res     max_memory_reserved over the step (reset after each step; the
               allocator cache is NOT emptied, so earlier turns' cache counts)
  kernels      (profiled steps only, ``--profile``) GPU time per kernel name
  hash         (``--hash``) sha256 of every micro-batch's per-token loss and of
               the active block's gradients at optimizer.step (after clipping)

A block forward call is the first pass when it happens outside ``backward()``
and the checkpoint recompute when inside. Hooks only record events; they add no
syncs except one per optimizer step.

``--perturb-ulp`` moves one element of the first active parameter by one ulp
before the first forward: the falsifier for the hash check (it must change the
hash; if it does not, the hash cannot see a backward bug).
"""
import os, sys, json, time, argparse, hashlib

MAIN = r"C:\Users\komal\Documents\Codes\kramba-ai"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="jsonl output")
    ap.add_argument("--spb", type=int, default=3, help="steps per block (turn length)")
    ap.add_argument("--rotations", type=int, default=1)
    ap.add_argument("--profile", action="store_true",
                    help="torch.profiler (CUDA only) on the last step of each turn")
    ap.add_argument("--hash", action="store_true")
    ap.add_argument("--perturb-ulp", action="store_true")
    ap.add_argument("--perturb-grad", type=float, default=0.0,
                    help="falsifier: scale the gradient reaching the hidden states of the "
                         "first micro-batch of every step by (1 + this), e.g. 2**-20")
    ap.add_argument("--trace-dir", default=None)
    ap.add_argument("--val-every", type=int, default=100000000)
    ap.add_argument("--heldout-every", type=int, default=0,
                    help="run the #194 harness's offload-aware held-out eval after every "
                         "K-th step and record its peak reserved (no empty_cache)")
    ap.add_argument("--eval-windows", type=int, default=8)
    ap.add_argument("--extra", default="", help="extra trainer flags (space separated)")
    ap.add_argument("--set", default="", help="model attributes to set, e.g. _active_no_ckpt=1")
    ap.add_argument("--abort-peak", type=float, default=0.95,
                    help="stop the run if a step's peak reserved passes this share of "
                         "the card (past ~92%% WDDM spills and a step takes minutes)")
    ap.add_argument("--no-frozen-fused", action="store_true",
                    help="frozen ternary weights: old ternary-then-fp8 route")
    ap.add_argument("--loss-recompute", action="store_true",
                    help="chunked loss: old checkpointed slices (grad_in_forward=False)")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
    os.chdir(REPO)
    import torch
    from torch.optim.optimizer import (register_optimizer_step_post_hook,
                                       register_optimizer_step_pre_hook)
    import hvn.core.block_coordinate as bc
    import hvn.edge.chunked_loss as CL
    from profile_bcd_memory import PROD_ARGV
    import train_edge_conversational as T
    import hvn.edge.fp8_fused as F8
    assert os.path.abspath(bc.__file__).startswith(REPO), bc.__file__
    F8.FROZEN_FUSED[0] = not a.no_frozen_fused
    print(f"[s202] frozen fused fp8 quantize: {F8.FROZEN_FUSED[0]}", flush=True)

    nb = 7
    total = nb * a.spb * a.rotations
    SCHED, MODEL = [], []
    EV = []                      # (label, cuda event) since the last step end
    S = {"bwd": False, "n": 0, "t": None, "prof": None, "loss_h": hashlib.sha256(), "mb": 0,
         "eval_peak": None}

    def mark(label):
        e = torch.cuda.Event(enable_timing=True)
        e.record()
        EV.append((label, e))

    # --- phase hooks -----------------------------------------------------
    orig_init = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *x, **k):
        orig_init(self, model, optimizer, *x, **k)
        SCHED.append(self); MODEL.append(model)
        assert len(self.blocks) == nb, [n for n, _ in self.blocks]
        for kv in filter(None, a.set.split(",")):
            k, v = kv.split("=")
            setattr(model, k, int(v))
            print(f"[s202] model.{k} = {int(v)}", flush=True)

        def mpre(mod, inp):
            if not mod.training:
                return
            if S["t"] is None:
                begin()
            mark("fwd.embed")

        def mpost(mod, inp, out):
            if mod.training:
                mark("fwd.post")
        model.register_forward_pre_hook(mpre)
        model.register_forward_hook(mpost)
        for i, blk in enumerate(model.blocks):
            def bpre(mod, inp, i=i):
                if mod.training:
                    mark(f"bwd.recomp.L{i}" if S["bwd"] else f"fwd.L{i}")

            def bpost(mod, inp, out, i=i):
                if mod.training:
                    mark(f"bwd.grad.L{i}" if S["bwd"] else "fwd.glue")
            blk.register_forward_pre_hook(bpre)
            blk.register_forward_hook(bpost)
    bc.BlockCoordinateScheduler.__init__ = init

    def wrap_slice(fn):
        def w(*args, **kw):
            mark("bwd.loss_recompute" if S["bwd"] else "loss.fwd")
            r = fn(*args, **kw)
            mark("bwd.loss_grad" if S["bwd"] else "loss.glue")
            return r
        return w
    CL._slice_loss = wrap_slice(CL._slice_loss)
    CL._slice_loss_padded = wrap_slice(CL._slice_loss_padded)

    o_cce = CL.chunked_cross_entropy

    def cce(*args, **kw):
        if a.loss_recompute:
            kw["grad_in_forward"] = False
        if a.perturb_grad and S["mb"] == 0 and torch.is_grad_enabled():
            args[0].register_hook(lambda g: g * (1.0 + a.perturb_grad))
        S["mb"] += 1
        out = o_cce(*args, **kw)
        if a.hash and torch.is_grad_enabled():
            S["loss_h"].update(out.detach().float().cpu().numpy().tobytes())
        return out
    CL.chunked_cross_entropy = cce

    o_bwd = torch.Tensor.backward

    def backward(self, *args, **kw):
        S["bwd"] = True
        mark("bwd.head")
        try:
            return o_bwd(self, *args, **kw)
        finally:
            mark("post_bwd")
            S["bwd"] = False
    torch.Tensor.backward = backward

    o_clip = torch.nn.utils.clip_grad_norm_

    def clip(*args, **kw):
        mark("clip")
        r = o_clip(*args, **kw)
        mark("pre_opt")
        return r
    torch.nn.utils.clip_grad_norm_ = clip

    VAL = []
    o_bdp = T.build_dataset_pipeline

    def bdp(*x, **k):
        dl, vl, st_ = o_bdp(*x, **k)
        VAL.append(vl)
        return dl, vl, st_
    T.build_dataset_pipeline = bdp
    windows = []

    @torch.no_grad()
    def heldout():
        # scripts/exp_bcd_blocksize_194.py heldout(), minus its empty_cache calls:
        # the trainer's own evaluate() is not offload-aware (it crashes on a
        # CPU-parked weight), so the production runs evaluate this way.
        model = MODEL[0]
        if not windows:
            for i, batch in enumerate(VAL[0]):
                x, y = batch[0], batch[1]
                for b in range(x.shape[0]):
                    windows.append((x[b:b + 1].clone(), y[b:b + 1].clone()))
                if len(windows) >= a.eval_windows:
                    break
            windows[:] = windows[:a.eval_windows]
        pool, act = model._offload_pool, model._offload_active_names
        model.eval()

        def seg(name, mods, fn):
            params = [p for m in mods for p in m.parameters()]
            if name in act or not pool._offloadable(params):
                return fn()
            pool.load(name, params)
            out = fn()
            pool.evict(name)
            return out
        dev, per = torch.device("cuda"), []
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in windows:
                x, y = x.to(dev), y.to(dev)
                pos = torch.arange(x.shape[1], device=dev).unsqueeze(0)
                z = seg("token_embedding", [model.token_embedding], lambda: model.token_embedding(x))
                for i, blk in enumerate(model.blocks):
                    z = seg(f"blocks.{i}", [blk], lambda: blk(z, pos))
                tl = seg("final_norm+decoder", [model.final_norm, model.decoder],
                         lambda: o_cce(model.final_norm(z), y, model.decoder,
                                       chunk=512, use_checkpoint=False))
                per.append(float(tl.float().mean()))
        model.train()
        return per

    o_eval = T.evaluate

    def evaluate(*args, **kw):
        torch.cuda.reset_peak_memory_stats()
        r = o_eval(*args, **kw)
        S["eval_peak"] = torch.cuda.max_memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        return r
    T.evaluate = evaluate

    # --- step boundaries -------------------------------------------------
    rows = open(a.out, "w")
    GRAD_H = {}

    def begin():
        torch.cuda.synchronize()
        S["t"] = time.perf_counter()
        pos = S["n"] % a.spb
        if a.profile and pos == a.spb - 1:
            from torch.profiler import profile, ProfilerActivity
            p = profile(activities=[ProfilerActivity.CUDA]); p.__enter__(); S["prof"] = p
        if a.perturb_ulp and S["n"] == 0:
            s = SCHED[0]
            p0 = [p for p in dict(s.blocks)[s.active_block_name] if p.is_cuda][0]
            with torch.no_grad():
                flat = p0.view(-1)
                flat[0] = torch.nextafter(flat[0], flat[0] + 1)
            print(f"[s202] perturbed {tuple(p0.shape)} [0] by one ulp", flush=True)

    def pre(opt, args, kw):
        mark("optim")
        if a.hash:
            s = SCHED[0]
            h = hashlib.sha256()
            n = nz = 0
            for p in dict(s.blocks)[s.active_block_name]:
                assert p.grad is not None, "active param without grad"
                g = p.grad.detach()
                h.update(g.float().cpu().numpy().tobytes())
                n += g.numel(); nz += int((g != 0).sum())
            GRAD_H.update(h=h.hexdigest(), n=n, nz=nz)

    def post(opt, args, kw):
        mark("end")
        torch.cuda.synchronize()
        sec = time.perf_counter() - S["t"]
        n = S["n"] + 1
        s = SCHED[0]
        ph = {}
        for (lab, e0), (_, e1) in zip(EV[:-1], EV[1:]):
            ph[lab] = ph.get(lab, 0.0) + e0.elapsed_time(e1) / 1000.0
        EV.clear()
        row = {"step": n, "sec": sec, "block": s.active_block_name,
               "pos_in_turn": (n - 1) % a.spb + 1, "phases": ph,
               "gpu_sum": sum(ph.values()),
               "lr": [g["lr"] for g in opt.param_groups],
               "peak_res": torch.cuda.max_memory_reserved(),
               "peak_alloc": torch.cuda.max_memory_allocated(),
               "total_mem": torch.cuda.get_device_properties(0).total_memory}
        if S["eval_peak"] is not None:
            row["eval_peak_res"] = S["eval_peak"]; S["eval_peak"] = None
        if a.heldout_every and n % a.heldout_every == 0:
            # As the trainer does right after optimizer.step(): zero_grad(set_to_none).
            for q in MODEL[0].parameters():
                q.grad = None
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            per = heldout()
            torch.cuda.synchronize()
            row["heldout"] = sum(per) / len(per)
            row["heldout_sec"] = time.perf_counter() - t0
            row["heldout_peak_res"] = torch.cuda.max_memory_reserved()
        if a.hash:
            row["loss_sha"] = S["loss_h"].hexdigest(); S["loss_h"] = hashlib.sha256()
            row["grad_sha"], row["grad_n"], row["grad_nz"] = GRAD_H["h"], GRAD_H["n"], GRAD_H["nz"]
        if S["prof"] is not None:
            p = S["prof"]; S["prof"] = None
            p.__exit__(None, None, None)
            kt = {}
            for ev in p.events():
                if ev.device_type == torch.autograd.DeviceType.CUDA:
                    kt[ev.name] = kt.get(ev.name, 0.0) + ev.device_time / 1e6
            row["kernels"] = dict(sorted(kt.items(), key=lambda kv: -kv[1])[:150])
        rows.write(json.dumps(row) + "\n"); rows.flush()
        print(f"[s202] step {n} {s.active_block_name[:16]} pos {row['pos_in_turn']} "
              f"{sec:.3f}s gpu {row['gpu_sum']:.3f}s peak {row['peak_res'] / row['total_mem']:.1%}"
              + (f" grad {row['grad_sha'][:12]} loss {row['loss_sha'][:12]}" if a.hash else ""),
              flush=True)
        S["n"] = n
        if row["peak_res"] / row["total_mem"] > a.abort_peak:
            print(f"[s202] ABORT: peak reserved over {a.abort_peak:.0%}", flush=True)
            rows.close(); sys.stdout.flush(); os._exit(3)
        if n >= total:
            rows.close(); sys.stdout.flush(); os._exit(0)
        torch.cuda.reset_peak_memory_stats()
        S["mb"] = 0
        # The next step starts here: the trainer's zero_grad / BCD switch / data
        # fetch belong to it ("post_opt").
        mark("post_opt")
        S["t"] = None
        begin()
    register_optimizer_step_pre_hook(pre)
    register_optimizer_step_post_hook(post)

    sys.argv = ["train_edge_conversational.py"] + list(PROD_ARGV) + [
        "--grad-accum", "8", "--bcd-steps-per-block", str(a.spb),
        "--bcd-layers-per-block", "4",
        "--max-steps", "2800", "--val-fraction", "0.001", "--val-every", str(a.val_every),
        "--save-every-steps", "100000",
        "--checkpoint-dir", os.path.join(os.environ.get("S202_CKPT", os.path.dirname(os.path.abspath(a.out))), "ckpt_s202"),
        "--data-dir", os.path.join(MAIN, "data_cache"), "--seed", "0",
        "--deterministic"] + (a.extra.split() if a.extra else [])
    T.main()


if __name__ == "__main__":
    main()
