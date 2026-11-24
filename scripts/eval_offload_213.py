"""#213: the trainer's own --val-every eval on the real 1B k4 --offload run.

Author: Rambarun Komaljeet

Runs the real ``train_edge_conversational.main`` with ``PROD_ARGV`` + the k4
production settings (4 layers per BCD block, grad_accum 8, --deterministic) and
short turns (``--spb``) so the run passes every one of the 7 BCD block
positions. The LR schedule spans 2800 steps with no warmup, so every step has a
real (nonzero, near-peak) LR; the run exits after ``--steps``.

Per optimizer step (one JSON row in ``steps_<tag>.jsonl``):
  loss_sha   sha256 of every training micro-batch's per-token loss
  grad_sha   sha256 of the active block's gradients at optimizer.step
  rng_sha    sha256 of the CPU + CUDA RNG states after optimizer.step
  sec        wall time of the training step (post-hook to post-hook, minus any
             eval in between)
  peak_res   max_memory_reserved over the training step (reset at its first
             forward, i.e. after the previous step's eval and empty_cache)
Per eval (``evals_<tag>.jsonl``): wall time, peak reserved during the eval, the
active block, every window's mean loss, the targets' sha256, and evaluate()'s
own overall number.

``--compare-harness``: after the LAST step's trainer eval, run the #209 harness
walk (scripts/ab_heldout_209.py ``heldout``) on the same windows at the same
state and record both per-window lists.

``--falsify``: the eval skips its RNG restore (torch.random.fork_rng made a
no-op: origin/main's behaviour) and nudges one element of the optimizer state
after every eval. Its hashes must differ from the eval-off arm, else the
identity check could not see an eval that changes training.
"""
import os, sys, json, time, argparse, hashlib, contextlib

MAIN = r"C:\Users\komal\Documents\Codes\kramba-ai"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--steps", type=int, default=14)
    ap.add_argument("--spb", type=int, default=2)
    ap.add_argument("--val-every", type=int, required=True)
    ap.add_argument("--val-batches", type=int, default=64)
    ap.add_argument("--compare-harness", action="store_true")
    ap.add_argument("--falsify", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
    os.chdir(REPO)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.optim.optimizer import (register_optimizer_step_post_hook,
                                       register_optimizer_step_pre_hook)
    import hvn.core.block_coordinate as bc
    import hvn.edge.chunked_loss as CL
    assert os.path.abspath(bc.__file__).startswith(REPO), bc.__file__
    from profile_bcd_memory import PROD_ARGV
    import train_edge_conversational as T

    if a.falsify:
        torch.random.fork_rng = lambda *x, **k: contextlib.nullcontext()
        print("[e213] FALSIFIER: eval without RNG restore + optimizer-state nudge", flush=True)

    VAL, SCHED, MODEL = [], [], []
    o_bdp = T.build_dataset_pipeline

    def bdp(*x, **k):
        dl, vl, st = o_bdp(*x, **k)
        assert vl is not None
        VAL.append(vl)
        return dl, vl, st
    T.build_dataset_pipeline = bdp

    S = {"loss": hashlib.sha256(), "grad": None, "nz": 0, "n": 0, "t": None,
         "eval_sec": 0.0, "evwin": None, "evsha": None, "res_start": None}
    o_init = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *x, **k):
        o_init(self, model, optimizer, *x, **k)
        SCHED.append(self); MODEL.append(model)

        def mpre(mod, inp):
            if mod.training and torch.is_grad_enabled() and S["res_start"] is None:
                torch.cuda.synchronize()
                S["res_start"] = torch.cuda.memory_reserved()
                torch.cuda.reset_peak_memory_stats()
                if S["t"] is None:
                    S["t"] = time.perf_counter()
        model.register_forward_pre_hook(mpre)
    bc.BlockCoordinateScheduler.__init__ = init

    o_cce = CL.chunked_cross_entropy

    def cce(*x, **k):
        out = o_cce(*x, **k)
        if torch.is_grad_enabled():
            S["loss"].update(out.detach().float().cpu().numpy().tobytes())
        elif S["evwin"] is not None:
            S["evwin"].append(float(out.float().mean()))
            S["evsha"].update(x[1].cpu().numpy().tobytes())
        return out
    CL.chunked_cross_entropy = cce

    steps_f = open(os.path.join(a.out, f"steps_{a.tag}.jsonl"), "w")
    evals_f = open(os.path.join(a.out, f"evals_{a.tag}.jsonl"), "w")

    def pre(opt, args, kw):
        s = SCHED[0]
        h, nz, n = hashlib.sha256(), 0, 0
        for p in dict(s.blocks)[s.active_block_name]:
            assert p.grad is not None, "active param without grad"
            g = p.grad.detach()
            h.update(g.float().cpu().numpy().tobytes())
            n += g.numel(); nz += int((g != 0).sum())
        S["grad"], S["nz"], S["gn"] = h.hexdigest(), nz, n

    def post(opt, args, kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        n = S["n"] + 1
        r = hashlib.sha256()
        r.update(torch.get_rng_state().numpy().tobytes())
        r.update(torch.cuda.get_rng_state().numpy().tobytes())
        row = {"step": n, "block": SCHED[0].active_block_name,
               "lr": [g["lr"] for g in opt.param_groups],
               "sec": now - S["t"] - S["eval_sec"],
               "peak_res": torch.cuda.max_memory_reserved(),
               "res_at_start": S["res_start"],
               "total_mem": torch.cuda.get_device_properties(0).total_memory,
               "loss_sha": S["loss"].hexdigest(), "grad_sha": S["grad"],
               "grad_nz": S["nz"], "grad_n": S["gn"], "rng_sha": r.hexdigest()}
        steps_f.write(json.dumps(row) + "\n"); steps_f.flush()
        print(f"[e213] step {n} {row['block'][:20]} {row['sec']:.2f}s "
              f"peak {row['peak_res'] / row['total_mem']:.1%} loss {row['loss_sha'][:10]} "
              f"grad {row['grad_sha'][:10]} rng {row['rng_sha'][:10]}", flush=True)
        S.update(n=n, t=now, eval_sec=0.0, loss=hashlib.sha256(), res_start=None)
        if n >= a.steps and not (n % a.val_every == 0):
            steps_f.close(); evals_f.close(); sys.stdout.flush(); os._exit(0)
    register_optimizer_step_pre_hook(pre)
    register_optimizer_step_post_hook(post)

    @torch.no_grad()
    def harness():
        """scripts/ab_heldout_209.py heldout(), same walk, same chunk."""
        model = MODEL[0]
        windows = []
        for x, y, _src in VAL[0]:
            for b in range(x.shape[0]):
                windows.append((x[b:b + 1].clone(), y[b:b + 1].clone()))
            if len(windows) >= a.val_batches:
                break
        windows = windows[:a.val_batches]
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
        per, dev = [], torch.device("cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for x, y in windows:
                x, y = x.to(dev), y.to(dev)
                pos = torch.arange(x.shape[1], device=dev).unsqueeze(0)
                z = seg("token_embedding", [model.token_embedding],
                        lambda: model.token_embedding(x))
                for i, blk in enumerate(model.blocks):
                    z = seg(f"blocks.{i}", [blk], lambda: blk(z, pos))
                tl = seg("final_norm+decoder", [model.final_norm, model.decoder],
                         lambda: o_cce(model.final_norm(z), y, model.decoder,
                                       chunk=512, use_checkpoint=False))
                per.append(float(tl.float().mean()))
        model.train()
        return per

    o_eval = T.evaluate

    def pool_bytes():
        pool = MODEL[0]._offload_pool
        free = sum(b.numel() * b.element_size() for sets in pool._free_buffers.values()
                   for bufs, _ in sets for b in bufs)
        res = sum(b.numel() * b.element_size() for e in pool._resident.values()
                  for b in e["buffers"])
        nsets = sum(len(s) for s in pool._free_buffers.values())
        return {"free": free, "resident": res, "free_sets": nsets}

    def evaluate(*x, **k):
        torch.cuda.synchronize()
        pb0, al0 = pool_bytes(), torch.cuda.memory_allocated()
        res0 = torch.cuda.memory_reserved()
        torch.cuda.reset_peak_memory_stats()
        S["evwin"], S["evsha"] = [], hashlib.sha256()
        t0 = time.perf_counter()
        out = o_eval(*x, **k)
        torch.cuda.synchronize()
        sec = time.perf_counter() - t0
        row = {"step": S["n"], "block": SCHED[0].active_block_name, "eval_sec": sec,
               "eval_peak_res": torch.cuda.max_memory_reserved(),
               "res_before": res0, "res_after_eval": torch.cuda.memory_reserved(),
               "alloc_before": al0, "alloc_after": torch.cuda.memory_allocated(),
               "pool_before": pb0, "pool_after": pool_bytes(),
               "total_mem": torch.cuda.get_device_properties(0).total_memory,
               "overall": out[0], "windows": len(S["evwin"]),
               "per_window": S["evwin"], "target_sha": S["evsha"].hexdigest()}
        S["evwin"] = None
        if a.falsify:
            st = SCHED[0].optimizer.state
            for p, s in st.items():
                for v in s.values():
                    if torch.is_tensor(v) and v.is_floating_point() and v.numel() > 1:
                        v.view(-1)[0] += 1e-3
                        break
                else:
                    continue
                break
        last = S["n"] >= a.steps
        if last and a.compare_harness:
            with torch.random.fork_rng(devices=[torch.device("cuda")]):
                t1 = time.perf_counter()
                hw = harness()
                torch.cuda.synchronize()
            row["harness_per_window"] = hw
            row["harness_sec"] = time.perf_counter() - t1
            row["harness_mean"] = sum(hw) / len(hw)
            row["bit_equal_windows"] = sum(int(p == q) for p, q in zip(hw, row["per_window"]))
            row["max_abs_diff"] = max(abs(p - q) for p, q in zip(hw, row["per_window"]))
        evals_f.write(json.dumps(row) + "\n"); evals_f.flush()
        print(f"[e213] EVAL after step {S['n']} ({row['block'][:20]}): {out[0]:.5f} "
              f"{len(row['per_window'])} windows {sec:.1f}s "
              f"peak {row['eval_peak_res'] / row['total_mem']:.1%}"
              + (f" | harness {row['harness_mean']:.5f} bit-equal "
                 f"{row['bit_equal_windows']}/{len(hw)} max|d| {row['max_abs_diff']:.3g}"
                 if "harness_mean" in row else ""), flush=True)
        S["eval_sec"] += time.perf_counter() - t0
        if last:
            steps_f.close(); evals_f.close(); sys.stdout.flush(); os._exit(0)
        return out
    T.evaluate = evaluate

    sys.argv = ["train_edge_conversational.py"] + list(PROD_ARGV) + [
        "--grad-accum", "8", "--bcd-steps-per-block", str(a.spb),
        "--bcd-layers-per-block", "4", "--warmup-steps", "0",
        "--max-steps", "2800", "--val-fraction", "0.001",
        "--val-every", str(a.val_every), "--val-batches", str(a.val_batches),
        "--save-every-steps", "100000",
        "--checkpoint-dir", os.path.join(a.out, "ckpt_" + a.tag),
        "--data-dir", os.path.join(MAIN, "data_cache"), "--seed", "0",
        "--deterministic"]
    T.main()


if __name__ == "__main__":
    main()
