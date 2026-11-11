"""#197 Stage 1: step-0 gradient of the REAL 1B production step, fp8 vs higher precision.

Author: Rambarun Komaljeet

Runs the real ``train_edge_conversational.main`` with ``PROD_ARGV`` (24L d2048
hybrid, seq 4096, batch 1, grad_accum 8, BCD offload, Muon) and ``--deterministic``
at one chosen BCD block, captures every trainable gradient at the pre-hook of
optimizer step 0 (lr 0, so step 0 is the only step whose inputs are identical
across arms), saves them, and exits. Nothing in production code is changed;
modes are monkeypatches applied before ``main`` runs.

Modes
-----
fp8      production: ``--fp8-linear`` (e4m3 x/W forward, e5m2 grad backward).
bf16     reference: ``--fp8-linear`` dropped -> every Linear in bf16 autocast.
fp8hb    fp8 forward, high-precision backward: same fp8 forward (e4m3 x/W) but
         the backward GEMMs use the bf16 incoming gradient (no e5m2) times the
         DEQUANTIZED saved e4m3 x/W, in a bf16 GEMM. Isolates the e5m2 part.

Integrity, recorded per run so a comparison can prove it compares the SAME step
  - sha256 of the active block's parameters at step 0 (same init),
  - sha256 of every micro-batch's targets (same data).
Falsifier: ``--pert`` multiplies the e5m2 scale of the FIRST fp8 backward call
by (1 + 2^-23), one fp32 ulp. A comparison that cannot see that is vacuous.
"""
import os, sys, json, time, argparse, hashlib

MAIN = r"C:\Users\komal\Documents\Codes\kramba-ai"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--active", type=int, required=True)
    ap.add_argument("--mode", choices=["fp8", "bf16", "fp8hb"], required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out", required=True, help="dir for grads/meta (scratch)")
    ap.add_argument("--pert", action="store_true")
    ap.add_argument("--pert-w", default="",
                    help="NAME: raise the first element of this (frozen) weight by one "
                         "bf16 ulp before parking -- the same tiny change to the "
                         "function for any mode, to measure each path's sensitivity")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
    os.chdir(REPO)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    import torch
    from torch.optim.optimizer import register_optimizer_step_pre_hook
    import hvn.core.block_coordinate as bc
    import hvn.edge.chunked_loss as CL
    import hvn.edge.fp8_fused as F8
    assert os.path.abspath(F8.__file__).startswith(REPO), F8.__file__
    from profile_bcd_memory import PROD_ARGV
    argv = list(PROD_ARGV)
    if a.mode == "bf16":
        argv.remove("--fp8-linear")

    cnt = {"bwd": 0}
    if a.mode == "fp8hb":
        def bwd(ctx, g):
            xq_t, xs, wq_t, ws = ctx.saved_tensors
            need_x, need_w = ctx.needs_input_grad[0], ctx.needs_input_grad[1]
            gb = g.to(torch.bfloat16)
            dx = dw = None
            if need_x:   # dx = g @ W ; wq_t is W^T [K, O]
                w = (wq_t.float() * ws).to(torch.bfloat16)
                dx = (gb @ w.t()).to(ctx.x_dtype)
            if need_w:   # dW = g^T @ x ; xq_t is x^T [K, N]
                xt = (xq_t.float() * xs).to(torch.bfloat16)
                dw = (gb.t() @ xt.t()).to(ctx.w_dtype)
            cnt["bwd"] += 1
            return dx, dw, None, None, None
        F8._FP8Linear.backward = staticmethod(bwd)
        print("[h197] fp8 forward, bf16 backward from dequantized e4m3 x/W", flush=True)
    if a.pert:
        orig_q = F8.fp8_quantize

        def q(t, dtype, transpose=False):
            out = orig_q(t, dtype, transpose)
            if dtype == F8.E5M2 and cnt["bwd"] == 0:
                cnt["bwd"] += 1
                inv = out[1] * (1 + 2 ** -23)
                assert inv.item() != out[1].item(), "perturbation is a no-op"
                print("[h197] PERT: first e5m2 scale x (1+2^-23)", flush=True)
                return out[0], inv, out[2]
            return out
        F8.fp8_quantize = q

    calls = []   # sha of each loss call's targets; the last 8 are step 0's micro-batches
    orig_cce = CL.chunked_cross_entropy

    def cce(hidden, targets, *x, **k):
        calls.append(hashlib.sha256(targets.detach().cpu().numpy().tobytes()).hexdigest())
        return orig_cce(hidden, targets, *x, **k)
    CL.chunked_cross_entropy = cce

    SCHED, MODEL = [], []
    orig_init = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *x, **k):
        k["initial_active_idx"] = a.active
        if a.pert_w:
            with torch.no_grad():
                w = dict(model.named_parameters())[a.pert_w]
                f = w.view(-1)
                v = f[0].to(torch.bfloat16).reshape(1)
                v2 = (v.view(torch.int16) + 1).view(torch.bfloat16)
                old = float(f[0]); f[0] = v2.float()[0]
                assert float(f[0]) != old
                print(f"[h197] PERT-W {a.pert_w}[0] {old!r} -> {float(f[0])!r}", flush=True)
        orig_init(self, model, optimizer, *x, **k)
        SCHED.append(self); MODEL.append(model)
    bc.BlockCoordinateScheduler.__init__ = init
    t0 = time.time()

    def pre(opt, args, kw):
        torch.cuda.synchronize()
        names = {id(p): n for n, p in MODEL[0].named_parameters()}
        grads, ph = {}, hashlib.sha256()
        for g in opt.param_groups:
            for p in g["params"]:
                if p.grad is None:
                    continue
                n = names[id(p)]
                grads[n] = p.grad.detach().float().cpu()
                ph.update(n.encode()); ph.update(p.detach().float().cpu().numpy().tobytes())
        assert grads and len(calls) >= 8, (len(grads), len(calls))
        tgt = hashlib.sha256("".join(calls[-8:]).encode())
        torch.save(grads, os.path.join(a.out, f"g_{a.tag}.pt"))
        meta = {"tag": a.tag, "mode": a.mode, "active": a.active, "pert": a.pert, "pert_w": a.pert_w,
                "block": SCHED[0].active_block_name, "n_blocks": len(SCHED[0].blocks),
                "param_sha": ph.hexdigest(), "target_sha": tgt.hexdigest(),
                "loss_calls": len(calls), "tensors": len(grads),
                "numel": sum(v.numel() for v in grads.values()),
                "peak_alloc": torch.cuda.max_memory_allocated(),
                "peak_res": torch.cuda.max_memory_reserved(),
                "total_mem": torch.cuda.get_device_properties(0).total_memory,
                "fp8": dict(F8.FP8_STATS), "hb_bwd_calls": cnt["bwd"],
                "wall_s": time.time() - t0}
        with open(os.path.join(a.out, f"m_{a.tag}.json"), "w") as f:
            json.dump(meta, f, indent=1)
        print("[h197]", meta, flush=True)
        sys.stdout.flush(); os._exit(0)
    register_optimizer_step_pre_hook(pre)

    import train_edge_conversational as T
    sys.argv = ["train_edge_conversational.py"] + argv + [
        "--grad-accum", "8", "--bcd-steps-per-block", "100", "--max-steps", "1000",
        "--checkpoint-dir", os.path.join(a.out, "ckpt_" + a.tag),
        "--data-dir", os.path.join(MAIN, "data_cache"), "--seed", "0", "--deterministic"]
    T.main()


if __name__ == "__main__":
    main()
