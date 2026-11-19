"""#218 gate 1, step 3: do the RAM cuts break the update? Real 1B group, >= 20 steps.

Author: Rambarun Komaljeet

The real 1B layers 4-7 (DeltaNet, DeltaNet, attention, DeltaNet; d 2048, 64 heads,
window 4096), built exactly as scripts/bench_streamed_group_217.py builds them
(production ``--fast --fp8-linear`` stack, ternary + fp8 GEMMs, Triton scan,
bf16 autocast, deterministic mode), 8 micro-batches x 4096 tokens, the
production Muon (per-layer optimizer, per-layer clip 1.0, as in #217).

Arms (host-state precision; everything on the GPU stays as in the trainer):
  ref      fp32 master, fp32 Muon momentum, fp32 parked group input   (#217 exact)
  lean     bf16 master with STOCHASTIC rounding, bf16 momentum (round-to-nearest),
           bf16 parked group input                                   (the 16 GB plan)
  rne      as lean but the master is rounded to NEAREST (no SR)     (falsifier)
  lean_m8  as lean but momentum in int8, absmax per 256-value block (extra lever)
Non-matrix (AdamW) params and their moments stay fp32 in every arm (5 MB/group).

How a step runs for one arm (this is the streamed design's math, made resident:
#217 showed streamed == resident bit for bit, so streaming is left out here):
  1. weights = the arm's stored host master, upcast to fp32 on the GPU
  2. forward all 8 mb under no_grad -> loss and dL/dy (y the group output)
  3. re-run from the PARKED group input (bf16-rounded in lean arms), per layer
     torch.utils.checkpoint, backward with dL/dy: this is the gradient the
     streamed design computes (its backward never sees the fp32 input)
  4. per-layer clip 1.0, production Muon/AdamW step in fp32 on the GPU
  5. write back: master fp32 / bf16-SR / bf16-RNE, momentum fp32 / bf16 / int8.

Objective: the group learns to imitate a teacher group (same architecture, other
seed) on Gaussian inputs: loss = mean((y - T)^2). 4 training batches are cycled;
batch 99 is never trained on (held-out loss). The objective is synthetic: this
is a NUMERICS proxy (does the update survive the cast?), not a quality result.
A held-out LM-loss A/B at 1B is still required.

Metrics per step and arm, vs ref at the same step:
  * kept   = <stored update, intended update> / |intended|^2 for this step
             (intended = the fp32 weight the optimizer produced, before the
             cast). 1 = the whole update reached the master; RNE at small lr
             should be well below 1 -- that is the falsifier.
  * cum_err = |U_arm - U_ref| / |U_ref|, U = master now - master at step 0
    cum_kept = <U_arm, U_ref> / |U_ref|^2
  * grad_rel = |g_arm - g_ref| / |g_ref| (same batch, own weights)
  * train loss on the step's batch, held-out loss every --eval-every steps.
Step 0 also attributes the gradient error: bf16 park only (fp32 weights) and
bf16 weights only (fp32 park).
"""
import os, sys, json, time, argparse, contextlib

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = (4, 5, 6, 7)
ATTN = {6, 18}
D, NH, S, MB = 2048, 64, 4096, 8
PEAK_MUON_LR, PEAK_LR, WD, CLIP = 0.02, 3e-4, 0.01, 1.0
ARMS = {
    "ref": dict(w="fp32", m="fp32", park="fp32"),
    "lean": dict(w="sr", m="bf16", park="bf16"),
    "rne": dict(w="rne", m="bf16", park="bf16"),
    "lean_m8": dict(w="sr", m="int8", park="bf16"),
    # control: full fp32 like ref, only the micro-batch accumulation order is
    # reversed (grads differ at the fp32 rounding level, ~1e-7). Shows how far a
    # harmless perturbation drifts through Muon + ternary thresholds in N steps.
    "ctrl_mbrev": dict(w="fp32", m="fp32", park="fp32", mbrev=True),
}
HELDOUT = 99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--muon-lr", type=float, default=PEAK_MUON_LR)
    ap.add_argument("--steps", type=int, default=24)
    ap.add_argument("--train-batches", type=int, default=4)
    ap.add_argument("--eval-every", type=int, default=4)
    ap.add_argument("--arms", default="ref,lean,rne,lean_m8")
    a = ap.parse_args()
    arms = a.arms.split(",")
    assert arms[0] == "ref"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from bench_streamed_group_217 import host_mem
    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint
    import train_edge_conversational as T          # before any CUDA call (pyarrow, #217)
    out = open(a.out, "w", encoding="utf-8")

    def emit(row):
        out.write(json.dumps(row) + "\n"); out.flush()

    dev = torch.device("cuda")
    import hvn.core.deltanet_triton as DT
    import hvn.edge.fused_bias as FB
    import hvn.core.fused_norm_residual as FNR
    from hvn.edge.ternary import apply_ternary_to_model
    from hvn.optimizers import build_optimizer
    from hvn.core.deltanet import DeltaNetBlock
    from hvn.core.transformer_block import TransformerBlock, set_deterministic_attention
    DT.SCAN_KERNEL[0] = "t"
    FB.set_mode("fused")
    FB.AMAX_HINTS[0] = False
    FNR.ENABLED[0] = False
    set_deterministic_attention(True)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=False)
    from torch.utils import deterministic as _tdet
    _tdet.fill_uninitialized_memory = False
    hm0 = host_mem()

    def build(seed):                                 # == bench_streamed_group_217.build
        torch.manual_seed(seed)
        blocks = []
        for L in LAYERS:
            if L in ATTN:
                blocks.append(TransformerBlock(d_model=D, n_heads=NH, dropout=0.0, window=4096))
            else:
                blocks.append(DeltaNetBlock(d_model=D, n_heads=NH, dropout=0.0, conv_kernel=4,
                                            use_triton_scan=False, moe_experts=1, moe_top_k=2,
                                            use_megakernel_scan=False, ffn_activation="gelu"))
        g = nn.Module()
        g.blocks = nn.ModuleList(blocks)
        targets = [n for n, m in g.named_modules() if isinstance(m, nn.Linear)
                   and not any(p in n for p in ("beta_proj", "alpha_proj", "decoder"))]
        apply_ternary_to_model(g, target_names=targets)
        T.apply_fast_stack(g, triton_scan=True, megakernel_scan=True, tf32=True,
                           compile_model=False, verbose=False, fused_ternary=True,
                           ternary_step_cache=True, fused_rmsnorm=True, fp8_linear=True)
        g.train()
        for m in g.modules():
            for k, b in list(m._buffers.items()):
                if b is not None:
                    m._buffers[k] = b.to(dev)
        return g

    pos = torch.arange(S, device=dev).unsqueeze(0)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    def make_x(b):
        gen = torch.Generator(device=dev); gen.manual_seed(1000 + b)
        return torch.randn(MB, S, D, device=dev, generator=gen) * 0.02

    def clear_caches(g):
        for m in g.modules():
            for k in ("_scache", "_fscale", "_f8cache"):
                if getattr(m, k, None) is not None:
                    setattr(m, k, None)

    # ---------------------------------------------------------------- teacher targets
    t0 = time.perf_counter()
    teacher = build(1).to(dev)
    batches = list(range(a.train_batches)) + [HELDOUT]
    TGT = {}
    with torch.no_grad(), amp():
        for b in batches:
            X = make_x(b)
            ys = []
            for m in range(MB):
                h = X[m:m + 1]
                for blk in teacher.blocks:
                    h = blk(h, position_ids=pos)
                ys.append(h.float())
            TGT[b] = torch.cat(ys).bfloat16().cpu()
            del X, ys
    del teacher
    torch.cuda.empty_cache()
    print(f"[n218] teacher targets for {batches} in {time.perf_counter() - t0:.1f}s", flush=True)

    # ---------------------------------------------------------------- student
    g = build(0)
    layers = []
    for i, blk in enumerate(g.blocks):
        opt, desc = build_optimizer(blk, "muon", lr=PEAK_LR, weight_decay=WD, muon_lr=PEAK_MUON_LR)
        mid = {id(p) for grp in opt.param_groups if grp.get("use_muon", False) for p in grp["params"]}
        params = list(blk.parameters())
        assert {id(p) for grp in opt.param_groups for p in grp["params"]} == {id(p) for p in params}
        for grp in opt.param_groups:
            grp["lr"] = a.muon_lr if grp.get("use_muon", False) else PEAK_LR * a.muon_lr / PEAK_MUON_LR
        layers.append(dict(blk=blk, opt=opt, params=params,
                           is_muon=[id(p) in mid for p in params]))
    allp = [(li, k, p, lay["is_muon"][k]) for li, lay in enumerate(layers)
            for k, p in enumerate(lay["params"])]
    # Every arm starts from the SAME bf16-representable matrices (the one-time
    # conversion of an fp32 checkpoint is not what is being tested; the smoke run
    # showed it alone moves the gradient ~6% through ternary threshold flips, which
    # would hide the per-step effect). Non-matrix params stay fp32 everywhere.
    W0 = [(p.detach().bfloat16().float() if mu else p.detach().clone()) for _, _, p, mu in allp]
    n_muon = sum(p.numel() for _, _, p, mu in allp if mu)
    n_aux = sum(p.numel() for _, _, p, mu in allp if not mu)

    # ---------------------------------------------------------------- host-state codecs
    def q8(m):
        flat = m.reshape(-1)
        pad = (-flat.numel()) % 256
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        blk = flat.view(-1, 256)
        s = blk.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127.0
        q = torch.round(blk / s).clamp_(-127, 127).to(torch.int8)
        return (q.cpu(), s.cpu(), m.shape, m.numel())

    def dq8(t):
        q, s, shape, n = t
        return (q.to(dev).float() * s.to(dev)).reshape(-1)[:n].view(shape)

    def mom_store(kind, m):
        if kind == "fp32":
            return m.cpu()
        if kind == "bf16":
            return m.bfloat16().cpu()
        return q8(m)

    def mom_load(kind, t):
        return dq8(t) if kind == "int8" else t.to(dev).float()

    def sr_bf16(x, gen):
        bits = x.contiguous().view(torch.int32)
        r = torch.randint(0, 1 << 16, x.shape, device=x.device, dtype=torch.int32, generator=gen)
        return ((bits + r) & -65536).view(torch.float32).bfloat16()

    # quick unbiasedness check of the SR codec (would catch a sign / mask bug)
    gtest = torch.Generator(device=dev); gtest.manual_seed(7)
    xt = W0[0].reshape(-1)[:1_000_000].to(dev) * 1.000123
    acc = torch.zeros_like(xt)
    for _ in range(64):
        acc += sr_bf16(xt, gtest).float()
    ulp = (xt.bfloat16().float() - xt).abs().mean()
    sr_bias = float((acc / 64 - xt).mean() / xt.abs().mean())
    sr_mean_abs_err = float((acc / 64 - xt).abs().mean())
    rne_mean_abs_err = float((xt.bfloat16().float() - xt).abs().mean())
    print(f"[n218] SR check: mean bias {sr_bias:.2e} (rel), |mean64(SR)-x| {sr_mean_abs_err:.3e} "
          f"vs |RNE-x| {rne_mean_abs_err:.3e}", flush=True)
    del xt, acc

    # GPU cost of the casts on one layer's matrices (42M values), median of 10:
    # bf16 SR, bf16 RNE, int8 blockwise quantize + dequantize (GPU part only).
    mats = [p.detach().to(dev) for (li, k, p, mu) in allp if li == 0 and mu]
    def _time(fn):
        ts = []
        for _ in range(12):
            e0 = torch.cuda.Event(enable_timing=True); e1 = torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record(); e1.synchronize(); ts.append(e0.elapsed_time(e1))
        return sorted(ts[2:])[5]
    def _q8gpu(m):
        flat = m.reshape(-1); pad = (-flat.numel()) % 256
        if pad:
            flat = torch.cat([flat, flat.new_zeros(pad)])
        blk = flat.view(-1, 256); s_ = blk.abs().amax(1, keepdim=True).clamp_min(1e-30) / 127.0
        q = torch.round(blk / s_).clamp_(-127, 127).to(torch.int8)
        return (q.float() * s_).reshape(-1)
    cast_cost = {"layer_matrix_params": sum(m.numel() for m in mats),
                 "sr_ms": _time(lambda: [sr_bf16(m, gtest) for m in mats]),
                 "rne_ms": _time(lambda: [m.bfloat16() for m in mats]),
                 "int8_q_dq_ms": _time(lambda: [_q8gpu(m) for m in mats])}
    del mats
    print(f"[n218] cast cost per layer (GPU): {cast_cost}", flush=True)

    # ---------------------------------------------------------------- arm states
    state = {}
    for name in arms:
        cfg = ARMS[name]
        W = []
        for (li, k, p, mu), w0 in zip(allp, W0):
            W.append(w0.clone() if (cfg["w"] == "fp32" or not mu) else w0.bfloat16())
        M = {j: None for j, (_, _, _, mu) in enumerate(allp) if mu}
        state[name] = dict(cfg=cfg, W=W, M=M, A={}, adam_step=[0] * len(layers),
                           W0=W0 if name == "ref" else None,
                           gen=torch.Generator(device=dev).manual_seed(12345))
    hm_state = host_mem()
    emit({"kind": "setup", "muon_lr": a.muon_lr, "steps": a.steps, "arms": arms,
          "n_muon": n_muon, "n_aux": n_aux, "train_batches": a.train_batches,
          "sr_bias_rel": sr_bias, "sr_mean64_abs_err": sr_mean_abs_err,
          "rne_abs_err": rne_mean_abs_err, "cast_cost": cast_cost, "host_start": hm0, "host_state": hm_state,
          "optim_desc": desc})
    print(f"[n218] setup muon_lr {a.muon_lr} arms {arms} muon params {n_muon:,} aux {n_aux:,} "
          f"host ws {hm_state['ws']:.2f} GB avail {hm_state['sys_avail']:.2f}", flush=True)

    def load(name):
        st = state[name]
        for (li, k, p, mu), w in zip(allp, st["W"]):
            p.data = w.to(dev).float()
            p.grad = None
        clear_caches(g)

    def park(x, kind):
        return x.bfloat16().float() if kind == "bf16" else x.clone()

    @contextlib.contextmanager
    def recompute_ctx():
        old = DT._in_backward
        DT._in_backward = lambda: True
        try:
            yield
        finally:
            DT._in_backward = old

    def run(h):
        for blk in g.blocks:
            h = blk(h, position_ids=pos)
        return h

    def eval_loss(b):
        X = make_x(b); Tg = TGT[b].to(dev).float()
        tot = 0.0
        with torch.no_grad(), amp():
            for m in range(MB):
                y = run(X[m:m + 1]).float()
                tot += float((y - Tg[m:m + 1]).pow(2).mean()) / MB
        del X, Tg
        return tot

    def fwd_bwd(b, park_kind, check_same=False, mbrev=False):
        """Gradient as the streamed design computes it. Returns train loss and
        the per-layer flat grads (fp32, GPU, before clipping)."""
        X = make_x(b); Tg = TGT[b].to(dev).float()
        tot, same = 0.0, None
        for m in (reversed(range(MB)) if mbrev else range(MB)):
            x = X[m:m + 1]
            with torch.no_grad(), amp():
                y = run(x).float()
            diff = y - Tg[m:m + 1]
            tot += float(diff.pow(2).mean()) / MB
            dy = 2.0 * diff / (diff.numel() * MB)
            xp = park(x, park_kind).requires_grad_()
            h = xp
            with amp():
                for blk in g.blocks:
                    h = checkpoint(blk, h, pos, use_reentrant=False)
            if check_same and m == 0:
                same = float((h.detach().float() - y).abs().max())
            torch.autograd.backward(h, dy.to(h.dtype))
            del y, diff, dy, xp, h
        del X, Tg
        grads = [torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                            for p in lay["params"]]) for lay in layers]
        return tot, grads, same

    def opt_step(name):
        st, cfg = state[name], state[name]["cfg"]
        for li, lay in enumerate(layers):
            torch.nn.utils.clip_grad_norm_(lay["params"], CLIP)
            opt = lay["opt"]
            for (lj, k, p, mu) in [x for x in allp if x[0] == li]:
                j = jidx[(lj, k)]
                if mu:
                    if st["M"][j] is not None:
                        opt.state[p]["momentum_buffer"] = mom_load(cfg["m"], st["M"][j])
                else:
                    if j in st["A"]:
                        m1, m2 = st["A"][j]
                        opt.state[p].update(step=st["adam_step"][li], exp_avg=m1.to(dev),
                                            exp_avg_sq=m2.to(dev))
            opt.step()
            st["adam_step"][li] += 1
            for (lj, k, p, mu) in [x for x in allp if x[0] == li]:
                j = jidx[(lj, k)]
                if mu:
                    st["M"][j] = mom_store(cfg["m"], opt.state[p]["momentum_buffer"])
                else:
                    s_ = opt.state[p]
                    st["A"][j] = (s_["exp_avg"].cpu(), s_["exp_avg_sq"].cpu())
            opt.state.clear()

    jidx = {(li, k): j for j, (li, k, p, mu) in enumerate(allp)}

    def write_back(name):
        """Cast the optimizer's fp32 weights into the arm's master format.
        Returns this step's kept fraction and rounding noise (Muon params)."""
        st, cfg = state[name], state[name]["cfg"]
        dot = nint = nerr = 0.0
        ev0 = torch.cuda.Event(enable_timing=True); ev1 = torch.cuda.Event(enable_timing=True)
        cast_ms = 0.0
        for j, (li, k, p, mu) in enumerate(allp):
            wnew = p.data
            if not mu or cfg["w"] == "fp32":
                stored = wnew
            else:
                ev0.record()
                stored = sr_bf16(wnew, st["gen"]) if cfg["w"] == "sr" else wnew.bfloat16()
                ev1.record(); ev1.synchronize(); cast_ms += ev0.elapsed_time(ev1)
            if mu:
                wprev = st["W"][j].to(dev).float()
                dint = (wnew - wprev).double()
                dsto = (stored.float() - wprev).double()
                dot += float((dsto * dint).sum()); nint += float((dint * dint).sum())
                nerr += float(((dsto - dint) ** 2).sum())
                del wprev, dint, dsto
            st["W"][j] = stored.cpu()
        return {"kept": dot / nint if nint > 0 else None,
                "noise": (nerr / nint) ** 0.5 if nint > 0 else None,
                "upd_norm": nint ** 0.5, "cast_ms": cast_ms}

    def cum_vs_ref(name):
        ref = state["ref"]
        st = state[name]
        dot = nref = nerr = 0.0
        for j, (li, k, p, mu) in enumerate(allp):
            if not mu:
                continue
            w0r = ref["W0"][j].to(dev)
            ur = (ref["W"][j].to(dev) - w0r).double()
            if name == "ref":
                ua = ur
            else:                                            # all arms start at W0
                ua = (st["W"][j].to(dev).float() - w0r).double()
            dot += float((ua * ur).sum()); nref += float((ur * ur).sum())
            nerr += float(((ua - ur) ** 2).sum())
            del w0r, ur, ua
        return {"cum_kept": dot / nref if nref > 0 else None,
                "cum_err": (nerr / nref) ** 0.5 if nref > 0 else None}

    def grel(gs, gr):
        num = sum(float((x - y).double().pow(2).sum()) for x, y in zip(gs, gr))
        den = sum(float(y.double().pow(2).sum()) for y in gr)
        cos = sum(float((x.double() * y.double()).sum()) for x, y in zip(gs, gr)) / (
            den ** 0.5 * sum(float(x.double().pow(2).sum()) for x in gs) ** 0.5)
        per = [float((x - y).norm() / y.norm()) for x, y in zip(gs, gr)]
        return (num / den) ** 0.5, cos, per

    # ---------------------------------------------------------------- run
    ref_g0 = None
    for s in range(a.steps + 1):
        b = s % a.train_batches
        ref_grads = None
        for name in arms:
            st = state[name]
            t_arm = time.perf_counter()
            load(name)
            row = {"kind": "step", "step": s, "arm": name, "batch": b}
            if s % a.eval_every == 0 or s == a.steps:
                row["heldout"] = eval_loss(HELDOUT)
            if s == a.steps:                                 # final eval only
                emit(row)
                print(f"[n218] final {name} heldout {row['heldout']:.6f}", flush=True)
                continue
            loss, grads, same = fwd_bwd(b, st["cfg"]["park"], check_same=(s == 0 and name == "ref"),
                                        mbrev=st["cfg"].get("mbrev", False))
            row["train_loss"] = loss
            if same is not None:
                row["ref_recompute_vs_forward_maxabs"] = same
            if name == "ref":
                ref_grads = grads
                row["grad_norm"] = sum(float(x.double().pow(2).sum()) for x in grads) ** 0.5
                # natural scale: how far the ref gradient on batch 0 moves over
                # real training steps (a lean arm's grad_rel is read against it)
                if s == 0:
                    ref_g0 = [x.bfloat16() for x in grads]
                elif b == 0:
                    r_, c_, _ = grel(grads, [x.float() for x in ref_g0])
                    row["ref_drift_vs_step0"] = r_
            else:
                r_, c_, per = grel(grads, ref_grads)
                row.update(grad_rel=r_, grad_cos=c_, grad_rel_layers=per)
            del grads
            opt_step(name)
            row.update(write_back(name))
            row["arm_sec"] = time.perf_counter() - t_arm
            emit(row)
            st["_last"] = row
        if s == a.steps:
            break
        # step-0 attribution of the gradient error (no update)
        if s == 0:
            attr = {}
            for label, pkind in (("park_bf16_only", "bf16"), ("park_fp32_repeat", "fp32")):
                for (li, k, p, mu), w0 in zip(allp, state["ref"]["W0"]):
                    p.data = w0.to(dev).float()
                    p.grad = None
                clear_caches(g)
                _, gs, _ = fwd_bwd(b, pkind)
                r_, c_, per = grel(gs, ref_grads)
                attr[label] = {"grad_rel": r_, "grad_cos": c_, "per_layer": per}
                del gs
                for _, _, p, _ in allp:
                    p.grad = None
            emit({"kind": "attr", "step": 0, **attr})
            print(f"[n218] step-0 grad attribution {json.dumps(attr)}", flush=True)
        del ref_grads
        for name in arms:
            c = cum_vs_ref(name)
            r = state[name]["_last"]
            emit({"kind": "cum", "step": s, "arm": name, **c})
            print(f"[n218] s{s:02d} {name:8s} loss {r['train_loss']:.6f} "
                  f"{'held ' + format(r['heldout'], '.6f') if 'heldout' in r else '':16s} "
                  f"kept {r['kept']:.4f} noise {r['noise']:.3f} "
                  f"grad_rel {r.get('grad_rel', 0):.2e} cum_err {c['cum_err']:.4f} "
                  f"cum_kept {c['cum_kept']:.4f} cast {r['cast_ms']:.1f}ms "
                  f"{'refdrift ' + format(r['ref_drift_vs_step0'], '.3e') + ' ' if 'ref_drift_vs_step0' in r else ''}"
                  f"t {r['arm_sec']:.1f}s", flush=True)
    emit({"kind": "done", "host_end": host_mem(),
          "peak_alloc": torch.cuda.max_memory_allocated() / 1e9,
          "peak_res": torch.cuda.max_memory_reserved() / 1e9})
    print(f"[n218] done peak alloc {torch.cuda.max_memory_allocated() / 1e9:.2f} GB "
          f"host ws {host_mem()['ws']:.2f}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
