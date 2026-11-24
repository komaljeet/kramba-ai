"""#217: one 4-layer group of streamed full backprop at 1B, measured on the GPU.

Author: Rambarun Komaljeet

Checks the #216 paper estimate (MR !271) with the real thing, for ONE group:
the production 1B layers 4-7 (DeltaNet, DeltaNet, attention, DeltaNet; d 2048,
64 heads, window 4096), built with the production ``--fast --fp8-linear`` stack
(ternary + fp8 GEMMs, Triton "t" scan + megakernel, fused RMSNorm / bias, TF32,
bf16 autocast), 8 micro-batches x 4096 tokens, run the way #216's design "L2"
would run it. No trainer code is changed; this file only drives the real
modules and the real Muon.

Per group step (layer-major):
  forward     the group input (8 mb, fp32) is copied to host RAM (the parked
              "segment input"); per layer, its fp32 master streams host->GPU and
              runs all 8 micro-batches under no_grad.
  re-forward  the segment input streams back; layers 4-6 re-run (no_grad) to
              rebuild the inputs of layers 5-7 for all 8 mb (kept on the GPU).
  backward    per layer from the top: master streams in again, per micro-batch
              recompute + backward (grad of the weights accumulates in fp32 on the
              GPU), then the layer's Muon momentum / AdamW moments stream in, the
              layer's gradient is clipped (per-layer clip, see below), the
              production Muon step runs on the GPU, and the updated fp32 master +
              momentum + moments are copied back to host RAM.
Host RAM holds every fp32 master, Muon momentum and AdamW moment (the "exact"
full-precision variant in #216). The GPU holds at most two layers' weights.

Copies run on two side streams (H2D, D2H) with events; ``--mode sync`` puts every
copy on the compute stream instead (the overlap falsifier: the step must grow by
about bytes / rate), ``--mode pageable`` uses plain pageable host memory instead
of page-locked (cudaHostRegister) memory.

Choices made here that the real build would also have to make (stated, not tuned):
  * grad clip: per layer at max-norm 1.0 (a global norm cannot exist before the
    first layer is updated, #216 item 5). Costs the same as global clipping.
  * the recompute forward runs the scan's fused checkpoint pass, as the
    trainer's checkpoint recompute does (DT._in_backward forced True there).
  * deterministic mode as in #213's measurements (the per-layer times the
    estimate used came from --deterministic runs).
  * the input is a dummy embedding lookup (std 0.02 table, random tokens); the
    upstream gradient is seeded noise (std 1e-5). The table and a same-size grad
    tensor stay on the GPU, standing in for the tied embedding + its gradient
    that full backprop keeps resident (0.82 GB).

``--check N`` (falsifier) runs N steps of this path against a plain trainer-style
reference on the SAME group (all weights resident, micro-batch-major, the
trainer's torch.utils.checkpoint per layer, one production Muon over the group)
from the same weights, inputs and upstream gradients, and compares, step by
step: group output, input gradient, every layer's weight gradient, and the
updated weights relative to the size of the update (an absolute weight compare
would pass on stale weights because lr is small). It also checks that each
layer's host master hash changes every step and that the GPU copy equals the host
copy after write-back. ``--break skip_d2h`` / ``--break skip_h2d`` are the
must-fail partners (one layer's write-back skipped / one layer's weights not
copied in); the check must FAIL under both.
"""
import os, sys, json, time, argparse, hashlib, contextlib, ctypes
from ctypes import wintypes

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LAYERS = (4, 5, 6, 7)
ATTN = {6, 18}
D, NH, S, MB, VOCAB = 2048, 64, 4096, 8, 50257
MUON_LR, LR, WD, CLIP = 0.02, 3e-4, 0.01, 1.0


# --------------------------------------------------------------------------- host RAM
class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (n, ctypes.c_size_t) for n in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
            "PrivateUsage")]


class _MSE(ctypes.Structure):
    _fields_ = [("dwLength", wintypes.DWORD), ("dwMemoryLoad", wintypes.DWORD)] + [
        (n, ctypes.c_ulonglong) for n in (
            "ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
            "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]


def host_mem():
    """Process working set / peak / private commit and system available RAM, GB."""
    pmc = _PMC(); pmc.cb = ctypes.sizeof(_PMC)
    k32 = ctypes.windll.kernel32
    k32.GetCurrentProcess.restype = wintypes.HANDLE
    k32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
    assert k32.K32GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    ms = _MSE(); ms.dwLength = ctypes.sizeof(_MSE)
    k32.GlobalMemoryStatusEx(ctypes.byref(ms))
    g = 1e9
    return {"ws": pmc.WorkingSetSize / g, "peak_ws": pmc.PeakWorkingSetSize / g,
            "private": pmc.PrivateUsage / g, "sys_avail": ms.ullAvailPhys / g,
            "sys_total": ms.ullTotalPhys / g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="jsonl output")
    ap.add_argument("--mode", choices=["overlap", "sync", "pageable"], default="overlap")
    ap.add_argument("--steps", type=int, default=7)
    ap.add_argument("--warmup", type=int, default=2, help="leading steps left out of the summary")
    ap.add_argument("--check", type=int, default=0, help="falsifier: N steps vs the reference")
    ap.add_argument("--break", dest="brk", choices=["none", "skip_d2h", "skip_h2d"], default="none")
    ap.add_argument("--copybench", action="store_true", help="only the copy-rate microbenchmark")
    ap.add_argument("--no-det", action="store_true")
    a = ap.parse_args()
    det = not a.no_det
    if det:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    hm_start = host_mem()
    sys.path.insert(0, REPO)
    os.chdir(REPO)
    import torch
    import torch.nn as nn
    from torch.utils.checkpoint import checkpoint
    # Before ANY CUDA call: the trainer imports pandas -> pyarrow, and loading
    # pyarrow after CUDA is initialised crashes this venv with a heap corruption
    # (0xc0000374, reproduced with `torch.zeros(1, device="cuda"); import pyarrow`).
    import train_edge_conversational as T
    out = open(a.out, "w", encoding="utf-8")

    def emit(row):
        out.write(json.dumps(row) + "\n"); out.flush()

    dev = torch.device("cuda")
    cudart = torch.cuda.cudart()
    REG = []

    def pin(t):
        """Page-lock an existing host tensor in place (exact size, no power-of-2
        rounding as torch's pinned caching allocator would do)."""
        r = cudart.cudaHostRegister(t.data_ptr(), t.numel() * t.element_size(), 0)
        assert int(r) == 0 or str(r).endswith("success"), f"cudaHostRegister failed: {r}"
        assert t.is_pinned()
        REG.append(t)
        return t

    def ev():
        return torch.cuda.Event(enable_timing=True)

    if a.copybench:
        copybench(torch, pin, ev, emit)
        out.close()
        return

    # ------------------------------------------------------------------ production stack
    import hvn.core.deltanet_triton as DT
    import hvn.edge.fused_bias as FB
    import hvn.core.fused_norm_residual as FNR
    from hvn.edge.ternary import apply_ternary_to_model
    from hvn.optimizers import build_optimizer
    from hvn.core.deltanet import DeltaNetBlock
    from hvn.core.transformer_block import TransformerBlock, set_deterministic_attention
    DT.SCAN_KERNEL[0] = "t"           # trainer default since #210
    FB.set_mode("fused")              # --fp8-bias default (#207)
    FB.AMAX_HINTS[0] = False          # --fused-bias-amax default off
    FNR.ENABLED[0] = False            # --fused-norm-residual default off
    if det:                           # as train_edge_conversational.main --deterministic
        set_deterministic_attention(True)
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True, warn_only=False)
        from torch.utils import deterministic as _tdet
        _tdet.fill_uninitialized_memory = False
    else:
        torch.backends.cudnn.benchmark = True

    def build(seed):
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
        # build_model_and_config: ternary on every Linear except the gates / decoder
        targets = [n for n, m in g.named_modules() if isinstance(m, nn.Linear)
                   and not any(p in n for p in ("beta_proj", "alpha_proj", "decoder"))]
        apply_ternary_to_model(g, target_names=targets)
        # --fast --fp8-linear as main() resolves them for the 1B PROD_ARGV
        T.apply_fast_stack(g, triton_scan=True, megakernel_scan=True, tf32=True,
                           compile_model=False, verbose=False, fused_ternary=True,
                           ternary_step_cache=True, fused_rmsnorm=True, fp8_linear=True)
        g.train()
        nbuf = 0
        for m in g.modules():
            for k, b in list(m._buffers.items()):
                if b is not None:
                    m._buffers[k] = b.to(dev); nbuf += b.numel()
        return g, len(targets), nbuf

    g, n_tern, nbuf = build(0)
    hm_built = host_mem()
    pos = torch.arange(S, device=dev).unsqueeze(0)
    amp = lambda: torch.autocast("cuda", dtype=torch.bfloat16)

    # Dummy tied embedding + its gradient, resident as in full backprop (#216 VRAM table).
    emb = torch.empty(VOCAB, D, device=dev).normal_(0.0, 0.02)
    emb_grad = torch.zeros_like(emb)

    def make_x0(step):
        gen = torch.Generator(device=dev); gen.manual_seed(1000 + step)
        tok = torch.randint(0, VOCAB, (MB, S), device=dev, generator=gen)
        return emb[tok]                                # [MB, S, D] fp32

    def make_r(step):
        gen = torch.Generator(device=dev); gen.manual_seed(2000 + step)
        r = torch.randn(MB, S, D, device=dev, generator=gen) * 1e-5
        return [r[m:m + 1] for m in range(MB)]

    @contextlib.contextmanager
    def recompute_ctx():
        """The trainer's recompute runs inside backward(), where the 't' scan
        writes its checkpoints during the forward (no separate pass)."""
        old = DT._in_backward
        DT._in_backward = lambda: True
        try:
            yield
        finally:
            DT._in_backward = old

    # ------------------------------------------------------------------ streamer
    class Streamer:
        def __init__(self, g, mode, brk):
            self.mode, self.brk = mode, brk
            self.cs = torch.cuda.current_stream()
            self.h2d_s = self.cs if mode == "sync" else torch.cuda.Stream()
            self.d2h_s = self.cs if mode == "sync" else torch.cuda.Stream()
            self.layers = []
            for i, blk in enumerate(g.blocks):
                params = list(blk.parameters())
                opt, desc = build_optimizer(blk, "muon", lr=LR, weight_decay=WD, muon_lr=MUON_LR)
                mid = {id(p) for p in opt.param_groups[0]["params"]}
                allid = {id(p) for grp in opt.param_groups for p in grp["params"]}
                assert allid == {id(p) for p in params}, "optimizer does not cover the layer"
                muon = [p for p in params if id(p) in mid]
                aux = [p for p in params if id(p) not in mid]
                n = sum(p.numel() for p in params)
                nm = sum(p.numel() for p in muon)
                na = sum(p.numel() for p in aux)
                master = torch.empty(n)
                off, slots = 0, []
                for p in params:
                    k = p.numel()
                    master[off:off + k].copy_(p.data.reshape(-1))
                    slots.append((off, k, tuple(p.shape)))
                    off += k
                mom = torch.zeros(nm)
                adam = torch.zeros(2 * na)
                lay = dict(i=i, L=LAYERS[i], blk=blk, params=params, muon=muon, aux=aux,
                           opt=opt, desc=desc, n=n, nm=nm, na=na, master=master, mom=mom,
                           adam=adam, slots=slots, adam_step=0, d2h_done=None)
                for p, v in zip(params, self.views(master, slots)):
                    p.data = v                             # evicted = points at the host master
                self.layers.append(lay)
            self.seg_host = torch.empty(MB * S * D)
            self.pinned = mode != "pageable"
            self.host_bytes = 4 * (self.seg_host.numel() + sum(
                l["master"].numel() + l["mom"].numel() + l["adam"].numel() for l in self.layers))
            if self.pinned:
                for l in self.layers:
                    for t in (l["master"], l["mom"], l["adam"]):
                        pin(t)
                pin(self.seg_host)
            self.copies = []

        @staticmethod
        def views(flat, slots):
            return [flat[o:o + k].view(sh) for o, k, sh in slots]

        def mom_views(self, lay, flat):
            out, o = [], 0
            for p in lay["muon"]:
                out.append(flat[o:o + p.numel()].view(p.shape)); o += p.numel()
            return out

        def adam_views(self, lay, flat):
            na, out, o = lay["na"], [], 0
            for p in lay["aux"]:
                k = p.numel()
                out.append((flat[o:o + k].view(p.shape), flat[na + o:na + o + k].view(p.shape)))
                o += k
            return out

        # copies -------------------------------------------------------------
        def h2d(self, host, tag, wait=None, zero=False):
            s = self.h2d_s
            with torch.cuda.stream(s):
                if wait is not None and s is not self.cs:
                    s.wait_event(wait)
                e0 = ev(); e0.record(s)
                dst = torch.empty(host.shape, dtype=host.dtype, device=dev)
                if zero:
                    dst.zero_()                      # must-fail partner: copy skipped
                else:
                    dst.copy_(host, non_blocking=True)
                e1 = ev(); e1.record(s)
            self.copies.append(("h2d", tag, host.numel() * host.element_size(), e0, e1))
            return dst, e1

        def use(self, t, e):
            if self.h2d_s is not self.cs:
                self.cs.wait_event(e)
                t.record_stream(self.cs)

        def d2h(self, gpu, host, tag, after):
            s = self.d2h_s
            with torch.cuda.stream(s):
                if s is not self.cs:
                    s.wait_event(after)
                e0 = ev(); e0.record(s)
                host.copy_(gpu, non_blocking=True)
                e1 = ev(); e1.record(s)
            if s is not self.cs:
                gpu.record_stream(s)
            self.copies.append(("d2h", tag, gpu.numel() * gpu.element_size(), e0, e1))
            return e1

        def attach(self, lay, dst):
            for p, v in zip(lay["params"], self.views(dst, lay["slots"])):
                p.data = v

        def evict(self, lay):
            for p, v in zip(lay["params"], self.views(lay["master"], lay["slots"])):
                p.data = v
                p.grad = None
            for m in lay["blk"].modules():
                for k in ("_scache", "_fscale", "_f8cache"):
                    if getattr(m, k, None) is not None:
                        setattr(m, k, None)

        # one group step -------------------------------------------------------
        def step(self, sidx, chk=None):
            cs, lays, n = self.cs, self.layers, len(self.layers)
            self.copies = []
            EV = []

            def mark(label):
                e = ev(); e.record(cs); EV.append((label, e)); return e

            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            hm0 = host_mem()
            X0 = make_x0(sidx)
            torch.cuda.synchronize()
            t_wall = time.perf_counter()
            t0 = mark("fwd.park")
            seg_done = self.d2h(X0.reshape(-1), self.seg_host, "seg.park", after=t0)
            x = [X0[m:m + 1] for m in range(MB)]
            del X0
            hashes0 = ([hashlib.sha256(l["master"].numpy().tobytes()).hexdigest() for l in lays]
                       if chk is not None else None)
            done = {}

            # ---- forward: stream each layer in, 8 micro-batches, no_grad
            pend = {0: self.h2d(lays[0]["master"], "w.fwd", wait=lays[0]["d2h_done"])}
            for i, lay in enumerate(lays):
                if i + 1 < n:
                    if i - 1 in done:
                        done[i - 1].synchronize()    # bound run-ahead: <= 2 layers resident
                    pend[i + 1] = self.h2d(lays[i + 1]["master"], "w.fwd",
                                           wait=lays[i + 1]["d2h_done"])
                dst, e = pend.pop(i)
                mark(f"fwd.stall.L{lay['L']}"); self.use(dst, e); self.attach(lay, dst)
                mark(f"fwd.L{lay['L']}")
                with torch.no_grad(), amp():
                    x = [lay["blk"](xm, position_ids=pos) for xm in x]
                done[i] = cs.record_event()
                self.evict(lay); del dst
            if chk is not None:
                chk["out"] = torch.cat(x).float().cpu()
            del x
            fwd_end = ev(); fwd_end.record(cs)

            # ---- re-forward: segment input back, layers 0..n-2 again (no_grad)
            segin, e_seg = self.h2d(self.seg_host, "seg.in", wait=seg_done)
            pend = {0: self.h2d(lays[0]["master"], "w.refwd")}
            mark("refwd.stall.seg"); self.use(segin, e_seg)
            inputs = {0: [segin.view(MB, S, D)[m:m + 1] for m in range(MB)]}
            pend_b = None
            done = {}
            for i in range(n - 1):
                lay = lays[i]
                if i - 1 in done:
                    done[i - 1].synchronize()
                if i + 1 < n - 1:
                    pend[i + 1] = self.h2d(lays[i + 1]["master"], "w.refwd")
                else:
                    pend_b = self.h2d(lays[n - 1]["master"], "w.bwd")
                dst, e = pend.pop(i)
                mark(f"refwd.stall.L{lay['L']}"); self.use(dst, e); self.attach(lay, dst)
                mark(f"refwd.L{lay['L']}")
                with torch.no_grad(), amp():
                    inputs[i + 1] = [lay["blk"](xm, position_ids=pos) for xm in inputs[i]]
                done[i] = cs.record_event()
                self.evict(lay); del dst
            del segin

            # ---- backward: per layer from the top, then its optimizer step
            G = make_r(sidx)
            pend_w = {n - 1: pend_b}
            prev_done = None
            for i in reversed(range(n)):
                lay = lays[i]
                mom_dst, e_m = self.h2d(lay["mom"], "mom.in")
                adam_dst, e_a = self.h2d(lay["adam"], "adam.in")
                if i - 1 >= 0:
                    if prev_done is not None:
                        prev_done.synchronize()
                    zero = self.brk == "skip_h2d" and sidx == 1 and lays[i - 1]["L"] == 6
                    pend_w[i - 1] = self.h2d(lays[i - 1]["master"], "w.bwd", zero=zero)
                dst, e = pend_w.pop(i)
                mark(f"bwd.stall.L{lay['L']}"); self.use(dst, e); self.attach(lay, dst)
                mark(f"bwd.L{lay['L']}")
                newG = []
                with recompute_ctx():
                    for m in range(MB):
                        xi = inputs[i][m].detach().requires_grad_()
                        with amp():
                            y = lay["blk"](xi, position_ids=pos)
                        torch.autograd.backward(y, G[m])
                        newG.append(xi.grad)
                        del y, xi
                G = newG
                inputs.pop(i)
                mark(f"optim.stall.L{lay['L']}")
                self.use(mom_dst, e_m); self.use(adam_dst, e_a)
                mark(f"optim.L{lay['L']}")
                if chk is not None:
                    gs = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                    for p in lay["params"]]).cpu()
                    chk.setdefault("grads", {})[lay["L"]] = gs
                torch.nn.utils.clip_grad_norm_(lay["params"], CLIP)
                opt = lay["opt"]
                for p, v in zip(lay["muon"], self.mom_views(lay, mom_dst)):
                    opt.state[p]["momentum_buffer"] = v
                for p, (m1, m2) in zip(lay["aux"], self.adam_views(lay, adam_dst)):
                    st = opt.state[p]
                    st["step"], st["exp_avg"], st["exp_avg_sq"] = lay["adam_step"], m1, m2
                opt.step()
                lay["adam_step"] += 1
                opt.state.clear()
                e_opt = cs.record_event()
                if not (self.brk == "skip_d2h" and sidx == 0 and lay["L"] == 5):
                    self.d2h(dst, lay["master"], "w.out", after=e_opt)
                self.d2h(mom_dst, lay["mom"], "mom.out", after=e_opt)
                lay["d2h_done"] = self.d2h(adam_dst, lay["adam"], "adam.out", after=e_opt)
                if chk is not None:
                    lay["d2h_done"].synchronize()
                    chk.setdefault("gpu_eq_host", {})[lay["L"]] = bool(
                        torch.equal(dst.cpu(), lay["master"]))
                self.evict(lay)
                del dst, mom_dst, adam_dst
                prev_done = cs.record_event()
            if chk is not None:
                chk["dx"] = torch.cat(G).float().cpu()
            del G
            comp_end = mark("tail")
            if self.d2h_s is not self.cs:
                cs.wait_stream(self.d2h_s)
            end = mark("end")
            torch.cuda.synchronize()
            wall = time.perf_counter() - t_wall
            if chk is not None:
                chk["hash_changed"] = {l["L"]: hashlib.sha256(l["master"].numpy().tobytes()).hexdigest() != h0
                                       for l, h0 in zip(lays, hashes0)}
            return self.report(sidx, EV, t0, fwd_end, comp_end, end, wall, hm0)

        def report(self, sidx, EV, t0, fwd_end, comp_end, end, wall, hm0):
            T0 = lambda e: t0.elapsed_time(e) / 1000.0
            phases, cats, comp_iv = {}, {}, []
            for (lab, e0), (_, e1) in zip(EV[:-1], EV[1:]):
                dt = e0.elapsed_time(e1) / 1000.0
                phases[lab] = phases.get(lab, 0.0) + dt
                cat = lab.rsplit(".L", 1)[0] if ".L" in lab else lab
                cats[cat] = cats.get(cat, 0.0) + dt
                if "stall" not in lab and lab not in ("fwd.park", "tail", "end"):
                    comp_iv.append((T0(e0), T0(e1)))
            ctag, h2d_iv, d2h_iv = {}, [], []
            for d, tag, nb, e0, e1 in self.copies:
                a_, b_ = T0(e0), T0(e1)
                c = ctag.setdefault(f"{d}.{tag}", {"bytes": 0, "sec": 0.0, "n": 0})
                c["bytes"] += nb; c["sec"] += b_ - a_; c["n"] += 1
                (h2d_iv if d == "h2d" else d2h_iv).append((a_, b_))
            for c in ctag.values():
                c["GBps"] = c["bytes"] / 1e9 / c["sec"] if c["sec"] > 0 else None
            u_h, u_d, u_c = union(h2d_iv), union(d2h_iv), union(comp_iv)
            hb = sum(c["bytes"] for k, c in ctag.items() if k.startswith("h2d"))
            db = sum(c["bytes"] for k, c in ctag.items() if k.startswith("d2h"))
            row = {
                "step": sidx, "mode": self.mode, "wall": wall,
                "gpu_total": T0(end), "compute_end": T0(comp_end), "fwd_end": T0(fwd_end),
                "phases": phases, "cats": cats, "copies": ctag,
                "h2d_bytes": hb, "d2h_bytes": db,
                "h2d_busy": total(u_h), "d2h_busy": total(u_d), "compute_busy": total(u_c),
                "h2d_d2h_overlap": total(inter(u_h, u_d)),
                "h2d_under_compute": total(inter(u_h, u_c)),
                "d2h_under_compute": total(inter(u_d, u_c)),
                "peak_alloc": torch.cuda.max_memory_allocated() / 1e9,
                "peak_res": torch.cuda.max_memory_reserved() / 1e9,
                "total_mem": torch.cuda.get_device_properties(0).total_memory / 1e9,
                "host_before": hm0, "host_after": host_mem(),
            }
            return row

    # ------------------------------------------------------------------ reference
    def ref_step(ref, ropt, sidx):
        ref.to(dev)
        for st in ropt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.to(dev)
        X0, R = make_x0(sidx), make_r(sidx)
        outs, dxs = [], []
        for m in range(MB):
            x = X0[m:m + 1].clone().requires_grad_()
            h = x
            with amp():
                for blk in ref.blocks:
                    h = checkpoint(blk, h, pos, use_reentrant=False)
            outs.append(h.detach().float().cpu())
            torch.autograd.backward(h, R[m])
            dxs.append(x.grad.float().cpu())
            del h, x
        grads = {}
        for L, blk in zip(LAYERS, ref.blocks):
            ps = list(blk.parameters())
            grads[L] = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1)
                                  for p in ps]).cpu()
            torch.nn.utils.clip_grad_norm_(ps, CLIP)
        ropt.step()
        ropt.zero_grad(set_to_none=True)
        ref.cpu()
        for st in ropt.state.values():
            for k, v in list(st.items()):
                if torch.is_tensor(v):
                    st[k] = v.cpu()
        del X0, R
        torch.cuda.empty_cache()
        weights = {L: torch.cat([p.detach().reshape(-1) for p in blk.parameters()]).clone()
                   for L, blk in zip(LAYERS, ref.blocks)}
        return {"out": torch.cat(outs), "dx": torch.cat(dxs), "grads": grads, "weights": weights}

    def rel(a_, b_):
        den = float(b_.abs().max())
        return float((a_ - b_).abs().max()) / den if den > 0 else float("inf")

    # ------------------------------------------------------------------ run
    params_total = sum(p.numel() for p in g.parameters())
    if a.check:
        ref, _, _ = build(0)
        ref.load_state_dict(g.state_dict())
        ref.cpu()
        ropt, _ = build_optimizer(ref, "muon", lr=LR, weight_decay=WD, muon_lr=MUON_LR)
    st = Streamer(g, a.mode, a.brk)
    hm_state = host_mem()
    torch.cuda.synchronize()
    info = {"kind": "setup", "mode": a.mode, "check": a.check, "break": a.brk, "det": det,
            "params_group": params_total,
            "params_per_layer": {l["L"]: l["n"] for l in st.layers},
            "muon_per_layer": {l["L"]: l["nm"] for l in st.layers},
            "aux_per_layer": {l["L"]: l["na"] for l in st.layers},
            "optim_desc": st.layers[0]["desc"], "ternary_linears": n_tern, "buffers": nbuf,
            "host_state_bytes": st.host_bytes, "pinned": st.pinned,
            "host_start": hm_start, "host_built": hm_built, "host_state": hm_state,
            "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0)}
    emit(info)
    print(f"[s217] setup {json.dumps(info)}", flush=True)

    if a.check:
        ok_all, results = True, []
        for s in range(a.check):
            prev = {l["L"]: l["master"].clone() for l in st.layers}
            r = ref_step(ref, ropt, s)
            chk = {}
            row = st.step(s, chk=chk)
            res = {"step": s,
                   "out_rel": rel(chk["out"], r["out"]),
                   "out_identical": bool(torch.equal(chk["out"], r["out"])),
                   "dx_rel": rel(chk["dx"], r["dx"]),
                   "grad_rel": {L: rel(chk["grads"][L], r["grads"][L]) for L in LAYERS},
                   "grad_identical": {L: bool(torch.equal(chk["grads"][L], r["grads"][L])) for L in LAYERS},
                   "upd_rel": {L: float((l["master"] - r["weights"][L]).abs().max())
                               / max(float((r["weights"][L] - prev[L]).abs().max()), 1e-30)
                               for L, l in zip(LAYERS, st.layers)},
                   "w_identical": {L: bool(torch.equal(l["master"], r["weights"][L]))
                                   for L, l in zip(LAYERS, st.layers)},
                   "upd_size": {L: float((r["weights"][L] - prev[L]).abs().max()) for L in LAYERS},
                   "gpu_eq_host": chk["gpu_eq_host"], "hash_changed": chk["hash_changed"]}
            ok = (res["out_rel"] <= 1e-2 and res["dx_rel"] <= 1e-2
                  and all(v <= 1e-2 for v in res["grad_rel"].values())
                  and all(v <= 0.1 for v in res["upd_rel"].values())
                  and all(res["gpu_eq_host"].values()) and all(res["hash_changed"].values()))
            res["pass"] = ok
            ok_all &= ok
            results.append(res)
            emit({"kind": "check", **res, "gpu_total": row["gpu_total"]})
            print(f"[s217] check step {s}: {'PASS' if ok else 'FAIL'} {json.dumps(res)}", flush=True)
            del r, chk, prev
        import hvn.core.deltanet_triton as DT2
        emit({"kind": "check_done", "pass": ok_all, "break": a.brk,
              "scan_t_stats": dict(DT2.SCAN_T_STATS)})
        print(f"[s217] CHECK {'PASS' if ok_all else 'FAIL'} (break={a.brk}) "
              f"scan {dict(DT2.SCAN_T_STATS)}", flush=True)
        out.close()
        sys.stdout.flush()
        os._exit(0 if ok_all else 2)

    rows = []
    for s in range(a.steps):
        row = st.step(s)
        rows.append(row)
        emit({"kind": "step", **row})
        c = row["cats"]
        print(f"[s217] {a.mode} step {s} wall {row['wall']:.3f}s gpu {row['gpu_total']:.3f}s "
              f"fwd {c.get('fwd', 0):.3f} refwd {c.get('refwd', 0):.3f} bwd {c.get('bwd', 0):.3f} "
              f"optim {c.get('optim', 0):.3f} stalls "
              f"{sum(v for k, v in c.items() if 'stall' in k):.3f} park {c.get('fwd.park', 0):.3f} "
              f"tail {c.get('tail', 0):.3f} | H2D {row['h2d_bytes'] / 1e9:.2f}GB busy "
              f"{row['h2d_busy']:.3f}s D2H {row['d2h_bytes'] / 1e9:.2f}GB busy {row['d2h_busy']:.3f}s "
              f"dup {row['h2d_d2h_overlap']:.3f}s | peak alloc {row['peak_alloc']:.2f} res "
              f"{row['peak_res']:.2f} GB | ws {row['host_after']['ws']:.2f} avail "
              f"{row['host_after']['sys_avail']:.2f} GB", flush=True)
    import hvn.core.deltanet_triton as DT2
    emit({"kind": "done", "scan_t_stats": dict(DT2.SCAN_T_STATS), "host_end": host_mem()})
    print(f"[s217] done scan {dict(DT2.SCAN_T_STATS)}", flush=True)
    out.close()


# --------------------------------------------------------------------------- intervals
def union(iv):
    out = []
    for a_, b_ in sorted(iv):
        if out and a_ <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b_))
        else:
            out.append((a_, b_))
    return out


def inter(u1, u2):
    out, i, j = [], 0, 0
    while i < len(u1) and j < len(u2):
        a_, b_ = max(u1[i][0], u2[j][0]), min(u1[i][1], u2[j][1])
        if a_ < b_:
            out.append((a_, b_))
        if u1[i][1] < u2[j][1]:
            i += 1
        else:
            j += 1
    return out


def total(iv):
    return sum(b_ - a_ for a_, b_ in iv)


# --------------------------------------------------------------------------- copy rates
def copybench(torch, pin, ev, emit):
    """H2D / D2H GB/s, pinned vs pageable, one direction and both at once, and with
    a GEMM load on the compute stream. One layer's fp32 master (41.9M floats)."""
    dev = torch.device("cuda")
    n = 41_943_040                                        # 2048*2048*10 floats = 168 MB
    nb = n * 4
    hp_a, hp_b = pin(torch.empty(n)), pin(torch.empty(n))
    hq_a, hq_b = torch.empty(n), torch.empty(n)
    hp_a.normal_(); hq_a.normal_()
    g_a, g_b = torch.empty(n, device=dev).normal_(), torch.empty(n, device=dev).normal_()
    s1, s2 = torch.cuda.Stream(), torch.cuda.Stream()
    reps = 10

    def timed(fn, label):
        res = []
        for r in range(reps + 2):
            torch.cuda.synchronize()
            t = time.perf_counter()
            evs = fn()
            torch.cuda.synchronize()
            wall = time.perf_counter() - t
            if r >= 2:
                res.append({"wall": wall, "gpu": [e0.elapsed_time(e1) / 1000 for e0, e1 in evs]})
        walls = sorted(x["wall"] for x in res)
        gpus = [sorted(x["gpu"][k] for x in res) for k in range(len(res[0]["gpu"]))]
        row = {"kind": "copy", "label": label, "bytes_each": nb, "reps": reps,
               "wall_med": walls[len(walls) // 2], "wall_min": walls[0], "wall_max": walls[-1],
               "gpu_med": [gg[len(gg) // 2] for gg in gpus]}
        row["GBps_each_gpu"] = [nb / 1e9 / t for t in row["gpu_med"]]
        row["GBps_total_wall"] = len(row["gpu_med"]) * nb / 1e9 / row["wall_med"]
        emit(row)
        print(f"[s217] copy {label}: wall {row['wall_med'] * 1000:.1f} ms "
              f"({row['GBps_total_wall']:.2f} GB/s total), per-copy GPU GB/s "
              f"{[round(x, 2) for x in row['GBps_each_gpu']]}", flush=True)
        return row

    def one(dst, src, s):
        def f():
            with torch.cuda.stream(s):
                e0 = ev(); e0.record(s); dst.copy_(src, non_blocking=True); e1 = ev(); e1.record(s)
            return [(e0, e1)]
        return f

    def both(h_src, d_dst):
        def f():
            with torch.cuda.stream(s1):
                a0 = ev(); a0.record(s1); g_a.copy_(h_src, non_blocking=True); a1 = ev(); a1.record(s1)
            with torch.cuda.stream(s2):
                b0 = ev(); b0.record(s2); d_dst.copy_(g_b, non_blocking=True); b1 = ev(); b1.record(s2)
            return [(a0, a1), (b0, b1)]
        return f

    timed(one(g_a, hp_a, s1), "h2d_pinned")
    timed(one(g_a, hq_a, s1), "h2d_pageable")
    timed(one(hp_b, g_b, s2), "d2h_pinned")
    timed(one(hq_b, g_b, s2), "d2h_pageable")
    timed(both(hp_a, hp_b), "duplex_pinned")
    timed(both(hq_a, hq_b), "duplex_pageable")

    # copies under a GEMM load on the compute stream (bf16 4096^3 GEMMs)
    A = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    B = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    cs = torch.cuda.current_stream()

    def gemms(k):
        e0 = ev(); e0.record(cs)
        for _ in range(k):
            torch.mm(A, B)
        e1 = ev(); e1.record(cs)
        return e0, e1
    for _ in range(3):
        gemms(20)
    torch.cuda.synchronize()
    res = {}
    for label, fn in (("gemm_alone", None), ("gemm+h2d", "h"), ("gemm+d2h", "d"),
                      ("gemm+duplex", "hd")):
        out = []
        for r in range(5):
            torch.cuda.synchronize()
            ge = gemms(60)
            cev = []
            if fn:
                for _ in range(4):
                    if "h" in fn:
                        cev += one(g_a, hp_a, s1)()
                    if "d" in fn:
                        cev += one(hp_b, g_b, s2)()
            torch.cuda.synchronize()
            out.append((ge[0].elapsed_time(ge[1]) / 1000,
                        [e0.elapsed_time(e1) / 1000 for e0, e1 in cev]))
        gt = sorted(o[0] for o in out)
        ct = sorted(t for o in out for t in o[1])
        res[label] = {"gemm_sec_med": gt[len(gt) // 2],
                      "copy_GBps_med": (nb / 1e9 / ct[len(ct) // 2]) if ct else None}
        print(f"[s217] {label}: 60 GEMMs {res[label]['gemm_sec_med'] * 1000:.1f} ms, copy GB/s "
              f"{res[label]['copy_GBps_med']}", flush=True)
    emit({"kind": "copy_under_gemm", **res})


if __name__ == "__main__":
    main()
