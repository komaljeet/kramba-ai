"""#206 step 1: what limits the DeltaNet scan kernels at the 1B shape?

Author: Rambarun Komaljeet

Shapes: batch 1, 64 heads, head dim 32, seq 4096 (N = 64 sequences), the production
strided-IO kernels from hvn/core/deltanet_triton.py (`_gated_delta_fwd_kernel_s`,
`_mk_ckpt_kernel_s`, `_mk_bwd_kernel_s`), inputs as in #203 (q, k, v bf16, q/k
L2-normalised, alpha = sigmoid(3 + N(0,1)) fp32, beta = sigmoid(N(0,1)) fp32).

Measurements (CUDA events, mean of n launches after one warm-up):
  base      each kernel at the production launch config, plus registers, spills,
            shared memory, warps, theoretical occupancy and SM clock during the loop
  nsweep    each kernel at N = 1 .. 128 sequences (same per-sequence work)
  tune      num_warps x num_stages sweep per kernel, time + bit-equality vs production

Hypotheses, written before running, and what refutes them:
  H1 latency: each program is a serial chain whose per-step latency, not the SM's
     issue rate, sets the time. Predicts time(N=1) ~ time(N=20) (one program per SM).
     Refuted if time(N=20) > 1.5 x time(N=1).
  H2 issue-bound at N=64: the SMs are full at N=64. Predicts time(128) ~ 2 x time(64).
     Refuted if time(128) < 1.3 x time(64).
  H3 memory: the scan is limited by DRAM bandwidth. Predicts achieved bytes/s near the
     card's 170-187 GB/s (#203). Refuted if achieved < 50% of that.
"""
import json, sys, os, subprocess, threading, time
import torch

REPO = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import hvn.core.deltanet_triton as DT

dev = torch.device("cuda")
S, D = 4096, 32
PROPS = torch.cuda.get_device_properties(0)


def inputs(Bt, H, seed=0, S=S, D=D):
    g = torch.Generator(device=dev).manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(Bt, S, H, D, device=dev, generator=g), dim=-1).bfloat16()
    k = torch.nn.functional.normalize(torch.randn(Bt, S, H, D, device=dev, generator=g), dim=-1).bfloat16()
    v = torch.randn(Bt, S, H, D, device=dev, generator=g).bfloat16()
    a = torch.sigmoid(3 + torch.randn(Bt, S, H, D, device=dev, generator=g))
    b = torch.sigmoid(torch.randn(Bt, S, H, 1, device=dev, generator=g))
    return q, k, v, a, b


def ev_time(fn, n=10):
    fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


def sm_clock_during(fn, n=40):
    """Median SM clock (MHz) sampled by nvidia-smi while n launches run."""
    fn(); torch.cuda.synchronize()
    clocks = []
    stop = [False]

    def sampler():
        while not stop[0]:
            try:
                r = subprocess.run(["nvidia-smi", "--query-gpu=clocks.sm", "--format=csv,noheader,nounits"],
                                   capture_output=True, text=True, timeout=5)
                clocks.append(int(r.stdout.strip().splitlines()[0]))
            except Exception:
                pass
            time.sleep(0.05)
    th = threading.Thread(target=sampler); th.start()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    stop[0] = True; th.join()
    clocks.sort()
    return clocks[len(clocks) // 2] if clocks else None, len(clocks)


class Prob:
    """One problem instance with production-layout buffers."""

    def __init__(self, Bt, H, seed=0):
        self.q, self.k, self.v, self.a, self.b = inputs(Bt, H, seed)
        self.Bt, self.H, self.N = Bt, H, Bt * H
        self.beta3 = self.b.reshape(Bt, S, H).contiguous()
        self.sB, self.sS, self.sH, _ = self.q.stride()
        self.o = torch.empty(self.N, S, D, device=dev, dtype=torch.float32)
        self.B = DT._MegaScan.block
        self.nblk = S // self.B
        self.ckpt = torch.empty(self.N, self.nblk, D, D, device=dev, dtype=torch.float32)
        g = torch.Generator(device=dev).manual_seed(seed + 100)
        self.do = torch.randn(self.N, S, D, device=dev, generator=g)
        self.dq, self.dk, self.dv = (torch.empty_like(self.q) for _ in range(3))
        self.da = torch.empty_like(self.a); self.db = torch.empty_like(self.beta3)

    def fwd(self, BV=16, **kw):
        return DT._gated_delta_fwd_kernel_s[(self.N, D // BV)](
            self.q, self.k, self.v, self.a, self.beta3, self.o, self.H, self.sB, self.sS, self.sH,
            S=S, D=D, BV=BV, **kw)

    def ck(self, **kw):
        return DT._mk_ckpt_kernel_s[(self.N,)](
            self.k, self.v, self.a, self.beta3, self.ckpt, self.nblk, self.H, self.sB, self.sS, self.sH,
            S=S, D=D, B=self.B, **kw)

    def bw(self, **kw):
        return DT._mk_bwd_kernel_s[(self.N,)](
            self.q, self.k, self.v, self.a, self.beta3, self.ckpt, self.do,
            self.dq, self.dk, self.dv, self.da, self.db, self.nblk, self.H,
            self.sB, self.sS, self.sH, S=S, D=D, B=self.B, **kw)

    def outs(self):
        return [t.clone() for t in (self.o, self.ckpt, self.dq, self.dk, self.dv, self.da, self.db)]


def meta(ck, threads_per_warp=32):
    """Registers / spills / shared / warps and theoretical occupancy on this SM."""
    nw = ck.metadata.num_warps
    regs = ck.n_regs
    smem = ck.metadata.shared
    # sm_89: 64K regs, 48 warps, 24 blocks, 100 KB smem per SM; regs alloc per warp in 256 units
    regs_per_warp = ((regs * threads_per_warp + 255) // 256) * 256
    by_regs = 65536 // max(1, regs_per_warp * nw)
    by_warps = 48 // nw
    by_smem = (100 * 1024) // smem if smem else 24
    blocks = min(by_regs, by_warps, by_smem, 24)
    return {"num_warps": nw, "num_stages": ck.metadata.num_stages, "regs": regs,
            "spills": ck.n_spills, "shared": smem, "blocks_per_sm": blocks,
            "warps_per_sm": blocks * nw, "occupancy": blocks * nw / 48}


def ptx_stats(ck):
    ptx = ck.asm.get("ptx", "")
    return {"bar_sync": ptx.count("bar.sync"), "shfl": ptx.count("shfl.sync"),
            "ld_global": ptx.count("ld.global"), "st_global": ptx.count("st.global"),
            "ptx_lines": ptx.count("\n")}


def run_base(out):
    p = Prob(1, 64)
    r = {"sm_count": PROPS.multi_processor_count, "name": PROPS.name, "N": p.N, "S": S, "D": D}
    for name, fn in (("fwd", p.fwd), ("ckpt", p.ck), ("bwd", p.bw)):
        ck = fn(); torch.cuda.synchronize()
        t = ev_time(fn, 20)
        mhz, ns = sm_clock_during(fn, 60)
        rr = {"ms": t, **meta(ck), **ptx_stats(ck), "sm_mhz": mhz, "clock_samples": ns}
        rr["ns_per_step"] = t * 1e6 / S
        if mhz:
            rr["cycles_per_step"] = t * 1e-3 / S * mhz * 1e6
        r[name] = rr
        print(name, rr, flush=True)
    # bytes moved (lower bound: every input read once, outputs written once)
    el = p.N * S * D
    r["fwd"]["bytes"] = el * (2 + 2 + 2 + 4) + p.N * S * 4 + el * 4
    r["ckpt"]["bytes"] = el * (2 + 2 + 4) + p.N * S * 4 + p.ckpt.numel() * 4
    r["bwd"]["bytes"] = el * (2 + 2 + 2 + 4 + 4) + p.N * S * 4 + p.ckpt.numel() * 4 + el * (2 * 3 + 4) + p.N * S * 4
    for n in ("fwd", "ckpt", "bwd"):
        r[n]["GBps"] = r[n]["bytes"] / (r[n]["ms"] * 1e-3) / 1e9
    out["base"] = r


def run_nsweep(out):
    res = {}
    for N in (1, 2, 4, 10, 20, 40, 64, 128):
        p = Prob(1, N)
        res[N] = {"fwd": ev_time(p.fwd, 5), "ckpt": ev_time(p.ck, 5), "bwd": ev_time(p.bw, 5)}
        print(N, res[N], flush=True)
        del p; torch.cuda.empty_cache()
    out["nsweep"] = res


def run_tune(out):
    p = Prob(1, 64)
    p.fwd(); p.ck(); p.bw(); torch.cuda.synchronize()
    ref = p.outs()
    res = []
    for kname, fn, idx in (("fwd", p.fwd, [0]), ("ckpt", p.ck, [1]), ("bwd", p.bw, [2, 3, 4, 5, 6])):
        cfgs = [(w, s) for w in (1, 2, 4, 8) for s in (1, 2, 3)]
        for bv in ((8, 16, 32) if kname == "fwd" else (None,)):
            for w, s in cfgs:
                kw = {"num_warps": w, "num_stages": s}
                if bv is not None:
                    kw["BV"] = bv
                try:
                    if kname == "bwd":
                        p.ck()  # same checkpoint for every bwd config
                    ck = fn(**kw); torch.cuda.synchronize()
                    cur = p.outs()
                    eq = all(torch.equal(cur[i].view(torch.int16 if cur[i].dtype == torch.bfloat16 else torch.int32),
                                         ref[i].view(torch.int16 if ref[i].dtype == torch.bfloat16 else torch.int32))
                             for i in idx)
                    md = max((cur[i].float() - ref[i].float()).abs().max().item() for i in idx)
                    t = ev_time(lambda: fn(**kw), 10)
                    row = {"kernel": kname, "BV": bv, "num_warps": w, "num_stages": s, "ms": t,
                           "bitequal": eq, "maxdiff": md, "regs": ck.n_regs, "spills": ck.n_spills}
                except Exception as e:  # compile failures are data too
                    row = {"kernel": kname, "BV": bv, "num_warps": w, "num_stages": s, "error": repr(e)[:200]}
                res.append(row); print(row, flush=True)
                # restore production outputs for the next comparison
                p.fwd(); p.ck(); torch.cuda.synchronize()
    out["tune"] = res


if __name__ == "__main__":
    path = sys.argv[1]
    which = sys.argv[3].split(",") if len(sys.argv) > 3 else ["base", "nsweep", "tune"]
    out = {}
    if os.path.exists(path):
        out = json.load(open(path))
    for w in which:
        {"base": run_base, "nsweep": run_nsweep, "tune": run_tune}[w](out)
        json.dump(out, open(path, "w"), indent=1)
