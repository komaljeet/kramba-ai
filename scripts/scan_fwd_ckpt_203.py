"""#203 prototype measurement: can the scan FORWARD write the backward's state checkpoints?

Author: Rambarun Komaljeet

In a checkpointed layer's backward the scan runs three serial passes over the
sequence: the recompute's forward (_gated_delta_fwd_kernel_s, needed for o), then
_mk_ckpt_kernel_s (the SAME state recurrence again, to store the state every B=4
tokens), then _mk_bwd_kernel_s. The checkpoint pass is 64 programs on 20 SMs and
latency bound (scripts/scan_occupancy_203.py). The forward already holds every state.

This times a forward kernel that also stores the checkpoints (each program stores its
own BV rows of the [D, D] state) and checks the stored states are BIT-equal to
_mk_ckpt_kernel_s's and o is bit-equal to the production forward's. If the states
differ in any bit, the backward would not be bit-identical and the idea fails as a
drop-in.
"""
import json, sys
import torch
import triton
import triton.language as tl
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else ".")
import hvn.core.deltanet_triton as DT


@triton.jit
def _fwd_ckpt_kernel_s(q_ptr, k_ptr, v_ptr, alpha_ptr, beta_ptr, o_ptr, ckpt_ptr, nblk,
                       H, sB, sS, sH, S: tl.constexpr, D: tl.constexpr, BV: tl.constexpr,
                       B: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_v = tl.program_id(1)
    bi = pid_n // H
    hi = pid_n % H
    offs_k = tl.arange(0, D)
    offs_v = pid_v * BV + tl.arange(0, BV)
    base = bi * sB + hi * sH
    obase = pid_n * S * D
    cbase = pid_n * nblk * D * D
    state = tl.zeros((BV, D), dtype=tl.float32)
    for t in range(S):
        if t % B == 0:
            tl.store(ckpt_ptr + cbase + (t // B) * D * D + offs_v[:, None] * D + offs_k[None, :], state)
        row = base + t * sS
        k_t = tl.load(k_ptr + row + offs_k).to(tl.float32)
        q_t = tl.load(q_ptr + row + offs_k).to(tl.float32)
        a_t = tl.load(alpha_ptr + row + offs_k).to(tl.float32)
        v_t = tl.load(v_ptr + row + offs_v).to(tl.float32)
        b_t = tl.load(beta_ptr + (bi * S + t) * H + hi).to(tl.float32)
        state = state * a_t[None, :]
        Sk = tl.sum(state * k_t[None, :], axis=1)
        u = v_t - Sk
        state = state + b_t * (u[:, None] * k_t[None, :])
        o_t = tl.sum(state * q_t[None, :], axis=1)
        tl.store(o_ptr + obase + t * D + offs_v, o_t.to(o_ptr.dtype.element_ty))


dev = torch.device("cuda")
S, D, H, Bt = 4096, 32, 64, 1
g = torch.Generator(device=dev).manual_seed(0)
q = torch.nn.functional.normalize(torch.randn(Bt, S, H, D, device=dev, generator=g), dim=-1).bfloat16()
k = torch.nn.functional.normalize(torch.randn(Bt, S, H, D, device=dev, generator=g), dim=-1).bfloat16()
v = torch.randn(Bt, S, H, D, device=dev, generator=g).bfloat16()
a = torch.sigmoid(3 + torch.randn(Bt, S, H, D, device=dev, generator=g))
beta3 = torch.sigmoid(torch.randn(Bt, S, H, device=dev, generator=g))
N = Bt * H
B = DT._MegaScan.block
nblk = S // B
sB, sS, sH, _ = q.stride()
o_ref = torch.empty(N, S, D, device=dev); o_new = torch.empty_like(o_ref)
ck_ref = torch.empty(N, nblk, D, D, device=dev); ck_new = torch.empty_like(ck_ref)


def t(fn, n=10):
    fn(); torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


ref_fwd = lambda: DT._gated_delta_fwd_kernel_s[(N, 2)](q, k, v, a, beta3, o_ref, H, sB, sS, sH, S=S, D=D, BV=16)
ref_ck = lambda: DT._mk_ckpt_kernel_s[(N,)](k, v, a, beta3, ck_ref, nblk, H, sB, sS, sH, S=S, D=D, B=B)
res = {"fwd_ms": t(ref_fwd), "ckpt_ms": t(ref_ck)}
for BV in (32, 16, 8):
    new = lambda: _fwd_ckpt_kernel_s[(N, D // BV)](q, k, v, a, beta3, o_new, ck_new, nblk, H, sB, sS, sH,
                                                   S=S, D=D, BV=BV, B=B)
    res[f"fused_bv{BV}_ms"] = t(new)
    new(); ref_fwd(); ref_ck(); torch.cuda.synchronize()
    res[f"fused_bv{BV}_o_bitequal"] = bool(torch.equal(o_ref.view(torch.int32), o_new.view(torch.int32)))
    res[f"fused_bv{BV}_ckpt_bitequal"] = bool(torch.equal(ck_ref.view(torch.int32), ck_new.view(torch.int32)))
    res[f"fused_bv{BV}_ckpt_maxdiff"] = float((ck_ref - ck_new).abs().max())
print(res, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1)
