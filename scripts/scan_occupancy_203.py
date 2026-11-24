"""#203: is the DeltaNet scan at the 1B shape limited by parallelism (too few programs)?

Author: Rambarun Komaljeet

The production scan runs N = batch x heads = 64 independent sequences of 4096 steps
on a 20-SM card: forward grid (64, 2) = 128 programs, checkpoint and backward grids
(64,) = 64 programs. Each program is a serial chain over the sequence.

Test: time each kernel at N = 32, 64, 128 (same per-sequence work). If the time at
N = 128 is about the time at N = 64, the card had room for twice the programs and a
kernel that splits each sequence's work across more programs could approach 2x.
If time doubles with N, the card is already full and splitting cannot help.
(Falsifier: the N = 32 point. If 32 -> 64 also does not change the time, the kernel is
latency bound per program; if 32 -> 64 doubles it, the card was already full at 32.)

Also: the forward with BV = 8 (256 programs) instead of 16, timed and compared bit for
bit to BV = 16 (each program owns whole rows of the state, so the result should not
depend on BV unless the in-row reduction order changes with the tile shape).
"""
import json, sys
import torch
sys.path.insert(0, sys.argv[2] if len(sys.argv) > 2 else ".")
import hvn.core.deltanet_triton as DT

dev = torch.device("cuda")
S, D = 4096, 32


def inputs(Bt, H, seed=0):
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


out = {}
for Bt, H in ((1, 32), (1, 64), (2, 64), (4, 64)):
    q, k, v, a, b = inputs(Bt, H)
    N = Bt * H
    beta3 = b.reshape(Bt, S, H).contiguous()
    sB, sS, sH, _ = q.stride()
    o = torch.empty(N, S, D, device=dev, dtype=torch.float32)
    B = DT._MegaScan.block
    nblk = S // B
    ckpt = torch.empty(N, nblk, D, D, device=dev, dtype=torch.float32)
    do = torch.randn(N, S, D, device=dev)
    dq, dk, dv = (torch.empty_like(q) for _ in range(3))
    da = torch.empty_like(a); db = torch.empty_like(beta3)

    def fwd(BV=16):
        DT._gated_delta_fwd_kernel_s[(N, D // BV)](q, k, v, a, beta3, o, H, sB, sS, sH, S=S, D=D, BV=BV)

    def ck():
        DT._mk_ckpt_kernel_s[(N,)](k, v, a, beta3, ckpt, nblk, H, sB, sS, sH, S=S, D=D, B=B)

    def bw():
        DT._mk_bwd_kernel_s[(N,)](q, k, v, a, beta3, ckpt, do, dq, dk, dv, da, db, nblk, H,
                                  sB, sS, sH, S=S, D=D, B=B)
    r = {"N": N, "fwd_bv16_ms": ev_time(fwd), "ckpt_ms": ev_time(ck), "bwd_ms": ev_time(bw)}
    fwd(16); o16 = o.clone()
    r["fwd_bv8_ms"] = ev_time(lambda: fwd(8))
    fwd(8)
    r["fwd_bv8_bitequal"] = bool(torch.equal(o16.view(torch.int32), o.view(torch.int32)))
    if D // 4 >= 1:
        r["fwd_bv4_ms"] = ev_time(lambda: fwd(4))
        fwd(4)
        r["fwd_bv4_bitequal"] = bool(torch.equal(o16.view(torch.int32), o.view(torch.int32)))
    out[f"B{Bt}_H{H}"] = r
    print(r, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
