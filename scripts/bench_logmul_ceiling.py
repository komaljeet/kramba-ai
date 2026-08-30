"""Ceiling gate for the log-multiply idea.

Question asked BEFORE building anything: on THIS card, can a multiply-free
matmul (integer adds on CUDA cores) ever beat a bf16 tensor-core GEMM?

Arm A: cuBLAS bf16 matmul               -> tensor cores, the thing we'd replace
Arm B: Triton tl.dot bf16               -> hand-written tensor-core reference
Arm C: Triton add-only reduction        -> the L-Mul / Mitchell ceiling.
        Same loop structure, same memory traffic, NO tl.dot: the inner product
        is int32 add + sum. This is STRICTLY OPTIMISTIC for log-mul -- it does
        not even pay for the exponent/mantissa repack, the bias correction,
        or the int->float conversion. If C loses here, it loses everywhere.
"""
import torch, triton, triton.language as tl, time, json

DEV = "cuda"

@triton.jit
def _dot_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0.0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :],
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))

@triton.jit
def _addonly_kernel(a_ptr, b_ptr, c_ptr, M, N, K,
                    BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """Same tiling / same traffic, inner product via integer ADD only."""
    pm = tl.program_id(0); pn = tl.program_id(1)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    acc = tl.zeros((BM, BN), dtype=tl.int32)
    for k0 in range(0, K, BK):
        rk = k0 + tl.arange(0, BK)
        a = tl.load(a_ptr + rm[:, None] * K + rk[None, :],
                    mask=(rm[:, None] < M) & (rk[None, :] < K), other=0)
        b = tl.load(b_ptr + rk[:, None] * N + rn[None, :],
                    mask=(rk[:, None] < K) & (rn[None, :] < N), other=0)
        # Mitchell: multiply -> add of exponent+mantissa bit patterns.
        # Mitchell reduces a*b to an ADD of the bit patterns:
        # [BM,BK] + [BN,BK] -> [BM,BN,BK], then sum over K.
        prod = a[:, None, :] + tl.trans(b)[None, :, :]
        acc += tl.sum(prod, axis=2)
    tl.store(c_ptr + rm[:, None] * N + rn[None, :], acc,
             mask=(rm[:, None] < M) & (rn[None, :] < N))


def timeit(fn, warmup=10, iters=50):
    for _ in range(warmup): fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(iters): fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / iters


def run(M, N, K, tag):
    flops = 2 * M * N * K
    a16 = torch.randn(M, K, device=DEV, dtype=torch.bfloat16)
    b16 = torch.randn(K, N, device=DEV, dtype=torch.bfloat16)
    c32 = torch.empty(M, N, device=DEV, dtype=torch.float32)
    ai = torch.randint(0, 1 << 20, (M, K), device=DEV, dtype=torch.int32)
    bi = torch.randint(0, 1 << 20, (K, N), device=DEV, dtype=torch.int32)
    ci = torch.empty(M, N, device=DEV, dtype=torch.int32)

    res = {}
    t = timeit(lambda: torch.matmul(a16, b16)); res["cublas_bf16"] = t

    BM, BN, BK = 64, 64, 32
    grid = (triton.cdiv(M, BM), triton.cdiv(N, BN))
    t = timeit(lambda: _dot_kernel[grid](a16, b16, c32, M, N, K, BM, BN, BK))
    res["triton_tl_dot_bf16"] = t

    BM2, BN2, BK2 = 32, 32, 16   # [BM,BN,BK] register tile must stay small
    grid2 = (triton.cdiv(M, BM2), triton.cdiv(N, BN2))
    try:
        t = timeit(lambda: _addonly_kernel[grid2](ai, bi, ci, M, N, K, BM2, BN2, BK2),
                   warmup=3, iters=10)
        res["triton_add_only_int32"] = t
    except Exception as e:
        res["triton_add_only_int32"] = None
        res["add_only_error"] = str(e)[:300]

    print(f"\n=== {tag}  M={M} N={N} K={K}   ({flops/1e9:.1f} GFLOP) ===")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"  {k:26s} {v*1e3:8.3f} ms   {flops/v/1e12:7.2f} TFLOP/s")
        else:
            print(f"  {k:26s} {v}")
    if isinstance(res.get("triton_add_only_int32"), float):
        r = res["triton_add_only_int32"] / res["cublas_bf16"]
        print(f"  --> add-only is {r:.1f}x SLOWER than the tensor-core GEMM")
        res["slowdown_vs_cublas"] = r
    res.update(M=M, N=N, K=K, tag=tag)
    return res

out = []
out.append(run(4096, 2048, 2048, "1B projection (d=2048, seq 4096)"))
out.append(run(2048, 512, 512, "48M projection (d=512, seq 2048)"))
print("\n" + json.dumps(out, indent=1))
