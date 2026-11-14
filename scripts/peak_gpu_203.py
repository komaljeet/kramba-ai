"""#203: measured peak fp8 / bf16 GEMM throughput and memory bandwidth on this card.

Author: Rambarun Komaljeet

The lower bound for a layer phase is its FLOPs at the measured GEMM peak plus its
bytes at the measured bandwidth. Datasheet numbers are not used: the GeForce fp8
rate with fp32 accumulate, the laptop clocks and the power limit all decide it.

GEMM shapes are the layer's own (M = 4096 tokens, d = 2048, FFN 4096) plus one
large square shape for the card's best case. Each is timed with CUDA events over
50 back-to-back calls after 10 warm-up calls, median of 5 repeats.

Falsifier: if the fp8 number came out at or below the bf16 one, the fp8 path is not
on the fp8 tensor cores and the "fp8 bound" would be wrong.
"""
import json, sys
import torch

dev = torch.device("cuda")
E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2


def timeit(fn, n=50, rep=5):
    for _ in range(10):
        fn()
    ts = []
    for _ in range(rep):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        torch.cuda.synchronize(); a.record()
        for _ in range(n):
            fn()
        b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b) / 1e3 / n)
    ts.sort()
    return ts[len(ts) // 2]


out = {"device": torch.cuda.get_device_name(0),
       "sms": torch.cuda.get_device_properties(0).multi_processor_count, "gemm": [], "bw": {}}
shapes = [(4096, 2048, 2048), (4096, 4096, 2048), (4096, 2048, 4096), (4096, 2048, 64),
          (2048, 2048, 4096), (8192, 8192, 8192)]
one = torch.ones((), device=dev)
for M, N, K in shapes:
    a = torch.randn(M, K, device=dev).to(torch.bfloat16)
    b = torch.randn(N, K, device=dev).to(torch.bfloat16)
    t_bf = timeit(lambda: a @ b.t())
    fl = 2 * M * N * K
    row = {"M": M, "N": N, "K": K, "bf16_s": t_bf, "bf16_tflops": fl / t_bf / 1e12}
    if N % 16 == 0 and K % 16 == 0:
        a8 = a.to(E4M3); b8 = b.to(E4M3)
        t8 = timeit(lambda: torch._scaled_mm(a8, b8.t(), scale_a=one, scale_b=one,
                                             out_dtype=torch.bfloat16))
        g8 = a.to(E5M2)
        t85 = timeit(lambda: torch._scaled_mm(g8, b8.t(), scale_a=one, scale_b=one,
                                              out_dtype=torch.bfloat16))
        row.update(fp8_s=t8, fp8_tflops=fl / t8 / 1e12, fp8_e5m2_tflops=fl / t85 / 1e12)
    out["gemm"].append(row)
    print(row, flush=True)
    del a, b

n = 256 * 2 ** 20                       # 256M fp32 = 1 GiB
x = torch.empty(n, device=dev); y = torch.empty_like(x)
t = timeit(lambda: y.copy_(x), n=20)
out["bw"]["copy_GBps"] = 2 * x.numel() * 4 / t / 1e9          # read + write
t = timeit(lambda: x.sum(), n=20)
out["bw"]["read_GBps"] = x.numel() * 4 / t / 1e9
del x, y
# PCIe: pinned host -> device, 84 MB (one frozen layer's bf16 weights)
h = torch.empty(42 * 2 ** 20, dtype=torch.bfloat16).pin_memory()
d = torch.empty_like(h, device=dev)
t = timeit(lambda: d.copy_(h, non_blocking=True), n=10)
out["bw"]["h2d_pinned_GBps"] = h.numel() * 2 / t / 1e9
print(out["bw"], flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1)
