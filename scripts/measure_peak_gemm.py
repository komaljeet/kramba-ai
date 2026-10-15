"""Fresh bf16 GEMM peak on this card (2026-10-15), for the 1B MFU denominator.

Author: Rambarun Komaljeet

Reuses roofline_1b_step.measure_peak_flops (its three square-ish shapes) and adds
the 1B step's own GEMM shapes, plus a SUSTAINED run (~10 s of back-to-back GEMMs)
because a laptop GPU's boost clock can sag over a 16 s training step.
"""
import json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
from scripts.roofline_1b_step import measure_peak_flops, timeit

dev = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
print(f"[peak] {torch.cuda.get_device_name(0)} torch {torch.__version__}", flush=True)
out = {"roofline_measure_peak_flops": measure_peak_flops(dev)}
print(f"[peak] roofline measure_peak_flops(): {out['roofline_measure_peak_flops']/1e12:.2f} TFLOP/s", flush=True)
shapes = {"sq4096": (4096, 4096, 4096), "proj 4096x2048x2048": (4096, 2048, 2048),
          "proj 4096x2048x8192": (4096, 2048, 8192), "dec chunk 512x2048x50257": (512, 2048, 50257),
          "8192x8192x8192": (8192, 8192, 8192)}
per = {}
for name, (m, k, n) in shapes.items():
    a = torch.randn(m, k, device=dev, dtype=torch.bfloat16)
    b = torch.randn(k, n, device=dev, dtype=torch.bfloat16)
    t = timeit(lambda: a @ b, warmup=10, iters=50)
    per[name] = 2 * m * k * n / t
    print(f"[peak] {name:<28} {per[name]/1e12:6.2f} TFLOP/s", flush=True)
    del a, b
out["per_shape"] = per
# sustained ~10 s
a = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
b = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
for _ in range(20):
    a @ b
torch.cuda.synchronize()
windows = []
t_end = time.perf_counter() + 12
while time.perf_counter() < t_end:
    t = time.perf_counter()
    for _ in range(100):
        a @ b
    torch.cuda.synchronize()
    windows.append(100 * 2 * 4096 ** 3 / (time.perf_counter() - t))
out["sustained_windows"] = windows
print(f"[peak] sustained 4096^3 over {len(windows)} windows (~12 s): first {windows[0]/1e12:.2f} "
      f"last {windows[-1]/1e12:.2f} min {min(windows)/1e12:.2f} max {max(windows)/1e12:.2f} TFLOP/s", flush=True)
out["peak"] = max(max(per.values()), out["roofline_measure_peak_flops"], max(windows))
with open(sys.argv[1], "w") as fh:
    json.dump(out, fh, indent=1)
print(f"[peak] best {out['peak']/1e12:.2f} TFLOP/s", flush=True)
