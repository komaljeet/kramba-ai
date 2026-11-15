"""#207 step 4: what SM clock does this card reach when power is NOT the limit?

Runs a light load (one small bf16 matmul at a time, ~10 s) while nvidia-smi samples
clocks/power every 200 ms. If the light-load clock equals the clock seen during the
real k4 step (~2535 MHz median), the step is not power-throttled: it runs at the
card's own boost ceiling. Writes results/round2_207/smi_light_load.csv.
"""
import os, subprocess, sys, time
import torch

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "results", "round2_207", "smi_light_load.csv")
Q = ("timestamp,clocks.sm,clocks.mem,clocks.max.sm,power.draw,enforced.power.limit,temperature.gpu,"
     "utilization.gpu,clocks_throttle_reasons.active,clocks_throttle_reasons.sw_power_cap,pstate")
p = subprocess.Popen(["nvidia-smi", f"--query-gpu={Q}", "--format=csv", "-lms", "200"],
                     stdout=open(OUT, "w"))
a = torch.randn(512, 512, device="cuda", dtype=torch.bfloat16)
t0 = time.time()
while time.time() - t0 < 10:
    for _ in range(50):
        a = (a @ a).clamp_(-1, 1)
    torch.cuda.synchronize()
p.terminate()
print("wrote", OUT)
