"""#211: hold a fixed amount of VRAM or host RAM, to test what slows the embedding turn.

Author: Rambarun Komaljeet

  --vram-mib N   allocate N MiB on the GPU from THIS process (another WDDM client,
                 like a desktop app holding VRAM). Idle by default; with --vram-touch
                 it adds 1 to every element every --touch-sec seconds (an app that
                 keeps using its memory, so WDDM keeps it resident).
  --host-mib N   allocate N MiB of host RAM and re-touch every page every --touch-sec
                 seconds, so it stays in this process's working set and the OS has to
                 trim other processes (the trainer) instead.
Runs until --seconds elapse or it is killed. Prints one line when the memory is held.
"""
import argparse, time, sys

ap = argparse.ArgumentParser()
ap.add_argument("--vram-mib", type=int, default=0)
ap.add_argument("--host-mib", type=int, default=0)
ap.add_argument("--touch-sec", type=float, default=2.0)
ap.add_argument("--vram-touch", action="store_true")
ap.add_argument("--seconds", type=float, default=3600)
a = ap.parse_args()
held = []
if a.vram_mib:
    import torch
    x = torch.empty(a.vram_mib * 2**20, dtype=torch.uint8, device="cuda")
    x.fill_(1); torch.cuda.synchronize()
    held.append(x)
if a.host_mib:
    import numpy as np
    h = np.ones(a.host_mib * 2**20, dtype=np.uint8)
    held.append(h)
print(f"[ballast] holding vram {a.vram_mib} MiB host {a.host_mib} MiB", flush=True)
t_end = time.time() + a.seconds
while time.time() < t_end:
    if a.host_mib:
        h[::4096] += 1          # one write per page keeps every page resident
    if a.vram_mib and a.vram_touch:
        x.add_(1); torch.cuda.synchronize()
    time.sleep(a.touch_sec)
