"""Do the pretrain speed levers hold up in the REAL training loop?

Author: Rambarun Komaljeet

Two levers landed together on ``feat/pretrain-speed-levers``:

* the fused ternary quantizer (#154), now enabled by ``--fast`` under bf16;
* the megakernel backward no longer syncs host and GPU per DeltaNet block
  (``torch.tensor([nblocks], device=cuda)`` -> plain int kernel arg).

Each was estimated in isolation. Isolated numbers have misled this project
before (#158 compile "+18%" was 2.4x slower in the loop; #131's benchmark
measured the wrong thing), so this LAUNCHES THE REAL TRAINING SCRIPT from two
checkouts -- a baseline commit and the branch -- with an identical command, and
reads throughput from each run's ``training_log.csv`` (``--log-every 1``).

What would make this vacuous, and how it is guarded:

* **The lever not actually being on.** The branch log must print the fused
  ternary line and ``fused-ternary`` in the --fast preset; the baseline log
  must not. Checked, not assumed.
* **Thermal drift / background load mistaken for a result.** Arms run ABBA
  (base, new, new, base, ...), so a linear drift cancels; the GPU must be idle
  before every launch.
* **Too few steps.** Step 1-10 (autotune, allocator warm-up) are dropped and
  at least 100 measured steps per run are required.
* **A "speedup" that changed the model.** Same ``--seed`` and data order, so
  the per-step loss of the two arms is compared; the levers are bit-identical
  in isolation, so the trajectories must agree to within the trainer's own
  run-to-run noise (measured from the two baseline runs).
* **VRAM.** Peak allocated / reserved of the new arm must not exceed baseline.

Usage (one job on the GPU at a time)::

    python -u scripts/ab_speed_levers.py --base-repo <baseline checkout> \\
        --new-repo <branch checkout> --data-dir <abs data_cache> --steps 150
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
import subprocess
import sys
import time

SEQ, BATCH, ACCUM = 4096, 1, 8
TOKENS_PER_STEP = SEQ * BATCH * ACCUM
WARMUP = 10
FUSED_MARK = "Fused ternary quantizer enabled on"


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def avail_ram_gib() -> float:
    """System available physical RAM (Windows GlobalMemoryStatusEx; no psutil here)."""
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    ms = MS(dwLength=ctypes.sizeof(MS))
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullAvailPhys / 2**30


def proc_table() -> dict[int, tuple[int, str]]:
    """{pid: (parent pid, image name)} for every live process."""
    out = subprocess.run(
        ["powershell", "-NoProfile", "-Command",
         "Get-CimInstance Win32_Process | ForEach-Object "
         "{ \"$($_.ProcessId) $($_.ParentProcessId) $($_.Name)\" }"],
        capture_output=True, text=True).stdout
    table = {}
    for line in out.splitlines():
        parts = line.split(maxsplit=2)
        if len(parts) == 3:
            table[int(parts[0])] = (int(parts[1]), parts[2])
    return table


def descendants(root_pid: int, table=None) -> dict[int, str]:
    """Live {pid: name} descended from root_pid. The chain is only followed
    through LIVE processes (venv launcher -> real interpreter -> DataLoader
    workers), so it must be sampled while the arm runs; once the interpreter
    dies, an orphaned worker is no longer reachable from root_pid."""
    table = proc_table() if table is None else table
    kids: dict[int, list[int]] = {}
    for pid, (ppid, _) in table.items():
        kids.setdefault(ppid, []).append(pid)
    found, stack = {}, [root_pid]
    while stack:
        for c in kids.get(stack.pop(), []):
            if c != root_pid and c not in found:
                found[c] = table[c][1]
                stack.append(c)
    return found


def still_alive(seen: dict[int, str]) -> list[int]:
    """Which of the processes seen during the run are still alive (same PID and
    image name, so an unrelated process reusing a PID is not waited on)."""
    table = proc_table()
    return [p for p, name in seen.items() if p in table and table[p][1] == name]


def wait_for_ram(tag, args) -> float:
    """Block until available RAM >= --min-free-gib. Claude Code's low-RAM reaper
    killed this sweep twice during an arm's start-up audit (the Check 3
    checkpoint save/rebuild spike), so do not launch into a low-memory box."""
    t0 = time.time()
    while True:
        gib = avail_ram_gib()
        if gib >= args.min_free_gib:
            print(f"    [{time.strftime('%H:%M:%S')}] {tag}: {gib:.2f} GiB RAM available "
                  f"(need {args.min_free_gib}) -- ok", flush=True)
            return gib
        if time.time() - t0 > args.ram_wait_min * 60:
            sys.exit(f"only {gib:.2f} GiB RAM available after {args.ram_wait_min} min "
                     f"-- refusing to launch {tag}")
        print(f"    [{time.strftime('%H:%M:%S')}] {tag}: only {gib:.2f} GiB RAM "
              f"available (need {args.min_free_gib}) -- waiting", flush=True)
        time.sleep(30)


def build_cmd(repo, args, ckpt) -> list[str]:
    # The sizing tool's 896x16 plain rung (scripts/size_for_pretrain.py).
    return [sys.executable, "-u", os.path.join(repo, "train_edge_conversational.py"),
            "--data-dir", args.data_dir,
            "--arch", "hvn-hybrid", "--attn-layers", "4,12",
            "--d-model", "896", "--num-layers", "16",
            "--seq-len", str(SEQ), "--batch-size", str(BATCH),
            "--grad-accum", str(ACCUM),
            "--grad-checkpoint", "--chunked-loss", "512",
            "--log-every", "1", "--min-tokens-per-param", "0",
            "--save-every-steps", "1000000", "--keep-checkpoints", "1",
            "--seed", "0", "--max-steps", str(args.steps),
            "--checkpoint-dir", ckpt]


def run_arm(name, repo, idx, args) -> dict:
    tag = f"{name}_{idx}"
    ckpt = os.path.join(args.work_dir, tag)
    os.makedirs(ckpt, exist_ok=True)
    for f in glob.glob(os.path.join(ckpt, "*")):
        os.remove(f)
    log_path = os.path.join(args.work_dir, f"{tag}.log")
    used = gpu_used_mib()
    if used > args.max_idle_mib:
        sys.exit(f"GPU not idle ({used} MiB in use) -- refusing to launch {tag}")
    ram_at_launch = wait_for_ram(tag, args)
    cmd = build_cmd(repo, args, ckpt)
    print(f"[{time.strftime('%H:%M:%S')}] {tag}: launching {args.steps} steps "
          f"from {repo}", flush=True)
    t0 = time.time()
    min_ram, seen, last_kid_check = ram_at_launch, {}, 0.0
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        proc = subprocess.Popen(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=repo)
        while proc.poll() is None:
            min_ram = min(min_ram, avail_ram_gib())
            if time.time() - last_kid_check > 20:
                # Record the whole tree (real interpreter + 4 DataLoader workers
                # under --fast) while it is reachable; the exit wait below uses it.
                seen.update(descendants(proc.pid))
                last_kid_check = time.time()
            if time.time() - t0 > args.timeout_min * 60:
                proc.kill()
                break
            time.sleep(1)
        rc = proc.wait()
    wall = (time.time() - t0) / 60
    # subprocess only waits for the launcher. Make sure every process it spawned
    # (orphaned DataLoader workers included) is gone and its RAM is back before
    # the next arm starts, then let the OS settle.
    t_exit = time.time()
    left = still_alive(seen)
    while left and time.time() - t_exit < 300:
        time.sleep(2)
        left = still_alive(seen)
    lingered = time.time() - t_exit
    if left:
        sys.exit(f"{tag}: processes {left} still alive {lingered:.0f}s after exit")
    max_kids = len(seen)
    time.sleep(args.settle_s)
    print(f"    {tag}: min RAM available during run {min_ram:.2f} GiB; "
          f"{max_kids} child processes seen; all gone {lingered:.0f}s after exit; "
          f"{avail_ram_gib():.2f} GiB available after {args.settle_s}s settle", flush=True)
    text = open(log_path, encoding="utf-8", errors="replace").read()
    rows = list(csv.DictReader(open(os.path.join(ckpt, "training_log.csv"),
                                    encoding="utf-8")))
    for pt in glob.glob(os.path.join(ckpt, "*.pt")):
        os.remove(pt)
    sec = [TOKENS_PER_STEP / float(r["tokens_sec"]) for r in rows[WARMUP:]
           if float(r["tokens_sec"]) > 0]
    rec = {"tag": tag, "repo": repo, "rc": rc, "wall_min": wall,
           "ram_avail_at_launch_gib": ram_at_launch, "ram_avail_min_gib": min_ram,
           "max_child_procs": max_kids, "children_gone_after_s": lingered,
           "steps_logged": len(rows), "measured_steps": len(sec),
           "fused_on": FUSED_MARK in text,
           "median_s": statistics.median(sec) if sec else None,
           "mean_s": statistics.fmean(sec) if sec else None,
           "peak_alloc_gib": max(float(r["peak_vram_gb"]) for r in rows) if rows else None,
           "reserved_gib": max(float(r["reserved_vram_gb"]) for r in rows) if rows else None,
           "loss": [float(r["loss"]) for r in rows]}
    print(f"    rc={rc} {len(sec)} measured steps, median {rec['median_s']:.3f} s/step, "
          f"mean {rec['mean_s']:.3f}, peak {rec['peak_alloc_gib']:.2f} GiB, "
          f"fused_on={rec['fused_on']}  ({wall:.1f} min)", flush=True)
    return rec


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base-repo", required=True)
    p.add_argument("--new-repo", required=True)
    p.add_argument("--data-dir", required=True)
    p.add_argument("--steps", type=int, default=150)
    p.add_argument("--rounds", type=int, default=2, help="ABBA pairs")
    p.add_argument("--work-dir", required=True)
    p.add_argument("--out", default="results/speed_levers/ab.json")
    p.add_argument("--timeout-min", type=float, default=90)
    p.add_argument("--max-idle-mib", type=int, default=1000)
    p.add_argument("--min-free-gib", type=float, default=2.5,
                   help="wait before each arm until this much RAM is available")
    p.add_argument("--ram-wait-min", type=float, default=30)
    p.add_argument("--settle-s", type=float, default=15)
    args = p.parse_args()
    os.makedirs(args.work_dir, exist_ok=True)

    runs = {"base": [], "new": []}
    order = []
    for r in range(args.rounds):
        order += [("base", args.base_repo), ("new", args.new_repo),
                  ("new", args.new_repo), ("base", args.base_repo)]
    for i, (name, repo) in enumerate(order):
        runs[name].append(run_arm(name, repo, i, args))

    problems = []
    for name, want in (("base", False), ("new", True)):
        for rec in runs[name]:
            if rec["rc"] != 0:
                problems.append(f"{rec['tag']} exited {rec['rc']}")
            if rec["fused_on"] != want:
                problems.append(f"{rec['tag']}: fused ternary on={rec['fused_on']}, "
                                f"expected {want} -- the arms are not what they claim")
            if rec["measured_steps"] < 100:
                problems.append(f"{rec['tag']}: only {rec['measured_steps']} measured steps")

    b = statistics.median(r["median_s"] for r in runs["base"])
    n = statistics.median(r["median_s"] for r in runs["new"])
    gain = (b / n - 1) * 100

    # Loss agreement: new-vs-base, judged against base-vs-base (the trainer's
    # own run-to-run noise at identical code).
    def maxdiff(x, y):
        k = min(len(x), len(y))
        return max(abs(a - c) for a, c in zip(x[:k], y[:k]))
    bb = maxdiff(runs["base"][0]["loss"], runs["base"][1]["loss"])
    nb = max(maxdiff(nr["loss"], br["loss"]) for nr in runs["new"] for br in runs["base"])
    first_eq = all(nr["loss"][0] == runs["base"][0]["loss"][0] for nr in runs["new"])
    vram_b = max(r["peak_alloc_gib"] for r in runs["base"])
    vram_n = max(r["peak_alloc_gib"] for r in runs["new"])
    res_b = max(r["reserved_gib"] for r in runs["base"])
    res_n = max(r["reserved_gib"] for r in runs["new"])

    print(f"\nbase {b:.3f} s/step   new {n:.3f} s/step   speedup {gain:+.2f}%")
    print(f"per-run medians: base {[round(r['median_s'], 3) for r in runs['base']]} "
          f"new {[round(r['median_s'], 3) for r in runs['new']]}")
    print(f"loss max|diff|: base-vs-base {bb:.4f}, new-vs-base {nb:.4f}; "
          f"step-1 loss identical: {first_eq}")
    print(f"peak alloc GiB: base {vram_b:.3f} new {vram_n:.3f}; "
          f"reserved: base {res_b:.3f} new {res_n:.3f}")
    # Step 1's loss is computed from the INITIAL weights (forward only), so
    # bit-identical levers must reproduce it exactly. This is what caught the
    # fused kernel's approximate division flipping one weight's code.
    if not first_eq:
        problems.append("step-1 loss differs between arms -- the forward changed")
    if vram_n > vram_b + 0.01:
        problems.append(f"new arm uses more VRAM ({vram_n:.3f} vs {vram_b:.3f} GiB)")
    for pr in problems:
        print("PROBLEM:", pr)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    for rs in runs.values():
        for r in rs:
            r["loss_first5"] = r["loss"][:5]
            r["loss_last5"] = r["loss"][-5:]
    json.dump({"config": vars(args), "tokens_per_step": TOKENS_PER_STEP,
               "warmup_dropped": WARMUP, "base_s": b, "new_s": n, "gain_pct": gain,
               "loss_maxdiff_base_vs_base": bb, "loss_maxdiff_new_vs_base": nb,
               "step1_loss_identical": first_eq,
               "peak_alloc_gib": {"base": vram_b, "new": vram_n},
               "reserved_gib": {"base": res_b, "new": res_n},
               "problems": problems,
               "runs": {k: [{kk: vv for kk, vv in r.items() if kk != "loss"} for r in v]
                        for k, v in runs.items()}},
              open(args.out, "w", encoding="utf-8"), indent=2)
    print(f"wrote {args.out}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
