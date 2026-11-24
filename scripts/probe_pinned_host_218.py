"""#218 gate 1, step 1: how much page-locked (pinned) host RAM can ONE process hold
on this laptop, and what happens to the rest of the machine when it does.

Author: Rambarun Komaljeet

One process per target size (run_pinned.sh launches them, one log each). The
process makes a CUDA context (as a trainer would), then page-locks ``--gb`` GB in
``--chunk-mb`` chunks with cudaHostAlloc (``--api alloc``, called directly from
cudart64_12.dll so there is no torch power-of-2 rounding) or with
cudaHostRegister on plain malloc'd tensors (``--api register``, what #217 used).

Recorded per chunk: call time, system available RAM, process working set, commit.
After the target is held:
  * proof the memory is really pinned: torch.is_pinned() on the first and last
    chunk, and the H2D rate from them (pinned ~13 GB/s on this laptop, #217;
    pageable is slower AND makes the host block). A "pinned" buffer that was
    secretly pageable would fail is_pinned or show the lower rate.
  * responsiveness while held: time to allocate+touch 512 MB of ordinary memory in
    this process, and time to spawn `cmd /c exit` 5 times (a stand-in for "is
    the desktop sluggish": both need free pages; if Windows is paging other
    processes out to make room, both slow down).
  * hold ``--hold`` seconds, sampling available RAM and commit every second.
Then everything is freed and the same responsiveness probes run again.

System paging during the run is sampled outside this script by typeperf
(Pages Output/sec = other processes' pages written to the pagefile; Pages
Input/sec = hard faults reading them back), see run_pinned.sh.
"""
import os, sys, json, time, argparse, ctypes, subprocess
from ctypes import wintypes


class _PMC(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
        (n, ctypes.c_size_t) for n in (
            "PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
            "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
            "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage",
            "PrivateUsage")]


class _PI(ctypes.Structure):
    _fields_ = [("cb", wintypes.DWORD)] + [(n, ctypes.c_size_t) for n in (
        "CommitTotal", "CommitLimit", "CommitPeak", "PhysicalTotal", "PhysicalAvailable",
        "SystemCache", "KernelTotal", "KernelPaged", "KernelNonpaged", "PageSize")] + [
        (n, wintypes.DWORD) for n in ("HandleCount", "ProcessCount", "ThreadCount")]


K32 = ctypes.windll.kernel32
K32.GetCurrentProcess.restype = wintypes.HANDLE
K32.K32GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.c_void_p, wintypes.DWORD]
PSAPI = ctypes.windll.psapi


def mem():
    """GB: process working set / private commit, system available / commit."""
    pmc = _PMC(); pmc.cb = ctypes.sizeof(_PMC)
    K32.K32GetProcessMemoryInfo(K32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
    pi = _PI(); pi.cb = ctypes.sizeof(_PI)
    PSAPI.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    ps, g = pi.PageSize, 1e9
    return {"ws": pmc.WorkingSetSize / g, "priv": pmc.PrivateUsage / g,
            "pf": pmc.PageFaultCount,
            "avail": pi.PhysicalAvailable * ps / g, "commit": pi.CommitTotal * ps / g,
            "commit_limit": pi.CommitLimit * ps / g, "syscache": pi.SystemCache * ps / g,
            "kernel_np": pi.KernelNonpaged * ps / g}


def responsiveness(torch):
    t = time.perf_counter()
    x = torch.empty(512 * 2 ** 20, dtype=torch.uint8)
    x.fill_(1)
    touch = time.perf_counter() - t
    del x
    spawns = []
    for _ in range(5):
        t = time.perf_counter()
        subprocess.run(["cmd", "/c", "exit"], check=False)
        spawns.append(time.perf_counter() - t)
    return {"touch512MB_s": touch, "spawn_med_s": sorted(spawns)[2], "spawn_max_s": max(spawns)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gb", type=float, required=True)
    ap.add_argument("--chunk-mb", type=int, default=256)
    ap.add_argument("--api", choices=["alloc", "register"], default="alloc")
    ap.add_argument("--hold", type=int, default=15)
    ap.add_argument("--min-avail-gb", type=float, default=0.25,
                    help="guard: stop allocating if system available RAM drops below this")
    ap.add_argument("--max-chunk-s", type=float, default=20.0,
                    help="guard: stop if one chunk takes longer than this (machine thrashing)")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = open(a.out, "w", encoding="utf-8")

    def emit(row):
        out.write(json.dumps(row) + "\n"); out.flush()

    m0 = mem()
    import torch
    torch.zeros(1, device="cuda")                        # CUDA context, as a trainer has
    torch.cuda.synchronize()
    m_ctx = mem()
    cudart = ctypes.CDLL(os.path.join(os.path.dirname(torch.__file__), "lib", "cudart64_12.dll"))
    cudart.cudaHostAlloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t, ctypes.c_uint]
    cudart.cudaHostAlloc.restype = ctypes.c_int
    cudart.cudaFreeHost.argtypes = [ctypes.c_void_p]
    cudart.cudaFreeHost.restype = ctypes.c_int
    cudart.cudaHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    cudart.cudaHostRegister.restype = ctypes.c_int
    cudart.cudaHostUnregister.argtypes = [ctypes.c_void_p]
    cudart.cudaHostUnregister.restype = ctypes.c_int
    cudart.cudaGetLastError.restype = ctypes.c_int
    resp0 = responsiveness(torch)
    emit({"kind": "start", "gb": a.gb, "api": a.api, "chunk_mb": a.chunk_mb,
          "mem_start": m0, "mem_ctx": m_ctx, "resp_before": resp0})
    print(f"[p218] start gb={a.gb} api={a.api} avail {m_ctx['avail']:.2f} GB ws {m_ctx['ws']:.2f} "
          f"commit {m_ctx['commit']:.2f}/{m_ctx['commit_limit']:.2f} resp {resp0}", flush=True)

    chunk = a.chunk_mb * 2 ** 20
    target = int(a.gb * 2 ** 30)
    held, bufs, status, err = 0, [], "ok", 0
    t_all = time.perf_counter()
    while held < target:
        n = min(chunk, target - held)
        t = time.perf_counter()
        if a.api == "alloc":
            p = ctypes.c_void_p()
            r = cudart.cudaHostAlloc(ctypes.byref(p), n, 0)
            obj = p.value
        else:
            tt = torch.empty(n, dtype=torch.uint8)
            tt.fill_(0)                                  # commit the pages first
            r = cudart.cudaHostRegister(tt.data_ptr(), n, 0)
            obj = tt
        dt = time.perf_counter() - t
        if r != 0:
            cudart.cudaGetLastError()
            status, err = "fail", r
            emit({"kind": "chunk_fail", "held_gb": held / 2 ** 30, "err": r, "sec": dt, "mem": mem()})
            print(f"[p218] FAIL at {held / 2 ** 30:.2f} GB: cuda error {r} after {dt:.2f}s", flush=True)
            break
        bufs.append((obj, n))
        held += n
        m = mem()
        emit({"kind": "chunk", "held_gb": held / 2 ** 30, "sec": dt, "mem": m})
        if len(bufs) % 4 == 0 or held >= target:
            print(f"[p218] held {held / 2 ** 30:.2f} GB  chunk {dt * 1000:.0f} ms  avail {m['avail']:.2f} "
                  f"ws {m['ws']:.2f} commit {m['commit']:.2f}", flush=True)
        if dt > a.max_chunk_s:
            status = "guard_slow"; break
        if m["avail"] < a.min_avail_gb and held < target:
            status = "guard_avail"; break
    t_alloc = time.perf_counter() - t_all
    m_full = mem()

    # proof of pinning: is_pinned + H2D rate from the first and last chunk
    def as_tensor(obj, n):
        if a.api == "register":
            return obj
        arr = (ctypes.c_uint8 * n).from_address(obj)
        return torch.frombuffer(arr, dtype=torch.uint8)
    proof = []
    if bufs:
        dst = torch.empty(chunk, dtype=torch.uint8, device="cuda")
        for idx in sorted({0, len(bufs) - 1}):
            h = as_tensor(*bufs[idx])
            nb = h.numel()
            rates = []
            for _ in range(4):
                torch.cuda.synchronize()
                t = time.perf_counter()
                dst[:nb].copy_(h, non_blocking=True)
                torch.cuda.synchronize()
                rates.append(nb / 1e9 / (time.perf_counter() - t))
            proof.append({"chunk": idx, "is_pinned": bool(h.is_pinned()), "h2d_GBps_med": sorted(rates)[2]})
        # pageable reference of the same size
        hq = torch.empty(chunk, dtype=torch.uint8); hq.fill_(1)
        rates = []
        for _ in range(4):
            torch.cuda.synchronize(); t = time.perf_counter()
            dst.copy_(hq, non_blocking=True); torch.cuda.synchronize()
            rates.append(chunk / 1e9 / (time.perf_counter() - t))
        proof.append({"chunk": "pageable_ref", "is_pinned": bool(hq.is_pinned()), "h2d_GBps_med": sorted(rates)[2]})
        del hq, dst

    resp_held = responsiveness(torch)
    samples = []
    for s in range(a.hold):
        time.sleep(1.0)
        samples.append(mem())
    emit({"kind": "held", "status": status, "err": err, "held_gb": held / 2 ** 30, "alloc_sec": t_alloc,
          "mem_full": m_full, "proof": proof, "resp_held": resp_held,
          "hold_avail_min": min(x["avail"] for x in samples) if samples else None,
          "hold_avail_last": samples[-1]["avail"] if samples else None,
          "hold_ws_last": samples[-1]["ws"] if samples else None})
    print(f"[p218] HELD {held / 2 ** 30:.2f}/{a.gb} GB status={status} in {t_alloc:.1f}s  avail "
          f"{m_full['avail']:.2f} GB ws {m_full['ws']:.2f} commit {m_full['commit']:.2f}  proof {proof}  "
          f"resp {resp_held}  hold avail min {min(x['avail'] for x in samples) if samples else None}",
          flush=True)

    t = time.perf_counter()
    for obj, n in bufs:
        if a.api == "alloc":
            cudart.cudaFreeHost(obj)
        else:
            cudart.cudaHostUnregister(obj.data_ptr())
    bufs.clear()
    t_free = time.perf_counter() - t
    time.sleep(3)
    resp_after = responsiveness(torch)
    m_end = mem()
    emit({"kind": "freed", "free_sec": t_free, "mem_end": m_end, "resp_after": resp_after})
    print(f"[p218] freed in {t_free:.1f}s avail {m_end['avail']:.2f} resp {resp_after}", flush=True)
    out.close()


if __name__ == "__main__":
    main()
