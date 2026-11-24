"""#200 (A): at 1B k4, is streaming frozen weights over PCIe a stall?

Author: Rambarun Komaljeet

Runs the real ``train_edge_conversational.main`` with the #194 k4 production
settings (``PROD_ARGV``, grad_accum 8, fp8, balanced partition, 4 layers per block,
seed 0, deterministic, LR schedule over 2800 steps) but with SHORT turns
(``--spb``, default 4) so every block position is visited in a few minutes. Turn
length does not change what one step does, only how often the block switches.

Three passes over all 7 blocks, ``--spb`` steps per block each:
  normal   production code path, side-stream prefetch on
  sync     POSITIVE CONTROL 1: backward-recompute prefetch off
           (``model._offload_bwd_prefetch = False``), so every frozen block's
           recompute load is a synchronous pageable copy on the compute stream.
           The stall number must rise by about the step-time rise. (Turning ALL
           prefetch off crashes: see "found on the way" in the #200 docs entry --
           the tail load without prefetch fetches final_norm only, and the
           decoder bias stays on the CPU.)
  slow     POSITIVE CONTROL 2: every prefetch also copies its own size again
           (pinned 16 MiB scratch, repeated) on the side stream before its
           ready-event, i.e. the side-stream copy takes ~2x as long. Same weights.

In each turn: step 1 carries the block switch (timed separately around
``_activate``), step 2 is unprofiled (profiler-overhead reference), the rest are
profiled with CUDA-only torch.profiler. From the device timeline of a profiled step:
  compute stream     the stream with the most kernel time
  busy               union of compute-stream kernels/memsets
  h2d_on_compute     HtoD memcpy ON the compute stream (sync load misses)
  copy_wait          compute-stream idle time overlapping an in-flight side-stream
                     HtoD copy, in gaps that END within 2 ms of a side copy ending
                     (the compute stream resumed because the copy finished)
  copy_wait_loose    same overlap, any gap (upper bound)
  idle_other         remaining compute-stream idle (host-bound: Python, launches,
                     host staging memcpy, optimizer host work, data)
  stall              h2d_on_compute + copy_wait
Bytes and achieved GB/s come from the trace's memcpy records (bytes per record).
Host-side time inside GPUBufferPool.load / prefetch is also accumulated per step.

Output: ``<out>/stall_steps_<tag>.jsonl`` (one row per step), ``<out>/switch_<tag>.jsonl``.
"""
import os, sys, json, time, argparse

MAIN = r"C:\Users\komal\Documents\Codes\kramba-ai"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def union_len(iv):
    tot, cur_s, cur_e = 0.0, None, None
    for s, e in sorted(iv):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                tot += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        tot += cur_e - cur_s
    return tot


def merge(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def overlap(a0, a1, merged):
    t = 0.0
    for s, e in merged:
        if e <= a0:
            continue
        if s >= a1:
            break
        t += min(a1, e) - max(a0, s)
    return t


def analyse_trace(path, resume_tol_us=2000.0):
    d = json.load(open(path))
    acts = [e for e in d["traceEvents"]
            if e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset") and "dur" in e]
    if not acts:
        return None
    ktime = {}
    for e in acts:
        if e["cat"] == "kernel":
            ktime[e["args"]["stream"]] = ktime.get(e["args"]["stream"], 0) + e["dur"]
    cs = max(ktime, key=ktime.get)
    t0 = min(e["ts"] for e in acts)
    t1 = max(e["ts"] + e["dur"] for e in acts)
    comp_all, comp_busy, side_h2d = [], [], []
    h2d_comp = h2d_comp_bytes = side_bytes = side_dur = 0.0
    other_copy = {}
    for e in acts:
        s, en, st = e["ts"], e["ts"] + e["dur"], e["args"]["stream"]
        is_h2d = e["cat"] == "gpu_memcpy" and "HtoD" in e["name"]
        if st == cs:
            comp_all.append((s, en))
            if is_h2d:
                h2d_comp += e["dur"]; h2d_comp_bytes += e["args"].get("bytes", 0)
            elif e["cat"] != "gpu_memcpy":
                comp_busy.append((s, en))
            else:
                other_copy[e["name"]] = other_copy.get(e["name"], 0) + e["dur"]
        elif is_h2d:
            side_h2d.append((s, en)); side_bytes += e["args"].get("bytes", 0); side_dur += e["dur"]
    cm = merge(comp_all)
    sm = merge(side_h2d)
    side_ends = sorted(e for _, e in sm)
    gaps = [(t0, cm[0][0])] if cm and cm[0][0] > t0 else []
    gaps += [(cm[i][1], cm[i + 1][0]) for i in range(len(cm) - 1)]
    if cm and cm[-1][1] < t1:
        gaps.append((cm[-1][1], t1))
    import bisect
    cw = cwl = idle = 0.0
    for g0, g1 in gaps:
        idle += g1 - g0
        ov = overlap(g0, g1, sm)
        cwl += ov
        if ov > 0:
            j = bisect.bisect_left(side_ends, g0)
            if j < len(side_ends) and side_ends[j] <= g1 and g1 - side_ends[j] <= resume_tol_us:
                cw += ov
    us = 1e-6
    return {"window": (t1 - t0) * us, "compute_stream": cs, "busy": union_len(comp_busy) * us,
            "h2d_on_compute": h2d_comp * us, "h2d_on_compute_gb": h2d_comp_bytes / 1e9,
            "side_h2d_gb": side_bytes / 1e9, "side_h2d_dur": side_dur * us,
            "side_h2d_union": union_len(side_h2d) * us,
            "idle": idle * us, "copy_wait": cw * us, "copy_wait_loose": cwl * us,
            "idle_other": (idle - cw) * us, "stall": (h2d_comp + cw) * us,
            "gbps_side": side_bytes / (side_dur * us) / 1e9 if side_dur else None,
            "gbps_compute": h2d_comp_bytes / (h2d_comp * us) / 1e9 if h2d_comp else None,
            "other_copy_on_compute": {k: v * us for k, v in other_copy.items()},
            "n_kernels": sum(1 for e in acts if e["cat"] == "kernel")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--spb", type=int, default=4)
    ap.add_argument("--modes", default="normal,sync,slow")
    ap.add_argument("--tag", required=True, help="output file suffix")
    ap.add_argument("--trace-dir", required=True, help="scratch dir for chrome traces (deleted after parse)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True); os.makedirs(a.trace_dir, exist_ok=True)
    sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
    os.chdir(REPO)
    import torch
    from torch.optim.optimizer import register_optimizer_step_post_hook
    from torch.profiler import profile, ProfilerActivity
    import hvn.core.block_coordinate as bc
    from profile_bcd_memory import PROD_ARGV
    import train_edge_conversational as T
    assert os.path.abspath(bc.__file__).startswith(REPO), bc.__file__

    modes = a.modes.split(",")
    nb = 7
    total = len(modes) * nb * a.spb
    SCHED, MODEL = [], []
    H = {"load_hit": 0.0, "load_miss": 0.0, "prefetch": 0.0, "n_miss": 0, "n_pref": 0,
         "miss_bytes": 0, "pref_bytes": 0}
    MODE = {"m": modes[0]}
    SLOW = {}

    orig_init = bc.BlockCoordinateScheduler.__init__

    def init(self, model, optimizer, *x, **k):
        orig_init(self, model, optimizer, *x, **k)
        SCHED.append(self); MODEL.append(model)
        assert len(self.blocks) == nb, [n for n, _ in self.blocks]
    bc.BlockCoordinateScheduler.__init__ = init

    sw = open(os.path.join(a.out, f"switch_{a.tag}.jsonl"), "w")
    orig_act = bc.BlockCoordinateScheduler._activate

    def activate(self, idx):
        if not SCHED:          # constructor's first activation: not a switch
            return orig_act(self, idx)
        torch.cuda.synchronize(); t = time.perf_counter()
        r = orig_act(self, idx)
        torch.cuda.synchronize()
        sw.write(json.dumps({"after_step": st["n"], "to": self.blocks[idx][0],
                             "activate_sec": time.perf_counter() - t,
                             "mode": MODE["m"]}) + "\n"); sw.flush()
        return r
    bc.BlockCoordinateScheduler._activate = activate

    P = bc.GPUBufferPool
    o_load, o_pref = P.load, P.prefetch

    def nbytes(pool, params):
        return sum(pool._cpu_home[id(q)].numel() * pool._cpu_home[id(q)].element_size()
                   for q in pool._offloadable(params))

    def load(self, key, params):
        hit = key in self._resident
        if not hit:
            self.register_cpu_home(params)
            H["miss_bytes"] += nbytes(self, params); H["n_miss"] += 1
        t = time.perf_counter()
        r = o_load(self, key, params)
        H["load_hit" if hit else "load_miss"] += time.perf_counter() - t
        return r

    def prefetch(self, key, params, behind=False):
        fresh = self._prefetch_stream is not None and key not in self._resident
        t = time.perf_counter()
        r = o_pref(self, key, params, behind=behind)
        if fresh and key in self._pending_events:
            b = nbytes(self, params)
            H["pref_bytes"] += b; H["n_pref"] += 1
            if MODE["m"] == "slow":
                # Positive control 2: copy the same number of bytes again on the
                # side stream, then move the ready-event behind it.
                if not SLOW:
                    SLOW["h"] = torch.empty(8 << 20, dtype=torch.bfloat16).pin_memory()
                    SLOW["d"] = torch.empty(8 << 20, dtype=torch.bfloat16, device="cuda")
                chunk = SLOW["h"].numel() * 2
                with torch.cuda.stream(self._prefetch_stream):
                    for _ in range(max(1, round(b / chunk))):
                        SLOW["d"].copy_(SLOW["h"], non_blocking=True)
                    self._pending_events[key].record(self._prefetch_stream)
        H["prefetch"] += time.perf_counter() - t
        return r
    P.load, P.prefetch = load, prefetch

    rows = open(os.path.join(a.out, f"stall_steps_{a.tag}.jsonl"), "w")
    st = {"n": 0, "t": None, "prof": None}

    def begin(n_next):
        # profile the 3rd.. steps of each turn
        pos = (n_next - 1) % a.spb
        H.update({k: 0 for k in H})
        if pos >= 2:
            p = profile(activities=[ProfilerActivity.CUDA])
            p.__enter__()
            st["prof"] = p
        torch.cuda.synchronize()
        st["t"] = time.perf_counter()

    def post(opt, args, kw):
        torch.cuda.synchronize()
        sec = time.perf_counter() - st["t"]
        n = st["n"] + 1
        s = SCHED[0]
        row = {"step": n, "sec": sec, "block": s.active_block_name, "mode": MODE["m"],
               "pos_in_turn": (n - 1) % a.spb + 1, "profiled": st["prof"] is not None,
               "host": dict(H), "lr": [g["lr"] for g in opt.param_groups],
               "peak_res": torch.cuda.max_memory_reserved()}
        if st["prof"] is not None:
            p = st["prof"]; st["prof"] = None
            p.__exit__(None, None, None)
            tp = os.path.join(a.trace_dir, f"s{n}.json")
            t = time.perf_counter()
            p.export_chrome_trace(tp)
            row["trace"] = analyse_trace(tp)
            row["parse_sec"] = time.perf_counter() - t
            os.remove(tp)
        rows.write(json.dumps(row) + "\n"); rows.flush()
        tr = row.get("trace")
        print(f"[s200] step {n} {MODE['m']} {s.active_block_name[:14]} pos {row['pos_in_turn']} "
              f"{sec:.2f}s" + (f" stall {tr['stall']:.2f}s idle_other {tr['idle_other']:.2f}s "
                               f"side {tr['side_h2d_gb']:.1f}GB @{tr['gbps_side'] or 0:.1f}GB/s"
                               if tr else ""), flush=True)
        st["n"] = n
        if n >= total:
            rows.close(); sw.close(); sys.stdout.flush(); os._exit(0)
        # switch modes at rotation boundaries (after the scheduler's own switch,
        # which happens right after this hook; the mode applies to the next step)
        m = modes[n // (nb * a.spb)]
        if m != MODE["m"]:
            MODE["m"] = m
        MODEL[0]._offload_bwd_prefetch = (m != "sync")
        torch.cuda.reset_peak_memory_stats()
        begin(n + 1)
    register_optimizer_step_post_hook(post)

    # The first begin() must run before the trainer's first forward: a one-shot
    # forward pre-hook on the model.
    started = {"done": False}

    def first_fwd(mod, inp):
        if not started["done"]:
            started["done"] = True
            MODEL[0]._offload_bwd_prefetch = (MODE["m"] != "sync")
            begin(1)
    orig_init2 = bc.BlockCoordinateScheduler.__init__

    def init2(self, model, optimizer, *x, **k):
        orig_init2(self, model, optimizer, *x, **k)
        model.register_forward_pre_hook(first_fwd)
    bc.BlockCoordinateScheduler.__init__ = init2

    sys.argv = ["train_edge_conversational.py"] + list(PROD_ARGV) + [
        "--grad-accum", "8", "--bcd-steps-per-block", str(a.spb),
        "--bcd-layers-per-block", "4",
        "--max-steps", "2800", "--val-fraction", "0.001", "--val-every", "100000000",
        "--save-every-steps", "100000",
        "--checkpoint-dir", os.path.join(a.out, "ckpt_stall"),
        "--data-dir", os.path.join(MAIN, "data_cache"), "--seed", "0",
        "--deterministic"]
    T.main()


if __name__ == "__main__":
    main()
