"""#217: summarize the timed runs and compare with the #216 estimate.

Usage: python analyze.py  (reads the *.jsonl next to this file, prints the tables)
"""
import json, os, statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
WARMUP = 2
PRED = {"fwd": 0.553, "refwd": 0.423, "bwd": 1.917, "optim": 0.521}   # #216 section 5
PRED_GROUP = 3.41
# Not measured here; #216 took them from #213's measured phases (phases_default.txt).
REST = {"loss + embedding (trainable decoder)": 2.19, "AdamW embedding": 0.05, "clip, glue": 0.03}
BCD = 9.89
STEPS = 925_000


def rows(name, kind="step"):
    p = os.path.join(HERE, name + ".jsonl")
    if not os.path.exists(p):
        return []
    return [r for r in (json.loads(l) for l in open(p)) if r.get("kind") == kind]


def med(xs):
    return st.median(xs) if xs else float("nan")


def spread(xs):
    return f"{med(xs):.3f} (min {min(xs):.3f}, max {max(xs):.3f}, n {len(xs)})"


def layer_phase(rs, cat, L):
    return med([r["phases"].get(f"{cat}.L{L}", 0.0) for r in rs])


def full_step(rs):
    """24 layers from the measured per-layer times (DeltaNet = mean of the group's
    DeltaNet layers, attention = layer 6), as #216 counts them:
    forward 22 DN + 2 attn, re-forward 16 DN + 2 attn, backward and optimizer 22 + 2.
    Per-layer stalls are counted with their layers. Per-group extras (park, seg-in
    stall, tail) are counted 6 times."""
    dn, at = (4, 5, 7), 6
    out = {}
    for cat, ndn, nat, dnl in (("fwd", 22, 2, dn), ("refwd", 16, 2, (4, 5)),
                               ("bwd", 22, 2, dn), ("optim", 22, 2, dn)):
        for c in (cat, cat + ".stall"):
            v_dn = st.mean(layer_phase(rs, c, L) for L in dnl)
            v_at = layer_phase(rs, c, at)
            out[c] = ndn * v_dn + nat * v_at
    per_group = med([r["cats"].get("fwd.park", 0) + r["cats"].get("refwd.stall.seg", 0)
                     + r["cats"].get("tail", 0) for r in rs])
    out["per-group extras x6 (park, seg-in wait, copy tail)"] = 6 * per_group
    for k, v in REST.items():
        out[k + " [#213, not measured here]"] = v
    return out


def main():
    print("=== copy rates (copybench, 168 MB = one layer's fp32 master) ===")
    for r in rows("copybench", "copy"):
        print(f"  {r['label']:16s} per-copy GPU GB/s {[round(x, 2) for x in r['GBps_each_gpu']]}"
              f"  total by wall {r['GBps_total_wall']:.2f} GB/s"
              f"  (wall {r['wall_min'] * 1e3:.1f}-{r['wall_max'] * 1e3:.1f} ms)")
    for r in rows("copybench", "copy_under_gemm"):
        for k, v in r.items():
            if k != "kind":
                print(f"  {k:12s} 60 GEMMs {v['gemm_sec_med'] * 1e3:.1f} ms  copy GB/s {v['copy_GBps_med']}")

    print("\n=== falsifier ===")
    for name in ("check", "mustfail_skip_d2h", "mustfail_skip_h2d"):
        for r in rows(name, "check"):
            print(f"  {name:18s} step {r['step']} {'PASS' if r['pass'] else 'FAIL'}"
                  f" out_rel {r['out_rel']:.2e} dx_rel {r['dx_rel']:.2e}"
                  f" grad_rel max {max(r['grad_rel'].values()):.2e}"
                  f" upd_rel {({k: round(v, 4) for k, v in r['upd_rel'].items()})}"
                  f" gpu==host {all(r['gpu_eq_host'].values())} hash_changed {all(r['hash_changed'].values())}"
                  f" w_identical {all(r['w_identical'].values())}")
        for r in rows(name, "check_done"):
            print(f"  {name:18s} OVERALL {'PASS' if r['pass'] else 'FAIL'}  scan {r['scan_t_stats']}")

    for name in ("time_overlap_1", "time_overlap_2", "time_sync", "time_pageable"):
        rs = rows(name)[WARMUP:]
        if not rs:
            continue
        setup = rows(name, "setup")[0]
        print(f"\n=== {name} ({len(rs)} timed steps after {WARMUP} warm-up) ===")
        print(f"  group step, GPU timeline, until last copy lands: {spread([r['gpu_total'] for r in rs])} s"
              f"   vs #216 {PRED_GROUP} s -> x{med([r['gpu_total'] for r in rs]) / PRED_GROUP:.2f}")
        print(f"  group step, compute stream done:                 {spread([r['compute_end'] for r in rs])} s")
        print(f"  host wall:                                       {spread([r['wall'] for r in rs])} s")
        cats = sorted({k for r in rs for k in r["cats"]})
        for c in cats:
            v = med([r["cats"].get(c, 0.0) for r in rs])
            p = PRED.get(c)
            print(f"    {c:22s} {v:7.3f} s" + (f"   #216 {p:.3f} (x{v / p:.2f})" if p else ""))
        for L in (4, 5, 6, 7):
            print(f"    layer {L}: " + "  ".join(
                f"{c} {layer_phase(rs, c, L):.3f}" for c in ("fwd", "refwd", "bwd", "optim",
                                                             "fwd.stall", "refwd.stall",
                                                             "bwd.stall", "optim.stall")))
        tags = sorted({k for r in rs for k in r["copies"]})
        for t in tags:
            gb = med([r["copies"][t]["bytes"] / 1e9 for r in rs if t in r["copies"]])
            sec = med([r["copies"][t]["sec"] for r in rs if t in r["copies"]])
            print(f"    copy {t:14s} {gb:6.3f} GB  {sec:6.3f} s  {gb / sec if sec else 0:6.2f} GB/s")
        print(f"  H2D {med([r['h2d_bytes'] for r in rs]) / 1e9:.2f} GB busy {med([r['h2d_busy'] for r in rs]):.3f} s;"
              f" D2H {med([r['d2h_bytes'] for r in rs]) / 1e9:.2f} GB busy {med([r['d2h_busy'] for r in rs]):.3f} s;"
              f" both at once {med([r['h2d_d2h_overlap'] for r in rs]):.3f} s;"
              f" H2D under compute {med([r['h2d_under_compute'] for r in rs]):.3f} s;"
              f" D2H under compute {med([r['d2h_under_compute'] for r in rs]):.3f} s")
        print(f"  peak VRAM allocated {max(r['peak_alloc'] for r in rs):.2f} GB, reserved "
              f"{max(r['peak_res'] for r in rs):.2f} GB of {rs[0]['total_mem']:.2f} "
              f"({max(r['peak_res'] for r in rs) / rs[0]['total_mem']:.0%})")
        print(f"  host: state {setup['host_state_bytes'] / 1e9:.2f} GB pinned={setup['pinned']};"
              f" ws start {setup['host_start']['ws']:.2f} built {setup['host_built']['ws']:.2f}"
              f" state {setup['host_state']['ws']:.2f} after steps {rs[-1]['host_after']['ws']:.2f}"
              f" (peak {rs[-1]['host_after']['peak_ws']:.2f}, private {rs[-1]['host_after']['private']:.2f});"
              f" system free {setup['host_start']['sys_avail']:.2f} -> {rs[-1]['host_after']['sys_avail']:.2f} GB")
        fs = full_step(rs)
        tot = sum(fs.values())
        print(f"  full 1B step from these per-layer times: {tot:.2f} s ({tot / BCD:.2f}x BCD 9.89 s), "
              f"{STEPS * tot / 86400:.0f} days for 925,000 steps   (#216 likely 21.83 s)")
        for k, v in fs.items():
            print(f"      {k:55s} {v:6.2f}")


if __name__ == "__main__":
    main()
