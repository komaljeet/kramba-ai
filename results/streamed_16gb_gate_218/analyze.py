"""#218: tables from the pinned sweep and the numerics runs. Run from anywhere."""
import json, glob, os

D = os.path.dirname(os.path.abspath(__file__))


def rows(p):
    return [json.loads(l) for l in open(p, encoding="utf-8") if l.strip()]


print("PINNED HOST MEMORY (one process per size; GiB = 2^30 bytes)")
print(f"{'run':28s} {'asked':>6s} {'held':>6s} {'status':>11s} {'alloc s':>7s} {'avail after GB':>14s} "
      f"{'H2D GB/s':>8s} {'spawn ms':>8s} {'touch ms':>8s}")
for p in sorted(glob.glob(os.path.join(D, "pinned_*.jsonl")) + glob.glob(os.path.join(D, "shared_counter_*.jsonl"))):
    r = rows(p)
    st = next(x for x in r if x["kind"] == "start")
    h = next((x for x in r if x["kind"] == "held"), None)
    if h is None:
        continue
    pr = [x for x in h["proof"] if x["chunk"] != "pageable_ref"]
    print(f"{os.path.basename(p)[:-6]:28s} {st['gb']:6.2f} {h['held_gb']:6.2f} {h['status']:>11s} "
          f"{h['alloc_sec']:7.2f} {h['mem_full']['avail']:14.2f} "
          f"{min(x['h2d_GBps_med'] for x in pr):8.2f} {h['resp_held']['spawn_med_s'] * 1000:8.1f} "
          f"{h['resp_held']['touch512MB_s'] * 1000:8.1f}   (before: spawn "
          f"{st['resp_before']['spawn_med_s'] * 1000:.1f}, touch {st['resp_before']['touch512MB_s'] * 1000:.1f}, "
          f"avail {st['mem_ctx']['avail']:.2f})")

for p in (sorted(glob.glob(os.path.join(D, "numerics_lr*.jsonl")), key=lambda s: -float(s.split("lr")[-1][:-6]))
          + sorted(glob.glob(os.path.join(D, "control_lr*.jsonl")), key=lambda s: -float(s.split("lr")[-1][:-6]))):
    print(f"\n### {os.path.basename(p)}")
    if "control" in p:
        other = p.replace("control_", "numerics_")
        if os.path.exists(other):
            ra = [x for x in rows(p) if x["kind"] == "step" and x["arm"] == "ref"]
            rb = [x for x in rows(other) if x["kind"] == "step" and x["arm"] == "ref"]
            same = all(x.get("train_loss") == y.get("train_loss") and x.get("heldout") == y.get("heldout")
                       for x, y in zip(ra, rb)) and len(ra) == len(rb)
            print(f"  determinism: ref in this process == ref in {os.path.basename(other)}: {same}")
    r = rows(p)
    setup = r[0]
    lr = setup["muon_lr"]
    steps = [x for x in r if x["kind"] == "step"]
    cums = [x for x in r if x["kind"] == "cum"]
    attr = next((x for x in r if x["kind"] == "attr"), None)
    arms = setup["arms"]
    n = setup["steps"]
    print(f"\nNUMERICS muon lr {lr} ({n} steps, 4 train batches cycled, held-out batch never trained)")
    if attr:
        print(f"  step-0 grad error from bf16 park alone: {attr['park_bf16_only']['grad_rel']:.4f} "
              f"(cos {attr['park_bf16_only']['grad_cos']:.6f}); fp32 park repeated: "
              f"{attr['park_fp32_repeat']['grad_rel']:.1e} (determinism)")
    ref_rec = next((x.get("ref_recompute_vs_forward_maxabs") for x in steps
                    if x["arm"] == "ref" and x["step"] == 0), None)
    print(f"  ref: recompute vs forward max|diff| at step 0 = {ref_rec}")
    drift = [(x["step"], x["ref_drift_vs_step0"]) for x in steps if "ref_drift_vs_step0" in x]
    print(f"  ref grad on batch 0 vs step 0 (natural drift): " + ", ".join(f"s{s} {v:.3f}" for s, v in drift))
    held = {a: {x["step"]: x["heldout"] for x in steps if x["arm"] == a and "heldout" in x} for a in arms}
    hs = sorted(held["ref"])
    r0, rN = held["ref"][hs[0]], held["ref"][hs[-1]]
    print(f"  held-out loss by step: " + "  ".join(f"s{s}" for s in hs))
    for a in arms:
        print(f"    {a:8s} " + "  ".join(f"{held[a][s]:.6f}" for s in hs)
              + f"   gap at end {held[a][hs[-1]] - rN:+.2e} = "
              f"{(held[a][hs[-1]] - rN) / (r0 - rN) * 100 if r0 != rN else float('nan'):+.2f}% of ref's drop "
              f"({r0 - rN:.2e})")
    print(f"  {'arm':8s} {'kept/step':>9s} {'min':>6s} {'SR noise':>8s} {'grad_rel':>8s} "
          f"{'cum_err':>7s} {'cum_kept':>8s} {'train loss end':>14s} {'cast ms/step':>12s}")
    for a in arms:
        ss = [x for x in steps if x["arm"] == a and "kept" in x]
        cc = [x for x in cums if x["arm"] == a]
        k = [x["kept"] for x in ss]
        gr = [x.get("grad_rel", 0.0) for x in ss]
        print(f"  {a:8s} {sum(k) / len(k):9.4f} {min(k):6.3f} {sum(x['noise'] for x in ss) / len(ss):8.3f} "
              f"{sum(gr) / len(gr):8.4f} {cc[-1]['cum_err']:7.4f} {cc[-1]['cum_kept']:8.4f} "
              f"{ss[-1]['train_loss']:14.6f} {sum(x['cast_ms'] for x in ss) / len(ss):12.1f}")
    print("  cum_err by step: " + "; ".join(
        f"{a} " + ",".join(f"{x['cum_err']:.3f}" for x in cums if x["arm"] == a and x["step"] % 4 == 3)
        for a in arms if a != "ref"))
    done = next((x for x in r if x["kind"] == "done"), None)
    if done:
        print(f"  peak VRAM alloc {done['peak_alloc']:.2f} GB, host ws end {done['host_end']['ws']:.2f} GB")
    print(f"  cast cost per layer (GPU, 42M matrix values): {setup['cast_cost']}")
