"""#218 step 2: lean host-RAM budget for streamed full backprop at 1B, from exact
parameter counts (results/bcd_memory/ga8_k4.log, #216) and this issue's measurements.
GB = 1e9 bytes. Pinned cap and availability come from the step-1 sweep."""
GB = 1e9
MUON = 1_009_516_544               # 210 matrices
EMB = 102_926_336                  # tied embedding (AdamW)
AUX = 103_543_249 - EMB            # norms / gates / biases in the 24 layers (AdamW), 616,913
MB, S, D = 8, 4096, 2048
GROUPS = 6                         # 24 layers in groups of 4
PARK = MB * S * D                  # values in one parked group input

# measured
PIN_CAP = 6.75 * 2 ** 30 / GB      # hard cudaHostAlloc / cudaHostRegister cap (3 runs, both APIs)
PIN_SAFE = 6.00 * 2 ** 30 / GB     # largest size held with no slowdown, ~0.8 GB under the cap
AVAIL_REST = 8.11                  # lowest "available" before a sweep run (VS Code, Claude, Avast open)
AVAIL_TRIM = 10.12                 # after Windows trimmed idle apps during the sweep (same apps open)
OVH_FLOOR = 1.471                  # overhead.log: torch + trainer import + CUDA + all kernels, no host state
OVH_BENCH = 3.79 - 1.61            # #217 group bench: working set minus its host state
OVH_TRAINER = 3.07                 # #216: trainer working set minus its parked weights (4 workers excluded)

items = {
    "masters, matrices": {"fp32": MUON * 4, "bf16+SR": MUON * 2},
    "masters, non-matrix (fp32)": {"fp32": AUX * 4, "bf16+SR": AUX * 4},
    "Muon momentum": {"fp32": MUON * 4, "bf16": MUON * 2, "int8/256": MUON + MUON / 256 * 4},
    "AdamW moments, non-matrix": {"fp32": AUX * 8},
    "AdamW moments, embedding": {"fp32": EMB * 8, "bf16": EMB * 4, "on GPU": 0},
    "parked group inputs": {"6 x fp32": GROUPS * PARK * 4, "6 x bf16": GROUPS * PARK * 2,
                            "5 x bf16 (group 1 = embedding lookup, recomputed)": (GROUPS - 1) * PARK * 2,
                            "recomputed (none)": 0},
}
for k, v in items.items():
    print(f"{k:30s} " + "  ".join(f"{kk}: {vv / GB:.3f}" for kk, vv in v.items()))

def total(choice):
    return sum(items[k][c] for k, c in choice.items()) / GB

exact = {"masters, matrices": "fp32", "masters, non-matrix (fp32)": "fp32", "Muon momentum": "fp32",
         "AdamW moments, non-matrix": "fp32", "AdamW moments, embedding": "fp32",
         "parked group inputs": "6 x fp32"}
plan = dict(exact, **{"masters, matrices": "bf16+SR", "masters, non-matrix (fp32)": "bf16+SR",
                      "Muon momentum": "bf16",
                      "parked group inputs": "5 x bf16 (group 1 = embedding lookup, recomputed)"})
small = dict(plan, **{"Muon momentum": "int8/256", "AdamW moments, embedding": "bf16"})
print()
for name, ch in (("exact (#216/#217)", exact), ("lean plan", plan), ("smallest (int8 momentum, bf16 emb moments)", small)):
    st = total(ch)
    print(f"{name:44s} pinned state {st:.2f} GB (cap {PIN_CAP:.2f}, safe {PIN_SAFE:.2f}: margin {PIN_SAFE - st:+.2f})")
    for oname, o in (("measured floor", OVH_FLOOR), ("#217 bench", OVH_BENCH), ("trainer today", OVH_TRAINER)):
        t = st + o
        print(f"   + process {oname} {o:.2f} = {t:.2f} GB; margin vs available at rest {AVAIL_REST:.2f}: "
              f"{AVAIL_REST - t:+.2f}; vs after trimming {AVAIL_TRIM:.2f}: {AVAIL_TRIM - t:+.2f}")
