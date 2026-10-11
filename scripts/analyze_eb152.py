"""Grade the #152 effective-batch sweep, and check the harness first.

Author: Rambarun Komaljeet

Reads the trainer's per-step CSVs and ``eval/heldout_loss.py`` JSONs written by
``scripts/run_eb152_sweep.sh``. The verdict is held-out NLL only -- training
loss is never graded (#132).

Checks, each of which fails loudly rather than letting a vacuous run through:

1. **Repro cell** (replaces #152's original known-answer cell, which pointed at
   #148's 6.03552 -- a different model, corpus, tokenizer, scorer and length,
   so it could never pass). Two seeded 40-step ga=8 runs must give the same
   per-step loss sequence and the same held-out NLL.
2. **Prefix cell:** the real ga=8 arm's first 40 losses must equal the repro
   run's. During warmup the LR does not depend on --max-steps, so this ties
   the 3,052-step arm itself to the repro cell -- a sequence, not one step.
3. **Data identity across arms:** this trainer's first optimizer step runs at
   LR 0 (warmup scale 0/W), so a ga=16 arm's step-1 loss is the mean of ga=8's
   steps 1 and 2 at the same init weights, and ga=32's step 1 is the mean of
   ga=16's steps 1 and 2. That holds only if every arm reads the same
   micro-batches in the same order -- the matched-token claim, measured.
4. **The flag does something:** ga=8 and ga=16 loss sequences must differ.
5. **Vacuity guards:** every arm logged every step, saw exactly the same token
   count, and was scored on the same held-out token count.
"""

from __future__ import annotations

import csv
import json
import os
import sys

REPO = sys.argv[1] if len(sys.argv) > 1 else "."
CK = os.path.join(REPO, "checkpoints")
RES = os.path.join(REPO, "results", "effective_batch")

ARMS = {"ga8": (8, 3052), "ga16": (16, 1526), "ga32": (32, 763)}
TOKENS = 24416 * 4096                 # 100,007,936 -- identical in every arm
#: #159: step-to-step held-out wobble along ONE run (not seed-to-seed variance).
NOISE_LO, NOISE_HI = 0.0035, 0.0053
#: Data-identity tolerance: the logged loss is a sum of per-micro-batch means in
#: a different grouping, so it carries float rounding, nothing more.
ID_TOL = 1e-4

failures: list[str] = []


def losses(name: str) -> list[tuple[int, float]]:
    with open(os.path.join(CK, f"eb152_{name}", "training_log.csv"),
              encoding="utf-8") as fh:
        return [(int(r["step"]), float(r["loss"])) for r in csv.DictReader(fh)]


def heldout(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)["rows"][-1]


def check(ok: bool, msg: str) -> None:
    print(("  PASS  " if ok else "  FAIL  ") + msg)
    if not ok:
        failures.append(msg)


def main() -> int:
    out: dict = {"issue": 152, "tokens_per_arm": TOKENS, "checks": {}, "arms": {}}

    print("[1] repro cell: two seeded 40-step ga=8 runs")
    a, b = losses("repro_a"), losses("repro_b")
    check(len(a) == 40 and len(b) == 40, f"both logged 40 steps ({len(a)}, {len(b)})")
    max_d = max(abs(x[1] - y[1]) for x, y in zip(a, b))
    check(max_d == 0.0, f"loss sequences identical (max |diff| {max_d:.2e})")
    ha = heldout(os.path.join(RES, "repro_a.json"))
    hb = heldout(os.path.join(RES, "repro_b.json"))
    dh = abs(ha["heldout_nll"] - hb["heldout_nll"])
    check(dh == 0.0, f"held-out NLL identical ({ha['heldout_nll']:.6f} vs "
                     f"{hb['heldout_nll']:.6f}, diff {dh:.2e})")
    # Guard: the repro cell must have moved off init, or "identical" is trivial.
    check(a[-1][1] < a[0][1] - 0.5, f"repro run actually trained "
                                    f"({a[0][1]:.3f} -> {a[-1][1]:.3f})")
    out["checks"]["repro"] = {"max_loss_diff": max_d, "heldout_diff": dh,
                              "heldout_nll": ha["heldout_nll"]}

    present = [n for n in ARMS
               if os.path.exists(os.path.join(RES, f"final_{n}.json"))]
    if not present:
        print("\n(no finished arms yet)")
        return 1 if failures else 0

    L = {n: losses(n) for n in present}
    print("\n[2] prefix cell: real ga=8 arm vs repro run, first 40 steps")
    if "ga8" in L:
        pd = max(abs(x[1] - y[1]) for x, y in zip(L["ga8"][:40], a))
        check(pd == 0.0, f"ga8 arm reproduces the repro sequence (max |diff| {pd:.2e})")
        out["checks"]["prefix_max_diff"] = pd

    print("\n[3] data identity across arms (step 1 runs at LR 0)")
    for big, small in (("ga16", "ga8"), ("ga32", "ga16")):
        if big in L and small in L:
            want = (L[small][0][1] + L[small][1][1]) / 2
            got = L[big][0][1]
            check(abs(got - want) < ID_TOL,
                  f"{big} step 1 = mean({small} steps 1,2): {got:.6f} vs {want:.6f}")
            out["checks"][f"identity_{big}"] = got - want

    print("\n[4] the flag does something")
    if "ga8" in L and "ga16" in L:
        n = min(len(L["ga8"]), len(L["ga16"]))
        diff = sum(1 for i in range(n) if L["ga8"][i][1] != L["ga16"][i][1])
        check(diff > n // 2, f"ga8 and ga16 loss sequences differ ({diff}/{n} steps)")

    print("\n[5] vacuity guards")
    for n in present:
        ga, steps = ARMS[n]
        rows = L[n]
        check([s for s, _ in rows] == list(range(1, steps + 1)),
              f"{n}: logged steps 1..{steps} with none missing ({len(rows)} rows)")
        check(steps * ga * 4096 == TOKENS, f"{n}: {steps} x {ga} x 4096 = {TOKENS:,} tokens")
        f = heldout(os.path.join(RES, f"final_{n}.json"))
        check(f["tokens_scored"] == 398970,
              f"{n}: scored on the pinned held-out set ({f['tokens_scored']:,} tokens)")
        check(f["step"] == steps, f"{n}: final checkpoint is step {f['step']}")
        out["arms"][n] = {"grad_accum": ga, "optimizer_steps": steps,
                          "tokens_per_step": ga * 4096, "tokens": TOKENS,
                          "final_heldout_nll": f["heldout_nll"],
                          "final_heldout_ppl": f["heldout_ppl"]}
        tp = os.path.join(RES, f"traj_{n}.json")
        if os.path.exists(tp):
            with open(tp, encoding="utf-8") as fh:
                out["arms"][n]["trajectory"] = [
                    {"tokens": r["tokens_seen"], "heldout_nll": r["heldout_nll"]}
                    for r in json.load(fh)["rows"]]

    print("\n=== held-out NLL at 100,007,936 tokens, fixed muon_lr=0.02 ===")
    base = out["arms"].get("ga8", {}).get("final_heldout_nll")
    print(f"{'arm':>6}{'rows/step':>11}{'tokens/step':>13}{'opt steps':>11}"
          f"{'held-out':>11}{'vs ga8':>10}  reading")
    for n, r in out["arms"].items():
        d = None if base is None else r["final_heldout_nll"] - base
        if d is None or n == "ga8":
            reading = "baseline (current default)"
        elif abs(d) < 2 * NOISE_LO:
            reading = "noise (under 2x the low floor)"
        elif abs(d) < 2 * NOISE_HI:
            reading = "borderline (between 2x low and 2x high floor)"
        else:
            reading = ("REAL: " + ("worse" if d > 0 else "better"))
        r["delta_vs_ga8"] = d
        r["reading"] = reading
        print(f"{n:>6}{r['grad_accum']:>11}{r['tokens_per_step']:>13,}"
              f"{r['optimizer_steps']:>11}{r['final_heldout_nll']:>11.5f}"
              f"{'' if d is None else f'{d:+.5f}':>10}  {reading}")
    print(f"\nnoise floor (#159, within-run step-to-step): +/-{NOISE_LO}-{NOISE_HI}; "
          f"'real' needs > {2 * NOISE_HI:.4f}")

    out["failures"] = failures
    with open(os.path.join(RES, "sweep.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {os.path.join(RES, 'sweep.json')}")
    if failures:
        print(f"\n*** {len(failures)} CHECK(S) FAILED -- the sweep is not trustworthy:")
        for f in failures:
            print("   - " + f)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
