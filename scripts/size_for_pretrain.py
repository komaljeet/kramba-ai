"""Which model size should the next pretrain be? Measured, not estimated.

Author: Rambarun Komaljeet

For each candidate shape, from ~80M up to the ~1.6B ceiling of this card, this
LAUNCHES THE REAL TRAINING SCRIPT (train_edge_conversational.py) for a short
calibration run, then reports measured throughput, measured peak VRAM, and the
days it would take to reach 30 tokens per BODY parameter.

Why launch the real script instead of a harness: every wrong call on
2026-09-29 came from quoting a number from a different setup (a synthetic
32K-vocab bench standing in for GPT-2's 50K in the real loop, a scaled
throughput standing in for a measured one). scripts/size_vs_time_for_code.py
is the harness version of this question; this is the real-loop version.

Config held fixed (the one validated on 2026-09-29, docs/state.md):
    --arch hvn-hybrid, attention at L//4 and 3L//4, tied GPT-2 embeddings,
    ternary, --chunked-loss 512, --grad-checkpoint, Muon, --fast,
    seq 4096, batch 1, grad-accum 8 (32,768 tokens per optimizer step).

Batch size is NOT optimised here. Issue #152 (is batch ever chosen for
anything but "what fits"?) is open. Batch 1 x accum 8 is held constant so the
sizes are comparable, and because batch 2 already fails the 85% guard at
1024x20 without BCD. This tool does not answer #152.

Per candidate, the simplest config that fits is chosen by TRYING, in order:
    plain    -> no BCD (every parameter trains every step; fastest per token
                of learning, see the BCD note below)
    bcd      -> --block-coordinate (balanced partition, the default since !181)
    offload  -> --block-coordinate --offload (layer partition; balanced is not
                supported with offload yet)
A rung "fits" when the run completes with peak reserved VRAM under 85% of the
card (the training script's own spill guard, plus the max over every logged
step). A rung is skipped without launching only when a hard LOWER bound on its
memory (fp32 weights + grads + optimizer state, no activations) is already
over 85% -- launching those just makes WDDM spill for minutes.

How the throughput is measured (and why it is not vacuous):
  * --log-every 1, so every optimizer step is its own CSV row with its own
    tok/s. Step 1 (Triton autotune, allocator warm-up) is dropped.
  * BCD steps are NOT all the same cost: autograd stops at the earliest
    trainable layer, so the embedding block pays a full-depth backward and the
    last group a shallow one. Measuring only the first block (what a 100-step
    run at the real --bcd-steps-per-block 100 does) would understate BCD
    throughput by up to ~1.6x. So calibration uses --bcd-steps-per-block 1 and
    averages over whole rotations: every block is measured equally often, which
    is exactly the long-run mix at any steps-per-block. Block switches without
    offload are free; with offload they move one block over PCIe (~tens of ms
    against multi-second steps).
  * The block count the log reports is checked against the one computed here;
    a mismatch would misalign the rotation window and is reported as an error.
  * Throughput stability: first vs second half of the window must agree within
    15%, otherwise the row is flagged (mid-run spill or thermal throttle).
  * VRAM peak for BCD rungs therefore covers EVERY block, not only the first
    step the training script's guard sees.

BCD caveat that the days column cannot show: under BCD a token only updates the
active block, so BCD needs more tokens than full training for the same loss.
In the 8-layer A/B (results/bcd_partition/ab_d1024_L8.json) full training
reached balanced-BCD's 4000-step loss by ~step 1250 (~3.2x fewer steps) and the
old layer-BCD's by ~step 580 (~6.9x). The table prints the measured days and,
for BCD rows, the days multiplied by that A/B ratio -- labelled as an estimate,
because it was measured at 8 layers, not at these shapes.

Usage (from the repo root; one job on the GPU at a time):
    python -u scripts/size_for_pretrain.py --data-dir data_cache
    python -u scripts/size_for_pretrain.py --only 640x12,768x14   # subset
    python -u scripts/size_for_pretrain.py --report                # reprint table
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

#: (d_model, num_layers). Round shapes, small to the card's ceiling. The first
#: four are 2026-09-29's measured sweep (re-measured here as the harness check);
#: the rest cover BCD-only and offload-only territory up to ~1.6B.
CANDIDATES = [
    (640, 12), (768, 14), (896, 16), (1024, 20),
    (1280, 24), (1536, 24), (2048, 24), (2048, 36),
]

SEQ, BATCH, ACCUM = 4096, 1, 8
TOKENS_PER_STEP = SEQ * BATCH * ACCUM
TOKENS_PER_BODY_PARAM = 30          # #159, per BODY param (vocab excluded)
GUARD_PCT = 85.0
VOCAB = 50257                       # GPT-2; tied, so one table
#: 8L A/B step ratio (full vs BCD at equal loss), see module docstring.
BCD_TOKEN_PENALTY = {"balanced": 3.2, "layer": 6.9}
GIB = 2 ** 30


def attn_layers(L: int) -> tuple[int, int]:
    return L // 4, (3 * L) // 4


def plan_shape(d: int, L: int) -> dict:
    """Param counts and BCD block counts, built on the meta device (no memory)."""
    import torch
    from hvn.edge.model import EdgeHVN
    from hvn.core.block_coordinate import balance_blocks, partition_named_blocks
    a = attn_layers(L)
    plan = ["transformer" if i in a else "deltanet" for i in range(L)]
    with torch.device("meta"):
        m = EdgeHVN(vocab_size=VOCAB, d_model=d, num_layers=L, n_heads=d // 32,
                    block_plan=plan, attn_window=SEQ, tie_embeddings=True)
    total = sum(p.numel() for p in m.parameters())
    embed = m.token_embedding.weight.numel()
    layer_blocks = partition_named_blocks(m)
    sizes = [sum(p.numel() for p in ps) for _, ps in layer_blocks]
    return {"total": total, "body": total - embed, "embed": embed,
            "blocks": {"layer": len(layer_blocks),
                       "balanced": len(balance_blocks(layer_blocks))},
            "largest_block": max(sizes)}


def lower_bound_gib(rung: str, shape: dict) -> float:
    """Memory that rung cannot avoid, ignoring activations. Used only to SKIP
    rungs that are impossible, never to decide that one fits."""
    total, big = shape["total"], shape["largest_block"]
    if rung == "plain":     # fp32 weights + grads + Muon momentum (>= 12 B/param)
        return 12 * total / GIB
    if rung == "bcd":       # all fp32 weights resident + active block grad+state
        return (4 * total + 12 * big) / GIB
    return 12 * big / GIB   # offload: roughly the active block only


def build_cmd(d, L, rung, args, ckpt_dir) -> list[str]:
    a = attn_layers(L)
    cmd = [sys.executable, "-u", os.path.join(REPO, "train_edge_conversational.py"),
           "--data-dir", args.data_dir,
           "--arch", "hvn-hybrid", "--attn-layers", f"{a[0]},{a[1]}",
           "--d-model", str(d), "--num-layers", str(L),
           "--seq-len", str(SEQ), "--batch-size", str(BATCH),
           "--grad-accum", str(ACCUM),
           "--grad-checkpoint", "--chunked-loss", "512",
           "--log-every", "1", "--min-tokens-per-param", "0",
           "--save-every-steps", "1000000", "--keep-checkpoints", "1",
           "--checkpoint-dir", ckpt_dir]
    # --fast, muon, ternary and tied embeddings are the script's defaults.
    if rung in ("bcd", "offload"):
        cmd += ["--block-coordinate", "--bcd-steps-per-block", "1"]
    if rung == "offload":
        cmd += ["--offload"]
    return cmd


def gpu_used_mib() -> int:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0])


def card_gib() -> float:
    out = subprocess.run(["nvidia-smi", "--query-gpu=memory.total",
                          "--format=csv,noheader,nounits"],
                         capture_output=True, text=True, check=True).stdout
    return int(out.strip().splitlines()[0]) / 1024


def parse_run(log_path: str, csv_path: str, n_blocks: int | None) -> dict:
    text = open(log_path, encoding="utf-8", errors="replace").read()
    r: dict = {}
    if "VRAM SPILL" in text:
        r["fail"] = "spill guard (>85% after step 1)"
    elif "OutOfMemoryError" in text or "CUDA out of memory" in text:
        r["fail"] = "CUDA OOM"
    m = re.search(r"total params:\s*([\d,]+)", text)
    if m:
        r["total_logged"] = int(m.group(1).replace(",", ""))
    m = re.search(r"Block-coordinate-descent: (\d+) blocks \((\w+) partition\)", text)
    if m:
        r["blocks_logged"], r["partition"] = int(m.group(1)), m.group(2)
    m = re.search(r"\[heads\].*using (\d+)", text)
    if m:
        r["n_heads"] = int(m.group(1))
    m = re.search(r"\[vram\] peak (\d+)% of card", text)
    if m:
        r["guard_pct"] = int(m.group(1))
    if not os.path.exists(csv_path):
        r.setdefault("fail", "no training_log.csv (crashed before step 1)")
        return r
    rows = list(csv.DictReader(open(csv_path, encoding="utf-8")))
    r["steps_logged"] = len(rows)
    if rows:
        r["peak_alloc_gib"] = max(float(x["peak_vram_gb"]) for x in rows)
        r["reserved_gib"] = max(float(x["reserved_vram_gb"]) for x in rows)
    # Drop step 1 (autotune/warm-up). For BCD keep whole rotations only.
    sec = [TOKENS_PER_STEP / float(x["tokens_sec"]) for x in rows[1:]
           if float(x["tokens_sec"]) > 0]
    if n_blocks:
        sec = sec[: (len(sec) // n_blocks) * n_blocks]
    if len(sec) < max(8, n_blocks or 0):
        r.setdefault("fail", f"too few measured steps ({len(sec)})")
        return r
    r["measured_steps"] = len(sec)
    r["s_per_step"] = sum(sec) / len(sec)
    r["tok_s"] = TOKENS_PER_STEP / r["s_per_step"]
    h = len(sec) // 2
    if n_blocks:
        if len(sec) < 2 * n_blocks:
            r.setdefault("fail", "under 2 rotations measured")
            return r
        h = (len(sec) // n_blocks // 2) * n_blocks
    a, b = statistics.fmean(sec[:h]), statistics.fmean(sec[h:])
    r["half_drift"] = abs(b - a) / min(a, b)
    return r


def run_rung(name, d, L, rung, shape, args, card) -> dict:
    partition = "layer" if rung == "offload" else "balanced"
    n_blocks = shape["blocks"][partition] if rung != "plain" else None
    tag = f"{name}_{rung}"
    ckpt = os.path.join(args.work_dir, tag)
    log_path = os.path.join(args.log_dir, f"{tag}.log")
    rec = {"rung": rung, "log": os.path.relpath(log_path, REPO)}

    lb = lower_bound_gib(rung, shape)
    if lb > card * GUARD_PCT / 100:
        rec["fail"] = (f"skipped: memory lower bound {lb:.2f} GiB > "
                       f"{GUARD_PCT:.0f}% of {card:.2f} GiB")
        print(f"  [{rung}] {rec['fail']}", flush=True)
        return rec

    # Plain: 1 warm-up + 24 measured. BCD: 1 warm-up + at least 2 whole
    # rotations and at least 16 steps (two rotations so the drift check compares
    # like with like: rotation vs rotation, not cheap blocks vs dear ones).
    if rung == "plain":
        steps = 25
    else:
        steps = 1 + max(2, -(-16 // n_blocks)) * n_blocks

    used = gpu_used_mib()
    if used > args.max_idle_mib:
        sys.exit(f"GPU not idle ({used} MiB in use) -- refusing to launch {tag}")
    if os.path.exists(os.path.join(ckpt, "training_log.csv")):
        os.remove(os.path.join(ckpt, "training_log.csv"))
    cmd = build_cmd(d, L, rung, args, ckpt) + ["--max-steps", str(steps)]
    print(f"  [{rung}] launching {steps} steps -> {rec['log']}", flush=True)
    t0 = time.time()
    with open(log_path, "w", encoding="utf-8") as fh:
        fh.write("$ " + " ".join(cmd) + "\n\n")
        fh.flush()
        try:
            rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=REPO,
                                timeout=args.timeout_min * 60).returncode
        except subprocess.TimeoutExpired:
            rc = "timeout"
    rec["wall_min"] = (time.time() - t0) / 60
    rec["returncode"] = rc
    rec.update(parse_run(log_path, os.path.join(ckpt, "training_log.csv"), n_blocks))
    if rc != 0 and "fail" not in rec:
        rec["fail"] = f"exit {rc}"
    # Calibration checkpoints are useless and large (6+ GB at 1.6B).
    for pt in glob.glob(os.path.join(ckpt, "*.pt")):
        os.remove(pt)

    if "fail" not in rec:
        pct = 100 * max(rec.get("reserved_gib", 0), card * rec.get("guard_pct", 0) / 100) / card
        rec["vram_pct"] = pct
        if pct > GUARD_PCT:
            rec["fail"] = f"reserved {pct:.0f}% > {GUARD_PCT:.0f}% at some logged step"
        if n_blocks and rec.get("blocks_logged") != n_blocks:
            rec["fail"] = (f"block count mismatch: log {rec.get('blocks_logged')} vs "
                           f"planned {n_blocks} -- rotation window misaligned")
        if rec.get("total_logged") and rec["total_logged"] != shape["total"]:
            rec["warn_params"] = (f"logged {rec['total_logged']:,} vs planned "
                                  f"{shape['total']:,}")
    status = rec.get("fail") or (f"{rec['tok_s']:,.0f} tok/s, "
                                  f"{rec['vram_pct']:.0f}% VRAM, "
                                  f"drift {rec['half_drift']:.1%}")
    print(f"  [{rung}] {status}  ({rec['wall_min']:.1f} min)", flush=True)
    return rec


def report(results: list[dict], data_tokens: float) -> None:
    hdr = (f"{'shape':<8} {'body':>7} {'total':>7}  {'config':<22} {'tok/s':>6} "
           f"{'VRAM GB (card%)':>16} {'tokens':>7} {'data x':>6} {'days':>6} "
           f"{'BCD-adj est':>11}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in results:
        s = r["shape"]
        ok = r.get("chosen")
        head = f"{r['name']:<8} {s['body']/1e6:>6.0f}M {s['total']/1e6:>6.0f}M  "
        if not ok:
            print(head + "does not fit: " + "; ".join(
                f"{a['rung']}: {a.get('fail')}" for a in r["attempts"]))
            continue
        tokens = TOKENS_PER_BODY_PARAM * s["body"]
        days = tokens / ok["tok_s"] / 86400
        cfg = {"plain": "no BCD", "bcd": f"BCD balanced/{ok.get('blocks_logged')}",
               "offload": f"BCD layer/{ok.get('blocks_logged')}+offload"}[ok["rung"]]
        cfg += f" b{BATCH}x{ACCUM}"
        vram = max(ok["reserved_gib"], ok["vram_pct"] / 100 * r["card_gib"])
        adj = ""
        if ok["rung"] != "plain":
            adj = f"{days * BCD_TOKEN_PENALTY[ok.get('partition', 'layer')]:.0f}"
        flag = " *" if ok["half_drift"] > 0.15 else ""
        print(head + f"{cfg:<22} {ok['tok_s']:>6,.0f} "
              f"{vram:>6.2f} ({ok['vram_pct']:>3.0f}%)    {tokens/1e9:>6.2f}B "
              f"{tokens/data_tokens:>6.2f} {days:>6.1f} {adj:>11}{flag}")
    print(f"\n tokens = {TOKENS_PER_BODY_PARAM} x body params; days = tokens / measured tok/s."
          f"\n data x = tokens / the {data_tokens/1e9:.2f}B-token fineweb_edu.bin cache"
          f" (>1 re-reads data unless more is tokenized)."
          f"\n BCD-adj est = days x {BCD_TOKEN_PENALTY['balanced']} (balanced) or "
          f"x {BCD_TOKEN_PENALTY['layer']} (layer): the 8-layer A/B's extra-steps ratio"
          f"\n   for BCD to reach full training's loss. An ESTIMATE, not measured at these shapes."
          f"\n batch 1 x accum 8 x seq 4096 for all rows (not optimised; #152 open)."
          f"\n * = throughput drifted >15% between halves of the window.")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--data-dir", default="data_cache")
    p.add_argument("--only", default=None, help="comma list of shapes, e.g. 640x12")
    p.add_argument("--out", default="results/pretrain_sizing/results.json")
    p.add_argument("--log-dir", default="logs/pretrain_sizing")
    p.add_argument("--work-dir", default="checkpoints/pretrain_sizing")
    p.add_argument("--timeout-min", type=float, default=180)
    p.add_argument("--max-idle-mib", type=int, default=1000,
                   help="refuse to launch if the GPU already uses more than this")
    p.add_argument("--report", action="store_true", help="reprint the saved table")
    args = p.parse_args()
    for k in ("out", "log_dir", "work_dir", "data_dir"):
        v = getattr(args, k)
        if not os.path.isabs(v):
            setattr(args, k, os.path.join(REPO, v) if k != "data_dir"
                    else os.path.abspath(v))

    bin_path = os.path.join(args.data_dir, "tokenized", "fineweb_edu.bin")
    data_tokens = os.path.getsize(bin_path) / 2 if os.path.exists(bin_path) else float("nan")

    if args.report:
        saved = json.load(open(args.out, encoding="utf-8"))
        report(saved["results"], saved["data_tokens"])
        return 0

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.work_dir, exist_ok=True)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    card = card_gib()
    wanted = set(args.only.split(",")) if args.only else None
    results = []
    if os.path.exists(args.out) and wanted:
        # --only appends to / replaces rows of an earlier run.
        results = [r for r in json.load(open(args.out, encoding="utf-8"))["results"]
                   if r["name"] not in wanted]

    for d, L in CANDIDATES:
        name = f"{d}x{L}"
        if wanted and name not in wanted:
            continue
        shape = plan_shape(d, L)
        print(f"\n=== {name}: body {shape['body']/1e6:.1f}M / total "
              f"{shape['total']/1e6:.1f}M, attn {attn_layers(L)}, blocks "
              f"layer {shape['blocks']['layer']} / balanced "
              f"{shape['blocks']['balanced']} ===", flush=True)
        entry = {"name": name, "d_model": d, "num_layers": L,
                 "attn_layers": attn_layers(L), "shape": shape,
                 "card_gib": card, "attempts": [], "chosen": None}
        for rung in ("plain", "bcd", "offload"):
            rec = run_rung(name, d, L, rung, shape, args, card)
            entry["attempts"].append(rec)
            if "fail" not in rec:
                entry["chosen"] = rec
                break
        results.append(entry)
        results.sort(key=lambda r: r["shape"]["total"])
        with open(args.out, "w", encoding="utf-8") as fh:   # save after each shape
            json.dump({"config": {"seq": SEQ, "batch": BATCH, "grad_accum": ACCUM,
                                  "tokens_per_body_param": TOKENS_PER_BODY_PARAM,
                                  "guard_pct": GUARD_PCT},
                       "data_tokens": data_tokens, "results": results}, fh, indent=2)

    report(results, data_tokens)
    print(f"\nwrote {os.path.relpath(args.out, REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
