"""scripts/sweep_looped_early_exit.py — N/T hyperparameter search for Issue #122.

Sweeps max_loops N x entropy_threshold T on an EXISTING checkpoint (no
retraining), measuring near/far synthetic recall, avg loops/token on the
easy/hard TinyStories-style buckets, peak VRAM, and tokens/sec -- all
through hvn.edge.looped's forward_step-based incremental decode.

IMPORTANT scope note: eval/synthetic_recall_probe.py's own generate()
re-forwards the whole sequence every step via model(x) (full forward()),
NOT forward_step(). docs/state.md documents that forward() and
forward_step() use different scan implementations
(_scan_chunked vs _scan_sequential) that MEASURABLY DIVERGE on a trained
checkpoint. Since the loop wrapper hooks forward_step specifically (to keep
that contract exact, per the Issue #122 spec), this script implements its
own forward_step-based generation harness rather than reusing that script's
generate() -- and always includes an N=1 (no-loop) run through this SAME
harness as the true baseline, rather than assuming the previously-published
35.5%/8.5% full-forward()-path numbers transfer unchanged to the
forward_step path.

Usage (fast feasibility pass, small n-trials):
    python scripts/sweep_looped_early_exit.py \
        --checkpoint checkpoints/hybrid_tooluse_v1.pt \
        --n-trials 30 --gen-tokens-for-speed 40 \
        --output results/looped_early_exit/sweep_results.json

Re-run with --n-trials 300 on a single --only-n/--only-t combo for the
final validation-gate numbers once a candidate is chosen.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402
from eval.grounded_qa import set_attention_window, EOS             # noqa: E402
from eval.synthetic_recall_probe import build_prompt, whole_word_hit  # noqa: E402
from eval.loop_exit_probe import run_loop_exit_probe, BUNDLED_SAMPLE_TEXT  # noqa: E402
from hvn.edge.looped import enable_looped_early_exit, pop_loop_log  # noqa: E402

DEFAULT_N_GRID = [2, 4, 8, 16]
DEFAULT_T_GRID = [0.1, 0.3, 0.5, 0.7]

# Baseline numbers from docs/state.md, measured via the DIFFERENT
# (full-forward()) generation path -- reported alongside, not compared
# against directly. See module docstring.
DOC_BASELINE_NEAR = 35.5
DOC_BASELINE_FAR = 8.5


@torch.no_grad()
def generate_incremental(model, tok, device, prompt, max_new=20, temp=0.4, top_p=0.9, rep_pen=1.3):
    """Same sampling recipe as eval/synthetic_recall_probe.py's generate(),
    but through forward_step (so the loop wrapper actually engages) instead
    of a full model(x) re-forward every step."""
    ids = tok.encode(prompt)
    states = model.init_state(1, device=device)
    pos = 0
    logits_step = None
    for t in ids:
        cur = torch.tensor([[t]], device=device)
        logits_step, states = model.forward_step(cur, states, position_idx=pos)
        pos += 1

    out: list[int] = []
    next_logits = logits_step[0, -1].float()
    for _ in range(max_new):
        logits = next_logits.clone()
        for tkn in set(out):
            logits[tkn] /= rep_pen
        probs = F.softmax(logits / temp, dim=-1)
        sp, si = torch.sort(probs, descending=True)
        cum = torch.cumsum(sp, dim=-1)
        sp[cum - sp > top_p] = 0
        sp = sp / sp.sum()
        nxt = int(si[torch.multinomial(sp, 1)])
        if nxt == EOS:
            break
        out.append(nxt)
        cur = torch.tensor([[nxt]], device=device)
        logits_step, states = model.forward_step(cur, states, position_idx=pos)
        pos += 1
        next_logits = logits_step[0, -1].float()
    return tok.decode(out).strip()


def measure_recall(model, tok, device, n_trials, seed, n_distractors, probe_window):
    if probe_window is not None:
        set_attention_window(model, probe_window)
    else:
        set_attention_window(model, None)  # restore full window between conditions

    rng = torch.Generator().manual_seed(seed)
    hits = 0
    for _ in range(n_trials):
        prompt, correct = build_prompt(rng, n_distractors)
        answer = generate_incremental(model, tok, device, prompt)
        hits += whole_word_hit(answer, correct)
    return hits / n_trials


def measure_speed_and_vram(model, tok, device, n_tokens):
    prompt = "<|user|>\nSay a short sentence.\n<|assistant|>\n"
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    generate_incremental(model, tok, device, prompt, max_new=n_tokens, temp=0.7, top_p=0.95, rep_pen=1.0)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0
    peak_vram_mb = (torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                    if device.type == "cuda" else None)
    tokens_per_sec = n_tokens / elapsed if elapsed > 0 else None
    return tokens_per_sec, peak_vram_mb


def evaluate_combo(model, tok, device, N, T, n_trials, gen_tokens_for_speed, seed):
    enable_looped_early_exit(model, max_loops=N, entropy_threshold=T)

    near = measure_recall(model, tok, device, n_trials, seed=seed, n_distractors=0, probe_window=None)
    far = measure_recall(model, tok, device, n_trials, seed=seed + 1, n_distractors=16, probe_window=128)
    set_attention_window(model, None)  # restore for the speed/loop-count measurement below

    pop_loop_log(model)  # clear anything accumulated during recall measurement
    with torch.no_grad():
        loop_result = run_loop_exit_probe(model, tok, device, BUNDLED_SAMPLE_TEXT)

    tokens_per_sec, peak_vram_mb = measure_speed_and_vram(model, tok, device, gen_tokens_for_speed)

    return {
        "N": N, "T": T,
        "near_window_recall_pct": round(near * 100, 2),
        "far_window_recall_pct": round(far * 100, 2),
        "avg_loops_overall": loop_result["avg_loops_overall"],
        "avg_loops_easy": loop_result["avg_loops_easy"],
        "avg_loops_hard": loop_result["avg_loops_hard"],
        "n_easy": loop_result["n_easy"],
        "n_hard": loop_result["n_hard"],
        "peak_vram_mb": round(peak_vram_mb, 1) if peak_vram_mb is not None else None,
        "tokens_per_sec": round(tokens_per_sec, 2) if tokens_per_sec is not None else None,
    }


def apply_gates(record, baseline_tokens_per_sec):
    """Issue #122 target criteria, evaluated against the record's own
    measured N=1 baseline (tokens/sec) -- NOT the doc's full-forward()
    numbers, see module docstring."""
    N = record["N"]
    checks = {
        "near_recall_ge_30": record["near_window_recall_pct"] >= 30.0,
        "far_recall_ge_7": record["far_window_recall_pct"] >= 7.0,
        "avg_loops_below_half_N": (record["avg_loops_overall"] is not None
                                   and record["avg_loops_overall"] < N / 2),
        "tokens_per_sec_improves": (baseline_tokens_per_sec is not None
                                    and record["tokens_per_sec"] is not None
                                    and record["tokens_per_sec"] > baseline_tokens_per_sec),
    }
    checks["all_pass"] = all(checks.values())
    return checks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--n-trials", type=int, default=30,
                    help="recall trials per (N,T,near/far) cell -- default is a fast "
                         "feasibility pass; use 300 for the final validation gate")
    ap.add_argument("--gen-tokens-for-speed", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-grid", type=int, nargs="+", default=DEFAULT_N_GRID)
    ap.add_argument("--t-grid", type=float, nargs="+", default=DEFAULT_T_GRID)
    ap.add_argument("--output", default="results/looped_early_exit/sweep_results.json")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    model, step, _ = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    print(f"Loaded {args.checkpoint} (step {step}) on {device}")

    # Baseline: N=1 (T irrelevant -- always exits after the first, only, loop).
    print("\n=== Baseline N=1 (no looping) ===")
    baseline = evaluate_combo(model, tok, device, N=1, T=1.1, n_trials=args.n_trials,
                              gen_tokens_for_speed=args.gen_tokens_for_speed, seed=args.seed)
    print(json.dumps(baseline, indent=2))

    records = [baseline]
    for N, T in itertools.product(args.n_grid, args.t_grid):
        print(f"\n=== N={N} T={T} ===")
        rec = evaluate_combo(model, tok, device, N=N, T=T, n_trials=args.n_trials,
                             gen_tokens_for_speed=args.gen_tokens_for_speed, seed=args.seed)
        rec["gates"] = apply_gates(rec, baseline["tokens_per_sec"])
        print(json.dumps(rec, indent=2))
        records.append(rec)

    passing = [r for r in records if r.get("gates", {}).get("all_pass")]
    passing.sort(key=lambda r: (-r["near_window_recall_pct"], r["avg_loops_overall"] or 1e9))
    recommendation = passing[0] if passing else None

    payload = {
        "checkpoint": args.checkpoint,
        "step": step,
        "n_trials_per_cell": args.n_trials,
        "doc_baseline_full_forward_path": {
            "near_window_recall_pct": DOC_BASELINE_NEAR,
            "far_window_recall_pct": DOC_BASELINE_FAR,
            "note": "measured via model(x) full forward(), NOT forward_step -- "
                    "see module docstring; the 'baseline' record above (N=1) is "
                    "the apples-to-apples comparison for this sweep",
        },
        "results": records,
        "recommendation": recommendation,
    }

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote {args.output}")

    if recommendation:
        print(f"\nRecommended: N={recommendation['N']} T={recommendation['T']}")
    else:
        print("\nNo (N,T) combination passed all four gates.")


if __name__ == "__main__":
    main()
