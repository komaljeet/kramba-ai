"""eval/loop_exit_probe.py — measures average exit-loop count per token (Issue #122).

Runs a checkpoint, wrapped with hvn.edge.looped's entropy-gated early exit,
token-by-token over a held-out text sample via forward_step (the incremental-
decode path the loop wrapper hooks into). Buckets each token position as
"easy" or "hard" by whether the model's own greedy prediction from the
PREVIOUS step matched this actual token (a predictable continuation is an
"easy" token; a surprising one is "hard") and reports the average loop count
per bucket, plus an ASCII visualization (no plotting dependency).

Note on the text sample: this repo has no TinyStories loader wired up (see
docs/-- eval/curriculum.py's own pattern of falling back to a small bundled
offline sample when a real corpus isn't available locally). Rather than add
a new download dependency for this probe, a small hand-written set of
TinyStories-STYLE short sentences is bundled below (BUNDLED_SAMPLE_TEXT) --
simple, common-word-heavy children's-story prose, the same register as the
real corpus, but not the real downloaded dataset. Pass --text to use a real
local file instead.

Usage:
    python -m eval.loop_exit_probe --checkpoint checkpoints/hybrid_tooluse_v1.pt \
        --max-loops 8 --entropy-threshold 0.3 --output results/looped_early_exit/loop_probe_N8_T0.3.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402
from hvn.edge.looped import enable_looped_early_exit, pop_loop_log  # noqa: E402

BUNDLED_SAMPLE_TEXT = """
Once upon a time, there was a little girl named Lily. She loved to play in
the sunny garden with her dog. One day, Lily found a small red ball under
the big oak tree. She picked it up and threw it for her dog to catch. The
dog ran fast and caught the ball in his mouth. Lily laughed and clapped her
hands. "Good boy!" she said. They played all afternoon until the sun began
to set. Lily's mother called her inside for dinner. "Coming, Mom!" Lily
shouted. She gave her dog a big hug before going inside. That night, Lily
dreamed about all the fun games she would play tomorrow. The next morning,
she woke up early and ran to the window. It was raining outside, so Lily
and her dog stayed inside and built a fort out of blankets and pillows.
They pretended to be brave explorers searching for hidden treasure. Under
the couch, they found an old toy boat that Lily had forgotten about. She
was so happy to find it again. Her dog barked with excitement too.
""".strip()


def run_loop_exit_probe(model, tok, device, text: str) -> dict:
    """Feed `text` through forward_step token-by-token, recording loops-used
    per position and whether the model's PRIOR prediction matched this token
    (its "easy"/"hard" label). Returns a dict with per-bucket stats and the
    raw per-position series.
    """
    ids = tok.encode(text)
    if len(ids) < 2:
        raise ValueError("sample text too short to probe")

    states = model.init_state(1, device=device)
    cur = torch.tensor([[ids[0]]], device=device)
    logits, states = model.forward_step(cur, states, position_idx=0)
    pop_loop_log(model)  # discard the seed-token loop count (no "prediction" to grade yet)

    per_position = []  # list of {"pos", "loops", "easy", "token_id"}
    prev_pred = int(torch.argmax(logits[0, -1]).item())

    for pos in range(1, len(ids)):
        actual = ids[pos]
        easy = (prev_pred == actual)

        cur = torch.tensor([[actual]], device=device)
        logits, states = model.forward_step(cur, states, position_idx=pos)
        loops = pop_loop_log(model)[0]

        per_position.append({"pos": pos, "loops": loops, "easy": easy, "token_id": actual})
        prev_pred = int(torch.argmax(logits[0, -1]).item())

    easy_loops = [p["loops"] for p in per_position if p["easy"]]
    hard_loops = [p["loops"] for p in per_position if not p["easy"]]
    all_loops = [p["loops"] for p in per_position]

    def _avg(xs):
        return (sum(xs) / len(xs)) if xs else None

    return {
        "n_positions": len(per_position),
        "n_easy": len(easy_loops),
        "n_hard": len(hard_loops),
        "avg_loops_overall": _avg(all_loops),
        "avg_loops_easy": _avg(easy_loops),
        "avg_loops_hard": _avg(hard_loops),
        "per_position": per_position,
    }


def render_ascii(result: dict, max_loops: int) -> str:
    lines = []
    lines.append(f"Per-position loop count (E=easy H=hard), 1 char = 1 loop, max={max_loops}:")
    for p in result["per_position"]:
        tag = "E" if p["easy"] else "H"
        bar = "#" * p["loops"]
        lines.append(f"  [{tag}] pos={p['pos']:>4} loops={p['loops']:>2} {bar}")
    lines.append("")
    lines.append(f"avg loops overall = {result['avg_loops_overall']:.2f}  "
                 f"(easy n={result['n_easy']} avg={_fmt(result['avg_loops_easy'])}, "
                 f"hard n={result['n_hard']} avg={_fmt(result['avg_loops_hard'])})")
    return "\n".join(lines)


def _fmt(x):
    return f"{x:.2f}" if x is not None else "n/a"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--max-loops", type=int, default=8)
    ap.add_argument("--entropy-threshold", type=float, default=0.3)
    ap.add_argument("--text", default=None, help="path to a local text file (default: bundled sample)")
    ap.add_argument("--output", default=None, help="JSON output path")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from transformers import GPT2Tokenizer
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    model, step, _ = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    enable_looped_early_exit(model, max_loops=args.max_loops, entropy_threshold=args.entropy_threshold)

    text = open(args.text, encoding="utf-8").read() if args.text else BUNDLED_SAMPLE_TEXT

    with torch.no_grad():
        result = run_loop_exit_probe(model, tok, device, text)

    print(render_ascii(result, args.max_loops))

    if args.output:
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        payload = {
            "checkpoint": args.checkpoint, "step": step,
            "max_loops": args.max_loops, "entropy_threshold": args.entropy_threshold,
            **result,
        }
        with open(args.output, "w") as f:
            json.dump(payload, f, indent=2)
        print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
