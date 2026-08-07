#!/usr/bin/env bash
# Issue #132 follow-up 2: spend packing's speedup on PASSES instead of wall-clock.
#
# The OASST re-gate matched arms on REAL TOKENS: padded 2000 steps (35.7 min) vs
# packed 1379 steps (24.7 min). Packing won 1.445x wall-clock but lost ~0.06 nats
# of train loss -- most plausibly because matching on tokens also hands the packed
# arm 1.45x FEWER optimizer updates (1379 vs 2000).
#
# This arm removes that confound by taking the speedup as extra passes rather than
# as saved time: packed at the SAME 2000 steps as the padded control.
#
#   rows/step = batch_size 4 * grad_accum 8 = 32
#   padded  2000 steps -> 64,000 rows over 3,465 OASST examples  = 18.5 passes
#   packed  1379 steps -> 44,128 rows over 2,395 packed rows     = 18.4 passes (matched tokens)
#   packed  2000 steps -> 64,000 rows over 2,395 packed rows     = 26.7 passes (THIS ARM)
#
# So this single arm is simultaneously:
#   - equal-WALL-CLOCK vs the padded control (~35.7 min both), and
#   - step-matched vs the padded control (2000 updates both) -- the third arm the
#     re-gate writeup said was needed to separate "fewer updates" from "denser rows
#     are worse steps".
#
# CAVEAT, stated up front: 1.45x more passes over a FIXED corpus is not 1.45x more
# DATA. OASST is 3,465 examples and the control is already at ~18 passes, so this
# tests "more epochs", the only version of the idea available locally. If loss
# improves, packing's speedup converts to quality; if it does not, the cost is
# either real or the corpus is already saturated -- and 26.7 vs 18.5 passes on a
# fixed corpus cannot distinguish those two. Read it accordingly.
#
# Same harness as the re-gate on purpose: continue_ft_hybrid.py --chat-mix 1.0,
# triton+megakernel scan on unconditionally in every arm, so --pack/--max-steps
# are the only differences from the control.
set -u

PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python
TRUNK=/c/Users/komal/Documents/Codes/kramba-ai/checkpoints/edge_chinchilla83m/edge_conversational_step101000.pt
DATA=data/grounded_qa_expanded.jsonl     # present but weighted to 0 by --chat-mix 1.0
OUT=results/packing
CKPT_DIR=checkpoints/regate_oasst_packed_2000
mkdir -p "$OUT" logs

echo "=== [$(date)] PACKED, 2000 steps (equal wall-clock / step-matched vs padded control) ==="
"$PY" continue_ft_hybrid.py \
  --hybrid-trunk-init "$TRUNK" --grounded-data "$DATA" --chat-mix 1.0 --pack \
  --max-steps 2000 --save-every 1000 \
  --checkpoint-dir "$CKPT_DIR" --fast
echo "=== [$(date)] TRAINING done ==="

CKPT=$(ls -t ${CKPT_DIR}/*.pt 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "!! no checkpoint written -- aborting probes"; exit 1; fi

echo "=== [$(date)] PROBE near on ${CKPT} ==="
"$PY" -m eval.synthetic_recall_probe --checkpoint "$CKPT" --n-trials 300 \
  2>&1 | tee "${OUT}/regate_oasst_packed2000_near.txt"

echo "=== [$(date)] PROBE far ==="
"$PY" -m eval.synthetic_recall_probe --checkpoint "$CKPT" --n-trials 300 \
  --probe-window 128 --n-distractors 16 \
  2>&1 | tee "${OUT}/regate_oasst_packed2000_far.txt"

echo "=== [$(date)] EQUAL-WALL-CLOCK ARM COMPLETE ==="
