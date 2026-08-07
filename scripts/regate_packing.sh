#!/usr/bin/env bash
# Issue #132 recall re-gate: padded vs packed FT from the same warm start,
# matched on REAL TOKENS (same passes over the corpus), then the 300-trial
# synthetic recall probe on both.
#
# The two arms are identical except for --pack. Matching real tokens (not
# optimizer steps) is the point: packing's whole claim is "same training, less
# wall-clock", so the packed arm takes 5.18x fewer steps over the same data.
# That also means the packed arm gets 5.18x FEWER optimizer updates -- which is
# exactly the risk being measured, not a flaw in the design.
#
# Deviations from the historical hybrid_a4_v2 recipe, both deliberate:
#   - trunk is edge_conversational_step101000.pt (the step51000 that a4_v2 used
#     is not present in this checkout), so absolute numbers are NOT comparable
#     to the doc's 35.5%/8.5% -- which is why BOTH arms are measured fresh here.
#   - --chat-mix 0 (pure grounded-QA): keeps the packing ratio exactly 5.18x and
#     drops the OASST cache dependency. Identical in both arms.
set -u

PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python
TRUNK=/c/Users/komal/Documents/Codes/kramba-ai/checkpoints/edge_chinchilla83m/edge_conversational_step101000.pt
DATA=data/grounded_qa_expanded.jsonl     # 5,240 examples -- NOT the 40-example default
OUT=results/packing
mkdir -p "$OUT" logs

PADDED_STEPS=5000
PACKED_STEPS=965                          # 5000 / 5.18 == same passes over the corpus

echo "=== [$(date)] ARM 1/2: PADDED (control), ${PADDED_STEPS} steps ==="
"$PY" continue_ft_hybrid.py \
  --hybrid-trunk-init "$TRUNK" \
  --grounded-data "$DATA" \
  --chat-mix 0 \
  --max-steps "$PADDED_STEPS" \
  --save-every 1000 \
  --checkpoint-dir checkpoints/regate_padded \
  --fast
echo "=== [$(date)] ARM 1 done ==="

echo "=== [$(date)] ARM 2/2: PACKED, ${PACKED_STEPS} steps (same real tokens) ==="
"$PY" continue_ft_hybrid.py \
  --hybrid-trunk-init "$TRUNK" \
  --grounded-data "$DATA" \
  --chat-mix 0 \
  --pack \
  --max-steps "$PACKED_STEPS" \
  --save-every 500 \
  --checkpoint-dir checkpoints/regate_packed \
  --fast
echo "=== [$(date)] ARM 2 done ==="

# --- Probes: near-window and far-window (W=128, 16 distractors), 300 trials ---
for ARM in padded packed; do
  CKPT=$(ls -t checkpoints/regate_${ARM}/*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then
    echo "!! no checkpoint found for arm ${ARM} -- skipping its probes"
    continue
  fi
  echo "=== [$(date)] PROBE ${ARM} (near) on ${CKPT} ==="
  "$PY" -m eval.synthetic_recall_probe --checkpoint "$CKPT" --n-trials 300 \
    2>&1 | tee "${OUT}/regate_${ARM}_near.txt"
  echo "=== [$(date)] PROBE ${ARM} (far, W=128, 16 distractors) ==="
  "$PY" -m eval.synthetic_recall_probe --checkpoint "$CKPT" --n-trials 300 \
    --probe-window 128 --n-distractors 16 \
    2>&1 | tee "${OUT}/regate_${ARM}_far.txt"
done

echo "=== [$(date)] RE-GATE COMPLETE ==="
