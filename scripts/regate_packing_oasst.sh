#!/usr/bin/env bash
# Issue #132 follow-up: does packing hold on OASST chat data (ratio 1.45x)?
#
# The grounded-QA re-gate passed at 5.18x. OASST is the OTHER end of the length
# distribution -- median 359 tokens, >=10% at the 512 cap -- so packing reclaims
# much less (1.45x) and perturbs gradient composition much less. This measures
# it rather than inferring it from the grounded result.
#
# Run through continue_ft_hybrid.py with --chat-mix 1.0 (NOT
# finetune_edge_conversational.py) on purpose: that script only enables the
# Triton scan under --pack, so an A/B there would confound packing with a change
# of scan path. continue_ft_hybrid.py has triton+megakernel on unconditionally in
# both arms, leaving --pack as the only difference.
#
# --chat-mix 1.0 gives the grounded corpus zero sampling weight, so this is
# effectively pure OASST while reusing the identical harness as the first re-gate.
#
# Arms matched on REAL TOKENS: packed steps = padded steps / 1.45.
set -u

PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python
TRUNK=/c/Users/komal/Documents/Codes/kramba-ai/checkpoints/edge_chinchilla83m/edge_conversational_step101000.pt
DATA=data/grounded_qa_expanded.jsonl     # present but weighted to 0 by --chat-mix 1.0
OUT=results/packing
mkdir -p "$OUT" logs

PADDED_STEPS=2000
PACKED_STEPS=1379                         # 2000 / 1.45 == same passes over OASST

echo "=== [$(date)] OASST ARM 1/2: PADDED (control), ${PADDED_STEPS} steps ==="
"$PY" continue_ft_hybrid.py \
  --hybrid-trunk-init "$TRUNK" --grounded-data "$DATA" --chat-mix 1.0 \
  --max-steps "$PADDED_STEPS" --save-every 1000 \
  --checkpoint-dir checkpoints/regate_oasst_padded --fast
echo "=== [$(date)] OASST ARM 1 done ==="

echo "=== [$(date)] OASST ARM 2/2: PACKED, ${PACKED_STEPS} steps (same real tokens) ==="
"$PY" continue_ft_hybrid.py \
  --hybrid-trunk-init "$TRUNK" --grounded-data "$DATA" --chat-mix 1.0 --pack \
  --max-steps "$PACKED_STEPS" --save-every 1000 \
  --checkpoint-dir checkpoints/regate_oasst_packed --fast
echo "=== [$(date)] OASST ARM 2 done ==="

for ARM in padded packed; do
  CKPT=$(ls -t checkpoints/regate_oasst_${ARM}/*.pt 2>/dev/null | head -1)
  if [ -z "$CKPT" ]; then echo "!! no checkpoint for ${ARM} -- skipping probes"; continue; fi
  echo "=== [$(date)] OASST PROBE ${ARM} (near) on ${CKPT} ==="
  "$PY" -m eval.synthetic_recall_probe --checkpoint "$CKPT" --n-trials 300 \
    2>&1 | tee "${OUT}/regate_oasst_${ARM}_near.txt"
  echo "=== [$(date)] OASST PROBE ${ARM} (far) ==="
  "$PY" -m eval.synthetic_recall_probe --checkpoint "$CKPT" --n-trials 300 \
    --probe-window 128 --n-distractors 16 \
    2>&1 | tee "${OUT}/regate_oasst_${ARM}_far.txt"
done

echo "=== [$(date)] OASST RE-GATE COMPLETE ==="

# Persist the raw artifacts BEFORE the machine goes down -- there will be no
# session left to write them up, so at minimum the numbers must survive.
git add -A "$OUT" scripts/regate_packing_oasst.sh 2>&1
git commit -q -m "Results: OASST packing re-gate raw artifacts (issue #132 follow-up)

Ratio 1.45x on OASST (vs 5.18x grounded-QA). Both arms via continue_ft_hybrid.py
--chat-mix 1.0 so triton+megakernel scan is on in BOTH -- --pack is the only
difference (finetune_edge_conversational.py would have confounded packing with a
scan-path change).

Raw probe output only; interpretation and docs/state.md writeup still pending.

Co-Authored-By: Claude Opus 5 <noreply@anthropic.com>" 2>&1
git push -q origin issue-132-pack-default 2>&1 | tail -3
echo "=== [$(date)] ARTIFACTS PUSHED ==="

echo "=== [$(date)] SHUTTING DOWN in 120s -- run 'shutdown /a' to abort ==="
shutdown //s //t 120 //c "kramba-ai OASST re-gate complete"
