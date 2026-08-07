#!/usr/bin/env bash
# Issue #132 follow-up 2: held-out generalization check across all three OASST arms.
#
# Waits for the equal-wall-clock run's probes to release the GPU first -- this repo
# has a documented finding that concurrent GPU work distorts measurements, and a
# 6 GB card cannot host both comfortably.
#
# Two scorings of the same unseen corpus, both paired across arms:
#   answer-span (default) -- the grounded skill itself, ~14K supervised tokens,
#                            but sitting near ln(50257)=10.82 on checkpoints that
#                            never trained it, so high-variance and near-random.
#   --all-tokens          -- plain held-out LM over the whole rendered example,
#                            ~490K tokens: the low-variance instrument for
#                            "generalization or just more memorization".
set -u

PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python
OUT=results/packing
mkdir -p "$OUT" logs

echo "=== [$(date)] waiting for the GPU (equal-wall-clock probes) ==="
until grep -q "EQUAL-WALL-CLOCK ARM COMPLETE" logs/equal_wallclock_oasst.log; do sleep 15; done
echo "=== [$(date)] GPU free -- starting held-out eval ==="

CKPTS="checkpoints/regate_oasst_padded/hybrid_a4_step2000.pt \
       checkpoints/regate_oasst_packed/hybrid_a4_step1379.pt \
       checkpoints/regate_oasst_packed_2000/hybrid_a4_step2000.pt"
LABELS="A_padded_2000 B_packed_1379_tokenmatched C_packed_2000_equalwallclock"

echo "=== [$(date)] SCORING 1/2: all real tokens (held-out LM) ==="
"$PY" -m eval.heldout_grounded_loss --checkpoints $CKPTS --labels $LABELS \
  --batch-size 8 --all-tokens --json-out "${OUT}/heldout_alltokens.json"

echo "=== [$(date)] SCORING 2/2: answer span only (grounded skill) ==="
"$PY" -m eval.heldout_grounded_loss --checkpoints $CKPTS --labels $LABELS \
  --batch-size 8 --json-out "${OUT}/heldout_answerspan.json"

echo "=== [$(date)] HELD-OUT EVAL COMPLETE ==="
