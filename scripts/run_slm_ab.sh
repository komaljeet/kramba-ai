#!/usr/bin/env bash
# Rho-1 selective-token A/B, plus the downstream probe that decides it.
#
# Val loss alone cannot settle this: the selective arm deliberately trains on a
# hard subset, so its loss over ALL tokens can be flat or worse while downstream
# quality improves -- and the paper's general-domain result was +6.8% on downstream
# benchmarks, NOT a perplexity win. So both arms get eval/synthetic_recall_probe.py
# at the 300-trial bar, and the verdict reads both numbers.
set -u

MT=/c/Users/komal/Documents/Codes/kramba-ai
PY=$MT/.venv/Scripts/python
STEPS=1500

mkdir -p logs results/slm

echo "=== [$(date)] SLM A/B: ${STEPS} steps/arm, baseline vs top-60% tokens ==="
"$PY" -u scripts/ab_selective_tokens.py \
  --steps $STEPS --val-every 250 --val-batches 20 \
  --data-dir $MT/data_cache \
  --reference $MT/checkpoints/hybrid_chat_v1.pt \
  --target-init $MT/checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt \
  --out results/slm/ab_selective_tokens.json
echo "=== [$(date)] TRAINING done ==="

for ARM in baseline selective; do
  CK="checkpoints/slm_${ARM}/final.pt"
  [ -f "$CK" ] || { echo "!! missing $CK"; continue; }
  echo "=== [$(date)] PROBE ${ARM} (near-window, 300 trials) ==="
  "$PY" -u -m eval.synthetic_recall_probe --checkpoint "$CK" --n-trials 300 \
    2>&1 | tee "results/slm/probe_${ARM}_near.txt"
done

echo "=== [$(date)] SLM A/B COMPLETE ==="
