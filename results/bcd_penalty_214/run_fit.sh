#!/bin/bash
# #214 step 1: largest d (24 layers, head dim 32) where FULL backprop (no BCD, no offload,
# grad checkpoint) fits at batch 1 x seq 4096, grad_accum 8. 5 steps each, held-out eval
# after step 3 so the step after an eval is measured too. One process, one log per size.
cd /c/Users/komal/Documents/Codes/kramba-ai-wt214
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/bcd_penalty_214/probe
export AB214_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s214_ckpt"
for d in "$@"; do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
  [ "$used" -lt 1000 ] || { echo "GPU busy ($used MiB), abort"; exit 1; }
  echo "=== d$d $(date)"
  $PY -u scripts/ab_bcd_penalty_214.py --tag fit_full_d$d --out $O --mode full --d-model $d \
    --n-heads $((d / 32)) --steps 4000 --stop-at 5 --eval-at 3 --save-every 100000 \
    > $O/fit_full_d$d.log 2>&1
  echo "exit $? $(date)"
done
