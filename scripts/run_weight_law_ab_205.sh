#!/usr/bin/env bash
# #205 step 4: arms A (fixed 100), B (pre-registered rule), C (fixed = B's mean), seeds 7, 8.
set -u
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
R=results/weight_law_205
S=scripts/exp_weight_law_205.py
gpu_idle() {
  local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
  if [ "$used" -ge 1000 ]; then echo "GPU busy ($used MiB), abort" >> $R/ab_done.log; exit 1; fi
}
run() {  # name, args...
  local n=$1; shift
  gpu_idle
  $PY $S --steps 2800 --heldout-every 50 --out $R/$n "$@" > $R/$n.log 2>&1
  echo "$n exit=$? $(date +%T)" >> $R/ab_done.log
}
echo '{}' > $R/rule_params.json
for seed in ${SEEDS:-7 8}; do
  run A_s$seed --mode fixed --spb 100 --init-seed $seed
  run B_s$seed --mode rule --rule-json $R/rule_params.json --val-every 10 --init-seed $seed
  C=$($PY -c "import json;print(round(json.load(open('$R/B_s7/summary.json'))['mean_turn_len_completed']))")
  run C_s$seed --mode fixed --spb $C --init-seed $seed
done
echo DONE >> $R/ab_done.log
