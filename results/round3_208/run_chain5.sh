#!/bin/bash
# #208 chain 5: the two flags together. 3 alternating repeats, --scan-kernel t alone vs
# --scan-kernel t --fused-norm-residual (profile_step_202.py, spb 3, as chains 1 and 3).
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr3
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/round3_208
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s208_ckpt"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
for r in 1 2 3; do
  run time_t2_$r --spb 3 --extra "--scan-kernel t"
  run time_tfnr_$r --spb 3 --extra "--scan-kernel t --fused-norm-residual"
done
echo "=== chain done $(date)"
