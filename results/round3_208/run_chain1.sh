#!/bin/bash
# #208 chain 1: re-time --scan-kernel t on the post-#207 defaults (1B k4 production
# settings, profile_step_202.py, spb 3 so every block position is visited).
#   def = no flags (origin/main 1cbb6bf defaults, scan kernel v1)
#   t   = --scan-kernel t
# 3 alternating repeats, then one CUDA-profiled default run for the leftover-cost hunt.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr3
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/round3_208
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s208_ckpt"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
for r in 1 2 3; do
  run time_def_$r --spb 3
  run time_t_$r --spb 3 --extra "--scan-kernel t"
done
run prof_def --spb 3 --profile
echo "=== chain done $(date)"
