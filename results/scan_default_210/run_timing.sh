#!/bin/bash
# #210 timing check: new default (scan kernel t) vs the --scan-kernel v1 opt-out,
# 1B k4 production settings, profile_step_202.py, spb 3 (all 7 block positions),
# 3 alternating repeats.
cd /c/Users/komal/Documents/Codes/kramba-ai-wt210
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/scan_default_210
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s210_ckpt"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
for r in 1 2 3; do
  run time_def_$r --spb 3
  run time_v1_$r --spb 3 --extra "--scan-kernel v1"
done
echo "=== chain done $(date)"
