#!/bin/bash
# #211 E4: cost of --bcd-switch-empty-cache with nothing else on the GPU.
# profile_step_202.py, t kernels, spb 3, 2 rotations, alternating, 2 repeats each
# (plus e3_base / e3_fix_nob, which ran with --hash).
cd /c/Users/komal/Documents/Codes/kramba-ai-wt211
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/embed_turn_211
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s211_ckpt"
run() { tag=$1; shift; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
  echo "=== $tag gpu used $used MiB $(date)"
  $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl --rotations 2 "$@" > $O/$tag.log 2>&1
  echo "exit $? $(date)"; sleep 5; }
for r in 1 2; do
  run e4_off_$r --spb 3 --extra "--scan-kernel t"
  run e4_on_$r  --spb 3 --extra "--scan-kernel t --bcd-switch-empty-cache"
done
echo "=== e4 done $(date)"
