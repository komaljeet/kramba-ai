#!/bin/bash
# #211 E3: once part of the trainer's memory has been pushed to host RAM, does the
# trainer recover by itself when the pressure goes away, or only if its cache is
# emptied? profile_step_202.py, t kernels, spb 3, 2 rotations (14 turns).
#   e3_base      no ballast, --hash                         (reference times + bits)
#   e3_fix_nob   no ballast, --hash, --bcd-switch-empty-cache (cost + same bits?)
#   e3_nofix     ACTIVE 900 MiB VRAM ballast for the first 200 s, then released
#   e3_fix       same ballast, --bcd-switch-empty-cache
cd /c/Users/komal/Documents/Codes/kramba-ai-wt211
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/embed_turn_211
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s211_ckpt"
run() {  # tag ballast-seconds -- profile-args
  tag=$1; bsec=$2; shift 2
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
  echo "=== $tag ballast ${bsec}s gpu used $used MiB $(date)"
  if [ "$bsec" -gt 0 ]; then
    $PY -u scripts/ballast_211.py --vram-mib 900 --vram-touch --touch-sec 0.5 --seconds $bsec > $O/$tag.ballast.log 2>&1 &
    for i in $(seq 1 60); do grep -q holding $O/$tag.ballast.log 2>/dev/null && break; sleep 1; done
    echo "ballast up $(date)"
  fi
  $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl --rotations 2 "$@" > $O/$tag.log 2>&1
  echo "exit $? $(date)"; wait; sleep 5
}
run e3_base    0   --spb 3 --hash --extra "--scan-kernel t"
run e3_fix_nob 0   --spb 3 --hash --extra "--scan-kernel t --bcd-switch-empty-cache"
run e3_nofix   200 --spb 3 --extra "--scan-kernel t"
run e3_fix     200 --spb 3 --extra "--scan-kernel t --bcd-switch-empty-cache"
echo "=== e3 done $(date)"
