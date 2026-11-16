#!/bin/bash
# #211 E1: can a VRAM or host-RAM squeeze reproduce the slow embedding turn?
# Each run = profile_step_202.py, 1B k4 production settings, spb 3 (all 7 block
# positions, embedding turn first), with a ballast process started before the run.
# v1 = the kernel the slow turns were seen with. Monitor CSV runs alongside.
cd /c/Users/komal/Documents/Codes/kramba-ai-wt211
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/embed_turn_211
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s211_ckpt"
run() {  # tag ballast-args -- profile-args
  tag=$1; bal=$2; shift 2
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
  echo "=== $tag ballast [$bal] gpu used $used MiB $(date)"
  bpid=""
  if [ -n "$bal" ]; then
    $PY -u scripts/ballast_211.py $bal --seconds 1200 > $O/$tag.ballast.log 2>&1 &
    bpid=$!
    for i in $(seq 1 60); do grep -q holding $O/$tag.ballast.log 2>/dev/null && break; sleep 1; done
    cat $O/$tag.ballast.log
  fi
  $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1
  echo "exit $? $(date)"
  [ -n "$bpid" ] && kill $bpid 2>/dev/null; wait $bpid 2>/dev/null
  sleep 5
}
run e1_v1_base     ""                 --spb 3 --extra "--scan-kernel v1"
run e1_v1_vram900  "--vram-mib 900"   --spb 3 --extra "--scan-kernel v1"
run e1_v1_host4000 "--host-mib 4000"  --spb 3 --extra "--scan-kernel v1"
run e1_t_vram900   "--vram-mib 900"   --spb 3 --extra "--scan-kernel t"
echo "=== e1 done $(date)"
