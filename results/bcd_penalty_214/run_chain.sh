#!/bin/bash
# #214: BCD penalty vs full backprop, d832 x 24 layers (208.7M), production flags, grad_accum 8.
# d832 is the largest 24-layer size measured where full backprop fits cleanly (probe/).
# Timing (probe/): full 6.68 s/step, BCD k4 4.0 s/step (mean over the 7 blocks).
# 4000 steps: full ~7.5 h, BCD ~4.5 h. 16 h cap => ONE seed per arm at 4000 steps (cut from 2),
# seed 1 for both arms only to step 500 (seed-spread check), bcd spb 200 only to step 2000,
# spb 50 only if it still ends before the cap. All arms share the 4000-step schedule
# (warmup 400 = 10%, cosine to 0 at 4000), so --stop-at runs are comparable at equal steps.
# Data order pinned (shuffle seed 0); same 64 held-out windows; seed = model init only.
cd /c/Users/komal/Documents/Codes/kramba-ai-wt214
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/bcd_penalty_214
export AB214_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s214_ckpt"
DEADLINE=$(( 1794994496 + 16 * 3600 + 1800 ))
EVAL=1,100,200,300,400,500,600,700,1000,1500,2000,2500,3000,3500,4000
COMMON="--d-model 832 --n-heads 26 --steps 4000 --warmup 400 --eval-at $EVAL --eval-windows 64 --save-every 500"

gpu_idle() {
  for i in $(seq 1 60); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
    if [ "$used" -lt 1000 ]; then echo "gpu idle ($used MiB) $(date)"; return 0; fi
    echo "gpu busy ($used MiB), waiting $(date)"; sleep 60
  done
  echo "GPU never went idle -- aborting chain $(date)"; exit 1
}
last_step() { tail -1 $O/ab/steps_$1.jsonl 2>/dev/null | sed -n 's/.*"step": \([0-9]*\).*/\1/p'; }

ab() {
  tag=$1; target=$2; shift 2
  for att in 0 1 2 3; do
    log=$O/ab_$tag.log; [ $att -gt 0 ] && log=$O/ab_${tag}_r$att.log
    gpu_idle
    echo "=== $tag attempt $att $(date) -> $log"
    $PY -u scripts/ab_bcd_penalty_214.py --tag $tag --out $O/ab $COMMON "$@" > $log 2>&1
    rc=$?; ls=$(last_step $tag)
    echo "exit $rc last step ${ls:-none} $(date)"
    if [ "${ls:-0}" -ge "$target" ]; then rm -rf "$AB214_CKPT/ckpt_$tag"; return 0; fi
    echo "!!! $tag stopped at step ${ls:-none} of $target -- resuming"
  done
  echo "!!! $tag failed 4 times, moving on $(date)"
}
fits() { [ $(( $(date +%s) + $1 )) -lt $DEADLINE ]; }

ab bcd100_s0 4000 --mode bcd --spb 100 --seed 0
ab full_s0 4000 --mode full --seed 0
ab bcd100_s1 500 --mode bcd --spb 100 --seed 1 --stop-at 500
ab full_s1 500 --mode full --seed 1 --stop-at 500
if fits 8700; then ab bcd200_s0 2000 --mode bcd --spb 200 --seed 0 --stop-at 2000
else echo "=== bcd200 SKIPPED (cap) $(date)"; fi
if fits 8700; then ab bcd50_s0 2000 --mode bcd --spb 50 --seed 0 --stop-at 2000
else echo "=== bcd50 SKIPPED (cap) $(date)"; fi
echo "=== chain done $(date)"
