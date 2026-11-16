#!/bin/bash
# #209: long held-out A/B, --scan-kernel t vs default, 1B k4, 100 steps per BCD block.
# 1800 steps (18 turns, ~2.6 passes over the 7 blocks), warmup 180 (10%, as #208),
# cosine to 0 at 1800. Seeds 0/1 = model init only; data order pinned (shuffle seed 0);
# the same 64 held-out windows after step 1 and every turn end (every 100 steps).
# Asked for 2000 steps; cut to 1800 because the #208 per-block step times (default
# 11.15 s, t 9.94 s mean) put 4 x 2000 steps at ~25.2 h of a 26 h hard stop.
# Order interleaved by seed so a stop still leaves a pair. Each run auto-resumes from
# its own checkpoints (every 200 steps) if it dies; each attempt gets its own log.
# Falsifier last, only if it still fits before the deadline: t with the scan backward
# returning zero gradients, 300 steps of the same 1800-step schedule.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr4
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/long_ab_209
export AB209_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s209_ckpt"
DEADLINE=$(date -d "2026-11-16 20:04" +%s)
EVAL=1$(for s in $(seq 100 100 1800); do printf ",%d" $s; done)

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
    $PY -u scripts/ab_heldout_209.py --tag $tag --out $O/ab --steps 1800 --spb 100 \
      --warmup 180 --eval-at $EVAL --eval-windows 64 --save-every 200 "$@" > $log 2>&1
    rc=$?; ls=$(last_step $tag)
    echo "exit $rc last step ${ls:-none} $(date)"
    if [ "${ls:-0}" -ge "$target" ]; then rm -rf "$AB209_CKPT/ckpt_$tag"; return 0; fi
    echo "!!! $tag stopped at step ${ls:-none} of $target -- resuming"
  done
  echo "!!! $tag failed 4 times, moving on $(date)"
}

ab def_s0 1800 --seed 0
ab t_s0 1800 --seed 0 --extra=--scan-kernel=t
ab def_s1 1800 --seed 1
ab t_s1 1800 --seed 1 --extra=--scan-kernel=t
if [ $(( $(date +%s) + 4700 )) -lt $DEADLINE ]; then
  ab tbroken_s0 300 --seed 0 --extra=--scan-kernel=t --break-scan --stop-at 300
else
  echo "=== falsifier SKIPPED: would end past the deadline $(date)"
fi
echo "=== chain done $(date)"
