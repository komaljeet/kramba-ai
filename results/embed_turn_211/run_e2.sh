#!/bin/bash
# #211 E2: the #209 harness at short turns. v1 (the kernel that showed the slow turns),
# 10 steps per block, eval (8 windows) + empty_cache after every turn as in #209,
# 150 steps = embedding turns at 1-10, 71-80, 141-150. No checkpoint saves.
cd /c/Users/komal/Documents/Codes/kramba-ai-wt211
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/embed_turn_211
export AB209_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s211_ckpt"
EVAL=1$(for s in $(seq 10 10 150); do printf ",%d" $s; done)
run() { tag=$1; shift; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' \r')
  echo "=== $tag gpu used $used MiB $(date)"; rm -rf "$AB209_CKPT/ckpt_$tag"
  $PY -u scripts/repro_embed_turn_211.py --tag $tag --out $O/e2 --steps 150 --spb 10 --warmup 15 \
    --eval-at $EVAL --eval-windows 8 --save-every 100000 "$@" > $O/$tag.log 2>&1
  echo "exit $? $(date)"; }
run e2_v1_s1 --seed 1
echo "=== e2 done $(date)"
