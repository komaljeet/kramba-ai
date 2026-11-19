#!/bin/bash
# #217: the whole GPU session, one process and one log per run, nothing else on the GPU.
# Order: copy rates, falsifier (check + two must-fail partners), then timed runs.
W=/c/Users/komal/Documents/Codes/kramba-ai-wtsgb
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=$W/results/streamed_group_bench_217
cd $W
gpu_free() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')
  if [ "$used" -gt 400 ]; then echo "GPU NOT EMPTY ($used MiB used) - abort"; exit 9; fi
  echo "gpu used ${used} MiB"
}
run() { tag=$1; shift; gpu_free; echo "=== $tag $(date)"; $PY -u scripts/bench_streamed_group_217.py "$@" --out $O/$tag.jsonl > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run copybench --copybench
run check --check 3
run mustfail_skip_d2h --check 3 --break skip_d2h
run mustfail_skip_h2d --check 3 --break skip_h2d
run time_overlap_1 --mode overlap --steps 7
run time_sync --mode sync --steps 7
run time_pageable --mode pageable --steps 7
run time_overlap_2 --mode overlap --steps 7
echo "=== all done $(date)"
