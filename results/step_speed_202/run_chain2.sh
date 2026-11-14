#!/bin/bash
# #202 chain 2: active no-checkpoint 2 and 3 layers (on top of B), and the backward falsifier.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtspeed
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/step_speed_202
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run time_B_A2 --spb 3 --no-frozen-fused --set _active_no_ckpt=2 --abort-peak 0.86
run time_B_A3 --spb 3 --no-frozen-fused --set _active_no_ckpt=3 --abort-peak 0.86
run hash_B_falsify --hash --spb 2 --no-frozen-fused --perturb-grad 9.5367431640625e-07
