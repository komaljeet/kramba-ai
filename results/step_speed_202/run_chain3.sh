#!/bin/bash
# #202 chain 3: identity of A (active no-ckpt 1 layer) and C (one-pass frozen fp8),
# step-time A/B with a baseline repeat, eval VRAM, final profile.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtspeed
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/step_speed_202
A1="--active-no-ckpt-layers 1"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run hash_B_A1 --hash --spb 2 --no-frozen-fused --extra "$A1"
run hash_B_A1_C --hash --spb 2 --extra "$A1"
run time_B_A1_C --spb 3 --extra "$A1"
run time_B_A1r --spb 3 --no-frozen-fused --extra "$A1"
run time_base3 --spb 3 --loss-recompute --no-frozen-fused
run eval_B_A1_C --spb 3 --val-every 7 --extra "$A1"
run profile_B_A1_C --spb 3 --profile --extra "$A1"
