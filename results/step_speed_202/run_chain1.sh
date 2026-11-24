#!/bin/bash
# #202 chain 1: identity hashes (baseline x2, 1-ulp falsifier, change B) and step-time A/B.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtspeed
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/step_speed_202
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run hash_base1 --hash --spb 2 --loss-recompute
run hash_base2 --hash --spb 2 --loss-recompute
run hash_perturb --hash --spb 2 --loss-recompute --perturb-ulp
run hash_B --hash --spb 2
run time_base1 --spb 3 --loss-recompute
run time_B --spb 3
run time_base2 --spb 3 --loss-recompute
run time_B_A1 --spb 3 --set _active_no_ckpt=1 --abort-peak 0.92
