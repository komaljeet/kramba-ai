#!/bin/bash
# #202 chain 5: VRAM with the production held-out eval after EVERY step (2 steps per
# turn, so momentum + grads are live: steady state), base vs final (A1) vs a per-block
# no-checkpoint list; then step time of the per-block list without eval.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtspeed
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/step_speed_202
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run evalv_base --spb 2 --heldout-every 1 --loss-recompute --no-frozen-fused
run evalv_final --spb 2 --heldout-every 1 --extra "--active-no-ckpt-layers 1"
run evalv_turn --spb 2 --heldout-every 1 --abort-peak 0.86 --extra "--active-no-ckpt-layers 1,1,2,2,3,3"
run time_turn --spb 3 --abort-peak 0.86 --extra "--active-no-ckpt-layers 1,1,2,2,3,3"
