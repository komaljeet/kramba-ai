#!/bin/bash
# #202 chain 4: VRAM during the production (offload-aware) held-out eval, base vs final,
# eval after every step at every block position; held-out losses must match too.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtspeed
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/step_speed_202
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run eval_base --spb 1 --heldout-every 1 --loss-recompute --no-frozen-fused
run eval_final --spb 1 --heldout-every 1 --extra "--active-no-ckpt-layers 1"
