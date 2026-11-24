#!/bin/bash
# #204 chain 1: identity (hash) runs, then interleaved timing repeats, then VRAM with
# the production held-out eval after every step. All at the #202 shipped settings
# (--active-no-ckpt-layers 1). "base" = --no-frozen-codes (the #202 behaviour).
cd /c/Users/komal/Documents/Codes/kramba-ai-wt204
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/frozen_codes_204
X="--active-no-ckpt-layers 1"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run hash_codes --spb 2 --hash --extra "$X"
run hash_base --spb 2 --hash --no-frozen-codes --extra "$X"
run hash_codes_falsify --spb 2 --hash --perturb-grad 9.5367431640625e-07 --extra "$X"
run time_base1 --spb 3 --no-frozen-codes --extra "$X"
run time_codes1 --spb 3 --extra "$X"
run time_base2 --spb 3 --no-frozen-codes --extra "$X"
run time_codes2 --spb 3 --extra "$X"
run time_base3 --spb 3 --no-frozen-codes --extra "$X"
run time_codes3 --spb 3 --extra "$X"
run evalv_base --spb 2 --heldout-every 1 --no-frozen-codes --extra "$X"
run evalv_codes --spb 2 --heldout-every 1 --extra "$X"
echo "=== chain done $(date)"
