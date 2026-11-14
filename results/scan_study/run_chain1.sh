#!/bin/bash
# #206 chain 1 at the 1B k4 production settings (#202/#204: --active-no-ckpt-layers 1).
#   base = --no-scan-v1-warps (current main), v1 = tuned warps (bit-identical claim),
#   t = --scan-kernel t (not bit-identical).
# Identity (hash) runs, then interleaved timing repeats, then VRAM with the held-out
# eval after every step.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtscan
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/scan_study
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s206_ckpt"
X="--active-no-ckpt-layers 1"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run hash_v1 --spb 2 --hash --extra "$X"
run hash_base --spb 2 --hash --extra "$X --no-scan-v1-warps"
run hash_v1_falsify --spb 2 --hash --perturb-grad 9.5367431640625e-07 --extra "$X"
run hash_t --spb 2 --hash --extra "$X --scan-kernel t"
for r in 1 2 3; do
  run time_base$r --spb 3 --extra "$X --no-scan-v1-warps"
  run time_v1$r --spb 3 --extra "$X"
  run time_t$r --spb 3 --extra "$X --scan-kernel t"
done
run evalv_base --spb 2 --heldout-every 1 --extra "$X --no-scan-v1-warps"
run evalv_v1 --spb 2 --heldout-every 1 --extra "$X"
run evalv_t --spb 2 --heldout-every 1 --extra "$X --scan-kernel t"
echo "=== chain done $(date)"
