#!/bin/bash
# #207 chain 2 (chain 1 was stopped after its hash runs, so the amax-hint change
# could join the same interleaved timing). 1B k4 production settings.
#   A0 = --active-no-ckpt-layers 0 --fp8-bias off   (pre-#207 trainer defaults)
#   A1 = --fp8-bias off                             (step 1: no-ckpt 1 default; = main + flag)
#   A2 = --fp8-bias add                             (step 2: 2-D bias add kernel, dead adds skipped)
#   A3 = --no-fused-bias-amax                       (step 3a: bias fused into SiLU / GELU / residual)
#   A4 = (no flags)                                 (step 3b: + amax computed by the fused kernels; new default)
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr2
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/round2_207
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s207_ckpt"
A0="--active-no-ckpt-layers 0 --fp8-bias off"
A1="--fp8-bias off"
A2="--fp8-bias add"
A3="--no-fused-bias-amax"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run hash_A4 --spb 2 --hash
for r in 1 2 3; do
  run time_A0_$r --spb 3 --extra "$A0"
  run time_A1_$r --spb 3 --extra "$A1"
  run time_A2_$r --spb 3 --extra "$A2"
  run time_A3_$r --spb 3 --extra "$A3"
  run time_A4_$r --spb 3
done
run hash_A4_falsify --spb 2 --hash --perturb-grad 9.5367431640625e-07
run evalv_A1 --spb 2 --heldout-every 1 --extra "$A1"
run evalv_A4 --spb 2 --heldout-every 1
echo "=== chain done $(date)"
