#!/bin/bash
# #207 chain 1 at the 1B k4 production settings (profile_step_202.py).
#   A0 = --active-no-ckpt-layers 0 --fp8-bias off  (pre-#207 trainer defaults)
#   A1 = --fp8-bias off                            (step 1: no-ckpt 1 now the default; = main + flag, 11.285 s)
#   A2 = --fp8-bias add                            (step 2: 2-D bias add kernel, dead adds skipped)
#   A3 = (no flags)                                (step 3: bias fused into SiLU / GELU / residual; new default)
# Identity (hash) runs first, then interleaved timing repeats, then VRAM with the
# held-out eval after every step.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr2
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/round2_207
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s207_ckpt"
A0="--active-no-ckpt-layers 0 --fp8-bias off"
A1="--fp8-bias off"
A2="--fp8-bias add"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run hash_A3 --spb 2 --hash
run hash_A1 --spb 2 --hash --extra "$A1"
run hash_A0 --spb 2 --hash --extra "$A0"
run hash_A2 --spb 2 --hash --extra "$A2"
run hash_A3_falsify --spb 2 --hash --perturb-grad 9.5367431640625e-07
for r in 1 2 3; do
  run time_A0_$r --spb 3 --extra "$A0"
  run time_A1_$r --spb 3 --extra "$A1"
  run time_A2_$r --spb 3 --extra "$A2"
  run time_A3_$r --spb 3
done
run evalv_A1 --spb 2 --heldout-every 1 --extra "$A1"
run evalv_A3 --spb 2 --heldout-every 1
echo "=== chain done $(date)"
