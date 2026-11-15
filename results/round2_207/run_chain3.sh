#!/bin/bash
# #207 chain 3: chain 2's A3 runs failed at argparse ("--extra --no-fused-bias-amax"
# was read as an option; fixed with --extra=...). A3 timed here, interleaved with
# three more A4 repeats (A4 repeats 4-6), same settings as chain 2.
#   A3 = --no-fused-bias-amax   (bias fusions, no amax hints)
#   A4 = defaults               (bias fusions + amax hints)
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr2
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/round2_207
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s207_ckpt"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
for r in 1 2 3; do
  run time_A3_$r --spb 3 --extra=--no-fused-bias-amax
  run time_A4_$((r + 3)) --spb 3
done
echo "=== chain done $(date)"
