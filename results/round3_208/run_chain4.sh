#!/bin/bash
# #208 chain 4: held-out loss A/B for --fused-norm-residual (same harness, steps,
# seeds, data and held-out windows as chain 2; the def_s0 / def_s1 arms are reused),
# then the scan kernels' error vs fp64 re-measured on this branch (#206 harness).
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr3
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/round3_208
export AB208_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s208_ckpt"
ab() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/ab_heldout_208.py --tag $tag --out $O/ab \
  --steps 140 --spb 10 --warmup 14 --eval-at 1,70,140 --eval-windows 64 "$@" > $O/ab_$tag.log 2>&1; echo "exit $? $(date)"; }
ab fnr_s0 --seed 0 --extra=--fused-norm-residual
ab fnr_s1 --seed 1 --extra=--fused-norm-residual
echo "=== scan_fp64 $(date)"; $PY -u scripts/scan_study2_206.py $O/scan_fp64.json accuracy > $O/scan_fp64.log 2>&1; echo "exit $? $(date)"
echo "=== chain done $(date)"
