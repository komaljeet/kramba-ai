#!/bin/bash
# #208 chain 3 (after chain 2): candidate 3 -- residual gradient folded into the RMSNorm
# backward (--fused-norm-residual). Kernel bench first, then 1B timing, 3 alternating
# repeats vs the defaults (profile_step_202.py, spb 3, as chain 1).
# First launch (14:58, log chain3_first.log) ran bench_norm_res + time_def2_1, then every
# fnr run exited at once: `--extra "--fused-norm-residual"` is read by argparse as a new
# option (a value with a space, like "--scan-kernel t", is not). Stopped during
# time_def2_2 and relaunched as `run_chain3.sh resume` with --extra=... from time_fnr_1.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr3
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/round3_208
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s208_ckpt"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
if [ "$1" != "resume" ]; then
  echo "=== bench_norm_res $(date)"; $PY -u scripts/bench_norm_res_208.py $O/bench_norm_res.json > $O/bench_norm_res.log 2>&1; echo "exit $? $(date)"
  run time_def2_1 --spb 3
fi
run time_fnr_1 --spb 3 --extra=--fused-norm-residual
for r in 2 3; do
  run time_def2_$r --spb 3
  run time_fnr_$r --spb 3 --extra=--fused-norm-residual
done
echo "=== chain done $(date)"
