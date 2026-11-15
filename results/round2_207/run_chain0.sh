#!/bin/bash
# #207 chain 0 on current main (after #206): kernel profile of every turn's last step,
# then one plain timing run with nvidia-smi sampled every 250 ms (throttling check).
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr2
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/round2_207
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s207_ckpt"
X="--active-no-ckpt-layers 1"
Q=timestamp,clocks.sm,clocks.mem,clocks.max.sm,clocks.max.mem,power.draw,power.limit,enforced.power.limit,temperature.gpu,utilization.gpu,clocks_throttle_reasons.active,clocks_throttle_reasons.gpu_idle,clocks_throttle_reasons.sw_power_cap,clocks_throttle_reasons.hw_slowdown,clocks_throttle_reasons.hw_thermal_slowdown,clocks_throttle_reasons.hw_power_brake_slowdown,clocks_throttle_reasons.sw_thermal_slowdown,pstate
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run prof_main --spb 2 --profile --extra "$X"
nvidia-smi --query-gpu=$Q --format=csv -lms 250 > $O/smi_time_main.csv &
SMI=$!
run time_main_smi --spb 3 --extra "$X"
kill $SMI
echo "=== chain done $(date)"
