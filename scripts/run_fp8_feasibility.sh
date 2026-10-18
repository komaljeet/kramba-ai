#!/usr/bin/env bash
# Phase-1 FP8 feasibility arms, run ONE AT A TIME (one GPU process, one log each).
set -u
P=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
W=/c/Users/komal/Documents/Codes/kramba-wt-fp8
O=$W/results/fp8_feasibility
mkdir -p "$O"
cd "$W"
run() {  # name, extra args...
  local name=$1; shift
  echo "=== $name start $(date)" >> "$O/runner.log"
  PYTHONUNBUFFERED=1 "$P" -u scripts/exp_fp8_feasibility.py --out "$O/$name" "$@" > "$O/$name.log" 2>&1
  echo "=== $name exit=$? $(date)" >> "$O/runner.log"
}
run bf16_s0        --arm bf16      --seed 0 --steps 3000
run fp8_all_s0     --arm fp8_all   --seed 0 --steps 3000
run fp8_mixed_s0   --arm fp8_mixed --seed 0 --steps 3000
run bf16_s1        --arm bf16      --seed 1 --steps 3000
run fp8_noscale_s0 --arm fp8_noscale --seed 0 --steps 1000
echo "=== ALL DONE $(date)" >> "$O/runner.log"
