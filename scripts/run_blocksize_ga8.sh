#!/usr/bin/env bash
# #192: real 1B step time per block size at the PRODUCTION grad_accum=8.
set -u
PY="$1"; DATA="$2"
mkdir -p results/bcd_memory
for spec in "2 13" "3 10" "4 8" "6 5"; do
  set -- $spec; k=$1; blocks=$2
  rm -rf checkpoints/bcd_memory_scratch
  echo "=== ga8 budget_layers=$k" >> results/bcd_memory/ga8.log
  timeout 1700 "$PY" scripts/profile_bcd_memory.py --data-dir "$DATA" --budget-layers $k --blocks $blocks --grad-accum 8 \
     --out "results/bcd_memory/ga8_k$k.jsonl" > "results/bcd_memory/ga8_k$k.log" 2>&1
  echo "exit=$? rows=$(wc -l < results/bcd_memory/ga8_k$k.jsonl)" >> results/bcd_memory/ga8.log
done
echo DONE >> results/bcd_memory/ga8.log
