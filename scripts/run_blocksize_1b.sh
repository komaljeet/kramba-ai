#!/usr/bin/env bash
# #192: real 1B profile with bigger BCD blocks. Usage: bash scripts/run_blocksize_1b.sh <python> <data_dir>
set -u
PY="$1"; DATA="$2"
mkdir -p results/bcd_memory
for spec in ${SPECS:-"4 8" "5 7" "6 6"}; do
  set -- $spec; k=$1; blocks=$2
  rm -rf checkpoints/bcd_memory_scratch
  echo "=== budget_layers=$k" >> results/bcd_memory/blocksize_emb.log
  timeout 900 "$PY" scripts/profile_bcd_memory.py --data-dir "$DATA" --budget-layers $k --blocks $blocks \
     --out "results/bcd_memory/blocksize_emb_k$k.jsonl" > "results/bcd_memory/blocksize_emb_k$k.log" 2>&1
  echo "exit=$? rows=$(wc -l < results/bcd_memory/blocksize_emb_k$k.jsonl)" >> results/bcd_memory/blocksize_emb.log
done
echo DONE >> results/bcd_memory/blocksize_emb.log
