#!/usr/bin/env bash
# #192: same 1B profile under different PYTORCH_CUDA_ALLOC_CONF settings, one run each.
# Usage: bash scripts/run_alloc_sweep.sh <python.exe> <data_dir>
set -u
PY="$1"
DATA="$2"
mkdir -p results/bcd_memory
for cfg in "none" "garbage_collection_threshold:0.7" "max_split_size_mb:128" \
           "garbage_collection_threshold:0.7,max_split_size_mb:128"; do
  tag=$(echo "$cfg" | tr -c 'a-zA-Z0-9\n' '_')
  rm -rf checkpoints/bcd_memory_scratch
  if [ "$cfg" = "none" ]; then unset PYTORCH_CUDA_ALLOC_CONF; else export PYTORCH_CUDA_ALLOC_CONF="$cfg"; fi
  echo "=== $cfg" >> results/bcd_memory/sweep.log
  timeout 900 "$PY" scripts/profile_bcd_memory.py --data-dir "$DATA" \
      --out "results/bcd_memory/alloc_${tag}.jsonl" > "results/bcd_memory/alloc_${tag}.log" 2>&1
  echo "exit=$? rows=$(wc -l < results/bcd_memory/alloc_${tag}.jsonl)" >> results/bcd_memory/sweep.log
done
echo DONE >> results/bcd_memory/sweep.log
