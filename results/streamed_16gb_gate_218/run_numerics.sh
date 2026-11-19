#!/bin/bash
# #218 step 3: lean host-state numerics on the real 1B group (layers 4-7).
# One process and one log per Muon lr; nothing else on the GPU.
#   0.02   = peak lr (start of the cosine)
#   0.002  = 10% of peak (~80% through the cosine)
#   0.0005 = 2.5% of peak (~90% through the cosine) -- where RNE should lose the update
W=/c/Users/komal/Documents/Codes/kramba-ai-wt16gb
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=$W/results/streamed_16gb_gate_218
cd $W
gpu_free() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')
  if [ "$used" -gt 400 ]; then echo "GPU NOT EMPTY ($used MiB used) - abort"; exit 9; fi
  echo "gpu used ${used} MiB"
}
for lr in ${LRS:-0.02 0.002 0.0005}; do
  gpu_free
  tag=numerics_lr$lr
  echo "=== $tag $(date)"
  $PY -u scripts/bench_lean_numerics_218.py --muon-lr $lr --steps ${STEPS:-24} --out $O/$tag.jsonl > $O/$tag.log 2>&1
  echo "exit $? $(date)"
  tail -3 $O/$tag.log
done
echo "=== numerics done $(date)"
