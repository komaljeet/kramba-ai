#!/bin/bash
# #218 step 3b: chaos control. ref again + a full-fp32 arm whose only change is the
# micro-batch accumulation order. ref must reproduce numerics_lr*.jsonl bit for bit
# (determinism); ctrl shows how far a ~1e-7 perturbation drifts in 24 steps.
W=/c/Users/komal/Documents/Codes/kramba-ai-wt16gb
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=$W/results/streamed_16gb_gate_218
cd $W
gpu_free() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')
  if [ "$used" -gt 400 ]; then echo "GPU NOT EMPTY ($used MiB used) - abort"; exit 9; fi
  echo "gpu used ${used} MiB"
}
for lr in 0.02 0.002 0.0005; do
  gpu_free
  tag=control_lr$lr
  echo "=== $tag $(date)"
  $PY -u scripts/bench_lean_numerics_218.py --muon-lr $lr --steps 24 --arms ref,ctrl_mbrev --out $O/$tag.jsonl > $O/$tag.log 2>&1
  echo "exit $? $(date)"
  tail -2 $O/$tag.log
done
gpu_free
echo "=== overhead probe $(date)"
$PY -u scripts/probe_process_overhead_218.py $O/overhead.json > $O/overhead.log 2>&1
echo "exit $? $(date)"
echo "=== control done $(date)"
