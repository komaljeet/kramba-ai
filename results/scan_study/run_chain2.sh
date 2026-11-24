#!/bin/bash
# #206 chain 2: held-out loss A/B, v1 vs t scan kernels, on the d512 stand-in
# (ab_local_loss.py, the run_penalty_vs_n.sh harness: d512, 16 heads -> d_head 32,
# 12 layers, attention at 3 and 9). Same seed -> same init, batches and held-out
# probes for both kernels. Arms: bcd_k2 (BCD, production regime) and full
# (all layers trained, grad checkpointing -> the fused-checkpoint recompute).
# Plus a same-seed v1 repeat to measure run-to-run noise.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtscan
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/scan_study
DATA=/c/Users/komal/Documents/Codes/kramba-ai/data_cache
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/ab_local_loss.py --arms bcd_k2 full \
  --steps 2000 --d-model 512 --n-heads 16 --num-layers 12 --attn-layers 3 9 --data-dir $DATA \
  --out $O/$tag.json "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run lossab_s7_v1 --seed 7 --scan-kernel v1
run lossab_s7_t --seed 7 --scan-kernel t
run lossab_s8_v1 --seed 8 --scan-kernel v1
run lossab_s8_t --seed 8 --scan-kernel t
run lossab_s7_v1_repeat --seed 7 --scan-kernel v1
echo "=== chain done $(date)"
