#!/bin/bash
# #208 chain 2: chunked-scan microbench (candidate 2), op-level profile of the default
# step (candidate 3), then the 1B held-out loss A/B (candidate 1).
# A/B: 140 steps = 2 full passes over the 7 BCD blocks at 10 steps per block (production
# is 100 per block; 4 runs at 100 would be ~17 h, over the 6 h budget). Warmup 14,
# cosine to 0 at 140. Seeds 0 and 1 = model init; data order pinned (shuffle seed 0);
# 64 fixed held-out windows after steps 1, 70, 140.
#   def_s*     defaults (scan kernel v1)
#   t_s*       --scan-kernel t
#   tbroken_s0 --scan-kernel t with the scan backward returning zero gradients (falsifier)
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr3
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/round3_208
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s208_ckpt"
export AB208_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s208_ckpt"
echo "=== scan_chunk $(date)"; $PY -u scripts/scan_chunk_208.py $O/scan_chunk.json > $O/scan_chunk.log 2>&1; echo "exit $? $(date)"
echo "=== prof_ops $(date)"; $PY -u scripts/profile_step_202.py --out $O/prof_ops.jsonl --spb 2 --profile --profile-ops > $O/prof_ops.log 2>&1; echo "exit $? $(date)"
ab() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/ab_heldout_208.py --tag $tag --out $O/ab \
  --steps 140 --spb 10 --warmup 14 --eval-at 1,70,140 --eval-windows 64 "$@" > $O/ab_$tag.log 2>&1; echo "exit $? $(date)"; }
ab def_s0 --seed 0
ab t_s0 --seed 0 --extra "--scan-kernel t"
ab def_s1 --seed 1
ab t_s1 --seed 1 --extra "--scan-kernel t"
ab tbroken_s0 --seed 0 --extra "--scan-kernel t" --break-scan
echo "=== chain done $(date)"
