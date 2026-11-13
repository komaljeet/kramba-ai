# #194 k5 gate A + smoke, sequential, one log per run. From the worktree root.
#
# Gate rule, written BEFORE running (checked by gate_k5.py):
#  (a) no training turn and no eval above 85% of the card (peak reserved);
#  (b) no block's median step time above 1.5x what its depth predicts from the
#      k2/k4 long-run per-block medians (linear fit: start layer, layers in block;
#      the k6 WDDM spill ran 41-88 s where 12-20 s was expected);
#  (c) mean step over one full rotation (mean of per-block medians, 6 blocks)
#      at most 13.9 s (k4 12.6 s + ~10%).
# Any gate fails -> do not launch the long run.
set -u
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/bcd_blocksize_194/smoke
H="scripts/exp_bcd_blocksize_194.py --out $O --max-steps 2800"
# 1. k5: one full rotation (6 blocks x 3 steps) at full LR (warmup 0), every turn
#    fingerprinted, eval at step 1 and at the end of every layer-block turn
#    (5-layer blocks active at 6..15, the 4-layer+tail block at 18).
$PY -u $H --k 5 --expect-blocks 6 --tag fp_k5 --steps 18 --spb 3 --warmup 0 --eval-every 0 --eval-at 1,6,9,12,15,18 --fingerprint > $O/log_fp_k5.log 2>&1; echo "fp_k5 exit $?" >> $O/smoke_k5.status
# 2. negative control: warmup 1 leaves lr at 0 (measured in the k4 smoke), so NOTHING may change
$PY -u $H --k 5 --expect-blocks 6 --tag neg_k5 --steps 6 --spb 1 --warmup 1 --eval-every 0 --eval-at "" --fingerprint > $O/log_neg_k5.log 2>&1; echo "neg_k5 exit $?" >> $O/smoke_k5.status
echo SMOKE_K5_DONE >> $O/smoke_k5.status
