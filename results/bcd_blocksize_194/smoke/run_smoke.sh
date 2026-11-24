# #194 smoke + falsifiers, sequential, one log per run. From the worktree root.
set -u
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/bcd_blocksize_194/smoke
H="scripts/exp_bcd_blocksize_194.py --out $O --max-steps 2800"
rm -rf $O/ckpt_* $O/*_fp_k4.* $O/layout_fp_k4.json
# 1. k4: full rotation at full LR (warmup 0), every turn fingerprinted, eval at 1 and 21
$PY -u $H --k 4 --expect-blocks 7 --tag fp_k4 --steps 21 --spb 3 --warmup 0 --eval-every 0 --eval-at 1,21 --fingerprint > $O/log_fp_k4.log 2>&1; echo "fp_k4 exit $?" >> $O/smoke.status
# 2. negative control: warmup 1 leaves lr at 0 (measured), so NOTHING may change
$PY -u $H --k 4 --expect-blocks 7 --tag neg_k4 --steps 6 --spb 2 --warmup 1 --eval-every 0 --eval-at "" --fingerprint > $O/log_neg_k4.log 2>&1; echo "neg_k4 exit $?" >> $O/smoke.status
# 3. k2: full rotation at full LR, fingerprinted, eval at 1 and 39
$PY -u $H --k 2 --expect-blocks 13 --tag fp_k2 --steps 39 --spb 3 --warmup 0 --eval-every 0 --eval-at 1,39 --fingerprint > $O/log_fp_k2.log 2>&1; echo "fp_k2 exit $?" >> $O/smoke.status
# 4. resume: run to 8 with a checkpoint at 4, then resume from 4 and redo 5..8
$PY -u $H --k 4 --expect-blocks 7 --tag rs --steps 8 --spb 3 --warmup 0 --eval-every 0 --eval-at 8 --save-every 4 > $O/log_rs_a.log 2>&1; echo "rs_a exit $?" >> $O/smoke.status
for f in steps eval turns; do cp $O/${f}_rs.jsonl $O/${f}_rs_full.jsonl; done
$PY -u $H --k 4 --expect-blocks 7 --tag rs --steps 8 --spb 3 --warmup 0 --eval-every 0 --eval-at 8 --save-every 4 > $O/log_rs_b.log 2>&1; echo "rs_b exit $?" >> $O/smoke.status
echo SMOKE_DONE >> $O/smoke.status
