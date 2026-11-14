# #200 smoke, before the long runs. From the worktree root.
# k4 at 10 steps per block (the shortest arm): one full rotation (70 steps) at full LR
# (warmup 0, so bf16 fingerprints can see every update), every turn fingerprinted:
# all params must update over the rotation, frozen params must never change, the
# optimizer must hold NO state at a turn start (the trainer deletes it at a switch)
# and only finite state at a turn end. Eval at 1 and 70.
set -u
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/bcd_spb_200/smoke
rm -rf $O/ckpt_*
$PY -u scripts/exp_bcd_blocksize_194.py --out $O --max-steps 2800 --k 4 --expect-blocks 7 \
  --tag fp_s10 --steps 70 --spb 10 --warmup 0 --eval-every 0 --eval-at 1,70 --fingerprint \
  > $O/log_fp_s10.log 2>&1
echo "fp_s10 exit $?" >> $O/smoke.status
echo SMOKE_DONE >> $O/smoke.status
