# #194 long run, k5 arm (5 layers per BCD block: embedding, then 5,5,5,5,4 layers, 6
# training blocks, 600 steps per rotation). Same flags as run_194.sh's k2/k4 arms.
# Launch from the worktree root:  bash results/bcd_blocksize_194/run_194_k5.sh
# Re-running the same command resumes (trainer auto-resume from ckpt_k5, saved
# every 700 steps); stops once steps_k5.jsonl reaches 2800.
# Do not launch with anything else on the GPU.
set -u
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
OUT=results/bcd_blocksize_194
COMMON="--out $OUT --steps 2800 --eval-every 100 --eval-at 1 --save-every 700 --keep 2"
ST=$OUT/run_194_k5.status
done_steps() {   # last step logged by an arm (0 if none)
  $PY -c "import json,sys
try: print(max(json.loads(l)['step'] for l in open(sys.argv[1]) if l.strip()))
except Exception: print(0)" "$OUT/steps_$1.jsonl"
}
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 1000 ]; then echo "$(date '+%F %T') GPU busy ($used MiB used), not launching" >> $ST; exit 1; fi
tag=k5; k=5; nb=6
for attempt in 1 2 3; do
  last=$(done_steps $tag)
  if [ "$last" -ge 2800 ]; then break; fi
  log=$OUT/log_${tag}.a${attempt}.$(date '+%m%d-%H%M').log   # never reuse a log file
  echo "$(date '+%F %T') start $tag attempt $attempt (last step $last) -> $log" >> $ST
  $PY -u scripts/exp_bcd_blocksize_194.py --k $k --expect-blocks $nb --tag $tag $COMMON \
    > $log 2>&1
  echo "$(date '+%F %T') $tag attempt $attempt exit $?" >> $ST
done
echo "$(date '+%F %T') RUN_194_K5_DONE (last step $(done_steps $tag))" >> $ST
