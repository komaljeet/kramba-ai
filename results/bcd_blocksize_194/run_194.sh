# #194 long run: k2 then k4, one process at a time, one log file per attempt.
# Launch from the worktree root:  bash results/bcd_blocksize_194/run_194.sh
# Re-running the same command resumes (trainer auto-resume from ckpt_<tag>, saved
# every 700 steps); an arm whose steps file already reaches 2800 is skipped.
# Do not launch with anything else on the GPU.
set -u
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
OUT=results/bcd_blocksize_194
COMMON="--out $OUT --steps 2800 --eval-every 100 --eval-at 1 --save-every 700 --keep 2"
done_steps() {   # last step logged by an arm (0 if none)
  $PY -c "import json,sys
try: print(max(json.loads(l)['step'] for l in open(sys.argv[1]) if l.strip()))
except Exception: print(0)" "$OUT/steps_$1.jsonl"
}
used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
if [ "$used" -gt 1000 ]; then echo "GPU busy ($used MiB used), not launching" >> $OUT/run_194.status; exit 1; fi
for arm in "k2 2 13" "k4 4 7"; do
  set -- $arm; tag=$1; k=$2; nb=$3
  for attempt in 1 2 3; do
    last=$(done_steps $tag)
    if [ "$last" -ge 2800 ]; then break; fi
    echo "$(date '+%F %T') start $tag attempt $attempt (last step $last)" >> $OUT/run_194.status
    $PY -u scripts/exp_bcd_blocksize_194.py --k $k --expect-blocks $nb --tag $tag $COMMON \
      > $OUT/log_${tag}.a${attempt}.log 2>&1
    echo "$(date '+%F %T') $tag attempt $attempt exit $?" >> $OUT/run_194.status
  done
done
echo "$(date '+%F %T') RUN_194_DONE" >> $OUT/run_194.status
