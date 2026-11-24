# #200 long run: k4 with 50, then 25, then 10 steps per block turn. One process at a
# time, one log file per attempt. Same harness, seed, data, LR schedule and held-out set
# as the #194 k4@100 run (results/bcd_blocksize_194/*_k4.jsonl is the comparison arm).
# Launch from the worktree root:  bash results/bcd_spb_200/run_200.sh
# Re-running the same command resumes (trainer auto-resume from ckpt_<tag>, saved
# every 700 steps = a whole number of rotations for 350/175/70); an arm whose steps
# file already reaches 2800 is skipped. Refuses to start an attempt if the GPU has
# more than 1000 MiB in use.
set -u
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
OUT=results/bcd_spb_200
COMMON="--out $OUT --steps 2800 --eval-every 100 --eval-at 1 --save-every 700 --keep 2"
done_steps() {   # last step logged by an arm (0 if none)
  $PY -c "import json,sys
try: print(max(json.loads(l)['step'] for l in open(sys.argv[1]) if l.strip()))
except Exception: print(0)" "$OUT/steps_$1.jsonl"
}
for arm in "k4s50 50" "k4s25 25" "k4s10 10"; do
  set -- $arm; tag=$1; spb=$2
  for attempt in 1 2 3; do
    last=$(done_steps $tag)
    if [ "$last" -ge 2800 ]; then break; fi
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1)
    if [ "$used" -gt 1000 ]; then
      echo "$(date '+%F %T') GPU busy ($used MiB used), not starting $tag attempt $attempt; STOPPED" >> $OUT/run_200.status
      exit 1
    fi
    echo "$(date '+%F %T') start $tag attempt $attempt (last step $last)" >> $OUT/run_200.status
    $PY -u scripts/exp_bcd_blocksize_194.py --k 4 --expect-blocks 7 --spb $spb --tag $tag $COMMON \
      > $OUT/log_${tag}.a${attempt}.log 2>&1
    echo "$(date '+%F %T') $tag attempt $attempt exit $?" >> $OUT/run_200.status
  done
  last=$(done_steps $tag)
  if [ "$last" -lt 2800 ]; then
    echo "$(date '+%F %T') $tag FAILED after 3 attempts (last step $last); RUN_200_FAILED" >> $OUT/run_200.status
    exit 1
  fi
done
echo "$(date '+%F %T') RUN_200_DONE" >> $OUT/run_200.status
