#!/usr/bin/env bash
# Issue #152: the three effective-batch arms, one at a time, then held-out scoring.
#
# Matched TOKENS: 24,416 micro-batches x 4096 = 100,007,936 tokens in every arm
# (24,416 is the multiple of 32 nearest 100M/4096, so all three divide it exactly).
# Warmup is the same 16.38% of each arm's optimizer steps (= the same 4,000
# micro-batches / 16.4M tokens), and checkpoints land every 8.39M tokens.
#
# Restart policy: an arm without its final checkpoint is WIPED and rerun from
# step 0, never resumed. The trainer's resume path restarts the data iterator
# and skips warmup, so a resumed arm would not see the same tokens as the others.
# A seeded fresh rerun is bit-identical (checked by the repro cell), so the only
# cost of an interruption is time.
set -u
REPO=/c/Users/komal/Documents/Codes/kramba-ai
WT=$(cd "$(dirname "$0")/.." && pwd)
PY="$REPO/.venv/Scripts/python.exe"
cd "$REPO"
mkdir -p logs/eb152 results/effective_batch

for spec in "ga8 8 3052 500 256" "ga16 16 1526 250 128" "ga32 32 763 125 64"; do
  set -- $spec; NAME=$1; GA=$2; STEPS=$3; WARM=$4; SAVE=$5
  DIR="checkpoints/eb152_$NAME"
  for attempt in 1 2; do
    [ -f "$DIR/edge_conversational.pt" ] && break
    rm -rf "$DIR"
    echo "$(date '+%F %T') START $NAME attempt $attempt" >> logs/eb152/sweep.log
    bash "$WT/scripts/run_eb152_arm.sh" "$NAME" "$GA" "$STEPS" "$WARM" "$SAVE" \
      > "logs/eb152/$NAME.log" 2>&1
    echo "$(date '+%F %T') EXIT $NAME rc=$?" >> logs/eb152/sweep.log
  done
  if [ ! -f "$DIR/edge_conversational.pt" ]; then
    echo "$(date '+%F %T') FAILED $NAME -- stopping sweep" >> logs/eb152/sweep.log
    exit 1
  fi
  TPS=$((GA * 4096))
  "$PY" -u "$WT/eval/heldout_loss.py" --checkpoint "$DIR/edge_conversational.pt" \
    --expect-tokens 400000 --tokens-per-step $TPS \
    --out "results/effective_batch/final_$NAME.json" > "logs/eb152/heldout_final_$NAME.log" 2>&1
  "$PY" -u "$WT/eval/heldout_loss.py" --sweep "$DIR" \
    --expect-tokens 400000 --tokens-per-step $TPS \
    --out "results/effective_batch/traj_$NAME.json" > "logs/eb152/heldout_traj_$NAME.log" 2>&1
  echo "$(date '+%F %T') SCORED $NAME" >> logs/eb152/sweep.log
done
echo "$(date '+%F %T') ALL DONE" >> logs/eb152/sweep.log
