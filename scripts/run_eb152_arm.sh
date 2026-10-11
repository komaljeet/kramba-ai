#!/usr/bin/env bash
# Issue #152: one effective-batch arm through the REAL trainer.
# usage: run_eb152_arm.sh <name> <grad_accum> <max_steps> <warmup_steps> <save_every>
# Runs from the main repo (data_cache/, checkpoints/ live there); code from this worktree.
set -u
NAME=$1; GA=$2; STEPS=$3; WARM=$4; SAVE=$5
REPO=/c/Users/komal/Documents/Codes/kramba-ai
WT=$(cd "$(dirname "$0")/.." && pwd)
cd "$REPO"
"$REPO/.venv/Scripts/python.exe" -u "$WT/train_edge_conversational.py" \
  --code-corpus data_cache/code_corpus/stack_code.txt --fim \
  --tokenizer data_cache/tokenizer_32k_fim \
  --arch hvn-hybrid --attn-layers 3,9 --attn-window 4096 \
  --d-model 512 --num-layers 12 --n-heads 16 \
  --seq-len 4096 --batch-size 1 --grad-accum "$GA" \
  --grad-checkpoint --chunked-loss \
  --no-fast --precision bf16 --tf32 --triton-scan --megakernel-scan \
  --optimizer muon --muon-lr 0.02 \
  --seed 1234 --deterministic --max-steps "$STEPS" --warmup-steps "$WARM" \
  --log-every 1 --save-every-steps "$SAVE" --keep-checkpoints 0 \
  --checkpoint-dir "checkpoints/eb152_$NAME"
