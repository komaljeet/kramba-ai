#!/bin/bash
# Multi-epoch dropout A/B (#dropout follow-up). 384x8 hybrid (11.89M body params),
# 30 tok/body-param budget, pool capped so the run makes ~4.6 passes -- the same
# three ratios as the real 1B run (30.3B budget over 6.64B unique tokens).
#   pool  19,200 windows x 4096 = 78.6M unique tokens
#   steps 11,200 x 32,768 = 367M tokens = 30.9 tok/body-param = 4.67 passes
#   one pass = 2,400 steps; checkpoints every 800 = 3 per pass
# One run at a time, one log per run; gap eval runs after each arm.
set -u
WT=/c/Users/komal/Documents/Codes/kramba-wt-nodrop
R=/c/Users/komal/Documents/Codes/kramba-ai
PY=$R/.venv/Scripts/python.exe
D=$WT/results/dropout_repeat
cd $R  # relative data paths: the tokenizer cache signature is keyed on the path string
for SEED in 1234 5678; do
for P in 0.1 0.0; do
  N=rep384_p${P}_s${SEED}
  if ! grep -q "EXIT 0" $D/$N.log 2>/dev/null; then
  "$PY" -u $WT/train_edge_conversational.py --code-corpus data_cache/code_corpus/stack_code.txt --fim --tokenizer data_cache/tokenizer_32k_fim \
    --arch hvn-hybrid --attn-layers 2,6 --attn-window 4096 \
    --d-model 384 --num-layers 8 --n-heads 12 \
    --seq-len 4096 --batch-size 1 --grad-accum 8 \
    --grad-checkpoint --chunked-loss --fast \
    --optimizer muon --muon-lr 0.02 --dropout $P \
    --max-train-windows 19200 \
    --seed $SEED --deterministic --max-steps 11200 --warmup-steps 500 \
    --log-every 50 --save-every-steps 800 --keep-checkpoints 0 \
    --checkpoint-dir $WT/checkpoints/$N >> $D/$N.log 2>&1
  echo "EXIT $?" >> $D/$N.log
  fi
  "$PY" -u $WT/scripts/eval_repeat_gap.py --sweep $WT/checkpoints/$N --max-train-windows 19200 \
    --out $D/$N.gap.json >> $D/$N.gap.log 2>&1
  echo "EXIT $?" >> $D/$N.gap.log
done
done
echo DONE > $D/all_done.txt
