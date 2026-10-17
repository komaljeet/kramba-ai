#!/bin/bash
# Short 48M sanity A/B through the REAL trainer (#152's config), only --dropout differing.
# Waits for the 1B speed A/B to release the GPU. One run at a time, one log per run.
set -u
WT=/c/Users/komal/Documents/Codes/kramba-wt-nodrop
R=/c/Users/komal/Documents/Codes/kramba-ai
PY=$R/.venv/Scripts/python.exe
D=$WT/results/dropout_1b
until [ -f $D/ab_done.txt ]; do sleep 20; done
sleep 15
cd $R  # relative data paths: the tokenizer cache signature is keyed on the path string
for P in 0.1 0.0; do
  N=q48m_p$P
  "$PY" -u $WT/train_edge_conversational.py --code-corpus data_cache/code_corpus/stack_code.txt --fim --tokenizer data_cache/tokenizer_32k_fim     --arch hvn-hybrid --attn-layers 3,9 --attn-window 4096     --d-model 512 --num-layers 12 --n-heads 16     --seq-len 4096 --batch-size 1 --grad-accum 8     --grad-checkpoint --chunked-loss     --no-fast --precision bf16 --tf32 --triton-scan --megakernel-scan     --optimizer muon --muon-lr 0.02 --dropout $P     --seed 1234 --deterministic --max-steps 500 --warmup-steps 80     --log-every 1 --save-every-steps 250 --keep-checkpoints 0     --checkpoint-dir $WT/checkpoints/$N > $D/$N.log 2>&1
  echo "EXIT $?" >> $D/$N.log
  "$PY" -u $WT/eval/heldout_loss.py --sweep $WT/checkpoints/$N --tokenizer data_cache/tokenizer_32k_fim --corpus data_cache/code_corpus/stack_code.txt     --expect-tokens 400000 --out $D/$N.heldout.json > $D/$N.heldout.log 2>&1
  echo "EXIT $?" >> $D/$N.heldout.log
done
echo QDONE > $D/q_done.txt
