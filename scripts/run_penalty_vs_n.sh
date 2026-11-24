#!/usr/bin/env bash
# #198: BCD penalty vs blocks per rotation. 2 layers per block, narrow width so full fits.
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
DATA=/c/Users/komal/Documents/Codes/kramba-ai/data_cache
mkdir -p results/local_loss
for cfg in "12:3:9" "24:6:18"; do
  IFS=: read L A B <<< "$cfg"
  $PY scripts/ab_local_loss.py --arms bcd_k2 full --steps 12000 --d-model 512 --n-heads 16 \
    --num-layers $L --attn-layers $A $B --data-dir $DATA \
    --out results/local_loss/penalty_d512_L${L}.json > results/local_loss/penalty_d512_L${L}.log 2>&1
  echo "L$L exit=$? $(date +%T)" >> results/local_loss/penalty_done.log
done
echo DONE >> results/local_loss/penalty_done.log
