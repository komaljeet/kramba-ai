#!/usr/bin/env bash
# #199: sequential 1B arms, one GPU process at a time, one log per run.
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
DATA=/c/Users/komal/Documents/Codes/kramba-ai/data_cache
cd /c/Users/komal/Documents/Codes/kramba-ai-wt199
for arm in "base:" "forced:--force-prefix-grad" "nograd:--prefix-no-grad" "base2:"; do
  tag=${arm%%:*}; flags=${arm#*:}
  rm -rf checkpoints/bcd_prefix_scratch
  PYTHONUNBUFFERED=1 $PY scripts/profile_bcd_prefix.py --visit 0,1,6,12 --steps-per-block 3 \
     --grad-accum 8 --data-dir $DATA $flags --out results/bcd_prefix/$tag.jsonl \
     > results/bcd_prefix/$tag.log 2>&1
  echo "$tag exit=$? $(date +%T)" >> results/bcd_prefix/arms.log
done
echo DONE >> results/bcd_prefix/arms.log
