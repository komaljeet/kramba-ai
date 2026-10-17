#!/bin/bash
# Dropout on/off in scripts/attribute_1b_real_step.py (the 15.067 s/step harness), timing only.
WT=/c/Users/komal/Documents/Codes/kramba-wt-nodrop
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
D=$WT/results/dropout_1b
until [ -f $D/q_done.txt ]; do sleep 20; done
sleep 15
cd $WT
for A in on:0.1 off:0.0 on2:0.1 off2:0.0; do
  N=${A%%:*}; P=${A##*:}
  $PY -u scripts/attribute_1b_real_step.py --step-cache --no-profile --cycles-a 2 --dropout $P \
    --out $D/attrib_$N.json > $D/attrib_$N.log 2>&1
  echo "EXIT $?" >> $D/attrib_$N.log
done
echo ADONE > $D/attrib_done.txt
