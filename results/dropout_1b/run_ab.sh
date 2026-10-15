#!/bin/bash
# Alternating arms, one process at a time on the GPU, one log per run.
cd /c/Users/komal/Documents/Codes/kramba-wt-nodrop
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
D=results/dropout_1b
$PY -u scripts/ab_dropout_1b.py --dropout 0.0 --profile --out $D/off_1.json > $D/off_1.log 2>&1; echo "EXIT $?" >> $D/off_1.log
$PY -u scripts/ab_dropout_1b.py --dropout 0.1 --profile --out $D/on_1.json > $D/on_1.log 2>&1; echo "EXIT $?" >> $D/on_1.log
$PY -u scripts/ab_dropout_1b.py --dropout 0.0 --out $D/off_2.json > $D/off_2.log 2>&1; echo "EXIT $?" >> $D/off_2.log
$PY -u scripts/ab_dropout_1b.py --dropout 0.1 --out $D/on_2.json > $D/on_2.log 2>&1; echo "EXIT $?" >> $D/on_2.log
echo ALLDONE > $D/ab_done.txt
