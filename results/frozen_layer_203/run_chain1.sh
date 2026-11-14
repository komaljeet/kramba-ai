#!/bin/bash
cd /c/Users/komal/Documents/Codes/kramba-ai-wt203
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/frozen_layer_203
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/frozen_layer_203.py --out $O/$tag.json "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run A_after --active 1 --steps 3 --profile --hash
run B_before --active 3 --steps 3 --profile
run A_resident --active 1 --steps 3 --cache-blocks 8 --profile
