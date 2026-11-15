#!/bin/bash
# #207 chain 4: the shipped defaults (no flags: no-ckpt 1, --fp8-bias fused, amax
# hints OFF after chain 3 measured them at -0.040 s, under the 0.05 s bar).
# Identity hash vs hash_A1 and held-out eval VRAM at every turn.
cd /c/Users/komal/Documents/Codes/kramba-ai-wtr2
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/round2_207
export S202_CKPT="C:/Users/komal/AppData/Local/Temp/claude/s207_ckpt"
run() { tag=$1; shift; echo "=== $tag $(date)"; $PY -u scripts/profile_step_202.py --out $O/$tag.jsonl "$@" > $O/$tag.log 2>&1; echo "exit $? $(date)"; }
run hash_final --spb 2 --hash
run evalv_final --spb 2 --heldout-every 1
echo "=== chain done $(date)"
