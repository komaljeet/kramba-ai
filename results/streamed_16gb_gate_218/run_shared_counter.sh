#!/bin/bash
# #218 step 1c: does pinned host memory count against the GPU's WDDM "shared memory"
# budget (dxdiag: 7827 MB = half of RAM)? Read the NVIDIA adapter's Shared Usage
# counter while a probe holds 4 GiB pinned, and again after it exits.
W=/c/Users/komal/Documents/Codes/kramba-ai-wt16gb
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=$W/results/streamed_16gb_gate_218
cd $W
C='\GPU Adapter Memory(*)\Shared Usage'
echo "--- before"; typeperf "$C" -sc 1
$PY -u scripts/probe_pinned_host_218.py --gb 4 --api alloc --hold 25 --out $O/shared_counter_4gb.jsonl > $O/shared_counter_4gb.log 2>&1 &
P=$!
sleep 18
echo "--- while holding 4 GiB pinned"; typeperf "$C" -sc 1
wait $P
sleep 3
echo "--- after"; typeperf "$C" -sc 1
