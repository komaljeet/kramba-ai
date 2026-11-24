#!/bin/bash
# #218 step 1: pinned host RAM sweep, one process and one log per size, nothing else on the GPU.
# typeperf samples system paging for the whole sweep into paging.csv.
W=/c/Users/komal/Documents/Codes/kramba-ai-wt16gb
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=$W/results/streamed_16gb_gate_218
API=${API:-alloc}
SIZES=${SIZES:-"1 2 4 6 8 9 10 11 12"}
CHUNK=${CHUNK:-256}
SUF=${SUF:-}
cd $W
gpu_free() {
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' \r')
  if [ "$used" -gt 400 ]; then echo "GPU NOT EMPTY ($used MiB used) - abort"; exit 9; fi
  echo "gpu used ${used} MiB"
}
typeperf "\\Memory\\Available MBytes" "\\Memory\\Pages Output/sec" "\\Memory\\Pages Input/sec" \
  "\\Memory\\Committed Bytes" "\\Paging File(_Total)\\% Usage" -si 1 -y -o "$(cygpath -w $O/paging_$API$SUF.csv)" > /dev/null 2>&1 &
TP=$!
sleep 5
for gb in $SIZES; do
  gpu_free
  tag=pinned_${API}_${gb}gb${SUF}
  echo "=== $tag $(date)"
  $PY -u scripts/probe_pinned_host_218.py --gb $gb --api $API --chunk-mb $CHUNK --out $O/$tag.jsonl > $O/$tag.log 2>&1
  rc=$?
  echo "exit $rc $(date)"; grep -E "HELD|FAIL" $O/$tag.log
  if grep -qE "status=(fail|guard)|FAIL" $O/$tag.log || [ $rc -ne 0 ]; then echo "STOP: $tag did not hold its size"; break; fi
  sleep 20   # let Windows settle / refill standby before the next size
done
sleep 5
kill $TP 2>/dev/null; taskkill //PID $(ps -p $TP -o winpid= 2>/dev/null) //F > /dev/null 2>&1
echo "=== sweep done $(date)"
