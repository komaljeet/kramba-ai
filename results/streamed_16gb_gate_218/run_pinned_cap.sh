#!/bin/bash
# #218 step 1b: the first sweep failed at 6.75 GiB with ~2.1 GiB still available.
# Is that a fixed cap or a floor on free RAM? Three probes:
#   register API (cudaHostRegister, what #217 used)        6 / 7 / 8 GiB
#   alloc with 1 GiB chunks instead of 256 MiB             8 GiB
#   alloc while another process holds 3 GiB of touched RAM 8 GiB
#     fixed cap  -> still fails near 6.75 GiB (Windows pages the hog / others out)
#     free floor -> fails ~3 GiB earlier, near 3.75 GiB
W=/c/Users/komal/Documents/Codes/kramba-ai-wt16gb
O=$W/results/streamed_16gb_gate_218
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
cd $W
API=register SIZES="6 7 8" SUF=_reg bash $O/run_pinned.sh
sleep 20
API=alloc CHUNK=1024 SIZES="8" SUF=_c1g bash $O/run_pinned.sh
sleep 20
$PY -u -c "
import time, ctypes
n = 3 * 2**30
buf = ctypes.create_string_buffer(n)          # zeroed AND touched by ctypes
for i in range(0, n, 4096): buf[i] = b'x'      # make sure every page is resident
print('hog holds 3 GiB', flush=True)
time.sleep(150)
" > $O/hog3.log 2>&1 &
HOG=$!
sleep 25
API=alloc SIZES="8" SUF=_hog3 bash $O/run_pinned.sh
kill $HOG 2>/dev/null
wait
echo "=== cap probes done $(date)"
