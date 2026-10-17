#!/bin/sh
# Sequential 1B A/B arms, one log per run, nothing else on the GPU.
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
cd /c/Users/komal/Documents/Codes/kramba-wt-270fuse
run() { name=$1; shift; $PY -u scripts/bench_1b_fusion_sweep.py --out results/fusion_sweep_1b/$name.json "$@" > results/fusion_sweep_1b/$name.log 2>&1; }
run off_all --disable all
run on_all
run loo_read_gate --disable read_gate
run loo_sigmoid --disable sigmoid
run loo_residual --disable residual
run loo_ternary_bf16 --disable ternary_bf16
echo ALLDONE > results/fusion_sweep_1b/ab_done.txt
