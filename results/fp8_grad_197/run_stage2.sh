# #197 Stage 2 -- run the three arms back to back (one process, one log per arm).
# Plan and ETA: stage2_plan.txt. Do not launch with anything else on the GPU.
S=/c/Users/komal/AppData/Local/Temp/claude/2d79a0b2ce31e602/1a98aaa8-5cec-48a0-97a5-bc9bf11b194a/scratchpad/i197
COMMON="--spb 100 --warmup 20 --eval-windows 64"
bash $S/run2.sh A_fp8      fp8  180 1,100,140,180 $COMMON
bash $S/run2.sh B_bf16     bf16 180 1,100,140,180 $COMMON
bash $S/run2.sh A_fp8_pert fp8  180 1,100,140,180 $COMMON --pert-scale
echo STAGE2_DONE >> $S/done.txt
