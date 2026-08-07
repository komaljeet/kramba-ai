#!/usr/bin/env bash
# Does --block-coordinate cost model QUALITY? (Issue #112 pre-launch gate)
#
# WHY THIS EXISTS. The ~355M pretrain cannot fit on this card without
# --block-coordinate (measured: 6.34 GiB reserved with the default Muon optimizer
# vs a 5.9966 GiB card). So BCD is about to become load-bearing for a ~32-day run.
# But every measurement justifying BCD in docs/state.md is a VRAM measurement --
# #99 documents it purely as a memory optimisation. Nothing has ever measured what
# it does to the trained model, and the mechanism is not obviously free: only 1 of
# 14 blocks has requires_grad at a time, and the outgoing block's optimizer state
# is DELETED on every switch (hvn/core/block_coordinate.py:501-508), so each block
# restarts Muon's momentum from scratch once per cycle. Committing a month of GPU
# time to an untested assumption is the expensive mistake this repo keeps writing
# post-mortems about.
#
# WHY THE REAL SCRIPT, NOT A BENCH HARNESS. eval/scaling_laws.py is this repo's
# compute-matched A/B harness, and it would have been the cheaper option. It is
# deliberately NOT used here: the #112 investigation just found that every VRAM
# envelope in docs/training.md was measured by benches hardcoding VOCAB=8000
# (scripts/bench_block_coordinate_vram.py:31) with no ternary and no Triton scan --
# i.e. a different model than the one that actually trains. A harness that does not
# exercise the real path can be confidently wrong. Both arms here run
# train_edge_conversational.py itself, the same code path the 32-day run will use.
#
# THE COMPARISON. Identical everything except --block-coordinate. Same seed, same
# data mix, same optimizer, same step count -- so the arms are matched on TOKENS,
# which is how the flag would really be used (you do not get extra steps for free
# by using less memory). Control gets 4,200 full-model updates; BCD gets 4,200 steps
# each updating 1/14 of the model. That asymmetry IS the thing being measured, not
# a flaw in the design.
#
# SIZE. d=512/12L (~74M) is used because the control arm must fit WITHOUT BCD, and
# the real 355M config does not. 14 blocks here vs ~26 at 355M -- the same mechanism
# at a smaller block count, so this measures the effect's existence and rough size,
# not its exact magnitude at 355M. Stated as a limit, not hidden.
#
# LENGTH. One full BCD rotation is 14 blocks x 100 steps = 1,400 steps. 4,200 steps
# is 3 complete cycles -- enough that every block has been trained, dropped, and
# retrained twice. A run shorter than one cycle would flatter BCD by never paying
# the optimizer-state-reset cost on a block that had already learned something.
#
# FLAGS. --no-fast in BOTH arms: --fast is ON BY DEFAULT (#129) and forces
# torch.compile, which recompiles on every BCD block switch. Leaving it on would
# confound quality with a compile artifact and slow the BCD arm for reasons that
# have nothing to do with convergence. bf16/tf32/triton-scan are re-added by hand
# to match what --fast would otherwise have set.
set -u

PY=.venv/Scripts/python
STEPS=4200                 # 3 full BCD cycles (14 blocks x 100 steps)
# NOTE: --pretrain-mix takes FOUR fractions (slim,oasst,wiki,fineweb_edu) since
# Issue #120 added the 4th source. The three-value form recorded in older notes is
# now a hard argparse error -- caught by smoke-testing this before the real launch.
COMMON="--arch hvn --optimizer muon --pretrain-mix 0.85,0.0,0.15,0.0 \
        --no-fast --precision bf16 --tf32 --triton-scan --grad-checkpoint \
        --num-workers 0 --seed 0 --max-steps $STEPS \
        --val-fraction 0.05 --val-every 200"

mkdir -p logs results/bcd_quality

echo "=== [$(date)] ARM 1/2: CONTROL (no --block-coordinate), $STEPS steps ==="
$PY -u train_edge_conversational.py $COMMON \
    --checkpoint-dir checkpoints/ab_bcd_control
echo "=== [$(date)] ARM 1 done ==="

echo "=== [$(date)] ARM 2/2: --block-coordinate, $STEPS steps (same tokens) ==="
$PY -u train_edge_conversational.py $COMMON \
    --block-coordinate --bcd-steps-per-block 100 \
    --checkpoint-dir checkpoints/ab_bcd_blockcoord
echo "=== [$(date)] ARM 2 done ==="

echo "=== [$(date)] BCD QUALITY A/B COMPLETE ==="
echo "Val curves are in the per-run CSV logs; compare final val loss at matched steps."
