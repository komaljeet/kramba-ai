# #194 VRAM spill onset: ONE sweep point per call. From the worktree root:
#   bash results/vram_spill_onset_194/run_point.sh <tag> <ballast_mib>
#
# VERDICT RULE, written BEFORE any point was run:
#  Per point: k5 at production 1B settings, rotation blocks.5-9 -> blocks.0-4 ->
#  blocks.10-14, 4 steps each (--spb 4, 12 steps), one held-out eval at step 4
#  (blocks.5-9 active). Block step time = median of steps 2-4 of its turn (step 1
#  carries the block switch / process warm-up). Pressure = peak reserved incl.
#  ballast / card total (the number the #163 guard checks).
#  1. Baseline = two no-ballast runs (base_a, base_b). Their per-block spread
#     |a-b|/mean is stated. If any block's spread is over 2.5%, a 5% threshold
#     is not resolvable and that is reported instead of an onset.
#  2. Spill at a point, per block: median step > 1.05 x mean baseline median of
#     that block, OR the training process's GPU "Shared Usage" counter peaks
#     more than 64 MiB above its no-ballast peak. Eval: eval_sec > 1.05 x baseline.
#  3. Onset = the lowest pressure (%) at which any block or the eval spills; the
#     safe level = the highest pressure measured with no spill below the onset.
#  4. Positive control: at least one point must show >= 1.5x slowdown or a shared
#     counter rise. If none does, the method is blind: no conclusion.
#  5. A 90% guard is supported only if every block/eval at >= 90% pressure up to
#     90% + margin is clean; the margin is reported in MiB (1% = 61.4 MiB).
#  Stop the sweep on a hard OOM.
set -u
PY=/c/Users/komal/Documents/Codes/kramba-ai/.venv/Scripts/python.exe
O=results/vram_spill_onset_194
TAG=$1; MIB=$2
if tasklist | grep -qi '^python'; then echo "ABORT: a python process is running"; tasklist | grep -i python; exit 1; fi
echo "{\"tag\": \"$TAG\", \"ballast_mib\": $MIB, \"wall\": $(date +%s), \"idle_smi\": \"$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits)\"}" >> $O/idle.jsonl
rm -f $O/stop_$TAG
powershell -NoProfile -ExecutionPolicy Bypass -File $O/sampler.ps1 -Out $O/host_$TAG.jsonl -Stop $O/stop_$TAG &
$PY -u scripts/exp_vram_spill_onset_194.py --ballast-mib $MIB --k 5 --expect-blocks 6 --tag $TAG \
  --out $O --steps 12 --max-steps 2800 --spb 4 --warmup 0 --eval-every 0 --eval-at 4 > $O/log_$TAG.log 2>&1
echo "$TAG exit $?" >> $O/status.txt
touch $O/stop_$TAG; wait
rm -f $O/stop_$TAG
rm -rf $O/ckpt_$TAG
echo "$TAG DONE" >> $O/status.txt
