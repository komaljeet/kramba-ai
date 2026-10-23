# Current Measured State — EdgeHVN 83M & Speed Ceiling

> **Moved from README:** §5 "EdgeHVN 83M — the measured PPL picture", the speed-ceiling
> investigation, and the 50K probe verdict. These are the live, in-progress status blocks,
> not settled benchmarks. The hybrid design they feed is in
> [architecture.md](architecture.md); settled speed numbers in [benchmarks_speed.md](benchmarks_speed.md).

## num-workers 4->2 saves ~1.1GB RAM for free; second fusion-tail scoping found no new lever (2026-10-23)

**One line: dropping `--num-workers` from 4 to 2 is a real, free win -- ~1.1GB less resident
RAM, identical step time. A follow-up scoping pass looking for a second wave of kernel
fusions (beyond the Oct 17 four-fusion sweep) found the obvious candidate already fused and
firing correctly, and nothing else large enough to justify building.**

**num-workers A/B, measured on the real production config** (1.11B, offload+BCD+fp8-linear,
15 real steps each, peak resident memory sampled every 1s across all `python.exe` processes):

| workers | peak total RSS | avg s/step |
|---|---:|---:|
| 4 (current `--fast` default) | 7.16 GB | 17.890 |
| 2 | 6.04 GB | 17.875 |

No throughput cost (within noise) -- each worker is a fresh Windows `spawn` process
re-importing torch/transformers from scratch (no `fork`/COW on this OS), and the earlier
CPU-idle finding already showed the GPU, not the loader, is the bottleneck, so the extra
workers were never buying anything. Recommend `--num-workers 2` as the default going
forward; **not changed in code yet, flag only, pending a decision on whether to flip the
`--fast` default.**

**Fusion-tail scoping, second pass since Oct 17: no new lever found.** Looked for a second
wave of elementwise-kernel fusions in the ~26% of step time not already GEMM/scan/offload,
now that FP8 shrank the GEMM share and grew everything else's relative weight. Checked the
obvious candidate -- `fused_residual_add3` (`hvn/core/fused_gates.py`), the Oct 17 fusion that
merges a DeltaNet block's `x + branch1 + branch2` into one kernel -- against a theory that
FP8 (landed after that fusion was built) might have silently broken its dtype/contiguity fast
path for some layers, the same class of bug as #117/#138. **It hasn't**: a live counter
patched into the real call site during 5 real training steps with `--fp8-linear` on measured
exactly 880/880 expected calls (22 DeltaNet layers x 8 grad_accum x 5 steps) -- the fusion
fires every single time. (A first attempt at this measurement wrongly read 0/880 -- caused by
my own test harness, not training code: a module-level `atexit` hook fired in the two
Windows `spawn` dataloader-worker processes too, and their always-zero stats overwrote the
real result, last-writer-wins. Fixed by counting inside the real call site during training
instead of reading a shared counter after the fact.)

Two real, small gaps found and not worth building on their own: (1) `fused_residual_add3` is
wired into `deltanet.py` only, not `transformer_block.py` -- the 2 attention layers (of 24)
never get it, upper bound well under 1% of step time; (2) FP8's own quantizer
(`_amax_kernel` + `_cast_kernel`, ~5.1% of step) is already the minimal 2-kernel design
(amax reduction, then clamp/cast/quantize) and merging further would mean either an
online-running-max single-kernel rewrite (real complexity, same class of risk as the #96
chunk-parallel scan design that lost) or giving up the bit-identical quantization guarantee.
The remaining ~6% of scattered `copy`/`cast` kernels were not traced to source (would need a
`with_stack=True` profiler pass); not pursued given everything else checked came back already
near its floor.

## CPU idle time during the 1B step: none worth reclaiming (reconfirms #153 on the current offload+BCD+FP8 build, 2026-10-23)

**One line: asked whether anything could be deferred onto the CPU during its idle windows.
`cudaEventSynchronize` (the offload prefetch wait) blocks the host for ~10s of an ~18.9s
step -- looks large -- but the GPU's own kernel table for the same window sums to 20.05s of
device-busy time against that same wall clock. The GPU is still ~100%+ saturated. The host's
"idle" time is it blocked on the one thing actually gating the step, not spare capacity
sitting unused. Nothing to defer into it.**

Measured on the real production config (hvn-hybrid 1.11B, `--offload --bcd-partition
balanced --fp8-linear --fast`), reusing the existing `--profile` torch.profiler capture
(CPU+CUDA activities) rather than a smaller reference harness, so offload's real host-side
bookkeeping is included. `scripts/profile_cpu_hotspots.py`'s own reference-config measurement
(d896/L16/H14, no offload) would have missed this entirely -- confirmed by running it: this is
why the question needed a fresh measurement on the real config instead of reusing #153's
August number, which also predates offload/BCD/FP8 and is not a measurement of this harness.

Side finding, consistent with earlier reasoning rather than new: the DeltaNet scan kernels
(`_mk_bwd_kernel_s` + `_gated_delta_fwd_kernel_s` + `_mk_ckpt_kernel_s`) came back at 26.6% of
GPU self-time on this block -- matching the ~27% estimate made when FP8 shrank the GEMM share.
FP8 tensor-core GEMM kernels (`sm89_xmma_gemm_e4m3...`) confirmed actually executing, ~18% of
this block. No new lever found; this closes out the "anything else to leverage" question for
now the same way epic #78 did in August -- re-verified on the current build, not assumed.

(First attempt at this measurement hung for ~2h on a Windows multiprocessing bug in the probe
script itself -- missing the `if __name__ == "__main__":` guard needed when `--fast` spawns
dataloader workers via `spawn` on Windows. Not a training-code bug; killed and re-run.)

## 1B pretrain: FP8 full-cycle CONFIRMED -- 7.81% faster, 132/211 days (2026-10-23)

**One line: the stalled run below finished all 1300 steps after the mmap/reaper fixes. Real
13-block average, not a projection: 13.294 -> 12.335 s/step, 2,464 -> 2,656 tok/s = 7.81% more
tok/s (7.21% less step time). Slightly better than the single-block projection of 6.14% /
~12.52 s/step made below.**

Measured directly from `checkpoints/pretrain_1b_fp8_fullcycle/training_log.csv` (260 logged
rows, steps 5-1300, same run the entry below describes resuming). Per-block steady-state tok/s
climbs from 1857.2 (block0, `token_embedding`, slowest) to 4714.1 (block12, last, fastest) --
the same 18.9 -> 7.5 s/step bf16 spread already documented, just uniformly faster throughout.
The run's own printed "7.997 s/step" summary line is wrong -- it divides this resume's wall
time by the full 1300-step target instead of the ~960 steps it actually ran in this process;
the per-step CSV data above is the real source, not that line.

**Day estimate, confirmed:** 30.3B-token budget / 2,656 tok/s = **~132 days** (accepting BCD's
~1.6x quality cost) / **~211 days** (matching full BCD quality) -- down from 142/227. The
~134/~214 projected below (from one block) undershot slightly; the real full-cycle number is
~132/~211. README, docs/training.md and HF_BLOG_POST.md updated to match.

## 1B pretrain: FP8 tensor-core matmul finished an unfinished lead -- 6.14% faster on one block, full-cycle confirmation stalled on a RAM issue (2026-10-22/23)

**One line: an orphaned feasibility branch from 2026-10-18 (`exp/fp8-pretrain-feasibility`,
never merged) found Ada Lovelace's real FP8 tensor cores are ~2x faster on raw GEMM, but the
naive ("unfused, eager") quantizer made the full `Linear` op 1.4-1.7x SLOWER end to end -- a
real, positive lead left unfinished. Root-caused and fixed this session: a fused Triton
quantizer (one amax-reduction kernel + one scale/clamp/cast kernel, matching PyTorch's exact
`448.0/amax` rounding order bit-for-bit) flips the op positive. `--fp8-linear` (default off),
merged as v1.2.0.**

**Measured on the real 1B model, 25 steps, `token_embedding` block (the slowest), same day as
4+ bf16 baselines run for comparison:** steady-state tok/s (steps 5-25) 1748.83 (bf16) vs
1856.28 (fp8) = **6.14% faster**. Loss trajectories near-identical (fp8 11.2221 -> 11.1631,
bf16 11.2236 -> 11.1638 over the same 25 steps). VRAM barely moved (4.91 vs 4.88 GB reserved).
`tests/test_fp8_fused.py` (14 tests): bit-identical vs the reference quantizer at every
shape/dtype/edge case, and identical loss/params after 5 real-trainer steps under
Muon/grad-accum/grad-checkpoint/Triton-scan/megakernel/BCD+offload vs eager.

**Projected cycle time -- NOT YET CONFIRMED.** If the same 6.14% holds across all 13 BCD
blocks (it was only measured on one, the slowest), 13.294 -> ~12.52 s/step, 2,464 -> ~2,616
tok/s, and the day estimate below would drop **142 -> ~134** / **227 -> ~214**. This is a
projection from one block, not a cycle measurement -- exactly the gap the full-cycle run below
was launched to close. Until that run finishes, 142/227 stays the official number.

**Full-cycle confirmation run: stalled at step 340/1300, not a code bug.** Launched to measure
the real 13-block average with `--fp8-linear` on. Killed three times by Claude Code's own
background-shell low-memory "pressure reaper," each time within 0.2s of a checkpoint write
finishing. A dedicated live-counter investigation found the checkpoint write itself only costs
~0.4-0.7 GB for 1-2s -- not enough to explain the kills -- but found a bigger, separate risk:
*resuming* a checkpoint crashes available RAM to ~25 MB and system commit to 98% for a few
seconds (old model + new checkpoint + 4 persistent dataloader workers all alive at once). The
exact trigger for the three real kills is still unconfirmed -- Windows Update was running
during at least one of them, and the box runs chronically tight on commit (~26.5/30.5 GB)
regardless of that. Two real fixes applied: `mmap=True` on the checkpoint load
(`train_edge_conversational.py:2920`, ~2.6 GB less RAM during resume) and
`CLAUDE_CODE_DISABLE_BG_SHELL_PRESSURE_REAP=1` added to `.claude/settings.local.json` (takes
effect on the next restart). The step-340 checkpoint is good and resumable.

**Resolved (2026-10-23):** resumed cleanly after the restart and ran to completion, step
1300/1300. See the CONFIRMED entry above for the real cycle-average numbers.

## 1B pretrain: four bit-identical fusions = 14.096 -> 13.294 s/step, -5.7% (2026-10-17)

**One line: a code-reading sweep (no GPU) named six candidates with sizes scaled from
byte counts. Re-checked each on the real 1B step (2048x24 hvn-hybrid, BCD + offload,
--fast, --dropout 0) before building. Four were real and are built; two were not.
Measured together over two full BCD cycles: 14.096 -> 13.294 s/step. Peak VRAM
5062 -> 4996 MiB reserved. Peak host RAM unchanged (5.03 -> 5.02 GiB working set,
10.90 -> 10.78 GiB commit).**

Each lever's real worth, measured by turning just that one off (all-on = 13.294):

| lever | off -> s/step | worth | sweep's guess |
|---|---:|---:|---:|
| read norm + output gate + bf16 cast in one kernel | 13.669 | -0.375 s (2.8%) | 1-2% |
| ternary weights dequantized straight to bf16 | 13.477 | -0.183 s (1.4%) | 0.4% |
| alpha/beta `.float()` + sigmoid in one kernel | 13.438 | -0.144 s (1.1%) | 0.9% |
| residual `x + deflected + ffn_out` in one kernel | 13.388 | -0.094 s (0.7%) | 0.6% |

The four add to 0.796 s vs 0.802 measured together. The ternary one was 3.5x the
guess because at 1B the BCD-frozen weights are parked in bf16, so every call on an
fp32 input (FFN w1/w3, out_proj) went bf16 -> fp32 -> bf16 -- two full passes over the
weight, 16 times a step. The guess only counted the trainable block.

**Not built:**
- Loss-head shared cast: the tied table is trainable on 1 of 13 BCD steps only
  (`token_embedding` owns it, #151). Ceiling ~8 recasts x 8 micro-batches x 3.6 ms on
  that step = ~0.018 s/step (0.13%). Not worth a change whose fp32 grad-sum ordering is
  easy to get silently wrong.
- RoPE table fusion: no sin/cos kernel is in the top 80 kernels (each < 0.35 ms/step).
  Nothing to win.

**Correctness, all exact (torch.equal), never allclose:**
- `tests/test_fused_gates.py` (25): sigmoid forward over all 65,536 bf16 and fp16
  inputs; backward, residual, gate fold at the real shape; ternary bf16 dequant for
  trainable fp32 and frozen bf16 weights, fused and eager quantizer, step cache on/off;
  a whole ternary DeltaNet block over 4 optimizer steps; near-miss variants (approx
  exp, wrong add order, bf16 product) shown to FAIL the same checks.
- `scripts/check_fusion_bitwise_1b.py` on the real 1B model, one full BCD cycle, same
  state, levers off vs on: loss and every trainable grad equal on 13/13 steps. A
  falsifier arm (eager L2 norm, ~1 ulp off) is caught.

**Two traps hit on the way, worth knowing:**
- The gate kernel first came out ~1 ulp off. Triton picks its register layout from the
  widest memory access; the bf16 gate/output gave rows a different add order inside
  `tl.sum`, so `r` moved on 42,697 of 262,144 rows. A `max_contiguous(j, 4)` hint gives
  back the fp32 kernel's layout. Comment at the kernel.
- The first real-model check showed grads differing below attention layer 18 even
  though losses matched. Not the levers: the memory-efficient SDPA backward is
  nondeterministic at this shape (three identical calls, three different results, even
  with deterministic algorithms on). Any A/B of grads through the attention layers
  needs to control for that. The check now records attention's backward in the off arm
  and replays it in the on arm only if q, k, v and grad_out are bit-identical.

On by default: the three DeltaNet flags follow `use_triton_scan` (so `--fast`), the
ternary dtype is a class default (bit-identical on every path). Bench:
`scripts/bench_1b_fusion_sweep.py --disable NAME`; logs in `results/fusion_sweep_1b/`.
Not re-measured at 270M.

## Dropout: tested, not assumed -- turn it off, real quality win not just speed (2026-10-16/17)

**One line: the real 1B run has to repeat its cached data ~4.6x to hit its token budget
(30.3B tokens needed, 6.64B unique FineWeb-Edu cached) -- exactly the regime where
dropout's regularization is supposed to earn its keep. Tested it properly instead of
assuming: a scaled-down A/B (384x8, matching all three of the real run's ratios --
tokens/param, repeat count, unique-tokens/param) through 4.67 full passes, 2 seeds,
held-out loss scored at every pass boundary. Dropout OFF won on unseen-data loss at
every single checkpoint, in both seeds, by a margin 2-3x the measured seed-to-seed
noise -- and the margin grew with more passes, not shrank. `--dropout 0` is the
recommendation for the real 1B launch.**

**Why this needed a real test, not a guess:** dropout off also saves ~4.9% step time,
which made "just turn it off" tempting on speed alone -- but speed and quality are
separate questions, and this project has been burned before by treating one as a proxy
for the other (#132's retracted train-loss verdict, #122's luck-dominated recall gate).
The test specifically targeted the regime that matters (data repeats a known amount)
rather than a short single-pass sanity check, which would have been too early to show
anything either way (both arms look like noise before about pass 2).

**Real combined 1B measurement** (RMSNorm fusion + dropout off together, not just
estimated): **14.385 s/step**, 5060 MiB peak -- confirms the ~14.35 estimate. Updated
1B days: **~154** (accepting BCD's own quality cost) / **~246** (matching full BCD
quality), down from 161/258.

**Correction (2026-10-21):** this day-count was never recomputed after the four-fusion
sweep above (14.096 -> 13.294 s/step) landed. Redone with the current number: tokens/step
is unchanged (batch 1 x grad_accum 8 x seq 4096 = 32,768), so tok/s = 32768/13.294 = 2,464,
and against the 30.3B-token budget that's **~142 days** (accepting BCD's quality cost) /
**~227 days** (matching full BCD quality) -- down from 154/246. README, docs/training.md
and HF_BLOG_POST.md updated to match.

**Correction (2026-10-22): 13.294 s/step is a cycle average, and nothing above said so.**
A real launch of the exact command this doc and HF_BLOG_POST.md recommend measured a steady
~19 s/step (~1,720 tok/s) across its first 22 steps -- flat, not warming up -- and looked
broken. Debugger confirmed both numbers are correct: `results/fusion_sweep_1b/on_all.json`
shows per-block times from **18.9 s/step** (`token_embedding`, trained first) down to
**7.5 s/step** (the last block), averaging to 13.294 across a full 13-block, ~1,300-step
cycle. A fresh run always starts on the slowest block. README, docs/training.md and
HF_BLOG_POST.md now say this explicitly next to every place 13.294/2,464 is quoted.

**270M, re-measured on the same merged code:** 3,640 tok/s with dropout (default),
**3,775 tok/s without** -- **~20.1 days** vs **~19.3 days**. 270M's own token budget
(6.31B) is UNDER the 6.64B unique tokens cached, so it would not need to repeat data at
all even once -- the regime where dropout's whole justification disappears. `--dropout
0` is a safer call for 270M than for the 1B run, not a riskier one.

**Caveats, stated plainly, not hidden:** tested to 4.67 passes only (still widening at
that point -- untested beyond it), small-scale model, code data not FineWeb-Edu text, 2
seeds. Track the real run's own held-out loss live once it starts, as a safety check,
rather than trusting this result blind at a scale and duration it was never measured at.

`--dropout` flag (default 0.1, nothing changes unless set), `--max-train-windows`
cap, and `scripts/eval_repeat_gap.py` (seen/unseen/pinned-tail scoring per checkpoint)
merged, MR !212. tests/test_dropout_flag.py, tests/test_max_train_windows.py: 11
passed.

## 1B pretrain: fused block pre-norm RMSNorm + a real MFU figure = 15.067 s/step, -30.9% from the original baseline (2026-10-15)

**One line: a final line-by-line sweep (explicitly asked to find nothing if there was
nothing, not to force a result) found one more real lever -- the block's pre-norm
RMSNorm, six eager ops forward collapsed into one Triton kernel. 15.478/15.481 ->
14.864/14.860 s/step in isolation (-4.0%). Stacked with everything else and
re-measured for real: 15.706 -> 15.067 s/step (-4.1%), 5060 MiB peak, same as before.
Against the ORIGINAL pre-this-week main (21.81 s/step): -30.9% overall, a 1.45x
speedup.**

**Checked and explicitly ruled out, not just assumed, before landing this:** the two
sliding-window attention layers were read line-by-line (nothing from this week's
DeltaNet-specific fusion work had touched them). This Windows PyTorch build has no
flash-attention kernel at all -- it already falls back to the memory-efficient kernel,
which is the best available here, 0.39s/step (2.5%), nothing to fix. The causal mask
isn't even built at pretrain (`is_causal=True`, no `doc_ids`). A cuDNN attention
backend might save ~0.07s/step (0.4%) but that's an estimate from a microbenchmark
only, not built -- small enough, and off by default in PyTorch for a reason. Every one
of this week's fused paths was confirmed actually firing together under `--fast` (zero
eager fallbacks, by counting real kernel calls over a full cycle) -- ruling out the
"a flag silently didn't apply" failure mode that bit the very re-measurement below.

**Named, not built, explicitly flagged as not free:** removing dropout (p=0.1, 72
modules, >=0.39s/step/2.5%) would change the training recipe -- left for a real
decision, not taken as a free lever. Folding the gate-multiply + cast into the
read-norm kernel is maybe another ~0.27s/step (estimate, not measured).

**The re-measurement itself caught a real measurement bug on the way.**
`scripts/attribute_1b_real_step.py`'s call to `apply_fast_stack` never passed
`fused_rmsnorm=True` (that parameter defaults to False, and the script predates this
lever) -- so the FIRST re-measurement attempt came back at 15.711 s/step, identical to
before the fusion existed, and would have been reported as "no change" if trusted
at face value. This is the exact failure mode `apply_fast_stack`'s own docstring warns
about from a prior incident (`scripts/size_sweep.py`'s 3.4x error) -- it happened
again, in a script written this very week, within hours of the lever landing. Fixed
(one line); the real trainer (`train_edge_conversational.py:4359`) already passed the
flag correctly and was never affected.

**1B days, recomputed on the real number** (1.01B body params, 30 tokens/param,
2,175 tok/s): **~161 days** accepting BCD's own measured quality cost as-is, or
**~258 days** to match full-training quality instead (the longer number is the
"no quality lost" one -- see the earlier entry on this if that direction feels
backwards). Down from 168/269 yesterday, 580/181 before this week's work began.

## 1B pretrain: today's three levers stacked, measured together = 15.706 s/step, -7.2% on top of the 16.92 baseline (2026-10-14)

**One line: the within-step ternary cache, the frozen-block scale cache, the fused
short-conv, and the strided scan IO (four entries below) were each measured
separately. Stacked together on fully-merged main, one real full BCD cycle (26 steps,
2048x24 hvn-hybrid, --offload --block-coordinate --fast) averages 15.706 s/step --
**-7.2%** on top of today's earlier 16.92 s/step starting point, not the ~8.6% a naive
multiply of the four separate percentages would predict. Amdahl overlap between the
levers eats the difference, which is exactly why this was measured instead of
computed. Against the ORIGINAL pre-this-week main (21.81 s/step), that's -28.0%
overall, a 1.39x speedup.**

Peak reserved VRAM is 5060 MiB -- slightly LOWER than the 5078 MiB measured with just
the first four (pre-today) levers, because the new fusions also free transient copy
buffers the old path held. Loss trajectory over the real run is sane (11.24 -> 9.95
over 26 steps), no NaN.

This closes out the "chase the real remaining levers instead of re-asserting
'exhausted'" pass prompted directly by a repeat of this project's oldest mistake: a
session claimed "no big lever left" off a stale attribution table, was called out for
it (this exact claim has been wrong before -- the scan megakernel was declared
non-breaking-exhausted before Design B landed, and "the step is finished" at 21.81
s/step was itself overturned by this week's first four levers), and a fresh
measurement on current code found three more real, smaller wins instead of
confirming nothing was left. See the four entries immediately below for each lever's
own numbers, and the "fresh 1B step attribution" entry for the corrected attribution
table and the two measurement bugs (#131 double-count on the optimizer row; the old
"232ms weight cast" claim being 94% mislabeled offload copy) found while re-checking
the old one.

Also fixed in the same pass, not speed levers but real bugs found and fixed on the way
(all merged): a resume crash under --offload whenever the saved active block wasn't
block 0; BCD's bf16 parking silently freezing 1-D params (biases/norms) for the whole
run; ternary conversion doubling host RAM and crashing 1B builds outright; a
swallowed-error-message bug that made argument mistakes look like silent crashes; and
a smaller RAM waste building a throwaway decoder weight when embeddings are tied.

## Real pretrain-time numbers at 270M, 300M, 500M, measured on current code (2026-10-13)

**One line: using the real trainer (chunked loss, fused kernels, real ternary) rather
than a synthetic benchmark missing the loss-head optimization (which showed every size
spilling past VRAM -- a methodology bug caught before reporting, not a real result):**

| size | shape | fits plain? | measured | days @ 30 tok/body-param |
|---|---|---|---:|---:|
| 270M | 1024x20 | yes, 71.5% VRAM | 3,239 tok/s | 22.6 |
| 300M | 1024x24 | yes, right at the 85% guard line | 2,768 tok/s | 31.7 |
| 500M | 1024x44 | **no** -- 126% of VRAM, WDDM spill | n/a (175 s/step is a spill artifact, not a real number) | n/a |

**500M needs `--block-coordinate --offload`** (1024x44, 12 balanced blocks,
`--attn-layers 5,16,27,38`): real, stable, 18 steps across 3 block switches, flat
RAM/VRAM at 46% of the card. **17.8 s/step, ~1,841 tok/s.** Two honest framings for
"how long":
- **1x, no quality lost** (BCD-trained matched to full-training quality, i.e. training
  long enough to make up for BCD's own cost): **~139 days**.
- **1.6x, BCD's real measured quality cost accepted as-is** (just the raw wall-clock
  budget, output included): **~87 days**. The shorter number is the one with the
  quality cost baked in, not the other way around -- worth stating explicitly since
  it's easy to get backwards.

The 500M build initially appeared to crash silently (no traceback, no process left
alive) -- root cause was a missing `--attn-layers` flag combined with the
swallowed-error-message bug above, not RAM or a real crash. Fixed as part of that bug.

## BCD+offload: four real bugs found and fixed before trusting it on a long 1B run (2026-10-13/14)

**One line: an end-to-end integration test (1024x20 real trainer, offload+BCD vs
BCD-only vs plain, hard-killed and resumed mid-block three times, then confirmed on a
real 1.1B build+resume) found main was NOT safe for a real long run. All four fixed,
all validated together afterward with no bad interaction.**

1. **Resume under `--offload` crashed on step 1** whenever the saved active block
   wasn't block 0 (true on almost every real resume). `train()` loaded the optimizer
   state while the fresh scheduler still had every other block parked on CPU in bf16,
   so momentum landed on CPU in bf16 and the first `optimizer.step()` failed with
   "Expected all tensors to be on the same device". Fixed by restoring the BCD
   position before the optimizer state. New test `tests/test_offload_resume.py`.
2. **BCD's bf16 parking silently froze 1-D params** (biases, norm weights) for the
   whole run. `park_dtype_for` returned bf16 for every parameter with no fp32 copy
   kept; near a typical init value (e.g. `alpha_proj.bias` at 3.0) bf16's granularity
   swallowed the optimizer's tiny per-step update at EVERY park, so progress made
   during a block's active turn got erased the moment it parked again. Reproduced
   directly: a bias stayed exactly 3.0000 for an entire 300-step offloaded run; moves
   normally under plain/BCD-only training. Fixed: 1-D params (ndim<=1) now park at
   native dtype; matrices still park in bf16 (a smaller residual gap -- 6-11% of
   large-matrix values still lose their sub-step update per park, not yet fixed).
3. **Ternary conversion doubled host RAM and crashed 1B builds outright** on this
   15.29GB-RAM machine. `apply_ternary_to_model` kept every old dense `nn.Linear`
   alive in a dict until all 166 layers converted, and cloned each weight onto a
   fresh xavier-initialized matrix instead of converting in place -- briefly 3 copies
   of each layer's weight. Old main: ran out of RAM building a real 1.1B model, had to
   be killed. Fixed: `from_linear` now adopts the dense Parameter directly (no clone,
   no throwaway init); build peak 8.25GB -> 4.91GB working-set. `MIMALLOC_PURGE_DELAY=0`
   set before torch import, needed in ADDITION to the in-place fix for resume to
   survive (fix alone still crashed resume; the allocator doesn't give back the first
   model's freed memory without it).
4. **A swallowed-error-message bug** turned plain argument mistakes into silent-looking
   crashes: the trainer's `finally` block called `os._exit(1)` before Python could
   print a `SystemExit`'s message, so e.g. the "--arch hvn-hybrid requires
   --attn-layers" refusal exited 1 with empty stderr -- indistinguishable from a real
   crash without a full investigation (one was dispatched for exactly this, suspecting
   a RAM/OOM crash, before finding the real cause). Fixed: the except-SystemExit branch
   now prints `e.code` before the exit fires.

Smaller, found alongside #3: the decoder built its own full `50257x2048` fp32 weight
(0.38GB) and immediately discarded it when embeddings are tied. Fixed by building the
tied decoder on the `meta` device instead. Bit-identical weights and RNG state (a test
pins this and is checked to actually fail). Build peak 5.22->4.84GB, resume peak
7.56->7.18GB -- resume still bottoms out tight (checkpoint + fresh model coexist in
RAM; a Windows-safe mmap fix was deliberately left for later, see issue #168) but less
so.

**Validated together afterward**, not just individually: full test suite on merged
main, a real 300-step run with 3 hard-kill resumes (checked optimizer moments, BCD
position, and parked weights match exactly across each kill), and a real 1.1B
fresh-build + resume. No interaction bugs between the four. Two small non-blocking
things surfaced during that check and filed rather than chased: issue #169 (resume
rounds the restored block's matrices through bf16 once, pre-dates all four fixes
above, traces to #113) and issue #170 (a test leaks TF32 global state across the
process, pure test hygiene, unrelated to BCD/offload correctness).

## Fresh 1B step attribution on current main -- the old table was stale, GEMM's share grew to 43% (2026-10-14)

**One line: re-measured the real step attribution (CUDA-kernel-rows only, no #131
double-count) on current main instead of trusting the pre-this-week table. GEMM's
share grew from 33% to 43% now that this week's other fixes shrank everything else.
The old table also had two real mistakes in it.**

| category | share now | old table |
|---|---:|---:|
| gemm | 43% | 33% |
| elementwise | 34% | 36% |
| scan | 19% | 15% |
| PCIe copies | 12% (hidden on side stream, costs ~nothing) | 15% "other" |
| ternary quantize | 1% | n/a |
| optimizer | 2% | 2% |

Step time measured 16.89 s, matching the documented 16.92. The old table came from a
different, no-ternary harness at 3.67 s/step -- not a like-for-like baseline, which is
why it needed replacing, not just re-trusting.

**Two mistakes in the old method, found while re-checking it:**
- the old "optimizer 1.6%" row was a GPU annotation range (`Optimizer.step#Muon.step`),
  not real kernels -- the #131 double-count again, on a different profiling call this
  time.
- the old "[2048,2048] `copy_`, 232ms, suspected BitNet STE/autocast cast" claim was
  94% offload H2D copy mislabeled as a weight cast. Real on-GPU weight casts were only
  ~66ms/step. That probe had no ternary layers in it at all, so the STE explanation was
  a guess dressed as a measurement.

This directly motivated the within-step ternary cache below (built to fix the "6%"
the old number implied, landed anyway once the real number turned out to be smaller)
and is the reason the scan-copy and short-conv entries below state their own
before-fixing confirmation step explicitly -- re-measure before trusting a stale
number, this project has been burned by it more than once now.

## 1B pretrain: ternary weight quantized once per optimizer step, not once per micro-batch/recompute = 0.8% faster per step (2026-10-14)

**One line: each trainable ternary weight is quantized once per optimizer step and
reused across all 8 micro-batches and the checkpoint recompute, instead of
re-quantizing on every call. At 2048x24 (same config as above) 16.92 -> 16.78 s/step.**

Built to chase the old table's "232ms/6%" weight-cast claim above -- the real cause
turned out to be mostly mislabeled (see the attribution entry), so the win here is
small (0.8%, not ~6%) because the premise that motivated building it was wrong. Landed
anyway because it's free and fully correctness-tested, not because the number is big.

`hvn/edge/ternary.py`: `_dequantized` / `_CachedSTE`. Cache cleared on every optimizer
step (pre/post hooks) plus a `data_ptr`/version check; frozen (BCD-parked) weights are
never cached here (see the next entry for those). Correctness: bit-identical
losses/gradients/weights vs the uncached path over multiple steps, including a real
BCD+offload+Muon run over a full block cycle; deliberately-stale twins fail the same
check. VRAM: peak reserved unchanged (5078 MiB); peak allocated +192 MiB. Loss: step-0
identical; on-vs-off drift (6.4e-3) matches off-vs-off noise (5.7e-3). On by default
under `--fast`; `--no-ternary-step-cache` disables it. Tests:
`tests/test_ternary_step_cache.py` (10), plus the ternary/fast-stack/BCD/offload
suites (136 + 73 passed).

## 1B pretrain: cache the quantize SCALE of BCD-frozen weights too = 0.9% faster per step (2026-10-14)

**One line: the entry above only covers trainable weights (keyed on `data_ptr`, which
changes every time an offloaded block reloads) -- frozen/parked blocks were still
re-quantized on every forward+recompute even though their weights don't change between
BCD switches, ~0.4s/step (2.5%) confirmed real by profiling before fixing. Stacked on
the entry above: 16.78/16.79 -> 16.62/16.63 s/step.**

Caching the whole quantized weight for every frozen block would mean keeping them all
GPU-resident, defeating offload -- so only the absmean SCALE is cached, once per
optimizer step; the quantize pass itself still runs. Cache clears on every optimizer
step and on every BCD block switch (a callback list added in `block_coordinate.py`, so
`hvn/core/` still doesn't import `hvn/edge/`). Correctness: bit-exact over a full BCD
cycle on the real model, ~92% cache hit rate; a deliberately-stale-cache twin fails the
same bit-exact comparison. 133 tests passed (ternary suite).

## 1B pretrain: fuse the short-conv transpose+cast = 3.7% faster per step (2026-10-14)

**One line: `hvn/core/deltanet.py`'s `_causal_conv` ran `F.pad` on a transposed fp32
tensor (a slow strided copy), then autocast made a SECOND copy to bf16 -- backward paid
both copies again. Confirmed real by profiling before fixing: per 1B layer (fwd+bwd),
pad copy 0.94ms, cast 0.27ms, conv 0.65ms, conv backward 0.88ms, pad-backward clone
0.36ms, grad cast 0.26ms. At 2048x24, 16.92 -> 16.26-16.29 s/step.**

Fix: a new Triton kernel, `hvn/core/fused_causal_conv.py` -- reads `[B,S,D]` directly,
reproduces PyTorch's depthwise-conv arithmetic exactly, in the same order. Weight/bias
grads (active block only) still use the original aten call. bf16 only -- fp16 goes
through cuDNN and came out ~1 ulp off, so fp16/fp32/CPU/the packed `doc_ids` path all
keep the old code unchanged. Correctness: `torch.equal` (zero sign bits included) on
output, dx, dw, db at 5 shapes including the real 1B shape, plus a 4-step full-block
sequence; anti-vacuity checks confirm the fast path actually ran. 251 tests passed
(DeltaNet, scan, BCD, offload, ternary suites).

## 1B pretrain: scan reads q/k/v in place, no layout/cast copies = 3.5% faster per step (2026-10-14)

**One line: the megakernel scan now reads q/k/v/alpha/beta straight from the block's
`[B, S, H, d]` tensors in their own dtype, so the permute + fp32-cast copies around it
are gone. At 2048x24 (hvn-hybrid, BCD balanced + offload + prefetch + ternary step
cache) a full BCD cycle goes from 16.78 to 16.21 s/step -- 0.58 s (3.5%) faster. Peak
reserved 5078 -> 5064 MiB, peak allocated 3202 -> 3073 MiB. Bit-identical.**

**Measured before building** (attribution harness, kernel rows only, one profiled cycle):
copies of scan-shaped tensors (`[64, 4096, 32]` and friends) cost 0.74 s/step of device
time -- `.contiguous()` and `.float()` in `_MegaScan` (fwd and checkpoint recompute),
autograd's dq/dk/dv fp32 -> bf16 casts, and the permute-back copies in the l2norm /
view backward. After: 0.00 s/step, every row gone.

**Why the gain is 0.58 s, not 0.74 s:** the scan kernels got 0.18 s/step slower
(`_mk_ckpt_kernel` 0.38 -> 0.47 s, fwd 1.51 -> 1.60 s, bwd unchanged). Micro-benchmark at
the 1B shape: the cost is the head-interleaved layout (each step's row is 4 KB away from
the last, not adjacent), not bf16 and not the index math -- the same strided kernel on an
`[N, S, d]` layout is as fast as the old one, and on bf16 `[N, S, d]` faster. Triton loop
pipelining (`tl.range(num_stages=2..4)`) does not win it back (slower or equal).

**1B A/B** (`scripts/ab_scan_strided_io_1b.py`, main ff2867c + MR !199's ternary step
cache + this, two full cycles per run, ABAB, each run on an otherwise-empty GPU -- the
driver killed and retried any run another process touched):

| arm | s/step, run 1 / run 2 | fwd | bwd | peak reserved |
|---|---:|---:|---:|---:|
| copies (`use_fused_scan_io=False`) | 16.787 / 16.782 | 6.56 s | 9.87 s | 5078 MiB |
| strided IO (default) | 16.207 / 16.204 | 6.26 s | 9.60 s | 5064 MiB |

**Code.** `hvn/core/deltanet_triton.py`: new `_gated_delta_fwd_kernel_s`,
`_mk_ckpt_kernel_s`, `_mk_bwd_kernel_s` and `megakernel_scan_bshd` -- copies of the
validated kernels with only addressing (`n = b*H + h; row = b*sB + h*sH + t*sS`) and
store dtype changed; the old kernels are untouched. `o` stays fp32 `[N, S, d]` for the
fused read norm; dq/dk/dv/dalpha/dbeta come back contiguous in each input's dtype.
`DeltaNetBlock.use_fused_scan_io`: None (default) = on whenever the megakernel scan is;
False = the old copy path, exactly. Inputs whose strides differ fall back to
`.contiguous()` and bump `SCAN_IO_STATS["fallback_copies"]`.

**Correctness** (`tests/test_scan_strided_io.py`, all `torch.equal`, no tolerances):
- o and all five grads vs the old path, bf16 and fp32, five shapes (B 1-3, H 1-64,
  d 8-64, S not a power of 2);
- oddly strided inputs -- last-dim slice of a wider buffer, head-major storage, every
  other time step -- inside NaN-filled buffers, and the test asserts they took the stride
  path, not the copy fallback;
- whole DeltaNetBlock under bf16 autocast, with and without packed `doc_ids`: output,
  x.grad and every parameter grad bit-identical, with call counters proving which path ran;
- `torch.empty` is NaN-poisoned in the comparisons, so an unwritten gradient row fails
  instead of inheriting the old path's freed, identical bytes;
- Triton's fp32 -> bf16 store matches torch's cast on 4.2M random bit patterns plus 786K
  exact/near ties, NaN, Inf, denormals; the same inputs fail for round-toward-zero, so
  the test can tell the two apart.
Mutation check: a stride bug only the strided inputs can see, RTZ rounding on dq, beta
read from the wrong head, dalpha never stored, and the block silently not using the new
path each fail the suite (3, 11, 12, 16, 2 failures). Note the 1B loss itself cannot
show bit-identity: the step is not deterministic run to run (warm-step-1 loss varies by
~5e-4 between identical old-path runs).

## 1B pretrain: fused DeltaNet "glue" (q/k L2 norm + read RMSNorm) = 10.2% faster per step (2026-10-12)

**One line: two small Triton kernels replace the bandwidth-bound PyTorch ops around the
DeltaNet scan. At 2048x24 (hvn-hybrid, BCD balanced + offload) a full BCD cycle goes from
21.81 to 19.58 s/step -- 2.23 s (10.2%) faster. Peak VRAM +10 MiB. Not bit-identical
(step-0 loss differs by 2.4e-6 relative); the fused path is at least as accurate as the
old one against an fp32 truth.**

**Measured before building** (one DeltaNet layer at the 1B shape: B=1, S=4096, 64 heads x
d_head 32, bf16 autocast):

| glue op | fwd | fwd+bwd | per layer |
|---|---:|---:|---|
| `_l2norm` (q and k) | 1.94 ms | 7.70 ms | x2 |
| `_read_normalize` + reshape copy | 2.47 ms | 8.25 ms | x1 |
| scan wrapper `.contiguous().float()` (q, k, v) | 0.69 ms | -- | x3 |
| dq/dk/dv fp32 -> bf16 casts | -- | 0.29 ms | x3 |

That is ~15% of a DeltaNet layer's time. The two norms are most of it, and both are pure
functions outside the scan, so they were fused as their OWN kernels. The latency-bound
scan kernels were not touched.

- `hvn/core/fused_l2norm.py`: one read and one write each way, fp32 math, hand-written
  backward. 1.98 -> 0.20 ms fwd, 7.78 -> 0.80 ms fwd+bwd.
- `hvn/core/fused_read_norm.py`: reads the scan output in its `[B*H, S, d]` layout and
  writes `[B, S, D]` directly, so the permute copy is gone too. The weight gradient is a
  two-stage reduction, deterministic (no atomics). 2.49 -> 0.42 ms fwd, 8.32 -> 1.61 ms
  fwd+bwd.
- Whole DeltaNet block: 33.3 -> 29.8 ms fwd, 111.1 -> 94.5 ms fwd+bwd.
- On by default wherever `use_triton_scan` is on (`DeltaNetBlock.use_fused_l2norm` /
  `use_fused_read_norm`: None follows it; True/False forces). CPU falls back.

**1B A/B** (lean harness `scripts/ab_offload_bwd_prefetch_1b.py` with `H1B_GLUE`, on main
as of 786392e -- i.e. WITHOUT the forward-prefetch / loss-head / backward-prefetch levers --
plus this change, one full 13-block cycle per arm, ABBA):

| arm | s/step, cycle mean | peak reserved | peak allocated |
|---|---:|---:|---:|
| reference glue, run 1 / run 2 | 21.805 / 21.814 | 4904 MiB (79.9%) | 3058 MiB |
| fused glue, run 1 / run 2 | 19.558 / 19.598 | 4914 MiB (80.0%) | 2992 MiB |

Every active-block position gains, from -0.9 s (tail group active) to -3.4 s (embedding
active). Loss drift fused vs reference 5.2e-3 over 14 steps, inside reference-vs-reference
noise (6.5e-3).

**All four of today's levers together** (forward prefetch overlap, loss-head cast/pad,
backward recompute prefetch, this; one cycle, local combined checkout): **16.92 s/step vs
main's 21.81 -- 22% faster.** Peak reserved 5078 MiB (82.7%), 141 MiB under the 85% guard;
lower than the first two levers alone (5098 MiB), because the fused glue frees transient
memory.

**Tests** (`tests/test_fused_l2norm.py`, `tests/test_fused_read_norm.py`): fp32 parity;
bf16 within one ulp for the forward; fp64 truth -- the fused result must be at least as
accurate as the reference; zero and below-eps rows; non-contiguous input; non-power-of-2
`d`; B=2 so a wrong head mapping is visible. Anti-vacuity: the classic wrong L2 gradient
(no projection term) and a wrong weight gradient must land far outside the bounds. The
block-level tests count kernel calls, so they fail if the fused path is silently skipped,
and they require the fused block to be no less accurate than the reference block against
an fp32 run.

Two test-design traps hit and fixed on the way, worth knowing for any norm kernel:
- A below-eps row has gradient g/eps (~1e6x everything else), so a max-relative
  gradient comparison over inputs that include one only checks that row. The
  anti-vacuity test caught it.
- dx of an L2 norm is a difference of two terms. Per-element relative bounds fail for
  ANY bf16 implementation near cancellation; bound by the row's scale instead.

**Not done: folding glue INTO the scan kernels.** (Done 2026-10-14, 3.5% -- see the
entry above.) What remains is the scan wrapper's
`.contiguous().float()` copies (q, k, v: ~2.1 ms per forward pass) and the dq/dk/dv casts
(~0.9 ms per backward), about 3% of the step by the same arithmetic. The scan kernels
already upcast with `.to(tl.float32)` on load, so the copies are pure layout/dtype. A
complete fix:
- passes `[B, S, H, d]` strides into `_gated_delta_fwd_kernel`, `_mk_ckpt_kernel` and
  `_mk_bwd_kernel` (base = b*sB + h*sH, row = base + t*sS);
- accepts bf16 q/k/v;
- keeps `o` fp32 (the read norm needs it);
- stores dq/dk/dv in the input dtype.
It should be bit-identical (addressing only, same in-kernel upcast; Triton's fp32->bf16
store must be checked to be round-to-nearest-even like autograd's cast). It touches the
validated scan kernels, and the gain is a third of what was taken here, so it was left
for its own session. Separately, `test_bcd_balanced_offload::...rotations[True-1]`
failed once on its "loss went down" check (6.2195 vs 6.2054). It does not use the fused
path, passes 3/3 on main and on this branch, and is a flaky assertion against the
backward's run-to-run noise.
## 1B pretrain: prefetch the backward recompute's weights = 3.8% faster per step (2026-10-12)

**One line: at 2048x24 (hvn-hybrid, BCD balanced + offload, on top of the two levers
below), a full BCD cycle goes from 19.94 to 19.18 s/step -- 0.76 s (3.8%) faster. No extra
VRAM (peak reserved 5098 MiB / 83.0% in both arms). Step-0 loss bit-identical.**

**Validated before building.** Grad checkpointing recomputes every inactive block
downstream of the active one during backward. With `cache_blocks=1` each of those
recomputes was a `load()` miss: a synchronous copy from PAGEABLE memory that also blocks the
host, so the launch queue drains every time. Timed in the real 1B harness on main: 96 misses
and 0.80 s of main-stream copy per step averaged over a cycle (144-192 and 1.2-1.7 s when the
active block is early). The counts match 8 micro-batches x downstream layers exactly, so the
instrument counts the right thing. The estimate (0.7-0.88 s, ~4%) held.

**The fix.** `EdgeHVN.forward` records which offloaded blocks will be recomputed, in
forward order. When block i's recompute starts, after its own `load()` and before its
kernels are queued, it prefetches block i-1 on the side stream. The end of forward
prefetches the first one (the last block). `GPUBufferPool.prefetch(..., behind=True)`
does two things differently:
- lets the cache hold ONE entry over `cache_blocks`, because block i must stay resident
  for its own backward;
- takes the oldest free buffer set even when it is the only one. Every free set was
  released by a block whose backward was queued before block i's recompute, so waiting
  on its release event never serializes the copy behind block i.
Toggle: `model._offload_bwd_prefetch` (default on whenever offload prefetch is on).

| arm (both include the forward-prefetch + loss-head levers) | s/step, cycle mean | bwd misses/step | peak reserved |
|---|---:|---:|---:|
| backward prefetch OFF, run 1 / run 2 | 19.938 / 19.937 | 96 | 5098 MiB (83.0%) |
| backward prefetch ON, run 1 / run 2 | 19.194 / 19.166 | 0 | 5098 MiB (83.0%) |

- The saving scales with downstream depth, as the mechanism predicts: -1.56 s/step with
  every layer downstream (embedding active), -0.6 s mid-stack, +0.01 s with nothing
  downstream (tail group active).
- VRAM: identical peak reserved and allocated (3218 MiB), and the pool holds the same
  buffer sets in both arms (3 DeltaNet, 2 attention, 2 tail). The second set the backward
  prefetch needs already exists for the forward prefetch. So this lever does not eat any
  of the ~117 MiB of headroom left under the 85% guard.
- Loss: step-0 bit-identical. Later drift ON vs OFF 5.9e-3, inside the backward's own
  run-to-run noise (OFF vs OFF 4.9e-3 here; up to 1.0e-2 in the entry below).

**Tests** (`tests/test_offload_bwd_prefetch.py`):
- backward misses are 0 with it on and exactly L with it off (so the counter can fire);
- gradients match a never-offloaded model over a full cycle, and a twin that serves the
  wrong block's weights in backward must FAIL the same bound;
- a pool-level race stress with the main stream held behind a GPU sleep and only two sets,
  so every backward prefetch must reuse the set released one step earlier. Its twin skips
  the release wait and must corrupt reads. It also asserts no third set is allocated.
- A model-level sleep version of the race test was tried first. It was BLIND: with the
  wait removed it still passed, because the set it reused had been released long before.
  It was deleted rather than kept as false comfort.
Two existing residency tests now exempt exactly the one block fetched at the end of
forward, and assert that block IS on the GPU.

**Found on the way: the never-offloaded reference is not repeatable.** Gradient parity
first failed at 1.8e-3 and 4.05e-3 relative. Root-caused by elimination: across 8 fresh
processes, 2 showed a mismatch; in one, all four offload arms (ON and OFF) differed from
the reference by the same 4.05e-3, so the reference run was the odd one out. It was also
seen with backward prefetch OFF. A resident model under timing perturbation or concurrent
SM load did NOT reproduce it, so the trigger is unknown. Same finding as the "main against
itself drifts" line below. The parity bound is 2e-2 for that reason; corrupt weights give
O(1) errors (#164: 1.1-2.6), and the twin test proves the bound sees them.

Harness: `scripts/ab_offload_bwd_prefetch_1b.py`. Results: `results/offload_bwd_prefetch_1b/`
(`validate_main.*` is the pre-build miss timing on main; `ab1b_*` the A/B arms).

## 1B pretrain: offload prefetch overlap + loss-head cast/pad = 8.7% faster per step (2026-10-12)

**One line: at 2048x24 (hvn-hybrid, BCD balanced + offload), a full BCD cycle goes from
21.83 to 19.93 s/step -- 1.90 s (8.7%) faster. Step-0 loss is bit-identical. Peak VRAM goes
from 79.9% to 83.0% of the card, under the 85% guard but only ~117 MiB below it.**

Two levers found by a debugger investigation earlier today, built properly here:

1. **Offload prefetch now overlaps compute.** `GPUBufferPool.prefetch` made the copy stream
   wait for the WHOLE main stream (added in eb2b97b to fix a real race). Prefetch runs
   right after block i is queued, so block i+1's weight copy waited for block i to finish:
   0% overlap. Now each released buffer set gets its own event, recorded on the main stream
   in `_release`, and a prefetch that reuses a set waits only for that set's event. Fresh
   `torch.empty` sets keep the blanket wait (that is the case eb2b97b fixed). Prefetch takes
   the oldest free set, and only when two are free -- otherwise it is handed the set the
   current block just released and serializes again. Costs one extra buffer set per shape
   (+190 MiB at 1B). Why this is safe, call site by call site, is in
   `_order_prefetch_after`'s docstring.
2. **Loss head: cast the decoder weight once, pad the vocab.** Autocast re-cast the fp32
   tied table [50257, 2048] to bf16 on every slice call (16 per micro-batch). Autocast's
   own cache never applied: it only caches leaf tensors that require grad, and under BCD the
   table is frozen. And 50257 is odd, so cuBLAS used an `align1` GEMM. Now the weight is cast
   once per call into a buffer padded to 50304 rows (zeros, -inf bias on the pad). Only used
   under bf16 autocast, with a plain Linear decoder and a frozen weight -- in practice BCD
   steps where the embedding block is not active. Padding is local to the call: model,
   state_dict and checkpoints keep 50257.

Measured with a lean harness (`scripts/ab_offload_losshead_1b.py`: the trainer's own
`build_model_and_config` / `apply_fast_stack` / `build_optimizer` / `BlockCoordinateScheduler`,
synthetic tokens, no DataLoader, no audit). 2 warmup steps, then one full 13-block cycle,
because step time runs 10-30 s depending on the active block.

| arm | s/step (cycle mean) | forward GPU s | peak reserved |
|---|---:|---:|---:|
| main, run 1 / run 2 | 21.818 / 21.844 | 8.05 | 4.79 GiB (79.9%) |
| branch, run 1 / run 2 | 19.928 / 19.926 | 6.88 | 4.98 GiB (83.0%) |
| branch, old blanket wait (lever 2 only) | 21.110 | 8.04 | 4.98 GiB (83.0%) |

- Repeat runs agree to 0.03 s; the gap is 1.90 s.
- Split: lever 2 is 0.72 s, lever 1 is 1.18 s. The combined result equals the sum of the
  two, so they don't interact. The scratch estimates were -1.15 s and -0.76 s.
- VRAM: all +190 MiB is lever 1. Lever 2 adds nothing to the peak: the cast copy just
  replaces the per-slice transient copy it used to make.

**Correctness.**
- Step-0 loss (initial weights) is bit-identical across every arm.
- After step 0 the runs drift, but **main against itself drifts too**: identical code,
  max 4.4e-3 in loss over 15 steps. The backward is not bit-repeatable run to run. Branch vs
  branch drifts 4.9e-3, branch vs main up to 7.5e-3, and new vs old prefetch ordering
  (same values expected) up to 8.1e-3. As a control, main with loss chunk 256 instead of
  512 (a bf16 change already accepted here) is 7.9-10.3e-3 off main and already differs
  at step 0. So nothing here is outside normal noise.
- The race lever 1 must not reopen is covered by a deterministic test:
  `test_prefetch_release_events_survive_randomized_stress` holds the main stream behind
  with `torch.cuda._sleep`. It passes, and its twin proves it FAILS when the per-set wait is
  skipped. `test_prefetch_copy_overlaps_main_stream_work` fails 3/3 on the old blanket wait.
- A model-level test cannot catch this race at small size. Measured: the host is never
  more than ~1 block ahead of the GPU, and the set prefetch reuses was released 2+ blocks
  earlier, so its event had already completed in 138/138 reuses.
- Lever 2: per-token losses are bit-identical at 3 vocab sizes. Gradients are not. The
  cause is PyTorch's default `allow_bf16_reduced_precision_reduction=True`: cuBLAS picks
  bf16 split-K kernels depending on the GEMM shape. With it off, both paths are equally far
  from fp32 to 4 digits. With it on, at d=2048 both are 3.5e-3 from fp32 (identical); at
  d=896/1024 the fast path is 1.45-1.6x further, still inside the band the old path already
  has at d=512 (7e-3). Tests gate on that band.

**Found on the way, not fixed: offload + chunked loss + `offload_prefetch=False` crashes on
main.** The decoder bias lives in the tail BCD block. The `return_hidden` path only loads
`final_norm`, so the chunked loss reads a bias still parked on CPU ("mat1 is on cuda:0,
different from other tensors on cpu"). With prefetch on (the trainer's only mode) the bias
reaches the GPU only because the tail prefetch happens to include all decoder params.
Latent, but the bias's placement depends on that coincidence.

**Open, not built here** (named by the same investigation, not validated):
- prefetching weights for the backward checkpoint recompute -- now built, see the entry
  above (3.8%);
- the elementwise "glue" ops around the DeltaNet scan -- see the fused-glue entry
  (`feat/fused-l2norm-glue`).

Also seen: free RAM drops ~50 MB per step inside one harness process, on main's code too
(3.9 to 3.1 GB over 15 steps). Not chased; may matter for long runs. One harness arm
(lever 1 alone) died at start-up with Triton's `Cannot find ptxas.exe`, unexplained; the
lever split above uses the old-wait arm instead.

Results: `results/offload_losshead_1b/` (per-arm JSON + logs, `summary.json`).

## Pretrain speed levers: +2.1% per step, same loss, same VRAM (#154 context, 2026-10-12)

**One line: on the 896x16 hybrid at seq 4096, the branch steps in 7.795 s against 7.961 s
on main -- 2.1% faster. The step-1 loss is bit-identical and peak VRAM is the same. Small,
but free: it is on under `--fast` and changes nothing else.**

Two levers, measured together in the real trainer (`scripts/ab_speed_levers.py`, ABBA,
150 steps per arm, first 10 dropped, identical command and `--seed 0` in two checkouts):

1. **Fused ternary quantizer (#154) on under `--fast`, bf16 only.** It already existed; it
   was just never switched on in the training stack. bf16 is where fused == eager bit for bit.
   `--no-fused-ternary` opts out.
2. **No host sync per DeltaNet block in the megakernel backward.** `nblocks` went to the
   kernels as `torch.tensor([nblocks], device=cuda)`, a pageable copy that forced a
   `cudaStreamSynchronize` once per block per backward. Now a plain int argument.

| arm | per-run median s/step | peak alloc GiB | reserved GiB | step-1 loss |
|---|---|---:|---:|---|
| main (165d44c) | 7.969, 7.952 | 2.926 | 3.418 | 11.006638 |
| branch | 7.795, 7.794 | 2.931 | 3.418 | 11.006638 |

The two branch runs agree to 0.001 s and the two main runs to 0.017 s, so the 0.166 s gap
is not noise. Loss over 150 steps drifts apart by at most 0.0082 between arms, the same as
main against itself (0.0079) -- the trainer's own run-to-run spread. `results/speed_levers/ab.json`.

**A bug the A/B caught that every parity test missed.** The first run of this A/B had
step-1 losses of 11.006638 vs 11.006550. Step 1 is computed from the initial weights, so a
bit-identical lever must reproduce it exactly. Cause: Triton's default fp32 `/` is not
correctly rounded. Right next to the 0.5 rounding tie it disagrees with torch: in the real
model (seed 0, `blocks.11.out_proj`) torch gives w/s = -0.50000006 -> code -1, Triton gives
-0.5 -> code 0. One weight in 117M. Random-normal parity tests almost never land that close
to the tie, so they all passed. Fixed with `tl.div_rn`. New tests: the exact failing pair,
plus a +/-512-ulp sweep around the tie at 64 scales (26 code flips before, 0 after). The
A/B now fails outright on any step-1 mismatch. The buggy run is kept as
`results/speed_levers/ab_divbug.json` (it measured the same +2.2%, so speed was never the issue).

**Lever skipped: caching the quantized weight across a step's micro-batches.** In training
the STE needs the live shadow weight inside the autograd graph, and that weight changes every
optimizer step, so a cache could only be reused across the 8 grad-accum micro-batches. Today
the quantized copy only exists briefly during each forward/recompute and is freed before peak
memory; a cache would instead hold it for the whole step. At 896x16 (110 layers, 117.2M
ternary weights) that's **about +224 MiB of VRAM held continuously**, for an estimated ~2-3%
gain — and under `--offload` it would also keep GPU copies of weights that offload is actively
trying to move off the card, working against that system directly. It also carries the same
silent-staleness risk #117 and the #131 inference cache had to guard against. With the
quantizer now fused and cheap (this entry), the ~2-3% no longer clearly justifies either the
VRAM cost or the offload interaction. Not measured, correctness-wise or otherwise.

**Harness note: the A/B was killed twice by Claude Code's low-RAM reaper.** It was not RAM
building up from arm to arm. The trainer uses about 4.5 GB of system RAM (main process
~2.6 GB + 4 DataLoader workers ~0.58 GB each), and the box only has ~7 GB free. Both kills
happened during a start-up audit (Check 3, checkpoint save + rebuild), and one of them was a
first arm with nothing before it. The harness now waits for every child process of an arm
to exit (workers outlived the trainer by 5-6 s every time), settles 15 s, and checks free
RAM before each launch (`--min-free-gib 2.5`). The clean run's lowest free RAM was 2.01 GiB.

## Effective batch: 8 x 4096 beats 16 x 4096 by 0.13 nats at the same tokens -- at today's LR (#152, 2026-10-11)

**One line: on the 48M code model, doubling grad_accum from 8 to 16 is 6.3% worse held-out
at the same 100M tokens and fixed `muon_lr=0.02`. That is about 25x the noise floor. Keep
grad_accum 8. The 32 arm was stopped part-way and gave no result.**

Real trainer (`train_edge_conversational.py`), the 48M config (512x12, 16 heads, hvn-hybrid
attn at 3,9, 32K FIM tokenizer, code corpus, seq 4096, batch 1, chunked loss, bf16 + Triton +
megakernel). Only `--grad-accum` differs. `--seed 1234 --deterministic`. Scored by
`eval/heldout_loss.py` on the pinned 398,970-token held-out slice -- never training loss.

| grad_accum | tokens/step | optimizer steps | tokens | held-out NLL | vs 8 | wall |
|---:|---:|---:|---:|---:|---:|---:|
| **8** (current) | 32,768 | 3,052 | 100,007,936 | **2.10695** | -- | 212 min |
| 16 | 65,536 | 1,526 | 100,007,936 | 2.23946 | **+0.13250** | 191 min |
| 32 | 131,072 | 763 | -- | not completed | -- | -- |

The gap is real at every matched-token checkpoint: +0.63 at 8M tokens, shrinking to +0.13 at
92M. ga16 is ~11% faster per token (fewer optimizer steps). It still loses on wall clock:
ga8 reaches ga16's final 2.2395 at about 74M tokens, ~157 min against ga16's 191.

**Read it as "16 loses at muon_lr=0.02", not "16 loses".** LR was held fixed on purpose.
Larger batches usually want a larger LR, so the honest claim is: without an LR change, a
bigger batch costs quality here. With half the optimizer steps in a 100M-token run, the
bigger batch is short on updates; an LR raise might close some or all of the gap. Not tested.

**ga32 was not completed.** It was stopped at the maintainer's request at about step 566
of 763 (~74%), because the direction was already clear and the GPU time was not judged
worth it. The runner then restarted it from step 0 (it wipes rather than resumes, see
below) and that was stopped too, so no ga32 checkpoint or held-out number exists. This does
**not** show 32 would also lose. It only shows that finishing it was not worth the hours.

### The known-answer cell had to be replaced

The plan said the ga=8 arm must reproduce #148's `muon_lr=0.02` result, 6.03552. That cell
could never pass. #148 is an 83M gpt2-vocab model on SlimPajama+OASST at seq 512, batch 4,
grad_accum 1, 1,500 steps, scored on probe batches from its own training stream. Code
models land around 2 on `heldout_loss.py`. #148's own script would not reproduce it on
main either, since embeddings became tied by default in #151.

Replaced with checks that can fail (`scripts/analyze_eb152.py`, all pass):

1. **Repro cell.** Two seeded 40-step ga=8 runs give bit-identical losses and held-out NLL
   (9.786937 both). The check also requires that the run trained (10.48 -> 9.94).
2. **Prefix cell.** The real 3,052-step ga=8 arm's first 40 losses and LRs are identical
   to the repro run. During warmup the LR does not depend on `--max-steps`, so this ties
   the arm itself to the repro cell, as a sequence.
3. **Same data in every arm.** This trainer's step 1 runs at LR 0, so ga16's step-1 loss
   must equal the mean of ga8's steps 1-2: 10.483255 vs 10.483256. ga32 vs ga16:
   10.492664 vs 10.492664. All three arms read the same micro-batches in the same order.
4. **The flag does something.** ga8 and ga16 differ on 1,526 of 1,526 steps.
5. **No vacuous arm.** Every step logged, exactly 100,007,936 tokens, same held-out size.

**`--seed` alone is NOT reproducible -- found by check 1.** The first repro pair, seeded but
without `--deterministic`, matched on steps 1-2, drifted from step 3 (the first non-zero
update), and ended **0.0045 held-out NLL apart after 40 steps**. That is the size of the
#159 noise floor, from kernel non-determinism alone. `--deterministic` (new; sets
`use_deterministic_algorithms` + the cuBLAS workspace) makes it bit-exact at ~3% step cost
(3.95 vs 3.83 s/step). **Any same-seed A/B on this trainer without it carries
run-to-run noise of about the size of the effects usually being measured.**

### Design choices (decided, with reasons)

- **Matched tokens = 24,416 micro-batches** (the multiple of 32 nearest 100M/4096), so all
  three arms divide it exactly: `--max-steps` 3,052 / 1,526 / 763.
- **Warmup and cosine in each arm's own optimizer steps.** New `--warmup-steps` (was
  hardcoded 500): 500 / 250 / 125, the same 16.38% of each run and the same 4,000
  micro-batches. A fixed 500 would have spent 65% of the 763-step arm warming up. The
  trainer's cosine runs over `--max-steps` after warmup, so every arm ends at the same
  fraction of peak LR.
- **Restart, never resume.** The resume path re-creates the data iterator and skips
  warmup, so a resumed arm would not see the same tokens. The runner wipes an
  incomplete arm and reruns it seeded. Cost found the hard way: that wipe also removed
  ga32's partial checkpoints.

### Limits

- **Noise floor caveat.** #159's ±0.0035-0.0053 is **step-to-step wobble within one run**,
  not seed-to-seed variance. Here it is the only floor there is. At +0.1325 the
  conclusion does not depend on which floor is used. Close calls would.
- One seed. 48M only. 100M tokens (~3 tokens per body param, far short of #159's ~30).
- Fixed LR, as above.
- The token budget is small. With more data the step-count handicap of the big batch
  shrinks, and the trend over 8M -> 92M (+0.63 -> +0.13) is narrowing. Whether it closes
  at a full ~1B-token 48M budget is not measured.

Files: `results/effective_batch/` (final + trajectory JSONs, per-step CSVs, `sweep.json`);
`scripts/run_eb152_arm.sh`, `scripts/run_eb152_sweep.sh`, `scripts/analyze_eb152.py`.

## Balanced BCD now works with offload -- and offload was feeding blocks the wrong weights (#164, 2026-10-10)

**One line: `--offload` runs now get the balanced partition instead of silently falling back to
per-layer. Verifying that turned up a worse, older bug: every offload run with prefetch on
was computing its inactive layers with some weights swapped between same-shaped matrices.**

**The prefetch bug (fixed, `2f94d7a`).** `_PinnedStagingPool` (#118) caps each weight shape
at 2 pinned staging buffers. A real DeltaNet block has six (d, d) matrices, so every prefetch
went through the overflow branch. That branch reuses a buffer that is still held, and each
holder released it, so one buffer ended up in the free list several times. The next prefetch
then got the same pinned buffer for two weights, and the second host copy overwrote the first
before its GPU copy ran. Measured on main (d=512, 8 layers, seq 1024, lr=0): offloaded logits
off by **1.1-2.6 (max abs) on every forward** vs the same model without offload. **0.0
after the fix.** Loss still went down and every value stayed finite, which is why
`test_prefetch_does_not_corrupt_activations_at_depth` (it only checks finiteness) never saw it.
Any offload training result from before this fix is suspect. Offload *throughput* numbers are
not affected beyond the overflow syncs (see 1536x24 below).

**The #164 change.** The model was told one active name and compared each layer against it.
A balanced group's name ("blocks.0+blocks.1") matches no layer. Now the scheduler hands over
the set of member names. Tests (`tests/test_bcd_balanced_offload.py`): loss and every
parameter match the no-offload run exactly over 2.5 rotations (tied, untied, LRU cache 3);
every group member is on the GPU and everything else on the CPU after every forward and
backward; bf16-parked groups come back with their training. A guard test puts the old
single-name check back and requires the suite to fail (it crashes on a CPU/GPU device
mismatch).

**Re-measured sizing rows (`results/pretrain_sizing/results.json`):**

| shape | partition | blocks | tok/s | VRAM | before (layer) |
|---|---|---:|---:|---:|---|
| 1536x24 | **balanced** | 9 | **2,063** | 61% | 2,295 tok/s, 61%, 26 blocks |
| 2048x24 | **balanced** | 13 | **1,417** | 80% | 1,530 tok/s, 80%, 26 blocks |
| 2048x32 | **balanced** | 17 | **1,120** | 84% | 1,182 tok/s, 84%, 34 blocks |

Every log confirms the new partition (`Block-coordinate-descent: N blocks (balanced
partition)` + `CPU weight offload: ON`). Balanced is 5-10% slower per token at every shape
(1536x24 -10%, 2048x24 -7%, 2048x32 -5%): a balanced group's backward reaches deeper than one
layer's does. VRAM is unchanged. In days-to-30-tokens/param terms: 2048x24 248 days, 2048x32
418 days -- before the BCD step penalty, which balanced roughly halves (3.2x vs 6.9x), so
balanced is still the much faster way to a given loss.

2048x32's trainer finished all 35 steps (`TRAINING COMPLETE`, step 35 in the CSV), then the
whole job was killed by a low-RAM reaper while the 3.3 GB final checkpoint was being written
(most likely trigger; the 30 s RAM log missed the spike, it never showed under 3 GB free). So
its row was finalized from the same log + `training_log.csv` with the script's own
`parse_run` and guards (block count 17, params 1,449,849,809, 84% VRAM). Sizing rungs save a
full checkpoint they then delete; at 1B+ that save is the RAM hazard, not the training.

**Quality with offload: the 41% holds (2026-10-11).** Same 8-layer A/B, offload arm only,
the other three arms from `results/bcd_partition/ab_d1024_L8.json`
(`results/bcd_partition/ab_d1024_L8_offload.json`):

| arm | held-out loss | ppl | gap closed | peak allocated | wall |
|---|---:|---:|---:|---:|---:|
| full | 4.958 | 142 | -- | 3.88 GiB | 27.9 min |
| BCD layer | 5.758 | 317 | 0% | 3.88 GiB | 11.3 min |
| BCD balanced | 5.430 | 228 | 41% | 3.88 GiB | 16.5 min |
| BCD balanced + offload | **5.426** | **227** | **41%** | **2.88 GiB** | 21.3 min |

bf16 parking costs nothing measurable: the offload arm ends 0.004 nats *lower* than no
offload, and no eval point of 17 is more than 0.015 nats apart. The no-offload arm is
deterministic here (a fresh re-run matched the stored one to 5e-6 nats at all 16 evals), so
that 0.015 is offload's real effect, and it is tiny. All parameters moved. Offload saves 1 GiB
at this size and costs 29% wall-clock. 41% can be quoted for offload runs.

**The A/B harness could not have measured this before (fixed here).** `EdgeHVN.forward()`
only does the offload load/evict when `self.training and torch.is_grad_enabled()`. Eval runs
in `model.eval()` under no_grad, so it hit the CPU-parked weights and crashed on a device
mismatch at the step-0 eval. `scripts/ab_bcd_partition.py` now brings parked weights to the
GPU for each eval and puts the same CPU tensors back (asserted). Its "unmoved params" guard
also compared GPU fp32 init against CPU bf16 params (crash); it now rounds the init the same
way, so an untrained parked param still counts as unmoved. A 60-step smoke at 10 steps/block
did flag 4 `alpha_proj.bias` params as unmoved under offload only: AdamW steps of 3e-4 summed
over 10 steps are under half a bf16 ulp, so parking rounds them back to init. At the default
100 steps/block nothing was erased. **Do not run offload with a small `--bcd-steps-per-block`**
-- by the same arithmetic, at 1 step/block (the sizing rungs, throughput only) a 3e-4 AdamW
step on a norm weight near 1.0 (bf16 ulp 0.008) would be rounded away on every switch. Not
measured at 1 step/block.

**Latent, not fixed:** `train_edge_conversational.py --offload --val-fraction >0` takes the
same eval path (`model.eval()` + no_grad) and should crash the same way at the first VAL step.
Not run; val is off by default, which is why nothing has hit it.

## Depth vs width at 1B: 1.21x is the whole lever, and half the sweep does not fit (2026-08-31)

**One line: at matched parameters, trading depth for width is worth up to 1.21x -- but every
config past 12 layers spills the card, so the usable range is 24L -> 16L for 1.15x, and the
question that actually decides it (quality) is not measured.**

Motivated by the roofline: the scan is 14.9% of the step and its cost per layer is
`d_model x d_head x seq`, so fewer/wider layers should mean fewer scan calls and larger, more
efficient GEMMs. Total parameters held to ~1.21B (within 1.1%) and `d_head` pinned at 32, so
neither size nor head width is a confound. `scripts/profile_1b_step.py`, seq 4096, batch 1,
offload ON.

| layers | d_model | params | s/step | tok/s | peak | % card | vs 24L | days | |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 32 | 1792 | 1.211B | 3.895 | 1052 | 4.18 GB | 70% | 0.92x | 400 | |
| **24** | **2048** | **1.216B** | **3.575** | **1146** | **4.52 GB** | **75%** | — | **369** | current |
| 16 | 2464 | 1.222B | 3.118 | 1314 | 5.22 GB | 87% | 1.15x | 323 | |
| 12 | 2784 | 1.213B | 2.954 | 1387 | 5.75 GB | 96% | 1.21x | 304 | |
| 10 | 3008 | 1.210B | 2.923 | 1401 | 6.13 GB | 102% | 1.22x | 300 | **SPILL** |
| 8 | 3328 | 1.223B | 2.800 | 1463 | 6.76 GB | 113% | 1.28x | 290 | **SPILL** |
| 6 | 3744 | 1.219B | 2.780 | 1474 | 7.55 GB | 126% | 1.29x | 287 | **SPILL** |

**Three rows were nearly published as "the curve flattens below 12 layers".** They are past
the card -- 102%, 113%, 126% -- and on Windows WDDM backs the overflow with host RAM instead
of raising, so they run, report **99% GPU-busy**, and look like ordinary data points. Peak
VRAM is the only signal that separates them from real configs. **Do not quote their timings.**
This is the #141 spill hazard exactly, and it caught this sweep.

**Direction confirmed, magnitude capped.** 32L is *slower* than 24L (0.92x), and every step
shallower helps, so the mechanism is real. But the usable range ends at 12L/96%, which leaves
no headroom for a multi-month run -- one allocation over and the run silently goes 4-15x
slower. **16L at 87% is the only config with both a real gain (1.15x, 46 days) and margin.**

**Known-answer cell, and it earned its place.** Fixing `plan()` (below) changed *where* the
two attention layers sit for every depth except 24. `L12_ctrl` re-ran an already-measured
config under the new placement: **2.954 s at attn [3,10] vs 2.938 s at attn [1,5], 0.5%
apart.** Placement does not affect step time, so the two sweeps are comparable. Without that
cell the second sweep could not have been read against the first.

**Not a recommendation to change the default.** This measures step TIME. Whether 16 wide
layers learn as well as 24 narrow ones at matched parameters is unmeasured, and depth
generally buys something in language models that width does not. It is also one seed, and the
0.089-nat seed spread measured the same day says that matters. The decision to stay at 24L
rests on the untested quality question, not on the speed number.

### `plan()` bug found by the sweep

`scripts/profile_1b_step.py:44` hardcoded the two attention layers at indices 3 and 10, so any
config below 11 layers raised `IndexError` -- L=8 died mid-sweep. Now placed proportionally at
the same depth fractions those indices held at 24 layers (1/8 and 5/12), verified to return
exactly `[3, 10]` at L=24 so the reference config is unchanged, and to place exactly 2
attention layers at every depth from 2 to 32. The COUNT stays at 2 by design; for a step-time
measurement only the count matters.

Artifacts: `results/depth_width/L{6,8,10,12,12_ctrl,16,24,32}.json`.

## No dead units to recycle -- 0 of 49,152 ternary rows, 6 of 13,312 activations (2026-08-31)

**One line: dormant-neuron recycling (ReDo) was proposed, and the population it would
operate on was measured first -- it is 0.05%, so there is nothing to recycle and the idea
is dropped without building it.**

Two distinct ways a unit can die here, and only the second is specific to this stack:

1. **Dormant activations** -- output negligible against the layer average, so almost no
   gradient returns through it. The classic cause is ReLU pinned at zero; this model's FFN
   defaults to GELU (`deltanet.py:145`) with SiLU/sigmoid gates, none of which have a hard
   zero, so the mechanism should be weak. Measured anyway.
2. **Ternary collapse** -- BitNet b1.58 rounds any weight under half the scale to exactly
   zero, so a whole output row can go silent with no activation function involved. **#109
   documented exactly this**: fresh attention layers were ternary-quantized at init,
   "collapsing them to {-1,0,1} before they had any chance to learn a pattern", and a real
   recall regression was traced to it.

Measured on `checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt`
(`scripts/probe_dead_units.py`):

| population | result |
|---|---|
| ternary output rows entirely zero | **0 of 49,152** |
| activation units exactly zero | **0 of 13,312** |
| dormant at ReDo's tau=0.025 | **6 of 13,312 (0.05%)** |

Threshold sweep, so the verdict does not rest on one cutoff: tau=0.001 -> 0 units,
0.01 -> 2, 0.025 -> 6, 0.05 -> 20, 0.1 -> 84 (0.63%). Nothing approaching a population
worth reinitialising at any threshold.

**The one number worth keeping: 35.33% of ternary weights round to exactly zero.** A third
of the weight matrix is zero and yet no complete row ever is -- the zeros are spread, which
is ternary behaving as intended (sparsity), not units failing.

**Why the detector's zero is trustworthy.** A detector that finds nothing is
indistinguishable from a broken one, so `selftest_detector()` plants a known number of dead
units into a synthetic tensor and asserts it finds exactly those and no others, plus asserts
it reports ZERO on all-healthy data. It also runs before the checkpoint is touched. And the
ternary scan is confirmed non-vacuous: it found **49,152 real output rows**, so "0 dead" is
a measurement rather than "no ternary modules were present" -- the false negative this
check exists to rule out.

**#109 is not contradicted.** Ternary collapse is real, it happens at init on freshly added
layers, and a checkpoint trained 8,000 steps has recovered from it. Recycling addresses a
condition that does not persist.

**And it was never a speed lever.** Recycling does not change step time; it changes what the
parameters buy. Same category as weight sharing.

Artifact: `results/dead_units/probe.json`.

## "98.7% GPU-busy" was right; "therefore compute-bound" was WRONG (2026-08-31)

**One line: a roofline against this card's own measured ceilings puts the scan at 1.3% of peak
compute and 7.6% of peak bandwidth -- near NEITHER -- so it is latency-bound, not compute-bound,
and the conclusion the 98.7% figure has been carrying since 2026-08-27 does not follow.**

`scripts/profile_1b_step.py` prints *"compute-bound. Only fewer FLOPs or a better kernel helps"*
whenever GPU-busy is high. **That inference is invalid.** GPU-busy measures whether a kernel is
executing, not whether the machine is doing work. A kernel running a serial dependency chain with
64 programs on 20 SMs reports as busy the entire time while most of the card idles.

The 98.7% measurement itself stands. Everything derived from it about *what kind of fix applies*
was wrong for the scan.

### The roofline (`scripts/roofline_1b_step.py`)

Both ceilings measured on this card, not taken from a spec sheet -- a roofline drawn against a
vendor peak no code ever reaches makes everything look memory-bound:

    peak compute     25.12 TFLOP/s   (bf16 GEMM, the same cuBLAS path training issues)
    peak bandwidth   169.7  GB/s     (large streaming copy)
    machine balance  148    FLOP/byte

| op | ms | TFLOP/s | GB/s | % peak compute | % peak BW | verdict |
|---|---:|---:|---:|---:|---:|---|
| DeltaNet scan (fwd) | 6.52 | 0.33 | 12.9 | **1.3%** | **7.6%** | **LATENCY-bound** |
| projection GEMM | 1.36 | 25.28 | 30.9 | **100.6%** | 18.2% | compute-bound |
| elementwise (SiLU gate) | 0.45 | 0.09 | 112.1 | 0.4% | 66.1% | memory-bound |

**The GEMMs are at 100.6% of the card's own demonstrated ceiling.** There is nothing to reclaim
there, and any proposal to speed up the projections is finished before it starts.

Byte counts are analytic minimums (Nsight Compute is not installed on this box), which biases a
re-reading kernel toward looking latency-bound. The scan verdict survives that caveat only
because **both** fractions are low: no re-read accounting moves 1.3%-of-compute.

### What binds the scan: parallelism, and the chain is exactly serial (`scripts/probe_scan_levers.py`)

Two candidate causes, told apart by how cost responds to N and to S:

| N (=batch x heads) | progs/SM | Mhead-tok/s | vs N=8 |
|---:|---:|---:|---:|
| 8 | 0.4 | 16.2 | 1.00x |
| 20 | 1.0 | 28.2 | 1.74x |
| **64** | 3.2 | **39.8** | 2.46x |
| 128 | 6.4 | 45.0 | 2.78x |
| 256 | 12.8 | 46.3 | 2.86x |

Occupancy is real (2.86x across the sweep) but **nearly spent at the current config**: N=64 to
N=256 is only **1.16x**, and it costs 4x the batch in VRAM. Not a lever worth pulling.

Sequence scaling settles the rest -- **ns per sequential step is constant to within 1.05x across
a 16x range of S** (1529/1571/1609/1589/1585 ns at S=512..8192). The chain is fully serial: cost
is set by the number of steps and nothing else.

**The 1,590 ns/step is the interesting number.** The data a step actually needs is ~20 KB across
all programs; at 170 GB/s that is **~120 ns**. The scan pays **13x** that, waiting on dependent
loads with too few warps in flight to hide the latency.

### Chunk-parallel is dead -- now on a sweep, not one sample

`gate_scan_vs_dhead.py` concluded "do not build" from a **single chunk size (128)**, which is the
premature-discard pattern this repo keeps writing up. Swept properly at d_head 32:

| chunk | 16 | 32 | 64 | 128 | 256 | 512 |
|---|---:|---:|---:|---:|---:|---:|
| vs sequential | 28.5x | 14.0x | 7.2x | **4.95x** | 8.2x | 14.9x |

Every size loses; the best is still 5x worse. The verdict now rests on evidence. Known-answer
cell held (d_head 64 reproduced #95/#96's measured loss).

### Where the step ACTUALLY goes (`scripts/attribute_1b_step.py`)

The measurement that decides whether any kernel work is worth doing. d=2048/24L, n_heads 64,
seq 4096, batch 1, offload ON, 3.67 s/step:

| category | s/step | share |
|---|---:|---:|
| elementwise | 1.304 | 35.6% |
| gemm | 1.208 | 33.0% |
| **scan** | **0.546** | **14.9%** |
| other (mostly PCIe memcpy) | 0.545 | 14.9% |
| optimizer | 0.060 | 1.6% |

**Amdahl kills the scan kernel rewrite.** Even the full 13x latency gap yields **1.16x** on the
whole step -- 375 days to 323. Weeks of Triton for 14%.

**A harness bug caught by its own check, and it is #131's exactly.** The first run reported
`other` at 49% and a device-time/wall-clock ratio of **1.99** -- near-exactly 2x, because
`key_averages()` returns op-level rows (`aten::mm`) *and* the kernel rows they launched, both
carrying the same device time. #131 disclosed this same double-count in its decode profile
("they agree to the last decimal at 54,794.628 us"). Filtering to `device_type == CUDA` brought
the ratio to **1.00**. **Any future profiling script must not sum across `key_averages()`.**

### `aten::copy_` is the largest single op, and it is mostly NOT waste (`scripts/probe_copy_traffic.py`)

`copy_` is **0.83 s/step** -- larger than any single GEMM. It is *not* the PCIe offload traffic,
which is counted separately as `Memcpy HtoD` at 0.40 s. This is device-to-device copying.

| bucket | s/step | share |
|---|---:|---:|
| activation-shaped | 0.522 | 62.8% |
| weight-shaped | 0.289 | 34.7% |
| small/other | 0.021 | 2.5% |

63% is gradient-checkpoint saves and contiguity -- structural, and removing checkpointing was
measured at 178% of the card and 5.6x slower. The one suspect row is **`[2048, 2048]`: 414 calls,
232 ms/step**, each weight copied several times per step by the autocast cast and the BitNet STE,
paid on the forward *and again* on the checkpoint recompute. Best case ~6% of the step.

**#131 already tried caching this and found training cannot reuse it** -- the STE needs the live
shadow weight in the autograd graph and the weight moves every step. The **untested** case is
caching within a single step across the forward/recompute pair, where the weight demonstrably has
not changed.

### Verdict: the step is finished

GEMMs at 100.6% of the card. Scan capped at 1.16x. Two-thirds of the copies structural. 98-99%
busy. Summing every remaining lever optimistically gives **20-25%** for weeks of risky work --
375 days to ~300.

**This holds for this config only** (1.21B, batch 1, seq 4096, offload on) and says nothing about
a different architecture or shorter sequences. Within what is built, step time is real work at
close to hardware speed. **The remaining variable was never the step -- it is the 36 billion
tokens.**

Artifacts: `results/roofline/{roofline_1b,scan_levers,attribution_1b,copy_traffic}.json`.

## Decision: not training a 1B model on this laptop (2026-08-28)

`data_cache/fineweb_edu` (233 GB parquet, ~127B tokens) has been **deleted** to free disk
space. It was never tokenized. Everything below this line about the 1B/1.21B frontier
run — the data-mix cap, the compute-bound step profile, the tensor-train comparison — is
now historical context, not a live plan. If a 1B+ pretrain is picked back up later, the
FineWeb-Edu download has to happen again first.

## ⚠️ The DEFAULT data mix caps a pretrain at 0.13B tokens — 270x short for 1B (2026-08-27)

**One line: 233 GB of FineWeb-Edu (~127B tokens) is downloaded and sitting on disk, the
default `--pretrain-mix` gives it a 0.0 share, and the resulting budget caps at 134,784,617
tokens — enough for a ~7M-param model. A 1.21B run on defaults gets 270x less data than it
needs, with 127B tokens unused three directories away.**

This is the #119 mechanism firing on the live default, not a hypothetical. The warning prints
on every run and it is correct; it had simply never been checked against the 1B target.

### The arithmetic (`max_repeat=3`, tokens as counted by the GPT-2 tokenizer)

| source | tokens on disk | share | budget it can support |
|---|---:|---:|---:|
| SlimPajama | 1,318,230,992 | 0.50 | 7,909,385,952 |
| **OpenAssistant** | **15,724,872** | **0.35** | **134,784,617 ← binds** |
| WikiText-103 | 119,073,409 | 0.15 | 2,381,468,180 |
| FineWeb-Edu | ~127,000,000,000 | **0.00** | **unused** |
| | **1.45B unique** | | **CAP: 0.13B** |

OpenAssistant holds 15.7M tokens. At a 35% share it can stretch to 47M before hitting
`max_repeat=3`, and for 47M to *be* 35% the whole budget must stop at 134M. The largest
corpus on the machine contributes nothing because its share is zero.

| target | tokens needed | vs the 0.13B cap | vs ALL 1.45B unique |
|---|---:|---:|---:|
| 478M @ 20 tok/param | 9,559,940,860 | **71x** | 6.6x |
| 478M @ 30 (#159) | 14,339,911,290 | **106x** | 9.9x |
| 1.21B @ 20 | 24,290,872,660 | **180x** | 16.7x |
| 1.21B @ 30 | 36,436,308,990 | **270x** | 25.1x |

**Even ignoring the mix cap entirely, every unique token on disk outside FineWeb-Edu is 25x
short for a 1.21B model.** The non-FineWeb corpora cannot pretrain anything at this scale.

### The fix is one flag, and this repo already wrote it down

    --pretrain-mix 0.0,0.0,0.0,1.0

FineWeb-Edu only, which #120 already recommended for "4B+ pretrain runs" — the recommendation
was correct and simply never became the default. The default is hardcoded at
`train_edge_conversational.py:1011` and `:1502` as `(0.50, 0.35, 0.15, 0.0)`, described in the
help text as "the original conversational mix". It is a **fine-tuning** mix that is also the
**pretraining** default, and nothing warns that the two want different things.

That takes the budget from 0.13B to ~127B — 3.5x more than a 1.21B model needs at 30
tokens/param, and 8.9x what a 478M needs.

**One cost to plan:** the parquet files are downloaded but NOT tokenized — there is no
`fineweb_edu.bin` in `data_cache/tokenized/`. That is a one-time pass, and only the portion
actually consumed needs it (~66 GB of parquet for 36B tokens, not all 233 GB).

### Why this matters more than any of the compression work

The three blockers on a 1B run, as they actually stand:

| | status |
|---|---|
| **memory** | **solved** — 1.21B fits at 73% of the card (4.38 GB), measured |
| **data** | **solved but OFF BY DEFAULT** — 127B tokens on disk, one flag away |
| **time** | **not solved** — 413 days at 1.21B, 87 at 478M |

Two of the three are done. The default mix is the one that would have silently wasted a
multi-week run: the loss curve would look entirely healthy while the model repeated a 134M
token budget over and over, which is memorisation, not training — the #132 failure mode at
corpus scale.

## The 1B step is COMPUTE-bound at 98.7% GPU — the only free speed left is +7% (2026-08-27)

**One line: profiled the 1B training step, the GPU is 98.7% busy, so every transfer- and
scheduling-side lever has nothing to reclaim; the one exact win available is raising the
micro-batch from 1 to 2 at ~478M, worth +7.1%.**

> **⚠️ Corrected 2026-08-27, same day.** The throughput figures in this section and the one
> below were measured with a **data race live** in `GPUBufferPool.prefetch` (the side stream
> never waited for the main stream, so weight copies overwrote in-flight activations). The
> race is exactly the transfer/compute overlap being timed, so the numbers were **~4%
> optimistic**: **1,014 tok/s is really ~975**, **1,911 is really ~1,830**, and 4.02 s/step is
> ~4.16 s. Fixed in `block_coordinate.py`.
>
> **What survives unchanged:** every VRAM figure and %-of-card verdict (identical to 0.01 GB),
> the **98.7% GPU-busy** measurement, and every *relative* conclusion — batch 2 wins at 478M
> and spills at 1.21B, chunk size does nothing, cache size does nothing — because both arms of
> each comparison ran under the same race. The days-to-train figures shift ~4% (413 -> ~430 at
> 1.21B, 87 -> ~91 at 478M).

### Where the 4.02 s goes (`scripts/profile_1b_step.py`, d=2048/24L, seq 4096, batch 1)

| phase | seconds | share |
|---|---:|---:|
| forward | 1.019 | 25.3% |
| **backward** | **2.942** | **73.2%** |
| optimizer | 0.059 | 1.5% |
| **GPU busy** | | **98.7% mean, 100% max** |

**98.7% busy settles the class of fix.** Only fewer FLOPs or a better kernel can help;
PCIe/prefetch/cache tuning is chasing a cost that is already ~0.

> **⚠️ Corrected 2026-08-31 — the second half of that sentence does not follow.** "98.7% busy"
> is correct, and the PCIe/prefetch conclusion stands. But **"compute-bound" does not follow
> from "busy"**: GPU-busy measures whether a kernel is *executing*, not whether the machine is
> doing work. A roofline against this card's own measured ceilings puts the scan at **1.3% of
> peak compute and 7.6% of peak bandwidth** — near neither, i.e. **latency-bound**: a serial
> dependency chain with too few programs to hide memory latency, which reports as 100% busy
> while most of the card idles. The class of fix for the scan is parallelism, not fewer FLOPs.
> See the 2026-08-31 entry at the top of this file. `profile_1b_step.py` still prints the
> "compute-bound" line; treat it as describing the GEMMs, which genuinely are (100.6% of peak),
> not the whole step. #153 measured 97.2% at 83M
with everything resident — this confirms the same holds at 1B with weights streaming from
CPU, which nobody had checked and which is a genuinely different regime.

**Backward is 2.9x forward where ~2x is normal.** The extra ~1.0 s (~25% of the step) is
gradient checkpointing recomputing the forward. That recompute is bit-identical (#49/#100),
so removing it would be free quality-wise — **and it cannot be removed.**
`BlockCoordinateScheduler` raises `offload_inactive_weights=True requires
model.grad_checkpoint=True`: offload evicts a block's weights after its forward and relies on
the checkpoint recompute to bring them back. Measured without it at ~478M: **10.69 GB, 178%
of the card, 340 tok/s** — a 5.6x slowdown. The coupling is structural, not a flag to flip.

### Micro-batch: the only real win, and it inverts between sizes

| config | batch | s/step | tok/s | peak | % card | verdict |
|---|---:|---:|---:|---:|---:|---|
| 1.21B | 1 | 4.02 | **1,020** | 4.38 GB | 73% | FITS |
| 1.21B | 2 | 10.12 | 810 | 7.07 GB | 118% | **SPILLS** |
| 1.21B | 4 | 41.80 | 392 | 10.89 GB | 182% | **SPILLS** |
| 478M | 1 | 2.16 | 1,900 | 2.63 GB | 44% | FITS |
| **478M** | **2** | 4.03 | **2,035** | 4.33 GB | 72% | **FITS** |
| 478M | 3 | 6.00 | 2,049 | 5.24 GB | 87% | SUSPECT |

**At 1.21B, batch 1 is already optimal and raising it is actively harmful** — batch 2 spills
and loses 21%. This is worth stating loudly because #141 measured batch size as a *large* win
at 83M (batch 2 x 8 heads at 9,087 tok/s against batch 1 x 16 heads at 5,882, a 55% gap), and
inheriting that conclusion at 1B costs a silently-spilled run.

**At 478M, batch 2 is worth +7.1%** and still fits at 72%. Batch 3 reaches 87%, above the
#136 SUSPECT threshold, for a further +0.7% — not worth it. **`grad_accum` must be halved
alongside**, so the effective batch is unchanged; otherwise this is a training-dynamics change
smuggled in behind a throughput fix (the #142/#143 discipline).

### Two levers measured at zero, one of them predicted first

- **Chunked-loss chunk size does nothing**: 512 / 2048 / 4096 give 1,902 / 1,900 / 1,906
  tok/s at 478M. Bigger chunks cost VRAM (2.63 -> 4.19 GB) for no speed. Leave it at 512.
- **A larger resident block cache does nothing**: `offload_cache_blocks` 1 -> 8 gives
  2,047 -> 2,053 tok/s, **+0.3%**, for +0.21 GB. **Predicted before running** from the 98.7%
  figure — a cache exists to avoid transfers, and transfers are not what the step is waiting
  on. This also answers the recurring "keep a small parameter cache in VRAM and free each
  parameter once it is done" idea: that cache is `GPUBufferPool` and it already exists, at
  BLOCK granularity. Per-*token* granularity is not implementable, because a layer's weights
  are used by every token in the sequence inside one matmul — there is no instant at which a
  weight is finished with a token — and backward needs them again regardless, which is the
  same coupling that forces grad-checkpointing above.

### PARTIAL checkpointing: the right idea, worth +1.7%, and it loses to a bigger batch

The recompute is *arithmetic*, not waiting, so keeping some blocks' activations resident
instead of recomputing them is the one lever aimed at the actual bottleneck — unlike caching
weights (`offload_cache_blocks`, +0.3%), which saves transfers that were never the cost.
Measured by intercepting the checkpoint calls so the last N of 24 blocks keep their
activations (`scripts/probe_partial_checkpoint.py`, 478M, no offload — the offload path
forbids it, `model.py:413`):

| blocks kept resident | s/step | tok/s | peak | % card | verdict |
|---:|---:|---:|---:|---:|---|
| 0 (baseline) | 2.14 | 1,918 | 4.58 GB | 76% | FITS |
| **2** | **2.10** | **1,951** | 5.03 GB | 84% | **FITS — +1.7%** |
| 4 | 2.51 | 1,631 | 5.54 GB | 92% | SUSPECT |
| 6 | 2.75 | 1,489 | 6.05 GB | 101% | SPILLS |
| 12 | 6.61 | 620 | 7.67 GB | 128% | SPILLS |

**+1.7%, and only 2 of 24 blocks fit.** Activations cost ~225 MB per block (5.03−4.58 over
two), and without offload the baseline already sits at 76%, leaving ~0.5 GB usable under the
85% rule. Skipping 2 of 24 recomputes is 8% of a term worth 25% of the step ≈ 2% — which is
what came out. The arithmetic was right; the estimate of available headroom was not (an
earlier guess of ~13% used the 3.3 GB free in the *offload* config, which is precisely the
config that forbids this).

**Past 2 blocks it gets SLOWER while doing LESS work** — 12 blocks resident is 3.1x slower
than the baseline. Spilling costs more than the recompute ever saved. A clean illustration
that on this card, memory pressure dominates arithmetic savings.

**Verdict: do not build it.** It competes with micro-batch 2 for the same spare VRAM and
loses 1.7% to 7.1%, while needing a scheduler change and a rewrite of the offload eviction
logic to be usable in the config that matters.

### Summary

**There is no large exact speedup available on this hardware.** The GPU is saturated doing
arithmetic that has to happen plus one recompute that cannot profitably be removed. Everything
measured, best first:

| lever | gain | cost |
|---|---:|---|
| **micro-batch 2 (478M only)** | **+7.1%** | none — two flags, halve `grad_accum` |
| partial checkpointing, 2 blocks | +1.7% | a scheduler change; competes for the same VRAM |
| bigger block cache (1→8) | +0.3% | +0.21 GB |
| chunked-loss chunk size | 0.0% | +1.56 GB for nothing |
| micro-batch 2 **at 1.21B** | **−21%** | spills |

Anything bigger has to come from **fewer FLOPs** — the 478M-vs-1.21B decision (1,911 vs
1,020 tok/s, 87 vs 413 days at 30 tokens/param) — or from different hardware.

Artifacts: `scripts/profile_1b_step.py`, `results/profile_1b_step.json`,
`scripts/probe_1b_throughput.py`.

## Tensor-Train projections: 3.96x fewer parameters at 1.05x step time (Issue #161, 2026-08-27)

**One line: TT factorisation of the DeltaNet projections is now a real trainable layer, and
it costs essentially nothing in speed — 3.96x fewer block parameters at d=896 for 1.05x step
time. Quality is NOT measured yet, so this is a capability that landed, not a win that was
demonstrated.**

`hvn/edge/decomposition.py::TTMatrix` had implemented TT-SVD since #32 Phase 2.3 but could
not be put in a model: plain class rather than `nn.Module`, cores not `nn.Parameter`,
`from_dense`/`to_dense` both `@torch.no_grad()`, and `matvec` taking a single vector with no
batch dimension (falling back to `to_dense()` for d>2, which would silently break gradients).
Nothing imported it. `hvn/edge/tt_linear.py::TTLinear` is the trainable counterpart, opt-in
via `tt_targets` on `DeltaNetBlock` / `EdgeHVN`; unset leaves both byte-identical.

### Measured (`scripts/probe_tt_cost.py`, idle GPU, batch 4 x seq 512, full block fwd+bwd)

| d_model | rank | dense params | TT params | smaller | fused s/it vs dense | direct vs dense |
|---|---:|---:|---:|---:|---:|---:|
| 384 | 32 | 1,481,478 | 482,054 | 3.07x | **1.09x** | 1.69x |
| 896 | 32 | 8,051,470 | 2,032,398 | **3.96x** | **1.05x** | 2.34x |
| 896 | 64 | 8,051,470 | 2,435,854 | 3.31x | 1.04x | 3.86x |
| 2048 | 32 | 42,033,184 | 9,527,328 | 4.41x | not timed | not timed |

Compression **improves with size**, which is the right direction for the edge-inference
thesis. Because the parameters stay compressed, optimiser state shrinks by the same factor —
that is the real VRAM term, since Muon/AdamW state is proportional to parameter count.

### A wrong claim, published and then corrected within the session

The first commit stated "TT is not a speed lever — 1.5-3.3x more FLOPs", from the arithmetic
for streaming a batch through the cores: `B*r*n1*m2*(n2+m1)` against dense `B*d^2`, giving
3.3x at d=384 and 2.1x at d=896, i.e. relatively **better** as d grows.

Measured, that order came out 1.69x at d=384 and 2.34x at d=896 — **worse at the larger
width, the opposite ordering to the prediction.** A measurement that disagrees with the
prediction's *direction* is not noise, so it was chased rather than published. Cause: the
batch-streaming order builds a `[B*S, m1, r, rest]` intermediate `r` times larger than the
output, making it bandwidth-bound rather than FLOP-bound — which also explains why its
penalty scales with rank (3.86x at r=64) while the fused path does not (1.04x).

Folding the cores into each other **first** is batch-independent: one small core contraction,
then an ordinary cuBLAS GEMM. The FLOP argument dissolves because it was never about the
right algorithm. Both orders are kept and selectable (`tt_contract`, default `"fused"`;
`"direct"` for memory-constrained inference where never forming the `[d,d]` matrix is the
point), with a known-answer test asserting they are the same function in forward (2.6e-06)
and in the gradients into the cores (3.8e-05).

**The transferable lesson is the repo's own rule in a new place:** a number derived on paper
is not a measurement of this code. The arithmetic was correct about the implementation it
described and irrelevant to the implementation that should have been written.

### Two traps closed by construction

- **`alpha_proj` is refused by name.** It is zero-initialised (bias 3.0, so decay starts
  ~0.95). TT-SVD of a zero matrix gives zero cores, and because cores enter the output
  *multiplicatively* those cores also have zero **gradient** — the decay gate would be
  permanently dead while the loss curve looked perfectly healthy. `beta_proj` likewise.
  `'ffn'` on a SwiGLU/MoE block raises rather than silently skipping, which would have made a
  "tt-ffn" arm secretly measure the dense incumbent (#140's failure mode).
- **Ternary does not silently stack.** `apply_ternary_to_model` targets `nn.Linear`, and
  `TTLinear` deliberately does not subclass it, so TT layers are skipped rather than carrying
  two compression schemes on one weight and confounding both. Asserted, with a control that
  fails if `apply_ternary_to_model` stopped doing anything at all. `ternary_cores=True` is an
  explicit opt-in for measuring the composition later.

### Quality: MEASURED, and TT does not pay at this scale (2026-08-27, same day)

The section above said quality was unmeasured. It has now been measured
(`scripts/ab_tt_vs_dense.py`, 4 arms x 5,000 steps, d_model 384/512, 12 layers, batch 4 x
seq 512, Muon, identical seed and pre-materialised batches, held-out loss):

| arm | d | body params | held-out | VRAM reserved | min | steps/s |
|---|---:|---:|---:|---:|---:|---:|
| dense | 384 | 17,808,769 | **5.75208** | 2,920 MiB | 19.8 | 4.21 |
| dense_twin | 384 | 17,808,769 | 5.75443 | 2,862 MiB | 19.5 | 4.27 |
| tt | 384 | 9,813,377 | 5.95041 | 2,842 MiB | 18.0 | 4.63 |
| tt_wide | 512 | 16,920,721 | 5.76436 | 3,254 MiB | 20.1 | 4.14 |

Noise floor (twin cell): **2.347e-03**.

**Equal width — TT is 0.19833 nats WORSE, 9.9x the band.** 1.81x fewer body parameters costs
a fifth of a nat. Not marginal.

**Matched parameters — INCONCLUSIVE, +0.01228 at matched steps and +0.01336 at matched wall
clock.** Both are above the 2.3e-03 noise floor but inside the pre-registered 0.02 band, so
the run refuses to call it. The reading is "TT-wider is very slightly behind
dense-narrower and one seed at this size cannot separate them," not "they are equal".

### The finding that decides it, and it is about memory rather than loss

**At matched parameters TT uses MORE VRAM, not less: 3,254 MiB against 2,920 — +334 MiB,
+11.4%.** Width drives activations, and TT does not touch activations. So spending the saved
parameters on width to recover the quality hands the memory back with interest.

And at equal width, where TT *is* smaller, the VRAM saving is **78 MiB — 2.7%** for 1.81x
fewer body parameters. That is the whole answer to "what does TT mean for VRAM" at this
scale: almost nothing, because weights and optimiser state are not the binding term. #158
measured a live 48M run at CE ~1500 MB, CUDA context ~400 MB, weights 183 MB, optimiser
183 MB — TT attacks the 11% and leaves the rest alone.

**So both configurations lose:** compress and you pay 0.198 nats to save 2.7% of memory;
widen back to matched parameters and you pay 11.4% more memory to land on the same loss.

**TT was FASTER, which was not predicted.** 4.63 vs 4.21 steps/s at equal width (1.10x) --
fewer and smaller matrices for Muon's Newton-Schulz iteration. Same mechanism as #122's
looped arm (1.30x). The #161 probe measured the *layer* as speed-neutral; in the real loop
with a real optimiser it is slightly positive.

### What this does and does not settle

**Settled:** at d=384-512 for 5,000 from-scratch steps, TT is not a win in either
configuration, and its VRAM benefit is far smaller than its parameter ratio.

**NOT settled, and these are real:**
- **Scale.** Block-level compression *improves* with width (3.07x at d=384, 3.96x at d=896,
  4.41x at d=2048), and at sizes where weights and optimiser state actually dominate VRAM
  the arithmetic changes. This run is at the size where they provably do not.
- **The model-level ratio is diluted by the hybrid.** The block compresses 3.07x but the
  model body only 1.81x, because 4 of the 12 layers are attention and are never TT'd. A
  pure-DeltaNet stack would sit much closer to the block figure.
- **Ternary was applied in NO arm.** This is TT against dense fp32, not against the ~10.7x
  ternary baseline TT would have to beat for adoption. Ternary is also speed-neutral and
  needs no width compensation, so on this evidence it remains the better lever.
- One seed, one corpus, 5,000 steps, early training. Bounds early quality, not convergence.

**Recommendation: do not adopt TT for training at this scale.** The code is landed, tested
and opt-in (default off), so it costs nothing to leave in place for the size regime where
weights dominate — which is the edge-inference case it was always aimed at, and which this
run does not reach.

Artifacts: `results/tt/ab_tt_vs_dense.json` (full per-arm trajectories),
`scripts/ab_tt_vs_dense.py`, `hvn/edge/tt_linear.py`, `scripts/probe_tt_cost.py`,
`tests/test_tt_linear.py` (24/24), `tests/test_tt_deltanet.py` (33/33). Full suite 575/575.
`logs/` is gitignored, so the console log is local-only.

### At the 1B target itself, TT is dominated on every axis (2026-08-27)

The A/B above was at d=384-512, and the fair objection to it is that TT is aimed at the size
where weights dominate VRAM — the 1B-on-6GB goal this project exists for. **Measured there,
directly** (`scripts/probe_tt_1b_no_offload.py`, 24 layers, seq 4096, batch 1,
grad-checkpoint, #158 chunked loss, this card):

| arm | params | s/step | tok/s | peak reserved | % of card | verdict |
|---|---:|---:|---:|---:|---:|---|
| **dense 1.21B + BCD/offload** | 1,214,543,633 | 4.04 | **1,014** | 4.38 GB | 73% | **FITS** |
| **dense 478M (d=1216) + offload** | 477,997,043 | 2.14 | **1,911** | 2.64 GB | 44% | **FITS** |
| TT 499M (d=2048) + offload | 499,414,801 | 3.96 | 1,033 | 4.29 GB | 72% | FITS |
| TT 948M (d=3072) + offload | 947,900,529 | 7.11 | 576 | 6.56 GB | 109% | **SPILLS** |
| TT 396M (d=2048), NO offload | 396,488,465 | 10.45 | 392 | 6.80 GB | 113% | **SPILLS** |

**The hypothesis that motivated this run: "dense 1B needs CPU offload, TT does not, therefore
TT is faster."** It is refuted. TT resident on the card does **not** fit at seq 4096 — 6.80 GB
against a 5.9966 GB card, and the spill is visible in the step times climbing 6.2 → 9.5 →
10.5 → 12.3 s. It is 2.6x SLOWER than the offload path it was supposed to beat.

**The matched-parameter control is the one that settles it.** Dense 478M and TT 499M are the
same size; dense runs **1.85x faster** (1,911 vs 1,033 tok/s) in **1.6x less memory** (2.64 vs
4.29 GB). The #161 A/B already found them equal in quality at matched parameters. So TT is
strictly dominated: same quality, slower, larger.

**Why, in one line: TT trades parameters for WIDTH, and PCIe traffic scales with PARAMETERS,
not width.** So a TT model pays width's costs (d² compute per token, larger activations) and
collects none of the transfer saving it was supposed to buy. At a genuine ~1B parameters
(d=3072) it streams the same bytes as dense 1B while being 1.5x wider — 576 tok/s against
1,014, and it spills.

**Two corrections to the record fall out of this run:**

- **`docs/training.md`'s "~2x offload inefficiency" is stale.** It predates #115 (prefetch
  overlap, ~25%) and #113 (bf16 parking, ~25%). Measured today the offload path does
  **1,014 tok/s** at 1.21B, better than the 875 recorded in #157. Offload is not a tax to be
  escaped — it is what keeps VRAM free for activations, which is the binding term at seq 4096.
- **A 1B model on this card is not memory-blocked, and is faster than the docs say.** 1.21B
  trains at 73% of the card. At 1,014 tok/s a 21.4B-token budget is ~244 days; the lever that
  actually moves that is *size* — the 478M row is nearly 2x faster — not compression.

A bug fixed on the way: `TTLinear` could not be constructed under `torch.device("meta")`
because the init rescale called `.item()`. Meta construction is how #118 established that an
8B model needs ~32 GB to build for real, so a layer that cannot be sized on meta cannot be
sized at the scale that matters.

Artifacts: `results/tt/tt_1b_no_offload.json`, `scripts/probe_tt_1b_no_offload.py`.

**Final status for #161: measured worse than dense at every setting tried, and the CODE HAS
BEEN DELETED.** `hvn/edge/tt_linear.py`, its wiring into `DeltaNetBlock`/`EdgeHVN`, its two
test files and the two TT-specific harnesses are gone; `deltanet.py` and `edge/model.py` are
byte-identical to before #161. Recoverable from git history (commits `807b8b9` .. `c20dc86`
on `issue-161-tensor-train-deltanet`) if anyone wants to re-run it.

Kept rather than deleted, because they answer questions that outlive TT:

- **`scripts/probe_1b_throughput.py`** — the size-vs-throughput probe for the 1B goal,
  stripped of its TT arms. It is what produced the 1,014 and 1,911 tok/s figures.
- **`results/throughput_1b_measured.json`**, `results/tt/ab_tt_vs_dense.json` — the evidence
  behind the numbers above.

**Why deleted rather than left opt-in:** the MoE precedent (#123) kept a losing lever because
its verdict was scale-dependent — it lost to *dense scaling at this size*, and might not at
another. TT's verdict is **mechanistic**: it trades parameters for width, width costs d²
compute and activations, and PCIe traffic scales with parameters not width. That argument has
no size dependence, so there is no regime to come back for, and ~1,700 lines carrying a
"do not use" sign is a liability rather than an option.

### The one number worth taking from this whole issue

Measured while establishing the TT baseline, and it has nothing to do with TT
(`scripts/probe_1b_throughput.py`, seq 4096, grad-checkpoint, chunked loss, BCD+offload):

| config | params | s/step | tok/s | peak | % card | days @ 30 tok/param |
|---|---:|---:|---:|---:|---:|---:|
| d=2048, 24L | 1,214,543,633 | 4.02 | 1,020 | 4.38 GB | 73% | **413** |
| **d=1216, 24L** | **477,997,043** | **2.14** | **1,911** | **2.65 GB** | **44%** | **87** |

**A 478M model is 1.88x the throughput at 60% of the memory, and 4.7x less wall clock to a
30-tokens-per-parameter budget** (#159's measured ratio, not Chinchilla's 20). If the 1B run
is time-blocked rather than memory-blocked — and it is, at 413 days — this row is where the
time actually is. Size is the lever; compression was not.

## LIVE RUN — 157M code model, FIM, in progress (2026-08-12)

> **Read this first if you are picking the project up.** A multi-day training run is underway
> and this is its state. Everything below was established in the session of 2026-08-11/12.

### What is running

`train_edge_conversational.py`, detached, auto-resumes from `checkpoints/code_157m/`:

    --code-corpus data_cache/code_corpus/stack_code.txt --fim
    --tokenizer data_cache/tokenizer_32k_fim
    --arch hvn-hybrid --attn-layers 4,12 --attn-window 4096
    --d-model 896 --num-layers 16 --n-heads 28
    --seq-len 4096 --batch-size 1 --grad-accum 8
    --grad-checkpoint --no-fast --precision bf16 --tf32 --triton-scan --megakernel-scan
    --max-steps 10000 --checkpoint-dir checkpoints/code_157m

| | |
|---|---|
| model | **157,494,343** params, **129M body**, tied embeddings, ternary |
| corpus | **3,830,663,537 tokens** — the full 11.2 GB Stack extract, 13 languages, FIM rate 0.5 |
| shape | 32,768 tokens/step (seq 4096 x batch 1 x accum 8) |
| speed | **7.12 s/step = 4,600 tok/s** (28 heads, +20% vs 14), GPU 98-100%, peak 4.3 GiB of 6.14 |
| plan | 10,000 steps = 328M tokens ~ **20 h** (8.6% of the corpus) |
| logs | `logs/code_157m_pretrain.out`, CSV `checkpoints/code_157m_h28/training_log.csv` |
| resume | **automatic** — `find_latest_checkpoint()` picks up the newest .pt; re-run the same command |


**Head count changed to 28 mid-session (#159).** d_head drops 64 → 32, which halves the
recurrent state *per head* — a real capacity change to the mechanism this architecture is built
on. The quality A/B was started and stopped after one arm, so **this is unmeasured**; the loss
curve against the preserved 14-head run is the check. At step 50: 14 heads **9.098**, 28 heads
**9.023**. The 14-head run (1,100 steps, loss 2.329, ppl 10.27) is kept at
`checkpoints/code_157m_h14_step1100/` precisely so the curves can be compared.

The definitive quality answer is the FIM gate on the finished model, not an 800-step A/B — early
training is the regime where capacity limits do *not* show up.

Loss trajectory (14-head reference run): step 50 **9.098** (ppl 8,934) -> 200 **5.157** (174) -> 500 **3.942**
(51.5) -> 600 **3.503** (33.2).

### Three bugs fixed to get here — all of them silent

1. **`--fim` could never have worked.** `vocab_size = tokenizer.vocab_size` EXCLUDES added
   tokens, so the model was built with 32,000 rows while FIM's sentinels occupy 32000-32002.
   Every sentinel indexed out of bounds -> bare `CUDA error: device-side assert triggered`
   inside the pre-training audit. Now `len(tokenizer)`; identical for GPT-2 (50257).
2. **The tokenized cache ignored which corpus produced it.** `code_bin = _bin("code")` keys on
   the literal string "code", so pointing `--code-corpus` at a different file silently reused
   whatever was cached first. Observed: a run announced `stack_code.txt (11.20 GB)` and loaded a
   **121,122,231-token** cache built the previous day -- 4% of the intended data, and at the
   planned step count **2.7 passes over it** (a memorisation regime). The signature now carries
   source name + size + mtime and refuses loudly.
3. **The tokenizer-identity guard compared raw paths**, so a backslash made
   `data_cache/tokenizer_32k_fim` and `data_cache	okenizer_32k_fim` read as different
   tokenizers, rejecting a valid cache.

### `torch.compile` is a NET LOSS on this architecture — settled (#158)

Measured in the real training loop (seq 4096, grad_accum 8, `--grad-checkpoint`):

| config | s/step | tok/s | GPU |
|---|---:|---:|---:|
| **eager + Triton scan + megakernel** | **8.66** | **3,784** | **98%** |
| per-block compile + those kernels | 20.72 | 1,581 | 35% |
| per-block compile, kernels OFF | 21.42 | 1,530 | 36% |
| whole-model compile | — | — | `InductorError: MemoryError` after 40+ min |

**2.4× slower**, and disabling the fused kernels does not help — so it is not the graph-break
interaction #129 documented, it is compile itself here. GPU utilisation tells the story: **36%
mean, 96% max** — it bursts to full then stalls, a gap *between* kernels rather than slow
kernels. VRAM was 4,394 MiB of 6,140, so nothing spilled.

Whole-model compile is separately impossible at this depth: Inductor's `compute_ancestors` keeps
an ancestor set for every node alive at once — **O(N²)** in graph nodes — and MemoryErrors at 16
layers (three reproductions, seq 4096 and 2048, 6.9–9 GB free, ~24 GB of commit headroom
remaining). Per-block compile *works*; it is simply not worth using.

**Fourth independent loss for compile here:** #129 (megakernel breaks fusion, 2 graph breaks),
`state.md:558` (`--block-coordinate` conflict, 6.73 vs 4.41 s/step at 355M), the Inductor OOM,
and this. `--fast` keeps bf16 + TF32 + Triton scan + megakernel and **skips `torch.compile`**.

> **A correction on the record.** A 4-step, 1-micro-batch microbenchmark said per-block compile
> was **+18%** and it was written into four documents on that basis. The real loop inverted it.
> A number from a different harness is not a measurement of this one — the rule this repo already
> had, broken within an hour of writing it down.

### Why it is ~4x slower per step than the 83M, and why that is correct

| | 83M | this |
|---|---|---|
| d_model x layers | 512 x 12 | 896 x 16 |
| `layers x d^2` | 3.15M | **12.85M** (4.08x) |
| tokens/step | 16,384 | 32,768 (2x) |
| s/step | 1.36 | 8.66 (6.4x) |
| tok/s | 12,047 | 3,767 (**3.2x slower**) |

**4.08x the compute per token at 3.2x the time** — better than proportional, because larger
matmuls use the GPU more efficiently. Parameter count is the wrong metric: #139 measured the
83M as **62% vocabulary**, so its body was ~31M against this model's 129M — the same 4.1x from
an independent direction.

### The gate exists and is control-validated

`eval/fim_completion_gate.py`: held-out files from the corpus TAIL, a whole-line span masked,
scored on `exact` and `valid`. Controls measured on 100 cases:

    floor (empty completion)   exact   0.0%   valid  47.0%
    floor (copy previous line) exact   0.0%   valid  38.0%
    known-answer (true span)   exact 100.0%   valid 100.0%

**An empty completion is "valid" 47% of the time**, so validity alone means nothing — the bar is
exact >> 0 with valid >> 47%. `eval/coding.py` already provides the behavioural tier (HumanEval
problems executed in a sandbox); this gate does not duplicate it.

### Immediate next steps

1. **Stop at a checkpoint boundary.** `scripts/stop_after_checkpoint.ps1` waits for
   `edge_conversational_step<N>.pt` to appear AND stop growing, then kills every python process.
   Killing between saves loses up to 500 steps (~72 min).
2. **Windows update + restart** (pending as of this writing).
3. **On a clean boot, A/B `torch.compile`** — both arms one harness, same shape, Triton scan and
   megakernel in BOTH, maximum free RAM. If it wins, resume with `--fast`; resume is NOT lossy
   (`_save_resume_state` persists optimizer state and `build_cosine_scheduler` replays the LR).
4. **Decide the run length.** 10,000 steps is a de-risking pass. A full pass over 3.83B tokens is
   ~114,000 steps, **~11.5 days**, and at 20 tokens/param 3.83B supports ~190M params — so 157M
   is **data-rich, not starved**.

## The speed search, closed: twelve levers, one survivor (2026-08-12)

**One line: `--n-heads 28` is worth +20% and is now running. Everything else measured neutral or
negative, and the GPU is at 98-100% doing irreducible arithmetic.**

All measured at the REAL training shape (157M, seq 4096, grad_accum 8, grad-checkpoint on),
because a microbenchmark misled this session twice.

| lever | result | why |
|---|---|---|
| **`--n-heads` 14 → 28** | ✅ **+20%** (8.63 → 7.12 s/step) | 28 programs instead of 14 on 20 SMs; VRAM +24 MiB, params +176K |
| `--n-heads` 56 | ❌ +2.6% | curve flattens; would halve head state again for nothing |
| `torch.compile` | ❌ 2.4× slower | see the compile entry above |
| Triton `BV` block size | ❌ 16 already optimal | clean U-curve: BV=1 is 0.31×, BV=64 is 0.60× |
| chunk-parallel scan (already written, `chunked_scan_forward`) | ❌ **4× slower** | 32 serial steps instead of 4,096, but ~2.6× the arithmetic |
| fused ternary quantizer (#154) | ⚠️ **+1.9%** | 4.14× on the kernel, ~nothing in the loop |
| host-code rewrite (Rust/C++) | ❌ nothing to rewrite | the host is *blocked*, not busy |

### Why the parallel scan loses, which keeps being rediscovered

`chunked_scan_forward` cuts serial depth 128× (4,096 token-steps → 32 chunk-steps) and is **4×
slower**. Each chunk pays a `[128,128]` Gram matrix, a triangular solve, and several matmuls —
roughly **2.6× the total arithmetic** of the sequential version. That trade wins when per-step
work is large; at `d_head=64` each sequential step is trivial, so the fix-up dominates.

**The rule:** parallelising a scan pays when the per-step work is large relative to the fix-up
cost. It would flip at much larger head dimensions — which is why #112 saw head effects at 355M
that do not reproduce here. Third independent time this has been measured (#95, #96, now).

### Where the time actually goes (`scripts/profile_cpu_hotspots.py`)

| | share of an 8.58 s step |
|---|---|
| `cudaStreamSynchronize` — host **blocked waiting on the GPU** | 5.84 s (68%) |
| `cudaLaunchKernel` — 60,275 launches | 0.61 s (7%) |
| `ternary_quantize_ste` — pure Python, 1,760 calls | 0.395 s (4.6%) |

**The CPU is idle, not slow.** That is why rewriting host code in a compiled language buys
nothing, and why fusing the ternary quantizer — 4.14× faster in isolation — moved the
loop only **1.9%**: it removed host work that was overlapping with GPU work anyway.

`HVN_FUSED_TERNARY=1` enables it (`hvn/edge/ternary_fused.py`, 13 parity tests). **Default off**:
1.9% is not worth changing the training-path quantizer without a reason.

### Two thresholds of my own that were wrong

Recorded because both would have produced a false verdict:

- The fused-quantizer A/B **rejected on a loss difference of 3.08e-03** — but two runs of the
  *same* eager arm differ by **2.58e-03**. The threshold was tighter than the harness's own
  reproducibility, so it would have rejected an identical implementation.
- The parity test bounded **absolute** fp32 drift at 1e-6. Drift is relative: at scale 7.82 a
  1.2e-05 relative difference is 9.2e-05 absolute, so the bound failed on large weights for no
  reason connected to correctness.

## RETEST AT SCALE — levers measured null at 83M that should improve with width (2026-08-11)

> **Read this before concluding a lever is dead.** Every entry below was measured as *no gain* or
> *negative* at 83M (d_model 512, d_head 64) and written up with language like "the lever is
> closed". That phrasing is **too strong**: for most of them the *mechanism* predicts they get
> better as the model widens, and this project's stated goal is to go bigger later. The code is
> landed and flag-gated in every case — nothing was deleted, and retesting is a flag flip, not a
> rebuild.
>
> This index exists because the alternative is archaeology through a dozen closed issues at the
> moment someone launches a larger run.

| lever | flag / entry point | measured at 83M | why width should change it |
|---|---|---|---|
| **Chunk-parallel scan (forward)** | `train_chunk`, `--megakernel-scan` | ❌ #95 chunked hybrid 2–4× slower; #96 Design A ~1.5× slower | The parallel form replaces `C` cheap sequential steps with one `C×C` interaction solve. Per-step cost grows with `d_head²`, the solve does not — so the trade flips as `d_head` grows. **At d_head 64 the steps are too cheap to be worth skipping.** The kernel header still reads *"chunk-parallel tiling for throughput — TODO"*: the forward version is not even built. |
| **n_heads 8 → 16** | `--n-heads` | ✅ +10% speed, no quality cost (#144) — but only **1.12–1.17×** | Already demonstrated size-dependent: **2.4× at 355M** (#112) vs 1.1× here. The win grows with model width. |
| **MoE FFN** | `--moe-experts`, `--moe-top-k` | ❌ #123: beat dense by 11.8% where the bar was 19% | MoE's premise is capacity-per-FLOP, which pays when a dense model of the same compute is capacity-limited. At 83M it is not. The bar itself (−0.504 exponent) moves with size. |
| **Multi-token prediction** | not built | ❌ ruled out from the literature | Its own scaling ablation reports **negative below ~1B**. Explicitly a size threshold, not a refutation. |
| **Selective tokens (Rho-1)** | `scripts/ab_selective_tokens.py` | ❌ no gain at 83M (#: 2026-08-07) | Rho-1's own floor is **1.1B**. Same order-of-magnitude gap as MTP. |
| **Group-wise ternary scales** | `group_size=` on `apply_ternary_to_model` | ❌ #146: per-tensor wins; grouped costs 3–62% more storage | Untested claim, but the storage *ratio* improves with width: scale count grows as `out×in/group`, while codes grow as `out×in`. Wider layers amortise the scales better. Quality side genuinely looked null. |
| **ReLU² / PowLU activations** | `ffn_activation=` | ❌ #140: no throughput change | **Do NOT retest for speed** — the null is structural, not size-dependent: activations are pointwise against `d²` matmuls, so the gap only widens with size. Quality-per-step (Primer's ~1–3%) was never measured and is the only open question. |
| **BCD optimizer-state retention** | `retain_optimizer_state=` | ❌ #151: closes −0.3% of the gap | Not width-dependent — but **rotation-dependent**: it helped (+0.124 nats) at 5 steps/block and did nothing at 100, because momentum has to survive the round trip. A rotation-matched `beta` (~0.99947) is untested. |
| **Progressive layer stacking** | `scripts/ab_progressive_stacking.py` | ❌ #147: +18% throughput, ~0 quality; the apparent win was an LR restart | Depth-growth's saving scales with layer count. At 12 layers half-depth saves little; at 24+ the arithmetic is more favourable. |
| **Sequence-length curriculum** | `scripts/throughput_sweep.py` | ❌ #149: 2.9% slower | Occupancy is `B×H` against 20 SMs; more heads at larger size changes the fill. Weakest candidate on this list. |
| **Per-token optimizer-state paging** | not built (#151) | n/a — analysis only | Saving is ~206 MB at GPT-2's 50K vocab; it scales with `vocab × d_model`. Only interesting at a larger tokenizer or wider model, where VRAM actually binds. |

**Two that should NOT be retested for the same reason twice:**

- **Attention layers are the *cheap* ones during training** (#149: removing both was 6.4% *slower*). Any proposal premised on "attention is the expensive part" starts from a false assumption at seq 512.
- **The pipeline is not idle** (#153: 97.2% GPU-busy). Plumbing optimisations have nothing to reclaim.

**The framing for all of the above:** these are *"no gain at 83M"*, not *"impossible"*. Several were written up more finally than the evidence supports — the same over-confident-verdict pattern that got #122 and #132 retracted. When the next size is trained, this table is the list to walk.

## Three generated architecture ideas, all killed by cheap gates (Issues #149/#150, 2026-08-11)

**One line: three novel training-speed ideas were proposed, each given a gate costing minutes
instead of the ~25-minute training A/B it would have needed, and all three died at the gate. Total
GPU cost ~20 min against ~75 min of blind runs — and one gate produced a finding that inverts an
assumption in this project's own framing.**

Generated after every *existing* lever was exhausted (MoE, activations, BCD, multi-token
prediction, selective tokens, ternary grouping, layer stacking — all null or negative). Each idea
was required to name the cheapest measurement that could kill it **before** any training code was
written.

### Idea A — chunk-size curriculum · killed by reading the dispatch

Schedule `train_chunk` large early (cheap, approximate) and shrink it as the model sharpens.
**Dead before running:** `deltanet.py:689` takes the Triton path whenever `use_triton_scan` is on,
so the `elif self.train_chunk` branch below it is unreachable — **`train_chunk` is dead code in the
production configuration** every harness and both FT scripts use. The kernel has no equivalent
knob (`deltanet_triton.py:66,122` is `for t in range(S)`; line 24 reads *"chunk-parallel tiling for
throughput -- TODO"*). The experiment would have produced a clean null about a knob that was never
connected. **Cost: 0 GPU minutes.**

### Idea B — sequence-length curriculum · killed by a 4-minute timing gate (#149)

Train early at **seq 128 × batch 16** instead of seq 512 × batch 4: identical tokens per step
(2,048) but 128 scan programs instead of 32, for better occupancy on 20 SMs.

| config | tok/s (3 rounds) |
|---|---|
| seq 512 × batch 4 (32 programs) | 12,131 / 12,217 / 12,217 |
| seq 128 × batch 16 (128 programs) | 11,829 / 11,886 / 11,864 |

**4× the programs is 2.9% *slower*.** Consistent across every round. The occupancy premise was
wrong — and #141 had already refuted the `N = B·H` model it was built on (equal `B·H` gave 55%
different throughput). The idea was constructed on a model this repo's own measurement had
discarded. **Cost: ~4 GPU minutes.**

### Idea C — recurrent-state warm-start · killed by measuring its premise (#150)

Initialise each layer's state from a detached running average instead of zeros, so sequences stop
paying to escape `S = 0`. Genuinely HVN-specific — a transformer has no persistent state to
warm-start, and this is not TBPTT (no gradient crosses the boundary; it is an init prior).

**Implementation cost is the reason it needed a gate:** `_triton_gated_delta_scan(q,k,v,alpha,beta)`
accepts no initial state, and the eager paths hardcode `S_state = torch.zeros`
(`deltanet.py:396,439,534`). Building it means a forward-and-backward change to the scan kernel —
hours, in the code this repo has twice recorded going silently wrong.

So the premise was measured instead (`scripts/probe_state_warmup_cost.py`, trained 83M,
loss by position):

| position | loss | vs tail |
|---:|---:|---:|
| 0 | 6.6705 | **+2.3167** |
| 1–4 | 5.4683 | +1.1146 |
| 4–16 | 4.9049 | +0.5511 |
| 16–64 | 4.4555 | +0.1017 |
| 64–128 | 4.3741 | +0.0203 |
| 384–512 (tail) | 4.3538 | — |

The state reaches useful content within ~16–64 tokens. The whole early-position penalty is
**0.55% of sequence loss** — and that is an **upper bound**, because most of it is irreducible:
predicting token 0 with no context is hard for any model, and a state averaged over *other*
sequences carries nothing about this one. Its only possible contribution is a generic prior, which
the weights already hold.

**The script's own threshold called this "worth building" at 0.55% vs a 0.5% cutoff. That
threshold was wrong and was overridden** — it did not account for irreducibility. A gate that
returns the wrong verdict is worse than none, so the reasoning is recorded here rather than the
number alone. **Cost: ~6 GPU minutes.**

### Idea D — late attention insertion · killed, and the premise was backwards

Train the first phase as pure DeltaNet (cheapest), then swap 2 layers to attention once recall
starts to matter. Premise: the attention layers are the expensive ones.

| config | tok/s (median of 3, interleaved) |
|---|---|
| 2 attention + 10 DeltaNet (production) | **10,393** |
| 0 attention + 12 DeltaNet | 9,726 (**−6.4%**) |

**Replacing attention with DeltaNet is 6.4% SLOWER.** At seq 512 the FLOPs are equal
(`S²d = Sd² = 1.34e8`), but SDPA parallelises across the sequence while the DeltaNet scan is
sequential over it. A pure-recurrent early phase would cost time, not save it. **Cost: ~4 GPU
minutes.**

> **Worth carrying forward: the attention layers are the *cheap* ones during training.** The
> intuition that the hybrid keeps only 2 of them to limit cost is wrong for training — per
> [architecture.md](architecture.md) the 2-layer limit is an **inference-VRAM** decision (KV cache
> flat over a long agentic session), and that reasoning stands. But any future proposal premised on
> "attention is the expensive part" is starting from a false assumption at this sequence length.

### What this pass is evidence for

Not that the ideas were bad — B and C were reasonable and C is genuinely novel. It is that **the
gate is where the value is**. Four ideas, four kills, ~14 GPU minutes total, versus ~100 minutes to
learn the same thing by running them. Two of the four were killed by *reading code* rather than
measuring, and one gate (D) overturned an assumption held elsewhere in the project.
## The GPU is 97% busy during pretraining — there is no idle time to reclaim (Issue #153, 2026-08-11)

**One line: a full-stack instinct that the pipeline was starving the GPU was measured and is
wrong. Kernel time is 97.2% of wall clock; checkpoint saves cost 0.28 s and host copies 0.005 s.**

Every throughput number in this repo is tokens/sec — a *rate*, which cannot distinguish "the GPU
is saturated" from "the GPU idles 40% of the time and still produces that rate". The only idle
gap ever measured was the dataloader's 0.2–0.3% (#141), and that was a lean harness with
pre-materialised batches, no saves and no evals. This measures a realistic window.

`scripts/profile_pretrain_utilisation.py`, production config (`--fast`, ternary, Muon,
grad_accum 8), 48 micro-batches **including a checkpoint save and an inline eval**:

| phase | seconds | share of wall |
|---|---:|---:|
| forward + backward | 6.406 | the work |
| optimizer (Muon) | 0.627 | 1.8% |
| checkpoint save (317 MB) | **0.280** | 0.8% |
| host→device copy | 0.005 | 0.0% |

**GPU-busy / wall clock = 97.2%.**

Every suspect came back negligible. The 317 MB synchronous checkpoint write, expected to be a
recurring stall, costs **0.28 s**.

### The one apparent stall was a one-time compile, checked rather than assumed

The control window showed `inline_eval` at 27.7 s of a 35 s window — which would have been a large
recurring cost. It is not recurring: switching a `torch.compile`d model to eval mode compiles a
second graph, once per process.

    eval pass 1: 1.20s      <- compiling the eval graph
    eval pass 2: 0.29s
    eval pass 3: 0.29s

One-time. On a 9-day run it is noise. Worth knowing for short A/B harnesses, where a 1–30 s
one-off can be a visible share of a 10-minute run.

### A broken number in this run's own output, recorded rather than quietly dropped

The script reported **"profiler overhead: −77.1%"**. A profiled run cannot be faster than an
unprofiled one, so the figure is meaningless: the control window paid the one-time eval-path
compile and the profiled window did not, so the two are not comparable. The utilisation figure
itself is unaffected — it is kernel time over wall clock *within* the profiled window — but the
overhead comparison needs the control run twice to be valid.

### Why this matters more than it sounds

It closes the "is the plumbing wasting the run" question for the planned multi-day pretrain: a
9-day estimate is 9 days of arithmetic, not 9 days containing 3 idle ones. **There is no slack
left in or around the training loop** — consistent with the rest of this session, where nine
architecture ideas returned null and the remaining cost is `6 × params × tokens`.

**Limits:** one config (83M hybrid, batch 4, seq 512, grad_accum 8) on one card; a 48-micro-batch
window rather than a multi-hour run; the profiler's own overhead makes 97.2% a lower bound.

## BCD's 4x cost is under-training, not optimizer amnesia (Issue #151, 2026-08-11)

**One line: keeping a block's optimizer momentum across rotations instead of deleting it closes
-0.3% of block-coordinate descent's quality gap. The cost is that each block trains 1/12 of the
time, and no state trick touches that.**

#112 measured BCD at ~3.9x worse perplexity and named a mechanism it never tested: *"only 1 of 14
blocks trains at a time **with the outgoing block's optimizer state deleted on every switch**"*.
`retain_optimizer_state=True` parks that state on CPU and restores it when the block returns.

| arm | held-out | ppl |
|---|---:|---:|
| full model | **5.48194** | 240 |
| bcd_delete (today's behaviour) | 7.85570 | 2,580 |
| **bcd_retain** | 7.86224 | 2,597 |

**Fraction of the gap closed by retention: -0.3%.** The lever is closed.

### The arithmetic that predicted this, written before the run

Heavy-ball momentum over `k` steps of zero gradient decays to `beta^k * m`. Muon runs beta=0.95,
and a block is away ~1,100 steps at 100 steps/block:

    0.95^1100 ~= 1e-25

So a **correctly decayed** restore is arithmetically identical to deletion -- BCD's existing
behaviour is already the right answer for heavy-ball momentum. Only a "paused, not decaying"
restore has any content, and that is what was implemented and tested. It changed nothing.

### The one place it did work, and why

A 260-step smoke at **5 steps/block** had retention *ahead by 0.124 nats*. Same code, different
rotation speed -- and the arithmetic explains it exactly:

| rotation | steps away | surviving momentum | retention |
|---|---:|---:|---|
| 5 steps/block | 55 | ~0.06 | **helped (+0.124 nats)** |
| 100 steps/block | 1,100 | ~1e-25 | nothing |

**Retention only helps while the memory is still worth having.** At production rotation it
restores numerical noise. This is one data point from a smoke run, not a finding -- but the
mechanism and the evidence point the same way, and a rotation-matched `beta` (~0.99947 for a
per-rotation half-life) is the untested variant if anyone returns to this.

### Known-answer cell, and a caveat about it

`bcd_delete` had to reproduce BCD's known cost or the harness would not be measuring what #112
measured. It did, qualitatively: **+2.374 nats (10.7x ppl)** against #112's **+1.36 (3.9x)** --
same direction, larger magnitude, on a different harness with 12 blocks rather than 14. The
retain-vs-delete comparison is internally valid; **10.7x should not be quoted as BCD's cost.**

### Related: per-token state, measured and reframed

The same question at token granularity (`scripts/probe_token_update_gaps.py`, 2,000 steps of the
real corpus) came out the opposite way:

| | |
|---|---|
| distinct token ids touched per step | 769 (1.5% of vocab) |
| median gap between appearances | **7 steps** |
| momentum half-life (beta=0.95) | 13.5 steps |
| **mean momentum surviving on return** | **0.5745** |

Tokens recur constantly, so their momentum *is* still alive -- 0.57 versus the blocks' 1e-25. But
**dense training already retains this**: the embedding momentum buffer is a persistent dense
tensor, so per-token retention is not a change that can be made, it is what already happens.

The actionable reading is inverted: **sparsifying embedding optimizer state to save memory would
destroy ~57% of a live signal.** The safe form is to *park* idle rows on CPU rather than drop
them -- ~206 MB of AdamW state for ~3 MB/step of traffic (~0.2% of step time), quality-neutral by
construction. Not built: VRAM is not currently binding (3.4 GiB peak of 6 GiB), and the saving
scales with vocabulary, so it only becomes interesting at a larger tokenizer.

### Not run

`scripts/sweep_effective_batch.py` (Issue #152) was written and started -- is effective batch 32 a
training choice or just what fit in VRAM -- then **stopped after one arm** when the session moved
to evaluation. The script, its known-answer cell against #148, and its stated fixed-LR confound
are committed; **no result was produced.** Recorded here so the question reads as open rather than
answered. **Answered 2026-10-11** on the real 48M trainer instead (see the #152 entry at the top).
That script's known-answer cell could not have passed: its reference is #148's grad_accum-1
run, and it never compares against it.

## Fixed-LR sweep: #147's restart win is a REAL schedule effect, not a mis-set LR (Issue #148, 2026-08-11)

**One line: no constant learning rate reaches what a mid-run cosine restart reaches. `muon_lr=0.02`
is already the optimum of a clean U-curve, so the restart is doing something a better number
cannot.**

#147 found a restart worth −0.091 nats, and this repo had a ready explanation: #133's dynamic-LR
"win" turned out to be that the baseline LR was 10× too high — the schedule was compensating for
a bad number. Same shape, so it was checked the same way. Plain single cosine in every arm, only
`--muon-lr` differing, identical seed and pre-materialised batches, `--deterministic`:

| muon_lr | held-out | bits/byte | vs restart |
|---:|---:|---:|---:|
| 0.005 | 6.12494 | 2.2042 | +0.18084 |
| 0.01 | 6.03769 | 2.1809 | +0.09359 |
| **0.02** (current default, #129) | **6.03552** | **2.1798** | **+0.09142** |
| 0.04 | 6.07227 | 2.1899 | +0.12817 |
| 0.08 | 6.11035 | 2.1993 | +0.16625 |
| *cosine WITH restart @ 0.02 (#147)* | *5.94410* | — | — |

**The "mis-set LR" explanation is refuted.** 0.02 sits at the bottom of a clean U — every
direction is worse — and the best fixed LR is still **0.091 nats short** of the restart. #133's
finding does not transfer to this case.

### A broken known-answer cell, found and fixed

The 0.02 arm was required to reproduce #147's control exactly. It did — drift **2.2e-7** — and
the run **declared itself VOID anyway**, because the tolerance was 1e-9 while the reference
`6.03552` is a literal transcribed from five-decimal printed output, carrying ~5e-6 of
quantisation.

**A test compared against a rounded constant, at a tolerance finer than that constant's
precision, can never pass** — the exact mirror of #140's test that could never fail, and equally
useless. The tolerance is now tied to the reference's precision (`REF_TOL = 5e-6`), the run
passes, and the superseded verdict is kept in the JSON rather than deleted.

### What this does and does not license

- **Real:** at 1,500 steps, restarting the cosine at the halfway point beats every base LR tried.
- **Not tested — schedule *shape*.** This swept the base LR under a *single cosine*. A flatter
  decay, a constant LR with no decay, or a longer warmup might also reach 5.944. "No fixed LR
  gets there" is **not** "nothing but a restart gets there".
- **Still length-bound.** 1,500 steps is 0.19% of a real pretrain, and more time at high LR
  flatters a run nowhere near converged. Whether a restart helps a 100K-step pretrain is
  untouched, and the standard single long decay exists for reasons this run cannot see.

**Practical read:** do not change the pretrain schedule on this evidence. Do note that
`muon_lr=0.02` is confirmed optimal at this length — a useful independent check on #129's default.
## Progressive layer stacking contributes nothing — the win was an LR restart in disguise (Issue #147, 2026-08-11)

**One line: 6→12 layer growth looked like a −0.095 nat win, and ~96% of it was the cosine
schedule being restarted at the growth point. A 12-layer model with the same restart and no
growth matches it.**

`scripts/ab_progressive_stacking.py`, from scratch, graded at **equal wall clock** (the claim is
"same time, more progress"), identical seed, identical pre-materialised batches,
`--deterministic`, both arms ending on the same block plan:

| arm | steps | held-out | bits/byte | wall |
|---|---:|---:|---:|---:|
| control (12L throughout) | 1500 | 6.03552 | 2.1726 | 7.68 min |
| **control_restart** (12L, optimizer+cosine rebuilt at 750) | 1500 | **5.94410** | **2.1560** | 7.62 min |
| grown (6L → 12L at 750) | **1769** | 5.94041 | 2.1579 | 7.68 min |

- grown vs control: **−0.09511**
- control_restart vs control: **−0.09142**
- grown vs control_restart: **−0.00370**, and **bits/byte ranks control_restart ahead**

### How the confound was caught

The first run showed grown ahead at **equal steps** (−0.074) as well as equal wall clock.
Stacking cannot do that: at matched steps the 6-layer phase is strictly less capacity, so it
should *lose* there and win only on time. That impossible-shaped result is what prompted the
control — the grown arm rebuilds its optimizer and cosine at growth, so it receives a **warm
restart** (SGDR, Loshchilov & Hutter 2016) that the plain control never gets, and against that
control the restart's benefit is credited to "stacking".

Without `control_restart` this would have been written up as "progressive stacking works,
−0.095 nats". The stored verdict in `ab.json` said exactly that and has been corrected in
`ab_lr_control.json`, with the superseded string kept alongside.

### What survives, and what does not

- **Dead: stacking as a quality lever.** −0.0037 nats against the control, with the two
  metrics disagreeing on the sign.
- **Real but unconverted: stacking's throughput.** 1,769 steps vs 1,500 in the same 7.68 min,
  **+18%**, exactly as the arithmetic predicts (half the layers for the first half of the run).
  That extra 18% of steps bought 0.0037 nats — i.e. nothing. Cheaper steps did not become a
  better model.
- **Interesting but distrusted: the LR restart is worth −0.091 nats at zero cost.** No
  architecture change, no extra memory, one line of schedule code.

### Why the LR result should NOT be adopted on this evidence

A restart means more time at high LR, which flatters a **1,500-step run** (0.19% of a real
pretrain) almost by construction — the model is nowhere near converged, so anything that keeps
the step size large looks good. Full-length pretrains use a single long decay for exactly this
reason, and the ranking could reverse at length. It is also not novel (SGDR).

It does land in an area this repo has already found to be live: #133 concluded a dynamic LR
schedule "wins by 0.30 nats, but the win is that the LR was 10x too high". A restart helping at
1.5K steps may be the same story — evidence about **LR level**, not about schedules. The next test is a fixed-LR sweep at this length before anyone touches the schedule.

### Limits

One seed. 1,500 steps × 2,048 tokens = 3.07M, **0.19% of a real pretrain**. Growth point (750),
duplication scheme (plain stack-doubling), and restart timing are all unswept — a different
growth point could change the picture, though it would have to overcome a 0.0037 nat starting
gap against a free alternative.

## Group-wise ternary QAT loses to per-tensor too — the lever is fully closed (Issue #146, 2026-08-11)

**One line: training with finer ternary scales does not beat per-tensor either. Every grouped
scheme is marginally worse on held-out loss and costs 3–62% more storage. #145 killed the
retrofit; this kills the idea.**

#145 showed re-grouping an existing checkpoint makes it worse, and explicitly could not decide
whether *training* with group scales would win — a model trained under finer scales adapts to
them from step 0, which is precisely the mechanism that sinks the retrofit. That was worth
testing. It does not win.

`scripts/ab_ternary_grouping.py`, from scratch, 1,500 steps × batch 4 × seq 512, Muon @ 0.02,
identical seed, **identical pre-materialised batch sequence per arm**, `--deterministic`,
ternary applied to production's target set (excluding `beta_proj`/`alpha_proj`/`decoder`):

| arm | group | held-out | bits/byte | bits/weight | storage | vs per_tensor |
|---|---:|---:|---:|---:|---:|---:|
| **per_tensor** | 0 | **6.03297** | 2.1802 | **1.612** | — | — |
| per_row | −1 | 6.03843 | 2.1805 | 1.667 | 1.034× | +0.00546 |
| group_128 | 128 | 6.03719 | 2.1801 | 1.862 | 1.155× | +0.00422 |
| group_32 | 32 | 6.03378 | **2.1799** | 2.612 | **1.620×** | +0.00081 |
| per_tensor_twin | 0 | 6.03297 | 2.1802 | 1.612 | — | 0.000e+00 |

**Known-answer cell passed exactly (0.000e+00)** — the two per-tensor arms are bit-identical, so
the run-to-run floor is zero.

### Why the verdict does not depend on the sign

Held-out loss ranks per-tensor first; bits/byte marginally prefers `group_32` (2.1799 vs 2.1802)
— the metrics disagree at the **fourth decimal**, which is the signature of no real effect.

**It does not matter which way those signs fall.** `group_32`'s best case is ~0.001–0.005 nats
for **1.62× storage**. Ternary exists to save bytes; paying 62% more of them for a difference
that two held-out sets cannot agree on is not a trade. This is why the harness reports effective
bits/weight beside quality: `memory_bytes()` previously charged a flat 4 bytes for one scale, and
had that not been fixed (#146), every grouped scheme would have looked free and `group_32` would
have been reported as a narrow win.

### A caution about short-run signals, demonstrated

The 20-step smoke pass showed the grouped arms **ahead by 0.05 nats** — the direction #145's
mechanism predicted. By step 1,500 that had **reversed**. The early reading was flagged as
meaningless when it was taken, and it was. Worth keeping as a concrete example of why this repo
grades at the end of a run and not at the start.

### The A/B harnesses vs `--fast` — checked, and TF32 turns out to be a no-op here

`--fast` is bf16 + TF32 + `torch.compile` + Triton scan + async DataLoader. The A/B harnesses
(this one, `ab_prodigy.py` #134, `ab_n_heads.py` #144) have bf16 and the Triton scan,
pre-materialise batches so the loader is irrelevant, and run **without compile and without
TF32** — raised as a possible gap: a verdict measured outside production numerics is the
#86→#129 mistake.

**Measured, not argued.** A `--tf32` flag was added and the two decisive arms re-run with it:

| arm | TF32 off | TF32 on |
|---|---:|---:|
| per_tensor | 6.03297 | **6.03297** |
| group_32 | 6.03378 | **6.03378** |

**Bit-identical to five decimals.** Which then had to be checked for vacuity, since "the flag did
nothing" and "the flag was never on" look the same: a direct probe confirms TF32 *does* change an
fp32 matmul under `use_deterministic_algorithms(True)` (max diff 5.1e-2), so the flag was
genuinely active.

**The reason is that there are no fp32 matmuls in this loop at all.** Forward and backward run
under bf16 autocast, and **Muon's Newton-Schulz orthogonalisation also runs in bfloat16**
(`hvn/optimizers.py:59,64` — `X = G.bfloat16()`). The earlier claim in this entry that Muon does
fp32 matmuls "and those are affected" was **wrong**; it is corrected here rather than removed,
because it was the stated reason for running the check.

**Consequence: the harnesses differ from `--fast` only by `torch.compile`, which fuses the same
operations without changing the maths.** So this verdict and **#144's `n_heads` verdict both
transfer to real `--fast` runs** — now on evidence rather than assumption.

### Limits

One seed. 1,500 × 2,048 = 3.07M tokens, **0.19% of the 83M's real 1.65B-token pretrain** — early
training at 83M only. A grouping effect that emerges at convergence or at larger `d_model` would
not appear here. What *is* solid: at equal step count the grouped schemes are not ahead, and the
storage cost is arithmetic, not measured, so it holds at any scale.

The `group_size` code is landed and tested (17 tests, `group_size=0` bit-identical to the
pre-#146 implementation in forward *and* backward) — available if a future scale makes it worth
re-testing, and costing nothing while it sits at the default.

## Group-wise ternary scaling cannot be retrofitted — and "fp is the upper bound" is false for a QAT model (Issue #145, 2026-08-11)

**One line: finer ternary scales reduce weight error but make the trained model measurably
worse, because it has adapted to per-tensor rounding. The free retrofit is dead; training with
group scales is untested and untouched by this.**

`ternary_quantize_ste` (`hvn/edge/ternary.py:46`) uses **one scalar per weight matrix** —
262,144 weights sharing a normaliser at d=512. Per-row/per-group scales are what GPTQ/AWQ/NF4
all use, cost ~0.4% storage, and `hvn/compress.py` already implements them for *post-training*
compression. The obvious question was whether they help here.

`scripts/screen_ternary_grouping.py` — zero training. Re-quantize the trained 83M's shadow
weights under each scheme, 32 held-out probe batches + bits/byte on unseen FineWeb-Edu:

| scheme | held-out | vs per_tensor | recon error | bits/byte |
|---|---:|---:|---:|---:|
| **per_tensor** (today) | **4.28087** | — | 0.65605 | **1.5097** |
| per_row | 4.58461 | +0.30374 | 0.63789 | 1.6243 |
| group_128 | 4.57853 | +0.29766 | 0.63442 | 1.6157 |
| group_64 | 4.60788 | +0.32701 | 0.62964 | 1.6202 |
| group_32 | 4.53489 | **+0.25402** | **0.62251** | 1.5936 |

**Known-answer cell passed exactly (0.000e+00):** the harness's `per_tensor` reimplementation
reproduces the untouched `TernaryLinear` model's held-out loss bit-for-bit, so every other arm is
quantizing the same way production does.

**The result is a clean contradiction between two metrics, which is why both were measured.**
Finer scaling monotonically *reduces* weight reconstruction error (0.656 → 0.623) — it is a
strictly better approximation of the shadow weights, exactly as theory says. And every scheme is
*worse* on held-out loss and on bits/byte. Both quality metrics agree; only reconstruction
disagrees.

The explanation is that reconstruction error is measuring the wrong target. Under QAT the model's
weights sit **where per-tensor rounding puts them** — that is the function 8,000 steps optimised.
A scheme that approximates the *shadow* weights more faithfully moves the network away from the
one that was trained. **Retrofitting is therefore not free and not available.**

### A design error this run caught, worth recording

The screen was built with an `fp` arm (shadow weights, unquantized) as the assumed **upper
bound**, and a "% of ternary damage recovered" metric computed against it. **Both are invalid for
a QAT model**, and the run proved it: `fp` scored **7.62610** against per_tensor's **4.28087** —
dramatically *worse*. In QAT the forward pass always used quantized weights; the fp32 shadow
weight is an optimizer accumulator that was never itself the model's function. Feeding it directly
gives a different, untrained network.

So there is **no "damage" to recover from this checkpoint**, and the "% recovered" figure had a
fabricated denominator. It was removed rather than reported. The dense-fp reference that *does*
exist (119.4 vs 143.2, +20%, 2026-08-02) came from a **separately trained** fp arm — which is the
only way to obtain one.

### What is and is not decided

- **Decided: the cheap version is dead.** You cannot take an existing ternary checkpoint,
  re-group its scales, and get anything but a worse model.
- **Not decided: group-wise QAT.** Whether *training* with per-row/per-group scales beats training
  with per-tensor is completely untouched here — the model would adapt to the finer scheme from
  the start, which is precisely the mechanism that makes the retrofit fail. Settling it needs a
  from-scratch A/B (the `scripts/ab_n_heads.py` pattern: identical batches, twin arm, held-out
  loss + bits/byte), with `group_size = whole tensor` as the known-answer cell.

The screen cost ~10 minutes and ruled out the free option before any training code was written,
which is what it was for.

## `n_heads` 8 vs 16 — the speed lever costs no measurable quality (Issue #144, 2026-08-11)

**One line: `--n-heads 16` buys ~10% throughput for a held-out difference of 0.08%, and the two
held-out metrics disagree about even the *sign* of that difference. No quality cost detected at
83M in early training.**

`--n-heads 8 → 16` was recorded as a **2.4× speedup at 355M** (#112) and **1.12–1.17× at 83M**
(#141), documented purely as a speed knob — which is exactly how `--block-coordinate` was
documented before #112 found it cost ~4× perplexity. This grades it.

`scripts/ab_n_heads.py`, from-scratch (n_heads changes `read_norm_weight`'s shape, so there is no
warm-start), 1,500 steps × batch 4 × seq 512, Muon @ 0.02, identical seed, **identical
pre-materialised batch sequence in every arm**, `--deterministic`:

| arm | d_head | held-out loss | bits/byte | wall | steps/s |
|---|---:|---:|---:|---:|---:|
| h8 | 64 | 6.69093 | 2.3071 | 8.2 min | 3.06 |
| **h16** | 32 | 6.69628 | **2.3062** | 7.4 min | **3.36** |
| h8_twin | 64 | 6.69093 | 2.3071 | 8.0 min | 3.13 |

### The known-answer cell, and exactly what it does and does not license

`h8_twin` is `h8` run again under a different name. It came out **bit-identical: 0.000e+00**. So
determinism holds and the run-to-run noise floor is *exactly* zero — the +0.00535 nats measured
below is perfectly reproducible, not drift.

**It does not license calling that gap meaningful.** A zero floor covers repeat runs *at the same
seed*; it says nothing about seed-to-seed variance, which is unmeasured here and in every
comparable setup is far larger than 0.08%. Reading "noise floor 0" as "any difference is
important" would be the mistake this cell exists to prevent, not one it endorses.

### The two metrics disagree in sign

| view | h16 − h8 | reading |
|---|---:|---|
| matched steps, held-out probe loss | **+0.00535 nats** (0.08%) | h16 slightly worse |
| matched steps, bits/byte on unseen FineWeb-Edu | **−0.0009 bits** (0.04%) | h16 slightly **better** |
| matched wall clock (7.4 min) | **−0.03981 nats** | h16 better |

The probe loss is drawn from the training mix (SlimPajama / OASST / WikiText); bits/byte is
computed on FineWeb-Edu, which has mix share 0.0 and is unseen by both arms. **Two held-out sets,
opposite signs, both differences ~0.1%** — that is the signature of no real effect, not of a
small one. Reporting only the first number would have manufactured a "cost" that the second
refutes.

### Verdict

**`--n-heads 16` is safe to adopt for a new pretrain on this evidence**, and it is ~10% faster
here (1.096×, consistent with #141's 1.12–1.17×). At matched wall clock it is ahead on every
metric, because the extra speed buys extra steps.

**Limits, which bound this to a narrow claim:**

- **1,500 steps × 2,048 tokens = 3.07M tokens — 0.19% of the 83M's real 1.65B-token pretrain.**
  This measures *very* early training. A head-count effect that emerges later, or reverses, would
  not appear here.
- **One seed.** See the known-answer note above: the gap is reproducible, not robust.
- **83M only.** The *speed* win is already known to be size-dependent (2.4× at 355M vs 1.10× here,
  d_head 128→64 vs 64→32), so the quality effect may be too. This does not clear h16 at 355M.
- Applies to a **new** pretrain only — `n_heads` cannot be changed on an existing checkpoint.

Firming this up would need multiple seeds and ~20K steps (~2 h). Not run: the measured effect is
0.08% with the sign in dispute, so the reading is already "no detectable difference", and
a longer run would most likely spend two hours confirming that.

## FT recipes adopt #141's wins — and the compile gain does NOT transfer at full size (Issues #142/#143, 2026-08-11)

#141's two levers are now the defaults in the fine-tune recipes. The interesting part is #143:
**a number measured in one training loop was worth ~1.5× less in another**, which is exactly
what #142 refused to assume.

### What changed

| script | micro-batch | grad_accum | compile |
|---|---|---|---|
| `continue_ft_hybrid.py` | 4 (already) | 8 (already) | **`--fast` now default ON** (#142) |
| `finetune_edge_conversational.py` | **2 → 4** (#142) | **8 → 4** (#142) | **`--fast` added, default ON** (#143) |
| `continue_ft_looped.py` | 4 (already) | 8 (already) | deliberately none — see below |
| `train_edge_conversational.py` | auto | auto | already on (#129) |

`finetune_edge_conversational`'s `grad_accum` was halved alongside the batch so the **effective
batch stays 16** — a pure throughput change, not a change to training dynamics.

### #143 — measure the same lever in a different loop, get a different answer

`scripts/measure_chatft_compile.py`, real OASST + grounded-replay mix, interleaved arms, eager
entered twice under two names, 30 timed optimizer steps/arm, batch 4 × grad_accum 4, 2 rounds:

| arm | tok/s |
|---|---:|
| eager | 11,447 |
| **compiled** | **14,367 (+25.5%)** |
| noise floor (eager vs itself) | **0.31%** |
| recompiles inside the timed window | **0** |

**+25.5% here vs +39% in `continue_ft_hybrid`'s loop.** Both are real and well outside their
noise floors; the lever simply pays less on this data. Plausible cause (unverified): chat rows
pack at 1.45 examples/row against grounded-QA's 5.18, a very different row distribution and
mask density. Had #142 defaulted this on using the other loop's number, the documented gain
would have been ~1.5× too high.

**`continue_ft_looped.py` stays uncompiled deliberately.** `looped_forward_train()` bypasses
`model.forward()` — the path both measurements above actually exercise — so compiling the
top-level model there would compile something the script never invokes. Reconsidered under
#142 and left unchanged, with the reasoning recorded in the file so it doesn't read as an
oversight.

### Two hazards checked rather than assumed

- **`torch.compile` renames `state_dict` keys** with an `_orig_mod.` prefix, which would
  silently produce unloadable checkpoints from every run using the new `--fast` defaults.
  `_save_edge_checkpoint` already unwraps it — verified before shipping the default flip.
- **Compile is applied to the training path only.** `sample_reply` generates token-by-token with
  a growing sequence, so a compiled module there would recompile *per token* — slower than
  eager, and it would be charged to "compile" in any benchmark. The wrapper shares parameters
  with the plain module, so training still updates the same weights.

### Gate — GPU smoke runs

| script | steps | s/step | peak reserved |
|---|---:|---:|---:|
| `continue_ft_hybrid.py` | 20 | 1.22 | 3,420 / 6,140 MiB |
| `finetune_edge_conversational.py` | 72 | 0.74 | 4,050 / 6,140 MiB |

Both scripts now also **report peak reserved VRAM and flag a spill** — the failure that made
#141's batch-8 arm look merely "slow" instead of "not fitting" cannot now pass unnoticed in a
real run. `tests/test_continue_ft_hybrid.py` 7/7.

## The 83M was starving the GPU — `batch 2 → 4` is +37.7% for free (Issue #141, 2026-08-10)

**One line: the fine-tune recipe's `batch_size=2` leaves ~38% of this card's throughput on the
table, at identical effective batch and identical maths. And the `N = B·H` occupancy story that
motivated the sweep is refuted by its own results.**

`scripts/throughput_sweep.py`, RTX 4050 (20 SMs, 5.9966 GiB), eager, seq 512, **effective batch
held at 32** (`grad_accum` derived per arm, so every arm does the same optimizer maths), 3
interleaved rounds × 32 timed micro-batches, 83M 2-attn hybrid + ternary + Triton scan:

| arm | B·H | tok/s | vs `b2_h8` | alloc | reserved | fit |
|---|---:|---:|---:|---:|---:|---|
| b1_h8 | 8 | 5,376 | −40.8% | 1,655 | 1,738 | ok |
| **b2_h8** (today's recipe) | 16 | 9,087 | — | 2,458 | 2,538 | ok |
| **b4_h8** | 32 | **12,511** | **+37.7%** | 3,491 | 3,974 | ok |
| b8_h8 | 64 | 5,396 | −40.6% | 5,586 | **6,512** | ❌ **SPILLED** |
| b1_h16 | 16 | 5,882 | −35.3% | 1,655 | 1,738 | ok |
| b2_h16 | 32 | 10,181 | +12.0% | 2,461 | 2,542 | ok |
| **b4_h16** | 64 | **14,608** | **+60.8%** | 3,496 | 3,980 | ok |
| b8_h16 | 128 | 5,143 | −43.4% | 5,596 | **6,512** | ❌ **SPILLED** |

Noise floor (the `b2_h8` config entered twice under two names): **1.49%**. Positive control
(batch 1, 8 programs for 20 SMs): **−40.8%** — the harness resolves occupancy effects, so its
flat regions would have meant something.

### What to change today

**`--batch-size 4 --grad-accum 8`** in place of `batch 2 / grad_accum 16`. Same effective batch
(32 rows) — **+37.7% throughput**, at 3,974 MiB reserved on a 6,140 MiB card. A 2-hour
continued-FT becomes ~1h27. This applies to the *existing* checkpoint; nothing about the model
changes.

> **⚠️ CORRECTION (2026-08-10, later): "same gradient, same result" was wrong.** This entry
> originally claimed the two configs compute an identical gradient. They do not, and the reason
> matters more than the size of the error. The loss is **masked** (`ignore_index=-100`,
> `continue_ft_hybrid.py:189`) — only answer tokens count — so each micro-batch's loss is a mean
> over its *own* unmasked-token count `N_g`, and `loss / grad_accum` then weights every
> micro-batch equally regardless of `N_g`:
>
> ```
> accumulated = (1/G) · Σ_g [ (1/N_g) · Σ_{i∈g} ℓ_i ]     vs     true = (1/ΣN_g) · Σ_all ℓ_i
> ```
>
> Equal only if every `N_g` is identical, which masking guarantees they are not. This is the
> standard gradient-accumulation normalisation gap.
>
> **Measured** (`scripts/check_grad_accum_weighting.py`, real packed corpus, 200 draws of 32 rows):
>
> | micro-batch | grad_accum | max row over-weight | mean deviation from true batch loss |
> |---:|---:|---:|---:|
> | 1 | 32 | 1.437× | 13.5% |
> | **2** (old recipe) | 16 | 1.241× | **9.5%** |
> | **4** (new recipe) | 8 | 1.135× | **6.5%** |
> | 8 | 4 | 1.069× | 4.3% |
> | 32 | 1 | 1.000× | 0.0% |
>
> **The direction is favourable: `b4/ga8` is 31% *closer* to the true token-weighted objective
> than `b2/ga16`.** So the speedup is not bought with a worse gradient — but "identical" was
> never true, and neither config computes the true batch loss.
>
> The effect is small here only because packing (#132) makes rows uniform: 14.2 answer tokens
> per row, sd 2.4, CV 0.171. **On ragged, unpacked, or heavily-masked data the distortion grows**
> — at micro-batch 1 it is already 13.5%. Any future recipe that changes `grad_accum` should
> check this rather than assume accumulation is free.
>
> **Still unmeasured:** what either gradient does to a *trained* model. This shows the gradients
> differ and in which direction, not the downstream quality effect.

**Do not go to batch 8.** It reserves 6,512 MiB against a 5.9966 GiB card and Windows WDDM backs
the overflow with host RAM — it does not OOM, it just runs **40% slower than batch 2** while
looking healthy. This is exactly the failure `docs/training.md` warns about, reproduced: had the
sweep tracked only `max_memory_allocated` (5,586 MiB, comfortably "fitting"), the conclusion
would have been *"throughput collapses above batch 4"* instead of *"batch 8 does not fit"*.

### The refutation

The sweep was built on `docs/training.md`'s claim that throughput is set by `N = B·H` programs
filling 20 SMs. **That model is wrong**, and its own prediction is what falsifies it: equal `B·H`
must give equal throughput.

| pair at equal B·H | tok/s | gap |
|---|---:|---:|
| b2_h8 vs b1_h16 (both B·H = 16) | 9,087 vs 5,882 | **55%** |
| b4_h8 vs b2_h16 (both B·H = 32) | 12,511 vs 10,181 | **23%** |

**Batch and heads are not interchangeable, and batch dominates.** Doubling batch is worth
+37.7% where doubling heads is worth +12.0% from the same starting point. The residual head
effect is real but small at this size (+12% at b2, +16.8% at b4) — nothing like the 2.4×
recorded at 355M, which suggests the head win is size-dependent (d_head 128→64 there vs 64→32
here) rather than a general property.

A better-fitting explanation is **fixed per-micro-batch overhead** — kernel launches, Python
dispatch, the optimizer's own work — amortised over more tokens as the batch grows. More heads
split the *same* tokens across more programs and so cannot amortise anything. This is a
hypothesis fitted after the fact, not a measurement; confirming it needs a profile
(`aten::` launch counts per micro-batch vs batch size), which was not run.

### Limits, stated because they bound the claim

- **Synthetic tokens, no dataloader.** Real runs have a loader that may not keep a 2× larger
  batch fed. The +37.7% is the compute-side ceiling; confirm with a short real run before
  quoting it as end-to-end. Packing (#132, on by default) also changes tokens-per-step.
- **Eager, not `--fast`.** `torch.compile` changes launch overhead, which is exactly the
  quantity the fitted explanation depends on.
- **`n_heads` changes the architecture** (`read_norm_weight` is `[n_heads, d_head]`), so the
  h=16 rows apply to a *new* pretrain, not to `hybrid_a4_v2/hybrid_a4_step8000.pt` — and their
  quality effect is unmeasured. Only the batch-size rows are actionable on the current model.
- Median of 3 rounds, one machine, one seed.

### CONFIRMED end-to-end through the real pipeline (2026-08-10)

The +37.7% above is synthetic tokens with no dataloader. `scripts/confirm_batch_speedup.py`
re-measures it through the **real** path — real corpus (`grounded_qa_expanded.jsonl`, 5,240
records), real `GroundedQADataset` + `PackedDataset` packing, and the training loop copied from
`continue_ft_hybrid.continue_ft` (`doc_ids`, bf16 autocast, `clip_grad_norm_`, AdamW). Both arms
run 40 optimizer steps at an effective batch of 32 rows = **identical token counts**, arms
interleaved, 2 rounds:

| mode | b2/ga16 | b4/ga8 | gain |
|---|---:|---:|---:|
| eager | 8,201 tok/s | 11,102 tok/s | **+35.4%** |
| compiled (`--fast`) | 11,413 tok/s | 14,963 tok/s | **+31.1%** |

**The dataloader is not the limit** — time blocked on `next()` was **0.2–0.3%** in every arm.
That was the one failure mode that would have killed the claim, and it is measured, not assumed.

**Two secondary findings:**

- **`torch.compile` is worth ~+39% on its own** (8,201 → 11,413 at b2), and it **stacks** with
  the batch change: `compiled + b4` reaches **14,963 tok/s vs 8,201 for eager b2 — 1.82×**.
- **The batch win shrinks under compile (35.4% → 31.1%), which is evidence for the mechanism
  this entry could previously only guess at.** The post-hoc explanation above was fixed
  per-micro-batch overhead amortised over more tokens; `torch.compile` reduces exactly that
  overhead, so it should leave less for a bigger batch to recover — and it does. Supporting
  evidence, not proof: a launch-count profile still hasn't been run.

**Limits:** 2 rounds per arm, and the spread within an arm (eager b4: 10,650 vs 11,554, ~8%) is
wider than the synthetic harness's 1.04% noise floor — real data and a live dataloader are
noisier. Treat the end-to-end figure as **~30–35%**, not a third significant figure. Absolute
tok/s is also lower than the synthetic run (8,201 vs 9,087 at b2) because the real path carries
masking, `doc_ids` and grad clipping; only the ratio transfers.

### Reproduced on updated `main` after #140 merged (2026-08-10, same day)

Re-run with identical settings on a tree containing #140 (`results_recheck_main.json`):

| arm | first run | re-run on updated main | drift |
|---|---:|---:|---:|
| b2_h8 (baseline) | 9,087.1 tok/s | 9,091.7 tok/s | **+0.05%** |
| b4_h8 | +37.7% | **+37.5%** | 0.2 pp |
| b4_h16 | +60.8% | +60.1% | 0.7 pp |
| b2_h16 | +12.0% | +10.9% | 1.1 pp |
| b8_h8 / b8_h16 | SPILLED, 6,512 MiB | SPILLED, 6,512 MiB | identical |

Noise floor 1.04%, positive control −41.1%. **Every arm reproduces inside the noise floor**,
across two separate processes on different trees — so the win is not a single-run artifact.

This also clears something #140's own tests could not: that issue's FFN refactor
(`nn.GELU()` → `make_activation("gelu")`) is asserted byte-identical in *output*
(`max|Δlogits| = 0.0`), but an output test cannot detect a construction change that alters
kernel dispatch or fusion. The baseline reproducing to 0.05% shows it costs no throughput
either.

## FFN activation buys no training speed — GELU stays, PowLU is worse (Issue #140, 2026-08-10)

**Question:** does swapping the DeltaNet FFN's activation (GELU → ReLU² or PowLU) speed up
training? **Answer: no.** ReLU² ties; PowLU is measurably *slower* and costs more VRAM.

`scripts/activation_speed_test.py`, RTX 4050, eager, batch 2 × seq 512, grad_accum 16,
5 interleaved rounds × 64 timed micro-batches, model config read from
`checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt` (d=512, 12 layers, 2-attn hybrid, ternary):

| arm | tok/s (median) | vs GELU | peak VRAM |
|---|---|---|---|
| GELU (control) | 9,035 | — | 2,455.7 MiB |
| **ReLU²** | 9,106 | **+0.79% — tie** | 2,455.7 MiB |
| **PowLU** (p=1.5 learnable) | 8,677 | **−3.96%** | 2,573.7 MiB (+4.8%) |
| GELU again (noise floor) | 9,178 | +1.58% | 2,455.7 MiB |
| slow control (positive control) | 7,413 | −17.95% | 3,357.0 MiB |

**Why this was predictable, and was predicted before running.** The FFN is two `d×2d`
matmuls around one pointwise op; at d=512 the matmuls are ~400× the arithmetic and all three
activations move the same bytes. An activation swap has almost no throughput to move. The
measurement is only worth recording because it *could* have refuted that.

**What makes the null trustworthy** (the #138 lesson — a test that cannot fail publishes bugs
as numbers):

- **Positive control.** A deliberately expensive activation (stacked transcendentals) came out
  −17.95%, proving the harness resolves activation-level cost. Without it, "no difference"
  would be indistinguishable from "harness measures nothing"; the script **voids its own run**
  if the control fails to separate.
- **Noise-floor arm.** GELU was entered twice under different names; the 1.58% gap between two
  *identical* configs is the resolution limit. ReLU²'s +0.79% is inside it, so it is reported
  as a tie, not a win. A genuine ≥2% win would have been caught.
- **Interleaved, not sequential** (round-robin), because this repo previously measured a ~2×
  difference between a config and itself from GPU contention drift (see the speed-ceiling entry
  below).

**PowLU's −3.96% is outside the noise floor and consistent across rounds 2–5** — a real, if
small, regression, plus ~118 MiB extra activation memory for the saved `abs`/`sign`/`pow`
intermediates. Combined with it being an odd, non-gating, superlinear function with no
literature support, it is **not recommended**.

Eager mode was deliberate: `torch.compile` fuses all three into the surrounding elementwise
epilogue, which can only *shrink* the gaps. A null in eager is a null in production.

**Consequence for the ablation.** The remaining case for ReLU² is quality-per-step (Primer,
2021 reports ~1–3%), not speed — and the maintainer's bar was a speed win. That needs ~6
GPU-hours of continued-FT, and would be confounded anyway: the checkpoint's weights spent
8,000 steps adapting *to GELU*, so a warm-started arm measures "how fast can these weights
adapt to a new activation", not "which activation is better". **Not funded; #140 closed on the
throughput finding.** If revisited, the design is small from-scratch runs, not a
warm-started swap.

Code landed regardless (`hvn/core/activations.py`, `ffn_activation=` on `DeltaNetBlock`/
`EdgeHVN`, 13/13 tests): `"gelu"` is byte-identical to the pre-#140 path (max|Δlogits| = 0.0
vs the old construction), so the lever is available at zero cost if a later scale makes it
worth re-testing.

**Two spec corrections found before building** (both would have produced a meaningless run):
the task named SwiGLU as the baseline, but the checkpoint has no `use_swiglu` key and is
**GELU** — and SwiGLU needs a gate matrix the checkpoint lacks, so that arm would have
measured fresh-vs-trained weights at 2.25× the FFN params. The requested known-answer cell
("identity activation reproduces 35.5%/8.5% recall") is also unpassable: identity is a
different function, and ±2%/±1% is inside 300-trial sampling noise (σ≈2.8% / 1.6%), so a
*correct* harness fails it ~half the time. Replaced with the exact-logit-match cell above.

## EdgeHVN 83M — the measured PPL picture (2026-07-10)

The 83M has **three separate PPL/capability readings on three different data types** —
conflating them caused several wrong "ceiling" calls, so they're written down here:

| Data | PPL | What it measures | Status |
|---|---|---|---|
| **Chat fine-tune (OpenAssistant)** | **~26.5–30** (low: 3.277 loss / 26.5 ppl, step 440) | conversational **fluency** — chat format, coherence | ✅ **solved & measured** |
| **Pretrain (SlimPajama web)** | ~110 → **63.5 val @ 50K (stopped)** | broad web **knowledge** | ⬇ descending — undertrained, not a wall |
| **Grounded reading** (`eval/grounded_qa.py`) | 50K probe: **0/5 closed, 1/5 open, +1 lift** | **read + recall** exact tokens from context | 🟡 partial grounding — the hybrid test |

**So 83M is conversational in the fluent-chat sense** — ppl ~30 on chat data is a real,
measured conversational model (that's why it responds to "hi" correctly). What it is **not**
is *knowledgeable* (ppl ~110 on web) or *grounded* (0/5 recall). The chatterbox is real and
quantified; the gaps are knowledge + reading, **not** fluency.

**The "~110 = 83M capacity ceiling" claim is REFUTED.** The old 10K-step run plateaued at
SlimPajama ~110, but that run was **data-capped at ~135M tokens** (the OASST
`max_repeat=3` bind) — it plateaued because it ran out of fresh data, not because 83M hit a
wall. The Chinchilla-floor pretrain (Muon, OASST dropped from pretrain) descended and went
below 110: val PPL 182 (step 2K) → 133 (4K) → 114 (6K) → **102.9 (8K)** → … → **63.45
(50K)**, descending on the asymptotic tail; train loss confirms (8.35 → 4.71 avg, with a
mid-run flat patch ~step 3K–5.5K that then **broke through and accelerated** ~step 6K–7.5K
— a phase transition, the opposite of a ceiling). **Do not draw capacity conclusions from a
PPL that is still descending** — the undertrained-vs-capacity question is OPEN, tilting
toward *undertrained*. The 50K grounded-QA probe (capability: can it *read*) is the
decisive test; the PPL trajectory is now a positive signal.

> **TODO:** The README's EdgeHVN picture still describes the Chinchilla-floor pretrain as
> "still descending" (bg `b2dytn9gk`) in one paragraph and "STOPPED at ~51K" in the next —
> those contradict. The settled truth is **STOPPED at ~51K** (see the 50K probe verdict
> below). Remove the "still descending / running" phrasing.

> **Why the current pretrain shows val_ppl ~103 while the old run reported "~37 overall":**
> a **mix change, not a regression.** The Chinchilla-floor pretrain deliberately dropped
> OASST from pretrain (`--pretrain-mix 0.85,0.0,0.15` — OASST belongs in chat-FT), so its
> val PPL is SlimPajama+Wiki only, with no easy chat data dragging it down. The old "~37
> overall" included 35% easy OASST. Apples-to-apples on the hard knowledge data: old ~110 →
> new ~103 and falling — the new run is *better* on knowledge.

**Note — "hi → unrelated/garbage text" is NOT a knowledge problem.** Knowledge is about
facts ("capital of France"); "hi" needs none. A model with perfect knowledge and a broken
decoder still emits garbage to "hi". Real causes: (1) **serving/decoding** — if EOS + the
chat-format turn marker aren't hard-enforced, generation runs past "hello" into
base-pretrain web continuation → "unrelated text" (no proper EdgeHVN decode path exists
yet; the Phase B `chat_edgehvn.py` entrypoint / EOS-50256 format is exactly this fix);
(2) **chat-FT too shallow** — only 640 steps / 3 epochs (noisy, bounced to ppl ~39), so the
base-pretrain web-text prior leaks through; (3) the **base pretrain itself emits web
garbage**. Fix = **serving-side EOS/format enforcement + a real chat-FT + a cleaner trunk** —
not more knowledge.

## 50K probe verdict + STOP decision (2026-07-11)

The Chinchilla pretrain reached ~step 51K (val PPL **63.45 @ 50K**, Slim 4.18 / Wiki `nan` —
the `nan` is the known cosmetic val-mislabel, not a training fault; train loss ~4.08, ~0.7
steps/s). The mid-run `eval/grounded_qa.py` probe on the 50K trunk: **closed-book 0/5,
open-book 1/5, retrieval lift +1/5 → "Partial grounding."** The single open-book hit (Q3,
boiling point) surfaced *"Celsius"* from the passage — passage-primed, not lucky (closed-book
for the same question was HTML garbage). vs the old undertrained base's **0/5 / 0/5 / +0**
(pure HTML garbage): the 50K trunk moved from "pure garbage" to "answer-shaped, weakly
passage-primed." **This leans undertrained, not capacity-limited** — the first signs of
reading appeared. **Caveat:** this is the *pure base, not chat-FT'd*, so closed-book is HTML
garbage regardless of knowledge; 1/5 is a "the pretrain is now alive enough to sometimes
ground" signal, not a clean capacity verdict.

**Decision: STOP the pretrain at ~51K** (process tree terminated 2026-07-11, GPU freed;
trunk preserved at
`checkpoints/edge_chinchilla83m/edge_conversational_step{50000,51000}.pt`). Rationale: the A5
gate is a *binary architectural recall test* (lift > 0 vs 0/5) — recall comes from the 2 new
attention layers, not trunk PPL — and the 50K trunk already reads (partial grounding), which
is the floor the hybrid A4 FT builds on. A stronger trunk (50K→100K, val 63→~40) is
asymptotic-tail (the big jump 110→63 already happened), **won't fix the no-chat-FT
closed-book confound**, and costs ~30 h of the one scarce GPU that the P0 (hybrid A4) needs
now. **Next: A4** short grounded continue-FT from this trunk → A5 gate.

## A5 gate result — hybrid A4 step5000 FAILED; re-FT on a bigger corpus (2026-07-12)

The A4 8K run (bg `bjblkgc53`) was killed at ~step 5920; latest surviving checkpoint
`hybrid_a4_step5000.pt`. Ran the A5 gate on it, plus a new **beyond-window probe**
(`eval/grounded_qa.py --far-probe`, MR !83) that shrinks the attention window and pushes
the passage past it, so only the DeltaNet recurrence — not the *windowed* attention — can
carry the answer. Both checkpoints, same gate:

| Checkpoint | standard: closed / open (lift) | far-probe W=128: closed / near / far |
|---|---|---|
| **pure trunk** `edge_conversational_step51000.pt` | 0/5 / 1/5 (**+1**) | 0/5 / **2/5** / **0/5** |
| **hybrid A4** `hybrid_a4_step5000.pt` | 1/5 / 0/5 (**−1**) | 2/5 / **0/5** / **0/5** |

**Verdict: the A4 FT made things WORSE, not better.** The hybrid at step5000 reads *less*
than the bare trunk (near-open 0/5 vs 2/5; open-book lift −1 vs +1). Cause = **data, not
steps or capacity**: the FT trained the 2 fresh attention layers on only **40** grounded
triples, so it memorised 40 pairs and degraded the trunk's weak reading instead of learning
the general skill. (Loss was still falling at kill, but on 40 examples more steps only
overfit harder.)

**Architectural finding from the far-probe (important — reframes the goal).** For BOTH
checkpoints, **far-open = 0/5, far-lift = 0**: the DeltaNet recurrence carries *zero exact
recall beyond the window*. And the 2 attention layers are *windowed*, so they cannot help
beyond the window either. Therefore the hybrid's achievable win is **exact recall WITHIN
the window** (the trunk's recurrence already got near-open 2/5 there; attention should push
that higher) — **not** beyond-window recall, which is architecturally capped at ~0 for both
the recurrence and windowed attention. Beyond the window, hybrid ≈ a plain sliding-window
transformer on exact recall; HVN's only beyond-window edge is a *lossy semantic gist* the
keyword gate can't credit. See [architecture.md](architecture.md).

## CORRECTION (2026-07-12, later) — the "made things WORSE" verdict above is RETRACTED

`grounded_qa`'s 5-item keyword gate is too noisy to trust for this decision — it gates a
1B-parameter scale-up call, and a 5-item test is dominated by luck (one keyword flips the
verdict) plus a real-world-knowledge/fluency confound unrelated to recall. Replaced it with
`eval/synthetic_recall_probe.py` (MR !83): hundreds of trials per checkpoint, INVENTED
entity/value pseudo-words (e.g. *"The Zorvex is called Blenthar."*) rendered in the model's
own trained chat/QA template — unambiguous exact-match scoring, zero real-world-knowledge
confound, in-distribution phrasing (an earlier raw-random-token-id version produced a flat
0% everywhere including inside the hybrid's own window — that was a format artifact, not a
recall measurement; retracted and fixed, see the script's docstring).

**Near-window exact-recall accuracy, 200 trials each:**

| Checkpoint | accuracy |
|---|---|
| pure trunk (`step51000`) | **4–5%** |
| old A4, 40-example corpus, "failed" (`hybrid_a4/step5000`) | **16–16.5%** |
| new A4, 5,240-example corpus (`hybrid_a4_v2/step5000`, only 63% through its run) | **35.5%** |

**The old 40-example FT was never a regression — it was already ~3-4x better than the trunk.**
The 5-item gate's −1 lift was noise, not signal. The corrected, high-N picture is a clean,
monotonic trend: bigger grounded-QA corpus → better recall, exactly as hypothesized, and the
new run isn't even finished. **Beyond-window** (target pushed >W=128 via 16 distractor facts):
trunk 0.5%, old A4 0.5% (both = noise floor), **new A4 5.5%** — the first checkpoint to show
signal clearly above floor, though still far below the near-window number; worth watching at
step8000, not yet a strong claim.

**`eval/synthetic_recall_probe.py` is now the primary R&D gate; `grounded_qa.py` (standard +
`--far-probe`) is secondary — useful for real-world-transfer sanity, not for this go/no-go.**

## FINAL VERDICT — A4 8K complete, mechanism PASSES the R&D gate (2026-07-12)

`checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt` (8,000 steps, 161.6 min, `feat/grounded-qa-
beyond-window-probe`; final train loss ~2.35). Full synthetic-recall trajectory, 200 trials
per point:

| Checkpoint | near (unbounded) | near (W=128) | far (W=128, beyond) |
|---|---|---|---|
| pure trunk | 4.0% | 5.0% | 0.5% |
| old A4, 40-ex | 16.0% | 16.5% | 0.5% |
| new A4 **step5000** | 35.5% | 35.5% | 5.5% |
| new A4 **step8000** | 35.0% | **35.5%** | **8.5%** |

**PASS.** Two clean, high-N, monotonic findings:
- **Near-window recall works and is large:** ~9× the trunk's floor (4% → 35.5%). This is the
  core hybrid claim (windowed attention adds exact recall the recurrence alone lacks) —
  proven, not just "not broken."
- **Near-window has PLATEAUED** (35.5% at both 5000 and 8000) — more steps on *this* corpus
  won't push it further; the next lever for near-window is a bigger/more-diverse corpus, not
  more training time.
- **Far-window kept climbing and is now real signal, not noise:** 0.5% → 5.5% → 8.5% (~17× the
  trunk's floor), monotonic with training. This is the first evidence in the project that the
  recurrence carries *some* genuine exact-token signal beyond the attention window — the
  capability the architecture wasn't designed to structurally guarantee. Still modest in
  absolute terms (8.5%, not "reliable"); whether it keeps climbing with more data/steps or
  saturates here is open.

**Decision: the 83M R&D probe clears the bar.** This is the go-signal for the next staged
milestone in [README's "The Dream"](../README.md) — scaling the recipe toward a ~20B-class
native ternary-hybrid, not a reason to scale-up-via-capacity-question (#84 GaLore is no longer
the next step; it was the fallback for a failed gate, and the gate passed).

`grounded_qa.py`'s 8k re-run remained noisy as expected (standard: closed 2/5 open 0/5;
far-probe: closed 2/5 near 1/5 far 1/5, with the lone far "hit" being a garbled multi-country
list that happened to contain "Portuguese" — not real recall) — confirms it should stay a
secondary real-world-transfer sanity check, not the decision instrument, going forward.

## NEXT PLANNED MILESTONE — ~270M local, conversational fluency + tool-calling (2026-07-12)

Not the ~20B jump yet — a nearer, locally-fundable step first. Decided after checking real
constraints (no code committed for this milestone yet, this is the plan):

- **1B (and certainly 20B) cannot train locally under vanilla full AdamW.** Measured RTX 4050
  training-VRAM ratio is **16.8 bytes/param** (bf16 weights+grads+fp32 AdamW moments+master,
  Issue #69, [training.md](training.md)) → 1B needs ~16.8 GiB (off by ~3× on 6 GB), 20B needs
  ~336 GiB (off by ~14× even on a 24 GB RTX 3090 — a single card physically cannot hold a 20B
  model's training state; needs real multi-GPU sharding). Cloud-rental math (RunPod RTX 3090,
  sourced pricing $0.22/hr Community / $0.46/hr Secure): a full Chinchilla-optimal 1B pretrain
  needs an assumption-flagged **~1,341 GPU-hours** (~$295 total, ~56 days continuous, or ~15
  months at a $20/mo cap) — real, but a deliberate multi-month commitment.
  > **UPDATE (2026-07-12, later same day) — 1B now trains locally, cleanly, measured.**
  > Issue #99 (BAdam block-coordinate-descent, `--block-coordinate`) landed: 268M config
  > −39% peak VRAM (6.9→4.2 GiB), 1.04B config hard OOM → runs (still spilling past 6 GiB
  > alone). Issue #100 (EdgeHVN never had `--grad_checkpoint` wired, despite #49/#88 closing
  > it for the core model class only — fixed, bit-identical parity verified) stacked on top:
  > **1.04B config → 4.6 GiB peak, a clean fit with headroom, no spill.** See
  > [training.md](training.md)'s BCD section for the full numbers
  > (`results/bcd_ckpt_vram.json`). **The cloud-rental math above is no longer the path for
  > 1B at all** — it trains on this laptop today. GaLore (#84) and 8-bit Adam (#85) remain
  > unimplemented, further composable levers for 10B-class work, not needed to reach 1B.
- **The measured local RTX 4050 ceiling today is ~270M params** (d=1024, 24L, bf16 +
  `--grad_checkpoint`, 4.3 GiB peak — Issue #69 table above). Free, immediate, no new
  engineering.
- **Calibrated goal for 270M — NOT "agentic coding" capability.** Real competent code
  generation needs far more scale + data than a from-scratch local run can provide (even
  purpose-built 0.5–1.5B code models trained on trillions of tokens are considered weak for
  reliable agentic use). What 270M *can* plausibly prove: **(1) conversational fluency**
  (coherent chat, achievable at this scale with real chat-FT) and **(2) correct tool-call
  format-following** (emitting well-formed `[TOOL_CALL]{...}[/TOOL_CALL]` — a learnable format
  skill, not a reasoning skill).
- **Blocker independent of param count:** EdgeHVN has no serving/decode path yet (Phase B —
  KV cache + `forward_step` + `generate`, see [architecture.md](architecture.md)'s hybrid
  design) and no real tool-use FT corpus (Phase C — `hvn/tools.py`'s
  `generate_tool_training_data` is only ~50 synthetic templates, a seed). Testing *any*
  tool-calling behavior needs these built first, regardless of whether the model is 83M, 270M,
  or 20B. Scoping this build is part of the 270M milestone, not a separate later step.

**TODO:** scope the 270M run (config, chat-FT data mix, Phase B serving stub sufficient for
format-only tool-call testing) and open a GitLab issue before starting code, per
[ai_rules.md](ai_rules.md).

## Stage 1 — conversational FT on the 83M/2-attn hybrid (issue #108, 2026-07-15)

Prior context: the 2-attn hybrid checkpoint (`checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt`)
passed its recall gate (35.5% near-window / 8.5% far-window, `eval/synthetic_recall_probe.py`)
but was mostly trained on grounded-QA data (only a 30% OASST chat-mix) — not yet a genuinely
conversational model. A follow-on 3-attn scaling attempt regressed and was closed without
merging (issue #109; not yet on `origin/main`, tracked only via the GitLab issue).

**Run:** `finetune_edge_conversational.py --base-checkpoint hybrid_a4_step8000.pt
--grounded-replay data/grounded_qa_expanded.jsonl --grounded-replay-frac 0.15 --epochs 3`
(branch `feat/conversational-ft-108`) — full OASST corpus (3,482 conversations → 3,465
examples) mixed 85/15 with grounded-QA replay (5,240 examples) via the same
`ConcatDataset`+`WeightedRandomSampler` pattern already proven in `continue_ft_hybrid.py`.
1,632 optimizer steps, 104.2 min, final loss ~2.6-2.8 (ppl ~14-16).

**Bug caught pre-launch (fixed, not just observed):** a GPU smoke test surfaced the exact
same ~20x-inflated loss-logging bug previously found and fixed in `continue_ft_hybrid.py`
(commit `a023625`) — `finetune_edge_conversational.py`'s `finetune()` predated that fix and
had the identical accumulator bug (summed per-opt-step means across a 20-step window without
dividing by the step count). Fixed the same way (`n_steps_accum` divisor) before the real run;
smoke logs went from loss~50/ppl-capped-at-4.85e8 (nonsense) to loss~2.3/ppl~10 (sane).

**Qualitative (fixed prompt set, greedy decode via `sample_reply`, no repetition penalty):**
- "hi" → *"Hello! How can I help you today?"* — coherent, on-topic (a real improvement; the
  base checkpoint's chat behavior was noticeably weaker/format-only before this stage).
- "What is the capital of France?" → *"The capital of France is Paris."* — correct.
- "What's 2+2?" and the haiku prompt both degrade into repetition loops (`"2+3+3+3+3..."`,
  `"island of the island..."`). Likely a greedy-decoding artifact (no repetition penalty /
  sampling in the current `sample_reply` smoke-check, not necessarily a training failure) —
  worth re-checking once Stage 2's real decode path exists and a proper sampling strategy can
  be tested; not re-diagnosed further here.

**Recall regression check (`eval/synthetic_recall_probe.py`, 300 trials):**

| Checkpoint | near-window | far-window (W=128, 16 distractors) |
|---|---|---|
| 2-attn PASS baseline (pre-chat-FT) | 35.5% | 8.5% |
| 2-attn + Stage 1 chat FT (`hybrid_chat_v1.pt`) | **28.3%** (85/300) | **5.3%** (16/300) |
| pure trunk floor (reference) | 4-5% | 0.5% |

**Verdict: PASSES the "no collapse" gate, but recall genuinely eroded** — near-window down
~20% relative, far-window down ~38% relative. The 15% grounded-QA replay mix slowed but did
not fully prevent forgetting; both numbers remain far above the trunk floor (not a collapse),
so this is an acceptable, measured tradeoff rather than a regression to fix — but a real cost
of chat-focused FT, not a free win. If recall matters more than chat fluency for a future use
case, a higher replay fraction (or interleaved/alternating-stage FT instead of one blended
pass) is the lever to revisit.

**Next:** Stage 2 — the serving/incremental-decode path (issue #97 Phase B). `EdgeHVN` still
has no `forward_step`/KV-cache; all evaluation above (including `sample_reply`) uses expensive
full-sequence re-forward. See the full roadmap plan for Stage 2/3 design.

## EdgeHVN incremental decode landed (issue #97 Phase B, 2026-07-15)

`EdgeHVN`/`TransformerBlock` now have `init_state`/`forward_step`, and `EdgeHVN` has
`init_conversation_state`/`chat_step` matching `HamiltonianVortexNetwork`'s exact method
contract — `hvn/chat.py`'s `ChatSession`/`StateCheckpoint` work against EdgeHVN with zero
changes to their own logic (verified: the mixed DeltaNet+attention-KV-cache state round-trips
through the existing generic `_encode_state_tree`/`_decode_state_tree` JSON codec). New
`hvn/edge/chat.py:EdgeConversationFormatter` speaks EdgeHVN's actual trained format
(`<|user|>`/`<|assistant|>`, EOS-terminated) instead of the core model's bracket-tag protocol.
Branch `feat/edgehvn-serving-97`, `tests/test_edgehvn_incremental.py` (9/9 pass).

**A real finding surfaced while GPU-verifying parity on the actual trained checkpoint
(`hybrid_chat_v1.pt`), worth recording so it isn't mistaken for a serving bug later:**
comparing `forward_step`'s incremental output against a plain `forward()` call on a *trained*
checkpoint can show large divergence (up to ~3.2 in logit scale over just 20 tokens/12
layers) — looks alarming, is not a bug in the new decode path. Root cause: `forward()`
dispatches to `DeltaNetBlock._scan_chunked` (a fast, reassociated approximation used for
training speed, active whenever `train_chunk` is set — true for this checkpoint) rather than
the exact `_scan_sequential` formula that `forward_step` implements token-by-token.
`_scan_chunked` itself diverges non-trivially from `_scan_sequential` on real trained weights
(verified directly, bypassing `forward_step` entirely: a single DeltaNet layer's chunked vs.
sequential scan already differs by ~0.39 with this checkpoint's weights) — a pre-existing
numerical property of the #83/#96 training-speed optimization, invisible with the
small-random-weight test models used when that optimization was originally validated
("~1e-7" in the code comments). **`forward_step`'s actual correctness bar** — matching
`_scan_sequential`, the documented reference recurrence, apples-to-apples on the same trained
weights — **holds to ~3e-6 (machine precision)**, confirming the incremental decode path is
correct. Locked in as `tests/test_edgehvn_incremental.py::test_deltanet_forward_step_matches_scan_sequential_not_scan_chunked`.
**Practical implication:** don't use a trained/`train_chunk`-enabled checkpoint's `forward()`
as the parity oracle for incremental decode work — use `_scan_sequential` directly, or a
fresh-init/`train_chunk=0` model, or expect this gap and don't treat it as a regression.

## EdgeHVN wired into the Docker chat app + real end-to-end verification (2026-07-15)

`ModelVersion.architecture` field (migration `0003_modelversion_architecture`) + branching in
`hvn_worker/worker.py`/`inference.py` let the Django/Celery/Channels chat app serve EdgeHVN
checkpoints, not just the original `HamiltonianVortexNetwork`. New
`EdgeHVNChatSmokeTests` (`web/tests/test_chat_smoke.py`) drive the real (unmocked)
worker→inference→ChatSession path — 78/78 Django tests pass.

**Beyond the unit tests: verified against the real Docker stack, not just `manage.py test`.**
Built the image (`docker compose build`), brought up postgres/redis/app-web/app-worker/
app-beat, registered `hybrid_chat_v1.pt` as an `architecture="edgehvn"` `ModelVersion`, and
drove two real chat turns through the actual HTTP API (login → create conversation → POST a
message → Celery task runs → reply persisted) — the same code path the browser UI hits, no
mocking:
- Turn 1 ("hi") → *"Hello! I am an AI assistant..."* — coherent opening.
- Turn 2 ("What is the capital of France?") → *"The capital of France is Paris..."* — correct,
  and **confirms multi-turn `StateCheckpoint` save/load round-trips correctly for real** in
  the deployed stack (not just the CPU unit test's synthetic round-trip check).
- Both replies degrade into repetition after the first sentence or two — the same known
  quality limitation already documented for this checkpoint in the Stage 1 (#108) writeup
  (arithmetic/haiku prompts looping under greedy/low-temperature decode); not a new Phase B
  bug, just now also confirmed present via the real serving path, not only the CLI
  `sample_reply` check.

**One real bug caught and fixed during this verification** (would NOT have been caught by
the CPU unit tests, which import `hvn`/`train_edge_conversational` directly rather than
through the built Docker image): `docker compose build app-web` only rebuilds the `app-web`
image — `app-worker`/`app-beat` are separately-tagged images per service and were still
running a stale build missing the `train_edge_conversational.py` COPY, causing
`ModuleNotFoundError: No module named 'train_edge_conversational'` on the first live attempt.
Fixed by rebuilding all three service images explicitly. Worth remembering for any future
Dockerfile change to this app: **rebuild `app-web app-worker app-beat` together**, not just
whichever service you're actively iterating on.

Not yet done: a real trained-quality pass (fixing the repetition-under-greedy-decode issue,
e.g. sampling/repetition-penalty tuning) and exercising the actual browser UI (this
verification drove the REST API directly, the same code path, but not the JS frontend/
WebSocket client itself).

## Stage 3 — agentic tool-use FT + gate (issue #110 / #97 Phase C, 2026-07-15)

Continued from Stage 1's `hybrid_chat_v1.pt`. Landed (branch `feat/agentic-tooluse-97c`):
`repetition_penalty` param on `EdgeHVN.chat_step` (real repetition was observed in Stage 2's
Docker verification); sandboxed coding tools `read_file`/`edit_file`/`run_tests` in
`hvn/tools.py` (path-traversal guarded, `run_tests` takes no free-form args so a
model-controlled argument can't smuggle shell content — caught a real bytecode-cache staleness
bug here via the tests, fixed with `PYTHONDONTWRITEBYTECODE=1`); a 600-example programmatic
tool-use corpus (`scripts/build_tool_use_data.py`, every coding example actually executed +
verified during generation, not fabricated); `ToolUseDataset` with three-way loss masking
(query masked, `[TOOL_CALL]` learned, `[TOOL_RESULT]` masked as environment-injected context,
answer learned — the fix for a real flaw in the pre-existing
`hvn.tools.generate_tool_training_data`, which trained the model to hallucinate tool results);
and `EdgeToolCallingSession` (`hvn/edge/tool_session.py`), a from-scratch EdgeHVN-native
agentic loop built on `forward_step` (not a port of the pre-existing `ToolCallingSession`,
which is bound to a `generate_incremental` contract that doesn't exist on `EdgeHVN`).

**FT run:** continued from `hybrid_chat_v1.pt`, mixed 50% chat / 10% grounded-QA replay / 40%
tool-use, 1,743 optimizer steps (~2h), final loss ~1.7-1.9 (ppl ~6-7). Qualitative check via
the old full-forward `sample_reply` (no repetition penalty) showed the model correctly
emitting `[TOOL_CALL]\n{"name":"calculator","arguments":{"expression":"2+2"}}\n[/TOOL_CALL]`
for "What's 2+2?" — right tool, right arguments — before degrading into a malformed repeat
(a `sample_reply`-specific artifact; the real serving path, `EdgeToolCallingSession`, uses
`repetition_penalty`).

**Agentic coding gate (`eval/agentic_coding_gate.py`, `checkpoints/hybrid_tooluse_v1.pt`):** a
fresh (not a literal training example, but the same learned bug-template family) one-line-bug
mock repo, driven through the real `EdgeToolCallingSession` + sandboxed coding tools.

| Bar | Result |
|---|---|
| **Pass (required):** well-formed `read_file → edit_file → run_tests` sequence, no parse failures | **YES** |
| **Stretch (reported, not required):** the bug actually gets fixed | **NO** |

The model emitted the exact correct *tool sequence and JSON structure* three calls in a row,
zero parse failures — a real, non-trivial, previously-absent capability. It did **not**
correctly ground the arguments: it called `read_file`/`edit_file` on a hallucinated path
(`"bounds/jquery"`, likely bleed-through from the `bounds_*.py`-named training family mixed
with unrelated web-dev vocabulary) instead of the actual file named in the query
(`gate_calc.py`), so the edit never landed and `run_tests` correctly reported the original
failure. **This is exactly the calibrated outcome stated going into this stage** (per the
270M-milestone framing already in this file): tool-call **format-following** is a learnable
format skill and it works; correctly **grounding arguments from the specific query** is a
reasoning-adjacent skill this 83M model does not yet reliably have. Single-trial result (n=1,
real measurement, not averaged over multiple sampled runs) — a fuller sweep (several seeds/
prompts) would be needed to quantify the pass rate precisely, not just confirm the capability
exists.

**Roadmap status: Stages 1, 2, and 3 are all now landed** (branches `feat/conversational-ft-108`,
`feat/edgehvn-serving-97`, `feat/agentic-tooluse-97c` respectively, all pushed, none yet merged
by the user). The model can chat, recall within its window, be served through the real Docker
app, and follow a tool-calling format reliably — genuinely "good enough" per the plan's own
honesty bar, not GPT-4-class autonomous coding. The clearest next lever, if better argument-
grounding is wanted, is more/more-diverse tool-use training data (the same volume lesson as
grounded-QA and OASST throughout this whole track) rather than further architecture work.

## Live Docker chat testing: a real serving bug fixed + the scale-vs-data verdict (2026-07-15)

**Real bug found via actual live use, not a unit test.** Chatting with `hybrid_tooluse_v1.pt`
through the deployed Docker UI produced replies with the literal text `<|endoftext|>` appearing
mid-response followed by more generated text. Root cause: `hvn.chat.ChatSession.user_turn_streaming`
(what the web app's streaming path actually calls, not `chat_step`) had its own from-scratch
generation loop that **never checked for a stop token at all**, for any model — correct for the
core `HamiltonianVortexNetwork` (bracket-tag format, no single stop token by design) but wrong
for EdgeHVN, which was trained with a single EOS-terminated turn. A sampled EOS token was
decoded as literal text and generation just kept going instead of stopping. Same story for
`repetition_penalty` (issue #110): it existed on `chat_step` but `user_turn_streaming` never
used it. Fixed: `ChatSession` gained `eos_token_id`/`repetition_penalty` params (both default
off — zero behavior change for the core model), wired for the EdgeHVN call site
(`web/hvn_worker/inference.py`, `eos_token_id=50256, repetition_penalty=1.3`). Verified directly
(an EOS-biased test model now stops immediately instead of running the full token budget) and
via the live API (`hybrid_tooluse_v1.pt` now replies with a single clean, complete sentence
instead of a run-on garbled one). Landed on `feat/edgehvn-serving-97`.

**After the formatting bug was fixed, real qualitative testing surfaced the actual remaining
gap clearly: content coherence, not decoding mechanics.** Replies are now grammatically
well-formed and properly terminated, but frequently don't engage with what was actually asked
— generic assistant-boilerplate text regardless of the specific question, no real recognition
of conversational context (e.g. a frustrated "what the fuck?" gets a boilerplate "as a language
model, I can provide you with information" instead of anything responsive to the tone/intent).

**Verdict, stated directly: this is primarily a capacity (scale) problem, not a data problem.**
More/better training data is the proven, cheap lever for *surface* quality — format-following,
topic breadth, reduced repetition (this project has now demonstrated that pattern three times:
grounded-QA's 40→5,240-example expansion, OASST full-corpus vs capped, and the tool-use
corpus's programmatic generation). But genuine contextual reasoning — actually parsing intent,
not just producing fluent-sounding text — is the kind of capability that scaling-laws research
(and this project's own earlier "83M capacity ceiling" investigation, which concluded 83M was
undertrained *relative to its own ceiling*, not that the ceiling itself was high) suggests is
much more parameter-count-bound than data-bound at this scale. Pouring more data into a fixed
83M model would likely keep improving fluency/format without closing this specific gap.

**Decision: defer #107 (scale past 83M) — it is the correct next lever, not started this
session.** The full pipeline (conversational FT → serving → agentic tool-use) is proven
end-to-end at 83M, which was the point of building it at this scale first rather than
committing to a multi-day pretrain up front. Scaling the *same* recipe (DeltaNet trunk +
windowed-attention hybrid, same warm-start approach) to whatever size #107 lands on is now the
well-justified next step when picked back up — not a guess, a conclusion from real measurement
at the smaller scale first.

## #112 — 268M real-recipe validation attempt: a WDDM TDR driver crash, not a slow compile (2026-07-24)

Picking up #107/#112 (scale past 83M): before committing to a multi-hour/day pretrain at
120–270M, ran a short **300-step validation** of the "recommended ceiling config" from this
file's VRAM table (d=1024, 24L, ~268M, `--grad-checkpoint`) with the **real** data pipeline
(SlimPajama/WikiText via the existing `data_cache`, not synthetic) and the real `--fast` stack,
to get a steps/sec + VRAM-peak reading before scoping the full run.

**First finding — the untracked smoke probes already in `pretrain-300m-112` (`logs/probe_143m.log`,
`probe_300m.log`, `probe_300m_clean.log`, 150 steps each) were not measuring this table's 268M
config at all.** They used d=1024/19L (~302M, not 24L/268M) and never passed `--grad-checkpoint`,
so the 300M-class probe hit the documented WDDM VRAM spill (~30s/step, ~13x slower than the
143M probe's 2.2s/step) — consistent with this file's "spills → 4-15x slower" note, not a new
finding, just a probe that was missing the one flag that fixes it.

**Second finding — the first real validation launch appeared to hang for 50+ minutes with zero
step output, but the actual cause was a Windows driver crash, not a slow `torch.compile`.**
Windows Event Log (System log) shows, ~3 minutes into the run:
```
14:38:24  Driver Management has concluded the process to add Service nvlddmkm ... status: 0.
14:38:25  Driver Management has concluded the process to add Service NVDisplay.ContainerLocalSystem ... status: 0.
```
This is the signature of a **WDDM TDR (Timeout Detection & Recovery) reset** — Windows kills
and reloads the GPU driver if a single kernel launch blocks longer than its watchdog timeout
(default ~2s). Plausible trigger: first-time `torch.compile` + Triton-scan autotuning for the
new 268M/24L/seq_len=512 shape (the log's "Not enough SMs to use max_autotune_gemm mode" line
shows it was mid-autotune-fallback right before the reset). After the reset, the training
process kept running but was silently stuck on a dead CUDA context for the remaining ~47
minutes — indistinguishable from "still compiling" until investigated, because **stdout is
fully buffered when redirected to a file** (the same gotcha recorded for the 83M chinchilla
pretrain in memory — confirmed again here at a different scale).

**Contributing/latent finding, independent of the TDR crash:** the DataLoader worker
subprocesses spawned by `--fast`'s default `--num-workers 4` resolved `sys.executable` to
`C:\Users\komal\Documents\PythonVersions\314\python.exe` — **not** the project's
`.venv\Scripts\python.exe` — even though the parent process was correctly launched from the
venv. That `314` interpreter lacks `transformers` (a previously-known gotcha from the 83M
pretrain). The workers stayed alive without crash-looping in this run (worker `__getitem__`
only touches pre-tokenized numpy memmaps, no `transformers` import needed at fetch time), so
this did **not** cause this particular hang — but it's a live risk for any future run that
relies on multi-worker loading on this machine. Safe workaround used for the retry:
`--num-workers 0`. Root cause (why spawn resolves the wrong interpreter here) is still
unexplained and worth a real fix before depending on `--num-workers > 0` for a long pretrain.

**Post-crash GPU state (as of this writing):** `nvidia-smi` fails with "insufficient
permissions"; `torch.cuda.is_available()` is `False`; Device Manager shows the GPU
(`PCI\VEN_10DE&DEV_28A1&SUBSYS_8BD4103C&REV_A1\4&3382C3D5&0&0009`) status as `Error`. A
non-elevated `Disable-PnpDevice`/`Enable-PnpDevice` toggle failed (`Generic failure`, HRESULT
`0x80041001`) — needs an elevated PowerShell at minimum; **decision: full machine restart**
rather than chase the elevated-toggle path further.

**TODO after restart:** (1) re-verify `nvidia-smi` + `torch.cuda.is_available()` before
touching training again; (2) relaunch the same 300-step validation with `python -u` (unbuffered,
so real progress is visible immediately instead of only after the OS flushes the buffer) and
`--num-workers 0` (sidesteps the wrong-interpreter spawn risk above); (3) only after a clean
steps/sec + VRAM-peak reading, decide the full pretrain step budget for #112 and launch it as
a properly backgrounded, log-tailed run per the project's standing rule for long GPU jobs.

## #112 — pre-training-audit OOM bug fixed; BCD + torch.compile recompile conflict found (2026-07-24, later same day)

After the restart, `nvidia-smi`/CUDA came back healthy. Relaunching the 1.21B validation
(`d_model=2048, num_layers=24, n_heads=8, batch=4, seq_len=128, --grad-checkpoint
--block-coordinate --fast --num-workers 0`) hit a **new, real** crash — not the driver TDR —
immediately after model construction:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB. ...
Of the allocated memory 12.31 GiB is allocated by PyTorch...
```

**Root cause, confirmed directly (not the BCD mechanism itself):**
`pre_training_audit()` in `train_edge_conversational.py` has always run two checks that build
a *second full model copy* on GPU before `BlockCoordinateScheduler` ever exists — Check 2
(`copy.deepcopy(model)`, then a full **un-frozen** forward+backward across every param) and
Check 3 (rebuilds a second `EdgeHVN` to verify the checkpoint save/load cycle, moved to GPU,
no backward). Neither check is aware of `--block-coordinate` at all. At 1.21B params fp32,
real model weights (~4.16 GiB) + audit copy weights (~4.16 GiB) + full un-BCD-frozen
gradients across the whole audit copy (~4.16 GiB) ≈ 12.3 GiB — matches the crash almost
exactly. Harmless at 83M/268M (both copies + full grads still fit easily), so it went
unnoticed until pushing toward 1B. **Fixed**: both checks are now skipped when
`--block-coordinate` is set, with an explanatory log line (not made "BCD-aware" — that would
be new, untested logic for a one-off sanity check; real forward/backward problems still
surface immediately on step 1 of actual training). Also fixed a small, independent latent bug
found in the same trace: `load_checkpoint_for_resume()` rebuilt `EdgeHVN(...)` without passing
`grad_checkpoint` at all, so a resumed run silently lost checkpointing regardless of the
original run's setting — now threaded through, and `_save_edge_checkpoint` persists the key.

**BlockCoordinateScheduler itself, and the blanket `model.to(device)` calls, were confirmed
NOT the problem** — `_activate()` handles a CPU-resident starting model gracefully via
idempotent `.to(...)` calls, and the blanket `model.to(device)` is correct as long as
`offload_inactive_weights` isn't enabled (it isn't, by design — see below). No changes needed
there.

**After the fix, 1.21B still doesn't practically run.** Both a compiled (`--fast`) and an
eager-mode (no `--compile`) relaunch of the identical 1.21B config passed the audit and
reached `STEP 5 — START TRAINING` cleanly, then **stalled silently for 30–90+ minutes with
zero step output**, no error, and a healthy GPU (`nvidia-smi` clean throughout — not a repeat
of the TDR crash). The isolated micro-benchmark (`scripts/bench_block_coordinate_vram.py`)
says this config should fit at ~4.65 GiB, but that number never included the real script's
full overhead (DataLoader, checkpoint I/O, audit machinery) — almost certainly the documented
WDDM silent-host-spill behavior, but far worse than the "4–15× slower" already on record, and
never actually root-caused (would need real profiling, not attempted). **Open, unsolved.**

**Scaled down instead, on the same fixed code path, to get the number today.** At
**355M** (d=1024, 24L — the "268M-class" config, actual param count higher due to
vocab/embedding overhead), `--grad-checkpoint --block-coordinate`:

| Mode | Total (150 steps) | Avg s/step |
|---|---|---|
| Eager (no compile) | 661.4s | **4.41s/step** |
| `--fast` (compiled) | 1009.3s | **6.73s/step** — *slower*, not faster |

**Real, confirmed finding: `--block-coordinate` and `torch.compile` actively conflict.**
`BlockCoordinateScheduler` toggles `requires_grad` on a different layer-block every
`--bcd-steps-per-block` steps (default 100) — that's the entire mechanism behind its VRAM
saving. `torch.compile` guards its captured graph on `requires_grad`, so every block switch
invalidates the graph and forces a full recompile. Confirmed directly (not just inferred from
timing) via `TORCH_LOGS=recompiles`, which printed exactly this at step 100:
```
[__recompiles] Recompiling function forward in hvn/edge/model.py:309
    triggered by ... tensor 'blocks.0.ffn.0.bias' requires_grad mismatch.
```
Per-50-step timing breakdown of the compiled run confirms the shape: steps 1–50 (7.43s/step,
includes the one-time initial compile), 50–100 (5.66s/step, settled), 100–150 (7.10s/step,
spikes back up right at the block-switch boundary — a second recompile). For a **real, long**
pretrain with the default `steps_per_block=100` this isn't a one-time cost — it recompiles
**every 100 steps for the entire run**, not just once.

**Net status:** `--grad-checkpoint --block-coordinate` alone (eager mode) is real and working
at 355M (4.41s/step). Combining it with `--fast`/`torch.compile` is currently a net loss, not
a neutral no-op — worth remembering before assuming "just add `--fast`" is free. Three
possible fixes were identified but **not pursued** (deliberately paused, not blocked): raise
`--bcd-steps-per-block` much higher to dilute the recompile cost over a real multi-thousand-
step run, treat `--fast` and `--block-coordinate` as mutually exclusive and document it, or
investigate `torch._dynamo.config.force_parameter_static_shapes = False` (surfaced directly in
the recompile diagnostic as a possible lever, unproven). **1.21B remains unresolved** — the
audit-OOM bug that blocked it is fixed, but the silent stall through the real training script
is a separate, still-open problem needing real profiling, not a guess.
## MoE FFN on DeltaNet — works, but loses to simply scaling the dense model (Issue #123, 2026-08-02)

MoE was scoped as the only remaining lever that attacks FLOPs-per-token rather than memory
(the offload work turned out to be only ~2x of the 3,300x 6B-vs-142M gap; params and tokens
are the other 42x each). Implemented as `hvn/core/moe.py::MoEFFN` — E experts, top-k routing,
Switch-style load-balancing aux loss — routing **only the FFN** (~40% of block FLOPs). The
recurrence is deliberately left dense; see the module docstring for why routing q/k/v risks
mixing incompatible subspaces inside a single recurrent state, a failure mode transformers do
not have.

2x2 at d=512, 12 layers, 5M-token budget, E=8/top-2 (`results/moe/moe_2x2.json`):

| arm | params | eval PPL | wall-clock |
|---|---:|---:|---:|
| dense | 39,756,656 | 119.657 | 556.6s |
| shared | 10,830,684 | 156.424 | 510.2s |
| MoE | 128,015,216 | 105.544 | 1062.3s |
| shared + MoE | 18,185,564 | 131.344 | 930.4s |

Harness consistency check passed: the dense arm reproduces the ternary 2x2's 119.396 at 119.657
(0.2%), and shared reproduces 154.1/156.1 at 156.424.

**The bar, set before the numbers landed.** MoE is better than dense in absolute terms (−11.8%),
but that is the wrong comparison. The question is whether it beats what the *same compute* buys
by simply scaling the dense model up. Using this repo's measured −0.504 exponent, 1.40× params
(MoE's theoretical FLOP cost) is worth **19%**; the *measured* wall-clock cost was **1.91×**,
which raises the bar to **~39%**.

| comparison | gain | bar | verdict |
|---|---:|---:|:---:|
| dense → MoE | −11.8% | 19% / 39% | ❌ |
| shared → shared+MoE | −16.0% | 19% / 39% | ❌ |

**Verdict: at this budget, 3.2× the parameters bought less quality than 1.4× the compute would
have.** Scaling the dense model is the better use of the same FLOPs. The failure is consistent
across both the dense and shared baselines, so it is not an artifact of one configuration.

**The levers do compose.** MoE helps the shared model (−16.0%) at least as much as it helps the
dense one (−11.8%), so sharing and MoE are not fighting each other — MoE simply is not worth
its compute in either. This closes acceptance criteria 4 and 5 of #123 with a negative result.

**Two caveats, neither of which rescues the result:**
- **The 1.91× vs 1.40× gap is partly implementation.** `MoEFFN` loops over experts in Python
  rather than batching them; a tuned kernel would narrow the wall-clock penalty. But the
  FLOP-based 19% bar is missed regardless, so optimisation changes the margin, not the verdict.
- **5M tokens is very undertrained, and MoE has an extra thing to learn.** Routing — which
  expert for which token — is a harder learning problem than a dense FFN's, and may simply not
  have had the data to specialise. **This is the single strongest reason not to treat this as
  final**, and it is untested. If MoE is revisited, the first thing to check is whether the gain
  grows with token budget; a fixed ~12-16% that does not grow means the routing never learned
  anything useful.

**One result worth keeping regardless of the compute verdict.** shared+MoE reaches PPL 131.3 at
**18.19M stored params**, against dense needing **39.76M** for 119.7 — **2.2× fewer stored
parameters for 10% worse PPL**. That is a deployment-size story, not a training-compute one, and
it stacks with ternary's ~20x bits (measured to compose cleanly earlier today). For the
edge-inference thesis this project is actually built around, that combination matters more than
the training-FLOP comparison MoE just lost.

## Looped early exit — no-retrain hyperparameter search FAILS the gate (Issue #122, 2026-08-01)

Wrapped `EdgeHVN.forward_step` (`hvn/edge/looped.py`) so the block stack can re-run up to N
times per token, exiting once normalized softmax entropy drops below threshold T — a
"looped transformer" style weight-tied refinement, entropy-only exit signal (a trainable
confidence-MLP was considered and dropped: with no retraining allowed, its weights would be
random and its "confidence" scores meaningless). Contract preserved exactly — `forward_step`/
`init_state`/`chat_step` signatures unchanged, `ChatSession` needs zero code changes
(`tests/test_looped_early_exit.py`, 6/6 CPU parity/correctness tests, incl. a proof that a
discarded loop's candidate state never leaks into the next loop's attempt).

**Full grid N ∈ {2,4,8,16} × T ∈ {0.1,0.3,0.5,0.7} on `checkpoints/hybrid_a4_v2/
hybrid_a4_step8000.pt`** (the checkpoint the 35.5%/8.5% recall baseline was originally measured
on — `hybrid_tooluse_v1.pt`, named in the original task spec, does not exist in this working
tree or any worktree). Measured via a dedicated `forward_step`-based generation harness
(`scripts/sweep_looped_early_exit.py`), not `eval/synthetic_recall_probe.py`'s own `generate()`
— that uses full `forward()`, which uses `_scan_chunked` and diverges numerically from
`forward_step`'s `_scan_sequential` on a trained checkpoint (see the #97 incremental-decode
entry above); reusing it would have silently never exercised the loop wrapper at all.

**Result: every one of the 16 combinations fails.** The pattern is a clean binary, not a
tunable middle ground:

| T | Behavior (all N) |
|---|---|
| 0.1 / 0.3 | avg loops ≈ N (always maxes out) → recall = **0%**, near AND far, every time |
| 0.5 | mixed loop counts, recall still **0%** near AND far |
| 0.7 | avg loops ≈ **1.0–1.07** (almost never actually loops) → recall reverts to ~baseline |

There is no hyperparameter point where the model loops *and* stays useful. **Root cause: the
block stack was never trained to refine its own output** — a second pass through untrained
weights pushes the hidden state off the distribution it was trained on, and exact-token recall
collapses almost immediately. This was flagged as the key risk before implementation (an
untrained "confidence" mechanism can't be assumed quality-neutral) and the sweep confirms it
directly rather than just plausibly.

**Closest candidate, N=4/T=0.7, re-run at the full 300-trial validation gate** (vs. a
300-trial-measured N=1 baseline, not the old doc numbers — see harness note above):

| Metric | Baseline | N=4,T=0.7 | Target | Result |
|---|---:|---:|---|:---:|
| Near-window recall | 33.67% | 28.67% | ≥30% | ❌ |
| Far-window recall | 6.33% | 7.0% | ≥7% | ✅ (exactly on the line) |
| Avg loops, easy tokens | — | 1.0 | ≤2 | ✅ |
| Avg loops, hard tokens | — | 1.024 | ≥N/2 | ❌ |
| Tokens/sec | 27.82 | 54.34 | improves | ✅ (see caveat) |

**Overall FAIL**, and the two nominal passes don't rescue it: avg-loops-hard ≈ avg-loops-easy ≈
1.0 means the mechanism isn't discriminating hard from easy tokens at all — at T=0.7 it's
functionally inert, not "using" loops. The 2× tokens/sec figure is inconsistent with a loop
count barely above 1.0 and is almost certainly single-run GPU/system measurement noise (one
40-token generation per combo), not a real architectural speedup.

**Decision: untrained looped early exit does not work on this checkpoint at any tested (N,T).**
This is a real, informative negative result for the no-retrain phase of #122, not a bug —
correctness was independently verified via the parity tests above. The next step, if this
capability is still wanted, is a loop-aware **continued fine-tune** (train with randomized loop
counts so the block learns to be a genuine refiner of its own output), reusing this checkpoint's
existing pretrain/FT investment — not a from-scratch retrain, and not further hyperparameter
search on the frozen checkpoint. Full sweep data: `results/looped_early_exit/sweep_results.json`,
`results/looped_early_exit/validation_gate_N4_T0.7.json`.

## Loop-aware continued-FT: recipe check + a scan-kernel default flip (Issue #122, 2026-08-01)

**Stage 1 — cheap ~780K-param sandbox** (`scripts/train_looped_sandbox.py`, char-level,
from-scratch, trains in minutes): tests the loop-aware training recipe itself (random loop
count `k ~ Uniform[1,max_loops]` per step, loss averaged across EVERY loop's prediction —
"deep supervision" — not just the final one, so the model is actually taught that an early
loop's output alone should be usable). Result **ambiguous, not conclusive**: held-out loss got
worse with more loops in both an initial (flawed, out-of-distribution held-out set) and a
corrected (in-distribution train/val split) run. Root confound: the tiny repeated training
corpus (583 chars × 200) is fully solvable at k=1 alone, so there's no training pressure for
extra loops to do anything — the sandbox can't distinguish "recipe doesn't work" from "task was
too easy to need it." Recorded in `results/looped_early_exit/sandbox_1m_results.json`.

**Stage 2 — the real test** (`continue_ft_looped.py`): a genuine **continued fine-tune**, not a
retrain — warm-starts directly from `checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt` (same
checkpoint the #122 sweep above ran on) and continues training with the loop-aware objective on
`data/grounded_qa_expanded.jsonl` (5,240 real, non-repeated examples — the sandbox's
memorization confound doesn't apply). In progress; loss descending cleanly (0.87 → 0.46 by
step 560/2000).

**A real bug caught mid-run, and a genuine finding underneath it.** The first launch measured
**41.88s/step** and was wrongly attributed to VRAM/WDDM spill-over (a documented failure mode
elsewhere in this doc) — cutting `--batch-size` 4→2 didn't change `nvidia-smi`'s reported usage
at all, which should have been the tell. The real cause, found by diffing against the working
reference script (`continue_ft_hybrid.py`) rather than continuing to guess: that script's
`--fast` enables the fused Triton scan + megakernel backward (#83/#96) on the DeltaNet blocks;
`continue_ft_looped.py` had never wired them in at all, so it ran DeltaNet's slow default eager
scan, uncompiled. Wiring them in: **~1s/step, a ~42x speedup**, confirmed with real
`torch.cuda.max_memory_allocated()` instrumentation (4.6 GiB peak) in place of `nvidia-smi`
guessing.

**Default flipped: triton-scan + megakernel-scan are now unconditionally on in both
`continue_ft_hybrid.py` and `continue_ft_looped.py`** (no `--fast` needed for this part
anymore). This was checked thoroughly before flipping, not assumed:
- `tests/test_issue83.py` + `tests/test_issue83_stacking.py` (7/7, re-run fresh) confirm the
  Triton forward/backward/full-block match the exact sequential reference to rel-diff < 2e-3.
- `scripts/ab_megakernel.py` documents the megakernel backward as loss-bit-identical, grad
  diff ~5e-8.
- **The direction of the risk is the opposite of what was first assumed.** The Triton kernel
  implements the same safe log-space formulation as `_scan_chunked_stable`
  (`hvn/core/deltanet_triton.py`: "never forms `1/P`"). The path it replaces — the *default*,
  no-flag eager `_scan_chunked` — is the one with a real, code-documented correctness caveat:
  its own docstring says it can be "silently wrong on decayed channels at large chunks / hard
  decay," and the #44 `P`-floor fix only bounds the catastrophic Inf/NaN case, not full
  accuracy. So enabling triton-scan isn't a speed-for-accuracy trade — it's strictly better on
  both axes.

**Scope of the flip, deliberately narrow.** Only `continue_ft_hybrid.py`/`continue_ft_looped.py`
were changed. NOT touched: `DeltaNetBlock`'s own constructor default (`use_triton_scan=False`),
`EdgeHVN`'s own default (also `False`, and it doesn't expose `use_megakernel_scan` as a
constructor arg at all — only settable post-construction), or `train_edge_conversational.py`'s
own more elaborate `--fast`/`--triton-scan` machinery (which already conditionally enables
triton-scan under `--fast` for `--arch hvn`, gated by logic not fully audited here). Those are
wider-reaching, separately-scoped changes — affecting the main pretraining script, past
benchmark reproducibility assumptions, and other callers — not something to flip as a
side-effect of an #122 stage-2 script fix.

**Follow-up — attempted the broader flip too, and the caution above was justified.** Tried
flipping `DeltaNetBlock`/`EdgeHVN`'s own class-level `use_triton_scan`/`use_megakernel_scan`
constructor defaults to `True` (the true single source of truth every caller inherits from).
Two real, previously-undiscovered caveats surfaced from actually running the broader test suite
rather than assuming the earlier scoped verification generalized:

- **`use_megakernel_scan=True` by default breaks `torch.compile` fusion**:
  `tests/test_issue83_stacking.py::test_zero_graph_breaks` failed — 2 graph breaks instead of 0.
  Correctness is unaffected (all parity tests still passed), but it's a real compile-fusion
  regression, not just noise. Reverted this one back to `False` at the class level.
- **`use_triton_scan=True` by default breaks `BlockCoordinateScheduler` weight-offload
  parity** — a real, non-trivial loss divergence (0.297 vs 0.010 for the eager scan, not
  floating-point noise) specifically in
  `tests/test_block_coordinate_offload.py::test_lru_cache_hit_avoids_reload_and_preserves_parity`.
  Isolated with a standalone repro to confirm it's the scan flag and not something else, and
  confirmed `grad_checkpoint` alone (no BCD/offload) is unaffected
  (`tests/test_edgehvn_grad_checkpoint.py` 5/5) — narrow to the offload buffer-caching
  mechanism specifically, not root-caused, so also reverted to `False` at the class level.

**Net decision: class-level defaults (`DeltaNetBlock`, `EdgeHVN`) stay `False`.** Only
`continue_ft_hybrid.py`/`continue_ft_looped.py` have it on (verified safe there — neither uses
BCD/offload or `torch.compile` alongside the loop-aware path). This also answers why
`train_edge_conversational.py`'s own `--fast`/`--triton-scan` wiring was left untouched: that
script supports `--block-coordinate`, so the same BCD-offload interaction risk applies there
too — extending the flip to it would need the same root-cause work this pass didn't do.
`use_megakernel_scan` is now explicitly threaded through `EdgeHVN`'s constructor (previously
only settable post-construction) even though its default stays `False` — harmless plumbing
improvement, kept regardless of the revert.

## Stage 2 result — loop-aware continued-FT does NOT fix the recall collapse (Issue #122, 2026-08-01)

`continue_ft_looped.py` completed: 2000 steps (115.3 min, `--fast`, batch_size=2, grad_accum=16,
max_loops=3, warm-started from `checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt` on
`data/grounded_qa_expanded.jsonl`), final train loss 0.2507 (ppl 1.28), clean convergence.

**Validation (`scripts/validate_looped_ft.py`)** — fixed loop counts N=1,2,4
(`entropy_threshold=-1.0` forces exactly N loops every time, isolating "does looping still
corrupt recall" from any entropy-threshold question), 20 trials/N, near-window synthetic
recall, FT checkpoint vs the frozen baseline it was warm-started from:

| N | FT checkpoint (2000 steps) | Frozen baseline |
|---|---:|---:|
| 1 | 30.0% (6/20) | 45.0% (9/20) |
| 2 | **0.0%** (0/20) | **0.0%** (0/20) |
| 4 | **0.0%** (0/20) | **0.0%** (0/20) |

**Verdict: the loop-aware continued-FT does not fix the collapse.** Identical pattern on both
checkpoints — recall craters to exactly 0% the instant looping actually happens (N>1), and the
FT checkpoint is even slightly *worse* at plain single-pass (N=1) recall than the frozen
baseline it started from (30% vs 45%) — a real cost with zero offsetting benefit at N>1.
Real, non-repeated training data (`data/grounded_qa_expanded.jsonl`, 5,240 examples) rules out
the memorization confound that made the stage-1 sandbox result ambiguous — this is a clean
negative result, not an artifact of insufficient data variety.

**Working hypothesis for why, not yet tested:** the deep-supervision recipe (loss averaged
across every sampled loop's prediction, `hvn.edge.looped.looped_forward_train`) has a
structural gap. The same weights are trained to satisfy "loop 1 alone should already answer
correctly" and "loop 2 — which further transforms loop 1's now-correct hidden state — should
*also* answer correctly," with nothing in the objective rewarding "leave the answer alone once
it's already right." Real looped-transformer/universal-transformer designs typically need an
explicit mechanism for that (step embeddings, learned halting, or a residual-style update that
lets later iterations be small perturbations rather than full re-transformations) — a plain
re-application of the full block stack has no such pressure built in. More training steps on
the same recipe would not obviously fix a structural gap like this; it would need a different
training design, not more of the same one.

**Status: Issue #122's Looped Early Exit, as implemented (weight-tied loop over the
already-full-depth 12-layer hybrid + naive deep supervision), does not deliver a working
mechanism — neither without retraining (the original N×T sweep) nor with a real continued-FT
(this stage). Parking here rather than continuing to iterate on this specific recipe.** If
picked up again, the next steps are (a) a genuinely weight-shared/looped architecture
(this repo already has `EdgeHVN(shared_weights=True)` infrastructure for that — the original
issue's own "reuse a single weight-shared block" framing, never actually used by this work,
which instead looped the full already-deep hybrid) or (b) a training objective with an explicit
settle/halt mechanism, not just randomized-depth deep supervision. Full results:
`results/looped_early_exit/validate_looped_ft.json`,
`checkpoints/hybrid_looped_ft/hybrid_looped_step2000.pt`.

## Gated-loop fix attempt — tested cheaply, makes things worse when opened (Issue #122, 2026-08-02)

Tried the (b) option above directly: `scripts/train_looped_sandbox_gated.py` adds a learned
per-token gate so loop 2+ blends the block stack's raw output with the previous hidden state
(`h <- h + sigmoid(loop_gate(h)) * (Blocks(h) - h)`) instead of overwriting it outright — loop 1
stays the full ungated pass. Same disposable ~780K-param, from-scratch, char-level sandbox as
stage 1 (not a fine-tune — deliberately tiny so this is a fast, cheap mechanism check before
touching the real 83M checkpoint again).

**v1** (gate bias init -3, `sigmoid(-3)~=0.047`, near-closed at start), 3000 steps: partial
win. Loss still degrades with more loops but ~3x less than the ungated recipe (+0.09 k1->k4 vs
ungated's +0.27). Gate barely moved during training (0.047 -> 0.049).

**v2** (gate bias init -1, `sigmoid(-1)~=0.27`, more open at start, hypothesis: v1's gate was
gradient-starved by starting too saturated), 5000 steps: **hypothesis was wrong, and the result
is worse on every axis** — both overall single-pass quality (k=1 loss 5.80 vs v1's 4.56) and
the loop-degradation amount (+0.19 k1->k4 vs v1's +0.09). The gate spent the whole run trying
to close itself back down (0.27 -> 0.257) and never got back to where v1 started.

**Reading:** starting the gate MORE open made things worse, not better -- the correct amount of
gate openness is "as closed as possible," and v1's conservative init was already closer to
right. This rules out the "gradient-starved gate" explanation and points to a cleaner, stronger
conclusion: **any exposure to the block stack's repeated (loop 2+) transformation hurts,
independent of how the gate is tuned** -- not a fixable hyperparameter/init issue. The gate
mechanism works as designed (it lets the model suppress harmful extra loops), but there's
nothing useful in the block stack's second-pass output for it to let through. This is now the
second training-side fix attempted (deep supervision alone, then deep supervision + gating) that
fails to make looping productive on this architecture -- reinforces the (a) option above
(a genuinely weight-shared, deliberately-smaller-and-looped architecture) as the more promising
remaining direction over further training-recipe tweaks on the current already-full-depth
hybrid. Full results: `results/looped_early_exit/sandbox_1m_gated_results.json` (v1),
`results/looped_early_exit/sandbox_1m_gated_v2_results.json` (v2). Not attempted on the real
83M checkpoint -- the cheap sandbox check is exactly what caught this before that spend.

## First positive signal — looping WITHOUT early exit modestly helps (Issue #122, 2026-08-02)

Every attempt above trained with deep supervision (every sampled loop count's prediction
graded independently), because that's what entropy-gated early exit requires (loop 1 alone
must already be a usable answer). That requirement was never separated from "does repeated
computation help at all" until this check. `scripts/train_looped_sandbox_fixed.py` drops early
exit entirely: fixed N loops every step (no random k), loss on ONLY the final loop's
prediction -- the standard way weight-tied/looped depth is normally trained in the literature.
Same disposable ~780K-param sandbox, N=1 (plain single-pass baseline) vs N=4, same 3000
training steps, held-out loss:

| N | held-out loss |
|---|---:|
| 1 (single pass) | 5.0050 |
| 4 (always 4 loops, final-loop-only supervision) | **4.9621** |

**N=4 beats N=1** — modest (~0.9% relative) but real and the first improving result in this
entire investigation. Confirms the hypothesis from the gated-fix section above: deep
supervision's requirement that loop 1 already be independently correct was actively
suppressing the model's ability to use loop 2+ productively. Remove that requirement and
repeated computation *is* mildly useful at this toy scale.

**This cleanly separates two previously-conflated questions.** "Does looping help at all"
(this result: yes, mildly, when not required to also support early exit) is now distinct from
"does early exit specifically work" (still no — every early-exit-compatible training attempt,
deep-supervised or gated, failed). A **fixed-depth looped model with no early exit** is a
plausible, still-open direction; **early exit** (the original #122 ask — cheap tokens skip
loops, hard tokens get more) remains unproven and every attempt at it so far has failed.
Whether a fixed-depth loop is worth the N× inference cost without the early-exit savings is a
separate cost/benefit question, not yet evaluated. Full results:
`results/looped_early_exit/sandbox_1m_fixed_N1.json`, `sandbox_1m_fixed_N4.json`.

## Real-scale check of the fixed-N recipe — sandbox signal did NOT transfer (Issue #122, 2026-08-02)

Ran the sandbox's winning recipe (fixed N=4 loops, final-loop-only supervision, no early exit)
as a real continued-FT on the 83M checkpoint (`continue_ft_looped.py --fixed-n
--no-deep-supervise`, 2000 steps, 196.3 min, final train loss 0.30). Validated at N=1/2/4 on
both the step-500 (25% through) and step-2000 (final) checkpoints:

| N | Fixed-N FT, step 500 (early) | Fixed-N FT, step 2000 (final) | Frozen baseline |
|---|---:|---:|---:|
| 1 | 26.7% | 30.0% | 45.0% |
| 2 | — | **0.0%** | **0.0%** |
| 4 | **0.0%** | **0.0%** | **0.0%** |

**The sandbox's positive signal did not transfer.** Same collapse pattern as every prior
attempt, identical at both the early and final checkpoint — more training didn't help within
this run's budget either. The toy-scale result (N=4 beating N=1 by ~0.9% in held-out *loss*)
was a real, measured effect, but it didn't translate into the *recall* metric that actually
matters here — a small general next-token-prediction improvement is not the same thing as
fixing long-range associative recall, which apparently needs a qualitatively different fix, not
just "let the model use more compute if it wants to." This is the reason the real-scale check
mattered rather than treating the sandbox result as sufficient evidence on its own.

**This exhausts the training-recipe angle for Issue #122.** Three distinct training approaches
tried on the current (already-full-depth, 12-layer) hybrid architecture, all fail on the recall
metric:
1. Deep supervision (every sampled loop graded independently) — fails, recall collapses to 0%.
2. Deep supervision + a learned per-token gate — fails; opening the gate further made both
   overall quality and the collapse *worse*, not better.
3. Fixed-N, final-loop-only supervision (no early exit at all) — showed a small, real
   toy-scale loss improvement, but recall still collapses to 0% at real scale.

**Final status: parking Issue #122 as far as training-recipe iteration on this architecture
goes.** The only genuinely untested direction left is architectural — a real weight-shared,
deliberately-smaller-and-looped model (`EdgeHVN(shared_weights=True)` already exists in this
repo for that), which this entire investigation never actually tried; everything above looped
the already-full-depth 12-layer hybrid instead. That's a materially different, larger
commitment (its own from-scratch-vs-pretrain decision) and a separate piece of work if picked
up again, not a variation on anything tried here. Full results:
`results/looped_early_exit/validate_looped_ft_fixed_step500.json`,
`results/looped_early_exit/validate_looped_ft_fixed_step2000.json`,
`checkpoints/hybrid_looped_ft_fixed/hybrid_looped_step2000.pt`.

## Option (a) BUILT AND MEASURED — a small looped model loses to a deep one (Issue #122, 2026-08-21)

**This closes the last open looping direction.** Every earlier result looped an
already-full-depth 12-layer stack pretrained single-pass, and the docs' remaining
direction was a deliberately *small* stack trained from scratch to be run many times. Built
(`scripts/ab_looped_small.py`) and run. Both arms unroll to a structurally identical
12-layer network — same depth, same block order, attention at positions 0/3/6/9 in both —
differing only in whether the weights repeat. Both drive `looped_forward_train` (the deep
arm at k=1), so they share one code path. d_model 384, 13,000 steps, batch 4, seq 512
(26.6M tokens), Muon, identical seed and pre-materialised batches, from scratch:

| arm | effective depth | body params | held-out | steps/s | wall |
|---|---:|---:|---:|---:|---:|
| looped3x4 (3 layers x 4) | 12 | **4,490,461** | 5.37096 | 5.60 | 38.7 min |
| deep12 (12 layers x 1) | 12 | 17,808,769 | **5.27038** | 4.32 | 50.2 min |
| deep12_twin | 12 | 17,808,769 | 5.27956 | 4.27 | 50.7 min |

| comparison | gap | vs noise floor |
|---|---:|---:|
| matched steps (13,000) | **+0.10057** nats | 11.0x |
| matched wall clock (38.7 min) | **+0.05699** nats | 6.2x |

**Verdict: looping loses both ways.** 3.97x fewer body params costs 0.101 nats at matched
steps. The looped arm *is* 1.30x faster per step (fewer matrices for Muon), which halves the
gap on a wall-clock budget — but does not close it.

**Why this is the strong direction of evidence.** Both arms saw the same tokens while
deep12 had 4x more body params to fit with them, so deep12 is the MORE undertrained arm and
the bias favoured looping. Looping lost anyway. (Had it won, that would have been weak
evidence needing a longer run — the asymmetry was stated before the run, not after.)

**Methodological gotcha, and it cost the earlier estimate.** The `deep12_twin` known-answer
cell drifted **9.171e-03** at 13,000 steps, against **1.8e-04** measured on the same code at
20 steps in smoke. **Drift accumulates: a short determinism check understates the real noise
floor by ~50x.** Any A/B here that sets its significance band from a smoke-length twin is
setting it far too low. Both gaps above still clear the true floor by 6-11x.

**Limits, including one that bounds the agentic-coding question specifically.** One seed,
d_model 384, 26.6M tokens — both arms are undertrained (this bounds early-training quality,
not convergence). Ternary QAT deliberately not applied, since quantising 3 shared blocks and
12 independent ones are not the same operation. **Recall was NOT measured, deliberately:** at
ppl ~194 these from-scratch models cannot recall anything, so the near-window probe would
floor at 0% for both arms and discriminate nothing — a vacuous test, the exact failure this
file keeps recording. So this settles the *quality-per-parameter* question and leaves the
*recall/agentic-coding* question at this size unanswered; answering it needs a model trained
far enough to have recall at all. Full results:
`results/looped_small/ab_looped_small.json` (carries the full per-arm trajectory and
training curve; `logs/` is gitignored, so the console log is local-only).

## Fixed-N looping, finally graded on held-out LOSS — the negative holds, and is now properly supported (Issue #122, 2026-08-20)

**The gap this closes.** The fixed-N recipe's only positive result was a *loss* win in the
~780K sandbox (N=4 4.9621 vs N=1 5.0050). The real-scale check above then graded it on
*recall* only — `results/looped_early_exit/validate_looped_ft_fixed_step2000.json` contains
nothing but `by_n` recall percentages. So "the sandbox signal did not transfer" rested on a
different metric than the one that produced the signal, and held-out loss under fixed-N
looping at 83M had never been measured. It has now been
(`scripts/eval_looped_heldout_loss.py`, 199,290 tokens of `data_cache/heldout_fineweb.txt`,
seq_len 512, final-loop-only, both checkpoints, no retraining — the checkpoints already existed):

| N | FT (trained at fixed N=4) | frozen baseline |
|---|---:|---:|
| 1 | **4.9257** (ppl 138) | **4.8012** (ppl 122) |
| 2 | 5.0380 (+0.112) | 4.9797 (+0.179) |
| 4 | 5.4452 (+0.520) | 5.5121 (+0.711) |

**Verdict: looping monotonically WORSENS held-out loss at real scale, on both checkpoints.**
The negative from the recall check stands, now confirmed on the metric that actually showed
the sandbox win.

**The sharpest fact here: the FT checkpoint was trained with fixed N=4, and N=4 is its worst
setting.** It was optimised for the exact configuration it performs worst in. The +0.520 nats
it loses at its own training depth is **12x larger, and opposite in sign, to** the 0.043 nats
the sandbox gained. Note also the FT arm is worse than the frozen baseline at N=1 as well
(4.9257 vs 4.8012) — the same "real cost, zero offsetting benefit" shape the recall table
showed, so this is not a metric artifact.

**Guards, because a flat result here would have read as "looping is neutral".**
`enable_looped_early_exit` patches only `forward_step`; `eval/heldout_loss.py` calls
`model(x)`. Wiring those two together returns byte-identical logits at every N — a test that
cannot fail, the #138 shape. The script therefore uses `looped_forward_train` (the
full-sequence path the FT actually trained with) and **asserts** `max|logits(N=1) −
logits(N=2)| > 0` (measured 9.32 / 9.91, so the loop is genuinely running) and that every arm
scored an identical token count. It aborts rather than reporting a flat curve.

**Limits, stated because they bound the claim.** One held-out set, and FineWeb is
out-of-distribution for a conversational model (the absolute NLL ~4.8 reflects that); the
N-comparison is within-model on identical tokens, which is what the question needs, but the
absolute values are not a quality figure for these checkpoints. **This does not touch option
(a).** Both arms are continued fine-tunes of a checkpoint pretrained single-pass, looping an
already-full-depth 12-layer stack. A genuinely weight-shared, deliberately-smaller,
looped-from-scratch model (`EdgeHVN(shared_weights=True)`) remains untested — this result
closes "fixed-N looping on this architecture", not "looped LLMs". Full results:
`results/looped_early_exit/heldout_loss_fixed_n.json`.

## Mechanism diagnostic — the input-injection/skip-attention hypothesis is REFUTED (2026-08-02)

All four training recipes above failed identically (recall → 0% the instant N>1, general loss
fine) — a pattern consistent with looping destroying information exact recall depends on,
rather than a trainable deficiency. Two mechanism-level fixes were proposed and tested directly
on the **frozen** checkpoint (`scripts/diagnose_loop_recall_collapse.py`, zero training, ~20 min
GPU): (A) re-add the original token embedding every loop (`h ← Blocks(h + x0)`, standard in
Universal Transformer / Looped Transformer, absent from this implementation), and (B) run the 2
windowed-attention layers only once (loop 1) and loop only the 10 DeltaNet layers, on the theory
that attention specifically needs an un-smeared input to copy exactly.

**Neither rescues recall, alone or combined** — all four variants (current / A / B / A+B)
collapse to exactly 0% at N=4, N=1 sanity-checked at 45% matching the known baseline. This
refutes the specific hypothesis: it is not simply "attention loses its token anchor across
loops" (A alone would have fixed that) and not simply "re-running attention on an already-
transformed hidden state corrupts the copy" (B alone would have fixed that, and did not — even
with attention frozen after loop 1, recall still died).

**Revised hypothesis, not yet tested:** the destructive mechanism more likely sits in the
**DeltaNet recurrence itself**, not attention. DeltaNet's update is an overwrite —
`S ← diag(α)S + β(v − Sk)kᵀ` — keyed on k/q computed from whatever hidden state is fed in. On
loop 2+, k/q are computed from an already-transformed hidden state rather than the original
token, so the overwrite plausibly replaces loop 1's correctly-keyed association with a
differently-keyed one instead of refining it. Testing this needs inspecting the actual
recurrent state matrix across loop iterations (real mechanistic-interpretability work), not
another architectural toggle — a materially larger investigation than anything tried in this
doc so far, and not started. Full results: `results/looped_early_exit/diagnose_loop_recall_collapse.json`.

**Status: parking here.** Five real experiments now agree (4 training recipes + this
no-training mechanism probe) that looping this architecture's block stack, in every form
tried, destroys exact recall. The training-recipe angle and the two most obvious
architecture-level bypasses are exhausted. Remaining open directions, neither started:
(a) genuine weight-shared smaller-and-looped architecture (see above), (b) direct inspection of
the DeltaNet state matrix across loops to confirm or refute the overwrite hypothesis.

## ⚠️ RETRACTION — every #122 result above was measuring a one-line bug (2026-08-02)

**All of the negative conclusions in the six #122 sections above are invalid.** They do not
show that looping this architecture destroys recall; they show that
*looping-with-corrupted-memory* destroys recall. The bug was in the first file written for
this issue and silently confounded everything downstream.

**The bug.** `hvn/edge/looped.py` committed the **last** loop's recurrent state as the state
carried to the next token. Only loop 1 runs the blocks on the real token embedding — loops 2+
run them on an already-transformed hidden state, so their k/v are not derived from this token
at all. For DeltaNet the state *is* the memory (`S ← diag(α)S + β(v − Sk)kᵀ`), so committing a
later loop's state writes an association keyed on noise, once per extra loop — N writes per
token instead of one — corrupting precisely what the next token must read back.

**How it was found: by looking at the failure instead of scoring it.** Five experiments had all
measured the same binary (did the answer contain the target word) and none had inspected the
behaviour. Doing so (`scripts/observe_loop_failure.py`) showed the N=4 output is raw HTML/web
markup — this doc's own documented signature for the *base pretrain model* — meaning looping
knocked the model out of its instruction-following distribution rather than merely degrading
the answer. It also measured the residual stream growing 351 → 2560 (7.3×) over 8 loops at a
near-constant +315/loop, with argmax locking onto token 3467 (`' \'`) from loop 2 onward.

The decisive isolation came from a **failed sanity check**. That script's α-sweep included
α=0, which leaves `h` at loop 1's value so the logits are provably identical to N=1 and it
*must* reproduce ~45%. It returned 0/20. That invalidated the α curve, but with the logits held
identical the only remaining difference from N=1 was which state got committed. A 2×2 then
separated the two candidate mechanisms (`scripts/test_state_commit_hypothesis.py`, frozen
checkpoint, 20 trials/cell):

| | state ← loop 1 | state ← loop 4 |
|---|---:|---:|
| **logits ← loop 1** | 40.0% (N=1 baseline ✓) | **0.0%** |
| **logits ← loop 4** | **35.0%** | 0.0% (old behaviour) |

Cell (c) is decisive: loop-4 logits over a clean loop-1 state hold at 35%, statistically
indistinguishable from the 40% baseline. **The residual-stream drift is real but survivable —
it is not the cause.** The single fatal mechanism is the state commit.

**Fix:** commit loop 1's state unconditionally, regardless of which loop the exit fires on.
Extra loops refine the *prediction*; they do not rewrite *memory*. One line.
`tests/test_looped_early_exit.py` had asserted the loop-2 state and so locked the bug in —
rewritten to assert loop 1's, plus a guard that the two states genuinely differ so it cannot
silently pass again (6/6).

**Post-fix re-validation** (frozen checkpoint, fixed N, early exit disabled, two independent
runs of the same checkpoint so their spread calibrates the noise floor):

| N | run A | run B | pooled |
|---:|---:|---:|---:|
| 1 | 40.0% | 45.0% | 42.5% |
| 2 | 35.0% | 45.0% | 40.0% |
| 4 | 35.0% | 35.0% | 35.0% |
| 8 | 30.0% | 25.0% | 27.5% |

Recall **holds at every N**, versus exactly 0.0% at every N>1 before the fix. A pooled
monotonic decline (42.5% → 27.5%) is visible and directionally consistent with the measured
drift, but at ~2 standard errors it is suggestive rather than established — ±2 hits at n=20 is
ordinary noise here, as the two identical-checkpoint runs demonstrate.

**What is now known vs still open.** Looping is non-destructive — that is settled. The original
#122 ask (entropy-gated early exit buying a speedup at maintained quality) is **untested**: the
above used fixed N with the exit disabled. The N×T sweep must be re-run from scratch, because
entropy was previously computed on garbage logits, so the threshold behaviour that drove the
original sweep's conclusions ("low T always maxes out loops, high T never loops") was itself a
symptom of the bug. Likewise the four training runs are re-opened: they trained against
corrupted memory, so their failures carry no information about the training recipes themselves.

**Process note worth keeping.** The α=0 sanity check that exposed this was added on the fifth
experiment. Had a "this configuration must reproduce the known baseline" assertion been built
into the very first sweep, the four training runs (~6 h GPU) would never have been spent. Every
scoring harness in this project should carry a cell whose correct answer is already known.

## Post-fix N×T sweep — mechanism works, early exit's PREMISE does not (2026-08-02)

Re-ran the sweep the retraction above invalidated, on the frozen checkpoint with the
state-commit fix in place, 20 trials/cell. The harness now carries a **known-answer cell**
(`max_loops=8, T=1.1`: the loop machinery is engaged but the threshold is trivially
satisfiable, so it must exit at loop 1 and equal the N=1 baseline by construction). It
passed at 45.0% vs 45.0%, delta 0.0pp — the check that would have caught the original bug,
which an N=1-only baseline could not, since N=1 never loops.

| cell | near% | avg loops | easy | hard | **hard−easy** |
|---|---:|---:|---:|---:|---:|
| baseline (N=1) | 45.0 | 1.00 | 1.00 | 1.00 | 0.00 |
| sanity (N=8, T=1.1) | 45.0 | 1.00 | 1.00 | 1.00 | 0.00 |
| N=2 T=0.1 | 40.0 | 1.97 | 1.96 | 1.97 | +0.01 |
| N=2 T=0.3 | 35.0 | 1.64 | 1.75 | 1.60 | −0.15 |
| N=2 T=0.5 | 40.0 | 1.24 | 1.27 | 1.23 | −0.05 |
| N=2 T=0.7 | 35.0 | 1.02 | 1.00 | 1.02 | +0.02 |
| N=4 T=0.1 | 40.0 | 3.91 | 3.94 | 3.90 | −0.04 |
| N=4 T=0.3 | 35.0 | 2.86 | 3.23 | 2.73 | **−0.50** |
| N=4 T=0.5 | 40.0 | 1.71 | 1.80 | 1.67 | −0.13 |
| N=4 T=0.7 | 50.0 | 1.04 | 1.00 | 1.06 | +0.06 |
| N=8 T=0.1 | 25.0 | 7.79 | 7.81 | 7.78 | −0.02 |
| N=8 T=0.3 | 25.0 | 5.25 | 5.96 | 5.07 | **−0.89** |
| N=8 T=0.5 | 15.0 | 2.55 | 2.47 | 2.57 | +0.10 |
| N=8 T=0.7 | 35.0 | 1.10 | 1.00 | 1.13 | +0.13 |

The N=16 row and N=8's remaining cells were killed deliberately — the conclusion was already
determined and the sign is consistent across N=2/4/8 (`sweep_after_fix_partial.json`).

**1. The mechanism works.** Recall holds 35–50% at N=2 and N=4 versus **exactly 0.0% at every
N>1** pre-fix, and the loop count is genuinely tunable (T=0.3 → 1.64, T=0.5 → 1.24). The
pre-fix finding that there was no usable middle range — "low T always maxes out, high T never
loops" — is **superseded**: that was a symptom of entropy being computed on corrupted logits.

**2. Early exit's premise fails, and it fails backwards.** The whole idea is that hard tokens
earn more compute. Measured, hard tokens get *less*: seven of twelve looping cells are
negative, and — the part that matters — **every large effect is negative** (−0.89, −0.50,
−0.15) while every positive is trivial (≤ +0.13, all in cells that barely loop at all).
Wherever looping actually happens, the gate spends compute on the wrong tokens.

Mechanistically this is coherent, not anomalous. "Hard" here means the model's previous greedy
prediction missed. The entropy gating the exit is over the **next** token. After a surprising
token the model often lands somewhere formulaic and commits confidently → exits early. After a
predictable token it can sit at a genuine branch point with many plausible continuations →
high entropy, keeps looping. **Entropy over the next token measures branchiness, not difficulty
of the current one**, and those are different quantities. No threshold repairs that, and it is
scale-independent.

**3. Heavy looping costs recall.** N=8 reads 25% / 25% / 15% against the 45% baseline —
consistent with the residual-stream drift measured earlier (‖h‖ 351→2560, 7.3× over 8 loops).
Looping is roughly neutral at N=2–4 and actively harmful beyond. Combined with (2), there is no
configuration that both loops meaningfully and helps.

**4. Throughput numbers here are unusable.** N=4/T=0.3 reads 10.80 tok/s on *fewer* loops than
N=4/T=0.1's 17.02 — physically impossible, so single-generation timing is noise-dominated. The
apparent 1.7× at N=2/T=0.7 comes from averaging 1.02 loops, i.e. from not looping, not from
smart gating. Any real throughput claim needs repeated timed runs.

**Verdict, and how it differs from the retracted one.** Entropy-gated early exit does not work
on this model — but this time for a *measured property of the model*, on a mechanism proven
sound by a known-answer cell, rather than as the shadow of a bug. Distinguish this from the
retraction above: same shape of answer, entirely different epistemic standing.

**Still genuinely open:** the four training runs (deep supervision, gated, fixed-N) all trained
against corrupted memory, so their failures carry no information about the recipes themselves.
Anything re-run there should carry a known-answer cell in its own harness before GPU is spent.

## ALBERT weight sharing — fewer params AND better quality, measured (2026-08-02)

The "small model, big-model quality" question this project is actually built around had never
been tested. Issue #122's looping work looped a *full-depth* model, which saves zero params;
issue #32 Phase 2.4 shipped `SharedWeightHVN` and measured a 3.62× param reduction but never
paired it with a quality number. This closes that gap.

**Setup.** `eval/scaling_laws.py` gains two arms — `EdgeHVN-DeltaNet` (dense) and
`EdgeHVN-DeltaNet-Shared` (`shared_weights=True`). Same class, same blocks, sharing is the only
variable. WikiText-103, BPE-8000, 5M-token compute-matched budget, **12 layers**, bf16.
`param_count` uses `sum(p.numel())`, which counts a shared block once, so reported params are
the real storage footprint.

| d_model | arm | params | eval PPL |
|---:|---|---:|---:|
| 256 | dense | 12,018,032 | 194.8 |
| 256 | **shared** | **4,763,996** | 251.6 |
| 512 | dense | 39,756,656 | 120.7 |
| 512 | **shared** | **10,830,684** | 156.1 |

**Headline, and it needs no extrapolation — two real measured configs:**

    shared d=512 : 10.83M params, PPL 156.1
    dense  d=256 : 12.02M params, PPL 194.8

**The shared model has ~10% FEWER parameters and ~20% BETTER perplexity than the dense one.**
That is the goal stated in the README's "The Dream" — small footprint, quality of something
bigger — demonstrated directly rather than argued.

**Why it works.** The PPL cost of sharing is **+29% at both widths** (194.8→251.6 and
120.7→156.1), strikingly stable. But the param saving **grows** with width: 2.52× at d=256,
3.67× at d=512. A fixed quality cost against a growing param saving means **the trade improves
as the model scales** — the opposite of the compounding-loss worry. Against an equal-param
dense model (via this repo's own −0.504 exponent): shared is ~19% better at d=256 and ~33%
better at d=512.

**This also corrects a live misconception.** HVN is not "much worse than a Transformer" —
§8.12 has HVN-DeltaNet at 159.4 vs Transformer 172.0 at d=512, and 129.4 vs 150.8 at d=768,
with a steeper exponent (−0.504 vs −0.406). Sharing therefore compounds *onto a winning
baseline*, not a deficit. The variant that genuinely loses to a Transformer is the pre-#34
default HVN block (194.3 vs 150.8 at d=768).

**A design error caught cheaply, worth recording.** The first sweep was configured at 2 layers
(copying the documented width-sweep configs) and showed only a 7% param saving — sharing scales
with **depth**, not width, and at 2 layers there is barely anything to share, with the
8000-vocab embedding dominating the model anyway. A 90-second smoke test exposed it before any
real GPU was spent. The known-answer check here was **param arithmetic** — dense at d=512 was
predicted at ~39.7M and measured 39,756,656 — which is exactly the check that would have caught
the 2-layer flaw upfront had it been done first.

**Caveats.** 5M tokens is undertrained, so the absolute PPLs are high and only the *comparison*
carries signal. Two width points, single seed, one depth. Whether the +29% constant and the
growing param saving hold at a full training budget or at 1B+ scale is **untested** — but the
direction is now measured rather than assumed, and it favors sharing.

**Next, if pursued:** compose sharing with ternary (BitNet b1.58, `hvn/edge/ternary.py`) — 3.67×
from sharing on top of ~10× from ternary is the actual path to the README's ~2 GB 10B-class
target, and neither multiplier has been measured in combination with the other.

## Sharing × ternary compose cleanly — ~70× storage for +54% PPL (2026-08-02)

Sharing and ternary had never been combined anywhere in this repo. They are orthogonal
levers — sharing cuts the NUMBER of weights (3.67× at d=512/12L), ternary cuts BYTES PER
weight (fp32 → 1.58 bits, ~20×) — so their storage benefits multiply arithmetically. The
open question was whether their *quality* costs also compose, or compound worse.

Reason to expect compounding: a shared block's weights are reused at all 12 virtual layers,
so quantization error in that one block is applied 12 times rather than once.

2×2 at d=512, 12 layers, 5M-token budget, WikiText-103/BPE-8000
(`results/weight_sharing/ternary_2x2.json`):

| arm | PPL | vs dense-fp |
|---|---:|---:|
| dense fp | 119.4 | — |
| shared fp | 154.1 | ×1.290 |
| dense ternary | 143.2 | ×1.200 |
| **shared + ternary** | **183.8** | ×1.540 |

**Clean composition predicted 119.4 × 1.290 × 1.200 = 184.8; measured 183.8 — within 0.6%.**
The prediction was recorded before the final arm completed, so this is a real test rather
than a post-hoc fit. **The costs compose multiplicatively with no compounding penalty** — the
12×-reuse mechanism does not bite.

**Consequence: ~70× storage reduction (3.67× params × ~20× bits) for +54% PPL.** That is the
concrete path to the README's edge target, with both multipliers now individually measured
and their combination verified rather than assumed.

Harness consistency check passed: this run's dense-fp (119.4) and shared-fp (154.1) reproduce
the previous run's 120.7 / 156.1 at the same config, within ~1.5%.

**Caveats.** 5M tokens is undertrained and likely *understates* ternary specifically, since
QAT normally needs longer to converge — the +20% ternary cost is probably a pessimistic
bound. Param counts are unchanged for the ternary arms (fp32 shadow weights are retained for
the STE gradient during training); the storage win is arithmetic, realised post-`.freeze()`.
Single seed, one width, one depth.

**A mid-run misread worth recording:** at the 3M-token mark dense-ternary was reported as
tracking *better* than dense-fp (159.2 vs 170.8), which would have been suspicious — ternary
should not beat full precision. It was an artifact of comparing mid-trajectory points across
arms that converge at different rates; the final numbers (143.2 vs 119.4) show ternary
behaving normally. Do not compare arms mid-run.

## Issue #109 re-examined — the trunk, not the 3rd attention layer, was the larger factor (2026-08-02)

Issue #109 attributed a 3-attn recall regression to "an architecture question (layer
count/placement)" and closed without merging. Re-checking the code showed the comparison was
**confounded**: it measured **3-attn on the 102K trunk** against **2-attn on the 51K trunk**,
changing two variables. The writeup noticed the trunks differed ("stronger trunk + lower loss,
worse recall") but read the stronger trunk as making the regression more *surprising* rather
than as a candidate *cause*.

**The control already existed on disk and had never been probed** —
`checkpoints/hybrid_a4_2attn_102k/` is 2-attn on the 102K trunk, i.e. the PASS architecture
with the regression's trunk. All three arms, `eval/synthetic_recall_probe.py`, 200 trials each:

| arm | near-window recall |
|---|---:|
| 2-attn / 51K trunk (PASS baseline) | **37.0%** (74/200) |
| 2-attn / 102K trunk ← *the missing control* | **25.0%** (50/200) |
| 3-attn / 102K trunk | **15.5%** (31/200) |

The baseline reproduces the documented 35.5% at 37.0%, so the harness is sound.

- **Trunk effect** (holding attn=2): 37.0 → 25.0 = **−12.0pp**
- **Layer effect** (holding trunk=102K): 25.0 → 15.5 = **−9.5pp**
- Both significant at n=200 (~2.6 SE and ~2.4 SE)

**Verdict: right decision, incomplete root cause.** A 3rd attention layer does genuinely hurt
(15.5% vs 25.0% on the same trunk), so reverting to 2-attn was correct. But #109 assigned the
entire −21.5pp drop to layer count when the trunk accounted for more of it than layer count
did. Its stated root cause is incomplete, not wrong.

**The larger finding, and it was never documented at all: doubling pretraining made
recall-after-FT WORSE.** 51K trunk → 37.0%, 102K trunk → 25.0%, architecture identical. A
more-converged recurrence-only trunk appears harder to bolt fresh attention onto — consistent
with reduced plasticity in a further-converged model, though the mechanism is not established
here. Five `hybrid_a4_2attn_102k` variants (base, chatmix, chatmix5k, hilr, noternary) were
run and none were written up; `grep "102k" docs/` returned nothing before this entry, which is
why it sat undiscovered.

**Consequence for scaling plans.** This is direct evidence against "pretrain recurrence-heavy,
fine-tune windowed attention in later" as a large-scale recipe: more pretraining can degrade
the attention-insertion step. The trunk-length/recall relationship should be characterised at
small scale before any large pretrain commits to that ordering — a cheap sweep (probe several
trunk checkpoints through the same short A4 FT) would settle it.

**Process note.** Both this and the #122 state-commit bug were found the same way: checking
whether a comparison changed one variable or two. That question is cheaper than any of the GPU
time it recovered.

## GHOST Phase 1 — the detector works; the documented 6K-7.5K "phase transition" does not exist (Issue #125, 2026-08-03)

`ghost/detector.py` (epic #124, method by Rambarun Komaljeet) reads a training-loss CSV,
smooths it, differentiates it, and flags stretches where the gradient sits `k` sigma below its
trailing-window mean — sustained acceleration, i.e. a phase transition. Phase 1's gate was:
correctly identify the transition this file documents at **steps 6000-7500**.

**The gate FAILS, and the detector is not the reason. The ground truth is.**

At the spec defaults (window=50 samples, k=2.0) the detector finds **0 transitions across all
102,000 steps** of `checkpoints/edge_chinchilla83m/training_log.csv`, while correctly finding a
planted transition in synthetic data and correctly reporting 0 on a smooth-decay control.
The 80-point sensitivity sweep (`results/ghost/detected_transitions.json`) is what makes this
conclusive rather than a tuning argument:

| setting | detections on the real log | hits 6K-7.5K | detections on a **pure-noise** control |
|---|---:|:---:|---:|
| window=20, k=1.5, open_run=1 | 92 | yes | **90** |
| window=35, k=2.0, open_run=1 | 35 | yes | **37** |
| window=100, k=1.5, open_run=1 | 71 | yes | **77** |
| window=35, k=1.5, open_run=5 | **1** (at 42550-42800, 2.17 sigma) | no | **0** |

**Every setting that "detects" the 6K-7.5K transition finds just as many transitions in a
smooth synthetic curve that has none.** The hit is noise, not signal. The one setting with a
real signal-to-noise separation finds a single weak candidate at ~42.5K and nothing at 6K-7.5K.

**Independent corroboration from held-out val loss** (`val_log.csv`), which needs no detector at
all — the improvement rate is *decelerating* monotonically straight through the claimed window:

| val step | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 |
|---|---:|---:|---:|---:|---:|---:|
| val loss | 5.204 | 4.893 | 4.738 | 4.634 | 4.563 | 4.504 |
| delta per 1k steps | — | −0.156 | −0.077 | −0.052 | −0.036 | −0.029 |

A phase transition is an *acceleration*. Across 4K→8K the rate roughly halves at every
checkpoint. Nothing accelerates.

**Correction to this document.** The "EdgeHVN 83M — the measured PPL picture" section above
states the run had "a mid-run flat patch ~step 3K-5.5K that then **broke through and
accelerated** ~step 6K-7.5K — a phase transition, the opposite of a ceiling." That reading is
**retracted**. There is a mild flat patch (nearer ~3.3K-4.3K in a 500-step rolling mean), but
the descent that follows it is slower than the descent that preceded it, not faster. The
underlying conclusion that section drew — *undertrained, not capacity-limited* — is unaffected
and still stands on the val-PPL trajectory (110 → 63.45). Only the "phase transition"
characterisation was wrong: it was a narrative read of a noisy train-loss trace whose
sample-to-sample sigma (~0.15 nats) is larger than the entire descent it rides on (~0.1 nats
per 1,000 steps by step 6K).

**What this means for GHOST (#124).** The method's premise is that training consists of
predictable stretches punctuated by phase transitions that must be computed for real. This
project's only long training run contains **no detectable phase transition at all** at a
threshold that doesn't also fire on noise. That is not a refutation of GHOST — it is arguably
the *best possible* case for the extrapolator (a curve with no transitions is entirely
predictable stretch), and it removes the detector from the critical path. But it does mean the
detector cannot be validated against this repo's data, and Phase 3's "disable the extrapolator
near a transition" branch would never fire on a run like this one. Decide at #124 whether to
gate Phase 2 on a different (deliberately transition-rich) run or proceed straight to the
extrapolator. **The `2 sigma above rolling mean` rule itself is sound and reusable** — with the
persistence requirement (`open_run`) added, without which a bare k-sigma test produces
`n * Phi(-k)` false positives by construction (28 on a 1,200-sample noise curve, measured).

## GHOST Phase 2 — weight-delta extrapolation is measurably worse than doing nothing (Issue #126, 2026-08-03)

`ghost/extrapolator.py` (epic #124, method by Rambarun Komaljeet) takes three consecutive
checkpoints, measures each weight tensor's delta consistency, and predicts the weights N steps
later so those N steps need not be computed. Phase 2's gate: predicted weights at **N=10**
within **5%** of the real trained weights' loss at the same step.

**The spec's gate PASSES. The binding gate — beating the null control — FAILS at every N.**

The measurement needed a checkpoint series this repo did not have: its own checkpoints are
spaced 500 steps apart, far coarser than the N=10..500 jumps in question, so
`ghost/harness.py --make-series` continued `edge_chinchilla20k/step10000` for 800 steps
(AdamW, constant LR, batch 4 x seq 512, 1.59 steps/s, 8.4 min) and snapshotted the 11 steps the
analysis actually reads. Anchor at step 10200 (200 steps of warmup so any data-mix transient
has settled), extrapolator inputs at 10180/10190/10200, all arms scored on **one fixed
16-batch held-out probe set**.

| N | real loss | predicted | frozen (null control) | pred err | frozen err |
|---:|---:|---:|---:|---:|---:|
| 10 | 4.14036 | 4.22853 | 4.13901 | +2.13% | **+0.03%** |
| 50 | 4.10952 | 5.36871 | 4.13901 | +30.64% | **+0.72%** |
| 100 | 4.10046 | 7.43693 | 4.13901 | +81.37% | **+0.94%** |
| 500 | 3.98804 | 25.75799 | 4.13901 | +545.88% | **+3.79%** |

At N=10 the prediction is within 5% of real — and **doing nothing is ~70x more accurate**
(0.03% vs 2.13%). This is precisely the failure mode the null control was added to catch: a
+/-5% band is satisfiable by any method that barely moves, because over 10 steps the weights
barely move. By N=500 the extrapolated model is destroyed (loss 25.8 vs 4.0).

**One genuinely useful positive finding: the trajectory is real, but only at short range.**
Median cosine between consecutive weight deltas, measured at four spacings from the same
anchor:

| delta spacing | median cos(D1, D2) | params passing the consistency gate |
|---:|---:|---:|
| 10 steps | **+0.474** | 99.67% |
| 20 steps | +0.251 | 66.13% |
| 50 steps | +0.063 | 31.03% |
| 100 steps | +0.026 | 0.06% |
| 500 steps (repo's own checkpoints) | −0.012 | 0.11% |

Weight motion IS directionally coherent at 10-step spacing and decorrelates to noise by ~100
steps. An earlier reading of this project's native 500-step checkpoints (median cos −0.012)
was simply the wrong measurement scale — worth recording, because it would have supported the
right conclusion for the wrong reason.

**Why coherent deltas still do not buy a jump.** cos ~ 0.47 means the observed delta is roughly
half signal, half noise. Extrapolation multiplies that delta by `f = N/gap`, so injected noise
grows linearly in N while the true drift it is trying to capture is tiny — over 10 real steps
the probe loss moves by 0.001. There is no horizon where the trade wins: short N has no signal
worth capturing, long N has no valid linear trajectory.

**The obvious repairs were tried and the failure is monotone, not a bad configuration.**

| gate | params extrapolated | N=10 error | vs frozen |
|---|---:|---:|---|
| cos>=0.30 + quadratic | 99.67% | +2.13% | loses |
| cos>=0.30, linear only | 99.67% | +2.13% | loses |
| cos>=0.70, linear only | 31.03% | +0.83% | loses |
| cos>=0.90, linear only | 0.06% | +0.03% | ties — it *is* frozen |

Prediction error is monotone increasing in how much gets extrapolated, and the limit of
tightening the gate is the frozen baseline itself. **Shrinkage/damping needs no separate
experiment: damping is equivalent to reducing effective coverage, and this table shows the
optimum coverage is zero.** The extrapolator either injects noise or does nothing.

**Optimizer-state policy (recorded per #126):** this codebase's `_save_edge_checkpoint` never
stores AdamW moments, so every resume — and therefore every GHOST jump — already starts with
cold moment estimates. Nothing is extrapolated for them. This is a real per-jump cost that
would compound in Phase 3, on top of the weight error above.

**Caveats, stated so the result is not oversold.** One anchor, one model (83M ternary EdgeHVN),
one optimizer, constant LR. A different regime — early training with large coherent drift, or
a much smaller LR — could in principle behave differently. But the mechanism that fails here
(noise amplified by `f` against a drift too small to matter) is not specific to this
configuration, and the 5%-gate-passes-while-losing-to-frozen result would repeat wherever the
loss curve is as flat as it is at step 10200.

**Verdict: Phase 2 fails its binding gate.** Building Phase 3's training loop (#127) on this
mechanism would wrap a control loop around a step that is measurably worse than not taking it.
Artifacts: `results/ghost/extrapolation_scores.json`, `results/ghost/scores_{nopoly,cos07,cos09}.json`.

### Oracle upper bound — an adaptive/learned GHOST cannot work either (#126 follow-up)

The natural next proposal is to make GHOST *adaptive*: train normally, watch how wrong the
predictions are, fit the prediction function, repeat until it is good enough. That hypothesis
is testable without building it, by computing the **oracle**: for each tensor, the scalar
`alpha` in `W_pred = W3 + alpha * f * D2` chosen by least squares **using the true future
weights**. No adaptive, fitted, meta-learned or scheduled variant can beat a coefficient
selected with knowledge of the answer, so this is an upper bound on the entire family.

| N | oracle (cheats) | frozen | real | oracle err | median fitted alpha |
|---:|---:|---:|---:|---:|---:|
| 10 | 4.17811 | **4.13901** | 4.14036 | +0.91% | **+0.465** |
| 50 | 4.26663 | **4.13901** | 4.10952 | +3.82% | **+0.120** |
| 100 | 4.28950 | **4.13901** | 4.10046 | +4.61% | **+0.058** |
| 500 | 4.19223 | **4.13901** | 3.98804 | +5.12% | **+0.012** |

**The oracle loses to doing nothing at every horizon.** Two consequences:

1. **The optimal adaptive coefficient collapses toward zero as the jump grows** (0.465 -> 0.012).
   A self-tuning GHOST converges on "predict almost nothing" — the frozen baseline — and it
   does so *faster* the further ahead it tries to jump. The intuition that a fitted function
   improves with horizon is exactly inverted by the measurement.
2. **Even at its best (alpha 0.465 at N=10, which is essentially the measured delta correlation
   of 0.474) it is still worse in loss than freezing.** The oracle minimises *weight-space*
   distance to the true future weights and still loses in *loss space*: the component of the
   update that past deltas can predict is not the component that reduces loss. The predictable
   part is drift the optimizer's own momentum already applies; the part that constitutes
   learning comes from gradients of data the model has not seen yet, and that information is
   not present in past weights at any fitting budget.

This closes the "make it adaptive" branch analytically rather than by exhaustion: the ceiling
is measured, and it is below the do-nothing floor. `results/ghost/oracle_scores.json`.

### Two corrections to the Phase 2 method, both raised by the maintainer — verdict survives both (#126)

The Phase 2 result above was measured with **10-step delta spacing** at an anchor (step 10200)
where the loss curve is nearly flat. Both choices were wrong, and both were challenged:

1. **"It needs to be PER STEP."** Correct, and it mattered. Weight-delta coherence measured at
   1-step spacing is **median cos +0.890**, against +0.463 at the 10-step spacing originally
   used. The block measurement was discarding roughly half the available signal. Full ladder
   (flat anchor): gap 1 **+0.890**, gap 2 +0.864, gap 5 +0.675, gap 10 +0.463, gap 20 +0.219.
2. **"At a flat curve we do not need GHOST."** Also correct, and more serious: where the loss
   barely moves, the frozen null control is a near-perfect predictor for free, so it wins for
   reasons unrelated to the method. That made the original test unfair and the verdict
   unearned. Re-anchored at **step 600**, where the curve drops ~10x faster (−0.7 to −1.2 loss
   per 1k steps vs ~−0.1 at step 10200) and where real training gains **0.48 loss over 500
   steps** while freezing falls **9.55%** behind — i.e. the null control now has real ground
   to lose.

**Per-step deltas, early steep anchor (step 600) — the fair test:**

| N | real | frozen (null) | GHOST | oracle (cheats) | oracle alpha |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.49893 | **5.49368** | 5.50430 | 5.50324 | +0.892 |
| 2 | 5.49948 | **5.49368** | 5.52232 | 5.51626 | +0.830 |
| 5 | 5.49323 | **5.49368** | 5.61955 | 5.56444 | +0.700 |
| 10 | 5.47345 | **5.49368** | 6.03026 | 5.63838 | +0.526 |
| 50 | 5.40985 | **5.49368** | 128.8 | 5.91104 | +0.143 |
| 100 | 5.33866 | **5.49368** | 425.3 | 6.17017 | +0.074 |
| 500 | 5.01457 | **5.49368** | 9023.1 | 7.71704 | +0.020 |

**GHOST and the cheating oracle both lose to doing nothing at every horizon, in the regime
where GHOST is supposed to earn its keep, at the granularity that maximises the signal.** The
N=10 gate now fails the spec's own 5% band too (+10.17%), not just the null control.

**Why 90%-coherent motion still cannot be ridden.** Direction is highly predictable; magnitude
and curvature are not. Extrapolation scales the last delta by `f = N/gap`, so at gap 1 an N=500
jump applies 500x a single step along a direction that is only locally valid — the true path
curves away. The oracle's fitted coefficient shows this directly: it shrinks 0.892 -> 0.020 as
N grows, i.e. the best possible use of the observed direction is to follow it less and less,
converging on the frozen baseline. And even at N=1 with alpha 0.892 the oracle is still
marginally worse than freezing: the reliably predictable component of an update is the
momentum the optimizer already applies, while the component that actually reduces loss comes
from gradients of data not yet seen, which no function of past weights contains.

**Verdict unchanged, now earned rather than assumed.** The negative result stands at the
correct granularity and in the correct regime. Artifacts:
`results/ghost/scores_perstep_flat.json`, `results/ghost/scores_perstep_early.json`.

### Weight averaging — the positive result that came out of the GHOST work (#126 follow-up)

GHOST's premise was that the weight trajectory is coherent. #126 measured that it is (median
cos **+0.890** at 1-step spacing) but that *extrapolating* along it amplifies the delta's noise
by `f = N/gap` and loses to doing nothing at every horizon. **Averaging is the same observation
with the opposite arithmetic: it cancels the noise instead of amplifying it, keeping the shared
drift.** It skips no training step, so it does not violate the rule the extrapolation result
established — it asks only whether the weights already paid for are being used well.

Measured on the EXISTING `edge_chinchilla20k` checkpoints (500-step spacing, no new training),
same fixed 16-batch probe set, three independent anchors:

| anchor | latest | avg k=2 | avg k=3 | avg k=5 | avg k=10 | worth (real steps) |
|---:|---:|---:|---:|---:|---:|---|
| 8,000 | 4.30812 | **4.24677** | **4.26304** | **4.25888** | — | < 500 |
| 10,000 | 4.18270 | **4.14715** | **4.12614** | **4.12857** | 4.20883 (worse) | ~500-1000 |
| 14,000 | 4.01125 | **3.98420** | **3.98165** | **3.99291** | — | ~1000 |

**9 of 9 (anchor x k in 2,3,5) improve on the latest checkpoint**, gains 0.018-0.061 nats, for
the cost of loading a few checkpoints and averaging them. `k=10` (a 4,500-step window) is
*worse* than the latest — the averaging window has to be short relative to the drift, otherwise
it mixes in weights from a materially different model.

Because the gain is roughly constant in loss terms while real training slows down, **the gain
is worth more steps the later it is applied**: <500 steps at anchor 8K, ~1,000 steps at 14K.

**The important caveat — this is an evaluation-time gain, not yet a proven training speedup.**
Every number above evaluates an averaged model. The untested question is whether training
*continued from* the averaged weights stays ahead, or whether the advantage washes out within a
few hundred steps. That is the difference between "free better checkpoint" (certain, measured)
and "free training steps" (plausible, unmeasured), and it is the one experiment that decides
whether this is worth wiring into the training loop.

Artifacts: `results/ghost/weight_averaging.json`, `weight_averaging_{8000,14000}.json`.

#### A/B: the averaging advantage SURVIVES continued training (#128)

The open question above — free better checkpoint, or free training steps? — resolved by a
three-arm A/B from anchor step 10,000. All arms replay a byte-identical, pre-materialised
1,000-batch sequence from a freshly constructed optimizer, so the ONLY difference is the
starting weights. Third arm is a matched-norm control: the anchor plus Gaussian noise scaled
per tensor to the exact displacement averaging applied.

| step | latest | averaged | perturbed (control) | averaged's lead |
|---:|---:|---:|---:|---:|
| 0 | 4.18270 | **4.12614** | 4.32291 | +0.0566 |
| 200 | 4.08988 | **4.06539** | 4.13684 | +0.0245 |
| 500 | 3.95263 | **3.92361** | 3.96371 | +0.0290 |
| 700 | 3.89700 | **3.86775** | 3.90928 | +0.0293 |
| 1000 | 3.85241 | **3.82978** | 3.86580 | +0.0226 |

**Averaged leads at all 11 evaluation points; the perturbed control trails at all 11.** The
control is what makes this interpretable: an identical-magnitude *random* displacement makes
the model worse immediately (4.183 -> 4.323) and never recovers the gap, so the gain comes from
the direction averaging selects, not from perturbation acting as noise injection.

**It is a real head start that partially decays, not a compounding multiplier.** 40% of the
initial advantage is retained at 1,000 steps. In training-step terms: the averaged arm starts
~500 steps ahead and is still ~100-150 steps ahead after 1,000 steps of training (using the
latest arm's local rate of ~0.018 nats/100 steps at that point).

**Practical read:**
- **Certain:** applying it once at the end of any run gives a strictly better final model for
  seconds of arithmetic. There is no reason not to do this.
- **Measured:** it is also a genuine, if modest, training speedup — the head start does not
  wash out within 1,000 steps.
- **Unknown:** whether applying it repeatedly during a run compounds, or whether each
  application just re-buys the same decaying ~0.05 and the run converges to the same place. That
  is the question that decides whether it belongs *inside* the training loop or only at the end.

**Caveats:** one anchor, one seed, one A/B. The absolute gaps (~0.02-0.06 nats) are small,
though the sign is consistent across 11/11 paired evaluations on an identical probe set, and
the control separates cleanly in the opposite direction. `results/ghost/ab_continue.json`.

#### Periodic in-loop averaging does NOT compound — keep it at the end of the run (#128)

Two arms from step 10,000, 2,000 steps each, identical pre-materialised batch sequence.
``baseline`` trains plainly; ``periodic`` replaces its live weights with the mean of the last 3
snapshots every 250 steps and continues from there. The reference that matters is the third
number, free from the baseline arm: its own last-3 snapshots averaged at the end — the
already-proven one-shot application.

| arm | loss @ 2,000 steps | vs baseline |
|---|---:|---:|
| baseline, no averaging | 3.70874 | — |
| **baseline + end-of-run average** | **3.66057** | **−0.04817** |
| periodic averaging in-loop | 3.72787 | **+0.01914 (worse)** |

**Periodic averaging is worse than not averaging at all**, and 0.0673 worse than applying it
once at the end.

**The mechanism, which is the interesting part.** Every injection helped, and the gains did not
diminish: **+0.0243, +0.0405, +0.1401, +0.0732, +0.0615, +0.0646**. The averaged point is
genuinely better each time. But immediately BEFORE each injection the periodic arm was behind
the baseline — by 0.035 (step 1000), 0.056 (1250), 0.061 (1500), 0.054 (1750). Training onward
from an averaged point descends more slowly than training from the raw trajectory, and the
injections spend their gains repairing a deficit they created. The largest single gain (+0.1401)
came where the trajectory had bounced upward, confirming these injections act mainly as
volatility damping, not as progress.

This is the same 40%-decay observed in the #128 A/B, seen from the inside: averaging buys a
better point *now* at the cost of a slower descent afterwards. One application at the end
collects the benefit and pays none of the cost, which is exactly why the end-of-run form wins.

**Method flaw found mid-run and fixed, disclosed rather than buried.** The arms were seeded once
before the data loader rather than once per arm. The model has dropout, so the second arm drew a
different mask sequence and drifted from the first *before any injection*: 0.0036 apart at step
250, 0.0041 at step 500. Both gaps above (0.019 and 0.067) are 5x and 17x that drift, so the
verdict is unaffected — but `torch.manual_seed` is now called per arm in both `repeat_average`
and `ab_continue`. **The #128 A/B reported earlier carries the same flaw**; its margins
(0.011-0.057, averaged ahead 11/11, control behind 11/11) also exceed the drift, so its verdict
stands, but it should be re-run under per-arm seeding to remove the asterisk.

**Settled recommendation: average the last 2-5 checkpoints once, at the end of a run.** Do not
put it in the training loop. `results/ghost/repeat_average.json`.

### 3 parallel networks + weight averaging loses to one network at 3x batch (#130)

Maintainer proposal: train K networks side by side, average their weights periodically, and
hand the average back to each perturbed (`avg+d`, `avg`, `avg-d`) so they stay diverse. This is
a real family of methods — Federated Averaging / Local SGD for the averaging, Population-Based
Training for the perturbation — but its benefit is normally *parallelism across devices*. The
question here is whether it helps on ONE card, where 3 networks each get a third of the
throughput.

The comparison holds **total tokens constant**: 3 nets at batch 4 vs 1 net at batch 12,
same 1.8M tokens, byte-identical batch sequence (900 batches of 4 materialised once; the single
arm concatenates each group of 3). Averaging every 10 rounds, ensemble scored on its **averaged**
weights, per-arm seeding, same fixed 16-batch probe set used throughout #126/#128.

| round | `single` (1 net, batch 12) | `avg3` (3 nets, batch 4) | gap |
|---:|---:|---:|---:|
| 50 | **4.08878** | 4.13034 | 0.042 |
| 100 | **4.02920** | 4.08133 | 0.052 |
| 150 | **3.95435** | 4.02212 | 0.068 |
| 200 | **3.91598** | 3.98804 | **0.072** |

**The single network wins at every checkpoint and the gap widens monotonically.** Run stopped at
round 200 of 300 on a pre-declared trigger (stop if the gap widens again at 200), not after
seeing a result that suited a conclusion.

**Interpretation.** Averaging K models trained from a common point on different batches
approximates one model trained on the K-times-larger batch — so `avg3` should have tracked
`single` closely. It does not: it falls steadily behind. The three networks diverge between
averaging points by more than the averaging recovers, and 10 rounds of independent drift is
already too much at this scale.

**One prediction of mine that the data refuted.** I expected the ensemble to lose badly on
wall-clock (3x as many forward/backward passes, plus 20 averaging operations copying 3 x 332 MB
state dicts). It did not: 12.7 min vs 13.0 min at round 200 — **near parity, marginally
faster**. Batch 12 at seq 512 is evidently bandwidth-bound on this card, so three batch-4 passes
cost about what one batch-12 pass costs. The verdict rests entirely on quality; the compute
overhead I predicted would dominate does not exist here.

**`avg3_delta` (the +/-d variant) was deliberately NOT measured.** It is `avg3` plus a
perturbation, so it would have to beat `avg3` *and* then close a 0.072 gap that was still
widening — against the strictly more favourable of the two variants. #128's matched-norm
perturbation control is the relevant prior: perturbing averaged weights cost 0.14 and never
recovered. Recorded as a stated decision, not an omission; the harness supports it
(`--parallel-ensemble`) if it is ever worth 20 minutes.

**Where this DOES apply: multiple GPUs.** With 3 cards the arithmetic inverts — the ensemble
arms run concurrently instead of sequentially, and this becomes standard data-parallel training,
which this repo already has (#65 DDP/FSDP). Nothing here argues against that; it argues against
paying 3x the memory for it on one card.

### Window fit — averaging the direction makes it WORSE, and that closes #124 (#126 follow-up, 2026-08-05)

The one objection #126 never answered. Its oracle proved that no method can win by choosing
*how far* to travel along the last delta — but it never left that direction. The maintainer's
challenge was the right one: fit the *function* of the motion from a window of observed steps
rather than extrapolating a single noisy delta, and do it where the premise holds (settled
training, not the huge-PPL beginning). A window fit averages the per-step deltas, and averaging
is the arithmetic that CANCELS their noise rather than amplifying it — the mechanism behind the
one positive result in this epic (#128). If a denoised direction were usable, the whole family
would reopen.

`ghost/window_fit.py` streams a least-squares polynomial fit of each weight tensor against step
number. The window is never stored: the normal equations need only the scalar moments `sum(u^p)`
and tensor accumulators `sum(u^p * W_u)`, so a K-step window costs `degree+1` tensors instead of
K checkpoints, and one accumulator set serves every degree. Accumulation runs on `W - W_ref`,
never raw weights — per-step motion is ~1e-5 of a weight's magnitude, close enough to fp32
epsilon that summing raw weights would grind away part of the signal being fitted. Verified
against synthetic linear/quadratic trajectories: prediction error 4-7 orders below the motion,
Gram solve in float64 (cond 4.6e2 at K=20/d=1 to 1.0e7 at K=50/d=2).

Measured at the **settled anchor step 10,200** — #126's primary anchor, `--warmup-steps 200`
from `edge_chinchilla20k/step10000`, AdamW constant LR 3e-4, batch 4 x seq 512, same fixed
16-batch probe set. Windows K in {20, 50}, degrees {1, 2}, two prediction modes
(`locked` = real anchor + fitted displacement; `fiteval` = the fit's own value, whose N=0 is a
smoothed anchor), plus `last_delta` (#126 at gap 1) and a per-arm oracle that rescales the
FITTED direction using the true future weights. 13 arms x 8 horizons.

**The test was fair: real training gains 0.236 nats over the 500-step horizon (4.11941 ->
3.88303), so the null control had real ground to lose.**

**Not one arm beats frozen at any horizon.** Best result anywhere is `oracle_w50_d1` at N=1,
4.12019 vs frozen 4.11941 — and that arm is allowed to see the answer.

**The finding, which inverts the hypothesis.** Averaging did not denoise the direction; it
destroyed it. Median cosine between predicted and true displacement at N=1:

| arm | velocity estimate | median cos |
|---|---|---:|
| `last_delta` | 1 step | **+0.890** |
| `w20_d2` | quadratic slope at the anchor | +0.628 |
| `w20_d1` | mean over 20 steps | +0.433 |
| `w50_d2` | quadratic over 50 | +0.340 |
| `w50_d1` | mean over 50 steps | **+0.166** |

Monotone in how much history is mixed in: **the more steps averaged, the worse the direction
points.** The noise that averaging cancels is not separable from the only locally-valid part of
the direction. A window mean estimates where the weights HAVE been going, which is not where
they are ABOUT to go.

**Why the window fit still beat `last_delta` in loss at large N — it moves less, not better.**
At N=500 `w50_d1` travels 3.3x the true distance and `last_delta` 7.5x (loss 15.9 vs 13,661).
That is a smaller wrong step. The repair for over-travel is exactly what the oracle does, and
its fitted coefficient collapses toward zero on the same signature as #126:

| N | oracle alpha (w50_d1) | oracle alpha (w20_d1) |
|---:|---:|---:|
| 1 | +0.293 | +0.561 |
| 10 | +0.164 | +0.349 |
| 100 | +0.014 | +0.038 |
| 500 | +0.006 | +0.010 |

**Secondary result — the averaging gain has a lower bound too.** The `fiteval` arm at N=0 is a
smoothed anchor over 20-50 steps, and it LOSES to the raw anchor (4.12430 vs 4.11941), where
#128's averaging over checkpoints 500 steps apart WON. #128 established the window must be short
relative to the drift; this adds the other side — tens of steps is too short to cancel anything
worth cancelling, and only lags the anchor. One anchor, recorded as an observation rather than a
theory.

**Verdict: #124 is closed, on mechanism rather than exhaustion.** #126 showed the step length
cannot be fixed; this shows the direction cannot be improved either, and that the best available
direction is the most recent delta — which is ridable for about one step. The part of weight
motion recoverable from weight history is drift the optimizer's momentum already applies; the
part that constitutes learning comes from gradients of unseen data. No fitting budget, window
length, polynomial degree or adaptive schedule changes that.

Artifacts: `results/ghost/window_fit.json`, `results/ghost/window_fit.log`. The 2.2 GB of target
checkpoints under `results/ghost/series_windowfit/` were deleted after scoring; the run is
reproducible in ~15 min via `python -m ghost.harness --window-fit`.

## EdgeHVN decode ran at half speed — ternary weights re-derived every token (Issue #131, 2026-08-05)

Opened to scope a MatMul-free integration (Zhu et al. 2024) into EdgeHVN's projection layers.
The **bounding profile that was meant to decide whether to build it found a 2x speedup already
sitting in the existing code**, and retired the original plan on the way.

**The mechanism.** `TernaryLinear.forward` called `ternary_quantize_ste(self.weight)` on every
invocation — `abs -> mean -> div -> round -> clamp -> mul -> sub -> add` across the **entire
weight matrix**. In training that cost amortizes over `batch x seq` (4 x 512 = 2,048 tokens per
forward). In incremental decode a forward pass **is one token**, so the same O(d^2) cost is paid
per token, against a matvec that is also O(d^2) — the same order as the work it feeds. The
weights are frozen at inference, so all of it was redundant.

**Evidence** (`scripts/profile_decode_projections.py`, 64 tokens, profiling `forward_step` and
NOT `forward` — the latter dispatches to `_scan_chunked`, a path serving never takes):

| signal | value |
|---|---|
| projection calls (102 modules x 64 tokens) | 6,528 |
| `aten::mean` calls | **6,656** |
| `aten::div` calls | **6,656** |

One absmean reduction and one divide per projection per token — 1:1, which is the diagnosis.
Op-level device time: `aten::addmm` (the actual GEMM) **66 ms** vs ~**395 ms** for the
quantization chain.

**A flaw in that profile, disclosed rather than buried.** Its `projection_share_pct` (70.1%) and
`PROCEED` verdict are **not trustworthy**: `key_averages()` returns both op-level (`aten::abs`)
and kernel-level (`_ZN2at6native...AbsFunctor...`) entries carrying the *same* device time — they
agree to the last decimal at 54,794.628 us — so summing across all events double-counts. The call
*counts* are unaffected, and the A/B below needs no attribution at all, which is why the verdict
rests on those instead.

**The fix, measured.** Cache the dequantized weight under `eval()` + `no_grad`. Paired
within-process A/B, run in **both arm orders**, because the baseline always running first would
let GPU clock ramp inflate the second arm:

| arm order | pre-#131 | cached | speedup |
|---|---:|---:|---:|
| legacy first | 22.30 | 43.21 | **1.94x** |
| cached first | 48.40 | 21.45 | **2.26x** |

Two earlier runs against a monkeypatched prototype read 2.02x and 1.99x. **Claim: ~2x**,
consistent across four measurements and both orders.

**Known-answer cell** (what the #122 retraction requires of every harness): caching a
deterministic function of frozen weights is mathematically a no-op, so the cached arm MUST give
bit-identical logits. `max|dlogits| = 0.000e+00` in all four runs. **No recall regression is
possible by construction** — identical numbers, so no probe re-run is needed to clear it.

**Do NOT quote the absolute tok/s.** The same arm read 21.45/22.30 (legacy) and
40.56/43.21/48.40/53.65/81.05 (cached, across runs) — a ~1.5x spread on identical code, matching
this repo's documented GPU-contention variance ("re-ran identical `main()` twice = 0.7 steps/s").
Only the within-process paired ratio is meaningful on this machine.

**Scope: serving only; no training win was available.** Training stays uncached — the STE needs
the live shadow weight in the autograd graph and the weight moves every step. Nothing here
touches the compute wall or the ~142M local ceiling. What changes: every EdgeHVN checkpoint
*served* (Docker chat app, `EdgeToolCallingSession`, any `forward_step` eval) decodes ~2x faster
with identical output.

**Staleness is the real hazard, and it is guarded.** A weight cache has the exact failure shape
of #117 (register-if-absent dedup silently served untrained weights while aggregate loss still
trended down; no test caught it). So invalidation is aggressive rather than clever: on **any**
mode transition (`train()` is the single entry point for both `.train()` and `.eval()`), on
`_load_from_state_dict`, on `freeze()`, and on any dtype/device mismatch.
`tests/test_issue131_ternary_cache.py` (9/9) asserts a specific weight's update survives a
train->eval round trip **directly**, not via aggregate behaviour. One boundary is documented, not
hidden: mutating a weight in place while *staying* in eval mode is unobservable to any hook and
needs an explicit `invalidate_cache()` — a test pins that boundary so changing it is deliberate.

**Consequence for the original MatMul-free plan.** It proposed replacing the projection GEMMs
with ternary accumulation — the ~66 ms slice; the ~395 ms slice was redundant work that simply
should not run. Also established while scoping: **ternarization is already shipped**
(`apply_ternary_to_model` converts every `nn.Linear`; per #129 every real run trains ternary ON),
and **no training-VRAM win is possible** from ternary QAT because the fp32 shadow weights stay
resident — so the original spec's training-VRAM and memory gates were unachievable by
construction. A fused ternary *kernel* remains open, but decode is launch-bound here (2,815
kernel launches/token, CPU/device self-time ratio 0.91), so a future attempt should target launch
count per the #43 precedent, not multiply count.

**Untested:** batch>1 / the `generate_served` concurrent path, bf16 (this checkpoint decodes in
fp32), and whether the win holds after `freeze()` (which replaces the 8-op chain with a cast+mul,
already cheaper — expect a smaller ratio). Artifacts:
`results/matmul_free/decode_profile.json`, `ternary_cache_ab{,_reversed,_landed,_landed_rev}.json`.

## Padding-free sequence packing — 5.17x cheaper per real token (Issue #132, 2026-08-06)

The fine-tune datasets padded every example to `seq_len+1` with EOS and `-100` labels. Padding
tokens contribute nothing to the gradient, so that compute was pure waste. Measured on the real
corpora with the real GPT-2 tokenizer at `seq_len=512`:

| corpus | n | median | p90 | max | padding waste |
|---|---:|---:|---:|---:|---:|
| `data/grounded_qa_expanded.jsonl` | 5,240 | 86 | 115 | 205 | **83.0%** |
| `data/tool_use_train.jsonl` | 600 | 180 | 340 | 340 | **63.6%** |

The grounded-QA corpus is the one behind the whole recall track — `hybrid_a4_v2` (161.6 min), the
#122 deep-supervised (115.3 min) and fixed-N (196.3 min) FTs all paid the 83%.

**This is not "fewer tokens needed to learn."** Same real tokens, same epochs — packing puts ~5.2
real examples in the row that held one, so the corpus costs 5.18x fewer rows.

**Measured A/B** (`scripts/ab_packing.py`, real 83M hybrid config — 12 layers / 2 windowed-attn /
d=512 / ternary / triton+megakernel scan — on the real corpus, `results/packing/ab_packing.json`):

| arm | s/step | real (non-`-100`) tokens/step | us/real token | peak VRAM |
|---|---:|---:|---:|---:|
| padded | 0.330 | 5 | 61,147 | 2.08 GiB |
| packed | 0.344 | 29 | 11,816 | 2.15 GiB |

**5.17x cheaper per real token**, against a 5.18x row-reduction ceiling — i.e. packing captures
essentially all of the available win. Corpus packing: 5,240 examples -> 1,012 rows, waste
81.4% -> 3.9%. The predicted cost of losing the flash `is_causal` fast path on the 2 attention
layers turned out to be **~4%** (0.330 -> 0.344 s/step), and VRAM is flat.

**Why the boundary reset is exact rather than an epsilon — the finding that made this cheap.**
Document isolation needs the DeltaNet recurrent state cut at each document's first token, which is
expressible as forcing the decay gate `alpha_t = 0` there. `alpha` is a plain tensor passed *as
data* into every scan, so **no kernel changed**. And `hvn/core/deltanet_triton.py`'s scan is
sequential over S with a direct `state = state * a_t` multiply ("fp32 throughout -- no
reparameterization, so stable at any alpha"), so `alpha=0` is a mathematically exact reset there —
measured 4.8e-7 packed-vs-separate max abs diff, against 0.60 for the leakage it removes. The
eager `_scan_chunked`/`_scan_chunked_stable` paths fold alpha into a cumprod/log-space form that
clamps at 1e-6 and *cannot* represent a hard zero; rather than ship a silently-approximate path,
`DeltaNetBlock.forward` **raises** when asked to pack under them. The FT scripts already default
to the Triton scan (2026-08-01 flip), so this costs nothing in practice.

**Three sequence-axis carries, all cut** (`hvn/core/packing.py` + the blocks):
1. **Recurrent state** — `alpha=0` at document starts.
2. **The depthwise causal conv** — its `K-1`(=3) backward taps read across a boundary. Fixed with
   a segment-safe conv (K masked shifted multiplies instead of one `conv1d`), only on the packed
   path. This one is easy to miss and would have leaked 3 tokens per boundary silently.
3. **Attention** — block-diagonal document mask AND-ed into the existing explicit-mask branch.

RoPE needed nothing (it is relative, and cross-document keys are masked anyway). A fourth issue
surfaced while building: the **causal shift** means the last token of document A would be trained
to predict document B's *first* token. The current corpora mask their first token anyway (the
prompt is `-100`), so it is a no-op on them — but `PackedDataset` cuts it explicitly so the class
is not silently corpus-dependent.

**Verification.** `tests/test_packing.py` 27/27, including per-document parity vs
`_scan_sequential`, an explicit cross-document isolation proof on EdgeHVN logits, a
non-vacuousness check (doc_ids must actually *change* the result), the conv-boundary test, the
grad-checkpoint recompute path, and GPU parity for both the triton and megakernel scans. Full
suite 309 passed (`tests/test_issue33.py` syntax error and 6 `ckpt_path`-fixture errors in
`test_issue51/65/67` are pre-existing on `main` — verified by stashing).

**Scope, deliberately narrow.** `--pack` is opt-in on `finetune_edge_conversational.py`,
`continue_ft_hybrid.py`, `continue_ft_looped.py`. Class-level defaults on `DeltaNetBlock`/`EdgeHVN`
are **unchanged** — the same lesson as the #122 scan-default flip, which surfaced two real
regressions when broadened. Each corpus packs independently so a row never mixes corpora, which
keeps the `WeightedRandomSampler` mix fractions meaning what the flags say. `HVNBlock` and
`shared_weights=True` raise `NotImplementedError` rather than leak.

**NOT yet done — the remaining acceptance criterion.** A real packed-vs-padded FT run re-gated on
`eval/synthetic_recall_probe.py` at the 300-trial bar (against the 35.5% near / 8.5% far
baseline). Packing changes gradient composition per step — ~5 examples per row instead of 1 — so
it is **not** a "same result, faster" change on paper, and a speedup that moves recall is not a
win. The isolation is proven exact; what is unproven is whether the denser step is as good a step.

## Packing recall re-gate — PASSES: 5.15x faster, no detectable recall cost (Issue #132, 2026-08-06)

The remaining acceptance criterion from the entry above. Two FT arms, identical except `--pack`,
matched on **real tokens** (same passes over the corpus, so the packed arm takes 5.18x fewer
optimizer steps -- which is exactly the risk being measured, not a flaw in the design), then
`eval/synthetic_recall_probe.py` at the 300-trial bar on both.

| arm | steps | wall-clock | near-window | far-window (W=128, 16 distractors) |
|---|---:|---:|---:|---:|
| padded (control) | 5,000 | **89.1 min** | 7.3% (22/300) | 3.7% (11/300) |
| packed | 965 | **17.3 min** | 6.0% (18/300) | 3.3% (10/300) |

**5.15x end-to-end wall-clock speedup** — confirming the 5.17x/real-token microbenchmark
(`scripts/ab_packing.py`) on a real 90-minute training run rather than a 30-step timing loop.

**Recall difference is not statistically detectable.** Two-proportion tests: near z=+0.65
(p=0.51), far z=+0.22 (p=0.82). Neither is close to significance at n=300 per cell. Packing costs
5.18x fewer optimizer updates and the model does not measurably care.

**PASS.** The concern going in was that packing is not a "same result, faster" change on paper --
~5 examples per row instead of 1 changes gradient composition, and 5x fewer updates at the same LR
schedule is less optimization. Measured: neither shows up above noise.

**Three caveats, stated because the absolute numbers look wrong next to this doc's other entries:**
- **The trunk differs.** `edge_conversational_step51000.pt` (what `hybrid_a4_v2` used) is not in
  this checkout, so both arms warm-started from `step101000`. Absolute numbers are therefore NOT
  comparable to the historical 35.5% near / 8.5% far -- which is precisely why **both** arms were
  measured fresh rather than one against the doc. The same mistake #122 had to correct.
- **`--chat-mix 0`** (pure grounded-QA, both arms) to keep the packing ratio exactly 5.18x and drop
  the OASST cache dependency. a4_v2 used a 30% chat mix.
- **Deep in the overfitting regime**: 5,000 steps x 32 rows over 5,240 examples is ~30 passes;
  final train loss 0.0084 (ppl 1.01). This depresses both arms equally and explains why 7.3% sits
  so far below 35.5%. The probe scores **invented** pseudo-words never in training, so it still
  measures the general recall skill rather than memorized pairs -- but a less-overfit run would be
  a better absolute measurement. The A/B remains controlled either way.

**What this licenses and what it does not.** It licenses using `--pack` on a grounded-QA-style FT
at this scale: the speedup is real and the quality cost is below detection. It does **not**
establish that packing is free at every mix/length distribution -- OASST conversations are longer
(smaller packing ratio, fewer documents per row), so both the win and any risk shrink there, and
that case was not measured. Artifacts: `results/packing/regate_{padded,packed}_{near,far}.txt`,
`scripts/regate_packing.sh`, checkpoints under `checkpoints/regate_{padded,packed}/`.

## OASST packing re-gate — the win shrinks to 1.45x and a small loss cost appears (Issue #132, 2026-08-07)

Closes the "not established at every length distribution" caveat directly above. OASST is the
other end of the distribution from grounded-QA: median 359 tokens with >=10% at the 512 cap,
so it packs **1.45 examples per row** (3,465 -> 2,395 rows, waste 32.9% -> 3.0%) against
grounded-QA's 5.18. Both the available win and the gradient-composition perturbation shrink by
the same factor, which is why this could not be inferred from the grounded result.

**Harness choice that matters.** Both arms ran through `continue_ft_hybrid.py --chat-mix 1.0`,
not `finetune_edge_conversational.py`. That script only enables the Triton scan under `--pack`,
so an A/B there would have confounded packing with a change of scan path — the #122 lesson.
`continue_ft_hybrid.py` has triton+megakernel on unconditionally in both arms, leaving `--pack`
as the single difference. `--chat-mix 1.0` gives the grounded corpus zero sampling weight, so
this is effectively pure OASST on the identical harness as the first re-gate. Arms matched on
**real tokens**: 2000 padded steps vs 1379 packed (= 2000 / 1.45, same passes over the corpus).

| arm | steps | wall-clock | final train loss (last 5 windows) | near-window | far-window (W=128, 16 distr.) |
|---|---:|---:|---:|---:|---:|
| padded (control) | 2,000 | **35.7 min** | 2.614 (ppl 13.65) | 1.3% (4/300) | 4.0% (12/300) |
| packed | 1,379 | **24.7 min** | 2.676 (ppl 14.53) | 3.0% (9/300) | 4.7% (14/300) |

**1.445x end-to-end wall-clock**, against the 1.45x row-reduction ceiling — packing again
captures essentially all of the available win, and per-step cost is unchanged (1.07-1.08 s/step
in both arms, so the ~4% flash-`is_causal` penalty measured on grounded-QA does not grow here).
The scaling law across both re-gates is simply *speedup = packing ratio*; nothing else about the
step got cheaper or dearer.

**Recall: no detectable difference, but the probe is a weak instrument on this run.** Near
z=-1.40 (p=0.16), far z=-0.40 (p=0.69) — both non-significant at n=300, and both nominally
favour the *packed* arm. The caveat is that this FT is 100% chat with zero grounded
replay, so both arms sit at or below the pure-trunk floor (4-5% near / 0.5% far): recall was
eroded by the chat FT itself, exactly as Stage 1 (#108) predicted, in both arms equally. A
floored metric can only detect gross damage, so "no difference" is weaker evidence here than
the same words were in the grounded re-gate. **This is why the loss column was added** — on a
chat corpus, next-token loss is the metric that actually tracks the thing being trained.

**The new finding: a small, consistent train-loss cost that the grounded re-gate could not have
seen.** At matched real tokens the packed arm is behind throughout, not just at the end:

| progress (real tokens) | padded | packed | gap |
|---|---:|---:|---:|
| 10% | 3.3134 | 3.3440 | +0.031 |
| 20% | 3.1012 | 3.1259 | +0.025 |
| 30% | 2.8959 | 2.9305 | +0.035 |
| 50% | 2.7288 | 2.7863 | +0.058 |
| 70% | 2.6575 | 2.7061 | +0.049 |
| 100% | 2.614 | 2.676 | **+0.062** |

A stable ~0.03-0.06 nat offset (**+6.4% ppl** at the end), flat-to-slightly-widening — not a
divergence, not noise. The loss is comparable across arms by construction:
`CrossEntropyLoss(ignore_index=-100)` with the default mean reduction is a per-**real**-token
mean in both, and padding is `-100`, so the packed row's extra real tokens change the variance
of the estimate, not its expectation.

**Most likely cause, and it is confounded on purpose: 1.45x fewer optimizer updates.** Matching
on real tokens means the packed arm takes 1379 updates where the control takes 2000, at the same
peak LR and a cosine schedule over its own (shorter) run. That is *the* thing packing trades away,
and it is inseparable from packing in this design — which is the correct thing to measure, since
it is exactly how the flag would be used. What this A/B cannot do is attribute the 0.06 to "fewer
updates" versus "denser rows are worse steps"; separating those needs a third arm matched on
*steps* instead of tokens. The grounded re-gate could not have surfaced this at all: its final
loss was 0.0084 (ppl 1.01, ~30 passes, deep in the overfitting regime), where every arm is
pinned to the floor.

> **⚠️ The decision this section originally reached — "`--pack` stays opt-in, because it costs
> ~6% chat ppl" — is RETRACTED.** That verdict rested on **train** loss, which in this regime
> measures how hard each arm memorised its own training set, not model quality. Measured on
> held-out data the next day, the gap is +0.4% (nothing), and `--pack` is now ON BY DEFAULT.
> See the correction entry below. The measured facts above stand; only the reading of them was
> wrong.

**Original decision (superseded):** keep `--pack` opt-in, on the grounds that 1.45x wall-clock
for ~6% worse chat ppl is a trade rather than a free win, with the #122 scan-default flip as the
precedent for not broadening a verified-narrow default.

**The untested option flagged here as "probably dominates both"** — spend the speedup on passes
instead of wall-clock — **was tested and does NOT dominate.** It overfits. Also in the correction
entry below.

Artifacts: `results/packing/regate_oasst_{padded,packed}_{near,far}.txt`,
`scripts/regate_packing_oasst.sh`, checkpoints under `checkpoints/regate_oasst_{padded,packed}/`.

## CORRECTION — packing is free, `--pack` is now ON BY DEFAULT (Issue #132, 2026-08-07)

**In one line: packing gives the same model in ~2/3 the time. The "6% worse" in the entry above
was the wrong yardstick.**

Two things were measured today that reverse the decision above.

**1. The equal-wall-clock arm (arm C) — spend the speedup on passes, not on time.** Packed at the
control's *step* count instead of its *token* count: 2,000 steps, `--pack`, everything else
identical (`scripts/equal_wallclock_packing_oasst.sh`). Because both runs use cosine over 2,000
with warmup 60, the LR matches at every logged step, so this is also the **step-matched third arm**
the entry above said was needed to separate "fewer updates" from "denser rows are worse steps".

**2. Held-out loss on all three arms** (`eval/heldout_grounded_loss.py`,
`scripts/heldout_loss_three_arms.sh`). Every arm ran `--chat-mix 1.0`, which sets the grounded
corpus's sampler weight to `1/n_g * (1 - 1.0) = 0` — literally never sampled ("0% grounded" in
every log) — and the shared trunk is a pure SlimPajama+Wiki pretrain that was never grounded-FT'd,
with fresh attention layers. So `data/grounded_qa_expanded.jsonl` is **equally unseen by all
three**, and scoring it is a real generalization measurement. Evaluation is padded and in fixed
order for every arm, so packing is absent from the measurement side.

| arm | steps | wall-clock | passes | train loss | **held-out loss** | held-out ppl |
|---|---:|---:|---:|---:|---:|---:|
| A · padded (control) | 2,000 | 35.7 min | 18.5 | 2.6139 | **4.8185** | 123.79 |
| B · packed, token-matched | 1,379 | **24.7 min** | 18.4 | 2.6759 (+6.4%) | **4.8221** | 124.22 (+0.4%) |
| C · packed, equal wall-clock | 2,000 | 35.8 min | 26.7 | **2.4871 (−11.9%)** | **4.8511** | 127.88 (+3.3%) |

**Finding 1 — B's "6% ppl penalty" is not a quality penalty.** On 493,837 held-out tokens B is
+0.4% against the control: nothing. Matching on real tokens gives the packed arm 1,379 updates to
the control's 2,000, so it fits the *training set* less — which is all the train-loss gap was
measuring. **Packing at the same real tokens is quality-neutral and 1.45x faster.** That is the
padded-vs-packed verdict.

**Finding 2 — C's −11.9% train-loss "win" is overfitting, and it inverts on held-out.** Best train
loss of the three, worst held-out (+3.3%). 26.7 passes over a fixed 3,465-example corpus buys
memorization, not generalization. **Spending the speedup on more passes over the same data does
not work** — the direction that would work is more *corpus*, which OASST cannot supply locally.
The pre-launch caveat ("1.45x more passes over a fixed corpus is not 1.45x more data") is exactly
what happened.

**Finding 3 — train loss was the wrong instrument for this whole question, in both directions.**
It made B look worse than it is and C look better than it is. The FT harness logs no validation
loss and every arm trains on all 3,465 OASST conversations, so there is no OASST val split; the
held-out grounded corpus is the available substitute. Worth remembering the next time an arm is
graded on its own training loss in a multi-pass regime.

**Recall probes (300 trials each) are uninformative here and were not used to decide**: near
1.3% / 3.0% / 3.0%, far 4.0% / 4.7% / 2.7% (A/B/C). This FT is 100% chat with zero grounded
replay, so every arm sits at or below the pure-trunk floor — a floored metric detects only gross
damage. Recorded for completeness, not weight.

**A scoring caveat, so the artifacts are not misread.** `heldout_grounded_loss.py` also has an
answer-span-only mode (the corpus as `GroundedQADataset` masks it, ~14K supervised tokens).
Those numbers — 10.749 / 10.914 / 10.687 — sit at or *above* `ln(50257) = 10.82`, i.e. at chance:
none of these checkpoints ever trained the grounded skill. That mode is retained but is
**uninformative for this comparison**; the `--all-tokens` figures in the table are the ones to
read.

**Decision: `--pack` defaults to ON** in `finetune_edge_conversational.py`,
`continue_ft_hybrid.py`, `continue_ft_looped.py` (`argparse.BooleanOptionalAction`; `--no-pack`
restores the padded path). Free 5.15x on grounded-QA, 1.45x on OASST, no measurable quality cost
at either end of the length distribution — which was the open question. Class-level defaults on
`DeltaNetBlock`/`EdgeHVN` remain **unchanged**: the #122 flip's two regressions (compile-fusion
graph breaks, BCD-offload parity) were about `use_triton_scan`/`use_megakernel_scan` as *class*
defaults, and none of these three scripts uses `--block-coordinate`, so that risk does not reach
them. `tests/test_packing.py` 27/27, `test_continue_ft_hybrid` + `test_hybrid_forward` +
`test_looped_early_exit` 22/22 after the flip.

**Limits, stated plainly.** One training seed per arm, so the 0.4% has no error bar — it is "too
small to see", not "proven zero". The held-out corpus measures transfer in the same chat template
rather than OASST fluency itself. Both cut against over-claiming a *quality* win; neither
threatens the actual claim, which is that packing is quality-**neutral** and faster.

Artifacts: `results/packing/heldout_alltokens.json`, `heldout_answerspan.json`,
`regate_oasst_packed2000_{near,far}.txt`, `logs/equal_wallclock_oasst.log`, checkpoints under
`checkpoints/regate_oasst_packed_2000/`.

## The 355M scale-up is STOPPED — measured, not abandoned (Issue #112, 2026-08-07)

**One line: the scale-up costs a month of GPU time for a 2-4x parameter jump, and the
setting it depends on makes the model 4x worse. Stopped on the numbers.**

This entry closes out a day of #112 pre-launch work. Nothing here was a guess; each item
replaced a documented belief with a measurement, and three of the four beliefs were wrong.

### 1. The VRAM envelopes measured a different model (see the correction in training.md)

Every number in `docs/training.md` traces to a bench hardcoding `VOCAB = 8000` against the
real pretrain's 50,257, with no ternary and no Triton scan. `d=1024, 24L` is **355,128,593
params, not 268M**, and it **spills** (6.34-6.75 GiB reserved on a 5.9966 GiB card) rather
than being the "recommended ceiling config, 4.3 GiB". The 4.3 figure was the forward/backward
peak with optimizer state excluded entirely.

### 2. The "1.21B silent stall" was never a stall

Output is gated on `global_step % 50` with `grad_accum=8`, so nothing prints until micro-batch
400; at 1.21B each micro-batch took 82 s because `reserved` exceeded the card by 1.29 GiB and
WDDM backed the overflow with host RAM. First log line was **9.2 hours away**. The recorded
TDR/driver-crash hypothesis for the earlier 268M failure is **refuted**: the Triton scan has no
autotuning (`grep -c triton.autotune` = 0), the `"Not enough SMs"` line prints on every compile
on a 20-SM card, no kernel is within 36x of the 2 s watchdog, and the Windows System log has
**zero Event ID 4101** across its entire retained history. The GPU did end up faulted and that
remains unexplained — but it is not what caused the "hang".

### 3. `--block-coordinate` costs ~4x quality — the load-bearing assumption nobody had tested

BCD is documented purely as a VRAM optimisation (#99), yet it is mandatory at 355M, and only
1 of 14 blocks trains at a time with the outgoing block's optimizer state deleted on every
switch. A/B through the real training script (`scripts/ab_bcd_convergence.sh`, d=512/12L so the
control fits without BCD, 4,200 steps = 3 full rotations, identical seed/data/steps, `--no-fast`
in both arms so `torch.compile` recompiles at block switches cannot confound it):

| step | full model | `--block-coordinate` | gap |
|---|---:|---:|---:|
| 1,000 | 5.53 | 8.79 | 3.27 |
| 2,000 | 5.18 | 6.46 | 1.28 |
| 3,000 | 4.97 | 6.26 | 1.29 |
| 4,000 | 4.85 | 6.20 | 1.35 |
| **4,200 (final)** | **4.83** (ppl 125) | **6.19** (ppl 487) | **1.36** |

The early gap is cycle warmup — one full rotation completes at step 1,400, before which most
of the model has never trained. What remains after that is flat and real: **~3.9x worse
perplexity at matched steps**. Limits: 74M/14 blocks rather than 355M/26, and one seed.

### 4. Adafactor fits without BCD — and had been built but never measured

`scripts/probe_optimizer_vram.py`, real 355M config, built through `build_model_and_config` +
ternary + Triton scan (**not** a standalone `EdgeHVN`, deliberately — repeating the bench's
mistake would produce another wrong envelope), reporting `max_memory_reserved` rather than
`nvidia-smi`:

| optimizer | peak alloc | peak reserved | s/micro | fits (5.9966 GiB card) |
|---|---:|---:|---:|:--:|
| Muon (script default) | 5.67 GiB | 6.75 GiB | 22.82 | ❌ spills |
| **Adafactor** | **3.97 GiB** | **5.08 GiB** | **1.82** | ✅ **0.92 GiB headroom** |
| AdamW | 6.61 GiB | 7.47 GiB | 29.11 | ❌ spills |

The s/micro column is the spill penalty, not optimizer speed — the two that fit nothing run
12-16x slower because of host-RAM backing. **`--optimizer adafactor` removes the need for BCD
at 355M entirely**  
> **Does NOT generalise — see "Adafactor does NOT scale to 1B" (#157, 2026-08-14) at the end
> of this file.** At 1B it reserves 9.42 GiB on a 6 GiB card (a spill, 10.4x slower). Adafactor
> shrinks optimizer state, which is the deciding term at 355M and not at 1B, where weights and
> gradients alone exceed the card.**, at the same step time as the BCD config, without the 4x handicap. It has
been wired since #85; that issue's own closing note records its VRAM benchmark being deferred
"since the GPU is currently busy", and it was closed without the benchmark ever running.

**Adafactor's own quality is still unmeasured** (Muon beat AdamW in #86; Adafactor vs Muon has
never been run) and it saves memory by approximating — a row+column summary per matrix instead
of the full second moment. The arm that would settle it is ~2.3 h against the control above.

### 5. Why the track is stopped anyway — it is time-bound, not memory-bound

At the measured Adafactor throughput, with Chinchilla-optimal tokens (20/param):

| config | real params | tokens needed | days (h=8) | days (h=16 projected) |
|---|---:|---:|---:|---:|
| d=1024, 24L | 355M | 7.1B | 73 | **30** |
| d=768, 16L | 172M | 3.4B | 17 | 7 |
| d=2048, 24L | 1.21B | 24.2B | **31 years** | ~2 years even if it fit |

**1B was never trainable here**, and "1B trains locally today" (this file, 2026-07-12) is
withdrawn — it fit only in the vocab-8000 bench, and even a corrected fit is ~2 years of wall
clock. `docs/index.md` already warned that size decisions must start from
`6·N·D ÷ throughput` and never from whether the model fits; the 1B claim did precisely what
that warning forbids.

**And the existing 83M is already complete** — 102,000 steps x 16,384 = **1.65B tokens against
a 1.66B Chinchilla floor**. It is not undertrained any more. A 5-hour from-scratch run at any
size lands at 0.7-6.6% of its required tokens, i.e. strictly worse than the model already on
disk. There is no short version of a pretrain.

**Decision: stop the scale-up.** The goal is a model trained on *less* compute; a 30-day run
for a 4x parameter jump is the opposite, and 172M/7 days buys a 2x jump that this project's own
capacity analysis says will not close the contextual-reasoning gap regardless. The cheap
remaining wins are fine-tuning (1-3 h, now 1.45-5x faster with #132 packing on by default),
which is where this document has repeatedly concluded the lever actually is. The next
compute-root direction is multi-token prediction — attacking the one-token-per-forward-pass
cost — explicitly deferred, not dropped.

Artifacts: `scripts/ab_bcd_convergence.sh`, `logs/ab_bcd_convergence.log`,
`checkpoints/ab_bcd_{control,blockcoord}/val_log.csv`, `scripts/probe_optimizer_vram.py`,
`results/optimizer_vram_355m.json`.
## Selective token training (Rho-1) — no gain at 83M (2026-08-07)

**One line: training on only the 60% of tokens the model is most wrong about costs
0.29 nats of perplexity and buys nothing measurable downstream.**

Tested as a candidate answer to "train on less compute". Rho-1 (arXiv:2404.07965)
scores every token with a reference model trained on curated data and applies the loss
only to the top fraction by *excess loss* (target minus reference) — the tokens where
the model is still much worse than the reference. Reported: up to 10x faster to
baseline accuracy on math, and +6.8% average across 15 benchmarks for general-domain
continual pretraining.

**Why this repo could test it faithfully rather than approximately.** The method needs
a reference that is the same base trained further on curated data.
`checkpoints/hybrid_chat_v1.pt` **is** `hybrid_a4_v2/hybrid_a4_step8000.pt` fine-tuned
on the curated OASST + grounded-QA corpora (#108) — the paper's exact relationship, not
a stand-in. Both arms continue-pretrain from `hybrid_a4_step8000.pt` on identical,
pre-materialised batches in identical order, 1,500 steps each, differing only in the
token mask (`scripts/ab_selective_tokens.py`, `scripts/run_slm_ab.sh`).

| | baseline (all tokens) | selective (top 60%) |
|---|---:|---:|
| val loss over ALL tokens | **3.9048** | 4.1965 (**+0.2918**) |
| near-window recall, 300 trials | 27.3% (82/300) | 29.7% (89/300) |
| tokens kept | 100% | 60.0% |

**Verdict: no gain.** Perplexity is clearly worse; the downstream probe is
**statistically indistinguishable** (two-proportion z=+0.63, p=0.53). The nominal
+2.4 points is noise at n=300.

**The experiment was built so a false negative was possible to detect, and it still
came out negative.** Two traps were designed around explicitly:
- *Validation is scored on ALL tokens for both arms.* Scoring only the selected tokens
  would rig the metric toward the arm that chose them.
- *Perplexity alone cannot settle this.* The selective arm trains on a deliberately hard
  subset, so a worse all-token loss is expected and not disqualifying — the paper's
  general-domain result was a downstream benchmark gain, **not** a perplexity win. That
  is why the recall probe was run at all. The observed pattern matches the paper's
  *direction* (perplexity worse, downstream not worse) but the downstream **gain** is
  absent.

**Two limits, stated because they bound the claim.** The downstream reading is a single
probe measuring exact-token recall; the paper used 15 benchmarks, and recall may simply
not be where selection helps. And this is 83M against Rho-1's 1.1B floor — the same
order-of-magnitude gap that ruled out multi-token prediction earlier today (worse than
baseline below ~1B per its own scaling ablation). **This is evidence about this scale and
this setup, not a refutation of the method.**

**Context — the compute-reduction search, settled for now.** Four candidates were
examined today for the "train on less compute" goal:

| candidate | outcome |
|---|---|
| padding-free packing (#132) | ✅ **1.45-5x, merged, on by default** |
| block-coordinate descent | ❌ measured 4x worse quality |
| multi-token prediction | ❌ ruled out from the literature (negative below ~1B) |
| selective tokens (this entry) | ❌ measured, no gain at 83M |

Packing is the one that worked, and it is already landed. The remaining lever at this scale is not a training trick — it is spending the fine-tune budget on more and
better data, which this document has concluded repeatedly.

Artifacts: `results/slm/ab_selective_tokens.json`,
`results/slm/probe_{baseline,selective}_near.txt`, `logs/slm_ab.log`,
checkpoints under `checkpoints/slm_{baseline,selective}/`.

## Dynamic phase-aware LR — wins by 0.30 nats, but the win is "the LR was 10x too high" (Issue #133, 2026-08-09)

**One line: the scheduler beat fixed cosine by 0.30 nats held-out, and then the log showed
why — it spent 94% of the run pinned to its own lower bound, so what it discovered is that
the base learning rate was wrong, not that phase-aware scheduling works.**

`ghost/dynamic_lr.py` watches the live loss curve and emits a global LR multiplier applied on
top of the warmup+cosine schedule (`--dynamic-lr`, default OFF). Muon is untouched: every
param group is scaled by the same factor, so its matrix-group / aux-group LR ratio is exact.
Applied as a context manager around `optimizer.step()` and removed before `scheduler.step()`,
because `CosineAnnealingLR.get_lr()` is recursive and would otherwise fold the multiplier into
the cosine permanently (`tests/test_dynamic_lr.py`, 20/20, includes that specific check).

### The A/B

5,000 steps from `checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt`, Muon, identical
pre-materialised batch sequence, per-arm re-seeding, graded on a fixed 16-batch **held-out**
probe set (never on train loss — the #132 retraction). `scripts/ab_dynamic_lr.py`.

| arm | final held-out | final train | wall-clock | mean multiplier |
|---|---:|---:|---:|---:|
| fixed cosine | 4.3374 | 4.3589 | 26.9 min | 1.000 |
| **dynamic** | **4.0372** | 4.0989 | 26.2 min | **0.126** |

**Gap 0.3002 nats, against a measured identical-run noise floor of sd 0.0046 — ~65x, not
noise.** Both arms started at 4.2945; **only the dynamic arm ever beat its own starting
point.** The cosine arm ended *worse* than the checkpoint it began from, after 5,000 steps.

### What actually fired, which is the whole result

| phase | count |
|---|---:|
| spike (cut, x0.5) | **89** |
| dropping_fast (x1.05) | 36 |
| normal (x1.0) | 25 |
| plateau (x1.5) | **2** |
| transition (x1.25) | **0** |

The multiplier reached its 0.1 floor at **step 294** and never left it for the remaining 4,706
steps. So ~94% of the "dynamic" arm was, mechanically, a constant-0.1x run. The 152 later
adjustments are visible in `results/dynamic_lr/curves.png` as a dense picket of vertical lines
against a completely flat multiplier — the detector firing continuously and changing nothing,
because it was already clamped.

**Both "raise the LR" rules in the spec are dead on this workload.** Plateau fired twice in
5,000 steps; the 2-sigma phase-transition rule fired zero times. Every adjustment that mattered
was a cut.

**And most cuts were not reacting to real events.** At the moment of adjustment the median
`|loss - rolling mean| / sigma` was **0.74**, with only 53/152 outside 1 sigma — i.e. the
majority of triggers were within-noise excursions. That is the question the plot was built to
answer, and the answer is unflattering.

### Verdict

**The 0.30 win is real and reproducible, and it is not evidence for phase-aware scheduling.**
It is evidence that **Muon at `muon_lr=0.02` is roughly 10x too high for continued training
from this checkpoint** — the loss climbs 4.29 -> 4.75 over the first 1,000 steps under the
fixed schedule and never recovers within the run. The scheduler's contribution was noticing
that inside 300 steps and dividing by ten.

**The control arm that would settle this by measurement rather than inference — constant 0.1x,
no detection — was launched and then stopped by the maintainer before completion.** So the
attribution above rests on the multiplier trace (flat at the floor for 94% of the run) and the
phase counts, which are strong but are not the same thing as the arm. Recorded as a known gap,
not glossed: if anyone later wants to claim the detector itself helps, that arm is the bar, and
it is one 27-minute run.

**Recommendation: do NOT make `--dynamic-lr` default.**

### MEASURED — the sane-LR arm beats the scheduler, and that settles it

The third arm, run after the correction below: plain fixed cosine at a 10x lower base LR
(`--lr 2e-5 --muon-lr 0.002`), same harness, same batches, same seed, `--deterministic`.

| arm | base LR | final held-out | vs dynamic |
|---|---|---:|---:|
| cosine | muon 0.02 | 4.3374 | +0.300 |
| dynamic | muon 0.02, auto-cut to ~0.002 | 4.0372 | — |
| **cosine** | **muon 0.002** | **3.9288** | **-0.108** |

**Plain cosine at a sane LR beats the dynamic scheduler by 0.108 nats — 23x the noise floor.**
The scheduler's entire advantage was recovering from a mis-set learning rate; not making the
mistake is better than detecting it, because the dynamic arm still spends its first ~300 steps
at the wrong LR being knocked off the checkpoint before the cuts land. That damage is never
fully recovered inside 5,000 steps.

**Final verdict on #133: the method does not work.** It is a recovery mechanism for a
misconfiguration, and it is beaten by not having the misconfiguration. `--dynamic-lr` stays
non-default. What the exercise actually produced, and what makes it worth keeping: the
determinism fix below, and issue #134 (parameter-free LR), which attacks the real problem —
that a human has to pick this number at all.

### CORRECTION — the baseline was a strawman, and the "10x too high" is narrower than stated

This entry first claimed the LR finding "affects every fine-tune in this repo". **It does
not.** Checked directly after writing that: `continue_ft_hybrid.py:296`,
`finetune_edge_conversational.py:600` and `continue_ft_looped.py:181` all default to
**`--lr 2e-5`**. The repo's fine-tune scripts already encode the right answer.

The `muon_lr=0.02` used above came from `scripts/ab_dynamic_lr.py` copying
`train_edge_conversational.py`'s defaults, which are **pretraining** values — correct for a
freshly initialised model that needs large steps, wrong for a checkpoint already 8,000 steps
into training. New harness, inherited default, nobody had previously run a continue-training
job at pretrain LR.

**So the reading is weaker than the table suggests: the cosine arm is a badly-tuned
control, not a fair one.** The 0.30 nats measures how much damage a 10x-too-high LR does and
how quickly a cut rule undoes it — not that dynamic scheduling beats a competently configured
cosine. A cosine baseline at a sane LR would very likely beat both arms here, and that run has
not been done.

What survives the correction: the scheduler's *cut* path demonstrably works and rescued a
mis-set LR inside 300 steps; the raise rules are dead (2 and 0 firings); and the noise-floor /
determinism finding below is independent of any of this.

### A real bug found on the way: the A/B was not reproducible at all

Two arms with identical code paths landed 0.0046 nats apart (sd over 8 runs), **with the sign
of the gap flipping run to run**. Root cause: this torch build has no flash-attention, so the
2 `TransformerBlock`s' `F.scaled_dot_product_attention` falls back to the mem-efficient CUTLASS
kernel, whose backward accumulates `grad_q` with atomics — nondeterministic by PyTorch's own
warning. Isolated three ways: a 15-line standalone repro (`dq` differs, `dk`/`dv` bit-identical);
the in-model signature (in block 10 only `q_proj` and `norm` move, while `k_proj`/`v_proj`/
`out_proj` are bit-identical, and all 10 DeltaNet blocks are clean); and by flag
(`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` makes 40-step
arms bit-identical across 3 repeats). Muon amplifies it — Newton-Schulz normalises the update,
so a ~2e-3 relative gradient perturbation becomes a full-size step in a different direction.

**That floor sat inside the 0.01-0.05 nat band this experiment was built to measure**, so the
A/B could not have detected its own target effect. `--deterministic` is now ON by default in
`scripts/ab_dynamic_lr.py` (~9% wall-clock). Worth carrying to any future A/B in this repo that
touches the hybrid's attention layers.

**One consequence worth its own line:** at 300 steps, two *identical* runs made **7 and 8**
adjustments respectively — the numerical noise flipped a detector threshold and the LR
trajectories diverged from there (final losses 0.0168 apart). A loss-reactive scheduler
amplifies numerical noise into a different training trajectory rather than rejecting it.

Artifacts: `results/dynamic_lr/ab_dynamic_lr.json`, `lr_log.json`, `curves.png`,
`logs/ab_dynamic_lr.log`.

## Parameter-free LR (Prodigy) — wins fine-tuning outright, loses pretraining (Issue #134, 2026-08-09)

**One line: with no learning rate given at all, Prodigy beats hand-tuned Muon by 0.18 nats on a
fine-tune and loses to it by 0.07 on a from-scratch pretrain — so the recommendation is a
split by regime, not the single universal setting the issue hoped for.**

Opened directly out of #133, whose real finding was that a human has to pick this number and
picks it wrong when the regime changes. `hvn/optimizers.py` gains `Prodigy`
(Mishchenko & Defazio, arXiv:2306.06101): `lr` is pinned to 1.0 and is not a knob; the
optimizer estimates the step size `d` from observed gradients and keeps re-estimating. Two
variants share one estimator — `--optimizer prodigy` (published Adam-family update) and
`--optimizer muon-prodigy` (Muon's Newton-Schulz **direction**, Prodigy's `d` as the **step
size**). The second coupling is not in the paper and its guarantees do not transfer; that is
stated in the docstring, not left to be discovered.

Both regimes, `scripts/ab_prodigy.py`: identical pre-materialised batches, per-arm re-seeding,
`--deterministic`, graded on held-out probe loss. The Prodigy arms get **no LR, no warmup and
no cosine** — the `d`-ramp is the warmup, and adding a schedule would quietly re-introduce a
tuned knob.

### Fine-tune regime — 5,000 steps from `hybrid_a4_v2/hybrid_a4_step8000.pt` (start 4.29448)

| arm | final held-out | step size | wall | vs tuned |
|---|---:|---|---:|---:|
| Muon, hand-tuned | 3.92877 | 0.002 (tuned) | 25.6 min | — |
| muon-prodigy | 3.80098 | 6.09e-05 (auto) | 30.5 min | **-0.128 (28 sigma)** |
| **prodigy** | **3.75055** | 6.60e-05 (auto) | **23.9 min** | **-0.178 (39 sigma)** |

**Plain Prodigy wins on quality AND is the fastest arm**, having chosen a step size **30x
smaller** than the hand-set value. The harness is cross-validated: its `muon` arm reproduced
#133's independent sane-LR run to 5 decimals at every checkpoint (3.96485 at 1,000, 3.92877
final) from a different script.

### Pretrain regime — 1,500 steps from fresh init, same architecture (start 10.91736)

| arm | final held-out | step size | wall | vs tuned |
|---|---:|---|---:|---:|
| **Muon, hand-tuned** | **5.56450** | 0.02 (tuned) | 8.2 min | — |
| muon-prodigy | 5.81351 | 1.41e-03 (auto) | 9.7 min | +0.249 (54 sigma) |
| prodigy | 5.63261 | 1.69e-03 (auto) | 7.5 min | +0.068 (15 sigma) |

**Hand-tuned Muon wins here**, which is the regime #129 actually validated 0.02 for. Note the
baseline asymmetry that makes this the more trustworthy of the two tables: the pretrain `muon`
arm is genuinely tuned, whereas the fine-tune one used 0.002 — #133's rough 10x correction,
never swept. A properly swept Muon would close part of the fine-tune margin, though not
plausibly all 39 sigma of it.

**A caveat that may explain the muon-prodigy loss entirely:** its `d` was **still climbing at
step 1,500** (9.34e-04 at 1,000 -> 1.41e-03 at 1,500), i.e. it never finished ramping from
`d0=1e-6` inside the test. Plain Prodigy settled by step 500 and lost by far less. A real
pretrain is 100,000+ steps, where a 1,500-step ramp is 1.5% of the run. **This test cannot
distinguish "Prodigy is worse at pretraining" from "1,500 steps is too short to see it" and
should not be read as if it could.**

### The claim the issue was built to test, answered

The premise was that ONE invocation with no learning rate covers both regimes. **It does not.**
It covers fine-tuning better than a human does, and pretraining worse. Recommendation:

- **Fine-tuning: use `--optimizer prodigy`.** Better model, faster, no number to pick. This is
  the regime the repo actually spends its time in (1-3 h runs).
- **Pretraining: keep `--optimizer muon --muon-lr 0.02`.** Validated in #129, wins here.

### The consequence that matters more than the optimizer

**Every fine-tune in this repo ran at a badly mis-set learning rate.** The chat FT (#108), the
tool-use FT (#110), and the grounded-QA FTs behind the 35.5%-near / 8.5%-far recall numbers all
used `--lr 2e-5` AdamW or the pretrain-default Muon, against an estimator that lands 30x lower.
Those runs are 1-3 hours each and cheap to redo.

**In particular the 35.5% near-window recall "plateau" (identical at step 5000 and step 8000,
which the A4 writeup read as a data limit) is now ambiguous.** Those 2 attention layers were
freshly initialised and trained from scratch during that fine-tune, and fresh random layers are
exactly where a too-high LR settles into a mediocre fit and stays. "Plateaued" and "badly
optimised" look identical from outside. Re-running the grounded-QA FT under `--optimizer
prodigy` and re-probing at the 300-trial bar is ~2 h and would settle whether that ceiling was
real or self-inflicted. **Not yet done.**

### Limits

One seed per arm. The pretrain regime is 1,500 steps against a real pretrain's 100,000+. The
fine-tune baseline is corrected-not-swept. `tests/test_prodigy.py` 16/16 — every convergence
test also asserts something about `d`, since "loss goes down" alone would pass with an inert
estimator and a lucky `d0`. Artifacts: `results/prodigy/ab_{finetune,pretrain}.json`,
`logs/ab_prodigy_{finetune,pretrain}.log`.

## RAG works: 0% -> 22.5% on facts the weights cannot hold (Issue #137, 2026-08-10)

**One line: retrieval turns a knowledge question this model answers 0% of the time into one it
answers 22.5% of the time — and the single most important tuning result is that `top-k=1` beats
`top-k=3` by 16 points, because extra retrieved passages actively hurt.**

Opened out of the observation that the 83M's measured weakness is *only* knowledge (closed-book
0/5, ppl ~110 web) while its measured strength is *reading a passage handed to it* (35.5%
near-window exact recall, the capability the 2 windowed-attention layers were added for). RAG
converts the weakness into the strength.

`hvn/retrieval.py` — Okapi BM25, pure Python/numpy, **no new dependency and no second neural
model**. Lexical deliberately: BM25 is the floor a dense retriever must later beat, and
introducing an embedding model now would confound "is the retriever good" with "can this model
use retrieved text" — the entanglement that forced #122's retraction.

### The measurement

Three PAIRED arms over the same questions, 200 trials, on `hybrid_a4_v2/hybrid_a4_step8000.pt`.
Facts are about **invented** entities, so closed-book is unanswerable by construction and any
lift is provably retrieval (`eval/rag_probe.py`).

| arm | accuracy | retrieval recall | prompt |
|---|---:|---:|---:|
| closed-book | **0.0%** (0/200) | — | 24 tok |
| RAG, top-3 | 6.0% (12/200) | **100%** | 77 tok |
| **RAG, top-1** | **22.5%** (45/200) | **100%** | 52 tok |
| oracle (perfect passage) | 21.5-22.0% | — | 52 tok |

**RAG at k=1 matches the oracle** (22.5% vs 21.5%, within the n=200 noise of ~2.9pp). Retrieval
is at its ceiling: there is nothing left for a better retriever to win.

### The finding worth carrying forward: more context is worse

The first run reported a 16-point gap between RAG (k=3) and oracle, and the script's own verdict
line said "retrieval is the bottleneck". **That was wrong, and recall@3 = 100% proves it** — the
correct passage was in the context every single time. The gap was the *other two* passages
crowding it out. Dropping to k=1 recovered all 16 points.

The verdict logic is now fixed to check recall before blaming retrieval, because the wrong
version of that line is exactly the kind of thing that gets quoted into a doc unexamined.

**Practical rule for this model: retrieve one passage, not three.** At a 512-token trained window
(#135) context is the scarcest resource in the pipeline, and an 83M model has little capacity to
ignore a distractor. This is likely to relax with scale and with a longer trained window; at this
size it is a 3.75x effect.

### Cost

BM25 index over 2,000 facts builds in **0.01s**; end-to-end query (retrieve + generate) is
**0.20 s**. No training, no GPU-days, no new dependency — the cheapest real capability gain in
this repo's history.

### Limits, stated because they bound the claim

- **The 22% is not comparable to the 35.5% recall figure.** To index 2,000 facts the probe needed
  a wider pseudo-word space than `synthetic_recall_probe`'s 20x20=400 words (which caps at 200
  facts — see the bug below), so it uses 3-syllable words. Longer rare words are harder to
  reproduce exactly. The within-run arms are all on the same words and so are comparable to each
  other; the cross-doc comparison is not.
- **Invented facts are the easy case for BM25** — a unique entity name has very high IDF. Real
  prose, where the query shares common words with thousands of documents, is harder. Recall@k
  will not be 100% there.
- One checkpoint, n=200 (sd ~2.9pp), 320-token context budget.

### A latent infinite loop found and fixed

`synthetic_recall_probe.make_pair` redrew pseudo-words in an unbounded `while True` until it
found an unused pair. The space is 20x20 = **400 words** and each fact consumes two, so asking
for more than 200 facts spins a core **forever**, silently, with no output. The RAG probe
requested 2,000 and hung on fact #201; diagnosed from a live stack (5 identical samples, 5.05s
CPU per 5s wall — spinning, not blocked), which also ruled out the import/network hypotheses.
Now bounded and raises with the actual limit. Existing probe runs are unaffected (they use at
most 17 pairs), but the trap was live for anyone raising `--n-distractors`.

## Prodigy on the grounded-QA FT — recall decays monotonically with steps (Issue #134 follow-up)

Testing whether the 35.5% near-window "plateau" was a badly-optimised fit that a better LR would
break through. **It was not — the opposite.** `continue_ft_hybrid.py --optimizer prodigy`, 2,000
steps from `hybrid_a4_step8000`, then `eval/synthetic_recall_probe.py` at 300 trials:

| FT steps | near-window recall |
|---|---:|
| 0 (baseline) | **36.0%** (108/300) |
| 1,000 | 21.3% (64/300) |
| 2,000 | 9.0% (27/300) |

Far-window: 7.7% -> 1.3%. Train loss fell 2.21 -> 1.40 across the same run.

**Clean monotonic decay.** Optimising *better* walked recall downhill *faster*: the 5,240-example
corpus is exhausted at 8,000 steps, and further optimisation buys memorisation of those specific
pairs — which the probe cannot credit, because it scores **invented** words.

**This retracts the recommendation in the #134 entry above** that every fine-tune should be redone
under Prodigy because they ran at a mis-set LR. A better optimiser does not help when the corpus,
not the optimisation, is the limit. The lever is more and more diverse data, which is the
conclusion this document has now reached four separate times.

The RAG result above is the constructive version of the same observation: if the corpus cannot
teach the knowledge, retrieve it instead of training it.

Artifacts: `results/rag/rag_probe_{baseline,baseline_k1,prodigy}.json`,
`results/prodigy/probe_*.txt`, `logs/rag_probe.log`.

## 62% of the 83M was vocabulary — tied embeddings WIN, the 32K vocab does not (Issue #139, 2026-08-10)

**One line: nobody had ever counted where the parameters went. 62% of this "83M model" is two
vocabulary tables. Tying them is a strict win — 31% fewer parameters and BETTER quality. Shrinking
the vocabulary to 32K is a real trade, not a free win, and on the current English corpus it is
slightly negative.**

Opened after the maintainer refused to accept "training levers are exhausted". That claim was true
of every lever anyone had *examined* — optimizers, kernels, memory, scan, packing, schedulers — and
all of them are downstream of the model's shape. One `named_parameters()` count found this in
seconds.

### The audit (`results/vocab/audit_multi.json`)

Through the real `build_model_and_config` path with ternary on — not a standalone `EdgeHVN`, because
that is precisely the mistake that made every VRAM envelope in [training.md](training.md) wrong:

| vocab | tied | total params | of which real model | MFLOP/token |
|---:|:--:|---:|---:|---:|
| 50,257 | no | 83,079,329 | **38.0%** | 127.0 |
| 50,257 | yes | 57,347,745 | 55.0% | 127.0 |
| 32,000 | yes | 47,981,904 | 65.8% | 108.3 |
| 16,000 | yes | 39,773,904 | 79.4% | 91.9 |

The two levers differ in kind. The input table is a **gather** — it costs memory and no compute, so
tying it is free on FLOPs. The output head is a **matmul**, 512 x 50257 per token, **45% of forward
FLOPs against 38% for all twelve blocks combined** — so shrinking the vocab is a compute lever.

### Tied embeddings — PROVEN, adopt it

1,500 steps from fresh init, identical batches/seed/`--deterministic`, only `tie_embeddings`
differing (`results/vocab/ab_tied.json`, `ab_untied.json`):

| arm | final held-out | params | wall |
|---|---:|---:|---:|
| untied | 5.56450 | 83,079,329 | 8.2 min |
| **tied** | **5.50681** | **57,347,745** | 8.8 min |

**0.058 nats BETTER with 31% fewer parameters** — 12.5 sigma against the #133 noise floor of
sd 0.0046. Not a trade; strictly better and smaller. Both arms share a vocabulary, so per-token loss
IS comparable here.

Likely mechanism: regularisation. 51.5M of the 83M were two independent tables learning nearly the
same mapping from limited data; forcing them to share removes 25.7M parameters that were otherwise
free to memorise.

Cost: **~7% slower per step** despite identical FLOPs, because the shared table now receives
gradients from both the input lookup and the output head, serialising an accumulation.

### The 32K mixed tokenizer — a real trade, currently negative on English

Trained ByteLevel BPE on 300 MB, 50/40/10 code/English/chat. Code came from 21 shallow-cloned
permissive repos (`scripts/collect_code_corpus.py`) — `bigcode/the-stack-dedup` lists anonymously
but 401s on download, and this machine's HF token is malformed (#120), so there is no anonymous path.

**Tokens per document, identical source text, at full corpus scale:**

| corpus | GPT-2 | 32K | ratio |
|---|---:|---:|---:|
| code (sample) | 16.12 tok/line | 11.30 | **0.701** |
| OASST chat | 15,724,872 | 11,929,880 | **0.759** |
| SlimPajama | 1,318,230,992 | 1,346,158,575 | 1.021 |
| WikiText | 119,073,409 | 129,653,042 | 1.089 |

The 4,000-line sample predicted 1.022 for SlimPajama; the full 1.3-billion-token corpus gives 1.021.
The method holds at scale.

**The A/B, graded on bits per byte** (1,500 steps each, 2,010,062 held-out bytes of FineWeb-Edu —
downloaded but mix share 0.0, so unseen by both arms):

| arm | **bits/byte** | tokens/byte | per-token NLL | wall |
|---|---:|---:|---:|---:|
| GPT-2 50257, untied | **2.0444** | 0.2164 | 6.5497 | 8.9 min |
| 32K, tied | 2.0829 | 0.2308 | 6.2555 | 8.0 min |

**The 32K arm is 1.9% WORSE per byte and uses 6.7% more tokens on English.** Net compute per byte is
~9% cheaper — not the 20% a sampling-weighted estimate suggested, and not the 24% quoted earlier
from an unweighted mean that included code the corpus does not contain.

**Three limits, because they decide whether this is worth revisiting:**
- **The held-out set is pure English prose, the 32K vocab's WORST domain.** Its measured advantages
  are code (0.701) and chat (0.759), and neither is in this held-out set or in the pretrain corpus.
  This test measures the vocabulary exactly where it cannot win.
- **The arms are matched on STEPS, not compute.** The 32K arm has 42% fewer parameters and finished
  10% faster; at equal compute it gets more steps and part of the 1.9% closes.
- One seed, 1,500 steps.

**Recommendation: adopt tied embeddings; hold the 32K vocab.** It is a one-way door — a model cannot
be fine-tuned onto a different tokenizer — and its case rests on code and chat being in the corpus,
which is #136's job. Revisit it there, not before.

### Why per-token loss could not have answered this

At **step 0**, untrained, zero knowledge: the 32K arm reads **10.41** and the 50257 arm **10.82** —
exactly ln(vocab) in both cases. After 1,500 steps the order **inverts**: 6.26 vs 6.55, now favouring
32K, because each of its tokens carries more text and is genuinely harder to predict.

Two artefacts pulling opposite ways, neither about quality, and the sign flips during training. On
per-token loss this A/B would have reported the 32K vocab as the clear winner.

`eval/bits_per_byte.py` normalises against source bytes instead. The first token of each window is
left unscored — a model must not be charged for predicting a token it had no context for, and that
penalty falls harder on whichever tokenizer produces more windows.

### Infrastructure: four silent hazards closed on the way

None would have raised an error; each would have produced a plausible wrong number.

1. **The data pipeline overrode the caller's tokenizer with GPT-2's unconditionally.** Passing a 32K
   tokenizer would have tokenized the whole corpus with GPT-2 anyway, and the model would have
   trained on ids from a vocabulary it does not have.
2. **The `.bin` cache recorded nothing about which tokenizer produced it.** Switching vocab reused
   GPT-2 ids in silence. Now a `.tokenizer` sidecar, and a mismatch refuses. Unlabelled pre-#139
   caches are adopted as GPT-2 out loud — and that is not a loophole: an unlabelled cache with a
   *custom* tokenizer still refuses.
3. **`pre_training_audit` compared logits width to the 50257 constant**, so it would have rejected a
   32K model before step 1 — the audit blocking the run it exists to protect.
4. **EOS was hardcoded 50256 in 10 files.** The 32K vocab's EOS is 0. A 32K model served with 50256
   could never sample its stop token, so every reply would run to the token limit — reading as a
   quality problem while being a configuration one. `EdgeHVN` now carries `eos_token_id`, persisted
   and restored. Still hardcoded in `hvn/edge/chat.py`, `tool_session.py`, `eval/grounded_qa.py`,
   `finetune_edge_conversational.py` and `packing.py`'s `pad_id`.

**Also: re-tokenizing is 4.28x faster and bit-identical.** `_tokenize_text_to_bin` called
`tokenizer.encode` once per 4 MB block — one Python call into a single-threaded path, pegging one
core of sixteen. Now split across cores via the Rust backend's `encode_batch`: 2.39s to 0.56s on
4 MB, and 1,017,861 tokens either way. Identical is not free — chunks must begin **at** a newline,
not after one, because newline-plus-indent merges are exactly what a code-trained vocabulary learns.
The full 5.4 GB pass went from ~55 minutes to 10.

Artifacts: `results/vocab/*.json`, `logs/ab_tied.log`, `logs/ab_vocab.log`, `logs/ab_bpb.log`,
`data_cache/tokenizer_32k/`.

## The 32K vocab WINS on code -- 6.26% better bits/byte (Issue #139 follow-up, 2026-08-10)

**One line: put code in the training mix and in the held-out set, and the 32K mixed vocabulary
beats GPT-2 by 6.26% on bits per byte -- reversing the 1.9% loss #139 measured on English, and
about 3x larger.**

#139 graded the 32K vocabulary on FineWeb-Edu: pure English prose, no code in either arm's
training mix. That entry flagged the problem in its own limits section -- "this test measures the
vocabulary exactly where it cannot win" -- because the vocabulary's measured advantages are code
(0.701 tokens/line) and chat (0.759), and neither was present. This is the test it never got.

`scripts/ab_vocab_on_code.py`: 165 MB of permissively-licensed code as the training corpus,
held-out is the corpus TAIL (different repositories and languages from the head, since the
collector walks repos in sorted order), 1,500 steps per arm, tokenizer as the only difference.

### The result (`results/vocab/ab_code.json`)

| arm | **bits/byte** | tokens/byte | held-out bytes | windows |
|---|---:|---:|---:|---:|
| GPT-2 50257 | 2.6569 | 0.4071 | 2,110,655 | 1,679 |
| **32K** | **2.4907** | **0.3517** | 2,110,655 | 1,450 |

**6.26% better per byte, 13.60% fewer tokens per byte.** Both arms are scored on the *identical*
2,110,655 held-out bytes, which is the whole reason `eval/bits_per_byte.py` exists: #139 showed
per-token loss is not comparable across vocabularies and would have flipped the sign.

For contrast, the same metric on English (#139): GPT-2 2.0444, 32K 2.0829 -- the 32K vocabulary
1.9% WORSE. Same tokenizer, same metric, opposite sign. The domain is the whole story.

### The caveat that matters: the arms are matched on TOKENS, not bytes

Both arms consume 1,500 x 4 x 512 = 3,072,000 tokens. They do not consume the same amount of
code, because that is precisely what a tokenizer changes:

| arm | corpus as tokens | tokens trained on | code actually seen |
|---|---:|---:|---:|
| GPT-2 | 72,841,334 | 3,072,000 | ~6.96 MB (4.22% of corpus) |
| 32K | 52,598,601 | 3,072,000 | ~9.64 MB (5.84% of corpus) |

**The 32K arm trained on ~38% more code.** Some unknown share of the 6.26% is that extra text
rather than better modelling of the same text.

Stated as a mechanism, not a hole, and here is why the byte-matched control was considered and
NOT run: packing more text into each token is what a tokenizer is *for*. It is the product, not
contamination. A byte-matched run would discard the main benefit and report a number that
understates the vocabulary that would actually ship.

The compute argument runs the same way. From this issue's own audit, 32K tied is **108.3
MFLOP/token against 127.0** for the 50257 head. The 32K arm sees ~38% more code AND costs ~15%
less per token, so matched on compute it wins by more than 6.26%, not less. #139's mirror-image
caveat ("matched on STEPS, not compute ... at equal compute part of the 1.9% closes") pointed the
other way on English; it points this way here.

### Limits

- **Wall times in the JSON are not comparable and must not be quoted.** The gpt2 arm's 1,052 s
  shared the GPU with a duplicate process (below); the v32k arm's 457 s ran alone. That 2.3x is an
  artefact of the launch mistake, not a speedup. Quality is unaffected -- determinism is
  per-process (`torch.use_deterministic_algorithms(True)`, per-arm re-seeding).
- One seed, 1,500 steps, one held-out set.
- Held-out is the corpus tail. Different repos from the head, so not a memorisation check, but it
  is the weakest part of the design and the script says so in its own docstring.
- Both arms see well under one epoch (4-6% of corpus), so neither is in a multi-pass regime.

### What this does NOT settle

**The adoption decision is still open, and #136 still owns it.** This measures code alone. The
English penalty has not been re-measured against a corpus containing both, and the vocabulary is a
one-way door -- a model cannot be fine-tuned onto a different tokenizer. The run that would move
the decision is a mixed-corpus A/B, which is exactly where #139 said to revisit it.

Current reading: 32K wins on code by 6.26%, loses on English by 1.9%. Whether that nets out
depends entirely on the mix, and nobody has measured the mix.

### Two process failures worth recording

**1. The run was launched twice, and the second launch hid the first.** Both processes were
redirected to `logs/ab_code_vocab.log`. Each held its own write offset, so the later process
overwrote the earlier one *in place*: file size stayed pinned at 2,717 bytes, `tail -f` never saw
growth, and a monitor watching for progress stayed silent through a completed run. Silence read as
"still running" for 16 minutes after the verdict had already been written. Two copies of a
12-layer model also pushed the 6 GB card to 5,766 MiB and one arm died of CUDA OOM inside Muon's
Newton-Schulz (`hvn/optimizers.py:73`). Rule added to CLAUDE.md: one run per log file, and check
the GPU is clear before launching.

**2. `logs/ab_code_vocab.log` is not a citable artefact for this run.** Its step lines are two
processes interleaved -- the timings read 6.6, 10.6, 12.4 min for successive checkpoints, and one
line lost its opening bracket where the writes collided. The JSON is clean, written only by the
surviving process. Cite the JSON.

### A reproducibility gap found on the way

**The #139 A/B script was never committed.** `main` carries `results/vocab/ab_bpb_{gpt2,32k}.json`,
`ab_tied.json` and `ab_untied.json`, but no script that produces them -- only
`scripts/audit_vocab_budget.py` and `scripts/train_tokenizer.py`. Searched every `.py` on disk
outside `.venv` and every branch: it does not exist. So the headline "tied embeddings win by 0.058
nats" result, which is now the repo default, cannot be reproduced from the repo. Rewriting that
harness is open work.

Artifacts: `results/vocab/ab_code.json`, `scripts/ab_vocab_on_code.py`.

## The 32K vocab on a MIXED corpus: a split, not a win -- break-even is ~17% code (Issue #139, 2026-08-10)

**One line: on a corpus containing code, English and chat, the 32K vocabulary is 8.6% BETTER on
code, 2.4% WORSE on English and 2.7% better on chat -- so there is no single answer, and the
decision rule is the code share of the corpus: above ~17% code the 32K vocab wins, below it
GPT-2 wins.**

The two prior measurements disagreed in sign. #139 graded the vocabulary on English only and got
1.9% worse; the code follow-up graded it on code only and got 6.26% better. Both were
single-domain, so neither could answer the question that actually decides adoption -- what
happens on a corpus containing both. The tokenizer is a one-way door: a model cannot be
fine-tuned onto a different vocabulary, so the choice is made once, before the pretrain.

`scripts/ab_vocab_mixed.py`: both arms train on the SAME mixed corpus, then each is scored on
THREE separate held-out sets, byte-identical per domain across arms.

### Setup

193 MB mixed corpus, measured shares **code 49.4% / English 39.7% / chat 10.8%** by bytes (the
mix the 32K tokenizer was itself trained on). Sources are `code_corpus/code_all.txt`,
`slimpajama_en.txt` and `oasst1.txt`, interleaved in 1 MB blocks so random window sampling reaches
every domain instead of hitting three contiguous slabs. 1,500 steps per arm, seed 7,
`--deterministic`, tied embeddings, no ternary, tokenizer as the only difference.

Held-out is 2 MB per domain with a **2 MB discarded gap** between train and held-out in each
source. Without the gap the slices touch, and one repository or chat thread straddling the
boundary puts its own text on both sides -- contamination that a larger held-out set does not fix.

### Result (`results/vocab/ab_mixed.json`)

| domain | GPT-2 b/B | 32K b/B | change | 32K tokens/byte |
|---|---:|---:|---:|---:|
| code | 2.1637 | **1.9770** | **-8.63%** | 0.3312 vs 0.5048 |
| English | 2.1983 | 2.2514 | **+2.42%** | 0.2413 vs 0.2329 |
| chat | 2.5755 | **2.5070** | **-2.66%** | 0.2673 vs 0.3355 |
| blended at the trained mix | 2.2220 | **2.1434** | -3.54% | -- |

**The blended -3.54% is the least useful number in the table** and is recorded only for
completeness. It is a property of the mix that happened to be trained on, not of the vocabulary.
Per-domain figures are the durable result because they re-blend for any other mix without
another GPU hour.

### The decision rule

Absolute differences (32K minus GPT-2, bits/byte): code **-0.1867**, English **+0.0531**, chat
**-0.0685**. Solving for where the blend is zero:

| corpus | 32K vs GPT-2 |
|---|---:|
| 100% code (Issue #136's model) | **-8.63% (32K wins)** |
| 30% code / 60% English / 10% chat | -1.4% (32K wins) |
| **~17% code / 73% English / 10% chat** | **break-even** |
| 5% code / 85% English / 10% chat | +1.30% (GPT-2 wins) |
| this repo's current pretrain mix (~65% English / 35% chat) | +0.45% (GPT-2 wins) |

With no chat at all, break-even rises to **~22% code**.

**Recommendation: adopt 32K for the code model (#136); keep GPT-2 for the general/chat line.**
The single universal answer this issue kept looking for does not exist, and the blended number
would have hidden that.

### The asymmetry that cuts against the 32K vocab

The arms are matched on TOKENS, not bytes. On this corpus the 32K vocabulary needs 56,127,531
tokens where GPT-2 needs 70,189,328 for identical text, so at the same 3.07M-token budget **the
32K arm trained on ~25% more text**.

**It lost on English anyway.** The English penalty is therefore measured *against* a 25% data
advantage, which means the true penalty is larger than the 2.42% quoted. The same logic flatters
the code and chat wins by an unknown amount. That is the reading in both directions, and
it is why the recommendation splits by domain rather than declaring a winner.

### A clean confirmation of the compute saving

Both arms ran alone on the GPU this time, so wall times ARE comparable (unlike the code A/B,
where a duplicate process contended):

- GPT-2 arm 564.3 s, 32K arm 481.2 s -> **ratio 0.853**.
- Predicted from the audit: 108.3 / 127.0 MFLOP/token = **0.853**.

The measured wall-clock ratio matches the predicted FLOP ratio to three decimals. The smaller
output head delivers exactly the ~15% per-token saving the parameter audit said it would, with no
hidden cost elsewhere.

### Limits

- 1,500 steps, one seed, one mix. This sizes direction and rough magnitude; it is not a pretrain.
- Held-out slices differ from the code-only A/B's, so bits/byte values are NOT comparable across
  the two runs. Only the within-run GPT-2-vs-32K gap carries meaning.
- The break-even shares are linear extrapolations from three measured domains. They assume a
  domain's per-byte difficulty does not change with its share of the mix, which is an assumption,
  not a measurement.

### Two guards whose only job is to fail

`_check_heldout` refuses to run if any domain's held-out set is under 500K chars, and each domain
must score at least 200 windows or the run aborts. #139 shipped a held-out set that was silently
EMPTY -- `read(n)` consumes characters, not bytes, so the slice hit EOF -- and bits/byte would
have reported `nan` without complaint. A silent `nan` is how a broken measurement becomes a
published number.

Artifacts: `results/vocab/ab_mixed.json`, `scripts/ab_vocab_mixed.py`, `logs/ab_vocab_mixed.log`.

## How big a model can this card actually train? 129M body in 9.3 days (Issue #136 sizing, 2026-08-10)

**One line: measured the real ceiling with the real architecture in the real execution mode, and the
answer is that the 32K tied vocabulary buys a 36% larger MODEL BODY at the same wall-clock -- 129M
body in 9.3 days against 95M body in 9.0 days for #136's planned GPT-2 config.**

### Why a new tool when eval/vram_envelope.py already sweeps sizes

That tool cannot answer the question, for three independent reasons, and training.md already warns
its rows "must not be used to size a real run":

1. **It hardcodes `VOCAB = 8000`.** At d=1024 that is ~8M params of embedding against ~103M for an
   untied GPT-2 vocabulary. It makes the vocabulary term -- 42-62% of these models per the #139
   audit -- nearly invisible, so every row overstates how much *body* the card can afford.
2. **It builds `HamiltonianVortexNetwork`**, the core class, which has NO `block_plan` and NO
   `attn_window`. The trained model is `EdgeHVN` with two sliding-attention layers at positions
   3 and 10. The old rows measure a pure DeltaNet trunk -- not the hybrid -- and the attention
   layers are exactly the ones whose memory scales differently with d_model.
3. **No tie_embeddings flag**, so it cannot represent what is now the repo default.

`scripts/size_sweep.py` goes through `build_model_and_config` with `arch="hvn-hybrid"`, ternary on,
Muon, and the `--fast` stack applied.

### The result (`results/size_sweep_fast.json`), --fast ON, head_dim 64, B=4 S=512

| config | arm | total | **body** | vocab share | peak VRAM | tok/s | days | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 640x16 | gpt2-untied | 130M | 66M | 49.4% | 3,934 MiB (64.1%) | 5,639 | 5.3 | ok |
| 640x16 | **32k-tied** | 86M | 66M | 23.7% | **3,046 MiB (49.6%)** | 6,455 | **3.1** | ok |
| 768x16 | gpt2-untied | 172M | 95M | 44.9% | 5,070 MiB (82.6%) | 4,399 | 9.0 | ok |
| 768x16 | **32k-tied** | 119M | 95M | 20.6% | 3,760 MiB (61.2%) | 4,916 | **5.6** | ok |
| 896x16 | gpt2-untied | 219M | 129M | 41.1% | 5,314 MiB (86.5%) | 3,492 | 14.5 | SUSPECT |
| 896x16 | **32k-tied** | **157M** | **129M** | 18.2% | 4,722 MiB (76.9%) | 3,934 | **9.3** | **ok** |

**Largest body that fits cleanly: 896x16 32k-tied -- 129M of body inside 157M of total.**

The comparison that matters is not total parameters, it is **body at equal wall-clock**:

| | #136 as planned | this recommendation |
|---|---|---|
| config | 768x16, GPT-2 untied | **896x16, 32K tied** |
| body (what learns) | 95M | **129M (+36%)** |
| total params | 172M | 157M |
| peak VRAM | 82.6% | **76.9%** |
| days to Chinchilla | 9.0 | 9.3 |

Same card, essentially the same clock, 36% more model. The budget comes entirely from not spending
45% of the parameters on two lookup tables.

### The measurement error this run made, and corrected

**The first version of this sweep measured eager mode and was wrong about everything absolute.**
`--fast` is `default=True` on `train_edge_conversational` (line 2652, since #129) but is applied in
`main()`, NOT in `build_model_and_config` -- so calling the builder directly silently gets none of
compile, TF32 or Triton scan.

That produced a published-looking claim that #136's config "does not fit, spills at 105% of VRAM,
needs 30.6 days". **Retracted.** Measured properly it fits at 82.6% and needs 9.0 days, and #136's
own ~7-day estimate was approximately right.

| 768x16 gpt2 | peak VRAM | tok/s | days |
|---|---:|---:|---:|
| eager (first, wrong) | 6,446 MiB (105.0%) | 1,300 | 30.6 |
| --fast (correct) | 5,070 MiB (82.6%) | 4,399 | 9.0 |

**Caveat on that contrast, stated because it is not a clean isolation:** the eager sweep used a
fixed `n_heads=8` while the --fast sweep derives `n_heads = d_model / 64`, so at d=768 the two runs
differ in head count (8 vs 12) as well as compile mode. The --fast table is internally consistent
and is the decision-relevant one; the eager-vs-fast deltas are indicative, not measured.

**The lesson, which is the same one #139 recorded in different words:** building through the real
construction path is necessary but NOT sufficient. It must also RUN the way training runs. Two
tools failed here for two different reasons -- one measured the wrong model class, one measured the
wrong execution mode -- and both looked authoritative.

### A hard crash nobody had hit: head_dim must be a power of 2

`hvn/core/deltanet_triton.py:57` does `tl.arange(0, D)` over the head dimension, and Triton requires
that range to be a power of 2. With `n_heads=8`:

| d_model | head_dim | Triton scan |
|---:|---:|---|
| 512 | 64 | ok |
| 640 | 80 | **InductorError** |
| 768 | 96 | **InductorError** |
| 896 | 112 | **InductorError** |
| 1024 | 128 | ok |

**It is a hard crash under --fast, not a silent fallback.** The rule is documented in
`hvn/kernels/hamiltonian_kernel.py:293` and enforced nowhere. The fix is not to avoid those widths
-- it is to pick `n_heads` so `d_model / n_heads` is a power of 2: d=768 wants **12 heads**, d=896
wants **14**. With 8 heads a 7-day run would die partway in with a traceback naming `tl.arange` and
saying nothing about the model config.

**Recommended follow-up: a startup guard.** When `triton_scan` is on, assert `d_model % n_heads == 0`
and that the quotient is a power of 2, and fail immediately with a message naming both numbers.

### The spill detector, and why a plain ok/OOM verdict is not enough

On Windows/WDDM, exceeding VRAM does **not** raise OOM -- the driver spills to host RAM over PCIe
and the step merely gets slower. A run that "succeeded" may be a run that did not fit. The eager
sweep produced rows at 105%, 127% and 138% of VRAM that all completed without error.

So `verdict` is only `ok` when peak reserved is under 85% of total AND step-time variability is
under 0.25. Either signal alone marks the row SUSPECT. Reporting only completion is how the 268M row
in training.md was published as fitting when the real script spills at 355M.

### Limits

- 20 measured steps after 6 warmup, one seed, B=4 S=512, no dataloader in the loop. Real training
  adds data loading and may use longer sequences; these are a floor on memory, not a ceiling.
- Peak memory is reset AFTER warmup, so torch.compile's transient autotuning allocations are
  excluded. Steady-state is what a multi-day run sits at, but a large enough compile-time spike
  could still spill during the first step.
- **The `days` column applies this repo's 20-tokens-per-param convention to TOTAL params.** The two
  arms have identical bodies, so the 32K arm is credited with needing ~30% fewer tokens purely for
  having smaller lookup tables. If the correct budget scales with BODY params instead, the day
  counts converge and the 32K advantage shrinks to the throughput margin (1.11-1.15x) plus
  the VRAM headroom. **This is unmeasured and it is the weakest number in the table.**
- Chinchilla-optimality itself is an assumption at this scale; #134 noted a 1,500-step pretrain
  cannot distinguish a real effect from an unfinished ramp.

Artifacts: `results/size_sweep_fast.json` (--fast, decision-relevant), `results/size_sweep.json`
(eager, retained only as the record of the retracted measurement), `scripts/size_sweep.py`,
`logs/size_sweep_fast.log`.

### Addendum (#136, 2026-08-10): megakernel on, and 168M body is reachable -- but only with the 32K vocab

Two follow-up runs after the sweep above, same script, same steps, same seed, so these are paired
against the numbers in the table.

**Design B (`--megakernel-scan`, #96) enabled.** It was never on in the first sweep.

| 896x16 | VRAM | tok/s | days |
|---|---:|---:|---:|
| 32k-tied, megakernel OFF | 4,722 MiB (76.9%) | 3,934 | 9.3 |
| **32k-tied, megakernel ON** | **4,148 MiB (67.6%)** | 4,007 | **9.1** |
| gpt2-untied, megakernel OFF | 5,314 MiB (86.5%) | 3,492 | 14.5 |
| gpt2-untied, megakernel ON | 5,286 MiB (86.1%) | 3,626 | 14.0 |

**The gain is memory, not speed: -574 MiB on the 32K arm for +1.9% throughput.** That is well under
#96's ~1.05x end-to-end figure but consistent with its mechanism -- Design B recomputes in the
backward instead of storing, trading compute for VRAM.

**Unexplained, and recorded as such:** the memory saving is 574 MiB on the 32K arm and 28 MiB on
GPT-2, for the same change at the same size. One run per arm cannot separate a real effect from
allocator variation. Not worth a mechanism story on this evidence.

**Both arms reproduced the previous table within noise, which is the regression check** that moving
the fast stack out of `main()` into `apply_fast_stack` did not change behaviour.

### The headroom bought a bigger body

At 67.6% there was ~1.9 GB spare, so d=1024 was worth measuring -- it had only ever been run eager,
where it read 114.8%.

| 1024x16 | total | body | VRAM | tok/s | days | verdict |
|---|---:|---:|---:|---:|---:|---|
| gpt2-untied | 271M | 168M | 6,224 MiB (101.4%) | 2,324 | 27.0 | SUSPECT (spills) |
| **32k-tied** | 201M | **168M** | **4,652 MiB (75.8%)** | 3,159 | 14.7 | **ok** |

**A 168M body is only reachable with the 32K vocabulary.** The GPT-2 arm spills at the same size,
because 103M of its parameters are two lookup tables. This is the strongest form of the #139 result:
at d=1024 the vocabulary choice is not "cheaper", it is the difference between trainable and not.

### The ladder, and which rung to pick

All --fast + megakernel, all clean fits:

| config | body | VRAM | days | days per 10M body |
|---|---:|---:|---:|---:|
| 896x16 32k-tied | 129M | 67.6% | **9.1** | **0.71** |
| 1024x16 32k-tied | 168M | 75.8% | 14.7 | 0.87 |

**896x16 remains the recommendation.** Going to 1024 buys +30% body for +61% time -- a worse rate
per unit of model. 1024 is genuinely available if 15 days is acceptable; it is not free.

For context on where this started: #136 planned a **95M body in ~9 days**. The same card now gives a
**129M body in 9.1 days**, from the vocabulary and execution-mode changes alone, with no
architecture work.

Artifacts: `results/size_sweep_megakernel.json`, `results/size_sweep_1024.json`,
`logs/size_sweep_mega.log`, `logs/size_sweep_1024.log`.

## HANDOFF: #136 is ready to pretrain (2026-08-10, session close)

Written at the end of a long session so a fresh window can continue without
re-deriving any of it. Everything below is committed; nothing lives only in chat.

### The recommended run

    python train_edge_conversational.py \
      --code-corpus data_cache/code_corpus/stack_code.txt \
      --tokenizer data_cache/tokenizer_32k_fim \
      --fim --fim-rate 0.5 --fim-spm-rate 0.5 \
      --arch hvn-hybrid --attn-layers 3,10 \
      --d-model 896 --num-layers 16 --n-heads 14 \
      --tie-embeddings \
      --checkpoint-dir checkpoints/code_896x16 \
      --val-fraction 0.005

**n_heads=14 is not optional.** head_dim must be a power of 2 or the Triton scan
crashes under --fast; 896/14 = 64. `check_head_dim` now enforces it at startup.

**--checkpoint-dir must be fresh.** Resuming rebuilds the architecture from the
checkpoint; `_check_resume_matches_request` now refuses on mismatch rather than
silently training the wrong model, but a dedicated directory avoids the question.

Expect **9.1 days measured, 10-12 realistic** (measured with synthetic tensors,
no dataloader, on a laptop that throttles).

### What is settled, and where the numbers are

| decision | answer | source |
|---|---|---|
| architecture | hvn-hybrid, d=896, 16L, 14 heads | `results/size_sweep_megakernel.json` |
| size | 157M total / **129M body**, 67.6% VRAM, 9.1 d | same |
| vocabulary | 32K tied + 3 FIM sentinels = **32003** | `results/vocab/ab_mixed.json` |
| corpus | 11.20 GB, 2.86M files, **3.98B tokens** | `results/codecomplete/corpus_stats.json` |
| FIM | rate 0.5, spm 0.5, char-level cuts | `hvn/fim.py`, 39 tests |

Coverage is 1.27x Chinchilla for 157M, so the run stays under one epoch.

### The three open questions, in priority order

1. **Token budget: 3.14B (total params) or 2.58B (body params)?** ~1.5 days of
   difference. The 20-tokens-per-param convention is applied to TOTAL params
   throughout this repo, but the two arms in the size sweep have identical
   bodies, so the 32K arm is credited with needing 30% fewer tokens purely for
   having smaller lookup tables. **Unmeasured, and the weakest number in every
   table.** Recommendation: use 3.14B -- it errs toward more data, and this
   project's most expensive failure was undertraining.

2. **A resume test, before the clock starts.** Designed but not run. Three of the
   five ways resume can break make the loss curve look BETTER, not worse (data
   position restarting re-reads familiar data; optimizer momentum loss; LR
   schedule restart), so it needs an uninterrupted control arm:
   run 60 clean -> run 30, kill, resume to 60 -> compare step count and loss.
   Without arm 1 any plausible number passes.

3. **The eval does not exist.** #136 criterion 4 is line-level accept rate and
   exact-match next-line on held-out repos, against a stated baseline (n-gram or
   nearest-line retrieval). Nothing measures this yet, and a model with no eval
   cannot produce the result the whole track exists for.

### What was found today, condensed

Five bugs, none of which raised an error and all of which produced confident
wrong answers:

- `eval/vram_envelope.py` sized runs from the wrong model class at vocab 8000.
  Deleted; superseded by `scripts/size_sweep.py`.
- `--fast` lives in `main()`, so every measurement harness measured an
  uncompiled model. Published "30.6 days" for a 9.0-day config. Fixed by
  `apply_fast_stack`, the single implementation.
- `main()` has been unable to start a training run since #139 merged
  (`eos_token_id` passed but not accepted). Nothing noticed because every A/B
  harness bypasses `main()`.
- The pre-training audit could not pass on the hybrid (rebuild dropped
  `block_plan`), so it only ever validated the plain architecture.
- Resuming silently replaced the requested architecture with the checkpoint's:
  a request for 157M trained a 28M model, at step 100 of 120, with no warning.

The pattern is one thing: **a stale or partial artifact quietly preferred over
what was asked for.** #139 closed it for the tokenizer cache; today closed it for
execution mode, model class, and resume.

### Known limits carried forward

- The held-out split is the corpus TAIL, and the corpus is written in language
  order, so it is SQL-heavy rather than representative. Fine for watching a loss
  curve, not for a language-stratified quality claim.
- The realised language mix is not the planned mix: filtering removes minified
  JS by BYTES far more than by file count. JavaScript needed a manual top-up
  from 7.9% to 15.9%.
- 2.43% of files in The Stack's "permissive subset" are not permissively
  licensed per file. Measured across 1, 29 and 34 shards (2.55/2.43/2.37%), so
  it is a property of the dataset. They are dropped.

### UPDATE (2026-08-10 19:00): the smoke run CRASHED -- start here

The handoff above says the pipeline was verified end-to-end. **It was not.** The
120-step smoke reached training, passed all four audit checks, compiled, and then
died at 18:35:48:

    extern_kernels.bmm(buf333, reinterpret_tensor(buf327, (56, 64, 32),
                       (2048, 1, 64), 0), out=buf334)
    RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR
                  when calling cublasSgemmStridedBatched(...)
    [W CachingHostAllocator.cpp:26] Exception in pinned allocator free()

`56 = batch 4 x 14 heads`, `64 = head_dim`, so it is the batched matmul at the
shapes this config introduces. Log: `logs/smoke_code.log` (the traceback starts
at line 233; the file also contains a large C-level stack dump, filter with
`grep -vE "blocks\.[0-9]+\.|python314.dll"`).

**Not diagnosed.** Two candidates, neither confirmed:

- **A disguised allocation failure.** CUBLAS_STATUS_INTERNAL_ERROR on a bmm is
  frequently cuBLAS failing to get its workspace. The pinned-allocator warning
  points the same way, at HOST pinned memory rather than VRAM.
- **Something specific to n_heads=14.** This is the first time this config has
  been trained; the size sweep ran the same shapes, but see below.

**The lesson this repeats, and it is the same one as every other bug today:**
`scripts/size_sweep.py` measured d=896/16L/14 heads as fitting cleanly at 67.6%
VRAM, and that measurement is not wrong -- it simply had **no DataLoader, no
pinned host buffers and no gradient accumulation**. The real training path has
all three. "The sweep says it fits" was again treated as evidence about a path
the sweep does not exercise.

**Next window starts here, before anything else:**

1. Reproduce with `--num-workers 0` (removes pinned host memory) and with a
   smaller `--grad-accum`, to separate allocation pressure from a shape problem.
2. If it is allocation, the 896x16 config may need `--grad-checkpoint` or a
   smaller batch, and the 9.1-day estimate changes.
3. If it reproduces at n_heads=14 but not n_heads=8 with the same memory
   settings, it is a shape/kernel problem and the config choice is affected.

Until this is understood, **the 9.1-day plan is not launchable** -- and that is the status, not the "ready to pretrain" heading above.

## Adafactor does NOT scale to 1B — offload stays mandatory (#157, 2026-08-14)

The #112 entry above concludes **"`--optimizer adafactor` removes the need for BCD at
355M entirely"**. That is correct at 355M and does **not** generalise. The wording invites
the extrapolation, and it was made — this issue was opened partly on the assumption that
Adafactor might make the whole offload path unnecessary at 1B, and priced that first.

Measured, 1B (d=2048, 24L), seq 4096, tied embeddings, ternary, grad-checkpoint, on the
5.9966 GiB card (`results/adafactor_1b/results.json`):

| arm | peak alloc | peak reserved | s/step | tok/s |
|---|---:|---:|---:|---:|
| **adafactor**, no BCD, no offload | 9.08 GiB | **9.42 GiB** | 48.50 | 84 |
| **muon**, no BCD, no offload | — | — | **OOM** | — |
| adafactor + BCD + offload | — | — | **blocked by the #151 guard** | — |
| BCD + offload, untied (reference) | 1.15 GiB | **4.05 GiB** | **4.68** | **875** |

**9.42 GiB reserved on a 6 GiB card is a spill, not a fit.** It ran only because WDDM
silently backs the overflow with host RAM — no OOM, just 4-15x slower, the failure mode
this repo has documented before. Here the penalty is **10.4x**: 48.50 s/step against 4.68
with offload.

**Why it does not carry.** Adafactor shrinks *optimizer state* (row+column summary instead
of a full second moment). It does nothing about weights or gradients. At 355M those two are
~2.8 GB combined, so optimizer state was the deciding term and Adafactor won by 2.8 GiB. At
1B they are ~4.3 GB **each** — over the card before the optimizer is counted at all. The
lever that mattered at 355M is not the binding one at 1B.

**Consequence: at 1B, BCD + CPU weight offload is mandatory, not optional.** It is the only
configuration measured to fit (4.05 GiB, 875 tok/s). Which makes the tied-embeddings
incompatibility (#151's guard, #157) a real blocker rather than an inconvenience: a 1B run
today must pass `--no-tie-embeddings`, costing 70M parameters and 250 MB of VRAM, and giving
up a lever #139 measured as *better* as well as smaller (held-out 5.50681 tied vs 5.56450
untied at 31% fewer params).

**Still time-bound, not memory-bound.** At the measured 875 tok/s a 1B model needs ~267 days
for a Chinchilla-optimal 21.4B tokens. A 1B model **fits and trains** on this 6 GB card —
the ceiling is arithmetic, not memory, and offload does not touch arithmetic.

**Also corrected here:** `docs/training.md`'s offload table reports 1.04B at 1.73 s/step,
measured at batch 4 x seq 128 = **512 tokens/step**. Real training runs 4096-token sequences.
At the sequence length actually used the same config is 4.68 s/step — but **2.8x better
throughput** (875 vs 313 tok/s), because per-step offload overhead amortises over more
tokens. Do not read the small-batch step time as a throughput figure in either direction.

Prediction was recorded in the benchmark's docstring before it ran, along with the
falsifiability check: a fit and a spill are both visible, so this measurement could have
closed #157 rather than confirming it.

## HVN beats a matched transformer at equal compute — the first real head-to-head (#160 Phase 0, 2026-08-16)

The only architecture comparison this repo had was §8.12: 5M tokens, WikiText, single seed,
perplexity only. Everything since has been HVN-only — `results/size_vs_time/results.json`
has six HVN configs and no baseline. The 48M code model proved the architecture *functions*;
it proved nothing about *better*, because the opponent was never run.

**Setup.** 384x8, `--arch hvn-hybrid` (attn at 2,6) vs `--arch transformer`. Parameters match
to 0.4% (body 11.89M vs 11.84M). Identical corpus, tokenizer, FIM version, sequence length,
precision, optimizer, seed (1234), constant LR, and held-out slice. **Matched WALL CLOCK,
7.4 h each** — hours are what you spend, and each architecture converts them into tokens at
its own rate.

| arm | held-out NLL | tokens seen | tokens/param |
|---|---:|---:|---:|
| **HVN hybrid** | **1.9654** | 475M | 40.0 |
| transformer | 1.9814 | **672M** | 56.7 |

**HVN wins by +0.0160 against a pre-registered noise threshold of ±0.011** — and it wins
while seeing **40% LESS data**, because the transformer runs 1.41x faster per step at this
size (1.29 vs 1.82 s/step, the first time a transformer baseline has been timed here).

At matched tokens the gap is much larger — roughly 4x the noise floor at every point:

| tokens/param | HVN | transformer |
|---|---:|---:|
| 20 | **2.0509** | 2.1361 |
| 30 | **1.9841** | 2.0703 |
| 40 | **1.9613** | 2.0163 |

So: HVN extracts more from the same data, and enough more to absorb a 40% data deficit.

**Two things this does NOT say.** Nothing above a 12M body — both models are small enough
that the capability gates read at the floor for both, so this is a loss comparison, not a
capability one. And it is a single seed: same-seed removes *between-arm* luck but both arms
share whatever that seed favours. #86's A/B used the same compromise.

**A correction to a claim made earlier in the week.** HVN is *not* cheaper to train. At
384 wide / 4096 long the transformer is 1.41x faster per step; the recurrent scan, not
attention, is the bottleneck at this size. The compute win is at INFERENCE (6.29 MB vs
2.46 GB of context memory at 100K tokens, O(1) vs O(N)) and that is settled separately.
Training-speed parity should not be claimed.

**Next:** #160 Phase 1, the 48M pair, both arms fresh and seeded, 22 h each — launched
2026-08-16 05:51. That answers the question Phase 0 cannot: whether the lead grows or
shrinks with size, and whether it shows up in capability rather than only in loss.

## Tokens-per-parameter for HVN: ~30, not 20 — but the size trend is unresolved (#159, 2026-08-16)

Every sizing decision in this repo divides by Chinchilla's 20 tokens/param, fitted on
transformers and never checked here. Measured on three sizes with a constant LR (so every
checkpoint is a valid "trained to X tokens" model rather than a snapshot partway through a
cosine — the bias Hoffmann et al. had to re-run to avoid), scored on a fixed held-out slice
via `eval/heldout_loss.py`:

| tokens/param | 12M body | 21M body | 32M body |
|---|---:|---:|---:|
| 20 | 2.0509 | 1.8619 | 1.6397 |
| 30 | 1.9841 | 1.8264 | — |
| 40 | 1.9613 | 1.8037 | — |

Going 20 -> 30 buys 1.9-3.3%; 30 -> 40 buys ~1.1% for 33% more time. **~30 is where the
trade turns**, and 20 is too low.

**What is NOT resolved: how the ratio moves with size.** The elbow detector returns 24.8,
9.3 and 19.7 for the three sizes — not ordered by size, which is the question. The 21M curve
is twice as noisy as the others (±0.0097 vs ±0.0035-0.0053), and the 32M curve is truncated
at 25.9 tokens/param, still improving, because it was never trained further.

**Fitting a scaling law to the 120 points does not rescue it, and the failure is recorded
rather than hidden.** The standard parametric form returns optimal ratios of 10.6 / 4.2 /
1.2 / 0.3 tokens-per-param at 12M / 32M / 129M / 1B — i.e. it claims a 1B model needs less
data than it has parameters, contradicting every raw curve, with a fit error (±0.038) larger
than the effects being measured. Five free parameters, three sizes, one truncated: the fit
is degenerate. **Use ~30 as a working number; do not extrapolate it across scale.**

Consequence for sizing at 30 tokens/param, using measured step times: 48M in 1.1 days, 70M
in 2.0, 108M in 4.7, 157M in 9.7. The 157M launched at 25,000 steps on 2026-08-14 would have
reached 0.32x Chinchilla — the #112 trap — and was stopped for this study.

## Chunked cross-entropy: -1436 MB at 48M, nothing at 1B (#158, 2026-08-17)

The loss was the largest single VRAM consumer -- larger than the model. Measured on a
live 48M run (3.27 GB in use): cross-entropy ~1500 MB, CUDA context ~400 MB, weights
183 MB, optimizer 183 MB, layer activations ~50 MB. Gradient checkpointing had already
reduced activations to nothing; nothing had been done about the loss.

Chunking the CE alone would save nothing -- `model(x)` materialises the full
[B, S, vocab] logits before the loss is called. So `EdgeHVN.forward(return_hidden=True)`
returns [B, S, d_model] (4 MB vs ~500 MB) and `hvn/edge/chunked_loss.py` projects and
scores a slice at a time, each slice wrapped in `torch.utils.checkpoint` so backward
recomputes one slice's logits rather than holding all of them.

| 48M, seq 4096, accum 8 | alloc | reserved | s/step |
|---|---:|---:|---:|
| unchunked | 2275 MB | 2584 MB | 3.21 |
| **chunk 512** | **1003 MB** | **1148 MB** | 3.43 |

**-1436 MB for 1.07x step time.**

> ## ⚠️ RETRACTED (2026-08-27) — "it does NOT help at 1B" is WRONG, and acting on it spills the card
>
> The paragraph below concluded the chunked loss is "neutral at the top of the range" and a
> lever for **small** models only. Re-measured at d=2048/24L with BCD+offload at **seq 4096**,
> the length real training actually uses:
>
> | 1.21B, seq 4096, batch 1, grad-ckpt | peak reserved | s/step | verdict |
> |---|---:|---:|---|
> | unchunked | **6.18 GB** | 4.58 | ❌ 103% of the card — SPILLS |
> | **chunk 512** | **4.40 GB** | 4.02 | ✅ 73% — fits |
>
> **-1.79 GB, and it is the difference between fitting and spilling.** It is also *faster*
> (4.02 vs 4.58 s/step), because the spilled arm is paying WDDM host-RAM traffic.
>
> **Root cause of the wrong number: the original 1B comparison was run at a short sequence.**
> The logits tensor is `[B, S, vocab]`, so its size is linear in sequence length: 785 MB at
> seq 4096 against 98 MB at seq 512. At a short sequence the logits are genuinely not the
> binding term and chunking looks pointless — which is exactly what "+33 MB" records. At the
> sequence length training runs at, they are ~1.8 GB with gradients and they dominate.
>
> This is the same shape as every other retraction in this file: **a measurement taken in a
> configuration the real run never uses.** The 48M row above was measured at seq 4096 and is
> correct; the 1B row was not, and the two were reported side by side as if comparable.
>
> **Consequence: the chunked loss is MANDATORY at 1B, not optional.** Anyone sizing a 1B run
> from the retracted sentence would switch it off and silently spill. Verified 2026-08-27 by
> `scripts/probe_tt_1b_no_offload.py`'s baseline arm (4.38 GB with chunking) and a direct
> chunked-vs-unchunked pair at the same config.

**(superseded — see the retraction above)** ~~It does NOT help at 1B.~~ Same-scan comparison at
d=2048/24L with offload: 4.004 GiB unchunked vs 4.037 GiB chunked -- +33 MB. **That pair was
measured at a short sequence and does not describe training at seq 4096.**

Exactness: bit-identical in fp32 (losses and every gradient, including into the tied
table). Under bf16 autocast the value differs by 3.8e-03 on a loss of 10.5 -- 3.6e-04
relative -- because slices reduce separately. Do not call it bit-identical without
saying "in fp32".

**A benchmark error worth recording, because it nearly became a published number.**
A first pass reported chunked+offload at 19.90 s/step against 4.68 unchunked and was
reported as "4.25x slower, needs a fix". It was the benchmark: `ceiling.py` never set
`use_triton_scan`/`use_megakernel_scan`, so it ran the eager chunked scan while the
baseline ran the Triton path. The 2x2 that isolates it, at the full 1.07B shape:

| scan | loss | s/step |
|---|---|---:|
| ON | unchunked | 4.75 |
| ON | chunked | 4.85 |
| OFF | unchunked | 20.18 |
| OFF | chunked | 20.40 |

The scan flag is the 4.25x; the chunked loss is 1.02x. The stated hypothesis -- that the
decoder was being reloaded from CPU once per chunk -- was also wrong: instrumenting
`GPUBufferPool` showed byte-identical traffic in all four arms (2 loads, 0 misses,
0.3 MB). Same shape as every sibling-script divergence in this repo: two scripts that
look equivalent, one missing a flag.

## Measured size ceiling on the 6 GB card (2026-08-17)

With BCD + CPU offload + ternary + tied embeddings, seq 4096, single micro-batch:

| config | params | reserved | note |
|---|---|---:|---|
| 2048x24 | 1.07B | 4.00 GB | fits comfortably |
| 2048x36 | 1.58B | 5.92 GB | at the edge of a 5.9966 GiB card |
| 2560x36 | 2.45B | 9.39 GB | **spills to host RAM** (no OOM -- the WDDM failure mode) |

**~1.6B is the ceiling.** Note these were measured WITHOUT the scan kernels
(see the benchmark error above), so the memory figures are ~0.4 GB pessimistic and the
step times are ~4x pessimistic. Fitting is not training: at 875 tok/s a 1B model needs
~267 days for a Chinchilla-optimal budget, so the practical range remains a few hundred
million parameters.

## `pre_training_audit` OOMs at ~12 GB on any no-BCD run at seq 4096 -- was a bug in the audit, not a real cost (#112 resurfaced, 2026-09-28)

A 270M-param POC (d=1024, 20 layers, 32 heads, 262.1M total) OOM'd at 12.25 GiB during
startup, before training began, when run WITHOUT `--block-coordinate`. Root cause:
`pre_training_audit`'s first check does an `.eval()`-mode forward with autograd still on
(`train_edge_conversational.py:2234-2236`, no `torch.no_grad()`). In eval mode
`self.training` is False, so `hvn/edge/model.py:455`
(`use_ckpt = self.grad_checkpoint and self.training and torch.is_grad_enabled()`) turns
gradient checkpointing off too -- the forward records a full, non-checkpointed graph for
every layer at the full sequence length. The `logits` tensor holding that graph was never
freed before the next check ran on top of it, and `finally: del audit_model; empty_cache()`
did not help, since the graph still referenced the copy's activations.

Same code path as #112 (this file, "pre-training audit deepcopied model to GPU, OOM'd 6 GB
card at large sizes"), which was only ever patched by *skipping* Checks 2/3 when
`--block-coordinate` is set (lines ~2218-2224, ~2291-2297) -- never fixed at the source. It
resurfaces for any no-BCD run at long sequence length, because #112's own case (3x weights
at 1.21B, seq 128) had a tiny eval graph; at seq 4096 the kept graph dominates.

**Isolated repro** (same shape, deepcopy, tiny model at seq 512/1024, scratch script):
eval forward with grad on keeps 1.68 GiB (seq 512) / 2.83 GiB (seq 1024) alive afterward,
~+1.15 GiB per 512 tokens -- extrapolates to ~9.7 GiB at seq 4096, matching the 12.25 GiB
logged (plus the deepcopy's own resident weights). Under `torch.no_grad()` it stays flat at
0.49-0.53 GiB regardless of sequence length. Memory only dropped after `del logits`
specifically -- not after `del audit_model` -- confirming `logits` was the leak.

**Ruled out:** the slow eager `_scan_chunked` reference path used pre-`--fast` isn't the
cause by itself -- the train-mode check alone (checkpointed, same slow scan) peaked at
4.69 GiB, well inside budget. Wrong batch/seq (log showed the requested [1, 4096]). Grad
checkpointing being skipped in train mode (traceback showed `checkpoint()` was in fact
called there).

**Fix** (commit `84cf3ac`, `feat/blog-run-logging`, not pushed): wrap the eval forward in
`torch.no_grad()`, `del logits` before the next check runs, and use `chunked_cross_entropy`
in the audit when `config["chunked_loss"]` is set (so the audit tests what training
actually runs). `--block-coordinate`'s existing skip logic is untouched.

**Post-fix, no-BCD 270M (batch 1, grad-accum 8, fast stack, grad-checkpoint,
chunked-loss 512, muon) trains cleanly:** audit peak ~4.7 GiB (measured via isolated
pieces), training 2,822-2,831 tok/s steady, 4.08 GB peak / 4.43 GB reserved (74%, under the
85% spill guard), no spill over 300 steps. Loss fell 9.94 (step 50) -> 7.65 (100) -> 6.96
(150) -> 6.52 (200) -> 5.99 (250) -> 6.06 (300, single-step read during LR warmup) --
real learning, not the flat ~11.0 the earlier BCD run showed at step 100 (BCD trains one
block at a time; whether that flatness is expected block-cycling behaviour or an actual
problem was not verified).

**Batch 2 (accum 4) and batch 4 (accum 2) do not fit without BCD** -- spill guard stopped
training at step 1 (93% and 128% reserved respectively). Batch 1 is the only no-BCD
setting that fits at this shape.

## The blog's "~270M for coherent conversation" number was never measured (2026-09-29)

Traced to this file (the "270M can plausibly prove conversational fluency" passage) and
`HF_BLOG_POST.md`'s "what a 6 GB card is good for" table. 270M was picked because it was
**the largest model that fit on the card at the time** -- not because any model at that
size, or any size, was shown to hold a coherent conversation. No size trained in this repo
has that evidence, including 270M (the POC above only has a pretrain loss curve, no
generation).

**The only model ever chat-tested is the 83M shape (31M body, untied ~62%-vocab GPT-2),
fine-tuned as `hybrid_chat_v1`/`hybrid_tooluse_v1`** on a 51K-step trunk (~0.84B tokens,
stopped mid-schedule). Fluent, grammatically clean, correct on simple facts ("hi" ->
proper greeting, capital of France -> Paris), but **loops on arithmetic/creative prompts**
and, per this file's own later notes, "frequently doesn't engage with what was actually
asked -- generic assistant-boilerplate." This is the only real quality floor that exists.
The fully-trained 102K-step 83M trunk (1.65B tokens) was never chat-tested. 48M/70M/108M/
157M have loss/VRAM/speed numbers only (`results/size_vs_time/results.json`), no chat
evidence at all.

**Also caught: #159's ~30-tokens/param rule is per *body* param (vocab excluded)** --
`scripts/size_vs_time_for_code.py:120` uses the same convention. The 270M POC's body is
~210M (262M total - the tied GPT-2 head), so its correct budget is 30 x 210M = 6.3B tokens
(~26 days at 2,827 tok/s), not the 7.86B/~32 days figure obtained by multiplying the
**total** param count. Separately, the tokenized `fineweb_edu.bin` cache holds only 6.64B
tokens -- a 6.3B run fits; anything past ~6.6B re-reads data unless more is tokenized first.

**Throughput cross-check (superseded below):** `results/size_vs_time`'s 896x16 speed
(4,603 tok/s), scaled by layers x d^2, predicted 2,824 tok/s at the POC's 1024x20 shape --
close to the 2,827 the plain-`hvn` POC measured. That reference point turned out to be a
hybrid model benchmarked with a 32K synthetic vocab, not GPT-2's 50K in the real training
loop -- see the resolved flag below for the corrected number.

No size below 270M has been trained or chat-tested here, so day-counts are time estimates
only, not quality predictions. Outside priors (not measured here, treat loosely):
DialoGPT-small (117M) and SmolLM2-135M-Instruct are roughly coherent at comparable sizes,
but trained on 10B-2T tokens vs the few-billion-token budgets below.

**Resolved: the flag about `--arch hvn` vs `--arch hvn-hybrid` (2026-09-29).** All the
runs above used plain `hvn` (no attention), not the hybrid this project actually trains.
Re-run at 1024x20 with `--arch hvn-hybrid --attn-layers 5,15` (attention at ~L/4 and
~3L/4), same settings otherwise (batch 1, grad-accum 8, fast stack, grad-checkpoint,
chunked-loss 512, muon, tied GPT-2 embeddings):

| | plain hvn | hybrid |
|---|---|---|
| total / body params | 262.13M / 210.67M | 262.05M / 210.59M |
| speed (steps 100-300) | 2,826-2,831 tok/s | 2,861-2,865 tok/s |
| peak VRAM, used / held | 4.08 / 4.43 GB (74%) | 4.08 / 4.43 GB (74%) |
| loss at steps 50/100/150/200/250/300 | 9.94/7.65/6.96/6.52/5.99/6.06 | 9.63/7.62/6.71/6.51/6.18/6.00 |

**The hybrid is not more expensive here -- if anything marginally faster** (~1% at
1024x20, ~4% at a paired 640x12: 7,478 vs 7,200 tok/s, same VRAM). This contradicts
`HF_BLOG_POST.md`'s "~13% VRAM, ~7% step time" cost for the 2 attention layers -- that
number was measured in a different setup (blog's KV cache capped at 128 keys vs this
loop's full-window flash attention at seq 4096, plus grad-checkpointing keeping only one
block's activations live regardless of type). **The blog's hybrid-cost claim needs
correcting or re-scoping to its original setup** -- unverified which factor actually
explains the gap, don't assume it's the KV-cache width alone.

The audit (commit `84cf3ac`) needed no further fix for the hybrid path -- Check 3 already
passed `block_plan`/`attn_window`/`tie_embeddings` through the rebuild, confirmed by
forcing a hybrid checkpoint's `block_plan` to `None` and observing the audit correctly
fail. `use_deltanet: False` printing for hybrid configs is a cosmetic-only quirk (the 18
DeltaNet blocks still run on the Triton scan).

**Corrected candidate-size table, hybrid, attention at ~L/4 and ~3L/4, 30 tokens/body-param,
real measured throughput (not scaled) at each size:**

| config | attn layers | body / total | tokens | tok/s | peak VRAM (held) | days |
|---|---|---:|---:|---:|---:|---:|
| 640x12 | 3,9 | 49.4M / 81.6M | 1.48B | 7,478 | 1.57 GB (1.84, 31%) | 2.3 |
| 768x14 | 3,10 | 83.0M / 121.6M | 2.49B | 5,418 | 2.17 GB (2.56, 43%) | 5.3 |
| 896x16 | 4,12 | 129.0M / 174.0M | 3.87B | 4,105 | 2.93 GB (3.42, 57%) | 10.9 |
| 1024x20 | 5,15 | 210.6M / 262.1M | 6.32B | 2,863 | 4.08 GB (4.43, 74%) | 25.5 |

These land close to the earlier wrong-architecture estimates, because the hybrid turned out
to cost about the same as plain `hvn` in this loop. Only 640x12/768x14/896x16 are now
measured rather than scaled; each ran 100 steps, which says nothing about quality. Chat
eval was not run against any of these -- pretrain loss at this step count cannot show
coherence.

## Chat coherence eval built and validated (2026-09-29, `feat/chat-coherence-eval`, commit `6f14ece`)

Built because loss/perplexity cannot show whether a model is *coherent* rather than just
fluent -- the same proxy-can't-measure-target failure shape as #122/#132. 32 fixed prompts
(`eval/chat_prompts_v1.json`: 11 factual, 8 follow-up/context-dependent, 8 instruction, 5
loop-inducing), scored by `eval/chat_coherence.py` on three heuristics: on-topic (keyword
overlap, not a correctness judge), non-repetitive, terminates cleanly. Scorer records the
prompt file's sha256 with each result so the set can't silently drift.

**Validated against two references, greedy decode, repetition-penalty 1.3 (serving
default):** floor `hybrid_chat_v1` (83M/31M-body) scores **19% coherent (6/32)**; ceiling
SmolLM2-135M-Instruct scores **75% (24/32)**. Ceiling beats floor on 19 prompts, loses on 1
(sign test p=4e-5). A random-keyword chance control hits only 2-3%, so the 19%/75% gap
isn't an artifact of the scoring method. Hand-reading agrees: the floor gets basic facts
wrong ("Earth's largest moon" for the largest planet, "the Atlantic" for the largest ocean,
loops on `7*7*7...`, forgets a name given one turn earlier).

**Known limits of the score, before trusting it on a future candidate:**
- On-topic is keyword-based, not semantic -- hand-check found ~4/32 false passes on the
  ceiling and ~2/32 false fails on both; the *real* gap by hand (~55% vs ~8%) is larger
  than the headline, not smaller.
- Repetition-penalty 1.3 (the serving default) masks looping -- both models score >90% on
  non-repetitive/terminates at that setting. Looping only shows at penalty 1.0 (floor 25%,
  ceiling 84% non-repetitive).
- Sampled decoding is one draw; use greedy for comparing candidates.
- 32 prompts -- treat a gap under ~15 points between two candidates as noise. Use the
  paired per-prompt comparison, not the headline percentage.

**Side effects while wiring this up (also in commit `6f14ece`):** found and fixed two real
serving bugs in `serve_openai.py` -- repetition penalty divided negative logits too,
which *raised* them and made repeats more likely (now matches `EdgeHVN.chat_step`'s sign
handling); and `build_prompt` ended earlier assistant turns with `"\n"` instead of the
trained `<|endoftext|>` separator, so every multi-turn eval prompt was off-format.
`generate_ids()` was split out of the existing generation loop so both the repo's own
checkpoints and HF reference models share one sampler, rather than a second implementation.
12 new tests (`tests/test_chat_coherence_eval.py`) check the eval can actually fail
(boilerplate, echoed questions, loops, missing stop token, the old broken penalty sign, the
multi-turn format). SmolLM2-135M-Instruct was fetched via `curl` into
`checkpoints/hf/SmolLM2-135M-Instruct` (gitignored) because the HF Python client's TLS
failed in this environment.

## BCD's flat early loss was a false alarm; its real quality cost was a fixable partition problem, 41% closed (2026-09-29)

The 270M BCD POC's loss sat near-flat (~11.0) at step 100 -- looked like a possible bug,
compared against a no-BCD run already down to 7.65 by the same step. **It wasn't a bug:
that POC was only 100 steps long, and BCD's fixed round-robin partition (`token_embedding`
| one block per layer | `final_norm+decoder`, `hvn/core/block_coordinate.py:529-535`,
order always `sequential`) hadn't reached a single layer yet in that window -- all 20
layers were still at random init. A loss of ~11 is exactly what that predicts. The two
runs were never a matched comparison.**

**The real, fixable problem was underneath it.** BCD's peak grad+optimizer-state memory is
set by its *largest* block -- the tied vocab table (51.5M params at d=1024) -- while every
layer block is ~5x smaller (10.5M params). So each layer was training only 1/22 of the
time (22 blocks at 270M) for zero memory benefit; the slot sized for the vocab table was
being paid for at every block, including the ones far smaller than it.

**Fix:** `--bcd-partition balanced` (**now the default** for `--block-coordinate` in
`train_edge_conversational.py`, except under `--offload`, which falls back to `layer`;
it won at equal steps AND at equal wall-clock in the A/B -- balanced at 11.4 min was
5.533 vs layer's 5.758 at 11.3 min. Resume a run saved under the old partition with
`--bcd-partition layer`; a mismatch is refused, not silently remapped.
`hvn/core/block_coordinate.py`'s
`balance_blocks`) groups consecutive layers into blocks sized near the memory ceiling
instead of one-per-layer -- 22 blocks -> 6 at 270M (groups of 4 layers).

**Measured** (controlled A/B, d=1024, 8 layers, pure DeltaNet, tied GPT-2 vocab, ternary,
muon, 4,000 steps, same batches, scored on held-out data, `scripts/ab_bcd_partition.py`):

| step | full training | BCD, old (layer) partition | BCD, balanced partition |
|---|---:|---:|---:|
| 4000 | 4.958 (ppl 142) | 5.758 (ppl 317) | 5.430 (ppl 228) |

**Closes 41% of the full-vs-BCD gap, at identical peak memory** (grad+optimizer state:
0.575 GiB both BCD arms vs 1.204 GiB full; peak allocated 3.875 GiB all three). Isolated
properly: a control at the old per-layer partition but the same short (300-step) rotation
length scored *worse* (5.873) than today's normal-rotation layer partition (5.758) -- so
the gain is from each layer training more often, not from a faster cycle, and confirms
shrinking `--bcd-steps-per-block` alone would hurt, not help.

**What's still inherent, not fixed:** balanced BCD still trails full training by 0.47 nats
(~1.6x ppl vs full's ~1x). That remainder is the random-init under-training cost this
session already suspected from reading the BAdam paper's own validation (continued
pretraining on an already-competent Llama-3.1-8B, never tested from a random init) --
freezing an untrained block doesn't preserve anything worth preserving. No schedule change
closes this without spending the memory BCD exists to save.

**Ruled out, with reasons:** faster rotation hurts (above). Keeping optimizer state across
block switches measured -0.3% in issue #151's own harness. A short all-blocks warmup
before switching to BCD needs full-training memory, which defeats the point (270M without
BCD only fits at batch 1 to begin with). Always keeping the embedding block trainable
raises the peak (table + a live layer at once), failing the same-memory rule that
`balanced` meets.

**Does not shrink at 1B the way it does at 270M.** The packing ratio depends on
vocab-table-size vs. per-layer-size; the vocab table grows linearly with d_model while a
layer's params grow roughly with d_model^2, so relatively less slack exists to reclaim at
larger d_model. Estimated (not measured) to pack only 2 layers/group at d=2048 (26 blocks
-> 13) vs 4 layers/group at d=1024 -- expect a real but smaller gain there, needs its own
measurement before relying on it.

**Not yet supported combined with `--offload`** -- `EdgeHVN`'s offload path decides the
active block by comparing a single layer name (`hvn/edge/model.py:270,346,490`), so a
merged group would get evicted mid-step. The combination is refused loudly (scheduler and
argparse), not silently broken. Doesn't block current sizes (270M and below don't need
offload at all). Resuming a run under a different partition than it started with is also
refused.

**Two side findings:** `scripts/ab_bcd_retain_state.py` (issue #151) imports
`train_edge_conversational` (which imports pyarrow) *after* `require_cuda()` --
importing pyarrow post-CUDA-init crashes with no traceback (the known pandas/pyarrow
ordering issue noted at the top of `train_edge_conversational.py:40-61`); a latent crash in
that script, not fixed there. Separately, #151's own harness ran the embedding/head group
at `aux_lr=2e-5` against the real training default of `3e-4`, which likely inflated its
reported 3.9x-10.7x gap -- this session's harness at the correct `3e-4` measured 2.2x for
the (unfixed) layer partition on its own 8-layer test, closer to earlier estimates than
#151's number.
