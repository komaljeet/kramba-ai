# #216 Streamed full backprop at 1B: paper estimate (no GPU run)

Date: 2026-11-19. Nothing was run on the GPU. Every number below is measured (source named),
arithmetic from sizes (inputs shown), or marked **unmeasured**. The arithmetic is in
`scripts/estimate_streamed_fullbp_216.py`; its output is `estimate_numbers.txt` in this folder.

## Verdict

- **Speed: passes.** Likely step 21.8 s (low 19.4, high 25.5). That is 2.2x the BCD step of
  9.89 s, well under the ~40 s bar (4x). 925,000 steps = 234 days (208 to 273).
- **VRAM: fits on paper.** About 3.8-4.1 GB allocated at the peak, ~1 GB under today's 1B k4 run
  (4.99 GB allocated, 82% reserved). Not measured.
- **Host RAM: does NOT fit on this laptop at full precision.** The trainer would need ~13.6 GB.
  Today it can grow to ~8.3 GB before free RAM hits zero (10.6 GB with 0 DataLoader workers).
  Short by 3-5 GB.
- **So: worth building only if RAM is solved first.** Two ways: (a) 32 GB RAM in the laptop
  (then the exact design fits with ~11 GB spare), or (b) store optimizer momentum, master
  weights and checkpoints in bf16 (needs 8.7 GB, fits only with 0 workers, 1.8 GB margin, and
  each cut needs a loss A/B first).
- Payoff if it works: at the 208M penalty (BCD needs 6.7x the steps, #214), full backprop
  reaches the loss BCD gets at 925,000 steps after ~138,000 steps = **~35 days instead of 106**.
  The penalty at 1B is unmeasured.

## 1. What is already measured

| quantity | value | source |
|---|---|---|
| H2D, pinned staging, side stream (real trainer, profiled) | 13.06 GB/s (15.49 GB in 1.187 s) | `results/bcd_stall_200/stall_steps_ctrl.jsonl`, token_embedding step 3 trace |
| H2D, pinned, one 84 MB copy | 13.44 GB/s | `results/frozen_layer_203/peak.json` |
| H2D, pageable, synchronous on compute stream | 10.74 GB/s (16.84 GB in 1.568 s) | same #200 control trace (`h2d_on_compute`) |
| D2H (device to host), any kind | **unmeasured** | nothing in the repo |
| H2D and D2H at the same time (duplex) | **unmeasured** | |
| PCIe volume BCD streams today, hidden | 13.5-32.3 GB/step, stall 0.056 s/step | docs/state.md #200 |
| Trainable DeltaNet layer, per step (8 mb): forward / recompute+backward with weight grads | 0.130 s / 0.430 s | `perlayer_default.txt` (copied from main's local `results/speed_hunt_213/`, #213) |
| Trainable attention layer (6, 18): forward / recompute+backward | 0.163 s / 0.627 s | same |
| Loss with trainable tied decoder (embedding turn): loss fwd + loss recompute + embed | 0.455 + 1.669 + 0.063 = 2.19 s | `phases_default.txt` (#213) |
| Muon step, per 4-layer block (168.2-168.4M params) | 0.521-0.530 s; 0.514 s alone | `phases_default.txt`; `muon_ns.json` (#213) |
| Muon per parameter on GPU | 3.13 ns (3.16 s / 1,009.5M) | arithmetic |
| AdamW, embedding (102.9M) | 0.052 s = 0.51 ns/param | `phases_default.txt` |
| CPU optimizer speed | **unmeasured** | nothing in the repo |
| Params | 1,113,059,793 total; Muon 1,009,516,544 (210 matrices); AdamW 103,543,249 (embedding 102,926,336) | `results/bcd_memory/ga8_k4.log` |
| System RAM | 16.42 GB (15.29 GiB), 2x8 GB DDR5-5600 | `results/frozen_codes_204/time_base*.jsonl` `sys_total`; docs/training.md |
| Trainer RAM today (codes off, 4 workers) | working set max 5.09 GB, lowest free 3.24 GB | `results/frozen_codes_204/time_base*.jsonl` |
| Trainer RAM with fp8 codes (default) | working set max 7.25 GB, lowest free 1.06 GB | `results/frozen_codes_204/time_codes*.jsonl` |
| DataLoader worker RAM | ~0.56 GB each (7.16 vs 6.04 GB total RSS, 4 vs 2 workers) | docs/state.md 2026-10-23 |
| VRAM today, 1B k4 | 4.99 GB allocated, 5.30 GB reserved (82%) of 6.44 GB | `results/frozen_codes_204/time_base*.jsonl` |
| VRAM spill onset | ~91-92% reserved; guard 85% | docs/state.md #194 |
| Full vs BCD at 208M | full 6.69 s/step, BCD 4.10 s/step; BCD needs 6.7x the steps at 4000 (lower bound) | `results/bcd_penalty_214/summary.txt` (!270) |

Gaps that matter: D2H speed, duplex, CPU optimizer speed, the BCD penalty at 1B.

## 2. Code: what exists, what is new

How BCD works today (main 692d130):
- `GPUBufferPool` (`hvn/core/block_coordinate.py:163`) streams frozen layers' weights H2D through
  bounded pinned staging buffers on a side stream (`prefetch`, line 470). It is **read-only**:
  eviction just points `p.data` back at the CPU home, no copy out (line 201).
- Frozen 2-D weights are parked in **bf16 with no fp32 master** (`park_dtype_for`, line 272;
  `_activate`, line 1287). Only the active block is restored to fp32 on the GPU (line 1248).
- `EdgeHVN._offload_wrap_checkpointed` (`hvn/edge/model.py:333`) checkpoints every layer and
  reloads its weights inside the checkpoint closure for the recompute; backward prefetches the
  next recompute's weights (lines 403-430, 716-723).
- The trainer loop is micro-batch-major: 8 x (forward + `loss.backward()`), then
  `clip_grad_norm_(max_norm=1.0)` and `optimizer.step()` (`train_edge_conversational.py:3654-3712`).
- The tied embedding is pinned on the GPU in fp32 and never offloaded (#169, line 1255).
- Muon keeps one fp32 `momentum_buffer` per matrix (`hvn/optimizers.py:131`); BCD deletes it
  at every switch.
- `_forward_offload_inference` (`hvn/edge/model.py:452`) already walks the model segment by
  segment for eval (#213).

Design chosen for the estimate: **layer-major** (call it L2). For each layer, run all 8
micro-batches before moving on. Then each layer's weights cross PCIe a few times per step,
not 16 times, and its gradient finishes on the GPU and is used at once.
- Forward: per layer, stream its fp32 master in, run 8 micro-batches. Every 4th layer's input
  (the segment input, 8 mb x 33.5 MB fp32) goes to host RAM.
- Backward, per 4-layer segment from the top: stream the segment input back; re-forward layers
  1-3 of the segment (no grad) to rebuild their inputs for all 8 mb (kept on GPU); then per
  layer from the top: recompute + backward for each of 8 mb, fp32 grad accumulated on GPU;
  stream momentum in, Muon step on the GPU, stream master + momentum out; free the grad.
- **Gradients live only on the GPU, one layer at a time.** They never go to host RAM.
- Whole-model checkpointing = two levels: segment inputs in host RAM, layer inputs rebuilt
  per segment. Costs one extra forward for 18 of 24 layers (2.41 s). L1 (park every layer's
  input) skips that but needs 6.4 GB of RAM for them.

Rejected: micro-batch-major (today's loop with every layer trainable). Grads must
accumulate over 8 micro-batches but cannot stay on the GPU (4.04 GB fp32), so they would go
out 8 times per step (16-32 GB D2H) and sit in host RAM (+4.04 GB). Worse on RAM and PCIe.

What is new:
1. fp32 host master for every weight (today: bf16, no master). Or bf16 + stochastic rounding
   (smaller, but a numerics change needing an A/B).
2. A D2H write-back path in the pool: pinned staging, side stream, event before the host copy.
3. A layer-major training driver: per-layer `torch.autograd.grad` on a recomputed layer, hidden
   grads for 8 mb, segment inputs parked to host. Replaces the trainer's micro-batch loop for
   this mode. Biggest piece.
4. Per-layer optimizer step inside backward, with momentum streamed in and out.
5. **Global grad clipping cannot run before the updates** (layer 23 is updated before layer 0's
   grad exists). Options: use the previous step's norm, clip per layer, or two passes (+4 GB
   RAM for parked grads, which does not fit). Needs a decision and a loss A/B.
6. Checkpoint save/resume of host masters and momentum; eval reading fp32 homes.
7. Turn off the #204 frozen fp8 codes (weights change every step; it also frees 1.7 GB RAM).
8. Tests: a small model, streamed vs plain full backprop, equal over a **sequence** of steps,
   with a must-fail partner (e.g. a layer whose update is skipped).

## 3. Budget at 1B (8 x 4096 tokens per step)

### Compute, s/step (measured per-layer times, all layers trainable)

| part | s | derivation |
|---|---:|---|
| forward, 24 layers | 3.19 | 22 x 0.130 + 2 x 0.163 |
| re-forward in segments (L2 only) | 2.41 | 16 x 0.130 + 2 x 0.163 (layers 6, 18 are re-forwarded) |
| recompute + backward with weight grads | 10.71 | 22 x 0.430 + 2 x 0.627 |
| loss + embedding, trainable decoder | 2.19 | embedding-turn phases |
| Muon, 210 matrices, GPU | 3.16 | sum of the 6 measured block optim times |
| AdamW embedding | 0.05 | measured |
| clip, glue | 0.03 | measured, each <= 0.012 |
| **total L2 / L1** | **21.73 / 19.33** | |

Cross-check: the same per-layer numbers rebuild the measured BCD steps (e.g. the embedding turn's
backward 7.57 s = 0.416 + 21 x 0.290 + 2 x 0.534).

Optimizer on CPU instead: **unmeasured**. Muon is ~6e13 FLOP/step (5 Newton-Schulz rounds,
~6 x 2048 x 1.01e9; it runs at ~20 TFLOP/s on the GPU, in line with the card's 24.6 bf16 peak).
To fit in the 40 s budget the CPU would need to sustain >3 TFLOP/s. The GPU does it in 3.16 s,
so CPU Muon is not needed and not planned.

### Transfers, GB/step (L2)

- H2D 18.6 = fp32 masters 3 times (forward, re-forward, backward: 3 x 4.04) + momentum 4.04 +
  segment inputs 1.61 + AdamW state 0.83.
- D2H 10.5 = segment inputs 1.61 + masters 4.04 + momentum 4.04 + AdamW state 0.83.
- H2D busy time at the measured pinned 13.06 GB/s: 1.42 s, 7% of the compute time. BCD today
  hides up to 32.3 GB/step (#200).

### Step time

| case | step | x BCD | days for 925,000 | assumptions |
|---|---:|---:|---:|---|
| low | 19.4 s | 1.96 | 208 | L1 (every layer input in RAM: needs 32 GB RAM), copies hidden as today, +0.1 s stall |
| **likely** | **21.8 s** | **2.21** | **234** | L2, copies on side streams hidden behind compute as BCD's are today, +0.1 s stall (#200 measured 0.056) |
| high | 25.5 s | 2.58 | 273 | L2, every copy serial with compute at the pageable 10.74 GB/s (D2H assumed the same: unmeasured), +4.3% whole-run overhead (10.32 vs 9.89 s in #209/#215) |
| BCD k4 today | 9.89 s | 1.00 | 106 | #210 |

What overlaps with what: H2D of layer i+1 and D2H of layer i-1 run on side streams while layer
i computes, as BCD's prefetch does today. Muon for layer i runs on the GPU after its backward;
its momentum load overlaps the backward. The only serial pieces are the first load of each pass.

Break-even: the step stays under 40 s unless D2H is slower than **0.64 GB/s** (even with H2D
also serial). So the speed verdict does not depend on the unmeasured D2H rate in any likely way.

Not counted: Muon with 3 Newton-Schulz steps (-0.18 s per block measured in #213, ~-1.1 s/step,
needs a loss A/B); batching the 8 micro-batches into larger GEMMs in layer-major order.

### Host RAM, GB

| item | exact (fp32) | lean |
|---|---:|---:|
| rest of trainer process (5.09 working set - 2.02 bf16 parked weights) | 3.07 | 3.07 |
| layer masters (1,009.5M) | 4.04 | 2.02 (bf16 + stochastic rounding) |
| Muon momentum | 4.04 | 2.02 (bf16) |
| AdamW moments (103.5M x 8 B) | 0.83 | 0.83 |
| segment inputs (6 x 8 mb x 33.5 MB) | 1.61 | 0.81 (bf16) |
| **total, L2** | **13.59** | **8.74** |
| total, L1 (all 24 layer inputs, 6.44) | 18.42 | |

Available to the trainer on this PC: 8.33 (5.09 working set + 3.24 lowest free, 4 workers);
10.57 with 0 workers. Everything else on the PC used ~8.1 GB in those runs.
- Exact L2: short by 5.3 GB (3.0 GB with 0 workers). **Does not fit.**
- Lean L2: fits only with 0 workers, 1.8 GB margin. Three numerics changes, each unmeasured.
- With 32 GB installed: ~24.8 available. Exact L2 fits (11 GB spare); L1 fits too.
- Pinned (page-locked) memory limits on Windows for the staging buffers: **unmeasured**
  (the #118 bug shows pinning large amounts fails; the bounded staging pool avoids it).

### VRAM at the backward peak, GB (L2, estimate, unmeasured)

| item | GB |
|---|---:|
| embedding table + its grad (fp32, tied, stays on GPU) | 0.82 |
| current layer master + grad + momentum, fp8 copies, next layer in, previous layer out | 1.09 |
| segment layer inputs, 4 layers x 8 mb | 1.07 |
| hidden-state grads in + out, 8 mb | 0.54 |
| one layer's recompute + backward working set (#202: ~0.3 GB kept per layer; backward transients unmeasured) | 0.3-0.6 |
| **total allocated** | **3.8-4.1** |

At today's reserved/allocated ratio (1.06) that is 4.1-4.4 GB reserved, 63-68% of the card,
under the 85% guard. Today's 1B k4: 4.99 allocated, 82% reserved.

## 4. If RAM is solved: minimal build plan

1. Gate first (no GPU): decide RAM. Either 32 GB installed, or a 208M loss A/B (#214 harness,
   full backprop, ~1.1 h per 600-step arm) of bf16 momentum + bf16 master with stochastic
   rounding + previous-step clip vs the #214 full arm. Without one of these, stop.
2. Pool: add fp32 homes and a D2H write-back (pinned staging, side stream, event).
3. Layer-major driver for one model (`EdgeHVN`), opt-in flag, BCD path untouched.
4. Per-layer Muon/AdamW step in backward, momentum streamed; clip policy from step 1.
5. Save/resume and eval for host-resident state.
6. Tests on a tiny model: equal to plain full backprop over 10+ steps (sequence, not one step),
   with a must-fail partner; RAM and VRAM asserted.

## 5. One cheap GPU check (NOT run; only after the maintainer approves)

`scripts/bench_streamed_segment_216.py` (to be written): the real 1B layers 4-7 (one attention
layer), production `--fast` stack, 8 micro-batches x 4096, layer-major forward, segment input
parked to host, re-forward, per-layer recompute + backward, Muon per layer, masters and
momentum streamed in and out on side streams. 5 timed steps after 2 warm-up steps.

- Predicted segment time: forward 0.553 + re-forward 0.423 + backward 1.917 + Muon 0.521 =
  **3.41 s**. Pass if measured is at most 1.2x that (full step <= ~26 s).
- Also records: D2H rate (pinned and pageable), whether H2D and D2H overlap, peak VRAM
  (predicted 4.1-4.4 GB reserved), process RAM.
- Falsifier (must move the number): rerun with every copy forced synchronous on the compute
  stream. The step must grow by about bytes / rate (~0.5 s for this segment). If it does not,
  the overlap measurement is blind.
- Cost: ~10-15 min of GPU time. Writing the script: about half a day.
