**Pre-training a 1.11B LLM on a 6 GB Laptop GPU**

[DOI 10.5281/zenodo.22725431](https://zenodo.org/records/22725431)

**Laptop Specs**

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU — 6141 MiB VRAM, 20 SMs, compute cap 8.9 (Ada Lovelace), power-limited to 120 W |
| Measured peak bf16 GEMM | 25.6 TFLOP/s (measured, not a spec-sheet number) |
| CPU | AMD Ryzen 7 7840HS |
| System RAM | 15.29 GB usable (2x8GB DDR5-5600, dual channel) |
| OS | Windows 11 Home, build 10.0.26200 (WDDM, not Linux) |
| PyTorch | 2.12.1+cu126 |

This is a power-limited laptop GPU on Windows with notably tight system RAM relative to its VRAM. Don't assume these numbers transfer to a desktop RTX 4050, a different OS, or more system RAM without re-measuring.

**Before / after — same hardware, one week of real fixes (2026-08 to 2026-10)**

| | August 2026 | October 2026 (now) |
|---|---|---|
| 1B step time | 21.81 s/step | **12.335 s/step** (-43%) |
| Throughput | 1,579 tok/s | **2,656 tok/s** |
| **BCD's own quality cost** | **about 4x worse perplexity** than full training (one block trains at a time, balance was naive) | **about 1.6x worse perplexity** (balanced block partitioning, issue #164) |
| 1B pretrain time | about 375 days | **about 132-211 days**, depending on how the BCD quality cost above is counted |
| Offload correctness | silent weight-corruption bug, present undetected for 76 days (2026-07-26 to 2026-10-10) | found, fixed, and validated end-to-end with real hard-kill-and-resume tests |
| Peak VRAM at 1.11B | 4.51 GB | **5.24 GB** (81.4% of the card) — up, not down; fused kernels/caches add some transient buffers even while speeding things up |

The BCD quality row is easy to miss but matters as much as the speed numbers: the **about 4x to about 1.6x figure is a real cut in how much worse the memory-saving training mode is** compared to training the full model at once, not a speed change at all. It came from changing how BCD splits the model into trainable blocks — one big slot sized for the vocabulary table wasted memory for every other block; grouping layers into balanced-size blocks reclaimed it, which let more layers train "live" at once for the same VRAM budget.

**12.335 s/step is a full-BCD-cycle average (with `--fp8-linear`), not every step.** Per block it ranges from about 17.6 s/step (the token_embedding block, slowest, trained first) down to about 6.95 s/step (the last block). A fresh launch starts on the slowest block, so the first about 100 steps genuinely run at that slower end — that is expected, not a regression. It steps down at each block switch and averages to 12.335 s/step over a full about 1,300-step cycle (about 4.5 hours). Without `--fp8-linear` the same cycle averages 13.294 s/step (18.9 down to 7.5 s/step per block).

**Correction (2026-10-11):** the CPU weight-offload path used in the runs below had a bug, present from 2026-07-26 to 2026-10-10, that could swap weights between same-shaped matrices during training — fixed in commit 2f94d7a, and independently confirmed on the original historical code. The **VRAM/fits-in-X-GB numbers throughout this post are unaffected** (that bug was about weight correctness, not memory). The **loss-curve and "it trains" claims for the offloaded runs below are not** — they need re-running on fixed code before being trusted again. See the repo's docs/state.md for the full writeup.

**Update (2026-10-14):** that offload+BCD path has since been validated end-to-end — a real run hard-killed and resumed mid-training three times, plus a real 1.1B fresh-build and resume — and four more real bugs it had were found and fixed along the way (details in docs/state.md). Separately, the step time below was stale at the time: four more real speed levers had just landed on top of the ones already in this post, and 375 days became 154-246 days depending on how you count BCD's own quality cost (explained where it first comes up below). The "nothing left, 20-25% at most" line further down in this post turned out to be wrong, too — it's corrected in place rather than deleted, so the mistake stays visible.

**Update (2026-10-21):** that 154-246 day figure was itself never recomputed after a fifth speed lever (four bit-identical kernel fusions) landed on 2026-10-17, bringing the measured step from 14.385 down to **13.294 s/step**. Redone with the current number: 154-246 days becomes **142-227 days**. Every number in this post below has been updated to match. This project has now corrected its own "current" number three times in six weeks — treat every figure here as a thing to re-check against docs/state.md, not a constant.

**Update (2026-10-23):** a sixth lever landed — a fused FP8 matmul path (**--fp8-linear**) using the GPU's real FP8 tensor cores, not just a faster cast. First measured at 6.14% faster on one block (the slowest), then confirmed over a real full 13-block cycle after a host-RAM resume bug (unrelated to the model, fixed) was worked through: **7.81% faster** (2,464 to 2,656 tok/s, 13.294 to 12.335 s/step), quality unaffected. 142-227 days is now about **132-211 days** — this is the confirmed number, not a projection. Every s/step, tok/s and day-count figure in this post below has been updated to match, the same way the 2026-10-21 update below did.

I looked everywhere to be able to **pre-train** an LLM on my low-end GPU and could not find one. Everywhere I looked, the same conclusion was reached: "You need an A100", "You can only train a 20M toy model on your hardware". So being 'broke', I did the only thing I could do — subscribe to Claude and try to find a way... after much prompting and testing, I finally stitched together the stack that allowed me to **pre-train** a 1B LLM on my laptop. After much testing, we are at **100.6% of the card's measured GEMM peak** (I don't know how 100.6% is even possible, I'm not a scientist, just a broke guy).

Though I can **pre-train** a 1B LLM on 6 GB VRAM, it still takes about **142 days** (accepting block-coordinate descent's own quality cost as measured) or **about 227 days** (if you want the output to match full-training quality instead) to finish **pretraining** it properly — down from 375 days measured in August, after real speed and quality fixes landed since. More details below.

**The Headline Numbers (Measured on RTX 4050 6 GB)**

| Metric | Value |
|--------|-------|
| **Model size** | 1.11B parameters (pretraining, not inference) |
| **Peak VRAM** | 4.51 GB, 1.5 GB to spare (August); **5.24 GB measured 2026-10-17**, 81.4% of the card — up, not down, from this week's fused kernels/caches |
| **Throughput** | 1,579 tokens/sec at 4,096 context (August measurement); **2,656 tok/s confirmed 2026-10-23** with `--fp8-linear` |
| **Pretraining time (Chinchilla-optimal)** | about 375 days for 36B tokens (August); **about 132-211 days now**, see the update note above |

**The 5-Technique Stack That Made Pretraining Possible**

No new algorithms — every piece is published and attributed. The work was making them run together and measuring what that costs.

| Technique | What it removes | Saving |
|-----------|-----------------|--------|
| **Block-coordinate descent (BAdam)** | Optimizer state for every block but the active one | about 8x on optimizer state |
| **CPU weight offload** | Inactive blocks leave the GPU entirely | Scales with depth |
| **Ternary weights (BitNet b1.58)** | fp32 weights to {-1, 0, +1} (1.58 bits), per-matrix fp32 scale | **2.33x compression** (weight tensors only) |
| **Tied embeddings** | One vocabulary table instead of two | 103M params at 1B scale |
| **Gradient checkpointing** | Stored activations — recomputed instead | Activations to about 50 MB |
| **Chunked cross-entropy** | Logits materialized slice-by-slice, recomputed in backward | about 1.5 GB saved at 4,096 x 50K vocab |

Ternary quantization details (BitNet b1.58): what 2.33x means is weight tensors only, not total VRAM — fp32 (32 bits) becomes ternary stored as int8 (8 bits) plus a per-matrix fp32 scale, theoretical 20x (log base 2 of 3, about 1.58 bits), practical 2.33x after scale overhead. About a third of weights round to exactly 0 (35% sparsity), spread across rows rather than collapsing whole rows. This only works trained from scratch with a straight-through estimator — post-hoc quantization loses the learned dynamics, so **--ternary** is on by default in production runs. It also buys no training-VRAM win: fp32 shadow weights stay resident for gradient updates, so ternary only saves checkpoint size and inference memory. Fresh layers can zero out before learning (Issue #109), fixed by warm-starting the trunk from a pre-trained checkpoint. Finer, group-wise scales were tested (Issue #146) and lost to one scale per tensor — the storage ratio improves with width, but quality doesn't.

Two of these did not previously work together. Tied embeddings share one tensor between the input lookup and the output layer, and the offload system rejected that outright — it assigns every parameter to exactly one block, and this tensor belongs to two. The combination was guarded off as unsupported.

The fix (Issue #157): pin the tied table — a parameter reachable from two modules is never evicted, because the forward pass uses it at both ends and no single block's residency window covers both uses. Cost: one vocabulary table held resident (about 250 MB at 1B scale). When combined with chunked cross-entropy (**--chunked-loss 512**), the previously OOM combination (tied embeddings + CPU offload) now fits without OOM — a 5-step fp32 test at 1.11B parameters ran at **79% of 6 GB = 4.85 GB**. The pinned tied table (about 206 MB bf16) plus chunked loss (about 1.5 GB saved) makes the four-technique stack (BCD + offload + tied + chunked) fit together in memory. (2026-10-11: "fits" is confirmed; "trains correctly" at this combination and scale was never separately checked before an unrelated offload bug was found and fixed on 2026-10-10 — see the correction above.)

**Architecture**

The model is a gated DeltaNet hybrid (10 DeltaNet blocks plus 2 sliding-window attention layers at layers 3 and 10). This architecture is a reimplementation of published work: the canonical gated DeltaNet update (GLA / Mamba-2 style decay gate), a per-head output RMSNorm normalizer (proper weighted average readout), and an output gate plus short causal depthwise convolution (Mamba / RWKV-7 style). Constant-memory inference — no KV cache, fixed state — is the defining property of this entire model family (Mamba, RWKV, RetNet), not something this project invented.

**Why Hybrid Attention? Inference-Time Exact Recall**

Pure recurrence (DeltaNet, Mamba, RWKV) compresses all history into a fixed state — great for throughput, but cannot recall exact tokens from context. The hybrid adds 2 sliding-window attention layers (at layers 3 and 10, window=128) specifically to solve this.

Measured on a 200-trial synthetic recall probe (invented entities, zero knowledge confound):

| Model | Near-window recall (<=128 tokens) | Far-window recall (>128 tokens) |
|-------|----------------------------------|---------------------------------|
| Pure DeltaNet trunk (10 layers) | 4-5% | 0.5% (noise) |
| **Hybrid (+2 attention layers, step 8K)** | **35.5%** (**9x trunk**) | **8.5%** (17x trunk) |

Key insight: the attention window (128 tokens) gives exact token recall inside the window — something the recurrent state mathematically cannot provide. Beyond the window, both recurrence and windowed attention fall to noise floor (about 0.5%).

This means at inference: the hybrid can read and quote exact passages from the last about 128 tokens of context, while pure recurrent models can only give a lossy semantic gist. For coding agents, tool-use, and grounded QA, this is the difference between "it hallucinates the function signature" and "it copies the exact signature from context."

The attention layers add only about 13% VRAM (2 TransformerBlocks with windowed KV cache capped at 128 keys each) and about 7% step time at pretraining — a small price for exact recall where it matters most.

**What You Get Without Any of This (Same Card, Standard Script — Pretraining)**

| Model | Standard Script | With This Stack |
|-------|-----------------|-----------------|
| 157M | 3.38 GB — fits | fits |
| 243M | 4.80 GB — fits | fits |
| 434M | 9.04 GB — spills | fits |
| **1.07B (recurrent)** | far past the card | **4.00 GB** |
| **1.07B (plain transformer)** | far past the card | **4.04 GB** |
| **1.58B** | far past the card | **5.92 GB — the ceiling** |

About 243M on a standard script, about 1.58B with the stack — roughly 6.5x. Both numbers measured on the same 6 GB card at 4,096 context, stopping at the first spill into system RAM (on Windows/WDDM, exceeding VRAM does not error — it silently gets 4-15x slower).

What the table does not show: a transformer and the recurrent hybrid need almost identical memory at these sizes (4.80 GB vs 5.09 GB at 243M), and a plain 1.07B transformer runs under the same stack in 4.04 GB. The 6.5x comes from the pretraining stack, not from the architecture — so it applies to whatever model you are pretraining, including an ordinary transformer.

**What This Means for Your GPU (Projected VRAM Ceilings — Extrapolated, Not Measured)**

**Only the 6 GB row below is measured.** All other rows are linear extrapolation from the 6 GB ceiling (about 1.58B) assuming fixed-ratio savings. Real VRAM ceilings will vary by GPU architecture, WDDM behavior, and batch size — treat the rest as upper-bound estimates.

The stack scales with VRAM. On a 6 GB card we measured about a 1.58B ceiling. Here's what the same stack projects for larger cards (linear extrapolation from measured points, stopping at first WDDM spill):

| GPU VRAM | Max Model (Recurrent Hybrid) | Max Model (Plain Transformer) | Pretrain Time (Chinchilla) |
|----------|------------------------------|-------------------------------|----------------------------|
| **6 GB** (RTX 4050 laptop) | **about 1.58B** | **about 1.58B** | about 132-211 days @ 1.1B (was about 375, Aug measurement) |
| **8 GB** (RTX 3070/4060 laptop) | **about 2.2B** | **about 2.2B** | about 61-97 days @ 1.5B |
| **12 GB** (RTX 3060/4070) | **about 3.5B** | **about 3.5B** | about 28-45 days @ 2.5B |
| **16 GB** (RTX 4080) | **about 4.8B** | **about 4.8B** | about 15-24 days @ 3.5B |
| **24 GB** (RTX 3090/4090) | **about 7.5B** | **about 7.5B** | about 6-11 days @ 5B |

(Pretrain-time column rescaled 2026-10-21 by the same factor the 6 GB row's own measurement moved by since the last pass — not independently re-extrapolated, since nothing on these other cards has actually been measured. This stacks a speed-lever update on top of an already rough linear extrapolation, so treat these four rows as even more approximate than the warning above already says.)

Key insight: the 6.5x multiplier (standard script to this stack) applies at every VRAM tier because the savings are proportional. BCD saves about 8x optimizer state (scales with params). Offload moves inactive blocks to CPU (scales with depth). Ternary gives 2.33x weight compression (fixed ratio). Tied embeddings save 103M params at 1B (fixed absolute). Chunked loss saves about 1.5 GB at 4K context (fixed absolute).

The ceiling is linear in VRAM because the dominant terms (weights + optimizer state) scale with model size, and our techniques compress them by fixed ratios. A 24 GB card doesn't just fit "a bit more" — it fits about 5x the parameters of a 6 GB card.

What changes at larger VRAM: you can drop **--offload** (no PCIe bottleneck) for about a 20% speedup; you can increase **--batch-size** for better GPU utilization; you hit the compute wall sooner (matmuls at 100% ceiling); and at 24 GB, a 7B pretrain is about 18 days — actually finishable.

**System RAM Requirements (Often Overlooked)**

| GPU VRAM | Min System RAM | Recommended | Why |
|----------|----------------|-------------|-----|
| **6 GB** | 16 GB | 32 GB | Offloaded weights + dataset + OS |
| **8 GB** | 16 GB | 32 GB | Offloaded weights grow with model depth |
| **12 GB** | 24 GB | 48 GB | Less offload needed, but dataset larger |
| **16 GB** | 32 GB | 64 GB | Can drop offload, but bigger datasets |
| **24 GB** | 48 GB | 96 GB | Full model in VRAM, but dataset + OS |

Two reasons system RAM matters. First, CPU offload (**--offload**): inactive blocks' weights are evicted to system RAM. At 1.11B with offload, about 3-4 GB of weights sit in system RAM — fine with 16 GB system RAM and 6 GB VRAM, but an 8 GB system will OOM on the host. Second, WDDM silent spill (Windows): exceeding VRAM does not raise OOM — the driver silently spills to system RAM over PCIe at 4-15x slower. We measured 82 s/micro-batch when a 2.45B model spilled. If your system RAM is full, the spill has nowhere to go, which is a hard crash.

**Rule of thumb:** system RAM should be at least 2x GPU VRAM for offload runs, at least 1.5x for no-offload runs. The 16 GB in my laptop was the minimum for 6 GB VRAM plus offload.

**Where the Memory Actually Goes (Measured at 48M, 4,096 Context)**

| Item | Size |
|------|------|
| **Computing the loss** — 4,096 positions x 50,257 vocabulary | about 1,500 MB |
| CUDA context, kernels, workspaces | about 400 MB |
| Weights (ternary) | about 960 MB |
| Optimiser state (BCD: one block at a time) | about 200 MB |
| Layer activations (grad checkpoint) | about 50 MB |
| **Pinned tied table (if tied)** | **about 250 MB** |

The largest consumer is not the model — it's the cross-entropy over the vocabulary (about 3x the weights). Chunked loss (512 positions at a time) recomputes logits in backward instead of keeping them, saving about 1.5 GB. This is what made the previously OOM combination (tied plus offload) work.

**The Trade-off: Tied Embeddings vs Offload**

| Config | Params | Tied | VRAM | Status |
|--------|--------|------|------|--------|
| Blog claim (no BCD offload) | 1.07B | yes | 4.00 GB | Target |
| 1.11B run (untied) | 1.11B | no | 3.48/4.51 GB | 1,350 steps, ran without OOM — loss curve needs re-verifying, see correction above |
| **Tied + offload + chunked_loss** | **1.11B** | **yes** | **4.85 GB (79%)** | **5 steps fp32, fits without OOM (fixed) — not a trained/graded run** |
| Tied + offload (no chunked) | 1.07B | yes | >6 GB | OOM (pinned 250 MB + overhead) |
| 57M test | 57M | yes | Works | All 5 techniques integrate |

**The Hard Reality: Fitting Is Not Training**

A 1B model ran at about 1,100 tokens/sec here in August; **2,656 tok/s confirmed 2026-10-23** with the FP8 matmul lever. Pretraining one properly still needs roughly 30 billion-plus tokens — **about 132-211 days** now, down from 375, depending on how you count block-coordinate descent's own quality cost. The ceiling this stack raises is what you can hold; the wall it does not touch is arithmetic — just a smaller wall than it was in August.

Why 375 days (now 142)? The card was first timed on the one thing it does best (GEMM) to find its real speed — not the number on the box. Then a pretraining step was broken into parts:

| Part of One Training Step | Share | How Close to Limit? |
|---------------------------|-------|---------------------|
| Multiplying matrices (the actual maths) | 33% | **100.6% — at the ceiling** |
| Small operations between the big ones | 36% | Limited by memory speed |
| Stepping through the sequence | 15% | Waiting, not calculating |
| Moving data around | 15% | The price of fitting in 6 GB |

The maths runs at 100.6% of the fastest this GPU was seen to go — that part held up. "Nothing left, 20-25% at most" did not: four more real, measured levers (fusing more DeltaNet glue into the scan kernels, fixing a short causal-conv transpose, two ternary-quantize caches) landed in one week, cutting the step a further 7% on top of what was already captured, for 28% total against the August baseline at that point. Since then a fifth lever (four more bit-identical kernel fusions) landed on 2026-10-17, taking the total to **39%** against the August baseline. This project has now been wrong about "exhausted" more than once — treat any future claim like it the same way: as a thing to re-measure, not trust.

**Why 30 Tokens/Param, Not 20?**

The Chinchilla paper (Hoffmann et al., 2022) recommended about 20 tokens/parameter for transformers. This architecture behaves differently. Measured across three model sizes at constant learning rate (every checkpoint is a model trained to that many tokens, not a snapshot part-way through a decaying schedule):

| Tokens/Param | Held-out Loss Improvement | Compute Cost |
|--------------|---------------------------|--------------|
| 20 to 30 | **1.9-3.3%** better | +50% tokens |
| 30 to 40 | about 1.1% better | +33% tokens |

30 is where the trade turns. Going from 20 to 30 buys significant quality for 50% more compute; going 30 to 40 buys marginal gains for 33% more compute. Why this architecture needs more data: the recurrent state compresses history differently than attention, exact-token recall (the hybrid's attention window) needs more exposure to learn the pattern, and the gated DeltaNet update has more learnable dynamics than a static attention matrix. How this moves with model size is unresolved — fitting the standard scaling form to all measured points gives a degenerate answer (it implies a 1B model needs less data than it has parameters, contradicting every raw curve), and three model sizes are not enough to separate the terms.

**Practical rule:** budget for 30 tokens per body-param minimum with this stack. A 1.11B model (about 1.01B body params) needs about 30B tokens — **about 132-211 days at 2,656 tok/s** (was 375 days at 1,579 tok/s in August), depending on how you count block-coordinate descent's own quality cost. At 20 tokens/param you'd stop earlier and leave quality on the table.

**What a 6 GB Card Is Good For**

| Goal | Size Needed | Training Time |
|------|-------------|---------------|
| Emitting correctly formatted tool calls | about 100-270M | Days |
| Holding a coherent conversation | about 270M | about 19-20 days |
| Writing code well enough to trust | Far more | Out of reach |

The first two are real projects here. The third is not — code models many times larger, trained on trillions of tokens, are still weak at it.

**Pretraining Config Used (1.11B Run)**

This is the recommended config: the hybrid architecture (10 DeltaNet layers + 2 sliding-attention layers at 6 and 18), tied embeddings, and the fully composable memory stack (offload + BCD + chunked loss), all confirmed compatible together per Issue #157. Shown here as the config to use going forward, not a literal transcript of any one past run.

Command: python train_edge_conversational.py --arch hvn-hybrid --d-model 2048 --num-layers 24 --n-heads 64 --attn-layers 6,18 --attn-window 4096 --batch-size 1 --seq-len 4096 --grad-accum 8 --chunked-loss 512 --offload --block-coordinate --bcd-partition balanced --bcd-steps-per-block 100 --optimizer muon --muon-lr 0.02 --lr 3e-4 --dropout 0 --tie-embeddings --pretrain-mix 0.0,0.0,0.0,1.0 --min-tokens-per-param 0

Key flags explained: **--arch hvn-hybrid** selects the gated DeltaNet trunk with 2 sliding-attention layers. **--attn-layers 6,18** places those attention layers for exact-token recall. **--tie-embeddings** shares the vocab table between embedding and decoder (saves 103M params; compatible with offload since Issue #157 pinned the shared table). **--offload** does CPU weight offload for inactive blocks (auto-enables **--block-coordinate** and **--grad-checkpoint**). **--block-coordinate** is BAdam: train one block at a time, clear its optimizer state. **--chunked-loss 512** scores loss 512 positions at a time, recomputing in backward (saves about 1.5 GB) — this is what makes tied embeddings plus offload fit without OOM. **--precision bf16** (on by default via **--fast**) gives 2-3x speed and 2x less VRAM, needs RTX 30/40 series. **--optimizer muon** needs 1.6x fewer steps to the same loss (about 12% slower per step, still a net win).

**If you run this command yourself, the first about 100 steps will print about 17.6 s/step with `--fp8-linear` (about 19 s/step without it), not the cycle average.** That is correct, not broken — BCD trains one block at a time, and a fresh launch always starts on the slowest block (token_embedding). It steps down at every block switch after that. See the before/after table above for the per-block range.

**Traps That Cost Real Hours**

1) **torch.compile is 2.4x SLOWER** on this architecture (4 independent measurements). Use **--fast**, which deliberately skips it.
2) Bare **--no-fast** silently drops you to fp32 — 74.6 s/step instead of 8.66. Re-add the precision flags by hand.
3) The default data mix caps a pretrain at 0.13 billion tokens. OpenAssistant is only 15.7M tokens but takes a 35% share, which binds the whole budget — enough for about a 7M model. A 1B run on defaults gets 270x less data than it needs, and the loss curve looks perfectly healthy while it re-reads the same tokens. Pass **--pretrain-mix 0.0,0.0,0.0,1.0** to use FineWeb-Edu instead.
4) Nothing prints until step 50. A short run looks hung when it's fine. Watch training_log.csv, which flushes every row.
5) tokenizer.vocab_size excludes added tokens. Use len(tokenizer) for embedding table sizing — otherwise sentinels index out of bounds (bare CUDA device-side assert).

**What Failed (The Dead Ends, Recorded on Purpose)**

11 ideas tested and rejected, measured not guessed: 1) original HVN architecture, lost by 13% at equal data vs gated DeltaNet (retracted). 2) torch.compile, 2.4x slower (4 independent measurements). 3) Mixture-of-Experts, loses to simply scaling the dense model. 4) Tensor-train factorisation, half params, 22% worse. 5) GHOST (weight extrapolation), loses to doing nothing. 6) Looped/weight-shared models, 3 layers x 4 loses to 12 layers by 11x noise floor. 7) Chunk-parallel scan, 4x slower (serial triangular solves). 8) Sequence-length curriculum, 2.9% slower (occupancy against 20 SMs). 9) Per-token entanglement rate, no gain at 83M. 10) Entropy-gated early exit, does not work (retracted). 11) Adafactor without BCD, spills to host RAM, 10x slower.

5 retractions — conclusions this project withdrew: 1) original HVN "novel architecture" claim, lost to published DeltaNet by 13%. 2) #117 cache serving untrained weights, whole epic graded on aggregate loss that still trended down. 3) #122 5-item recall gate, gating a scale-up decision on a measurement dominated by luck. 4) #132 verdict on train loss, a multi-pass regime that measures memorisation, not quality. 5) #138 CUDA graph "6.8x speedup", compared one replay against one eager step from the same state — both do the same thing, passed at max|dlogits|=0 while the context had silently stopped growing.

4 optimisations measured slower than doing nothing: 1) torch.compile, 2.4x slower (20.72 vs 8.66 s/step). 2) Chunk-parallel scan (Design A), 2-4x slower at d=64. 3) Sequence-length curriculum, 2.9% slower at 83M. 4) Attention layer removal, 6.4% slower (attention is the cheap part during training).

8 silent bugs caught by falsifiable tests: 1) #117, cache served untrained weights, only test watched aggregate loss which trended down anyway. 2) #122, 5-item recall gate passed by luck, not signal. 3) #132, graded on train loss in a multi-pass regime, which measures memorisation. 4) #138, CUDA graph compared one step against one step from the same state, both identical, passed at zero diff. 5) #155, tokenizer.vocab_size excludes added tokens, embedding sized 3 rows short, out-of-bounds CUDA assert. 6) #157, tokenized .bin cache keyed on the string "code", not the corpus, silently reused the old cache. 7) #112, pre-training audit deepcopied the model to GPU, OOM'd the 6 GB card at large sizes. 8) #159, cosine LR scheduler compounded decay on every resume, a sawtooth LR.

**Limits (Read Before Scaling)**

The architecture is a reimplementation of published work — gated DeltaNet, GLA/Mamba-2 gating, Mamba/RWKV-7 convolution and output gate. The code is original; the design is not. Constant-memory inference is a property of that entire model family, not of this work. Nothing above 48M has been trained to convergence here — every scaling statement is measured below that size and should not be extrapolated past it. The 100,000-token inference memory figure quoted in the repository is calculated, not measured, and is therefore not quoted on this page. Comparisons are single-seed: that removes luck between arms, not luck shared by both. Training the same model three times, changing only the starting random numbers, moved the result by 0.089. Some differences reported in this project are smaller than that and cannot be separated from luck without repeating each across several seeds.

**Resources**

Code: https://gitlab.com/komalbarun/kramba-ai (Apache-2.0). Documentation: https://gitlab.com/komalbarun/kramba-ai/-/tree/main/docs. Issues / Discussion: https://gitlab.com/komalbarun/kramba-ai/-/issues.

**Updates**

**2026-09-26:** the previously OOM combination (tied embeddings + CPU offload + BCD) now fits without OOM when combined with chunked cross-entropy (**--chunked-loss 512**). A 5-step fp32 test at 1.11B parameters ran at 79% of 6 GB = 4.85 GB without OOM. The pinned tied table (about 206 MB bf16) plus chunked loss (about 1.5 GB saved) makes the four-technique stack (BCD + offload + tied + chunked) fit together in memory. This was the missing piece — the blog's 1.11B run used untied embeddings (3.48/4.51 GB); the tied + offload combo was previously OOM (>6 GB) without chunked loss. (This was a memory-fit test only, 5 steps, not a graded training run.)

**2026-10-11:** a separate, unrelated bug in the CPU offload path (present since 2026-07-26, fixed in 2f94d7a) could swap weights between same-shaped matrices during training whenever **--offload** was active with prefetch on. It did not affect any VRAM/fit numbers in this post, only the correctness of the weights during offloaded training. Every "trains"/"works"/"loss drops" claim above that came from an **--offload** run (the 1,350-step 1.11B headline run included) needs re-running on fixed code before it can be trusted again. The fix itself has been verified correct — confirmed at small scale (8-layer test), with the quality finding below (BCD balanced partition) confirmed to hold under offload too.

**2026-10-21:** every s/step, tok/s, VRAM and day-count figure in this post has been re-measured against the current code and updated — 14.385 to 13.294 s/step, 2,278 to 2,464 tok/s, 5.31 GB to 5.24 GB peak, and the day estimate from 154-246 to 142-227. The **--offload** plus tied-embeddings compatibility question raised earlier (#151) is confirmed fixed and current (#157) — the config above uses tied embeddings together with offload without issue.

**License**

Apache 2.0 — free to use, modify, and sell, with attribution and a patent grant. The license covers the code, not trained weights. Checkpoint files are a separate work, all rights reserved. Weights you train are yours. Products you build are yours.
