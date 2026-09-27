# Pre-training a 1.11B LLM on a 6 GB Laptop GPU — Measured, Not Claimed

[DOI 10.5281/zenodo.22725431](https://zenodo.org/records/22725431)

I looked everywhere to be able to **pre-train** an LLM on my low-end GPU and could not find one. Everywhere I looked, the same conclusion was reached: "You need an A100", "You can only train a 20M toy model on your hardware". So being 'broke', I did the only thing I could do — subscribe to Claude and try to find a way... after much prompting and testing, I finally stitched together the stack that allowed me to **pre-train** a 1B LLM on my laptop. After much testing, we are at **100.6% of the card's *measured* GEMM peak** (I don't know how 100.6% is even possible, I'm not a scientist, just a broke guy).

Though I can **pre-train** a 1B LLM on 6 GB VRAM, it still takes about **375 days** to finish **pretraining** it properly. More details below:

---

## The Headline Numbers (Measured on RTX 4050 6 GB)

| Metric | Value |
|--------|-------|
| **Model size** | 1.11B parameters (pretraining, not inference) |
| **Peak VRAM** | 4.51 GB (1.5 GB to spare) |
| **Throughput** | 1,579 tokens/sec at 4,096 context |
| **Pretraining time (Chinchilla-optimal)** | ~375 days for 36B tokens |

---

## The 5-Technique Stack That Made Pretraining Possible

No new algorithms — every piece is published and attributed. The work was making them run together and measuring what that costs.

| Technique | What it removes | Saving |
|-----------|-----------------|--------|
| **Block-coordinate descent (BAdam)** | Optimizer state for every block but the active one | ~8× on optimizer state |
| **CPU weight offload** | Inactive blocks leave the GPU entirely | Scales with depth |
| **Ternary weights (BitNet b1.58)** | fp32 weights → {-1, 0, +1} (1.58 bits), per-matrix fp32 scale | 2.33× weight storage |
| **Tied embeddings** | One vocabulary table instead of two | 103M params at 1B scale |
| **Gradient checkpointing** | Stored activations — recomputed instead | Activations to ~50 MB |
| **Chunked cross-entropy** | Logits materialized slice-by-slice, recomputed in backward | ~1.5 GB saved at 4,096 × 50K vocab |

**Ternary quantization details (BitNet b1.58):**

- **What 2.33× means:** Weight tensors only (not total VRAM). fp32 (32 bits) → ternary stored as int8 (8 bits) + per-matrix fp32 scale. Theoretical 20× (log₂3 ≈ 1.58 bits), practical 2.33× after scale overhead.
- **35% sparsity:** ~1/3 of weights round to exactly 0 — zeros are spread across rows, not whole-row collapse.
- **QAT only:** Must train from scratch with straight-through estimator (STE). Post-hoc quantization loses the learned dynamics. `--ternary` is ON by default in production runs.
- **No training VRAM win:** fp32 shadow weights stay resident for gradient updates. Ternary saves checkpoint size and inference memory only.
- **Ternary collapse at init:** Fresh layers can zero out before learning (Issue #109). Fixed by warm-starting trunk from pre-trained checkpoint.
- **Group-wise scales tested (Issue #146):** Finer per-group scales lose to per-tensor — storage ratio improves with width but quality doesn't.

**Two of these did not previously work together.** Tied embeddings share one tensor between the input lookup and the output layer, and the offload system rejected that outright — it assigns every parameter to exactly one block, and this tensor belongs to two. The combination was guarded off as unsupported.

**The fix (Issue #157):** Pin the tied table — a parameter reachable from two modules is never evicted, because the forward pass uses it at both ends and no single block's residency window covers both uses. Cost: one vocabulary table held resident (~250 MB at 1B scale). **When combined with chunked cross-entropy (`--chunked-loss 512`), the previously OOM combination (tied embeddings + CPU offload) now works** — a 5-step fp32 test at 1.11B parameters ran at **79% of 6 GB = 4.85 GB** without OOM. The pinned tied table (~206 MB bf16) plus chunked loss (~1.5 GB saved) makes the four-technique stack (BCD + offload + tied + chunked) fully composable.

---

## Architecture

The model is a **gated DeltaNet hybrid** (10 DeltaNet blocks + 2 sliding-window attention layers at layers 3 and 10). This architecture is a reimplementation of published work:

- Canonical gated DeltaNet update (GLA / Mamba-2 style decay gate)
- Per-head output RMSNorm normalizer (proper weighted average readout)
- Output gate + short causal depthwise convolution (Mamba / RWKV-7 style)

Constant-memory inference — no KV cache, fixed state — is the defining property of this entire model family (Mamba, RWKV, RetNet), not something this project invented.

---

## Why Hybrid Attention? Inference-Time Exact Recall

Pure recurrence (DeltaNet, Mamba, RWKV) compresses all history into a fixed state — great for throughput, **but cannot recall exact tokens** from context. The hybrid adds **2 sliding-window attention layers** (at layers 3 and 10, window=128) specifically to solve this.

**Measured on 200-trial synthetic recall probe (invented entities, zero knowledge confound):**

| Model | Near-window recall (≤128 tokens) | Far-window recall (>128 tokens) |
|-------|----------------------------------|---------------------------------|
| Pure DeltaNet trunk (10 layers) | 4–5% | 0.5% (noise) |
| **Hybrid (+2 attention layers, step 8K)** | **35.5%** (**9× trunk**) | **8.5%** (17× trunk) |

**Key insight:** The attention window (128 tokens) gives **exact token recall inside the window** — something the recurrent state mathematically cannot provide. Beyond the window, both recurrence and windowed attention fall to noise floor (~0.5%).

This means at inference: the hybrid can **read and quote exact passages** from the last ~128 tokens of context, while pure recurrent models can only give a lossy semantic gist. For coding agents, tool-use, and grounded QA, this is the difference between "it hallucinates the function signature" and "it copies the exact signature from context."

The attention layers add only **~13% VRAM** (2 TransformerBlocks with windowed KV cache capped at 128 keys each) and **~7% step time** at pretraining — a small price for exact recall where it matters most.

---

## What You Get Without Any of This (Same Card, Standard Script — Pretraining)

| Model | Standard Script | With This Stack |
|-------|-----------------|-----------------|
| 157M | 3.38 GB — fits | fits |
| 243M | 4.80 GB — fits | fits |
| 434M | 9.04 GB — spills | fits |
| **1.07B (recurrent)** | far past the card | **4.00 GB** |
| **1.07B (plain transformer)** | far past the card | **4.04 GB** |
| **1.58B** | far past the card | **5.92 GB — the ceiling** |

**~243M on a standard script, ~1.58B with the stack — about 6.5×.** Both numbers measured on the same 6 GB card at 4,096 context, stopping at the first spill into system RAM (on Windows/WDDM, exceeding VRAM does not error — it silently gets 4–15× slower).

What the table does not show: a transformer and the recurrent hybrid need almost identical memory at these sizes (4.80 GB vs 5.09 GB at 243M), and a plain 1.07B transformer runs under the same stack in 4.04 GB. The 6.5× comes from the **pretraining stack, not from the architecture** — so it applies to whatever model you are pretraining, including an ordinary transformer.

---

## What This Means for Your GPU (Projected VRAM Ceilings — **EXTRAPOLATED, NOT MEASURED**)

> **⚠️ Only the 6 GB row is measured.** All other rows are linear extrapolation from the 6 GB ceiling (~1.58B) assuming fixed-ratio savings. Real VRAM ceilings will vary by GPU architecture, WDDM behavior, and batch size. Treat as upper-bound estimates.

The stack scales with VRAM. On a **6 GB card** we measured ~1.58B ceiling. Here's what the same stack projects for larger cards (linear extrapolation from measured points, stopping at first WDDM spill):

| GPU VRAM | Max Model (Recurrent Hybrid) | Max Model (Plain Transformer) | Pretrain Time (Chinchilla) |
|----------|------------------------------|-------------------------------|----------------------------|
| **6 GB** (RTX 4050 laptop) | **~1.58B** | **~1.58B** | ~375 days @ 1.1B |
| **8 GB** (RTX 3070/4060 laptop) | **~2.2B** | **~2.2B** | ~160 days @ 1.5B |
| **12 GB** (RTX 3060/4070) | **~3.5B** | **~3.5B** | ~75 days @ 2.5B |
| **16 GB** (RTX 4080) | **~4.8B** | **~4.8B** | ~40 days @ 3.5B |
| **24 GB** (RTX 3090/4090) | **~7.5B** | **~7.5B** | ~18 days @ 5B |

**Key insight:** The 6.5× multiplier (standard script → this stack) applies **at every VRAM tier** because the savings are proportional:

- BCD: saves ~8× optimizer state (scales with params)
- Offload: inactive blocks to CPU (scales with depth)
- Ternary: 2.33× weight compression (fixed ratio)
- Tied embeddings: saves 103M params at 1B (fixed absolute)
- Chunked loss: saves ~1.5 GB at 4K context (fixed absolute)

**The ceiling is linear in VRAM** because the dominant terms (weights + optimizer state) scale with model size, and our techniques compress them by fixed ratios. A 24 GB card doesn't just fit "a bit more" — it fits **~5× the parameters** of a 6 GB card.

**What changes at larger VRAM:**
- You can drop `--offload` (no PCIe bottleneck) → ~20% speedup
- You can increase `--batch-size` → better GPU utilization
- You hit the **compute wall** sooner (matmuls at 100% ceiling)
- At 24 GB, a 7B pretrain is ~18 days — actually finishable

---

## System RAM Requirements (Often Overlooked)

| GPU VRAM | Min System RAM | Recommended | Why |
|----------|----------------|-------------|-----|
| **6 GB** | 16 GB | 32 GB | Offloaded weights + dataset + OS |
| **8 GB** | 16 GB | 32 GB | Offloaded weights grow with model depth |
| **12 GB** | 24 GB | 48 GB | Less offload needed, but dataset larger |
| **16 GB** | 32 GB | 64 GB | Can drop offload, but bigger datasets |
| **24 GB** | 48 GB | 96 GB | Full model in VRAM, but dataset + OS |

**Two reasons system RAM matters:**

1. **CPU offload** (`--offload`): Inactive blocks' weights are evicted to **system RAM**. At 1.11B with offload, ~3–4 GB of weights sit in system RAM. If you have 16 GB system RAM and 6 GB VRAM, you're fine. If you have 8 GB system RAM, you'll OOM on the host.

2. **WDDM silent spill (Windows)**: On Windows, exceeding VRAM does **not** raise OOM — the driver silently spills to **system RAM over PCIe** at **4–15× slower**. We measured 82 s/micro-batch when a 2.45B model spilled. If your system RAM is full, the spill has nowhere to go → hard crash.

**Rule of thumb:** System RAM ≥ 2× GPU VRAM for offload runs; ≥ 1.5× for no-offload runs. The 16 GB in my laptop was the minimum for 6 GB VRAM + offload.

---

## Where the Memory Actually Goes (Measured at 48M, 4,096 Context)

| Item | Size |
|------|------|
| **Computing the loss** — 4,096 positions × 50,257 vocabulary | ~1,500 MB |
| CUDA context, kernels, workspaces | ~400 MB |
| Weights (ternary) | ~960 MB |
| Optimiser state (BCD: one block at a time) | ~200 MB |
| Layer activations (grad checkpoint) | ~50 MB |
| **Pinned tied table (if tied)** | **~250 MB** |

The largest consumer is **not the model** — it's the cross-entropy over the vocabulary (~3× the weights). **Chunked loss (512 positions at a time) recomputes logits in backward instead of keeping them, saving ~1.5 GB.** This is what made the previously OOM combination (tied + offload) work.

---

## The Trade-off: Tied Embeddings vs Offload

| Config | Params | Tied | VRAM | Status |
|--------|--------|------|------|--------|
| Blog claim (no BCD offload) | 1.07B | ✓ | 4.00 GB | Target |
| 1.11B run (untied) | 1.11B | ✗ | 3.48/4.51 GB | 1,350 steps, works |
| **Tied + offload + chunked_loss** | **1.11B** | **✓** | **4.85 GB (79%)** | **5 steps fp32, works (fixed)** |
| Tied + offload (no chunked) | 1.07B | ✓ | >6 GB | OOM (pinned 250 MB + overhead) |
| 57M test | 57M | ✓ | Works | All 5 techniques integrate |

---

## The Hard Reality: Fitting ≠ Training

A 1B model runs at ~1,100 tokens/sec here, and pretraining one properly needs roughly **36 billion tokens** — about **375 days**. The ceiling this stack raises is what you can *hold*; the wall it does not touch is arithmetic.

### Why 375 Days?

The card was first timed on the one thing it does best (GEMM) to find its real speed — not the number on the box. Then a pretraining step was broken into parts:

| Part of One Training Step | Share | How Close to Limit? |
|---------------------------|-------|---------------------|
| Multiplying matrices (the actual maths) | 33% | **100.6% — at the ceiling** |
| Small operations between the big ones | 36% | Limited by memory speed |
| Stepping through the sequence | 15% | Waiting, not calculating |
| Moving data around | 15% | The price of fitting in 6 GB |

The maths runs at **100.6%** of the fastest this GPU was seen to go. Nothing left there. Every remaining improvement added up optimistically comes to **20–25%**, turning 375 days into about 300. The year is not bad code — it is 36 billion tokens.

---

## Why 30 Tokens/Param, Not 20?

The Chinchilla paper (Hoffmann et al., 2022) recommended ~20 tokens/parameter for transformers. But this architecture behaves differently:

**Measured across three model sizes at constant learning rate** (every checkpoint is a model trained to that many tokens, not a snapshot part-way through a decaying schedule):

| Tokens/Param | Held-out Loss Improvement | Compute Cost |
|--------------|---------------------------|--------------|
| 20 → 30 | **1.9–3.3%** better | +50% tokens |
| 30 → 40 | ~1.1% better | +33% tokens |

**30 is where the trade turns.** Going from 20→30 buys significant quality for 50% more compute. Going 30→40 buys marginal gains for 33% more compute.

Why this architecture needs more data:
- The recurrent state compresses history differently than attention
- Exact-token recall (the hybrid's attention window) needs more exposure to learn the pattern
- The gated DeltaNet update has more learnable dynamics than a static attention matrix

**How this moves with model size is unresolved.** Fitting the standard scaling form to all measured points gives a degenerate answer — it implies a 1B model needs less data than it has parameters, contradicting every raw curve. Three model sizes are not enough to separate the terms.

> **Practical rule:** Budget for **30 tokens/param minimum** with this stack. A 1.11B model needs ~33B tokens (375 days at 1,579 tok/s). At 20 tokens/param you'd stop at ~22B tokens and leave ~2-3% quality on the table.

---

## What a 6 GB Card Is Good For

| Goal | Size Needed | Training Time |
|------|-------------|---------------|
| Emitting correctly formatted tool calls | ~100–270M | Days |
| Holding a coherent conversation | ~270M | ~24 days |
| Writing code well enough to trust | Far more | Out of reach |

The first two are real projects here. The third is not — code models many times larger, trained on trillions of tokens, are still weak at it.

---

## Pretraining Config Used (1.11B Run)

```bash
python train_edge_conversational.py \
  --arch hvn \
  --d-model 2048 \
  --num-layers 24 \
  --n-heads 64 \
  --seq-len 4096 \
  --batch-size 1 \
  --grad-accum 8 \
  --precision bf16 \
  --optimizer muon \
  --muon-lr 0.02 \
  --lr 3e-4 \
  --tie-embeddings \
  --offload \
  --block-coordinate \
  --bcd-steps-per-block 100 \
  --chunked-loss 512 \
  --pretrain-mix 0.0,0.0,0.0,1.0 \
  --min-tokens-per-param 0
```

**Key flags explained:**
- `--arch hvn` — gated DeltaNet blocks (the default)
- `--tie-embeddings` — share vocab table between embedding + decoder (saves 103M params)
- `--offload` — CPU weight offload for inactive blocks (requires `--grad-checkpoint`)
- `--block-coordinate` — BAdam: train one block at a time, clear its optimizer state
- `--chunked-loss 512` — score loss 512 positions at a time, recompute in backward (saves ~1.5 GB)
- `--precision bf16` — mixed precision (2-3× speed, 2× less VRAM, needs RTX 30/40 series)
- `--optimizer muon` — 1.6× fewer steps to same loss (but ~12% slower/step)

---

## Traps That Cost Real Hours

1. **`torch.compile` is 2.4× SLOWER** on this architecture (4 independent measurements). Use `--fast`, which deliberately skips it.
2. **Bare `--no-fast` silently drops you to fp32** — 74.6 s/step instead of 8.66. Re-add the precision flags by hand.
3. **The default data mix caps a pretrain at 0.13 billion tokens.** OpenAssistant is only 15.7M tokens but takes a 35% share, which binds the whole budget — enough for a ~7M model. A 1B run on defaults gets **270× less data than it needs**, and the loss curve looks perfectly healthy while it re-reads the same tokens. Pass `--pretrain-mix 0.0,0.0,0.0,1.0` to use FineWeb-Edu instead.
4. **Nothing prints until step 50.** A short run looks hung when it's fine. Watch `training_log.csv`, which flushes every row.
5. **`tokenizer.vocab_size` EXCLUDES added tokens.** Use `len(tokenizer)` for embedding table sizing — otherwise sentinels index out of bounds (bare CUDA device-side assert).

---

## What Failed (The Dead Ends, Recorded on Purpose)

### 11 Ideas Tested and Rejected (Measured, Not Guessed)

1. **Original HVN architecture** — lost by 13% at equal data vs gated DeltaNet (retracted)
2. **`torch.compile`** — 2.4× slower (4 independent measurements)
3. **Mixture-of-Experts** — loses to simply scaling the dense model
4. **Tensor-train factorisation** — half params, 22% worse
5. **GHOST (weight extrapolation)** — loses to doing nothing
6. **Looped/weight-shared models** — 3 layers × 4 loses to 12 layers by 11× noise floor
7. **Chunk-parallel scan** — 4× slower (serial triangular solves)
8. **Sequence-length curriculum** — 2.9% slower (occupancy against 20 SMs)
9. **Per-token entanglement rate** — no gain at 83M
10. **Entropy-gated early exit** — does not work (retracted)
11. **Adafactor without BCD** — spills to host RAM, 10× slower

### 5 Retractions (Conclusions This Project Withdrew)

1. **Original HVN "novel architecture" claim** — lost to published DeltaNet by 13%
2. **#117 cache serving untrained weights** — whole epic on aggregate loss that still trended down
3. **#122 5-item recall gate** — gating scale-up on luck-dominated measurement
4. **#132 verdict on train loss** — multi-pass regime measures memorisation, not quality
5. **#138 CUDA graph "6.8× speedup"** — compared one replay against one eager step from same state; both do same thing, passed at max|dlogits|=0 while context silently stopped growing

### 4 Optimisations Measured Slower Than Doing Nothing

1. **`torch.compile`** — 2.4× slower (20.72 vs 8.66 s/step)
2. **Chunk-parallel scan (Design A)** — 2–4× slower at d=64
3. **Sequence-length curriculum** — 2.9% slower at 83M
4. **Attention layer removal** — 6.4% slower (attention is the cheap part during training)

### 8 Silent Bugs Caught by Falsifiable Tests

1. **#117** — cache served untrained weights; only test watched aggregate loss (trended down)
2. **#122** — 5-item recall gate passed by luck, not signal
3. **#132** — graded on train loss in multi-pass regime (measures memorisation)
4. **#138** — CUDA graph compared 1 step against 1 step from same state; both identical, passed at 0 diff
5. **#155** — `tokenizer.vocab_size` EXCLUDES added tokens; embedding sized 3 rows short, OOB CUDA assert
6. **#157** — tokenized .bin cache keyed on string "code", not corpus; silently reused old cache
7. **#112** — pre-training audit deepcopied model to GPU, OOM'd 6 GB card at large sizes
8. **#159** — cosine LR scheduler compounded decay on every resume (sawtooth LR)

---

## Limits (Read Before Scaling)

- The architecture is a reimplementation of published work — gated DeltaNet, GLA/Mamba-2 gating, Mamba/RWKV-7 convolution and output gate. The code is original; the design is not.
- Constant-memory inference is a property of that entire model family, not of this work.
- **Nothing above 48M has been trained to convergence here.** Every scaling statement is measured below that size and should not be extrapolated past it.
- The 100,000-token inference memory figure quoted in the repository is **calculated, not measured**, and is therefore not quoted on this page.
- Comparisons are single-seed: that removes luck between arms, not luck shared by both. Training the same model three times, changing only the starting random numbers, moved the result by **0.089**. Some differences reported in this project are smaller than that and cannot be separated from luck without repeating each across several seeds.

---

## Resources

- **Code**: https://gitlab.com/komalbarun/kramba-ai (Apache-2.0)
- **Documentation**: https://gitlab.com/komalbarun/kramba-ai/-/tree/main/docs
- **Issues / Discussion**: https://gitlab.com/komalbarun/kramba-ai/-/issues

---

## Updates

**2026-09-26:** The previously OOM combination (tied embeddings + CPU offload + BCD) now works when combined with **chunked cross-entropy (`--chunked-loss 512`)**. A 5-step fp32 test at 1.11B parameters ran at **79% of 6 GB = 4.85 GB** without OOM. The pinned tied table (~206 MB bf16) plus chunked loss (~1.5 GB saved) makes the four-technique stack (BCD + offload + tied + chunked) fully composable. This was the missing piece — the blog's 1.11B run used untied embeddings (3.48/4.51 GB); the tied + offload combo was previously OOM (>6 GB) without chunked loss.

---

## License

Apache 2.0 — free to use, modify, and sell, with attribution and a patent grant. The license covers the code, not trained weights. Checkpoint files are a separate work, all rights reserved. **Weights you train are yours.** Products you build are yours.

---

