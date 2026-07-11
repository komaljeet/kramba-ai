# Development Roadmap

> **Moved from README:** §9 Development Roadmap. The Issues #32 & #33 execution plan
> (former §10) lives in [execution_plan.md](execution_plan.md); measured numbers referenced
> below are in the [benchmarks_*.md](index.md) files.
>
> **Status legend:** ✅ Done · 🔄 In Progress · ⬜ Planned

## Phase 1–3: Core Architecture, Stability, Performance — ✅ Complete

Core engine (`VortexInjectionLayer`, `HamiltonianMemoryBlock`, `HVNBlock`, `HVNDecoder`,
full model), theory + FLOP docs; velocity L2-norm + Frobenius clamp, power-iteration
low-rank truncation, gradient clipping, cosine LR + AdamW, char tokenizer, train/generate
scripts; multi-head Hamiltonian (PPL 9.48→8.37), gated vortex injection, chunked parallel
(35% faster fwd), incremental generation (O(1), 41% faster), speed/quality benchmarks.

## Phase 4: Quality Improvements — ✅ (all sub-items closed)

RoPE (#1), rank-k projector (#12), per-head vorticity projections (#16), residual H
connection (#13), input-dependent η (#14), RMSNorm + parallel residual (#15).

## Phase 5: Speed & Kernel Optimizations — 🔄 (4/6 done, Issue #33 closed)

Fused Triton kernel (Fix 1 ✅), torch.compile (#10 ✅), associative parallel scan (Fix 3 ✅),
CUDA Graph capture (Fix 6 ✅), multi-chunk barrier sync (Fix 4 ✅). Larger chunk size (#4/#7
✅). **Deferred:** Fix 2 flash-style tiled Hamiltonian (needed when d_model>1024), Fix 5
fully parallel with learned positional injection (research moonshot, loses causal structure).

## Phase 6: Evaluation & Scaling — ✅ Complete

Real-dataset benchmarks (#2), long-context N=256→4096 + needle-in-haystack (#2), scaling
laws (#18). **Original finding (now overturned):** HVN PPL ∝ params^-0.007 (nearly flat). **#35
re-measurement on real WikiText-103 (BPE, compute-matched, unsaturated):** HVN scales at **−0.42**
≈ Transformer −0.41 — the flat −0.007 was a 34K-char artifact; HVN benefits from scale normally
(see [benchmarks_scaling.md](benchmarks_scaling.md)).

## Phase 7: Inference Optimization & Production — ✅ / ⬜ mixed

Quantization INT8/NF4 (#20 ✅), BPE tokenizer (#11 ✅), HuggingFace integration (#3/#6 ✅),
mixed precision (#19 ✅). ✅ **Distributed training (DDP/FSDP + tensor-parallel) — epic #64
delivered (2026-07-06).** Every piece shipped with a laptop-runnable **parity artifact**
(the #46 `batched == single-stream` discipline extended to distribution):

- **#65 Data parallelism** — `train.py`/`auto_train.py --distributed [--fsdp]` under
  `torchrun` (`hvn/distributed.py`): 2-process gloo run == single-process weights
  (max|Δw| 1.2e-7, grad-accum `no_sync` included), per-epoch loss all-reduce (extends #50),
  rank-0-only atomic writes, single-GPU checkpoint layout unchanged (`tests/test_issue65.py` 5/5).
- **#66 Tensor parallelism** — head-parallel DeltaNet (`hvn/core/head_parallel.py`,
  Megatron-style f/g pair, one all-gather per block, 1/W per-rank state) + full design
  ([TENSOR_PARALLEL_DESIGN.md](TENSOR_PARALLEL_DESIGN.md), incl. within-head sharding and why
  the default block's `HPH` term is TP-hostile while DeltaNet needs zero recurrence collectives) —
  `tests/test_issue66.py` 5/5 forward+gradient parity.
- **#67 Multi-node launch + checkpoint I/O** — `hvn/launch.py` (torchrun/SLURM bring-up,
  FSDP full-state-dict consolidation → rank-0 atomic single-file layout) +
  [DISTRIBUTED_LAUNCH.md](DISTRIBUTED_LAUNCH.md); a 2-proc-DDP checkpoint serves **unchanged**
  through `generate.load_model` → `ModelVersion` (`tests/test_issue67.py`, param parity 2.4e-7).
- **#68 Frontier plan** — [FRONTIER_TRAINING_PLAN.md](FRONTIER_TRAINING_PLAN.md): config from
  the measured −0.504 DeltaNet exponent, Chinchilla-style budgets (~$8 pilot / ~$210 @1B /
  ~$21K @10B, MFU labelled), data plan extending #53, the edge/ternary lane fork costed, risk
  register, go/no-go Gates 0–3. **This is the frontier ramp.**

O(1) inference and #46 batched-equality preserved throughout; the cluster run itself remains
gated on budget/environment (Gate 1+ of the plan).

**Epic #64** (frontier-scale training readiness) tracked the above sub-issues: data-parallel
DDP/FSDP path (#65), tensor/model-parallel HVN block (#66), multi-node launch + sharded-checkpoint
I/O (#67), and the frontier training plan/spec (#68). Two companion issues quantify the on-box
ceiling — **RTX 4050 (6 GB)** and **RTX 4060 (8 GB)** training envelopes (#69/#70, see
[training.md](training.md)) — and record the hard reality that a **10B model cannot be trained (or
bf16-inferenced) on these cards** (10B ≈ 160 GB of training state, 20 GB of bf16 weights; off by
~30×). A 10B model *served* on 6 GB is only reachable via the **edge/ternary (BitNet b1.58) path**
(#32) at ~2 GB — but that is a **separate, quantization-aware training run**, not a free export of a
full-precision flagship (see #68's lane table).

## Phase 8: Fine-Tuning & Instruction Tuning — ✅ Complete

LoRA, QLoRA, instruction tuning, LoRA save/load, demo (#29).

## Phase 9: PPL Root-Cause Correction (Issues #34/#35/#36, 2026-07-04)

> **Diagnosis update:** the "fixed d×d matrix is a fundamental compression bottleneck"
> explanation is contradicted by our own benchmark — the SSM reaches PPL ~1.1 with a state
> *smaller* than HVN's d×d-per-head Hamiltonian. Fixed-size O(1) state is **not** the limiter.
> HVN is really a linear-attention fast-weight network; the fix is to borrow proven dynamics
> from DeltaNet / GLA / Mamba-2 / RWKV-7, not to add capacity.

- **9.1 Readout & update dynamics (#34)** — ✅ **Done** (`--use_deltanet`). New
  `DeltaNetBlock`: per-channel data-dependent decay gate + per-head output RMSNorm, canonical
  gated DeltaNet update with separate k/v projections (drops HPH), output gate + causal conv.
  A/B tiny-Shakespeare (d=128, 2L): PPL **8.81 → 3.42 = 61% better** (target ≥30%). O(1)
  preserved. Training via chunked-parallel delta rule (Yang et al. 2024): 27.5 → 3.7 s/epoch
  (~7.4×), numerically identical to sequential scan (max|diff| 4.8e-7). `test_issue34.py` 5/5.
- **9.2 Real-scale scaling validation (#35)** — ✅ **Done — the flat exponent was a small-data
  artifact.** Rewired `eval/scaling_laws.py` for a *methodologically correct* scaling study:
  **compute-matched** training (every size sees the same **token budget**, not the same epochs),
  **full held-out eval PPL** (whole valid split in `seq_len` chunks, not a single mid-text batch),
  **BPE** tokenizer support, **SSM** added to the sweep, and a per-model eval-PPL-vs-tokens
  trajectory reporting each model's **saturation token budget**. Added a streaming WikiText-103
  loader (`eval/datasets.py`, `--dataset wikitext103 --local-path <dir>`, char-capped reads to
  bound memory on the 539 MB corpus).

  **Headline run — real WikiText-103, BPE (vocab 8,000), 5M-token compute-matched budget** (12.4M
  unique train tokens, so unsaturated), d={128,256,512,768}, 2L, bf16, RTX 4050 (see
  [benchmarks_scaling.md](benchmarks_scaling.md)). **Refit exponents: HVN −0.4237, Transformer
  −0.4066, SSM −0.5311.** HVN scales *as well as* a Transformer (marginally steeper) and its PPL
  drops 521 → 206 across a 9× param increase — a ~100× steeper exponent than the original −0.007.
  **The "HVN doesn't benefit from scale" conclusion is refuted:** it was an artifact of fitting on
  34K characters (deep in the overfit/saturation regime). A **small-data control** on full
  Tiny-Shakespeare (char-level, 32× the original data) *reproduces* the flat exponent (HVN −0.0042)
  precisely because every size saturates at ~1M tokens there — confirming the artifact mechanism.
  **Caveat:** default HVN only (the #34 `--use_deltanet` fix was not swept); 5M-token reduced budget
  (absolute PPLs drop at a larger budget, but the exponent is already in the trustworthy unsaturated
  regime). Scale up via:
  ```bash
  python -m eval.scaling_laws --dataset wikitext103 --local-path wikitext-103-v1 \
      --tokenizer bpe --bpe_vocab 8000 --token_budget 100_000_000 \
      --d_models 128,256,512,768 --precision bf16 --compile \
      --output-json results/scaling_wt103.json --output-plot results/scaling_wt103.png
  ```
- **9.3 Generation-speed cleanups (#36)** — ✅ **Done**. Removed the dead full-`forward()`
  prompt prefill in `generate_incremental` (the prompt was processed twice), and moved decode
  sampling on-device: reuse one preallocated input buffer (no per-token host→device alloc),
  keep top-k + `multinomial` on-device, and sync generated ids to host once at the end (no
  per-step `.item()`). New `prefill_states` absorbs the prompt in a single pass with two exact
  paths: DeltaNet uses the algebraically-exact chunked scan (O(chunks) launches — **42.7×**
  fewer than token-by-token on a 512-token prompt); the default HVN path uses the exact
  sequential recurrence (its chunked forward is a within-chunk *approximation*, so prefilling
  through it would change output). Measured HEAD vs pre-#36 (d=128, 2L, RTX 4050, warmed
  best-of-5, 512-tok prompt + 200 tokens): HVN decode **80.7 → 197.7 tok/s (2.45×)**; DeltaNet
  time-to-first-token **1,612 → 23 ms (70×)** for a 512-token prompt (prompt-length dependent).
  Verified token-for-token identical to the previous implementation (`test_issue36.py` 5/5,
  `top_k=1` greedy parity), and **stackable** with #33 kernels (`--cuda_graph` 350 tok/s,
  `--fused_kernel` 354 tok/s). O(1) preserved; `demo.py`, `test_issue33.py`, `test_issue34.py`
  regressions pass.
