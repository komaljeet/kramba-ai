# Training & Usage Guide

> **Moved from README:** §6 Training & Usage Guide — HVN-vs-Transformer differences,
> what fits in 6 GB, key training flags, `--full` meta-flag, precision, generation, tips.
> VRAM-envelope issues #69/#70 live here; the closed-issue ledger is [features.md](features.md).

## HVN vs Transformer — Fundamental Differences

| Aspect | Transformer (MHA) | HVN |
|--------|----------------------|---------|
| Core mechanism | Self-Attention (Q·K^T over all tokens) | Vortex particles interacting with a fixed-size Hamiltonian |
| Memory of past | KV Cache (grows with N) | Hamiltonian matrix H (constant d×d) |
| Per-token inference | O(N·d) | O(d²·r) — constant in N |
| Inference memory | O(N·d) | O(d²) — fixed |
| Positional encoding | RoPE / learned / sinusoidal | RoPE in vortex injection |
| Parallelism | Fully parallel | Sequential H update (chunked/parallel-scan mitigation) |

## What Fits in 6 GB VRAM (RTX 4050)

| Config | d_model | Layers | Params | VRAM | Fits? |
|--------|---------|--------|--------|------|-------|
| Small (default) | 128 | 2 | ~265K | ~14 MB | ✅ Easily |
| Medium | 256 | 4 | ~1M | ~50–100 MB | ✅ Easily |
| Large | 512 | 8 | ~5–10M | ~200–500 MB | ✅ Yes |
| Very Large | 512 | 12 | ~15M | ~1–2 GB | ✅ Yes |

**Recommended balanced command:**
```bash
python train.py --text my_corpus.txt \
    --d_model 256 --num_layers 4 --bond_dim 64 \
    --batch_size 16 --seq_len 128 --precision bf16 --compile \
    --epochs 100 --generate --prompt "Chapter 1 "
```

**On OOM, reduce in order:** `--batch_size 8|4` → `--seq_len 64` → `--d_model 128`, then
add `--grad_accum 4` to recover effective batch.

### Measured RTX 4050 training envelope (Issue #69)

Measured with `python -m eval.vram_envelope` (DeltaNet block, vocab 8000, B=8, S=128, bf16,
fused AdamW; full grid in `results/vram_envelope_rtx4050.json`):

| Config | Params | Peak VRAM | s/step | Trains in 6 GB? |
|---|---:|---:|---:|---|
| d=512, 8L | 29M | 1.0 GiB | 0.39 | ✅ |
| d=768, 12L | 83M | 2.5 GiB | 0.49 | ✅ |
| d=1024, 12L | 143M | 3.6 GiB | 0.52 | ✅ **max without `--grad_checkpoint`** |
| d=1024, 24L | 268M | 6.9 GiB | 2.28 | ❌ (spills → 4.4× slower) |
| d=1024, 24L `+ckpt` | 268M | **4.3 GiB** | 1.13 | ✅ **recommended ceiling config** |
| d=1024, 24L, B=16, S=256 `+ckpt` | 268M | 4.7 GiB | 2.28 | ✅ (ckpt crushes activations) |
| d=1280, 24L `+ckpt` | 414M | 7.0 GiB | 1.46 | ❌ (fits 8 GB — see #70) |
| d=1536, 24L `+ckpt` | 592M | 10.0 GiB | 1.95 | ❌ |
| d=2048, 24L (±ckpt) | ~1B | — | — | ❌ hard OOM at allocation |

**Honest ceiling: ~270M params** (d=1024, 24L, bf16 + `--grad_checkpoint`, 4.3 GiB peak) —
measured **16.8 bytes/param** end-to-end (bf16 weights + grads + fp32 AdamW moments +
master). **1B (~16 GB) and 10B (~160 GB) cannot train on this card — off by ~3× and ~30×** —
and 10B can't even bf16-load (20 GB weights). Knobs ranked by measured effect: bf16
(#19/#44) → `--grad_checkpoint` (#49; −38% peak at d=1024/24L, the difference between
fitting and spilling) → smaller batch/seq (+`--grad_accum`) → 8-bit optimizer (not wired) →
CPU/NVMe offload (not wired).

> **Windows WDDM spill-over caveat:** past ~6 GiB the driver silently spills into system
> RAM instead of OOM-ing — steps get 4–15× slower (see the 6.9 GiB row) and the run *looks*
> alive. On Linux the same configs hard-OOM. Treat `peak < 6 GiB` as the envelope, not "it
> didn't crash"; the >6 GiB rows in the JSON record true allocator demand (for #70).

### Block-coordinate-descent (Issue #99) — the ~270M ceiling above is no longer the whole story

The **~48% of the 16.8 bytes/param budget that is AdamW's fp32 momentum+variance state**
doesn't have to be resident for the whole model at once. [BAdam](https://arxiv.org/abs/2404.02827)
(Luo et al., NeurIPS 2024) trains one parameter block at a time (round-robin over layers),
explicitly clearing the outgoing block's optimizer state on every switch — every block gets
properly trained over a full cycle, but only ONE block's Adam moments are ever VRAM-resident.
`--block-coordinate --bcd-steps-per-block N` (`hvn/core/block_coordinate.py`).

**Measured on this card** (`scripts/bench_block_coordinate_vram.py`, isolated subprocesses,
`results/bcd_vram.json`):

| Config | Params | Peak VRAM (vanilla) | Peak VRAM (`--block-coordinate`) | Reduction |
|---|---:|---:|---:|---|
| d=1024, 24L (no ckpt) | 268M | 6.9 GiB | **4.2 GiB** | **−39%** |
| d=2048, 24L | ~1.04B | ❌ hard OOM (12.6 GiB demand) | 9.0 GiB (still spills on 6 GiB) | OOM → *runs* |
| d=2048, 24L, B=2/S=64 | ~1.04B | — | 7.1 GiB (still ~1 GiB over) | closer, not clean yet |

**UPDATE (Issue #100) — stacked with `--grad_checkpoint`, 1B now CLEANLY fits on this 6 GB
card.** `EdgeHVN` never had gradient checkpointing wired (#49/#88 closed it only for the core
`HamiltonianVortexNetwork`, a different class with its own `forward()`) — fixed, mirroring
the exact core-model pattern, bit-identical parity verified (`tests/test_edgehvn_grad_checkpoint.py`,
5/5, incl. the hybrid `block_plan`). BCD shrinks *optimizer-state* memory; checkpointing
shrinks *activation* memory — independent axes, and they compose almost exactly as predicted:

| Config | Params | Vanilla | `--block-coordinate` | `--block-coordinate --grad-checkpoint` |
|---|---:|---:|---:|---:|
| d=1024, 24L | 268M | 6.9 GiB | 4.2 GiB (−39%) | — |
| d=2048, 24L | ~1.04B | ❌ hard OOM (12.6 GiB demand) | 9.0 GiB (spills) | **✅ 4.6 GiB — clean fit** |
| d=2048, 24L, B=2/S=64 | ~1.04B | — | 7.1 GiB (spills) | **✅ 4.4 GiB — clean fit, more headroom** |

(`results/bcd_ckpt_vram.json`.) **The "1B/10B cannot train on this card" framing above is
now corrected, not just softened: 1B trains locally today**, measured, no cloud/tuning
gamble required — a flat ~3× impossibility this morning, a working `--block-coordinate
--grad-checkpoint` combo by this afternoon. 10B (~336 GiB naive) is not re-evaluated here —
worth a follow-up probe now that both levers are proven, but likely still needs genuinely
different hardware (see the `2M` floor analysis below — none of the levers on this page
touch total weight storage).

### GaLore (Issue #84) — a real standalone win, but does NOT compound with #99

[GaLore](https://arxiv.org/abs/2403.03507) (Zhao et al., 2024) projects each 2-D weight
matrix's gradient into a rank-`r` subspace (via a periodically-refreshed SVD-derived
projection) *before* Adam tracks its moments — so Adam's state is `O(r)` instead of
`O(m·n)`. Unlike LoRA the **weights stay full-rank** the whole time; only the optimizer's
view of the gradient is compressed — still genuine full-parameter training.
`--optimizer galore --galore-rank N --galore-update-gap N` (`hvn/optimizers.py`).

**Measured on this card** (`scripts/bench_galore_vram.py`, `results/galore_vram.json`):

| Config | Params | Peak VRAM |
|---|---:|---:|
| d=1024, 24L, AdamW | 268M | 6.9 GiB |
| d=1024, 24L, **GaLore alone** | 268M | **5.2 GiB (−24%)** |
| d=2048, 24L, BCD+ckpt, AdamW | 1.04B | 4.65 GiB |
| d=2048, 24L, BCD+ckpt, **+ GaLore** | 1.04B | 4.43 GiB alloc / 4.69 GiB reserved — **flat, not additive** |
| d=3072, 32L, BCD+ckpt+GaLore | 3.07B | ❌ 12.5 GiB — hits the `2M` floor (see below) |

**Two honest findings, not oversold:**
- **GaLore alone gives a real 24% reduction** — well short of the paper's headline 65.5%,
  because that figure is *optimizer-state-only* on a 7B model where optimizer state
  dominates total memory. Here activation memory (untouched by GaLore) dilutes the
  total-peak number; 24% is what's actually measured, not a smaller version of 65.5%.
- **Stacked with #99 (block-coordinate-descent), the marginal gain is flat, not
  compounding.** BCD already limits optimizer state to *one block* at a time; GaLore then
  only compresses that single already-small block's state further — a small absolute
  saving on top of a big one, not a multiplicative win. Unlike #99+#100 (genuinely
  independent axes: optimizer state vs. activations), #84 and #99 **overlap** — both
  target optimizer state, so their benefits mostly don't add.
- **3.07B confirms the `2M` floor prediction below**: no optimizer-side lever (BCD, GaLore,
  checkpointing) touches weight storage, so pushing past the local ceiling hits a hard wall
  regardless of which of these three is used or how they're combined. **Directly tested and
  bracketed** (not just extrapolated): 1.38B fits at 5.96 GiB (~180 MB margin), 1.54B needs
  6.62 GiB and spills. **Real ceiling: ~1.3–1.4B params** (treat ~1.2B as the safe practical
  target with real headroom) — corrects an earlier, too-optimistic "~2–2.5B" extrapolation
  that a direct 2.05B test (8.4–8.6 GiB, spills badly) disproved.

### CPU weight offload (Issue #101) — the `2M` floor is fixed, but a new floor takes over

Extends #99 to attack the `2M` term directly: BCD already limits *optimizer state* to one
block at a time; this additionally streams every INACTIVE block's *weights* to CPU RAM,
evicting them from GPU entirely (not just their optimizer state). `hvn/core/
block_coordinate.py`'s `GPUBufferPool` (reusable, pre-allocated GPU buffers keyed by shape
signature — every DeltaNet layer at a given `d_model` shares one, so nothing is freshly
allocated after the first block of each shape is seen) + `BlockCoordinateScheduler
(offload_inactive_weights=True, device=...)`.

**Two real, subtle bugs found and fixed by testing before landing, not after:**
- A naive first design deferred eviction for every block downstream of the active one for
  its *entire* forward pass (not just its checkpoint recompute window) — technically
  correct but wasteful, leaving most of a deep model resident simultaneously. Fixed by
  exploiting that checkpoint's *initial* forward pass discards everything by design: safe
  to evict immediately after call #1 regardless of position; only the later recompute
  (call #2, only for downstream blocks) needs deferred, hook-based eviction.
- `register_cpu_home` was called *before* moving a block to CPU in one construction-time
  path, capturing the stale GPU tensor as the "home" to evict back to — evictions ran
  without error but silently re-pointed params at GPU instead of CPU. Caught by a dedicated
  device-residency test (gradient-parity alone did NOT catch this — values were untouched,
  only placement was wrong). Fixed by reordering + a defensive device-type guard in the pool.

**Measured on this card** (`scripts/bench_offload_vram.py`, `results/offload_vram_final.json`):

| Config | Params | Peak alloc | Peak reserved | sec/step |
|---|---:|---:|---:|---:|
| d=2048, 24L | 1.04B | 1.11 GiB | 4.21 GiB | 1.73s |
| d=2048, 36L | 1.54B | 1.15 GiB | 6.13 GiB | 4.54s |

**Peak allocated memory matches the design's promise** — ~1 GiB regardless of scaling
toward 1.5B, confirmed via an instrumented single-step trace (peaked at 716 MB before the
buffer pool, ~1.15 GiB after — both far below vanilla's multi-GB requirement). The buffer
pool also delivered a genuine **2.4× speedup** (10.7s → 4.5s/step at 1.54B) by eliminating
the CUDA allocator churn from constant fresh alloc/free on every block swap (measured
before the pool: 317 reserved memory segments, ~290 GB of cumulative allocate/free traffic
over just 6 steps).

**Peak *reserved* memory stayed ~6 GiB at 1.54B when this section was first written —
resolved by Issue #102 below. Do not read the 1.04B/1.54B table above as the final
ceiling; see #102's numbers instead.**

### CUDA reserved-memory floor (Issue #102) — root cause found and fixed, real ceiling ~3.3B

The `~6 GiB stuck reserved` gap above was investigated and its root cause found: it is
**not** offload-swap fragmentation (the #101 buffer pool already fixed that, evidenced by
its own 2.4× speedup) and **not** activation memory (cutting batch×seq 16× only moved
reserved 2.6%). A fine-grained trace of `torch.cuda.memory_reserved()` at every
construction phase found the actual cause: the **caller's own `model.to(device)`** (which
must happen before `BlockCoordinateScheduler` is even constructed) allocates ONE large
reserved CUDA segment sized for the *whole* model in a single shot. `_activate()`'s
construction-time eviction correctly frees almost everything (`memory_allocated()` drops
from ~5.9 GiB to ~62 MiB) — but PyTorch's caching allocator does not proactively return
freed segments to the driver, so `memory_reserved()` stays pinned at that original
~full-model size for the rest of the process.

**Fix: a single `torch.cuda.empty_cache()` call, once, immediately after
`BlockCoordinateScheduler`'s construction-time eviction** (`hvn/core/
block_coordinate.py`, `__init__`) — while the freed memory is still one clean, fully-empty
block from the original allocation, this call successfully releases nearly all of it.
Calling `empty_cache()` *later* (e.g. mid-training-loop, once the buffer pool's own
differently-sized allocations have mixed into the same segments, which is what earlier
attempts in this doc tried) barely helps — partial occupancy blocks segment release.
Timing is everything; the mechanism itself (`empty_cache()`) was correct all along.

**Measured effect at 1.54B**: peak reserved memory **6130 MiB → 1182 MiB** (an 81%
reduction), and training got **faster too** (4.54s/step → 2.53s/step) — less fragmented
reserved space makes the buffer pool's own allocations cheaper to satisfy.

**The real local ceiling, directly tested and bracketed** (`scripts/bench_offload_vram.py`,
`results/offload_ceiling_final.json`):

| Params | Peak reserved | sec/step |
|---:|---:|---:|
| 1.54B | 1.18 GiB | 2.53s |
| 2.05B | 1.23 GiB | 3.58s |
| 2.55B | 1.28 GiB | 5.33s |
| 3.05B | 1.33 GiB | 9.92s |
| 3.18B | 1.33 GiB | 18.97s |
| **3.30B** | **1.35 GiB** | **30.6s** |
| 3.39B | ❌ OOM (12.7 GiB demand) | — |

**Real ceiling: ~3.3B params** — a hard boundary (3.39B jumps straight to a 12.7 GiB
demand, not a gradual approach) — up from the earlier ~1.3–1.4B ceiling, itself up from a
flat "1B is impossible" this morning. **Important caveat: VRAM fitting and training speed
are separate claims here.** Step time climbs steeply with depth (2.5s at 1.5B → 30.6s at
3.3B) — the offload mechanism's per-step PCIe traffic scales with block count, so a config
that *fits* is not necessarily one that trains at a *useful pace*. Treat ~1.5–2B as the
practical sweet spot (comfortable VRAM margin, still reasonable step time); 3.3B is the
documented outer edge of what fits at all, not a recommended target.

**Extended finding — the 3.39B "hard boundary" above was NOT a per-block or layer-count
limit; it was the caller's own construction-time `model.to(device)` call, which is a
separate, further-fixable bottleneck.** Traced precisely: the OOM's traceback pointed
inside `EdgeHVN(...).to(device)` itself, not forward/backward — `nn.Module.to()` copies
parameters one at a time and needs the *whole model* transiently resident to do so.
EdgeHVN's default parameter dtype is fp32 (confirmed: 3.35B params × 4 bytes = 13.4 GiB),
which exceeds this card regardless of anything the offload scheduler does afterward, since
it hasn't been constructed yet at that point. Ruled out "it's about the number of
layers/blocks specifically" with a controlled comparison: a **wide-shallow** config (d=2896,
39 layers, ~3.32B params) fit fine (2.4 GiB) at a param count where the **narrow-deep**
79-layer config OOM'd — proving it's the fp32 `.to(device)` copy size, not block count.

**Fix: skip the caller's blanket `model.to(device)` entirely when offloading — construct
the model CPU-resident and let `BlockCoordinateScheduler` handle *all* placement from the
start**, including the very first block. No scheduler code changes were needed:
`_activate()`'s existing eviction logic already handles a CPU-resident starting model
correctly (`.to("cpu")` on an already-CPU tensor is a no-op). Wired as an opt-in
`skip_initial_to_device` flag in `scripts/bench_offload_vram.py`'s probe config (not yet
wired into `train_edge_conversational.py`'s main training path — that file has an internal
audit/checkpoint-verification step whose GPU-residency assumptions weren't traced under
time pressure; a deliberate, smaller scope than rewiring the main script today).

**Measured**: the 3.35B config (79 layers) that previously hard-OOM'd at construction
(12.7 GiB demand) now peaks at **~1 GiB** for a full forward+backward. Pushed further:
a **4.02B** config also completed forward+backward at ~1 GiB peak reserved — VRAM-wise,
the ceiling is now far higher than 3.3B and was not fully re-bracketed (diminishing value:
see the next paragraph). **The remaining constraint past ~3.3B is no longer VRAM at all —
it's wall-clock training speed and CPU-side model construction time** (the 4.02B
construction alone took ~21s on CPU, plus this machine's 15.29 GiB total RAM becomes the
next real ceiling for fp32 CPU-resident storage: a 4.65B config, needing ~18.6 GiB fp32,
timed out during construction from heavy page-file swapping — a soft, slow degradation,
not a hard OOM).

**Practical implication**: with this extended fix, model size is no longer VRAM-bounded on
this card at all in any range worth targeting — it is bounded by (a) CPU RAM for fp32
storage and (b) per-step training time.

### LRU residency cache (Issue #103) — landed, closes the "not yet built" lever above

`GPUBufferPool` now keeps a checkpointed block's buffers resident across its two per-step
invocations (call #1 forward, call #2 backward recompute) instead of evicting immediately,
sized via `offload_cache_blocks` (LRU eviction beyond that capacity). Verified correct
(gradient parity, cache-hit + no-pileup regression tests, 174/174 suite) and effective at
the practical ~1.5–2B sweet spot. **Honest limit found, not a bug**: on a 95-layer, ~4B
stress config, raising `cache_blocks` 8→32 did *not* help — backward recomputes in reverse
order, so a block must stay resident through nearly the whole stack to get reused, which no
affordable cache size reaches at that depth; bigger cache there only added VRAM pressure
(2.4→6.25 GiB) without a proportional hit-rate gain. Landed with #101/#102 in MR !90.

CPU-side fp32 storage for parked blocks and per-step transfer overlap remain open — see
issues #113 (bf16 parking), #114 (fp8 spike, gated on a real convergence check), and #115
(prefetch/double-buffer the next block's transfer against current compute).

### RTX 4060 (8 GB) envelope — measured demand, projected fit (Issue #70)

Memory **demand** is measured (card-independent; WDDM spill rows record true demand past
6 GiB); the 8 GB **fit** column is arithmetic against 8192 MiB — a **projection, not a
physical-4060 measurement** (step times don't transfer). Derived table:
`results/vram_envelope_rtx4060_projection.json`; validate on real hardware with
`python -m eval.vram_envelope`.

| Config | Params | Demand | 6 GB (4050) | 8 GB (4060) |
|---|---:|---:|---|---|
| d=1024, 12L | 143M | 3.6 GiB | ✅ | ✅ |
| d=1024, 24L (no ckpt) | 268M | 6.9 GiB | ❌ | ✅ **max without ckpt** |
| d=1024, 24L `+ckpt` | 268M | 4.3 GiB | ✅ | ✅ comfortable |
| d=1280, 20L (no ckpt) | 349M | 8.3 GiB | ❌ | ❌ just misses |
| d=1280, 24L `+ckpt` | 414M | 7.0 GiB | ❌ | ✅ **recommended 4060 ceiling** |
| d=1408, 24L `+ckpt` | 499M | 9.5 GiB | ❌ | ❌ |
| ~1B / 10B | — | ~16 / ~160 GB | ❌ | ❌ (off by ~2× / ~20×) |

**What the extra 2 GB buys:** the no-checkpoint ceiling roughly doubles (143M → ~270M) and
the checkpointed ceiling rises ~54% (270M → ~415M, d=1280/24L bf16 `+ckpt` at 7.0 GiB) — a
bigger comfortable config and more bf16 width headroom (#44), **not a categorical change**:
1B and 10B remain infeasible on 8 GB.

> **"RTX 4060" caveat:** laptop and desktop 4060 are 8 GB, but the desktop **4060 Ti** ships
> in 8 GB *or 16 GB* variants. A 16 GB Ti roughly doubles this envelope — still far short of
> training 1B comfortably or 10B at all.

## Key Training Flags

| Flag | Default | Description |
|------|---------|-------------|
| `--text` | (sample) | Path to training text |
| `--epochs` | 30 | Training epochs |
| `--lr` | 3e-4 | Learning rate (AdamW) |
| `--batch_size` / `--seq_len` | 16 / 64 | Batch size / sequence length |
| `--d_model` / `--num_layers` / `--bond_dim` | 128 / 2 / 32 | Model size / SVD rank |
| `--precision` | fp32 | fp32 / fp16 / bf16 (GPU for 16-bit) |
| `--grad_accum` | 1 | Gradient accumulation steps |
| `--compile` | flag | torch.compile (14× per-batch) |
| `--tokenizer` / `--bpe_vocab` | char / 1000 | char or bpe tokenizer |
| `--rank_k` | 1 | Rank-k entanglement (Issue #21 S1); 4/8 recommended |
| `--nonlinear_update` | flag | Nonlinear gated update (S4) |
| `--causal_reinjection` | flag | Multi-chunk causal re-injection (S2) |
| `--cross_head_mixing` | flag | Cross-head H mixing (S3) |
| `--swiglu` | flag | SwiGLU FFN (S6) |
| `--per_token_eta` | flag | Per-token entanglement rate (S5) |
| `--n_threads` | 1 | Independent H threads per head (Issue #22); 4 recommended |
| `--use_parallel_scan` | flag | Associative parallel scan (Issue #33 Fix 3) |
| `--use_deltanet` | flag | Gated-DeltaNet dynamics (Issue #34) |
| `--weightless` | flag | Freeze DeltaNet q/k/v/out as seed-regenerated random features (Issue #40; needs `--use_deltanet`) |
| `--grad_checkpoint` | flag | Exact gradient (activation) checkpointing (Issue #49): bit-identical gradients, ~20-30% slower/step, ~64% lower peak activation memory |
| `--full` | flag | Issue #50 meta-flag: enable the **complete recommended fast-training stack** in one switch. Explicit flags override. See below. |

**The `--full` meta-flag (Issue #50).** One switch that turns on the whole recommended
stack. Expands into two **honestly-labelled** categories:

| Category | What it enables | Honesty |
|----------|-----------------|---------|
| **A — exact speed levers** (model bit-identical) | fused AdamW · per-epoch loss sync · pinned-memory + `non_blocking` H2D · `--compile` · `--grad_checkpoint` | Speed **only** — trained model unchanged. |
| **B — recommended defaults** (change model/precision) | `--precision bf16` · `--use_deltanet` (the #39-recommended higher-quality block) | **Not** quality-neutral — `--full` trains the best/fastest *recommended* model. |

Does **not** enable `--use_parallel_scan` (a no-op under `--use_deltanet`). First run prints
a one-line banner listing what it turned on (`[A]`/`[B]` tags). **Measured:** steady-state
**1.58× faster** (5.53 → 3.51 s/epoch, d=256/3L/seq128/B32); Category-A primitives proven
exact (`tests/test_issue50.py`). Explicit flags always win.

## Precision Guide

| Precision | Speed | Memory | Stability | Notes |
|-----------|-------|--------|-----------|-------|
| fp32 | 1× | 1× | Best | Default |
| fp16 | 2–3× | 0.5× | Good | GradScaler; may overflow |
| bf16 | 2–3× | 0.5× | Best | No scaler (RTX 40-series+) |

## Generation

```bash
python generate.py --prompt "Hamiltonian " --tokens 300 --temp 0.5 --top_k 5
```

| Flag | Default | Description |
|------|---------|-------------|
| `--checkpoint` | hvn_checkpoint.pt | Trained model path |
| `--prompt` | "The " | Seed text |
| `--tokens` | 200 | Max new tokens |
| `--temp` | 0.8 | Temperature (0.1–0.3 deterministic, 0.5–0.7 focused, 0.8–1.0 balanced, 1.0+ wild) |
| `--top_k` | 10 | Top-k sampling (0 = off) |

## Tips

More/real data (bundled sample is only ~34K chars), 50–100+ epochs on small data, larger
model (`--d_model 256 --num_layers 4 --bond_dim 64`), longer sequences (`--seq_len 128+`),
BPE tokenizer for real projects, always `--precision bf16` + `--compile` on RTX 40-series.