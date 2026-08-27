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

> ## ⚠️ CORRECTION (2026-08-07) — this table does NOT describe the real training script
>
> **Every VRAM number in this section, and in the BCD/GaLore/offload sections below, was
> measured on a model with `VOCAB = 8000`. The real pretrain uses GPT-2's 50,257.** The
> constant is hardcoded in every bench: `scripts/bench_block_coordinate_vram.py:31`,
> `bench_harness.py:36`, `bench_galore_vram.py:32`, `bench_offload_vram.py:29`,
> `bench_offload_prefetch.py:22`, `bench_offload_bf16_park.py:29`,
> `bench_offload_4b_8b.py:25`. Only `scripts/measure_speed_ceiling.py:56` uses 50257.
>
> The benches also build a plain `EdgeHVN` with **no ternary quantization and no
> `use_triton_scan`** (`bench_block_coordinate_vram.py:47-55`), while the training script
> applies `TernaryLinear` by default and the Triton scan's backward allocates a transient
> `[N,S,D,D]` fp32 tensor (`hvn/core/deltanet_triton.py:216`) — ~1.0 GiB per layer at the
> 1B shape. So the benched model is smaller *and* missing two real allocations.
>
> **Consequences, re-measured through the real `train_edge_conversational.py` path:**
>
> | claim in this doc | reality |
> |---|---|
> | `d=1024, 24L` is **268M** | **355,128,593 params** (the extra 87M is the vocab: `2 × (50257−8000) × 1024`) |
> | `+ckpt` → **4.3 GiB**, "recommended ceiling config" | **5.92 GiB alloc / 6.34 GiB reserved** with the script's default Muon → **spills a 5.9966 GiB card**. The 4.3 GiB figure is the forward/backward peak only and **excludes optimizer state entirely** |
> | `d=2048, 24L` is **~1.04B** | **1,213,523,217 params** |
> | `d=2048, 24L` BCD+ckpt → **"✅ 4.6 GiB clean fit"** (line ~96) | **6.17 GiB alloc / 7.28 GiB reserved** — 1.29 GiB *over* the card, silently backed by host RAM, **82 s per micro-batch** |
>
> **What actually fits at d=1024/24L** (real vocab, real script, measured at the moment
> optimizer state materialises — the cliff is visible inside a single run):
>
> | optimizer | peak alloc | peak reserved | fits? |
> |---|---:|---:|:--:|
> | Muon (script default) | 5.92 GiB | 6.34 GiB | ❌ |
> | AdamW | 6.86 GiB | 8.31 GiB | ❌ |
> | AdamW + `--tie-embeddings` | 6.28 GiB | 7.54 GiB | ❌ |
> | **Muon + `--block-coordinate`** | **3.43 GiB** | **3.91 GiB** | ✅ |
>
> So **`--block-coordinate` is mandatory at this size**, not optional — see the gate note
> in [state.md](state.md) before relying on it, because it has only ever been measured as a
> VRAM optimisation, never for its effect on model quality.
>
> **Do not trust `nvidia-smi` for any of this.** All four failing configs above report a
> healthy-looking `nvidia-smi` while spilling. Use `torch.cuda.max_memory_allocated()` /
> `max_memory_reserved()`.
>
> The rows below are left as originally measured (they are internally consistent at vocab
> 8000) but **must not be used to size a real run**. Rows not re-measured above are of
> unknown accuracy for the real script — assume they understate.

> ### ⛔ SUPERSEDED (2026-08-10, #136) — use `scripts/size_sweep.py`
>
> **`eval/vram_envelope.py` has been deleted, and the table below is kept only as a
> historical record.** It could not size a real run for three independent reasons:
>
> 1. **vocab 8000.** At d=1024 that is ~8M params of embedding against ~103M for an untied
>    GPT-2 vocabulary. #139 measured the vocabulary at **42–62%** of these models, so this
>    table makes the single largest term nearly invisible and **overstates** how much model
>    body the card can afford.
> 2. **Wrong model class.** It built `HamiltonianVortexNetwork`, which has no `block_plan`
>    and no `attn_window`. Every trained model here is `EdgeHVN` **hybrid**, with two
>    sliding-attention layers. The rows describe a pure DeltaNet trunk.
> 3. **No `tie_embeddings`**, so it cannot represent the current default.
>
> Current numbers, measured through `build_model_and_config` in the real `--fast` execution
> mode, are in [state.md](state.md) ("How big a model can this card actually train?") and
> `results/size_sweep_fast.json`. Headline: **129M body in 9.3 days at `d=896, 16L` with the
> 32K tied vocabulary**, against 95M body in 9.0 days for `d=768, 16L` untied GPT-2.
>
> Also note **`d_model / n_heads` must be a power of 2** or the Triton scan crashes under
> `--fast`; `check_head_dim()` now enforces this at startup.

Measured with the removed `eval/vram_envelope.py` (DeltaNet block, **vocab 8000** — see the
corrections above, B=8, S=128, bf16, fused AdamW; full grid in
`results/vram_envelope_rtx4050.json`):

| Config | Params | Peak VRAM | s/step | Trains in 6 GB? |
|---|---:|---:|---:|---|
| d=512, 8L | 29M | 1.0 GiB | 0.39 | ✅ |
| d=768, 12L | 83M | 2.5 GiB | 0.49 | ✅ |
| d=1024, 12L | 143M | 3.6 GiB | 0.52 | ✅ **max without `--grad_checkpoint`** |
| d=1024, 24L | 268M | 6.9 GiB | 2.28 | ❌ (spills → 4.4× slower) |
| d=1024, 24L `+ckpt` | 268M | **4.3 GiB** | 1.13 | ⚠️ **NOT a ceiling config — see correction above** (real: 355M, 6.34 GiB reserved, spills) |
| d=1024, 24L, B=16, S=256 `+ckpt` | 268M | 4.7 GiB | 2.28 | ✅ (ckpt crushes activations) |
| d=1280, 24L `+ckpt` | 414M | 7.0 GiB | 1.46 | ❌ (fits 8 GB — see #70) |
| d=1536, 24L `+ckpt` | 592M | 10.0 GiB | 1.95 | ❌ |
| d=2048, 24L (±ckpt) | ~1B | — | — | ❌ hard OOM at allocation |

> **⚠️ The "~270M honest ceiling" below is wrong for the real script.** At vocab 50257 that
> config is 355M and needs 6.34 GiB reserved with the default optimizer — it spills. It runs
> only with `--block-coordinate` (3.91 GiB reserved), validated end-to-end at 4,587 MiB
> steady. The paragraph is kept for the bytes/param derivation, which still holds.

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

### Launching a real large run — the flags that matter (2026-08-07, Issue #112)

> **⚠️ The `N = B·H` occupancy explanation below is REFUTED by measurement (Issue #141,
> 2026-08-10).** Head count is still a real win, but not for the reason given, and it is the
> *smaller* half of the lever. If `N = B·H` programs were what set throughput, two configs with
> equal `B·H` would run at equal speed. They do not: at `B·H = 16`, batch 2 × 8 heads gives
> **9,087 tok/s** while batch 1 × 16 heads gives **5,882** — a 55% gap at identical occupancy.
> **Batch size dominates head count.** See the #141 entry in [state.md](state.md) for the full
> grid, and for the `batch 2 → 4` change that buys +37.7% on the *existing* 83M checkpoint.

**Head count is a 2.4× speedup at 355M (1.12–1.17× at 83M — Issue #141).** The Triton scan is
`O(S·D²)` per program with `N = B·H` programs to fill 20 SMs
(`hvn/core/deltanet_triton.py:218,222`). Same size, same memory, only `--n-heads` changed:

| | params | s/micro-batch | peak alloc | reserved |
|---|---:|---:|---:|---:|
| `--n-heads 8` (d_head 128) | 355,128,593 | 1.91 | 3.43 GiB | 3.91 GiB |
| `--n-heads 16` (d_head 64) | 355,325,393 | **0.79** | 3.36 GiB | 3.80 GiB |

This changes the schedule materially: at 16,384 tokens/step, `h=8` gives ~1,082 tok/s
(**~76 days** for a Chinchilla 7.1B-token budget at 355M), `h=16` gives ~2,580 tok/s
(**~32 days**). Re-run a short end-to-end gate at `h=16` before committing — the number above
is a micro-batch measurement, not a full-run one.

**`--fast` is ON BY DEFAULT (#129) and must be turned off for BCD runs.** It forces
`torch.compile`, which recompiles at every BCD block switch. There is no partial escape:
`--compile` is a bare `store_true`, so the only way to name it is to enable it. Use
`--no-fast` and re-add `--precision bf16 --tf32 --triton-scan` by hand.

**Nothing prints until micro-batch 400.** Console output is gated on
`global_step % log_every_steps` with `log_every_steps` hardcoded to 50
(`train_edge_conversational.py:1162`, no CLI flag) and `grad_accum` defaulting to 8. Even the
healthy 83M reference takes 2.3 minutes to reach its first line. Time that before concluding a
run is stuck — this exact silence was misdiagnosed twice, once as a slow compile and once as a
driver crash.

**Resume is lossy, and a month-long run will restart.** `load_checkpoint_for_resume()`
(`:1500-1549`) restores no optimizer state, `_save_edge_checkpoint()` (`:81-104`) saves none,
`train()` resets the cosine schedule and skips warmup on resume (`:1772-1777`), and
`BlockCoordinateScheduler` is rebuilt fresh at `:1744` — so **every restart rewinds the BCD
cycle to block 0**, systematically undertraining later blocks across many restarts. Fix this
before a multi-week BCD run, not after.

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
| d=2048, 24L | ~1.04B | ❌ hard OOM (12.6 GiB demand) | 9.0 GiB (spills) | ~~✅ 4.6 GiB — clean fit~~ ⚠️ **REFUTED: 7.28 GiB reserved, spills** |
| d=2048, 24L, B=2/S=64 | ~1.04B | — | 7.1 GiB (spills) | ~~✅ 4.4 GiB — clean fit~~ ⚠️ not re-measured; assume understated |

> **⚠️ CORRECTION (2026-08-07).** The two "clean fit" cells are wrong — same vocab-8000 bench
> problem as the top-of-page correction. Re-measured through the real script at `d=2048, 24L,
> h=8, B=4, S=128, --grad-checkpoint --block-coordinate`: **1,213,523,217 params** (not 1.04B),
> **6.17 GiB alloc / 7.28 GiB reserved** on a 5.9966 GiB card. WDDM backed the 1.29 GiB overflow
> with host RAM rather than OOM-ing, giving **82 s per micro-batch** — and since console output
> is gated on `global_step % 50` with `grad_accum=8`, the first log line would have appeared
> **9.2 hours in**. That is the entire explanation for the "1.21B stalls silently" report in
> [state.md](state.md); nothing was hung. Isolation: same `d_model`/`d_head`/kernel at 6 layers
> instead of 24 fits (4.40 GiB reserved) and runs at 1.66 s/micro — a **12.4× per-layer
> slowdown from memory pressure alone**, kernel held constant.

(`results/bcd_ckpt_vram.json`.) **The "1B/10B cannot train on this card" framing above is
now corrected, not just softened: 1B trains locally today**, measured, no cloud/tuning
gamble required — a flat ~3× impossibility this morning, a working `--block-coordinate
--grad-checkpoint` combo by this afternoon.
> **⚠️ That claim does not survive the 2026-08-07 re-measurement either.** 1B does **not**
> train locally at the real vocab — the config above spills. What *is* validated end-to-end
> (55 steps, real data pipeline, audit passed, checkpoint written, clean exit) is **355M**:
> `--d-model 1024 --num-layers 24 --n-heads 8 --grad-checkpoint --block-coordinate --no-fast
> --precision bf16 --tf32 --triton-scan --num-workers 0` → 4,587 MiB steady, 15.14 s/step. 10B (~336 GiB naive) is not re-evaluated here —
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

### P0 correctness bug found and fixed (Issue #117) — read before trusting any prior multi-cycle BCD+offload run

`GPUBufferPool.register_cpu_home()`'s register-if-absent dedup meant a block's CPU-resident
"home" tensor was captured once ever (its untrained init value) and never refreshed on later
deactivations — every block silently reset to its pristine pre-training value the instant it
went inactive, and retrained the identical trajectory every cycle. **Zero cross-cycle
learning was actually happening under `offload_inactive_weights=True`** before this fix,
confirmed via a direct repro (a tracked weight's diff-from-init reset to exactly 0.000000 on
every deactivation, bit-identical trajectory every subsequent cycle). The VRAM/step-time
numbers measured for #101/#102/#103 above are unaffected (single/few-step memory
benchmarks, not multi-cycle convergence runs) — but this means no real training run using
offload across multiple BCD cycles, before this fix, actually accumulated learning beyond
whichever block was in its first active cycle. Fixed via a `force` param on
`register_cpu_home`; new regression test
`test_deactivated_block_retains_training_across_cycles` checks a specific block's value
survives a deactivate→reactivate round trip directly (the gap the existing
`test_training_converges_with_offload` had — it only checks aggregate early-vs-late loss,
which still trended down under this bug). Landed in MR !99.

### bf16 parking for inactive blocks (Issue #113) — landed, real measured win on both axes

Parks INACTIVE (BCD-frozen) blocks' weights in bf16 on CPU instead of their native dtype
(fp32 by default), via `offload_park_dtype=torch.bfloat16` on `BlockCoordinateScheduler`. The
ACTIVE block is always restored to its native dtype the instant it's reactivated (`
GPUBufferPool.native_dtype_for`) — the optimizer's tiny per-step updates need full native
precision to be representable; only already-frozen weights get stored coarser.

**Measured** (`scripts/bench_offload_bf16_park.py`, `results/offload_bf16_park.json`):

| Config | Parked CPU MB (fp32 → bf16) | Peak reserved VRAM | sec/step |
|---|---:|---:|---:|
| 268M | 984.3 → **492.2** (exactly halved) | 488 → 380 MiB | 0.938 → **0.823** (−12%) |
| 1.04B | 3808.6 → **1904.3** (exactly halved) | 1290 → 1058 MiB | 1.552 → **1.158** (−25%) |

Both predictions from the issue confirmed directly, not just reasoned: CPU RAM exactly
halves (as expected — bf16 is literally half the bytes of fp32), and step time drops
meaningfully too (the offload mechanism is bandwidth-bound, so moving half the bytes per
block-swap over PCIe genuinely speeds up training, not just shrinks memory) — the win
compounds with depth (−12% at 268M vs −25% at 1.04B).

**One real caveat found and fixed, not just anticipated**: forwarding through a bf16-parked
(frozen) block alongside an fp32-active block mixes dtypes, which PyTorch ops like `Conv1d`
reject outside autocast with a cryptic low-level error ("Input type and bias type should be
the same"). Confirmed directly (crashes several frames deep inside
`torch.nn.modules.conv` without autocast). Fixed with a fail-fast guard in `EdgeHVN.forward()`
that raises a clear, actionable `RuntimeError` if `offload_park_dtype` is set without an
active `torch.is_autocast_enabled()` context — this project's training scripts already wrap
forward+loss in `torch.amp.autocast("cuda", dtype=torch.bfloat16)` (e.g.
`scripts/bench_offload_vram.py`), so this is a documented requirement, not a blocker.

`offload_park_dtype` is opt-in (default `None`, byte-identical to pre-#113 behavior). 12/12
`tests/test_block_coordinate_offload.py` + 7/7 `tests/test_block_coordinate.py` pass.

### Prefetch/double-buffer transfer overlap (Issue #115) — landed, on by default, real ~25% win

Overlaps the NEXT offloadable segment's CPU→GPU transfer (on a dedicated side CUDA stream,
with pinned CPU home memory + event-based sync) with the CURRENT segment's compute, instead
of paying each block's transfer serially when its own turn comes. `GPUBufferPool.prefetch()`
mirrors `load()`'s buffer-acquisition path but issues the copy on `_prefetch_stream` and
records a completion event; `load()` waits on that event (a GPU-side stream dependency, not
a host-blocking sync) if the block was prefetched but not yet consumed. Wired into
`EdgeHVN.forward()`'s layer loop: right after processing block `i`, prefetches `i+1` (or the
tail, `final_norm+decoder`, on the last layer) — issued *after*, not before, block `i`'s own
`load()` call, since GPU-level concurrency across streams doesn't depend on Python-side
call order, and issuing it before would let a `cache_blocks=1` pool evict block `i`'s
just-loaded buffer out from under its own in-flight compute.

**Measured** (`scripts/bench_offload_prefetch.py`, `results/offload_prefetch.json`, A/B
against `offload_prefetch=False`):

| Config | serial → prefetch (sec/step) | Δ | Peak reserved VRAM |
|---|---:|---:|---:|
| 268M | 0.825 → **0.625** | −24% | 488 MiB (unchanged) |
| 1.04B | 1.420 → **1.047** | −26% | 1290 MiB (unchanged) |
| 1.54B | 2.198 → **1.604** | −27% | 1338 MiB (unchanged) |

A consistent ~25% step-time win at every depth tested, at **zero VRAM cost** — confirms this
is a pure speed lever (hiding transfer latency behind already-happening compute), not a
memory/speed tradeoff. `offload_prefetch` defaults to `True` (unlike #113/#114, no precision
risk to opt into — it's a scheduling change only, and the existing gradient-parity test
already exercises this path since it's on by default). Set `offload_prefetch=False` to fall
back to the fully serial pre-#115 behavior.

**One real correctness hazard found and fixed while building this, not just anticipated**:
`GPUBufferPool`'s existing capacity-overflow eviction path (inside `_acquire_buffers`, used
by both `load()` and the new `prefetch()`) could evict a block whose prefetch copy was
*still in flight* on the side stream, handing its buffer to `_free_buffers` for reuse by a
different block while the old async copy was still writing into it — a genuine data race,
not just staleness. Fixed by synchronizing on any pending event before releasing a buffer,
in `evict()`, `evict_all()`, and `_acquire_buffers`'s own overflow-eviction loop (the last of
which the first two callers alone didn't cover).

14/14 `tests/test_block_coordinate_offload.py` + 7/7 `tests/test_block_coordinate.py` pass.

### P1 bug found and fixed at real depth (Issue #118) — pinned host memory grew unbounded with layer count

#115's `prefetch()`, as originally landed, pinned each block's ACTUAL CPU home tensor
directly and left it pinned indefinitely — only replaced (and un-pinned) the next time that
specific block completed a full BCD activate/deactivate cycle. Undetected by #115's own
tests (3-4 layer toy models) and benchmarks (24-36 layers, never enough cumulative pinning to
hit a ceiling). **Confirmed via a real 4.02B run (d=2048, 95 layers)**: crashed partway
through the very first forward pass, from inside `pin_memory()` itself, with a CUDA
out-of-memory error — traced and confirmed this was neither GPU VRAM (peak reserved was 64
MiB at the point of failure) nor regular CPU RAM (the model's bf16 CPU footprint is ~8 GiB,
well under this machine's ~15.3 GiB), but specifically the OS/driver's **pinned (page-locked)
memory ceiling**, far tighter than either — a single forward pass prefetches every layer in
sequence, not just the BCD-active one, so nothing released a block's pinning within one
forward pass; by ~95 layers, cumulative pinning approached ~8 GiB.

**Fix**: `GPUBufferPool` no longer pins home tensors directly. `_PinnedStagingPool` maintains
a small, bounded pool of reusable pinned staging buffers keyed by shape signature (this
project's blocks are shape-homogeneous per layer type, so the number of distinct shapes is
small and depth-independent). `prefetch()` copies a block's home data into a staging buffer
(a fast plain host memcpy, not PCIe-bound) before issuing the actual async H2D copy from it;
the staging buffer returns to the pool once its copy is confirmed complete via a **host-side**
`event.synchronize()` (not just the stream-level `wait_event()` used to make the GPU buffer
itself safe for compute — a plain host memcpy reusing the staging buffer needs a true host
block, not just a GPU-side dependency, or it could overwrite memory an in-flight async copy
was still reading from).

**Re-measured after the fix**: the same 4.02B config that crashed now completes cleanly —
**938 MiB peak allocated, 1102 MiB peak reserved, 4.25 sec/step**
(`scripts/bench_offload_4b_8b.py`, `results/offload_4b_8b.json`). An **8.01B** config
(d=2048, 190 layers) was also attempted at the same time — it fails, but at a different,
already-known, unrelated stage: real (non-meta) fp32 CPU construction of an 8B-param model
needs ~32 GiB, exceeding this machine's ~15.3 GiB RAM (the same pre-existing construction-time
ceiling documented in #102's extended finding for the 4.65 GiB config, just further out) —
confirmed directly, not assumed (`torch.device("meta")` gives an accurate param count without
allocating; the real, non-meta construction segfaults on its own, before the offload/GPU
machinery is ever reached). Not a regression from #118 or anything this session touched — a
separate, pre-existing hardware ceiling.

New regression test `test_prefetch_pinned_memory_bounded_at_depth`: exercises the actual root
cause (pinned-allocation count scaling with layer count) cheaply, via 30 tiny layers rather
than a slow multi-GB real-depth run — asserts the number of distinct pinned buffers ever
allocated stays bounded by (num shapes × max_per_shape), not by the 30-layer count. 15/15
`tests/test_block_coordinate_offload.py` + 7/7 `tests/test_block_coordinate.py` pass.

**Follow-up measurement: the real ceiling on this hardware is closer to ~6B than 4B, and the
8B block is specifically CPU-RAM-at-construction, not VRAM.** Bisected directly rather than
assumed:

| Params | fp32 construction RAM needed | Real construction result |
|---|---:|---|
| 4.02B | 16.1 GiB | ✅ works (~21s) |
| **5.995B** | 24.0 GiB | ✅ **works** (~33s — visibly slower, likely some page-file swap, but completes, not a crash) |
| 8.01B | 32.0 GiB | ❌ segfaults (hard OOM, not a slow degradation) |

**5.995B (d=2048, 142 layers) trains cleanly** with the full offload+bf16-park+prefetch
stack: 985 MiB peak alloc, 1168 MiB peak reserved, 6.92 sec/step at batch_size=2; **batch_size=8
also fits comfortably** (1612 MiB alloc, 1950 MiB reserved, 8.83 sec/step) — both far under
the 6 GiB card. VRAM stays small regardless of param count under this offload scheme (it's
governed by activation memory + the bounded per-block buffer/staging pools, not total
weights); the real ceiling for *this* hardware is CPU RAM at construction time, and bf16-park
being CPU RAM only bounds it via `torch.bfloat16` parked weights only in aggregate.

**Batch-size ceiling, measured at 4.02B** (`d_model` swept, `seq_len=128`):

| batch_size | peak_alloc | peak_reserved | sec/step |
|---:|---:|---:|---:|
| 2 | 938 MB | 1102 MB | 4.35 |
| 4 | 1076 MB | 1142 MB | 4.31 |
| 8 | 1424 MB | 1566 MB | 5.44 |
| 16 | 2128 MB | 2600 MB | 9.22 |
| 32 | 3536 MB | 4946 MB | 18.89 |
| 48 | — | — | stalled >300s (consistent with WDDM spill-over past ~6 GiB, not a hard OOM) |

**`offload_cache_blocks` (round-trip reduction) at 95-layer depth confirms the #103
structural limit directly, not just extrapolated**: raising it from 1→32 costs +2.8 GiB VRAM
for only ~8% faster steps (4.34s→4.00s) — a poor trade at this depth, because backward
recomputes in reverse layer order, so a block must survive resident nearly the entire stack
to get reused, which no practical cache size reaches at 95 layers. Not recommended past the
default at this scale; #103 already documented this working well at the shallower ~1.5-2B
sweet spot instead.

`scripts/bench_offload_4b_8b.py` gained an `offload_cache_blocks` passthrough (was silently
not wired to the probe config in the version committed with #118 -- caught while running this
sweep, not by any test, since the bug just made every `cache_blocks` value produce identical
results rather than erroring).

### RTX 4060 (8 GB) envelope — measured demand, projected fit (Issue #70)

Memory **demand** is measured (card-independent; WDDM spill rows record true demand past
6 GiB); the 8 GB **fit** column is arithmetic against 8192 MiB — a **projection, not a
physical-4060 measurement** (step times don't transfer). Derived table:
`results/vram_envelope_rtx4060_projection.json`; validate on real hardware with
`python scripts/size_sweep.py` (which replaced `eval/vram_envelope.py` in #136 — the
projection below inherits that tool's vocab-8000 and wrong-model-class caveats).

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

## Pretrain data readiness (Issues #119/#120) — memory/speed alone does not solve capability

A data-readiness audit (2026-07-26, prompted by the 4-6B ceiling above becoming trainable)
found the pretrain data pipeline had its own, separate bottleneck — solving VRAM/RAM/speed
doesn't solve data volume; this project's own `docs/roadmap.md` epic #78 closing note already
said as much in writing before this was investigated further.

**Chinchilla-optimal token targets** (20 tokens/param, this project's convention, per issue
#112): 4B params → 80B tokens; 6B params → 120B tokens.

### #119 — the mix silently caps total token budget, same mechanism as the old OASST bug

`MixedLocalDataset`'s `max_repeat` bind on whichever mix source is smallest relative to its
target share caps the *entire* dataset's token budget, not just that source's own
contribution — this is exactly what capped the 83M pretrain at ~135M tokens via OASST
(§ "EdgeHVN 83M" above). The same mechanism recurs with WikiText-103 (a fixed ~119M-token
corpus, nothing more to download) at its usual 15% pretrain share: confirmed directly, this
caps total achievable tokens around **~2.4B** with `max_repeat=3` — ~2-3% of the 80-120B
floor for 4-6B params.

**Fix**: rather than force a single "right" default (WikiText is fine at small scale, e.g.
the 83M/300M runs — the problem only bites once a run targets real Chinchilla volume),
`MixedLocalDataset` now prints a loud, correct warning every time, naming which of two
genuinely different cases applies:
- **MIX-RATIO bind** (`total_cap < sum(avail)`): a source's `max_repeat` is the active
  constraint — names the binding source, fixable by changing the mix.
- **DATA-VOLUME limit** (`sum(avail) <= total_cap`): every available window is already used,
  no source's repeat cap is binding — fixable only by downloading more data.

Caught a real bug in the first draft of this fix: it always blamed whichever source's
repeat-cap term was numerically smallest, even when that term wasn't the ACTIVE constraint.
Verified against this project's real cached `data_cache` (not synthetic): with only ~1.32B
SlimPajama tokens currently cached, the default 0.85/0.0/0.15 mix is presently
DATA-VOLUME-limited, not MIX-RATIO-limited by WikiText — though WikiText IS confirmed (via a
`max_repeat=1` test against the same real data) to become the binding mix-ratio constraint
once more data is downloaded, matching the original audit finding. `tests/test_issue119_mix_cap_warning.py`, 3/3 pass.

### #120 — switched pretrain data acquisition to FineWeb-Edu

SlimPajama's reupload mirror (`gmongaras/SlimPajama-627B_Reupload`) is impractical at
Chinchilla-optimal volume on this machine: ~6 GB/shard parquet, would need >1 TB disk
(exceeds the ~642 GB free) and ~13-18 hours at this connection's ~22.7 MB/s. **FineWeb-Edu**
(`HuggingFaceFW/fineweb-edu`) — already this project's own intended source per
`docs/FRONTIER_TRAINING_PLAN.md`, never executed until now — is strictly better on every
axis measured:

| | SlimPajama reupload | FineWeb-Edu |
|---|---|---|
| Quality | dedup-only cleaning | LLM-classifier-filtered for educational value |
| File granularity | ~6 GB/shard, 250 total | ~2.15 GB/file, pre-sized `sample-10BT`/`100BT`/`350BT` configs |
| Tokens per GB | ~0.44B tok / 1.72GB text-extracted | ~1.84 GB per 1B tokens (measured, `sample-350BT`: 300 files) |
| Auth | hit an unrelated expired-token issue (see below) | none needed, confirmed |
| 120B-token (6B-param) cost | doesn't fit in free disk | ~103 files, ~221 GB, ~2.7 hours (measured bandwidth) |

**Two real environment blockers found and fixed to get this far, unrelated to the data
question itself:**
- **SSL**: all HTTPS on this machine failed (`CERTIFICATE_VERIFY_FAILED: Basic Constraints of
  CA cert not marked critical`) — traced to Avast Antivirus's HTTPS-scanning root cert,
  malformed per strict X.509 rules (in the Windows trust store, which browsers use, but not
  Python's `certifi` bundle — and adding it to certifi wasn't sufficient, since the cert
  itself is structurally non-compliant, not just untrusted). Worked around with SSL
  verification disabled scoped specifically to `huggingface_hub`'s own HTTP client
  (`set_client_factory`), not system-wide.
- **Auth**: the cached HF token (825 bytes — far too large for a normal ~40-char token) is
  malformed/expired, causing `401 OAuth token signature verification failed`. Both
  SlimPajama's reupload and FineWeb-Edu work fine anonymously (`token=False`) for public
  files — no token needed at all.

**Integration**: `MixedLocalDataset`'s mix generalized from a 3-tuple (SlimPajama, OASST,
WikiText) to a 4-tuple (+ FineWeb-Edu). New `_load_fineweb_edu()` extracts text from locally
downloaded parquet files (`scripts/download_fineweb_edu.py` handles acquisition separately —
a 100+ file pull can take hours and shouldn't block a training script's startup) into its
OWN `.bin` cache, not merged into SlimPajama's — `_tokenize_text_to_bin`'s cache is keyed
only on the output `.bin` existing, so silently appending more text to SlimPajama's existing
1.32B-token cache would never get picked up without invalidating already-tokenized work.

**Real bug found and fixed while integrating this**: unlike the other 3 sources (all under
~5 GB, so preparing them unconditionally regardless of mix share is cheap, the existing
pattern), FineWeb-Edu can be hundreds of GB — an early version loaded/tokenized it
unconditionally, which caused genuinely slow test runs and, worse, a real file collision
(`PermissionError` on `fineweb_edu.bin.tmp`) between a stale unconditional-load test process
and a concurrent real download, that also silently wrote a runaway 66 GB extraction from a
still-in-progress download. Fixed: FineWeb-Edu is only loaded/tokenized when its mix share is
actually > 0. A nonzero share with nothing downloaded yet fails with a clear
`FileNotFoundError` pointing at `scripts/download_fineweb_edu.py`, rather than silently
renormalizing the other sources to quietly cover the gap.

`tests/test_issue120_fineweb_edu.py`, 5/5 pass (real parquet extraction, real end-to-end
`MixedLocalDataset` construction with FineWeb-Edu as the sole source — not mocked).

Recommended mix for 4B+ pretrain runs: `--pretrain-mix 0.0,0.0,0.0,1.0` (FineWeb-Edu only) —
sidesteps WikiText's #119 mix-ratio cap by construction rather than needing to tune shares.

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

### ⚠️ CORRECTED — `torch.compile` is a NET LOSS here, measured in the real loop (#158, 2026-08-12)

**The +18% below was wrong.** It came from a 4-step, 1-micro-batch microbenchmark — a shape
training never runs. Measured in the actual loop (seq 4096, grad_accum 8, `--grad-checkpoint`):

| config | s/step | tok/s | GPU |
|---|---:|---:|---:|
| **eager + Triton scan + megakernel** | **8.66** | **3,784** | **98%** |
| per-block compile + those kernels | 20.72 | 1,581 | 35% |
| per-block compile, kernels OFF | 21.42 | 1,530 | 36% |

**2.4× slower**, and turning the fused kernels off does not help — so it is *not* the
graph-break interaction #129 documented, it is compile itself on this architecture.

**The GPU signature is the tell: 36% mean, 96% max.** It bursts to full then stalls, which is a
gap *between* kernels rather than slow kernels. VRAM was 4,394 MiB of 6,140 — nothing offloaded,
nothing spilled. Unverified hypothesis: recomputation under `--grad-checkpoint` forces a sync per
checkpointed segment — 16 blocks × 8 micro-batches = 128 stalls per step, invisible at 1
micro-batch and dominant at 8.

**This is the FOURTH independent time compile has lost on this project**: #129 (megakernel breaks
fusion, 2 graph breaks), state.md:558 (`--block-coordinate` conflict, 6.73 vs 4.41 s/step at
355M), the Inductor `MemoryError` at 16 layers, and now this. `--fast` keeps bf16 + TF32 + Triton
scan + megakernel and **skips `torch.compile`**.

**What still stands from the section below:** whole-model compile is genuinely impossible at this
depth (`compute_ancestors` is O(N²), three reproductions). Per-block compile *works* — it just
isn't worth using.

### (superseded — see the correction above) Why whole-model `torch.compile` cannot work at this depth (#158)

**Read this before enabling `--fast` on anything deeper than ~12 layers, and before renting a
box to train something larger.**

`torch.compile(model)` on the whole 157M hybrid (16 layers, `--grad-checkpoint`) **fails**, every
time, with:

    torch._inductor.exc.InductorError: MemoryError
      scheduler.py compute_ancestors: ancestors |= name_to_ancestors[dep_node_name]

Reproduced **three times**: seq 4096 and seq 2048, with a browser open and on a fresh boot with
6.9 GB free. It is **not memory pressure** — it failed with ~24 GB of commit headroom left
(15.3 GB physical + 20 GB pagefile, ~10.8 GB committed).

**The mechanism, from the source.** `Scheduler.compute_ancestors` builds `name_to_ancestors`, a
complete ancestor set for **every node, all alive simultaneously**. That is **O(N²)** in graph
nodes. Gradient checkpointing makes it worse by adding the recomputation graph. At 16 layers, N²
strings do not fit in RAM.

**The fix: compile each block separately.** 16 graphs of N/16 nodes cost `16 × (N/16)² = N²/16`
— a **16× memory reduction** — while keeping nearly all the benefit, because fusion opportunities
live *within* a block (norm → projection → gate → scan → FFN), not across block boundaries.

| strategy | s/step | vs eager | compile |
|---|---:|---:|---|
| eager | 1.28 | — | — |
| **per-block** | **1.09** | **1.18×** | 26 s first, **3 s cached** |
| whole-model | — | — | **MemoryError after 40+ min** |

**The tuning knobs do nothing.** `realize_opcount_threshold=8` and `epilogue_fusion=False` both
land at *exactly* eager's 1.28 s — they do not compile anything useful. Only splitting the graph
works. (`scripts/test_compile_strategies.py` runs each strategy in a fresh process, because a
failed Inductor compile leaves cache state that flatters the next attempt.)

**Why this matters more at scale, not less.** The failure is quadratic in depth, so a 24-layer or
6B model would fail *harder*. Any plan to rent a multi-GPU box and enable `--fast` needs this —
whole-model compile would burn an hour per launch and then die. Per-layer compilation is what
large-model training frameworks do, for exactly this reason.

**A correction worth recording.** This was nearly dismissed: an early cross-harness comparison
suggested compile was worth ~1%, and that number was used to argue the whole question was not
worth chasing. The maintainer rejected it — *"1% on a 6B model is huge"* — and insisted it be made
to work. Measured properly, it is **18%**, which on a 9.5-day run is **~1.7 days**. The lesson is
the repo's own rule: a number from a different harness is not a measurement of this one.

### `--fast` — the EdgeHVN equivalent, and what every real run here actually used

`train.py`'s `--full` has a counterpart in `train_edge_conversational.py`: **`--fast`**, which
expands to **bf16 + TF32 + `torch.compile` + Triton scan + async DataLoader (4 workers)**.
Explicitly-passed levers override the preset (e.g. `--fast --precision fp32`).

> **Update (#142/#143):** `--fast` is now also the default on `continue_ft_hybrid.py` and
> `finetune_edge_conversational.py`, each measured in its own loop (+39% and +25.5%). See
> "Fine-tune recipe defaults" below. `continue_ft_looped.py` deliberately has no compile path.

**Use `--fast` for every real training run.** This is not a suggestion added after the fact —
it is what the project already does. `logs/chinchilla20k_pretrain.log`,
`logs/chinchilla83m_pretrain.log` and `logs/hybrid_a4_2attn_102k.log` all open with the
`[--fast]` banner. Until now the flag appeared **only in `--help`** and nowhere in `docs/` or
the README, so a reader following the documentation would have run ~5x slower than every
result in this file was produced at. (Measured directly while setting up the #129 A/B:
0.2 steps/s without it vs the ~0.8 steps/s the speed-ceiling work records with it, at
d=512/12L, batch 4, seq 512.)

**Do NOT use it for a controlled fp32 A/B.** The bare defaults (fp32, `num_workers=0`, no
compile) exist deliberately so both arms of a parity test are bit-comparable; `--fast`'s
bf16/TF32 change precision. The bit-identical levers (compile / Triton scan / workers) never
change the model, so an A/B that only needs *speed* parity can keep those.

**Known, expected noise under `--fast`:** `torch.compile` logs
`Not enough SMs to use max_autotune_gemm mode` on this card and falls back — documented in the
speed-ceiling investigation, not an error. `reduce-overhead`/CUDA-graph modes do crash on
`nn.Embedding` here; plain `torch.compile` (what `--fast` uses) is the supported configuration.

### Fine-tune recipe defaults (Issues #142/#143, 2026-08-11) — and why pretrain gained nothing

**The throughput wins from #141 were a fine-tuning story, not a pretraining one.** Pretrain
already had both levers; the FT scripts were the ones running slow. Current defaults:

| script | micro-batch | grad_accum | compile | changed? |
|---|---|---|---|---|
| `train_edge_conversational.py` (pretrain) | 4 (hardware scan) | 8 | `--fast` on (#129) | **no — already had both** |
| `continue_ft_hybrid.py` | 4 | 8 | **`--fast` now ON** | compile (#142) |
| `finetune_edge_conversational.py` | **4** (was 2) | **4** (was 8) | **`--fast` added, ON** | both (#142/#143) |
| `continue_ft_looped.py` | 4 | 8 | none, deliberately | no |

**Why pretrain was already fine:** `hardware_scan()` picks `batch_size 4 / grad_accum 8`
whenever free VRAM ≥ 2.5 GB, and only falls back to `batch 2 / grad_accum 16` below that.
On this card it has always chosen 4. `--fast` has been the default since #129.

> **The low-VRAM fallback now has a known price.** `hardware_scan()`'s `batch 2 / grad_accum 16`
> branch was written purely as a memory accommodation. #141 measured that config at **−27%
> throughput** versus `batch 4 / grad_accum 8` at the same effective batch. It is still the right
> choice when the memory isn't there — but it is a real cost, not a free substitution, and a
> machine that trips it will train materially slower for reasons the log does not explain.

**`finetune_edge_conversational.py`'s `grad_accum` was halved alongside its batch** so the
effective batch stays 16. The speedup comes from micro-batch size; changing the effective batch
would have been a training-dynamics change smuggled in behind a throughput fix.

**The compile gain is loop-specific — do not port the number.** Measured separately in each
loop, at the same model and card:

| loop | compile gain | noise floor |
|---|---:|---:|
| `continue_ft_hybrid` (grounded-QA) | **+39%** | 1.0–1.5% |
| `finetune_edge_conversational` (OASST + replay) | **+25.5%** | 0.31% |

Both real and far outside their floors; the lever simply pays less on chat data (whose rows pack
at 1.45 examples/row against grounded-QA's 5.18). Inheriting the first number would have
overstated the second by ~1.5×.

**`continue_ft_looped.py` is uncompiled on purpose.** `looped_forward_train()` bypasses
`model.forward()` — the path both measurements above exercise — so compiling the top-level model
would compile something the script never invokes. Reconsidered under #142 and left alone.

**Two hazards, both checked before the defaults shipped:**

- `torch.compile` prefixes `state_dict` keys with `_orig_mod.`, which would silently produce
  **unloadable checkpoints** from every run using the new defaults. `_save_edge_checkpoint`
  already unwraps it.
- Compile is applied to the **training path only** in `finetune_edge_conversational.py`. Its
  `sample_reply` generates token-by-token with a growing sequence, so a compiled module there
  would recompile *per token* — slower than eager.

**Both FT scripts now print peak reserved VRAM and flag a spill on exit**, so #141's silent-spill
failure (a config reading "fits" on allocated while reserving past the card and running 40%
slower) cannot pass unnoticed in a real run.

**`--n-heads` for pretrain — now graded (#144).** #112 measured `8 → 16` at **2.4× at 355M** and
#141 at **1.12–1.17× at 83M**, so the speed win is size-dependent. #144 measured the *quality*
side that had never been checked. At 83M, 1,500 from-scratch steps, identical seed and identical
batch sequence: held-out probe loss differs by **+0.00535 nats (0.08%)** while bits/byte on unseen
FineWeb-Edu differs by **−0.0009 (0.04%) in the opposite direction** — two held-out sets
disagreeing on the *sign* of a ~0.1% effect, which is the signature of **no detectable quality
cost**. A twin arm (h8 run twice) came out bit-identical, so the run-to-run floor is exactly zero.

**Safe to adopt for a new pretrain on this evidence.** It changes the architecture
(`read_norm_weight` is `[n_heads, d_head]`) so it cannot be applied to an existing checkpoint, and
the evidence covers only **early** training at 83M — 3.07M tokens, 0.19% of a real pretrain, one
seed. It does **not** clear h16 at 355M. Full limits in [state.md](state.md).

### Muon is the default optimizer (Issue #129) — confirmed with ternary ON

#86 measured Muon beating AdamW but ran `--no-ternary`, and every real run here trains with
ternary ON. #129 repeated the A/B in the production configuration: ternary on, `--fast`,
identical seed (init + data order), d=512/12L/8H, batch 4 x seq 512, 1,000 from-scratch steps,
only `--optimizer` differing.

| step | AdamW | Muon | Muon's lead |
|---:|---:|---:|---:|
| 100 | 8.8440 | 9.2837 | −0.44 |
| 200 | 7.5055 | **7.1158** | +0.39 |
| 300 | 7.0104 | **6.5695** | +0.44 |
| 500 | 6.5427 | **6.1374** | +0.41 |
| 700 | 6.3065 | **5.9200** | +0.39 |
| 1000 | 6.0969 | **5.7021** | **+0.395** |

**Steps to reach AdamW's own 1,000-step val loss (6.0969): AdamW 1,000, Muon ~570 — 1.75x
fewer.** Muon also passes 6.0, which AdamW never reaches in the whole run. Muon starts *behind*
(step 100) and overtakes by step 200, so a short smoke test can read the wrong way.

**Wall-clock, which #86 did not report: Muon runs at 0.7 steps/s vs AdamW's 0.8 — ~12% slower
per step.** The honest end-to-end win is therefore **~1.54x**, not 1.75x (1,250 s for AdamW's
1,000 steps vs ~814 s for Muon's 570).

**Ternary does not change Muon's relative advantage.** Final gap here (ternary on) is 0.395;
#86's (ternary off) was 0.406. Near-identical — though these are two separate runs, not a
controlled 2x2, so treat it as consistent rather than proven.

**Caveat, stated because it bounds the claim:** 1,000 from-scratch steps is 1% of a real
pretrain. No run in this project has taken Muon past that. Watch early val on a long run;
`--optimizer adamw` is one flag away.

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
## The compute wall — why "fits in VRAM" was never the binding constraint (2026-08-02)

Everything above this section is memory engineering: BCD (#99), grad checkpointing (#100), CPU
offload (#101), the reserved-memory fix (#102), LRU residency (#103), bf16 parking (#113),
prefetch (#115), bounded pinning (#118). It worked — this 6 GB card went from "1B is
impossible" to a **measured ~6B forward+backward at ~1 GiB peak**.

**Then the wall moved, and it is worth writing down exactly where it went.** Decomposing the
measured gap between the two real throughput points on this hardware:

| | measured |
|---|---|
| 142M, fits in VRAM | **9,102 tok/s** → Chinchilla (2.85B tok) in **3.6 days** |
| 6B, offloaded | **116 tok/s** (1024 tok / 8.83 s) → Chinchilla (120B tok) in **~33 years** |

That is a 3,300× gap. It decomposes as:

| factor | multiplier | fixable? |
|---|---:|---|
| 42× more params (FLOPs per token) | 42× | only by changing architecture |
| 42× more tokens (Chinchilla scales D with N) | 42× | only by accepting undertraining |
| **offload / PCIe inefficiency** | ~~**~2×**~~ **see correction** | yes — and it is the SMALLEST term |

> **⚠️ The ~2× offload figure is STALE (corrected 2026-08-27).** It was measured before #115
> (prefetch/transfer overlap, −25%) and #113 (bf16 parking of inactive blocks, −25%), both of
> which landed afterwards and both of which attack exactly this term. Re-measured on this card
> at the real training shape (d=2048/24L, **seq 4096**, batch 1, grad-checkpoint, chunked
> loss), the offload path runs at **1,014 tok/s / 4.04 s per step** at 1.21B — better than the
> 875 tok/s recorded in #157, and comfortably inside the card at 4.38 GB (73%).
>
> **Read the direction of this correction carefully: it makes offload look BETTER, which makes
> the section's conclusion STRONGER, not weaker.** If PCIe overhead is smaller than ~2×, then
> even less of the 3,300× gap is recoverable by memory engineering, and the N² argument below
> holds harder. Do not read "offload is faster than we thought" as "big models are closer than
> we thought".
>
> A practical consequence worth stating separately: **offload is not a tax to be escaped.** It
> is what keeps VRAM free for activations, which are the binding term at seq 4096. #161 tested
> the opposite hypothesis directly — a compressed model held resident on the card, avoiding
> PCIe entirely — and it was 2.6× SLOWER and spilled at 113%. See docs/state.md.

**Making PCIe transfer free would take 33 years to 16 years.** The memory engineering was real
and necessary to *fit* the model, but it was never where the training cost lived. Training cost
is `~6 · params · tokens`, and because Chinchilla scales tokens with params, it grows as **N²**.
No memory technique touches that.

**Why 8.83 s/step specifically.** At 6B the weights do not live on the GPU — every block is
streamed from CPU RAM over PCIe each step, ~12 GB of bf16 weights moved per step, twice
(forward + checkpoint recompute). It is data movement, not arithmetic. The corroborating
evidence is in this doc already: #113 cut step time 1.55 s → 1.16 s purely by **halving the
bytes moved**, which would do nothing if compute were the bottleneck.

**Levers that actually attack N², with measured or estimated multipliers:**

| lever | gain | status |
|---|---:|---|
| MoE / sparse activation | ~3–6× *predicted* | ❌ **measured worse than scaling dense** — see docs/state.md #123 |
| Data quality (FineWeb-Edu, dedup) | 2–5× | ✅ #120 |
| Muon optimizer | 1.6× | ✅ #86 |
| Accept undertraining (< 20 tok/param) | 2–5× | free, costs quality |

With MoE removed (it lost its A/B), the realistic stack is ~5–8×, which puts **~1B trainable
locally in 6–8 months** and leaves **6B out of reach by roughly 30×**.

**Practical ceilings on this hardware, stated plainly:**

| target | local time | verdict |
|---|---|---|
| **142M** | **3.6 days** | ✅ the honest local ceiling |
| 300M | ~129 days (measured, #112) | ❌ |
| 1B | 6–8 months | ❌ |
| 6B | 2.5–33 years | ❌ by ~30× |

**A note for future planning.** "It fits in VRAM" and "it can be trained" are different claims,
and this project spent considerable effort on the first while believing it was progress toward
the second. The 6B-at-1-GiB result is a genuine engineering achievement and simultaneously not
a training capability. Any future size decision should start from `6 · N · D ÷ throughput`, not
from whether the model loads.
