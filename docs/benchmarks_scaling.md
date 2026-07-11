# Benchmark Results — Real-Scale Scaling & Corrected Routing

> **Moved from README:** §8.4 Real-Scale Scaling Re-measurement (Issue #35) · §8.10
> Systemic Batch-Ordering Fix (Issue #46) · §8.12 Final Real-Scale Benchmark Pass (#45/#47).
> The DeltaNet/bf16/weightless sub-story is in [benchmarks_deltanet.md](benchmarks_deltanet.md).

## 8.4 Real-Scale Scaling Re-measurement (Issue #35)

Rebuilt scaling harness (`eval/scaling_laws.py`): **compute-matched** (fixed per-model token
budget, not epochs), **full held-out eval PPL** (whole valid split in `seq_len` chunks), BPE
+ SSM added, per-model saturation tracking, streaming WikiText-103 loader.

**Headline run — WikiText-103, BPE (vocab 8,000), 5M-token compute-matched budget** (12.4M
unique train tokens, so no repeats — models are **not** data-starved), d={128,256,512,768},
2L, seq_len=128, bf16, RTX 4050:

> **HVN column superseded by §8.12** (corrected #46 routing lowers it 6–12%: −0.397
> exponent); Transformer/SSM are unaffected by #46 and stand. This table is retained as the
> #35 record.

| Model | d=128 | d=256 | d=512 | d=768 | Params (d=768) | **Exponent** (PPL ∝ N^b) |
|-------|------:|------:|------:|------:|---------------:|:---:|
| **HVN** (default) | 521.1 | 351.9 | 236.1 | 205.8 | 21.75M | **−0.4237** |
| Transformer | 395.6 | 246.0 | 172.9 | 150.3 | 26.46M | −0.4066 |
| SSM | 366.8 | 225.2 | 138.5 | 110.6 | 22.92M | −0.5311 |

All three models are **still improving at 5M tokens** (unsaturated — the correct regime for
a scaling law). Absolute PPL is high because 5M tokens is a small budget for BPE-8000; the
**exponent**, not the absolute PPL, is the quantity of interest.

**Result — the flat exponent was a small-data artifact.** On real-scale data HVN scales at
**−0.42**, essentially matching Transformer (**−0.41**) and near SSM (−0.53) — a ~100×
steeper exponent than the original **−0.007**. HVN's PPL drops 521 → 206 (2.5×) across a 9×
param increase. The "HVN doesn't scale" narrative is **refuted**.

**Small-data control — full Tiny-Shakespeare, char-level, 2M-token budget** (32× the 34K the
−0.007 was fit on): HVN −0.0042, Transformer −0.0252, SSM −0.1060; HVN saturates at ~1M tokens
(all sizes). This *reproduces* the flat exponent — and shows why it is an artifact: at 1M
chars every size saturates early (data-bound), so the exponent measures saturation noise,
not architectural scaling. Move to a corpus where models aren't starved and the flatness
vanishes.

*Caveats:* default HVN only (the `--use_deltanet` #34 sweep is in
[benchmarks_deltanet.md](benchmarks_deltanet.md) — Issue #39); 5M-token reduced budget, not
the full ~100M — absolute PPLs will drop substantially at a larger budget, but the exponent
comparison is already in the trustworthy (unsaturated) regime. Scale up via:
```bash
python -m eval.scaling_laws --dataset wikitext103 --local-path wikitext-103-v1 \
    --tokenizer bpe --bpe_vocab 8000 --token_budget 100_000_000 \
    --d_models 128,256,512,768 --precision bf16 --compile \
    --output-json results/scaling_wt103.json --output-plot results/scaling_wt103.png
```

## 8.10 Systemic Batch-Ordering Fix — Batched `forward` == Single-Stream (Issue #46)

The head-misrouting bug that #43's served path worked around was **not** confined to
`generate_batched` — the same shared batched modules (`batched_vortex.py`,
`batched_memory.py`) back **training `forward`, batched eval PPL, and `parallel_forward`**,
so every batched path with `n_heads>1, B>1` was affected. The block lays the batch out
**batch-outer** (`bh = b·n_heads + h`), but the per-head params were expanded with
`repeat_interleave(B)` — **head-outer** ordering (`h·B + b`) — pairing each slot's data-head
with the *wrong* weight-head. The bug was self-consistent within a fixed batch size (so
training still converged), but it entangled learned per-head routing with the training batch
size and diluted multi-head specialization.

The fix **tiles** per-head params batch-outer (`repeat` instead of `repeat_interleave`) in
`MultiHead`/`MultiThread` vortex + memory. One source change corrects `forward`, eval,
`generate_batched`, and `parallel_forward` together.

**Result — the issue's own evidence snippet, `max|batched − per-sequence|`:**

| n_heads | Before (buggy) | After (#46) |
|--------:|---------------:|------------:|
| 1 | 6.0e-07 | 6.0e-07 |
| 4 | **2.42** | **6.0e-07** |
| 8 | (diverges) | 6.0e-07 |

Batched `forward` now equals single-stream at every width, across all flag-gated variants
and for `parallel_forward` and `generate_batched`. `B=1` is bit-unchanged.
`test_issue46.py` 5/5. **Caveat:** semantics-breaking — checkpoints trained under the old
routing and the pre-#46 §8.x benchmark numbers were produced under the buggy routing; the
re-measurement is tracked in #47 (folded into the final pass #45, see §8.12 below).

## 8.12 Final Real-Scale Benchmark Pass — Corrected Head Routing (Issues #45 / #47)

The single end-state benchmark run, executed once now that all feature work is settled. It
**supersedes** the default-HVN columns of §8.4/§8.6/§8.7 and the tiny-Shakespeare weightless
A/B of §8.8: every number here is produced under the **corrected #46 batch-outer head
routing**. SSM is omitted (a reference competitor, unaffected by #46 — its §8.4 numbers
stand); the comparison of interest is the three HVN variants vs the Transformer baseline.

**Config** (identical harness to §8.4/§8.7): WikiText-103, BPE-8000, 5M-token compute-matched
budget (12.4M unique train tokens — unsaturated), d ∈ {128,256,512,768}, 2L, seq_len=128,
**bf16**, RTX 4050 (`eval/scaling_laws.py --models HVN HVN-DeltaNet HVN-DeltaNet-Weightless
Transformer`, `results/scaling_final_pass.{json,png}`).

| d_model | HVN (default) | HVN-DeltaNet | HVN-DeltaNet-Weightless | Transformer |
|--------:|--------------:|-------------:|------------------------:|------------:|
| 128 | 469.1 | **412.3** | 455.2 | 395.6 |
| 256 | 307.2 | **251.1** | 275.2 | 245.6 |
| 512 | 218.8 | **159.4** | 183.3 | 172.0 |
| 768 | 194.3 | **129.4** | 157.0 | 150.8 |

**Refit exponents (PPL ∝ params^b):** HVN **−0.397**, HVN-DeltaNet **−0.504** (steepest),
HVN-DeltaNet-Weightless **−0.499**, Transformer **−0.406**.

**Finding 1 — correct head routing measurably *improves* the default HVN block (#47's A/B).**
The #46 fix (each batched slot now reads its own head's weights) lifts the default block at
every width vs the buggy-routing §8.7 numbers:

| d_model | HVN pre-#46 (§8.7, buggy) | HVN post-#46 (this pass) | Δ |
|--------:|--------------------------:|-------------------------:|:---:|
| 128 | 522.5 | 469.1 | **−10.2%** |
| 256 | 348.4 | 307.2 | **−11.8%** |
| 512 | 236.0 | 218.8 | −7.3% |
| 768 | 205.9 | 194.3 | −5.7% |

So #46 was **also a quality fix** — head-misrouting had been silently degrading the default
block by 6–12% PPL. **DeltaNet is unaffected** (within run variance of §8.7) because
`DeltaNetBlock` replaces the block entirely and never touches the buggy modules — confirming
#46 was localized to the default HVN path.

**Finding 2 — DeltaNet remains the quality leader and is the recommended default (feeds
#39).** It beats the default HVN block at every width (−12.1% → −33.4% PPL as d grows), has
the steepest exponent (−0.504 vs Transformer −0.406), and **beats the Transformer at d ≥
512**. Combined with #44 (bf16-stable at all widths) the last blocker on the #39 default-flip
is cleared: **`--use_deltanet` should become the documented default block.**

**Finding 3 — weightless (#40) real-scale cost, measured for the first time beyond
d=128/tiny.** See [benchmarks_deltanet.md](benchmarks_deltanet.md) §8.8 for the table; it
confirms the §8.8 caveat quantitatively (the −38% param saving was a small-vocab artifact;
at BPE-8000 the saving is modest at small d and grows with width to −19.6% at d=768).

O(1) per-token inference, no-KV, and forward↔incremental parity are unchanged by this pass
(training/measurement only; `test_issue46.py` guards the routing fix). **Deliverables:**
`results/scaling_final_pass.json` + `results/scaling_final_pass.png`.