# Benchmark Results — DeltaNet, bf16 Stability & Weightless

> **Moved from README:** §8.6 DeltaNet vs Default Block at Real Scale (Issue #39) · §8.7
> Large-Width bf16 Stability Fix (Issue #44) · §8.8 Hybrid Weightless Projections (Issue
> #40). The corrected-routing final pass that supersedes the default-HVN columns here is in
> [benchmarks_scaling.md](benchmarks_scaling.md).

## 8.6 DeltaNet vs Default Block at Real Scale (Issue #39)

The #34 DeltaNet win was previously only measured on 34K chars. Issue #39 swept it on the
same real-scale config as §8.4 (WikiText-103, BPE-8000, 5M-token compute-matched budget, 2L,
seq_len=128, RTX 4050) via a distinct `HVN-DeltaNet` model in `eval/scaling_laws.py`
(`--models HVN HVN-DeltaNet Transformer`):

| d_model | HVN (default) | HVN-DeltaNet | Transformer | DeltaNet vs HVN |
|--------:|--------------:|-------------:|------------:|:---:|
| 128 | 522.1 | **410.4** | 394.5 | **−21.4%** |
| 256 | 350.4 | **251.9** | 248.4 | **−28.1%** |
| 512 | 237.0 | **218.9** (fp32) / NaN (bf16) | 173.1 | −7.6% (fp32) |
| 768 | 206.1 | NaN (bf16) | 150.1 | — |

**DeltaNet is better on quality at every width** — cuts PPL 21–28% at d≤256 (essentially
matching the Transformer) and, in fp32, reaches 218.9 at d=512. **But under bf16** — the
project's recommended training precision — **it diverges to NaN at d≥512** (d=512 spiked
after 2M tokens at PPL 207; d=768 after 1M at 244, i.e. while *winning*). A fp32 rerun of
d=512 trains stably to 218.9, confirming the failure is a **large-width bf16
numerical-stability / loss-spike issue, not a quality deficit** — existing grad clipping was
insufficient at that width.

**Decision (at the time) — keep flag-gated, do not flip the default.** Acceptance criterion
#35 ("DeltaNet ≥ HVN block at real scale") held on *quality* but failed on *robustness*: a
default must train cleanly in the recommended precision. Flipping the default was blocked
on a follow-up (Issue #44). **That fix is now done — see §8.7 below.**

## 8.7 Large-Width bf16 Stability Fix for Gated-DeltaNet (Issue #44)

The §8.6 divergence was diagnosed to two numerical hot-spots in the bf16 path, both fixed
in `hvn/core/deltanet.py` (matmuls stay bf16; only the sensitive reductions move to fp32):

1. **fp32 islands** for the numerically sensitive ops — the L2-normalization of q/k
   (`_l2norm`), the per-head output RMSNorm on the read (`_read_normalize`, which squares a
   large read `S·q` — bf16 overflows above ≈1.8e19 after `pow(2)`), and the data-dependent
   decay/write gates `α`/`β`.
2. **Flooring the intra-chunk cumulative decay `P = cumprod(α)`** in the chunked-parallel
   scan. The algebra only uses the bounded causal ratios `P_i/P_j ≤ 1` (j≤i), but the
   intermediate `ktil = k/P` overflows fp32 to Inf once `P` underflows — at d≥512 enough decay
   channels shrink within a 32-token chunk that `(1e-6)^32` underflows. `P` is floored at
   `1e-16` (keys/queries are unit-norm, so `1/P ≤ 1e16`, well inside fp32), only perturbing
   channels already decayed to ≈0.
3. **Loss-spike guard** in the training loop (`eval/scaling_laws.py`): skip the optimizer
   step on a non-finite loss or a grad-norm spike (>1e3) so a transient spike can't poison
   the weights and AdamW moments.

**Result — DeltaNet now trains cleanly in bf16 at every width, with no NaN, and its quality
win *grows* with scale** (the instability had been silently degrading the large-width runs
even before the NaN). Full sweep, same config as §8.6 (WikiText-103, BPE-8000, 5M-token
compute-matched, 2L, seq_len=128, **bf16**, RTX 4050):

> **Superseded for the HVN-default column — see §8.12.** This table was produced under the
> pre-#46 buggy head routing. The corrected-routing final pass lowers the default-HVN
> numbers 6–12% (e.g. d=128 522.5 → 469.1); the DeltaNet column is unaffected (it doesn't use
> the buggy modules) and reproduces here within run variance.

| d_model | HVN (default) | HVN-DeltaNet | Transformer | DeltaNet vs HVN |
|--------:|--------------:|-------------:|------------:|:---:|
| 128 | 522.5 | **410.6** | 392.6 | **−21.4%** |
| 256 | 348.4 | **251.9** | 245.1 | **−27.7%** |
| 512 | 236.0 | **159.4** | 173.9 | **−32.4%** |
| 768 | 205.9 | **128.0** | 149.7 | **−37.8%** |

Refit exponents (bf16): HVN −0.423, **HVN-DeltaNet −0.507** (steepest of the three),
Transformer −0.404. DeltaNet now **beats both HVN and the Transformer at d=512 and d=768**
and has the steepest scaling exponent. All acceptance criteria met: no NaN at any width; bf16
d=512/768 PPL ≤ default HVN; O(1) preserved. `--use_deltanet` is now safe to run in bf16 at
all widths, which **unblocks the #39 default-flip decision** (left to a follow-up so this
change keeps defaults unchanged).

## 8.8 Hybrid Weightless Projections for Gated-DeltaNet (Issue #40)

*"Instead of using weights, use something else to keep file size / memory low — we already
won the no-qkv battle."* The Hamiltonian H is **already** weightless (runtime state, never
trained/stored). #40 targets the remaining trained tensors. A **fully** weightless net is
not viable — the #34 DeltaNet win came *specifically* from the **learned, data-dependent
dynamics** (`α_t=σ(W_α x)`, `β_t=σ(W_β x)`, output gate, causal conv), so freezing those
hands most of it back. So the shipped variant (`--weightless`, requires `--use_deltanet`,
`hvn/core/weightless.py`) is a **hybrid**: freeze the four big `d×d` "address/content"
projections (`q/k/v/out`) as **PRNG-seed-regenerated random features** — each stored as
*one integer*, not `d²` floats (`SeededLinear`, a non-persistent buffer regenerated from the
seed on load) — while keeping the tiny high-leverage learned core (`α/β/g` gates, conv, read
RMSNorm).

A/B on the bundled tiny-Shakespeare sample (the #34 scale — d=128, 2L, 20 epochs, bf16, RTX
4050), dense vs weightless DeltaNet, identical config/seed:

| Variant | Params | Eval PPL | Checkpoint |
|---------|-------:|---------:|-----------:|
| DeltaNet (dense) | 344,633 | 1.087 | 1,359 KB |
| **DeltaNet (weightless)** | **213,561 (−38%)** | **1.220 (+12.3%)** | **845 KB (−37.8%)** |

Freezing the four projections removes `4·d²·L` floats (here 131,072), cutting
params/checkpoint ~38% for a **+12.3% relative PPL** cost — the model stays reliable (both
memorize the sample near perfectly). O(1) inference, no-KV, fixed d×d state and
forward↔incremental parity are all preserved; checkpoints round-trip **exactly** by
regenerating the frozen matrices from the stored seed. `test_issue40.py` 6/6. *Scope:*
flag-gated, defaults unchanged; PPL is the memorization regime, so read the −38% size / +12%
PPL trade as the small-scale signal. At larger `d` the `4d²`-per-layer projection share grows
relative to the `vocab·d` embedding, so the size win grows.

**Real-scale cost — measured in the §8.12 final pass** (corrected routing; see
[benchmarks_scaling.md](benchmarks_scaling.md)): the hybrid weightless DeltaNet vs dense
DeltaNet:

| d_model | Dense PPL | Weightless PPL | PPL cost | Param saving |
|--------:|----------:|---------------:|:--------:|:------------:|
| 128 | 412.3 | 455.2 | +10.4% | −5.5% |
| 256 | 251.1 | 275.2 | +9.6% | −9.7% |
| 512 | 159.4 | 183.3 | +15.0% | −15.6% |
| 768 | 129.4 | 157.0 | +21.3% | −19.6% |

This **confirms the small-scale caveat quantitatively**: the −38% param saving at d=128 on
tiny-Shakespeare was a small-vocab artifact (char vocab ≈ 65, so the `vocab·d` embedding was
negligible and the frozen `4·d²·L` projections dominated). At BPE-8000 the embedding is large,
so at small `d` the saving is modest (−5.5%) and **grows with width** (−19.6% at d=768) as
the `4d²L` projection share overtakes the `vocab·d` embedding — exactly as predicted.
Weightless still beats the default HVN block at every width and keeps a near-DeltaNet
exponent (−0.499), so it remains the right pick when checkpoint size dominates; the PPL cost
is a modest, width-growing 10–21%.