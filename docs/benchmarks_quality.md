# Benchmark Results — Quality & Reasoning

> **Moved from README:** §8.2 Quality (Shakespeare) · §8.5 Honest Assessment · §8.11 Real
> Reasoning / `[THINK]`. Speed in [benchmarks_speed.md](benchmarks_speed.md); scaling in
> [benchmarks_scaling.md](benchmarks_scaling.md).

## 8.2 Quality (Shakespeare)

**d_model=64, 1 layer, 10 epochs:**

| Model | Params | Eval PPL | Train Time |
|-------|--------|----------|------------|
| HVN (old, single-head) | 30,056 | 9.48 | 236.1s |
| HVN (new, multi-head) | 34,225 | 8.98 | 172.5s |
| Transformer | 54,592 | 4.29 | 4.0s |
| SSM | 42,176 | 1.65 | 35.4s |

**d_model=128, 2 layers, 20 epochs:**

| Model | Params | Eval PPL | Train Time |
|-------|--------|----------|------------|
| HVN (multi-head) | 241,725 | 8.37 | 467.8s |
| Transformer | 405,248 | 1.90 | 3.4s |
| SSM | 306,432 | 1.12 | 68.6s |

## 8.5 Honest Assessment

**Where HVN wins:** memory efficiency (O(1) vs O(N)), parameter efficiency (43% fewer),
theoretical O(1) per-token scaling (flat N=256→4096), incremental generation, kernels close
the raw-speed gap (2–7×). **Where HVN loses:** raw speed (Flash Attention faster in absolute
terms), and the quality gap — addressed by Issue #34 (see [benchmarks_deltanet.md](benchmarks_deltanet.md)).

### The Quality Gap Summary

| Model | PPL (Shakespeare, d=128, 20ep) | Scaling exponent |
|-------|-------------------------------|------------------|
| SSM | ~1.16 | N/A |
| Transformer | ~2.0 | -0.090 |
| **HVN + Gated-DeltaNet (#34)** | **~3.4** (61% better than 8.8) | **−0.507** real-scale, −21–38% PPL vs HVN, beats Transformer at d≥512 (#39/#44; bf16-stable at all widths) |
| HVN (#22 n_threads=4 + all strategies) | ~8.13 | -0.0123 |
| HVN (#21 strategies) | ~8.37 | -0.0126 |
| HVN (original) | ~8.37 | -0.007 |

> **Exponent caveat (#35):** every "Scaling exponent" above was fit on **char-level
> Tiny-Shakespeare**, small enough that all model sizes saturate — so these measure
> saturation noise, **not** architectural scaling. On real WikiText-103 (BPE,
> compute-matched, unsaturated) HVN scales at **−0.42** ≈ Transformer −0.41 (see
> [benchmarks_scaling.md](benchmarks_scaling.md)). Read the flat char-Shakespeare exponents
> as a small-data artifact, not a property of HVN.

**Root cause (superseded → confirmed):** previously attributed to a "fundamental
compression bottleneck" of the d×d matrix. Now known **wrong** — the SSM reaches PPL ~1.1
with a *smaller* fixed state. The real culprit is the readout/update dynamics (velocity
L2-norm + Frobenius clamp discarding magnitude; non-standard symmetric HPH update),
validated by #34.

## 8.11 Real Reasoning / Thinking — Trainable `[THINK]` Chain-of-Thought (Issue #42)

Before #42 the `[THINK]`/`[/THINK]` special tokens (`hvn/tools.py`) were **only delimiter
parsing** — the model was never trained to put meaningful content inside them, so "thinking"
was a format convention, not a capability. #42 makes the blocks carry **real chain-of-thought
that measurably helps**, as a pure **data + training + decoding-convention** feature — no
architectural change, O(1) / no-KV inference untouched.

- **Data (`hvn/reasoning.py`).** A synthetic generator for **multi-step arithmetic**
  (chained add/sub, mul-then-add, order-of-operations, two-step word problems) —
  deterministic ground truth, so answers auto-grade. Each example pairs a question with an
  explicit step-by-step trace: `Answer: [THINK]\n6 * 2 = 12\n12 + 4 = 16\n[/THINK] The answer
  is 16.`
- **Training (`train.py --reasoning_training`).** Mirrors the tool-calling (#24) data path.
- **Decoding (`generate.py --hide_thinking`).** Strips `[THINK]` from the user-facing
  answer (via `ToolCallParser.strip_tool_blocks`) while logging the raw trace to stderr.
- **Eval (`eval/reasoning.py`).** Auto-graded **with-think vs. no-think** A/B on a held-out
  set: *think* lets the model reason before answering; *nothink* primes the answer prefix so
  it must commit immediately (same model, same problems).

**Result — reasoning measurably helps.** Char-level, `--use_deltanet`, d=256, 4L, 6,000
reasoning examples, 80 epochs, RTX 4050; 200 held-out problems, greedy decode:

| Mode | Exact-match accuracy |
|------|:---:|
| **think** (may emit `[THINK]`) | **53.5%** |
| nothink (must answer immediately) | 1.5% |
| **delta** | **+52.0pp** |

Externalising intermediate results as tokens turns the final answer into a *copy* of a number
the model just computed one step at a time, whereas the no-think path must compute the whole
expression in a single forward step and essentially cannot. Example (greedy):
`Question: 6 * 2 + 4` → `[THINK] 6 * 2 = 12  12 + 4 = 16 [/THINK] The answer is 16.` ✓
`test_issue42.py` 5/5.

**Also fixed — a latent tokenizer bug.** The char tokenizer was built over the *base* corpus
**before** the tool/reasoning/instruction data was appended, so any character absent from the
base text (notably **every digit**, and `*`/`=`) was OOV and **silently dropped at encode
time** — the reasoning examples trained with all numbers stripped. `train.py` now generates
the auxiliary data first and builds the vocab over the **full** corpus (base + tool +
reasoning + instruction), which also repairs the tool-calling (#24) and instruction-tuning
(#29) data paths.

*Scope / caveats:* small-scale, memorization-regime signal (single-digit operands; the
generalization gap on harder word problems is visible — think accuracy is 53%, not 100%).
Read the **+52pp delta** as the headline, not the absolute number; a real-scale re-measure
belongs in the #45 final pass.