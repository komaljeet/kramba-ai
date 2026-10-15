# kramba-ai — training language models on one 6 GB GPU

[![DOI]https://zenodo.org](https://doi.org/10.5281/zenodo.22725430)

A training and evaluation stack for linear-attention language models that runs on a single
consumer graphics card. The headline result, measured rather than estimated: **a 1.11-billion
parameter model trains in 4.51 GB**, and ~1.6B is the ceiling on the card.

That's the whole project. The rest of this page is about what that does and doesn't mean.

**Correction (2026-10-11):** the CPU weight-offload path used by the 1.11B run below had a bug
from 2026-07-26 to 2026-10-10 that could swap weights between same-shaped matrices during
training (fixed in commit `2f94d7a`, confirmed present as far back as the offload code itself,
independently reproduced on the original historical code). **The VRAM/fits-in-4.51GB number is
unaffected** — that bug was about weight correctness, not memory usage. **The loss curve from
that specific run is not** — it needs re-running on fixed code before the "it trains" claim
can be trusted again. See [`docs/state.md`](docs/state.md) for the full writeup.

**Update (2026-10-14):** the offload+BCD path that correction refers to has since been
validated end-to-end — a real 300-step run hard-killed and resumed mid-block three times,
plus a real 1.1B fresh-build and resume, checked against a plain and a BCD-only reference.
Along the way four more real bugs were found and fixed (a resume crash whenever the saved
active block wasn't block 0, BCD's bf16 parking silently freezing bias/norm params for the
whole run, ternary conversion doubling host RAM and crashing 1B builds outright, and an
error-swallowing bug that made argument mistakes look like crashes) — all merged, all
tested together, no bad interaction found. Separately, four more real step-time levers
landed this week, cutting the measured 1B step from 21.81 to **15.067 s/step** (-30.9%, a
1.45× speedup), stacked and re-measured together, not just added up on paper. **The
"reality check" section below is now corrected for these numbers — see
[`docs/state.md`](docs/state.md) for the full measured trail.**

Full documentation is in [`docs/`](docs/index.md). If you're picking this up cold, start with
[`docs/handoff.md`](docs/handoff.md).

## About the name

The repo, the `hvn/` package and `--arch hvn-hybrid` are named after the **Hamiltonian Vortex
Network** — the original block this project started from. That block was replaced in July 2026
and, when retested properly in August, it **lost**.

The name stayed because it's wired through the code. It doesn't describe the model in use, and
it isn't what the project is about.

## What's actually in the box

The default model is **10 gated-DeltaNet blocks plus 2 sliding-window attention layers**.

**This architecture is not original, and the code says so in its own docstring.** It's a
reimplementation of published work: a canonical gated DeltaNet update, a GLA/Mamba-2 decay
gate, and an output gate and causal convolution from Mamba/RWKV-7. It sits in the same family
as other published hybrids.

Constant-memory inference — no KV cache, fixed state — is the defining property of that whole
family (Mamba, RWKV, RetNet), not something this project invented. No comparison against Mamba
or RWKV has ever been run here.

**The original block was genuinely original, and it lost.** It framed tokens as fluid vortex
particles read against a fixed "entanglement Hamiltonian" (the maths is in
[`docs/theory.md`](docs/theory.md)). It's still in the tree as `--arch hvn-original-hybrid`,
kept as history. Retested at matched parameters, same seed, same data, it lost to gated
DeltaNet by 8–20× the noise floor. Novel in *framing*, not in *kind*.

**What's left is the engineering**, and that's the real contribution: block-coordinate descent,
CPU weight offload, ternary quantisation, tied embeddings, chunked cross-entropy, a hand-written
Triton scan and megakernel, and the fixes that make them work together.

## What's measured

| claim | status |
|---|---|
| 1.11B-parameter model **trains in 4.51 GB**, 1,579 tok/s | measured 2026-09-19 |
| **~1.6B is the size ceiling** (2.45B spills to system RAM) | measured 2026-08-17 |
| the stack is **not architecture-specific** — a plain 1.07B transformer also trains in 4.04 GB | measured 2026-08-17 |
| flat ms/token from 256 → 4096 tokens of context | measured, [benchmarks_speed.md](docs/benchmarks_speed.md) |
| beats a matched vanilla transformer at equal compute, 12M body | measured 2026-08-16, single seed |
| needs **~30 tokens per parameter**, not the usual 20 | measured 2026-08-16, size trend unresolved |
| chunked cross-entropy saves **1436 MB at 48M** | measured 2026-08-17 |
| **not cheaper to train** — a transformer is 1.41× faster per step | measured; the win is at inference only |
| the 1B step runs at **close to hardware limits** — matmuls hit 100.6% of the card's own measured GEMM ceiling | measured 2026-08-31 |
| `--n-heads 64` at 1B is **13% faster** than 32 (~430 → ~375 days) | measured 2026-08-31 — **stale**: current measured step is 15.067 s/step (~161-258 days depending on BCD quality framing, see the reality check below), from four more real speed levers landed 2026-10-12/14 |
| **1.11B fits and runs in 4.51 GB with 5-technique stack** (BCD + offload + ternary + grad checkpoint + chunked loss) | VRAM measured 2026-09-19, 1350 steps — **but see the correction above: this run used an offload path with a weight-corruption bug (fixed 2026-10-10), so the loss curve from this run needs re-verifying, the VRAM figure does not** |
| **tied embeddings + offload + BCD + chunked_loss fit without OOM** (previously OOM, fixed by pinning + chunked loss) | measured 2026-09-26, 79% VRAM (4.85 GB of 6 GB) at 1.11B fp32 — "composes" here means *fits*; whether it trains *correctly* at this combination and scale was never checked before 2026-10-10, and is only confirmed at small scale since |

One row deliberately removed: earlier versions led with "18.7% of attention's FLOPs" and
"6.29 MB vs 2.46 GB of memory". Both were **hand-derived, never benchmarked**, and both
described the *original* block — the one that lost. The analysis still lives in
[`docs/theory.md`](docs/theory.md), attached to the block it actually describes.

## What this is aiming at

Frontier-grade capability on a 6 GB laptop, eventually a phone. Two halves:

- **Fit the weights** — ternary (BitNet b1.58) quantisation puts a 20B-class model in ~4 GB
- **Keep memory flat as conversations grow** — recurrence plus windowed attention, so there's
  no cache that expands with context

The plan is to train such a model from scratch rather than distil an existing one. BitNet
showed ternary-from-scratch matches full precision at scale, so that's a real recipe, not a
compromise.

### The reality check, and it's a hard one

Training cost is roughly `6 × parameters × tokens`, and the tokens you need scale with the
parameters — so cost grows with the **square** of model size.

The memory engineering got a 6B model to run forward and backward on this card at
~1 GB peak. **That is not the same as being able to train it.** 6B is still out of reach here.
1B, with `--block-coordinate --offload`, is real but slow: at the current measured
**15.067 s/step** (2,175 tok/s), hitting this project's 30-tokens-per-body-param convention
takes **~161 days** accepting BCD's own ~1.6× quality cost as measured, or **~258 days**
if you want BCD's output to actually match full-training quality instead (that's the
multiplier's actual direction — the shorter number is the one with the quality cost baked
in, not the other way around). This was **375 days at a smaller, un-fixed config as of
2026-08-31**; four real speed levers and a from-scratch BCD-quality fix (balanced block
partitioning, cutting BCD's quality cost from ~4× to ~1.6×) landed since, measured together,
not assumed additive. See [`docs/state.md`](docs/state.md) for the full trail.

Everything smaller than 1B fits and trains *without* BCD/offload at all, measured directly
on current code: **270M in ~23 days, 300M in ~32 days** (both plain, no memory tricks). A
synthetic benchmark without the real loss-head optimization will make all three look like
they spill past VRAM — that's a measurement bug, not a real result; it's what a real-trainer
run with the actual chunked loss path shows.

**This is not a wall anymore, but it is still a real cost.** The roofline work (2026-08-31)
that put matmuls at 100.6% of the card's own GEMM ceiling and the scan capped at 1.16× by
Amdahl still holds broadly — GEMM's own share of the step has since grown to 43% as other
costs shrank. The remaining real levers found this week (fusing more of the DeltaNet glue
into the scan kernels, the short causal conv, a few percent of quantize overhead) added up
to real, measured wins, not the "nothing left" verdict this file used to state — don't
trust an "exhausted" claim here without a fresh number behind it, this project has been
wrong about that more than once.

So: the *inference* half of the goal is real and measured. The *train-it-yourself* half is
real up to ~1B on this hardware, at real but multi-month cost — not the "isn't achievable"
this section used to say, and not free either.

(A note for anyone reading the compute-wall analysis in [`docs/training.md`](docs/training.md):
its "~2× offload inefficiency" term is stale and the real figure is *smaller*, which makes that
conclusion stronger, not weaker.)

## Quick start

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install numpy

python demos/demo.py                                    # check it works
python train.py --epochs 50 --generate --prompt "The "
python generate.py --prompt "The " --tokens 200 --temp 0.7
```

More detail: [`docs/setup.md`](docs/setup.md), [`docs/usage.md`](docs/usage.md),
[`docs/training.md`](docs/training.md).

## Traps that have cost real hours

**`torch.compile` is 2.4× SLOWER here.** Four separate investigations have confirmed it. Use
`--fast`, which deliberately skips it.

**Bare `--no-fast` silently drops you to fp32** — 74.6 s/step instead of 8.66. Re-add the
precision flags by hand.

**The default data mix caps a pretrain at 0.13 billion tokens.** OpenAssistant is only 15.7M
tokens but takes a 35% share, which binds the whole budget — enough for a ~7M model. A 1B run
on defaults gets **270× less data than it needs**, and the loss curve looks perfectly healthy
while it re-reads the same tokens. Pass `--pretrain-mix 0.0,0.0,0.0,1.0` to use FineWeb-Edu
instead. Tracked in [issue #162](https://gitlab.com/komalbarun/kramba-ai/-/issues/162).

**Nothing prints until step 50.** A short run looks hung when it's fine. Watch
`training_log.csv`, which flushes every row.

## Documentation

| File | Covers |
|------|--------|
| [handoff.md](docs/handoff.md) | **Read first when picking up cold** — what's running, what's settled |
| [state.md](docs/state.md) | Every measured result, dated, including the retractions |
| [index.md](docs/index.md) | Overview, what is and isn't original, doc map |
| [training.md](docs/training.md) | Flags, VRAM envelopes, precision, the compute wall |
| [architecture.md](docs/architecture.md) | Components, project structure, the hybrid recall design |
| [theory.md](docs/theory.md) | Maths of the original block, FLOP analysis |
| [setup.md](docs/setup.md) · [usage.md](docs/usage.md) | Install and runtime features |
| [vscode.md](docs/vscode.md) | Running it as a local OpenAI-compatible server |
| [benchmarks_*.md](docs/index.md) | Speed, quality, scaling, DeltaNet comparisons |
| [features.md](docs/features.md) · [issues.md](docs/issues.md) · [roadmap.md](docs/roadmap.md) | History |
| [ai_rules.md](docs/ai_rules.md) | Workflow rules for contributors |

## Dead ends, recorded on purpose

Negative results sit in [`docs/state.md`](docs/state.md) with the same weight as the wins,
including results this project later **retracted** after finding the measurement itself was
wrong. The big ones:

- **Looped and weight-shared models lose** (#122). Every variant tried: looping a pretrained
  stack, loop-aware fine-tuning, a gated loop, and a small stack trained from scratch to run
  4×. Three layers × 4 loses to twelve layers by 11× the noise floor.
- **Tensor-train factorisation loses** (#161). Half the parameters, but 22% worse — and at
  matched parameters a plain dense model is 1.85× faster in less memory. Code deleted.
- **`torch.compile` is a net loss** — 2.4× slower, measured four times.
- **Mixture-of-experts loses to simply scaling the dense model** (#123).
- **Predicting future weights is worse than doing nothing** (GHOST, #126).
- **Selective token training gives no gain at 83M**, and the "phase transition" it was built on
  doesn't exist in the data.

## Requirements

GPU only — RTX 4050, CUDA 12.6. There's no CPU fallback path; if CUDA isn't available, fix the
install rather than running on CPU. See [`docs/ai_rules.md`](docs/ai_rules.md).

GitLab: https://gitlab.com/komalbarun/kramba-ai

## An open question, if you have a spare GPU

One thing nobody here has checked: **is the batch size right, or just whatever fit in memory?**

`hardware_scan()` picks it to fit VRAM and it became a constant nobody revisited. A critical
batch size exists and is well studied, but where it sits depends on your model and your data,
so it can't be looked up — and Muon, the optimiser used here, is far less characterised at
large batches than Adam.

The nice part: changing `--grad-accum` costs no extra memory, so any gaming GPU can answer it.
Three runs at 8 / 16 / 32, compared at matched *tokens* rather than matched steps, scored with
`eval/heldout_loss.py`. Full design, including a confound to control for, is in
[issue #152](https://gitlab.com/komalbarun/kramba-ai/-/issues/152).

If you run it, I'd like to know what you find.

## License

Apache 2.0 — see [`LICENSE`](LICENSE). Use it, modify it, redistribute it, build a product on
it, sell it. Two conditions: keep the attribution and the [`NOTICE`](NOTICE) file, and state
what you changed. It carries an explicit patent grant from every contributor.

This used to be a source-available licence that forbade commercial use. That changed on
2026-08-21 — the sole copyright holder relicensed it, and commercial use no longer needs to
ask.

**The licence covers the code, not trained weights.** Checkpoint files — `.pt`,
`.safetensors`, and any other serialised model state — are a separate work, not under Apache
2.0, all rights reserved. That holds for any such file in this repository, in its history, or
released later.

Two things that follow, stated plainly so the permissive part is not ambiguous:

- **Weights you train are yours.** Running this code does not make your resulting model a
  derivative of it. Keep it, sell it, no obligation. The carve-out is only about weights
  distributed *by this project*.
- **Products you build are yours.** Apache 2.0 has no copyleft, so a proprietary product on
  top of this stack is fine — the two conditions are just attribution and stating changes.

Reproduction and criticism were always explicitly permitted, and still are, because a result
nobody may check is worth nothing.

The licence covers this implementation, not the published algorithms it implements. Gated
DeltaNet, GLA/Mamba-2 gating, the Mamba/RWKV-7 conv and output gate, BitNet b1.58, BAdam,
ALBERT weight sharing, Chinchilla scaling and Bavarian et al.'s fill-in-the-middle are other
people's work, credited in the files that use them and listed in [`NOTICE`](NOTICE).

## The write-up

A single page with the measured numbers — what fits, what it costs, what failed — is at
<https://ai-research.rambarun.com>, source in [`site/index.html`](site/index.html).

---

*Maintainer note: when a session reaches a conclusion worth recording — a benchmark, a dead
end, a gate outcome, an architecture decision — update the relevant file in [`docs/`](docs/index.md)
before ending the session. Those files are the durable log; don't leave a settled result only in
a commit message. Full rules in [`docs/ai_rules.md`](docs/ai_rules.md).*
