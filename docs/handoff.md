# Handoff — read this first (updated 2026-08-17 08:10)

Author: Rambarun Komaljeet

State of play for a session picking this up cold, or for the same session after a
context reset. Everything below is measured unless it says otherwise.

---

## 1. Nothing is running

The GPU is idle. The #160 Phase 1 pair was killed deliberately on 2026-08-16: once the
architecture turned out to be a reimplementation of published gated DeltaNet, that
comparison became a reproduction of someone else's result rather than a finding.

## 2. What was settled on 2026-08-16/17, and it changes the project

**The architecture is not original.** `hvn/core/deltanet.py` reimplements published work
and says so in its own docstring: canonical gated DeltaNet, GLA/Mamba-2 decay gate,
Mamba/RWKV-7 output gate and causal conv. The README and docs/index.md called it novel
for six weeks after #34 replaced the original block on 2026-07-04; both are corrected.

**The original block was retested properly and lost.** `--arch hvn-original-hybrid`,
448x8 so body params match DeltaNet's within 0.9%, same seed, same corpus, same constant
LR, both arms keeping 2 attention layers so only the recurrent block differs. Held-out
NLL at matched tokens per parameter:

| tok/param | HVNBlock | DeltaNet | gap |
|---|---:|---:|---:|
| 4 | 2.7679 | 2.6715 | +0.096 |
| 10 | 2.3645 | 2.2428 | +0.122 |

Noise floor is ±0.011, so the gap is 8-20x noise. Closer than July's ~29% (this is ~13%
in perplexity), so the stack's fixes did help -- not enough. The gap is flat to slightly
widening with more data, so "needs more training" is not the explanation. It only reached
9.6 tok/param because the block has NO kernel support (`hvn/core/hvn_block.py` has zero
triton/megakernel references) and runs 4x slower; the comparison is at matched tokens, so
that does not affect the verdict.

**Constant-memory inference is not a differentiator.** It defines the whole
linear-attention/SSM family. No comparison against Mamba or RWKV has ever been run.

**What remains defensible:** the training stack. 1.07B trains in 4.00 GB on a 6 GB card;
~1.6B is the measured ceiling; the chunked loss returns 1436 MB at 48M. That is
engineering, not a research claim, and `site/index.html` is written around it.

## 3. Decisions already made — do not relitigate

- **Tokens per parameter ≈ 30 for this architecture, not Chinchilla's 20** (#159).
  20→30 buys 1.9–3.3%; 30→40 buys ~1.1% for 33% more time. **The size trend is
  UNRESOLVED** — the elbow estimates are not ordered by size, one curve is
  truncated, and a parametric fit to all 120 points is degenerate (it returns 0.3
  tokens/param at 1B, contradicting every raw curve). Use 30 as a working number;
  do not extrapolate it across scale.
- **The 157M is stopped on purpose** (`checkpoints/code_157m_v2`, step 100). At
  25,000 steps it would reach 0.32× Chinchilla — the #112 trap. At 30 tokens/param
  a 157M needs 9.7 days; a **108M needs 4.7** and is the better use of a week.
- **Phase 0 result stands:** at 12M body, matched hours, HVN 1.9654 vs transformer
  1.9814 (+0.016, above the ±0.011 threshold) — while seeing **40% less data**.
- **HVN is NOT cheaper to train.** The transformer is **1.41× faster per step** at
  384 wide / 4096 long. The compute win is at inference and only inference. Do not
  let this claim drift back.
- **The existing `code_48m_v2` cannot be used as an A/B arm** — it ran unseeded.

---

## 4. Gotchas that have already cost time

- **Every new benchmark script MUST set the scan kernels.** Three separate scripts
  this week omitted `use_triton_scan` / `use_megakernel_scan` after `.to(device)` (or
  the equivalent constructor args) and ran the eager path against a Triton baseline.
  One of them produced a "4.25x regression" that was entirely the missing flag, and
  another produced a hybrid reference 4.6x too slow. Both defaults are `False`. Copy the
  four-line block from `scripts/size_vs_time_for_code.py` into any new probe, and if two
  sibling scripts disagree by more than ~20%, diff their setup before believing either.
- **Check the clock every reply** (`Get-Date`). Extrapolating it has been wrong by
  2 hours and by 7 hours in this session alone.
- **Background watchers get killed.** Five in a row died mid-session. The detached
  PowerShell runners survive; harness watchers do not. Verify the artifact itself
  (checkpoint file, CSV rows, result JSON), never a watcher's exit status. A
  "DONE" that arrives implausibly fast means CRASHED, not finished.
- **Report results at the ETA you gave, unprompted.** Phase 0 finished at 02:04
  and was reported at 05:47 only because the maintainer asked.
- **Read the docs before designing anything.** Two experiments were specced this
  week that were already measured and recorded (#155's A/B, most of #156).
- **Ask whether the test could pass for the wrong reason, before running it.** The
  first scaling study had to be restarted 2 h in because mid-run checkpoints under
  a cosine schedule are not "models trained to X tokens".
- `tests/test_issue33.py` does not parse on `main` — exclude it from full runs.
  6 collection errors in `test_issue51/65/67` are also pre-existing.

---

## 5. Where things live

| | |
|---|---|
| held-out evaluator | `eval/heldout_loss.py` (fixed 400K-token slice, last 20 MB) |
| capability gates | `eval/next_token_gate.py`, `eval/fim_completion_gate.py` |
| scaling curves | `results/scaling/*.json` |
| the page | `site/index.html` — self-contained, mechanism deliberately removed |
| open issues | #158 chunked loss, #159 ratio, #160 head-to-head, #135, #152 |

**The page must not describe how the architecture works.** The mechanism sections
were removed on request; keep results, remove method. Check for leaks before
publishing (`grep -i` for the architecture name and its component vocabulary).
