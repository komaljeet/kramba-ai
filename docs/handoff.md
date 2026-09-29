# Handoff — read this first (updated 2026-09-29 14:00)

Author: Rambarun Komaljeet

State of play for a session picking this up cold, or for the same session after a
context reset. Everything below is measured unless it says otherwise.

---

## 0. LIVE: the 896x16 pretrain owns the GPU until ~2026-10-10 12:00

**Do not launch anything on the GPU while this runs.** It is the real candidate
conversational model, an ~11-day commitment, not a smoke test. A second process on
the card OOMs one of them or makes WDDM spill and slows both; a second writer on the
same log overwrites it in place (both happened 2026-08-10). CPU-only work is fine.

| | |
|---|---|
| started | 2026-09-29 13:45 (step 0, fresh — no warm start) |
| shape | 896x16 hvn-hybrid, attention at layers 4,12; 174,043,993 params (129,013,721 body) |
| config | tied GPT-2 embeddings, ternary, Muon, `--fast`, `--grad-checkpoint`, `--chunked-loss 512`, batch 1 x accum 8 x seq 4096, no BCD, `--seed 0` |
| data | FineWeb-Edu only (`--pretrain-mix 0.0,0.0,0.0,1.0`, passed explicitly), `data_cache/tokenized/fineweb_edu.bin` cache hit, 1,621,692 windows; the run uses 945K of them = **single pass, 0.58x of the data, no repeats** |
| budget | 30 tok / body param = 3.87B tokens → `--max-steps 118116` (ceil(3,870,411,630 / 32,768)); cosine LR over all 118,116 steps |
| speed | 4,102 tok/s, 7.99 s/step, VRAM 57% (2.93/3.42 GB) at step 50 — matches the sizing tool's 4,106 |
| finish | ~10.9 days → **~2026-10-10 12:00** |
| code | worktree `C:\Users\komal\Documents\Codes\kramba-ai-wt-pretrain896`, branch `run/pretrain-896x16` at main d9bb7c6 — pinned so branch switches in the shared main tree cannot change the code under a running/resuming job |
| log | `logs\pretrain_896x16_real.log` (main tree); per-50-step rows in `training_log.csv` in the checkpoint dir |
| checkpoints | `checkpoints\pretrain_896x16_real\` (main tree): every 250 steps (~33 min lost at worst), newest 3 kept, every 10,000th kept forever (#163 pruning) — ~11 GB total on disk |
| launch / resume | `scripts\launch_pretrain_896x16.cmd` in the worktree. The SAME command resumes (model + optimizer + LR schedule). Check the GPU is empty first. |

Mid-run checkpoints sit under a cosine schedule that has not decayed, so they are
**not** "a model trained to N tokens" (see §4). Grade only the final one. No held-out
val split in-loop: this is single-pass, so train loss is loss on unseen windows; use
`eval/heldout_loss.py` on the final checkpoint for a fixed-slice number.

Next, after it finishes (not before): chat fine-tune, then the chat coherence eval.

## 1. (2026-08-17) Nothing was running

The GPU was idle. The #160 Phase 1 pair was killed deliberately on 2026-08-16: once the
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
~1.6B is the measured ceiling; the chunked loss returns 1436 MB at 48M and
**1.79 GB at 1.21B** (re-measured 2026-08-27 at seq 4096 — the "does not help at
1B" line in state.md is retracted; without it a 1B run spills at 103% of the
card). That is
engineering, not a research claim, and `site/index.html` is written around it.

## 2b. READ BEFORE LAUNCHING ANY PRETRAIN (added 2026-08-27)

**The default `--pretrain-mix` caps the token budget at 134,784,617 tokens** — enough for a
~7M-param model. OpenAssistant holds only 15.7M tokens and takes a 35% share, so it binds the
whole budget via `max_repeat=3`. Meanwhile **233 GB of FineWeb-Edu (~127B tokens) is already
downloaded** and gets a 0.0 share.

A 1.21B run on defaults gets **270x less data than it needs**. The loss curve would look
healthy the whole way while the model re-read a 134M-token budget — memorisation, not
training.

    --pretrain-mix 0.0,0.0,0.0,1.0     # FineWeb-Edu only; ~127B tokens

FineWeb-Edu is downloaded but **not tokenized** (no `fineweb_edu.bin` in
`data_cache/tokenized/`) — budget a one-time pass over the portion you will consume (~66 GB
of parquet for 36B tokens, not all 233 GB). Full numbers in [state.md](state.md).

**Also settled 2026-08-27:** the 1B step is **98.7% GPU-bound**, so there is no free speed in
transfer/prefetch/cache tuning. Micro-batch 2 is worth +7.1% **at ~478M only** (halve
`grad_accum` alongside); at 1.21B batch 2 **spills** and loses 21%, so batch 1 is already
optimal there. Do not inherit #141's batch-size conclusion from 83M.

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
