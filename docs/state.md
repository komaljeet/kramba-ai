# Current Measured State — EdgeHVN 83M & Speed Ceiling

> **Moved from README:** §5 "EdgeHVN 83M — the measured PPL picture", the speed-ceiling
> investigation, and the 50K probe verdict. These are the live, in-progress status blocks,
> not settled benchmarks. The hybrid design they feed is in
> [architecture.md](architecture.md); settled speed numbers in [benchmarks_speed.md](benchmarks_speed.md).

## EdgeHVN 83M — the measured PPL picture (2026-07-10)

The 83M has **three separate PPL/capability readings on three different data types** —
conflating them caused several wrong "ceiling" calls, so they're written down here:

| Data | PPL | What it measures | Status |
|---|---|---|---|
| **Chat fine-tune (OpenAssistant)** | **~26.5–30** (low: 3.277 loss / 26.5 ppl, step 440) | conversational **fluency** — chat format, coherence | ✅ **solved & measured** |
| **Pretrain (SlimPajama web)** | ~110 → **63.5 val @ 50K (stopped)** | broad web **knowledge** | ⬇ descending — undertrained, not a wall |
| **Grounded reading** (`eval/grounded_qa.py`) | 50K probe: **0/5 closed, 1/5 open, +1 lift** | **read + recall** exact tokens from context | 🟡 partial grounding — the hybrid test |

**So 83M is conversational in the fluent-chat sense** — ppl ~30 on chat data is a real,
measured conversational model (that's why it responds to "hi" correctly). What it is **not**
is *knowledgeable* (ppl ~110 on web) or *grounded* (0/5 recall). The chatterbox is real and
quantified; the gaps are knowledge + reading, **not** fluency.

**The "~110 = 83M capacity ceiling" claim is REFUTED.** The old 10K-step run plateaued at
SlimPajama ~110, but that run was **data-capped at ~135M tokens** (the OASST
`max_repeat=3` bind) — it plateaued because it ran out of fresh data, not because 83M hit a
wall. The Chinchilla-floor pretrain (Muon, OASST dropped from pretrain) descended and went
below 110: val PPL 182 (step 2K) → 133 (4K) → 114 (6K) → **102.9 (8K)** → … → **63.45
(50K)**, descending on the asymptotic tail; train loss confirms (8.35 → 4.71 avg, with a
mid-run flat patch ~step 3K–5.5K that then **broke through and accelerated** ~step 6K–7.5K
— a phase transition, the opposite of a ceiling). **Do not draw capacity conclusions from a
PPL that is still descending** — the undertrained-vs-capacity question is OPEN, tilting
toward *undertrained*. The 50K grounded-QA probe (capability: can it *read*) is the
decisive test; the PPL trajectory is now a positive signal.

> **TODO:** The README's EdgeHVN picture still describes the Chinchilla-floor pretrain as
> "still descending" (bg `b2dytn9gk`) in one paragraph and "STOPPED at ~51K" in the next —
> those contradict. The settled truth is **STOPPED at ~51K** (see the 50K probe verdict
> below). Remove the "still descending / running" phrasing.

> **Why the current pretrain shows val_ppl ~103 while the old run reported "~37 overall":**
> a **mix change, not a regression.** The Chinchilla-floor pretrain deliberately dropped
> OASST from pretrain (`--pretrain-mix 0.85,0.0,0.15` — OASST belongs in chat-FT), so its
> val PPL is SlimPajama+Wiki only, with no easy chat data dragging it down. The old "~37
> overall" included 35% easy OASST. Apples-to-apples on the hard knowledge data: old ~110 →
> new ~103 and falling — the new run is *better* on knowledge.

**Note — "hi → unrelated/garbage text" is NOT a knowledge problem.** Knowledge is about
facts ("capital of France"); "hi" needs none. A model with perfect knowledge and a broken
decoder still emits garbage to "hi". Real causes: (1) **serving/decoding** — if EOS + the
chat-format turn marker aren't hard-enforced, generation runs past "hello" into
base-pretrain web continuation → "unrelated text" (no proper EdgeHVN decode path exists
yet; the Phase B `chat_edgehvn.py` entrypoint / EOS-50256 format is exactly this fix);
(2) **chat-FT too shallow** — only 640 steps / 3 epochs (noisy, bounced to ppl ~39), so the
base-pretrain web-text prior leaks through; (3) the **base pretrain itself emits web
garbage**. Fix = **serving-side EOS/format enforcement + a real chat-FT + a cleaner trunk** —
not more knowledge.

## 50K probe verdict + STOP decision (2026-07-11)

The Chinchilla pretrain reached ~step 51K (val PPL **63.45 @ 50K**, Slim 4.18 / Wiki `nan` —
the `nan` is the known cosmetic val-mislabel, not a training fault; train loss ~4.08, ~0.7
steps/s). The mid-run `eval/grounded_qa.py` probe on the 50K trunk: **closed-book 0/5,
open-book 1/5, retrieval lift +1/5 → "Partial grounding."** The single open-book hit (Q3,
boiling point) surfaced *"Celsius"* from the passage — passage-primed, not lucky (closed-book
for the same question was HTML garbage). vs the old undertrained base's **0/5 / 0/5 / +0**
(pure HTML garbage): the 50K trunk moved from "pure garbage" to "answer-shaped, weakly
passage-primed." **This leans undertrained, not capacity-limited** — the first signs of
reading appeared. **Caveat:** this is the *pure base, not chat-FT'd*, so closed-book is HTML
garbage regardless of knowledge; 1/5 is a "the pretrain is now alive enough to sometimes
ground" signal, not a clean capacity verdict.

**Decision: STOP the pretrain at ~51K** (process tree terminated 2026-07-11, GPU freed;
trunk preserved at
`checkpoints/edge_chinchilla83m/edge_conversational_step{50000,51000}.pt`). Rationale: the A5
gate is a *binary architectural recall test* (lift > 0 vs 0/5) — recall comes from the 2 new
attention layers, not trunk PPL — and the 50K trunk already reads (partial grounding), which
is the floor the hybrid A4 FT builds on. A stronger trunk (50K→100K, val 63→~40) is
asymptotic-tail (the big jump 110→63 already happened), **won't fix the no-chat-FT
closed-book confound**, and costs ~30 h of the one scarce GPU that the P0 (hybrid A4) needs
now. **Next: A4** short grounded continue-FT from this trunk → A5 gate.

## Hybrid A4 — grounded continue-FT pipeline (2026-07-11: MR !78 merged; 8K run LAUNCHED + healthy; loss-log fix !79)

**Status — for the next LLM taking over.** The Chinchilla pretrain was **stopped
at ~step 51K** (2026-07-11) after the 50K probe gave a *partial-grounding* result
(closed 0/5, open 1/5, +1 lift — leans **undertrained**, see the verdict above).
A stronger 100K trunk was judged unnecessary for the binary recall gate, so the GPU
was free and **A4 was unblocked**. The A4 *code* below landed, was CPU-tested
(7/7) + integration-smoke-tested (data → train → gate, no crash), and was **merged
to main via MR !78** (normal merge `1a355c2`). **The 8K run is now LAUNCHED
(2026-07-11 13:56) and healthy** — see "Monitoring the live run" below. The next
LLM's job is to **run the A5 gate when the run finishes (~16:30 ≈ 4:30 PM, ~2.5 h
total)**, not to re-launch it.

**The three files (branch `feat/hybrid-a4-grounded-ft`):**
- `scripts/build_grounded_qa_data.py` — builds the grounded-QA *training* corpus
  (`data/grounded_qa_train.jsonl`, 40 multi-domain `(passage, question, answer)`
  triples where `answer` is a verbatim span of `passage`, **disjoint from the 5
  eval items** — no train/test leakage; `--validate` asserts both invariants). It
  is a *seed* — extend `CORE` or add a Wiki/SlimPajama cloze augmenter for scale.
- `continue_ft_hybrid.py` — the A4 entrypoint. Builds the hybrid EdgeHVN
  (`--arch hvn-hybrid --attn-layers "3,10" --attn-window 4096`), applies ternary,
  warm-starts the 10 DeltaNet layers from a pure-HVN trunk via
  `load_hybrid_trunk_into_model` (2 attn layers stay fresh-init), sets triton-scan
  + megakernel-scan, `torch.compile` (all under `--fast`), then fine-tunes on the
  grounded-QA data rendered in the **gate's exact open-book chat format** with loss
  only on the answer turn. Optional `--chat-mix 0.3` blends OpenAssistant
  (cached) so the pure-base trunk also learns chat format (the gate emits HTML
  garbage otherwise). AdamW + cosine, bf16, grad-accum, deferred loss accumulator,
  saves model-only checkpoints with `block_plan` metadata. Reuses the train
  script's proven hybrid→ternary→trunk-init→scan→compile ordering.
- `tests/test_continue_ft_hybrid.py` — 7 CPU tests: builder invariants + no
  leakage, `GroundedQADataset` loss masking (only answer+EOS learned; truncation
  keeps the answer), `build_hybrid_model`+trunk-init (block_plan correct, 41 trunk
  keys copied / 18 attn fresh / 36 orphaned skipped, forward shape), checkpoint
  round-trip preserves `block_plan`. Run: `python tests/test_continue_ft_hybrid.py`.

**Commands (the canonical A4 run, ~5-10K steps on GPU):**
```bash
# 1. (re)build the seed data — already committed at data/grounded_qa_train.jsonl
python scripts/build_grounded_qa_data.py --validate
# 2. A4 continue-FT (background + tail the CSV; stdout is block-buffered)
python continue_ft_hybrid.py --fast --max-steps 8000 \
    --hybrid-trunk-init checkpoints/edge_chinchilla83m/edge_conversational_step51000.pt \
    --checkpoint-dir checkpoints/hybrid_a4
# 3. the A5 gate (pass = retrieval lift > 0; pure HVN = 0/5; 50K trunk = 1/5)
python -m eval.grounded_qa --checkpoint checkpoints/hybrid_a4/hybrid_a4_step8000.pt
```

**Monitoring the live run (2026-07-11 13:56 → ~16:30, bg id `bjblkgc53`):**
- Log: `logs/hybrid_a4_ft.log`; CSV (flushes every 20 opt-steps):
  `checkpoints/hybrid_a4/a4_ft_log.csv`. Run from the `hybrid-a4-grounded-ft`
  worktree; the trunk is loaded from the MAIN checkout
  `checkpoints/edge_chinchilla83m/edge_conversational_step51000.pt` by absolute
  path (the worktree has no `checkpoints/`).
- Healthy start: `loss ~6.0` (true mean), `ppl ~400`, descending; `~1.13 s/step`;
  GPU 3.7/6 GB; warmup 240 then cosine over 8000. Checkpoints at
  `checkpoints/hybrid_a4/hybrid_a4_step{1000,2000,…,8000}.pt` (save every 1000).
- ⚠️ The loss reads as the **true mean** (see the logging fix below). Do NOT be
  alarmed by a single-digit loss — that is correct. If loss reads ~100+ again, the
  logging regressed (or you are on pre-`!79` code).

**Loss-logging fix (MR !79, commit `a023625`).** The original `continue_ft`
accumulated `loss/grad_accum` per micro-batch but only zeroed the accumulator
every 20 *optimizer* steps → the logged "loss" was the **sum of ~20 per-step
means ≈ 20× the true mean**, and `ppl = exp(min(loss,20))` capped at `4.85e8`
(meaningless). A GPU diagnostic proved the *model* was healthy all along (fp32 ==
bf16 == scan+mega, `|logit|max=30`, no NaN; init loss ~10.4, training ~6) — the
logging was the only bug. Fixed by tracking `n_steps_accum` and dividing → true
mean. The 8K run uses the fixed code.

**A5 gate decision tree (what the next LLM does with the result):**
- **Lift > 0 (ideally 3-5/5 open vs 0-1/5 closed)** → the hybrid *recalls*; the
  architecture works. Proceed to **Phase B** (serving: KV cache + `forward_step`
  + `generate` + `chat_edgehvn.py` entrypoint) → B4 agentic-gate (mock-repo tool
  loop) → the P0 (agentic coding).
- **Lift 0 (still 0/5)** → either A4 was too short (re-run `--max-steps 20000`,
  or raise `--chat-mix` for fluency) **or** 83M is capacity-limited. If a longer
  A4 still fails → pivot to **scale-up via #84 GaLore ~1B with this same hybrid
  recipe** (scale-invariant: count fills in by depth, W stays task-fixed).
- **Marginal (1-2/5)** → re-run longer + check the closed-book isn't HTML garbage
  (if it is, raise `--chat-mix`; chat format isn't holding).

**Gotchas (Windows / this env):**
- `--fast` = bf16 + TF32 + `torch.compile` + triton-scan + megakernel-scan.
  `--megakernel-scan` requires `--triton-scan` (it swaps the scan *backward*); the
  script warns + ignores if `--triton-scan` is off. A4 is short → the safe place to
  prove `--megakernel-scan`'s long-run reliability (only benchmarked on short runs
  so far, see [[speed-ceiling-2x-needs-both-halves]]) and bank the ~1.05× step.
- The trunk is `checkpoints/edge_chinchilla83m/edge_conversational_step51000.pt`
  (model-only, ~317 MB; `step50000.pt` is the exact probe trunk). Both are
  gitignored (local disk) — they exist on the dev machine; if missing, re-pretrain
  or restore from backup.
- CPU testing: force `CUDA_VISIBLE_DEVICES=-1` (Windows ignores `=""`; see
  [[windows-distributed-gpu-quirks]]). `torch.amp.autocast("cuda")` on CPU is a
  tolerated no-op (emits a UserWarning).
- Long GPU runs: launch in the background + tail the CSV
  (`checkpoints/hybrid_a4/a4_ft_log.csv`); stdout is block-buffered when redirected.
- One feature branch per issue off **updated `origin/main`**; push the feature
  branch only; the user merges manually. Never push to `main`.

**Handoff pointers:** GitLab issue **#97** (this track: A4/A5 + Phase B/C); MR
**!78** (A4 pipeline — **merged** to main `1a355c2`); MR **!79** (loss-logging
fix, branch `fix/a4-loss-logging` — merge when convenient; the 8K run already
has it). A stray remote branch `feat/hybrid-a4-grounded-ft` was re-pushed (reused
a merged-branch name) — redundant, deletable via
`git push origin --delete feat/hybrid-a4-grounded-ft`. Plan:
`C:\Users\komal\.claude\plans\bubbly-waddling-sundae.md`. Memory:
`hybrid-token-recall-track`, `edgehvn-83m-rag-finding`,
`speed-ceiling-2x-needs-both-halves`.