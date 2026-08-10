# Current Measured State — EdgeHVN 83M & Speed Ceiling

> **Moved from README:** §5 "EdgeHVN 83M — the measured PPL picture", the speed-ceiling
> investigation, and the 50K probe verdict. These are the live, in-progress status blocks,
> not settled benchmarks. The hybrid design they feed is in
> [architecture.md](architecture.md); settled speed numbers in [benchmarks_speed.md](benchmarks_speed.md).

## FFN activation buys no training speed — GELU stays, PowLU is worse (Issue #140, 2026-08-10)

**Question:** does swapping the DeltaNet FFN's activation (GELU → ReLU² or PowLU) speed up
training? **Answer: no.** ReLU² ties; PowLU is measurably *slower* and costs more VRAM.

`scripts/activation_speed_test.py`, RTX 4050, eager, batch 2 × seq 512, grad_accum 16,
5 interleaved rounds × 64 timed micro-batches, model config read from
`checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt` (d=512, 12 layers, 2-attn hybrid, ternary):

| arm | tok/s (median) | vs GELU | peak VRAM |
|---|---|---|---|
| GELU (control) | 9,035 | — | 2,455.7 MiB |
| **ReLU²** | 9,106 | **+0.79% — tie** | 2,455.7 MiB |
| **PowLU** (p=1.5 learnable) | 8,677 | **−3.96%** | 2,573.7 MiB (+4.8%) |
| GELU again (noise floor) | 9,178 | +1.58% | 2,455.7 MiB |
| slow control (positive control) | 7,413 | −17.95% | 3,357.0 MiB |

**Why this was predictable, and was predicted before running.** The FFN is two `d×2d`
matmuls around one pointwise op; at d=512 the matmuls are ~400× the arithmetic and all three
activations move the same bytes. An activation swap has almost no throughput to move. The
measurement is only worth recording because it *could* have refuted that.

**What makes the null trustworthy** (the #138 lesson — a test that cannot fail publishes bugs
as numbers):

- **Positive control.** A deliberately expensive activation (stacked transcendentals) came out
  −17.95%, proving the harness resolves activation-level cost. Without it, "no difference"
  would be indistinguishable from "harness measures nothing"; the script **voids its own run**
  if the control fails to separate.
- **Noise-floor arm.** GELU was entered twice under different names; the 1.58% gap between two
  *identical* configs is the resolution limit. ReLU²'s +0.79% is inside it, so it is reported
  as a tie, not a win. A genuine ≥2% win would have been caught.
- **Interleaved, not sequential** (round-robin), because this repo previously measured a ~2×
  difference between a config and itself from GPU contention drift (see the speed-ceiling entry
  below).

**PowLU's −3.96% is outside the noise floor and consistent across rounds 2–5** — a real, if
small, regression, plus ~118 MiB extra activation memory for the saved `abs`/`sign`/`pow`
intermediates. Combined with it being an odd, non-gating, superlinear function with no
literature support, it is **not recommended**.

Eager mode was deliberate: `torch.compile` fuses all three into the surrounding elementwise
epilogue, which can only *shrink* the gaps. A null in eager is a null in production.

**Consequence for the ablation.** The remaining case for ReLU² is quality-per-step (Primer,
2021 reports ~1–3%), not speed — and the maintainer's bar was a speed win. That needs ~6
GPU-hours of continued-FT, and would be confounded anyway: the checkpoint's weights spent
8,000 steps adapting *to GELU*, so a warm-started arm measures "how fast can these weights
adapt to a new activation", not "which activation is better". **Not funded; #140 closed on the
throughput finding.** If revisited, the honest design is small from-scratch runs, not a
warm-started swap.

Code landed regardless (`hvn/core/activations.py`, `ffn_activation=` on `DeltaNetBlock`/
`EdgeHVN`, 13/13 tests): `"gelu"` is byte-identical to the pre-#140 path (max|Δlogits| = 0.0
vs the old construction), so the lever is available at zero cost if a later scale makes it
worth re-testing.

**Two spec corrections found before building** (both would have produced a meaningless run):
the task named SwiGLU as the baseline, but the checkpoint has no `use_swiglu` key and is
**GELU** — and SwiGLU needs a gate matrix the checkpoint lacks, so that arm would have
measured fresh-vs-trained weights at 2.25× the FFN params. The requested known-answer cell
("identity activation reproduces 35.5%/8.5% recall") is also unpassable: identity is a
different function, and ±2%/±1% is inside 300-trial sampling noise (σ≈2.8% / 1.6%), so a
*correct* harness fails it ~half the time. Replaced with the exact-logit-match cell above.

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

## A5 gate result — hybrid A4 step5000 FAILED; re-FT on a bigger corpus (2026-07-12)

The A4 8K run (bg `bjblkgc53`) was killed at ~step 5920; latest surviving checkpoint
`hybrid_a4_step5000.pt`. Ran the A5 gate on it, plus a new **beyond-window probe**
(`eval/grounded_qa.py --far-probe`, MR !83) that shrinks the attention window and pushes
the passage past it, so only the DeltaNet recurrence — not the *windowed* attention — can
carry the answer. Both checkpoints, same gate:

| Checkpoint | standard: closed / open (lift) | far-probe W=128: closed / near / far |
|---|---|---|
| **pure trunk** `edge_conversational_step51000.pt` | 0/5 / 1/5 (**+1**) | 0/5 / **2/5** / **0/5** |
| **hybrid A4** `hybrid_a4_step5000.pt` | 1/5 / 0/5 (**−1**) | 2/5 / **0/5** / **0/5** |

**Verdict: the A4 FT made things WORSE, not better.** The hybrid at step5000 reads *less*
than the bare trunk (near-open 0/5 vs 2/5; open-book lift −1 vs +1). Cause = **data, not
steps or capacity**: the FT trained the 2 fresh attention layers on only **40** grounded
triples, so it memorised 40 pairs and degraded the trunk's weak reading instead of learning
the general skill. (Loss was still falling at kill, but on 40 examples more steps only
overfit harder.)

**Architectural finding from the far-probe (important — reframes the goal).** For BOTH
checkpoints, **far-open = 0/5, far-lift = 0**: the DeltaNet recurrence carries *zero exact
recall beyond the window*. And the 2 attention layers are *windowed*, so they cannot help
beyond the window either. Therefore the hybrid's achievable win is **exact recall WITHIN
the window** (the trunk's recurrence already got near-open 2/5 there; attention should push
that higher) — **not** beyond-window recall, which is architecturally capped at ~0 for both
the recurrence and windowed attention. Beyond the window, hybrid ≈ a plain sliding-window
transformer on exact recall; HVN's only beyond-window edge is a *lossy semantic gist* the
keyword gate can't credit. See [architecture.md](architecture.md).

## CORRECTION (2026-07-12, later) — the "made things WORSE" verdict above is RETRACTED

`grounded_qa`'s 5-item keyword gate is too noisy to trust for this decision — it gates a
1B-parameter scale-up call, and a 5-item test is dominated by luck (one keyword flips the
verdict) plus a real-world-knowledge/fluency confound unrelated to recall. Replaced it with
`eval/synthetic_recall_probe.py` (MR !83): hundreds of trials per checkpoint, INVENTED
entity/value pseudo-words (e.g. *"The Zorvex is called Blenthar."*) rendered in the model's
own trained chat/QA template — unambiguous exact-match scoring, zero real-world-knowledge
confound, in-distribution phrasing (an earlier raw-random-token-id version produced a flat
0% everywhere including inside the hybrid's own window — that was a format artifact, not a
recall measurement; retracted and fixed, see the script's docstring).

**Near-window exact-recall accuracy, 200 trials each:**

| Checkpoint | accuracy |
|---|---|
| pure trunk (`step51000`) | **4–5%** |
| old A4, 40-example corpus, "failed" (`hybrid_a4/step5000`) | **16–16.5%** |
| new A4, 5,240-example corpus (`hybrid_a4_v2/step5000`, only 63% through its run) | **35.5%** |

**The old 40-example FT was never a regression — it was already ~3-4x better than the trunk.**
The 5-item gate's −1 lift was noise, not signal. The corrected, high-N picture is a clean,
monotonic trend: bigger grounded-QA corpus → better recall, exactly as hypothesized, and the
new run isn't even finished. **Beyond-window** (target pushed >W=128 via 16 distractor facts):
trunk 0.5%, old A4 0.5% (both = noise floor), **new A4 5.5%** — the first checkpoint to show
signal clearly above floor, though still far below the near-window number; worth watching at
step8000, not yet a strong claim.

**`eval/synthetic_recall_probe.py` is now the primary R&D gate; `grounded_qa.py` (standard +
`--far-probe`) is secondary — useful for real-world-transfer sanity, not for this go/no-go.**

## FINAL VERDICT — A4 8K complete, mechanism PASSES the R&D gate (2026-07-12)

`checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt` (8,000 steps, 161.6 min, `feat/grounded-qa-
beyond-window-probe`; final train loss ~2.35). Full synthetic-recall trajectory, 200 trials
per point:

| Checkpoint | near (unbounded) | near (W=128) | far (W=128, beyond) |
|---|---|---|---|
| pure trunk | 4.0% | 5.0% | 0.5% |
| old A4, 40-ex | 16.0% | 16.5% | 0.5% |
| new A4 **step5000** | 35.5% | 35.5% | 5.5% |
| new A4 **step8000** | 35.0% | **35.5%** | **8.5%** |

**PASS.** Two clean, high-N, monotonic findings:
- **Near-window recall works and is large:** ~9× the trunk's floor (4% → 35.5%). This is the
  core hybrid claim (windowed attention adds exact recall the recurrence alone lacks) —
  proven, not just "not broken."
- **Near-window has PLATEAUED** (35.5% at both 5000 and 8000) — more steps on *this* corpus
  won't push it further; the next lever for near-window is a bigger/more-diverse corpus, not
  more training time.
- **Far-window kept climbing and is now real signal, not noise:** 0.5% → 5.5% → 8.5% (~17× the
  trunk's floor), monotonic with training. This is the first evidence in the project that the
  recurrence carries *some* genuine exact-token signal beyond the attention window — the
  capability the architecture wasn't designed to structurally guarantee. Still modest in
  absolute terms (8.5%, not "reliable"); whether it keeps climbing with more data/steps or
  saturates here is open.

**Decision: the 83M R&D probe clears the bar.** This is the go-signal for the next staged
milestone in [README's "The Dream"](../README.md) — scaling the recipe toward a ~20B-class
native ternary-hybrid, not a reason to scale-up-via-capacity-question (#84 GaLore is no longer
the next step; it was the fallback for a failed gate, and the gate passed).

`grounded_qa.py`'s 8k re-run remained noisy as expected (standard: closed 2/5 open 0/5;
far-probe: closed 2/5 near 1/5 far 1/5, with the lone far "hit" being a garbled multi-country
list that happened to contain "Portuguese" — not real recall) — confirms it should stay a
secondary real-world-transfer sanity check, not the decision instrument, going forward.

## NEXT PLANNED MILESTONE — ~270M local, conversational fluency + tool-calling (2026-07-12)

Not the ~20B jump yet — a nearer, locally-fundable step first. Decided after checking real
constraints (no code committed for this milestone yet, this is the plan):

- **1B (and certainly 20B) cannot train locally under vanilla full AdamW.** Measured RTX 4050
  training-VRAM ratio is **16.8 bytes/param** (bf16 weights+grads+fp32 AdamW moments+master,
  Issue #69, [training.md](training.md)) → 1B needs ~16.8 GiB (off by ~3× on 6 GB), 20B needs
  ~336 GiB (off by ~14× even on a 24 GB RTX 3090 — a single card physically cannot hold a 20B
  model's training state; needs real multi-GPU sharding). Cloud-rental math (RunPod RTX 3090,
  sourced pricing $0.22/hr Community / $0.46/hr Secure): a full Chinchilla-optimal 1B pretrain
  needs an assumption-flagged **~1,341 GPU-hours** (~$295 total, ~56 days continuous, or ~15
  months at a $20/mo cap) — real, but a deliberate multi-month commitment.
  > **UPDATE (2026-07-12, later same day) — 1B now trains locally, cleanly, measured.**
  > Issue #99 (BAdam block-coordinate-descent, `--block-coordinate`) landed: 268M config
  > −39% peak VRAM (6.9→4.2 GiB), 1.04B config hard OOM → runs (still spilling past 6 GiB
  > alone). Issue #100 (EdgeHVN never had `--grad_checkpoint` wired, despite #49/#88 closing
  > it for the core model class only — fixed, bit-identical parity verified) stacked on top:
  > **1.04B config → 4.6 GiB peak, a clean fit with headroom, no spill.** See
  > [training.md](training.md)'s BCD section for the full numbers
  > (`results/bcd_ckpt_vram.json`). **The cloud-rental math above is no longer the path for
  > 1B at all** — it trains on this laptop today. GaLore (#84) and 8-bit Adam (#85) remain
  > unimplemented, further composable levers for 10B-class work, not needed to reach 1B.
- **The measured local RTX 4050 ceiling today is ~270M params** (d=1024, 24L, bf16 +
  `--grad_checkpoint`, 4.3 GiB peak — Issue #69 table above). Free, immediate, no new
  engineering.
- **Calibrated goal for 270M — NOT "agentic coding" capability.** Real competent code
  generation needs far more scale + data than a from-scratch local run can provide (even
  purpose-built 0.5–1.5B code models trained on trillions of tokens are considered weak for
  reliable agentic use). What 270M *can* plausibly prove: **(1) conversational fluency**
  (coherent chat, achievable at this scale with real chat-FT) and **(2) correct tool-call
  format-following** (emitting well-formed `[TOOL_CALL]{...}[/TOOL_CALL]` — a learnable format
  skill, not a reasoning skill).
- **Blocker independent of param count:** EdgeHVN has no serving/decode path yet (Phase B —
  KV cache + `forward_step` + `generate`, see [architecture.md](architecture.md)'s hybrid
  design) and no real tool-use FT corpus (Phase C — `hvn/tools.py`'s
  `generate_tool_training_data` is only ~50 synthetic templates, a seed). Testing *any*
  tool-calling behavior needs these built first, regardless of whether the model is 83M, 270M,
  or 20B. Scoping this build is part of the 270M milestone, not a separate later step.

**TODO:** scope the 270M run (config, chat-FT data mix, Phase B serving stub sufficient for
format-only tool-call testing) and open a GitLab issue before starting code, per
[ai_rules.md](ai_rules.md).

## Stage 1 — conversational FT on the 83M/2-attn hybrid (issue #108, 2026-07-15)

Prior context: the 2-attn hybrid checkpoint (`checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt`)
passed its recall gate (35.5% near-window / 8.5% far-window, `eval/synthetic_recall_probe.py`)
but was mostly trained on grounded-QA data (only a 30% OASST chat-mix) — not yet a genuinely
conversational model. A follow-on 3-attn scaling attempt regressed and was closed without
merging (issue #109; not yet on `origin/main`, tracked only via the GitLab issue).

**Run:** `finetune_edge_conversational.py --base-checkpoint hybrid_a4_step8000.pt
--grounded-replay data/grounded_qa_expanded.jsonl --grounded-replay-frac 0.15 --epochs 3`
(branch `feat/conversational-ft-108`) — full OASST corpus (3,482 conversations → 3,465
examples) mixed 85/15 with grounded-QA replay (5,240 examples) via the same
`ConcatDataset`+`WeightedRandomSampler` pattern already proven in `continue_ft_hybrid.py`.
1,632 optimizer steps, 104.2 min, final loss ~2.6-2.8 (ppl ~14-16).

**Bug caught pre-launch (fixed, not just observed):** a GPU smoke test surfaced the exact
same ~20x-inflated loss-logging bug previously found and fixed in `continue_ft_hybrid.py`
(commit `a023625`) — `finetune_edge_conversational.py`'s `finetune()` predated that fix and
had the identical accumulator bug (summed per-opt-step means across a 20-step window without
dividing by the step count). Fixed the same way (`n_steps_accum` divisor) before the real run;
smoke logs went from loss~50/ppl-capped-at-4.85e8 (nonsense) to loss~2.3/ppl~10 (sane).

**Qualitative (fixed prompt set, greedy decode via `sample_reply`, no repetition penalty):**
- "hi" → *"Hello! How can I help you today?"* — coherent, on-topic (a real improvement; the
  base checkpoint's chat behavior was noticeably weaker/format-only before this stage).
- "What is the capital of France?" → *"The capital of France is Paris."* — correct.
- "What's 2+2?" and the haiku prompt both degrade into repetition loops (`"2+3+3+3+3..."`,
  `"island of the island..."`). Likely a greedy-decoding artifact (no repetition penalty /
  sampling in the current `sample_reply` smoke-check, not necessarily a training failure) —
  worth re-checking once Stage 2's real decode path exists and a proper sampling strategy can
  be tested; not re-diagnosed further here.

**Recall regression check (`eval/synthetic_recall_probe.py`, 300 trials):**

| Checkpoint | near-window | far-window (W=128, 16 distractors) |
|---|---|---|
| 2-attn PASS baseline (pre-chat-FT) | 35.5% | 8.5% |
| 2-attn + Stage 1 chat FT (`hybrid_chat_v1.pt`) | **28.3%** (85/300) | **5.3%** (16/300) |
| pure trunk floor (reference) | 4-5% | 0.5% |

**Verdict: PASSES the "no collapse" gate, but recall genuinely eroded** — near-window down
~20% relative, far-window down ~38% relative. The 15% grounded-QA replay mix slowed but did
not fully prevent forgetting; both numbers remain far above the trunk floor (not a collapse),
so this is an acceptable, measured tradeoff rather than a regression to fix — but a real cost
of chat-focused FT, not a free win. If recall matters more than chat fluency for a future use
case, a higher replay fraction (or interleaved/alternating-stage FT instead of one blended
pass) is the lever to revisit.

**Next:** Stage 2 — the serving/incremental-decode path (issue #97 Phase B). `EdgeHVN` still
has no `forward_step`/KV-cache; all evaluation above (including `sample_reply`) uses expensive
full-sequence re-forward. See the full roadmap plan for Stage 2/3 design.

## EdgeHVN incremental decode landed (issue #97 Phase B, 2026-07-15)

`EdgeHVN`/`TransformerBlock` now have `init_state`/`forward_step`, and `EdgeHVN` has
`init_conversation_state`/`chat_step` matching `HamiltonianVortexNetwork`'s exact method
contract — `hvn/chat.py`'s `ChatSession`/`StateCheckpoint` work against EdgeHVN with zero
changes to their own logic (verified: the mixed DeltaNet+attention-KV-cache state round-trips
through the existing generic `_encode_state_tree`/`_decode_state_tree` JSON codec). New
`hvn/edge/chat.py:EdgeConversationFormatter` speaks EdgeHVN's actual trained format
(`<|user|>`/`<|assistant|>`, EOS-terminated) instead of the core model's bracket-tag protocol.
Branch `feat/edgehvn-serving-97`, `tests/test_edgehvn_incremental.py` (9/9 pass).

**A real finding surfaced while GPU-verifying parity on the actual trained checkpoint
(`hybrid_chat_v1.pt`), worth recording so it isn't mistaken for a serving bug later:**
comparing `forward_step`'s incremental output against a plain `forward()` call on a *trained*
checkpoint can show large divergence (up to ~3.2 in logit scale over just 20 tokens/12
layers) — looks alarming, is not a bug in the new decode path. Root cause: `forward()`
dispatches to `DeltaNetBlock._scan_chunked` (a fast, reassociated approximation used for
training speed, active whenever `train_chunk` is set — true for this checkpoint) rather than
the exact `_scan_sequential` formula that `forward_step` implements token-by-token.
`_scan_chunked` itself diverges non-trivially from `_scan_sequential` on real trained weights
(verified directly, bypassing `forward_step` entirely: a single DeltaNet layer's chunked vs.
sequential scan already differs by ~0.39 with this checkpoint's weights) — a pre-existing
numerical property of the #83/#96 training-speed optimization, invisible with the
small-random-weight test models used when that optimization was originally validated
("~1e-7" in the code comments). **`forward_step`'s actual correctness bar** — matching
`_scan_sequential`, the documented reference recurrence, apples-to-apples on the same trained
weights — **holds to ~3e-6 (machine precision)**, confirming the incremental decode path is
correct. Locked in as `tests/test_edgehvn_incremental.py::test_deltanet_forward_step_matches_scan_sequential_not_scan_chunked`.
**Practical implication:** don't use a trained/`train_chunk`-enabled checkpoint's `forward()`
as the parity oracle for incremental decode work — use `_scan_sequential` directly, or a
fresh-init/`train_chunk=0` model, or expect this gap and don't treat it as a regression.

## EdgeHVN wired into the Docker chat app + real end-to-end verification (2026-07-15)

`ModelVersion.architecture` field (migration `0003_modelversion_architecture`) + branching in
`hvn_worker/worker.py`/`inference.py` let the Django/Celery/Channels chat app serve EdgeHVN
checkpoints, not just the original `HamiltonianVortexNetwork`. New
`EdgeHVNChatSmokeTests` (`web/tests/test_chat_smoke.py`) drive the real (unmocked)
worker→inference→ChatSession path — 78/78 Django tests pass.

**Beyond the unit tests: verified against the real Docker stack, not just `manage.py test`.**
Built the image (`docker compose build`), brought up postgres/redis/app-web/app-worker/
app-beat, registered `hybrid_chat_v1.pt` as an `architecture="edgehvn"` `ModelVersion`, and
drove two real chat turns through the actual HTTP API (login → create conversation → POST a
message → Celery task runs → reply persisted) — the same code path the browser UI hits, no
mocking:
- Turn 1 ("hi") → *"Hello! I am an AI assistant..."* — coherent opening.
- Turn 2 ("What is the capital of France?") → *"The capital of France is Paris..."* — correct,
  and **confirms multi-turn `StateCheckpoint` save/load round-trips correctly for real** in
  the deployed stack (not just the CPU unit test's synthetic round-trip check).
- Both replies degrade into repetition after the first sentence or two — the same known
  quality limitation already documented for this checkpoint in the Stage 1 (#108) writeup
  (arithmetic/haiku prompts looping under greedy/low-temperature decode); not a new Phase B
  bug, just now also confirmed present via the real serving path, not only the CLI
  `sample_reply` check.

**One real bug caught and fixed during this verification** (would NOT have been caught by
the CPU unit tests, which import `hvn`/`train_edge_conversational` directly rather than
through the built Docker image): `docker compose build app-web` only rebuilds the `app-web`
image — `app-worker`/`app-beat` are separately-tagged images per service and were still
running a stale build missing the `train_edge_conversational.py` COPY, causing
`ModuleNotFoundError: No module named 'train_edge_conversational'` on the first live attempt.
Fixed by rebuilding all three service images explicitly. Worth remembering for any future
Dockerfile change to this app: **rebuild `app-web app-worker app-beat` together**, not just
whichever service you're actively iterating on.

Not yet done: a real trained-quality pass (fixing the repetition-under-greedy-decode issue,
e.g. sampling/repetition-penalty tuning) and exercising the actual browser UI (this
verification drove the REST API directly, the same code path, but not the JS frontend/
WebSocket client itself).

## Stage 3 — agentic tool-use FT + gate (issue #110 / #97 Phase C, 2026-07-15)

Continued from Stage 1's `hybrid_chat_v1.pt`. Landed (branch `feat/agentic-tooluse-97c`):
`repetition_penalty` param on `EdgeHVN.chat_step` (real repetition was observed in Stage 2's
Docker verification); sandboxed coding tools `read_file`/`edit_file`/`run_tests` in
`hvn/tools.py` (path-traversal guarded, `run_tests` takes no free-form args so a
model-controlled argument can't smuggle shell content — caught a real bytecode-cache staleness
bug here via the tests, fixed with `PYTHONDONTWRITEBYTECODE=1`); a 600-example programmatic
tool-use corpus (`scripts/build_tool_use_data.py`, every coding example actually executed +
verified during generation, not fabricated); `ToolUseDataset` with three-way loss masking
(query masked, `[TOOL_CALL]` learned, `[TOOL_RESULT]` masked as environment-injected context,
answer learned — the fix for a real flaw in the pre-existing
`hvn.tools.generate_tool_training_data`, which trained the model to hallucinate tool results);
and `EdgeToolCallingSession` (`hvn/edge/tool_session.py`), a from-scratch EdgeHVN-native
agentic loop built on `forward_step` (not a port of the pre-existing `ToolCallingSession`,
which is bound to a `generate_incremental` contract that doesn't exist on `EdgeHVN`).

**FT run:** continued from `hybrid_chat_v1.pt`, mixed 50% chat / 10% grounded-QA replay / 40%
tool-use, 1,743 optimizer steps (~2h), final loss ~1.7-1.9 (ppl ~6-7). Qualitative check via
the old full-forward `sample_reply` (no repetition penalty) showed the model correctly
emitting `[TOOL_CALL]\n{"name":"calculator","arguments":{"expression":"2+2"}}\n[/TOOL_CALL]`
for "What's 2+2?" — right tool, right arguments — before degrading into a malformed repeat
(a `sample_reply`-specific artifact; the real serving path, `EdgeToolCallingSession`, uses
`repetition_penalty`).

**Agentic coding gate (`eval/agentic_coding_gate.py`, `checkpoints/hybrid_tooluse_v1.pt`):** a
fresh (not a literal training example, but the same learned bug-template family) one-line-bug
mock repo, driven through the real `EdgeToolCallingSession` + sandboxed coding tools.

| Bar | Result |
|---|---|
| **Pass (required):** well-formed `read_file → edit_file → run_tests` sequence, no parse failures | **YES** |
| **Stretch (reported, not required):** the bug actually gets fixed | **NO** |

The model emitted the exact correct *tool sequence and JSON structure* three calls in a row,
zero parse failures — a real, non-trivial, previously-absent capability. It did **not**
correctly ground the arguments: it called `read_file`/`edit_file` on a hallucinated path
(`"bounds/jquery"`, likely bleed-through from the `bounds_*.py`-named training family mixed
with unrelated web-dev vocabulary) instead of the actual file named in the query
(`gate_calc.py`), so the edit never landed and `run_tests` correctly reported the original
failure. **This is exactly the calibrated outcome stated going into this stage** (per the
270M-milestone framing already in this file): tool-call **format-following** is a learnable
format skill and it works; correctly **grounding arguments from the specific query** is a
reasoning-adjacent skill this 83M model does not yet reliably have. Single-trial result (n=1,
real measurement, not averaged over multiple sampled runs) — a fuller sweep (several seeds/
prompts) would be needed to quantify the pass rate precisely, not just confirm the capability
exists.

**Roadmap status: Stages 1, 2, and 3 are all now landed** (branches `feat/conversational-ft-108`,
`feat/edgehvn-serving-97`, `feat/agentic-tooluse-97c` respectively, all pushed, none yet merged
by the user). The model can chat, recall within its window, be served through the real Docker
app, and follow a tool-calling format reliably — genuinely "good enough" per the plan's own
honesty bar, not GPT-4-class autonomous coding. The clearest next lever, if better argument-
grounding is wanted, is more/more-diverse tool-use training data (the same volume lesson as
grounded-QA and OASST throughout this whole track) rather than further architecture work.

## Live Docker chat testing: a real serving bug fixed + the scale-vs-data verdict (2026-07-15)

**Real bug found via actual live use, not a unit test.** Chatting with `hybrid_tooluse_v1.pt`
through the deployed Docker UI produced replies with the literal text `<|endoftext|>` appearing
mid-response followed by more generated text. Root cause: `hvn.chat.ChatSession.user_turn_streaming`
(what the web app's streaming path actually calls, not `chat_step`) had its own from-scratch
generation loop that **never checked for a stop token at all**, for any model — correct for the
core `HamiltonianVortexNetwork` (bracket-tag format, no single stop token by design) but wrong
for EdgeHVN, which was trained with a single EOS-terminated turn. A sampled EOS token was
decoded as literal text and generation just kept going instead of stopping. Same story for
`repetition_penalty` (issue #110): it existed on `chat_step` but `user_turn_streaming` never
used it. Fixed: `ChatSession` gained `eos_token_id`/`repetition_penalty` params (both default
off — zero behavior change for the core model), wired for the EdgeHVN call site
(`web/hvn_worker/inference.py`, `eos_token_id=50256, repetition_penalty=1.3`). Verified directly
(an EOS-biased test model now stops immediately instead of running the full token budget) and
via the live API (`hybrid_tooluse_v1.pt` now replies with a single clean, complete sentence
instead of a run-on garbled one). Landed on `feat/edgehvn-serving-97`.

**After the formatting bug was fixed, real qualitative testing surfaced the actual remaining
gap clearly: content coherence, not decoding mechanics.** Replies are now grammatically
well-formed and properly terminated, but frequently don't engage with what was actually asked
— generic assistant-boilerplate text regardless of the specific question, no real recognition
of conversational context (e.g. a frustrated "what the fuck?" gets a boilerplate "as a language
model, I can provide you with information" instead of anything responsive to the tone/intent).

**Verdict, stated directly: this is primarily a capacity (scale) problem, not a data problem.**
More/better training data is the proven, cheap lever for *surface* quality — format-following,
topic breadth, reduced repetition (this project has now demonstrated that pattern three times:
grounded-QA's 40→5,240-example expansion, OASST full-corpus vs capped, and the tool-use
corpus's programmatic generation). But genuine contextual reasoning — actually parsing intent,
not just producing fluent-sounding text — is the kind of capability that scaling-laws research
(and this project's own earlier "83M capacity ceiling" investigation, which concluded 83M was
undertrained *relative to its own ceiling*, not that the ceiling itself was high) suggests is
much more parameter-count-bound than data-bound at this scale. Pouring more data into a fixed
83M model would likely keep improving fluency/format without closing this specific gap.

**Decision: defer #107 (scale past 83M) — it is the correct next lever, not started this
session.** The full pipeline (conversational FT → serving → agentic tool-use) is proven
end-to-end at 83M, which was the point of building it at this scale first rather than
committing to a multi-day pretrain up front. Scaling the *same* recipe (DeltaNet trunk +
windowed-attention hybrid, same warm-start approach) to whatever size #107 lands on is now the
well-justified next step when picked back up — not a guess, a conclusion from real measurement
at the smaller scale first.

## #112 — 268M real-recipe validation attempt: a WDDM TDR driver crash, not a slow compile (2026-07-24)

Picking up #107/#112 (scale past 83M): before committing to a multi-hour/day pretrain at
120–270M, ran a short **300-step validation** of the "recommended ceiling config" from this
file's VRAM table (d=1024, 24L, ~268M, `--grad-checkpoint`) with the **real** data pipeline
(SlimPajama/WikiText via the existing `data_cache`, not synthetic) and the real `--fast` stack,
to get an honest steps/sec + VRAM-peak reading before scoping the full run.

**First finding — the untracked smoke probes already in `pretrain-300m-112` (`logs/probe_143m.log`,
`probe_300m.log`, `probe_300m_clean.log`, 150 steps each) were not measuring this table's 268M
config at all.** They used d=1024/19L (~302M, not 24L/268M) and never passed `--grad-checkpoint`,
so the 300M-class probe hit the documented WDDM VRAM spill (~30s/step, ~13x slower than the
143M probe's 2.2s/step) — consistent with this file's "spills → 4-15x slower" note, not a new
finding, just a probe that was missing the one flag that fixes it.

**Second finding — the first real validation launch appeared to hang for 50+ minutes with zero
step output, but the actual cause was a Windows driver crash, not a slow `torch.compile`.**
Windows Event Log (System log) shows, ~3 minutes into the run:
```
14:38:24  Driver Management has concluded the process to add Service nvlddmkm ... status: 0.
14:38:25  Driver Management has concluded the process to add Service NVDisplay.ContainerLocalSystem ... status: 0.
```
This is the signature of a **WDDM TDR (Timeout Detection & Recovery) reset** — Windows kills
and reloads the GPU driver if a single kernel launch blocks longer than its watchdog timeout
(default ~2s). Plausible trigger: first-time `torch.compile` + Triton-scan autotuning for the
new 268M/24L/seq_len=512 shape (the log's "Not enough SMs to use max_autotune_gemm mode" line
shows it was mid-autotune-fallback right before the reset). After the reset, the training
process kept running but was silently stuck on a dead CUDA context for the remaining ~47
minutes — indistinguishable from "still compiling" until investigated, because **stdout is
fully buffered when redirected to a file** (the same gotcha recorded for the 83M chinchilla
pretrain in memory — confirmed again here at a different scale).

**Contributing/latent finding, independent of the TDR crash:** the DataLoader worker
subprocesses spawned by `--fast`'s default `--num-workers 4` resolved `sys.executable` to
`C:\Users\komal\Documents\PythonVersions\314\python.exe` — **not** the project's
`.venv\Scripts\python.exe` — even though the parent process was correctly launched from the
venv. That `314` interpreter lacks `transformers` (a previously-known gotcha from the 83M
pretrain). The workers stayed alive without crash-looping in this run (worker `__getitem__`
only touches pre-tokenized numpy memmaps, no `transformers` import needed at fetch time), so
this did **not** cause this particular hang — but it's a live risk for any future run that
relies on multi-worker loading on this machine. Safe workaround used for the retry:
`--num-workers 0`. Root cause (why spawn resolves the wrong interpreter here) is still
unexplained and worth a real fix before depending on `--num-workers > 0` for a long pretrain.

**Post-crash GPU state (as of this writing):** `nvidia-smi` fails with "insufficient
permissions"; `torch.cuda.is_available()` is `False`; Device Manager shows the GPU
(`PCI\VEN_10DE&DEV_28A1&SUBSYS_8BD4103C&REV_A1\4&3382C3D5&0&0009`) status as `Error`. A
non-elevated `Disable-PnpDevice`/`Enable-PnpDevice` toggle failed (`Generic failure`, HRESULT
`0x80041001`) — needs an elevated PowerShell at minimum; **decision: full machine restart**
rather than chase the elevated-toggle path further.

**TODO after restart:** (1) re-verify `nvidia-smi` + `torch.cuda.is_available()` before
touching training again; (2) relaunch the same 300-step validation with `python -u` (unbuffered,
so real progress is visible immediately instead of only after the OS flushes the buffer) and
`--num-workers 0` (sidesteps the wrong-interpreter spawn risk above); (3) only after a clean
steps/sec + VRAM-peak reading, decide the full pretrain step budget for #112 and launch it as
a properly backgrounded, log-tailed run per the project's standing rule for long GPU jobs.

## #112 — pre-training-audit OOM bug fixed; BCD + torch.compile recompile conflict found (2026-07-24, later same day)

After the restart, `nvidia-smi`/CUDA came back healthy. Relaunching the 1.21B validation
(`d_model=2048, num_layers=24, n_heads=8, batch=4, seq_len=128, --grad-checkpoint
--block-coordinate --fast --num-workers 0`) hit a **new, real** crash — not the driver TDR —
immediately after model construction:

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 16.00 MiB. ...
Of the allocated memory 12.31 GiB is allocated by PyTorch...
```

**Root cause, confirmed directly (not the BCD mechanism itself):**
`pre_training_audit()` in `train_edge_conversational.py` has always run two checks that build
a *second full model copy* on GPU before `BlockCoordinateScheduler` ever exists — Check 2
(`copy.deepcopy(model)`, then a full **un-frozen** forward+backward across every param) and
Check 3 (rebuilds a second `EdgeHVN` to verify the checkpoint save/load cycle, moved to GPU,
no backward). Neither check is aware of `--block-coordinate` at all. At 1.21B params fp32,
real model weights (~4.16 GiB) + audit copy weights (~4.16 GiB) + full un-BCD-frozen
gradients across the whole audit copy (~4.16 GiB) ≈ 12.3 GiB — matches the crash almost
exactly. Harmless at 83M/268M (both copies + full grads still fit easily), so it went
unnoticed until pushing toward 1B. **Fixed**: both checks are now skipped when
`--block-coordinate` is set, with an explanatory log line (not made "BCD-aware" — that would
be new, untested logic for a one-off sanity check; real forward/backward problems still
surface immediately on step 1 of actual training). Also fixed a small, independent latent bug
found in the same trace: `load_checkpoint_for_resume()` rebuilt `EdgeHVN(...)` without passing
`grad_checkpoint` at all, so a resumed run silently lost checkpointing regardless of the
original run's setting — now threaded through, and `_save_edge_checkpoint` persists the key.

**BlockCoordinateScheduler itself, and the blanket `model.to(device)` calls, were confirmed
NOT the problem** — `_activate()` handles a CPU-resident starting model gracefully via
idempotent `.to(...)` calls, and the blanket `model.to(device)` is correct as long as
`offload_inactive_weights` isn't enabled (it isn't, by design — see below). No changes needed
there.

**After the fix, 1.21B still doesn't practically run.** Both a compiled (`--fast`) and an
eager-mode (no `--compile`) relaunch of the identical 1.21B config passed the audit and
reached `STEP 5 — START TRAINING` cleanly, then **stalled silently for 30–90+ minutes with
zero step output**, no error, and a healthy GPU (`nvidia-smi` clean throughout — not a repeat
of the TDR crash). The isolated micro-benchmark (`scripts/bench_block_coordinate_vram.py`)
says this config should fit at ~4.65 GiB, but that number never included the real script's
full overhead (DataLoader, checkpoint I/O, audit machinery) — almost certainly the documented
WDDM silent-host-spill behavior, but far worse than the "4–15× slower" already on record, and
never actually root-caused (would need real profiling, not attempted). **Open, unsolved.**

**Scaled down instead, on the same fixed code path, to get an honest number today.** At
**355M** (d=1024, 24L — the "268M-class" config, actual param count higher due to
vocab/embedding overhead), `--grad-checkpoint --block-coordinate`:

| Mode | Total (150 steps) | Avg s/step |
|---|---|---|
| Eager (no compile) | 661.4s | **4.41s/step** |
| `--fast` (compiled) | 1009.3s | **6.73s/step** — *slower*, not faster |

**Real, confirmed finding: `--block-coordinate` and `torch.compile` actively conflict.**
`BlockCoordinateScheduler` toggles `requires_grad` on a different layer-block every
`--bcd-steps-per-block` steps (default 100) — that's the entire mechanism behind its VRAM
saving. `torch.compile` guards its captured graph on `requires_grad`, so every block switch
invalidates the graph and forces a full recompile. Confirmed directly (not just inferred from
timing) via `TORCH_LOGS=recompiles`, which printed exactly this at step 100:
```
[__recompiles] Recompiling function forward in hvn/edge/model.py:309
    triggered by ... tensor 'blocks.0.ffn.0.bias' requires_grad mismatch.
```
Per-50-step timing breakdown of the compiled run confirms the shape: steps 1–50 (7.43s/step,
includes the one-time initial compile), 50–100 (5.66s/step, settled), 100–150 (7.10s/step,
spikes back up right at the block-switch boundary — a second recompile). For a **real, long**
pretrain with the default `steps_per_block=100` this isn't a one-time cost — it recompiles
**every 100 steps for the entire run**, not just once.

**Net status:** `--grad-checkpoint --block-coordinate` alone (eager mode) is real and working
at 355M (4.41s/step). Combining it with `--fast`/`torch.compile` is currently a net loss, not
a neutral no-op — worth remembering before assuming "just add `--fast`" is free. Three
possible fixes were identified but **not pursued** (deliberately paused, not blocked): raise
`--bcd-steps-per-block` much higher to dilute the recompile cost over a real multi-thousand-
step run, treat `--fast` and `--block-coordinate` as mutually exclusive and document it, or
investigate `torch._dynamo.config.force_parameter_static_shapes = False` (surfaced directly in
the recompile diagnostic as a possible lever, unproven). **1.21B remains unresolved** — the
audit-OOM bug that blocked it is fixed, but the silent stall through the real training script
is a separate, still-open problem needing real profiling, not a guess.
## MoE FFN on DeltaNet — works, but loses to simply scaling the dense model (Issue #123, 2026-08-02)

MoE was scoped as the only remaining lever that attacks FLOPs-per-token rather than memory
(the offload work turned out to be only ~2x of the 3,300x 6B-vs-142M gap; params and tokens
are the other 42x each). Implemented as `hvn/core/moe.py::MoEFFN` — E experts, top-k routing,
Switch-style load-balancing aux loss — routing **only the FFN** (~40% of block FLOPs). The
recurrence is deliberately left dense; see the module docstring for why routing q/k/v risks
mixing incompatible subspaces inside a single recurrent state, a failure mode transformers do
not have.

2x2 at d=512, 12 layers, 5M-token budget, E=8/top-2 (`results/moe/moe_2x2.json`):

| arm | params | eval PPL | wall-clock |
|---|---:|---:|---:|
| dense | 39,756,656 | 119.657 | 556.6s |
| shared | 10,830,684 | 156.424 | 510.2s |
| MoE | 128,015,216 | 105.544 | 1062.3s |
| shared + MoE | 18,185,564 | 131.344 | 930.4s |

Harness consistency check passed: the dense arm reproduces the ternary 2x2's 119.396 at 119.657
(0.2%), and shared reproduces 154.1/156.1 at 156.424.

**The bar, set before the numbers landed.** MoE is better than dense in absolute terms (−11.8%),
but that is the wrong comparison. The question is whether it beats what the *same compute* buys
by simply scaling the dense model up. Using this repo's measured −0.504 exponent, 1.40× params
(MoE's theoretical FLOP cost) is worth **19%**; the *measured* wall-clock cost was **1.91×**,
which raises the bar to **~39%**.

| comparison | gain | bar | verdict |
|---|---:|---:|:---:|
| dense → MoE | −11.8% | 19% / 39% | ❌ |
| shared → shared+MoE | −16.0% | 19% / 39% | ❌ |

**Verdict: at this budget, 3.2× the parameters bought less quality than 1.4× the compute would
have.** Scaling the dense model is the better use of the same FLOPs. The failure is consistent
across both the dense and shared baselines, so it is not an artifact of one configuration.

**The levers do compose.** MoE helps the shared model (−16.0%) at least as much as it helps the
dense one (−11.8%), so sharing and MoE are not fighting each other — MoE simply is not worth
its compute in either. This closes acceptance criteria 4 and 5 of #123 with a negative result.

**Two caveats, neither of which rescues the result:**
- **The 1.91× vs 1.40× gap is partly implementation.** `MoEFFN` loops over experts in Python
  rather than batching them; a tuned kernel would narrow the wall-clock penalty. But the
  FLOP-based 19% bar is missed regardless, so optimisation changes the margin, not the verdict.
- **5M tokens is very undertrained, and MoE has an extra thing to learn.** Routing — which
  expert for which token — is a harder learning problem than a dense FFN's, and may simply not
  have had the data to specialise. **This is the single strongest reason not to treat this as
  final**, and it is untested. If MoE is revisited, the first thing to check is whether the gain
  grows with token budget; a fixed ~12-16% that does not grow means the routing never learned
  anything useful.

**One result worth keeping regardless of the compute verdict.** shared+MoE reaches PPL 131.3 at
**18.19M stored params**, against dense needing **39.76M** for 119.7 — **2.2× fewer stored
parameters for 10% worse PPL**. That is a deployment-size story, not a training-compute one, and
it stacks with ternary's ~20x bits (measured to compose cleanly earlier today). For the
edge-inference thesis this project is actually built around, that combination matters more than
the training-FLOP comparison MoE just lost.

## Looped early exit — no-retrain hyperparameter search FAILS the gate (Issue #122, 2026-08-01)

Wrapped `EdgeHVN.forward_step` (`hvn/edge/looped.py`) so the block stack can re-run up to N
times per token, exiting once normalized softmax entropy drops below threshold T — a
"looped transformer" style weight-tied refinement, entropy-only exit signal (a trainable
confidence-MLP was considered and dropped: with no retraining allowed, its weights would be
random and its "confidence" scores meaningless). Contract preserved exactly — `forward_step`/
`init_state`/`chat_step` signatures unchanged, `ChatSession` needs zero code changes
(`tests/test_looped_early_exit.py`, 6/6 CPU parity/correctness tests, incl. a proof that a
discarded loop's candidate state never leaks into the next loop's attempt).

**Full grid N ∈ {2,4,8,16} × T ∈ {0.1,0.3,0.5,0.7} on `checkpoints/hybrid_a4_v2/
hybrid_a4_step8000.pt`** (the checkpoint the 35.5%/8.5% recall baseline was originally measured
on — `hybrid_tooluse_v1.pt`, named in the original task spec, does not exist in this working
tree or any worktree). Measured via a dedicated `forward_step`-based generation harness
(`scripts/sweep_looped_early_exit.py`), not `eval/synthetic_recall_probe.py`'s own `generate()`
— that uses full `forward()`, which uses `_scan_chunked` and diverges numerically from
`forward_step`'s `_scan_sequential` on a trained checkpoint (see the #97 incremental-decode
entry above); reusing it would have silently never exercised the loop wrapper at all.

**Result: every one of the 16 combinations fails.** The pattern is a clean binary, not a
tunable middle ground:

| T | Behavior (all N) |
|---|---|
| 0.1 / 0.3 | avg loops ≈ N (always maxes out) → recall = **0%**, near AND far, every time |
| 0.5 | mixed loop counts, recall still **0%** near AND far |
| 0.7 | avg loops ≈ **1.0–1.07** (almost never actually loops) → recall reverts to ~baseline |

There is no hyperparameter point where the model loops *and* stays useful. **Root cause: the
block stack was never trained to refine its own output** — a second pass through untrained
weights pushes the hidden state off the distribution it was trained on, and exact-token recall
collapses almost immediately. This was flagged as the key risk before implementation (an
untrained "confidence" mechanism can't be assumed quality-neutral) and the sweep confirms it
directly rather than just plausibly.

**Closest candidate, N=4/T=0.7, re-run at the full 300-trial validation gate** (vs. a
300-trial-measured N=1 baseline, not the old doc numbers — see harness note above):

| Metric | Baseline | N=4,T=0.7 | Target | Result |
|---|---:|---:|---|:---:|
| Near-window recall | 33.67% | 28.67% | ≥30% | ❌ |
| Far-window recall | 6.33% | 7.0% | ≥7% | ✅ (exactly on the line) |
| Avg loops, easy tokens | — | 1.0 | ≤2 | ✅ |
| Avg loops, hard tokens | — | 1.024 | ≥N/2 | ❌ |
| Tokens/sec | 27.82 | 54.34 | improves | ✅ (see caveat) |

**Overall FAIL**, and the two nominal passes don't rescue it: avg-loops-hard ≈ avg-loops-easy ≈
1.0 means the mechanism isn't discriminating hard from easy tokens at all — at T=0.7 it's
functionally inert, not "using" loops. The 2× tokens/sec figure is inconsistent with a loop
count barely above 1.0 and is almost certainly single-run GPU/system measurement noise (one
40-token generation per combo), not a real architectural speedup.

**Decision: untrained looped early exit does not work on this checkpoint at any tested (N,T).**
This is a real, informative negative result for the no-retrain phase of #122, not a bug —
correctness was independently verified via the parity tests above. The next step, if this
capability is still wanted, is a loop-aware **continued fine-tune** (train with randomized loop
counts so the block learns to be a genuine refiner of its own output), reusing this checkpoint's
existing pretrain/FT investment — not a from-scratch retrain, and not further hyperparameter
search on the frozen checkpoint. Full sweep data: `results/looped_early_exit/sweep_results.json`,
`results/looped_early_exit/validation_gate_N4_T0.7.json`.

## Loop-aware continued-FT: recipe check + a scan-kernel default flip (Issue #122, 2026-08-01)

**Stage 1 — cheap ~780K-param sandbox** (`scripts/train_looped_sandbox.py`, char-level,
from-scratch, trains in minutes): tests the loop-aware training recipe itself (random loop
count `k ~ Uniform[1,max_loops]` per step, loss averaged across EVERY loop's prediction —
"deep supervision" — not just the final one, so the model is actually taught that an early
loop's output alone should be usable). Result **ambiguous, not conclusive**: held-out loss got
worse with more loops in both an initial (flawed, out-of-distribution held-out set) and a
corrected (in-distribution train/val split) run. Root confound: the tiny repeated training
corpus (583 chars × 200) is fully solvable at k=1 alone, so there's no training pressure for
extra loops to do anything — the sandbox can't distinguish "recipe doesn't work" from "task was
too easy to need it." Recorded in `results/looped_early_exit/sandbox_1m_results.json`.

**Stage 2 — the real test** (`continue_ft_looped.py`): a genuine **continued fine-tune**, not a
retrain — warm-starts directly from `checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt` (same
checkpoint the #122 sweep above ran on) and continues training with the loop-aware objective on
`data/grounded_qa_expanded.jsonl` (5,240 real, non-repeated examples — the sandbox's
memorization confound doesn't apply). In progress; loss descending cleanly (0.87 → 0.46 by
step 560/2000).

**A real bug caught mid-run, and a genuine finding underneath it.** The first launch measured
**41.88s/step** and was wrongly attributed to VRAM/WDDM spill-over (a documented failure mode
elsewhere in this doc) — cutting `--batch-size` 4→2 didn't change `nvidia-smi`'s reported usage
at all, which should have been the tell. The real cause, found by diffing against the working
reference script (`continue_ft_hybrid.py`) rather than continuing to guess: that script's
`--fast` enables the fused Triton scan + megakernel backward (#83/#96) on the DeltaNet blocks;
`continue_ft_looped.py` had never wired them in at all, so it ran DeltaNet's slow default eager
scan, uncompiled. Wiring them in: **~1s/step, a ~42x speedup**, confirmed with real
`torch.cuda.max_memory_allocated()` instrumentation (4.6 GiB peak) in place of `nvidia-smi`
guessing.

**Default flipped: triton-scan + megakernel-scan are now unconditionally on in both
`continue_ft_hybrid.py` and `continue_ft_looped.py`** (no `--fast` needed for this part
anymore). This was checked thoroughly before flipping, not assumed:
- `tests/test_issue83.py` + `tests/test_issue83_stacking.py` (7/7, re-run fresh) confirm the
  Triton forward/backward/full-block match the exact sequential reference to rel-diff < 2e-3.
- `scripts/ab_megakernel.py` documents the megakernel backward as loss-bit-identical, grad
  diff ~5e-8.
- **The direction of the risk is the opposite of what was first assumed.** The Triton kernel
  implements the same safe log-space formulation as `_scan_chunked_stable`
  (`hvn/core/deltanet_triton.py`: "never forms `1/P`"). The path it replaces — the *default*,
  no-flag eager `_scan_chunked` — is the one with a real, code-documented correctness caveat:
  its own docstring says it can be "silently wrong on decayed channels at large chunks / hard
  decay," and the #44 `P`-floor fix only bounds the catastrophic Inf/NaN case, not full
  accuracy. So enabling triton-scan isn't a speed-for-accuracy trade — it's strictly better on
  both axes.

**Scope of the flip, deliberately narrow.** Only `continue_ft_hybrid.py`/`continue_ft_looped.py`
were changed. NOT touched: `DeltaNetBlock`'s own constructor default (`use_triton_scan=False`),
`EdgeHVN`'s own default (also `False`, and it doesn't expose `use_megakernel_scan` as a
constructor arg at all — only settable post-construction), or `train_edge_conversational.py`'s
own more elaborate `--fast`/`--triton-scan` machinery (which already conditionally enables
triton-scan under `--fast` for `--arch hvn`, gated by logic not fully audited here). Those are
wider-reaching, separately-scoped changes — affecting the main pretraining script, past
benchmark reproducibility assumptions, and other callers — not something to flip as a
side-effect of an #122 stage-2 script fix.

**Follow-up — attempted the broader flip too, and the caution above was justified.** Tried
flipping `DeltaNetBlock`/`EdgeHVN`'s own class-level `use_triton_scan`/`use_megakernel_scan`
constructor defaults to `True` (the true single source of truth every caller inherits from).
Two real, previously-undiscovered caveats surfaced from actually running the broader test suite
rather than assuming the earlier scoped verification generalized:

- **`use_megakernel_scan=True` by default breaks `torch.compile` fusion**:
  `tests/test_issue83_stacking.py::test_zero_graph_breaks` failed — 2 graph breaks instead of 0.
  Correctness is unaffected (all parity tests still passed), but it's a real compile-fusion
  regression, not just noise. Reverted this one back to `False` at the class level.
- **`use_triton_scan=True` by default breaks `BlockCoordinateScheduler` weight-offload
  parity** — a real, non-trivial loss divergence (0.297 vs 0.010 for the eager scan, not
  floating-point noise) specifically in
  `tests/test_block_coordinate_offload.py::test_lru_cache_hit_avoids_reload_and_preserves_parity`.
  Isolated with a standalone repro to confirm it's the scan flag and not something else, and
  confirmed `grad_checkpoint` alone (no BCD/offload) is unaffected
  (`tests/test_edgehvn_grad_checkpoint.py` 5/5) — narrow to the offload buffer-caching
  mechanism specifically, not root-caused, so also reverted to `False` at the class level.

**Net decision: class-level defaults (`DeltaNetBlock`, `EdgeHVN`) stay `False`.** Only
`continue_ft_hybrid.py`/`continue_ft_looped.py` have it on (verified safe there — neither uses
BCD/offload or `torch.compile` alongside the loop-aware path). This also answers why
`train_edge_conversational.py`'s own `--fast`/`--triton-scan` wiring was left untouched: that
script supports `--block-coordinate`, so the same BCD-offload interaction risk applies there
too — extending the flip to it would need the same root-cause work this pass didn't do.
`use_megakernel_scan` is now explicitly threaded through `EdgeHVN`'s constructor (previously
only settable post-construction) even though its default stays `False` — harmless plumbing
improvement, kept regardless of the revert.

## Stage 2 result — loop-aware continued-FT does NOT fix the recall collapse (Issue #122, 2026-08-01)

`continue_ft_looped.py` completed: 2000 steps (115.3 min, `--fast`, batch_size=2, grad_accum=16,
max_loops=3, warm-started from `checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt` on
`data/grounded_qa_expanded.jsonl`), final train loss 0.2507 (ppl 1.28), clean convergence.

**Validation (`scripts/validate_looped_ft.py`)** — fixed loop counts N=1,2,4
(`entropy_threshold=-1.0` forces exactly N loops every time, isolating "does looping still
corrupt recall" from any entropy-threshold question), 20 trials/N, near-window synthetic
recall, FT checkpoint vs the frozen baseline it was warm-started from:

| N | FT checkpoint (2000 steps) | Frozen baseline |
|---|---:|---:|
| 1 | 30.0% (6/20) | 45.0% (9/20) |
| 2 | **0.0%** (0/20) | **0.0%** (0/20) |
| 4 | **0.0%** (0/20) | **0.0%** (0/20) |

**Verdict: the loop-aware continued-FT does not fix the collapse.** Identical pattern on both
checkpoints — recall craters to exactly 0% the instant looping actually happens (N>1), and the
FT checkpoint is even slightly *worse* at plain single-pass (N=1) recall than the frozen
baseline it started from (30% vs 45%) — a real cost with zero offsetting benefit at N>1.
Real, non-repeated training data (`data/grounded_qa_expanded.jsonl`, 5,240 examples) rules out
the memorization confound that made the stage-1 sandbox result ambiguous — this is a clean
negative result, not an artifact of insufficient data variety.

**Working hypothesis for why, not yet tested:** the deep-supervision recipe (loss averaged
across every sampled loop's prediction, `hvn.edge.looped.looped_forward_train`) has a
structural gap. The same weights are trained to satisfy "loop 1 alone should already answer
correctly" and "loop 2 — which further transforms loop 1's now-correct hidden state — should
*also* answer correctly," with nothing in the objective rewarding "leave the answer alone once
it's already right." Real looped-transformer/universal-transformer designs typically need an
explicit mechanism for that (step embeddings, learned halting, or a residual-style update that
lets later iterations be small perturbations rather than full re-transformations) — a plain
re-application of the full block stack has no such pressure built in. More training steps on
the same recipe would not obviously fix a structural gap like this; it would need a different
training design, not more of the same one.

**Status: Issue #122's Looped Early Exit, as implemented (weight-tied loop over the
already-full-depth 12-layer hybrid + naive deep supervision), does not deliver a working
mechanism — neither without retraining (the original N×T sweep) nor with a real continued-FT
(this stage). Parking here rather than continuing to iterate on this specific recipe.** If
picked up again, the honest next steps are (a) a genuinely weight-shared/looped architecture
(this repo already has `EdgeHVN(shared_weights=True)` infrastructure for that — the original
issue's own "reuse a single weight-shared block" framing, never actually used by this work,
which instead looped the full already-deep hybrid) or (b) a training objective with an explicit
settle/halt mechanism, not just randomized-depth deep supervision. Full results:
`results/looped_early_exit/validate_looped_ft.json`,
`checkpoints/hybrid_looped_ft/hybrid_looped_step2000.pt`.

## Gated-loop fix attempt — tested cheaply, makes things worse when opened (Issue #122, 2026-08-02)

Tried the (b) option above directly: `scripts/train_looped_sandbox_gated.py` adds a learned
per-token gate so loop 2+ blends the block stack's raw output with the previous hidden state
(`h <- h + sigmoid(loop_gate(h)) * (Blocks(h) - h)`) instead of overwriting it outright — loop 1
stays the full ungated pass. Same disposable ~780K-param, from-scratch, char-level sandbox as
stage 1 (not a fine-tune — deliberately tiny so this is a fast, cheap mechanism check before
touching the real 83M checkpoint again).

**v1** (gate bias init -3, `sigmoid(-3)~=0.047`, near-closed at start), 3000 steps: partial
win. Loss still degrades with more loops but ~3x less than the ungated recipe (+0.09 k1->k4 vs
ungated's +0.27). Gate barely moved during training (0.047 -> 0.049).

**v2** (gate bias init -1, `sigmoid(-1)~=0.27`, more open at start, hypothesis: v1's gate was
gradient-starved by starting too saturated), 5000 steps: **hypothesis was wrong, and the result
is worse on every axis** — both overall single-pass quality (k=1 loss 5.80 vs v1's 4.56) and
the loop-degradation amount (+0.19 k1->k4 vs v1's +0.09). The gate spent the whole run trying
to close itself back down (0.27 -> 0.257) and never got back to where v1 started.

**Reading:** starting the gate MORE open made things worse, not better -- the correct amount of
gate openness is "as closed as possible," and v1's conservative init was already closer to
right. This rules out the "gradient-starved gate" explanation and points to a cleaner, stronger
conclusion: **any exposure to the block stack's repeated (loop 2+) transformation hurts,
independent of how the gate is tuned** -- not a fixable hyperparameter/init issue. The gate
mechanism works as designed (it lets the model suppress harmful extra loops), but there's
nothing useful in the block stack's second-pass output for it to let through. This is now the
second training-side fix attempted (deep supervision alone, then deep supervision + gating) that
fails to make looping productive on this architecture -- reinforces the (a) option above
(a genuinely weight-shared, deliberately-smaller-and-looped architecture) as the more promising
remaining direction over further training-recipe tweaks on the current already-full-depth
hybrid. Full results: `results/looped_early_exit/sandbox_1m_gated_results.json` (v1),
`results/looped_early_exit/sandbox_1m_gated_v2_results.json` (v2). Not attempted on the real
83M checkpoint -- the cheap sandbox check is exactly what caught this before that spend.

## First positive signal — looping WITHOUT early exit modestly helps (Issue #122, 2026-08-02)

Every attempt above trained with deep supervision (every sampled loop count's prediction
graded independently), because that's what entropy-gated early exit requires (loop 1 alone
must already be a usable answer). That requirement was never separated from "does repeated
computation help at all" until this check. `scripts/train_looped_sandbox_fixed.py` drops early
exit entirely: fixed N loops every step (no random k), loss on ONLY the final loop's
prediction -- the standard way weight-tied/looped depth is normally trained in the literature.
Same disposable ~780K-param sandbox, N=1 (plain single-pass baseline) vs N=4, same 3000
training steps, held-out loss:

| N | held-out loss |
|---|---:|
| 1 (single pass) | 5.0050 |
| 4 (always 4 loops, final-loop-only supervision) | **4.9621** |

**N=4 beats N=1** — modest (~0.9% relative) but real and the first improving result in this
entire investigation. Confirms the hypothesis from the gated-fix section above: deep
supervision's requirement that loop 1 already be independently correct was actively
suppressing the model's ability to use loop 2+ productively. Remove that requirement and
repeated computation *is* mildly useful at this toy scale.

**This cleanly separates two previously-conflated questions.** "Does looping help at all"
(this result: yes, mildly, when not required to also support early exit) is now distinct from
"does early exit specifically work" (still no — every early-exit-compatible training attempt,
deep-supervised or gated, failed). A **fixed-depth looped model with no early exit** is a
plausible, still-open direction; **early exit** (the original #122 ask — cheap tokens skip
loops, hard tokens get more) remains unproven and every attempt at it so far has failed.
Whether a fixed-depth loop is worth the N× inference cost without the early-exit savings is a
separate cost/benefit question, not yet evaluated. Full results:
`results/looped_early_exit/sandbox_1m_fixed_N1.json`, `sandbox_1m_fixed_N4.json`.

## Real-scale check of the fixed-N recipe — sandbox signal did NOT transfer (Issue #122, 2026-08-02)

Ran the sandbox's winning recipe (fixed N=4 loops, final-loop-only supervision, no early exit)
as a real continued-FT on the 83M checkpoint (`continue_ft_looped.py --fixed-n
--no-deep-supervise`, 2000 steps, 196.3 min, final train loss 0.30). Validated at N=1/2/4 on
both the step-500 (25% through) and step-2000 (final) checkpoints:

| N | Fixed-N FT, step 500 (early) | Fixed-N FT, step 2000 (final) | Frozen baseline |
|---|---:|---:|---:|
| 1 | 26.7% | 30.0% | 45.0% |
| 2 | — | **0.0%** | **0.0%** |
| 4 | **0.0%** | **0.0%** | **0.0%** |

**The sandbox's positive signal did not transfer.** Same collapse pattern as every prior
attempt, identical at both the early and final checkpoint — more training didn't help within
this run's budget either. The toy-scale result (N=4 beating N=1 by ~0.9% in held-out *loss*)
was a real, measured effect, but it didn't translate into the *recall* metric that actually
matters here — a small general next-token-prediction improvement is not the same thing as
fixing long-range associative recall, which apparently needs a qualitatively different fix, not
just "let the model use more compute if it wants to." This is the reason the real-scale check
mattered rather than treating the sandbox result as sufficient evidence on its own.

**This exhausts the training-recipe angle for Issue #122.** Three distinct training approaches
tried on the current (already-full-depth, 12-layer) hybrid architecture, all fail on the recall
metric:
1. Deep supervision (every sampled loop graded independently) — fails, recall collapses to 0%.
2. Deep supervision + a learned per-token gate — fails; opening the gate further made both
   overall quality and the collapse *worse*, not better.
3. Fixed-N, final-loop-only supervision (no early exit at all) — showed a small, real
   toy-scale loss improvement, but recall still collapses to 0% at real scale.

**Final status: parking Issue #122 as far as training-recipe iteration on this architecture
goes.** The only genuinely untested direction left is architectural — a real weight-shared,
deliberately-smaller-and-looped model (`EdgeHVN(shared_weights=True)` already exists in this
repo for that), which this entire investigation never actually tried; everything above looped
the already-full-depth 12-layer hybrid instead. That's a materially different, larger
commitment (its own from-scratch-vs-pretrain decision) and a separate piece of work if picked
up again, not a variation on anything tried here. Full results:
`results/looped_early_exit/validate_looped_ft_fixed_step500.json`,
`results/looped_early_exit/validate_looped_ft_fixed_step2000.json`,
`checkpoints/hybrid_looped_ft_fixed/hybrid_looped_step2000.pt`.

## Mechanism diagnostic — the input-injection/skip-attention hypothesis is REFUTED (2026-08-02)

All four training recipes above failed identically (recall → 0% the instant N>1, general loss
fine) — a pattern consistent with looping destroying information exact recall depends on,
rather than a trainable deficiency. Two mechanism-level fixes were proposed and tested directly
on the **frozen** checkpoint (`scripts/diagnose_loop_recall_collapse.py`, zero training, ~20 min
GPU): (A) re-add the original token embedding every loop (`h ← Blocks(h + x0)`, standard in
Universal Transformer / Looped Transformer, absent from this implementation), and (B) run the 2
windowed-attention layers only once (loop 1) and loop only the 10 DeltaNet layers, on the theory
that attention specifically needs an un-smeared input to copy exactly.

**Neither rescues recall, alone or combined** — all four variants (current / A / B / A+B)
collapse to exactly 0% at N=4, N=1 sanity-checked at 45% matching the known baseline. This
refutes the specific hypothesis: it is not simply "attention loses its token anchor across
loops" (A alone would have fixed that) and not simply "re-running attention on an already-
transformed hidden state corrupts the copy" (B alone would have fixed that, and did not — even
with attention frozen after loop 1, recall still died).

**Revised hypothesis, not yet tested:** the destructive mechanism more likely sits in the
**DeltaNet recurrence itself**, not attention. DeltaNet's update is an overwrite —
`S ← diag(α)S + β(v − Sk)kᵀ` — keyed on k/q computed from whatever hidden state is fed in. On
loop 2+, k/q are computed from an already-transformed hidden state rather than the original
token, so the overwrite plausibly replaces loop 1's correctly-keyed association with a
differently-keyed one instead of refining it. Testing this needs inspecting the actual
recurrent state matrix across loop iterations (real mechanistic-interpretability work), not
another architectural toggle — a materially larger investigation than anything tried in this
doc so far, and not started. Full results: `results/looped_early_exit/diagnose_loop_recall_collapse.json`.

**Status: parking here.** Five real experiments now agree (4 training recipes + this
no-training mechanism probe) that looping this architecture's block stack, in every form
tried, destroys exact recall. The training-recipe angle and the two most obvious
architecture-level bypasses are exhausted. Remaining open directions, neither started:
(a) genuine weight-shared smaller-and-looped architecture (see above), (b) direct inspection of
the DeltaNet state matrix across loops to confirm or refute the overwrite hypothesis.

## ⚠️ RETRACTION — every #122 result above was measuring a one-line bug (2026-08-02)

**All of the negative conclusions in the six #122 sections above are invalid.** They do not
show that looping this architecture destroys recall; they show that
*looping-with-corrupted-memory* destroys recall. The bug was in the first file written for
this issue and silently confounded everything downstream.

**The bug.** `hvn/edge/looped.py` committed the **last** loop's recurrent state as the state
carried to the next token. Only loop 1 runs the blocks on the real token embedding — loops 2+
run them on an already-transformed hidden state, so their k/v are not derived from this token
at all. For DeltaNet the state *is* the memory (`S ← diag(α)S + β(v − Sk)kᵀ`), so committing a
later loop's state writes an association keyed on noise, once per extra loop — N writes per
token instead of one — corrupting precisely what the next token must read back.

**How it was found: by looking at the failure instead of scoring it.** Five experiments had all
measured the same binary (did the answer contain the target word) and none had inspected the
behaviour. Doing so (`scripts/observe_loop_failure.py`) showed the N=4 output is raw HTML/web
markup — this doc's own documented signature for the *base pretrain model* — meaning looping
knocked the model out of its instruction-following distribution rather than merely degrading
the answer. It also measured the residual stream growing 351 → 2560 (7.3×) over 8 loops at a
near-constant +315/loop, with argmax locking onto token 3467 (`' \'`) from loop 2 onward.

The decisive isolation came from a **failed sanity check**. That script's α-sweep included
α=0, which leaves `h` at loop 1's value so the logits are provably identical to N=1 and it
*must* reproduce ~45%. It returned 0/20. That invalidated the α curve, but with the logits held
identical the only remaining difference from N=1 was which state got committed. A 2×2 then
separated the two candidate mechanisms (`scripts/test_state_commit_hypothesis.py`, frozen
checkpoint, 20 trials/cell):

| | state ← loop 1 | state ← loop 4 |
|---|---:|---:|
| **logits ← loop 1** | 40.0% (N=1 baseline ✓) | **0.0%** |
| **logits ← loop 4** | **35.0%** | 0.0% (old behaviour) |

Cell (c) is decisive: loop-4 logits over a clean loop-1 state hold at 35%, statistically
indistinguishable from the 40% baseline. **The residual-stream drift is real but survivable —
it is not the cause.** The single fatal mechanism is the state commit.

**Fix:** commit loop 1's state unconditionally, regardless of which loop the exit fires on.
Extra loops refine the *prediction*; they do not rewrite *memory*. One line.
`tests/test_looped_early_exit.py` had asserted the loop-2 state and so locked the bug in —
rewritten to assert loop 1's, plus a guard that the two states genuinely differ so it cannot
silently pass again (6/6).

**Post-fix re-validation** (frozen checkpoint, fixed N, early exit disabled, two independent
runs of the same checkpoint so their spread calibrates the noise floor):

| N | run A | run B | pooled |
|---:|---:|---:|---:|
| 1 | 40.0% | 45.0% | 42.5% |
| 2 | 35.0% | 45.0% | 40.0% |
| 4 | 35.0% | 35.0% | 35.0% |
| 8 | 30.0% | 25.0% | 27.5% |

Recall **holds at every N**, versus exactly 0.0% at every N>1 before the fix. A pooled
monotonic decline (42.5% → 27.5%) is visible and directionally consistent with the measured
drift, but at ~2 standard errors it is suggestive rather than established — ±2 hits at n=20 is
ordinary noise here, as the two identical-checkpoint runs demonstrate.

**What is now known vs still open.** Looping is non-destructive — that is settled. The original
#122 ask (entropy-gated early exit buying a speedup at maintained quality) is **untested**: the
above used fixed N with the exit disabled. The N×T sweep must be re-run from scratch, because
entropy was previously computed on garbage logits, so the threshold behaviour that drove the
original sweep's conclusions ("low T always maxes out loops, high T never loops") was itself a
symptom of the bug. Likewise the four training runs are re-opened: they trained against
corrupted memory, so their failures carry no information about the training recipes themselves.

**Process note worth keeping.** The α=0 sanity check that exposed this was added on the fifth
experiment. Had a "this configuration must reproduce the known baseline" assertion been built
into the very first sweep, the four training runs (~6 h GPU) would never have been spent. Every
scoring harness in this project should carry a cell whose correct answer is already known.

## Post-fix N×T sweep — mechanism works, early exit's PREMISE does not (2026-08-02)

Re-ran the sweep the retraction above invalidated, on the frozen checkpoint with the
state-commit fix in place, 20 trials/cell. The harness now carries a **known-answer cell**
(`max_loops=8, T=1.1`: the loop machinery is engaged but the threshold is trivially
satisfiable, so it must exit at loop 1 and equal the N=1 baseline by construction). It
passed at 45.0% vs 45.0%, delta 0.0pp — the check that would have caught the original bug,
which an N=1-only baseline could not, since N=1 never loops.

| cell | near% | avg loops | easy | hard | **hard−easy** |
|---|---:|---:|---:|---:|---:|
| baseline (N=1) | 45.0 | 1.00 | 1.00 | 1.00 | 0.00 |
| sanity (N=8, T=1.1) | 45.0 | 1.00 | 1.00 | 1.00 | 0.00 |
| N=2 T=0.1 | 40.0 | 1.97 | 1.96 | 1.97 | +0.01 |
| N=2 T=0.3 | 35.0 | 1.64 | 1.75 | 1.60 | −0.15 |
| N=2 T=0.5 | 40.0 | 1.24 | 1.27 | 1.23 | −0.05 |
| N=2 T=0.7 | 35.0 | 1.02 | 1.00 | 1.02 | +0.02 |
| N=4 T=0.1 | 40.0 | 3.91 | 3.94 | 3.90 | −0.04 |
| N=4 T=0.3 | 35.0 | 2.86 | 3.23 | 2.73 | **−0.50** |
| N=4 T=0.5 | 40.0 | 1.71 | 1.80 | 1.67 | −0.13 |
| N=4 T=0.7 | 50.0 | 1.04 | 1.00 | 1.06 | +0.06 |
| N=8 T=0.1 | 25.0 | 7.79 | 7.81 | 7.78 | −0.02 |
| N=8 T=0.3 | 25.0 | 5.25 | 5.96 | 5.07 | **−0.89** |
| N=8 T=0.5 | 15.0 | 2.55 | 2.47 | 2.57 | +0.10 |
| N=8 T=0.7 | 35.0 | 1.10 | 1.00 | 1.13 | +0.13 |

The N=16 row and N=8's remaining cells were killed deliberately — the conclusion was already
determined and the sign is consistent across N=2/4/8 (`sweep_after_fix_partial.json`).

**1. The mechanism works.** Recall holds 35–50% at N=2 and N=4 versus **exactly 0.0% at every
N>1** pre-fix, and the loop count is genuinely tunable (T=0.3 → 1.64, T=0.5 → 1.24). The
pre-fix finding that there was no usable middle range — "low T always maxes out, high T never
loops" — is **superseded**: that was a symptom of entropy being computed on corrupted logits.

**2. Early exit's premise fails, and it fails backwards.** The whole idea is that hard tokens
earn more compute. Measured, hard tokens get *less*: seven of twelve looping cells are
negative, and — the part that matters — **every large effect is negative** (−0.89, −0.50,
−0.15) while every positive is trivial (≤ +0.13, all in cells that barely loop at all).
Wherever looping actually happens, the gate spends compute on the wrong tokens.

Mechanistically this is coherent, not anomalous. "Hard" here means the model's previous greedy
prediction missed. The entropy gating the exit is over the **next** token. After a surprising
token the model often lands somewhere formulaic and commits confidently → exits early. After a
predictable token it can sit at a genuine branch point with many plausible continuations →
high entropy, keeps looping. **Entropy over the next token measures branchiness, not difficulty
of the current one**, and those are different quantities. No threshold repairs that, and it is
scale-independent.

**3. Heavy looping costs recall.** N=8 reads 25% / 25% / 15% against the 45% baseline —
consistent with the residual-stream drift measured earlier (‖h‖ 351→2560, 7.3× over 8 loops).
Looping is roughly neutral at N=2–4 and actively harmful beyond. Combined with (2), there is no
configuration that both loops meaningfully and helps.

**4. Throughput numbers here are unusable.** N=4/T=0.3 reads 10.80 tok/s on *fewer* loops than
N=4/T=0.1's 17.02 — physically impossible, so single-generation timing is noise-dominated. The
apparent 1.7× at N=2/T=0.7 comes from averaging 1.02 loops, i.e. from not looping, not from
smart gating. Any real throughput claim needs repeated timed runs.

**Verdict, and how it differs from the retracted one.** Entropy-gated early exit does not work
on this model — but this time for a *measured property of the model*, on a mechanism proven
sound by a known-answer cell, rather than as the shadow of a bug. Distinguish this from the
retraction above: same shape of answer, entirely different epistemic standing.

**Still genuinely open:** the four training runs (deep supervision, gated, fixed-N) all trained
against corrupted memory, so their failures carry no information about the recipes themselves.
Anything re-run there should carry a known-answer cell in its own harness before GPU is spent.

## ALBERT weight sharing — fewer params AND better quality, measured (2026-08-02)

The "small model, big-model quality" question this project is actually built around had never
been tested. Issue #122's looping work looped a *full-depth* model, which saves zero params;
issue #32 Phase 2.4 shipped `SharedWeightHVN` and measured a 3.62× param reduction but never
paired it with a quality number. This closes that gap.

**Setup.** `eval/scaling_laws.py` gains two arms — `EdgeHVN-DeltaNet` (dense) and
`EdgeHVN-DeltaNet-Shared` (`shared_weights=True`). Same class, same blocks, sharing is the only
variable. WikiText-103, BPE-8000, 5M-token compute-matched budget, **12 layers**, bf16.
`param_count` uses `sum(p.numel())`, which counts a shared block once, so reported params are
the real storage footprint.

| d_model | arm | params | eval PPL |
|---:|---|---:|---:|
| 256 | dense | 12,018,032 | 194.8 |
| 256 | **shared** | **4,763,996** | 251.6 |
| 512 | dense | 39,756,656 | 120.7 |
| 512 | **shared** | **10,830,684** | 156.1 |

**Headline, and it needs no extrapolation — two real measured configs:**

    shared d=512 : 10.83M params, PPL 156.1
    dense  d=256 : 12.02M params, PPL 194.8

**The shared model has ~10% FEWER parameters and ~20% BETTER perplexity than the dense one.**
That is the goal stated in the README's "The Dream" — small footprint, quality of something
bigger — demonstrated directly rather than argued.

**Why it works.** The PPL cost of sharing is **+29% at both widths** (194.8→251.6 and
120.7→156.1), strikingly stable. But the param saving **grows** with width: 2.52× at d=256,
3.67× at d=512. A fixed quality cost against a growing param saving means **the trade improves
as the model scales** — the opposite of the compounding-loss worry. Against an equal-param
dense model (via this repo's own −0.504 exponent): shared is ~19% better at d=256 and ~33%
better at d=512.

**This also corrects a live misconception.** HVN is not "much worse than a Transformer" —
§8.12 has HVN-DeltaNet at 159.4 vs Transformer 172.0 at d=512, and 129.4 vs 150.8 at d=768,
with a steeper exponent (−0.504 vs −0.406). Sharing therefore compounds *onto a winning
baseline*, not a deficit. The variant that genuinely loses to a Transformer is the pre-#34
default HVN block (194.3 vs 150.8 at d=768).

**A design error caught cheaply, worth recording.** The first sweep was configured at 2 layers
(copying the documented width-sweep configs) and showed only a 7% param saving — sharing scales
with **depth**, not width, and at 2 layers there is barely anything to share, with the
8000-vocab embedding dominating the model anyway. A 90-second smoke test exposed it before any
real GPU was spent. The known-answer check here was **param arithmetic** — dense at d=512 was
predicted at ~39.7M and measured 39,756,656 — which is exactly the check that would have caught
the 2-layer flaw upfront had it been done first.

**Caveats.** 5M tokens is undertrained, so the absolute PPLs are high and only the *comparison*
carries signal. Two width points, single seed, one depth. Whether the +29% constant and the
growing param saving hold at a full training budget or at 1B+ scale is **untested** — but the
direction is now measured rather than assumed, and it favors sharing.

**Next, if pursued:** compose sharing with ternary (BitNet b1.58, `hvn/edge/ternary.py`) — 3.67×
from sharing on top of ~10× from ternary is the actual path to the README's ~2 GB 10B-class
target, and neither multiplier has been measured in combination with the other.

## Sharing × ternary compose cleanly — ~70× storage for +54% PPL (2026-08-02)

Sharing and ternary had never been combined anywhere in this repo. They are orthogonal
levers — sharing cuts the NUMBER of weights (3.67× at d=512/12L), ternary cuts BYTES PER
weight (fp32 → 1.58 bits, ~20×) — so their storage benefits multiply arithmetically. The
open question was whether their *quality* costs also compose, or compound worse.

Reason to expect compounding: a shared block's weights are reused at all 12 virtual layers,
so quantization error in that one block is applied 12 times rather than once.

2×2 at d=512, 12 layers, 5M-token budget, WikiText-103/BPE-8000
(`results/weight_sharing/ternary_2x2.json`):

| arm | PPL | vs dense-fp |
|---|---:|---:|
| dense fp | 119.4 | — |
| shared fp | 154.1 | ×1.290 |
| dense ternary | 143.2 | ×1.200 |
| **shared + ternary** | **183.8** | ×1.540 |

**Clean composition predicted 119.4 × 1.290 × 1.200 = 184.8; measured 183.8 — within 0.6%.**
The prediction was recorded before the final arm completed, so this is a real test rather
than a post-hoc fit. **The costs compose multiplicatively with no compounding penalty** — the
12×-reuse mechanism does not bite.

**Consequence: ~70× storage reduction (3.67× params × ~20× bits) for +54% PPL.** That is the
concrete path to the README's edge target, with both multipliers now individually measured
and their combination verified rather than assumed.

Harness consistency check passed: this run's dense-fp (119.4) and shared-fp (154.1) reproduce
the previous run's 120.7 / 156.1 at the same config, within ~1.5%.

**Caveats.** 5M tokens is undertrained and likely *understates* ternary specifically, since
QAT normally needs longer to converge — the +20% ternary cost is probably a pessimistic
bound. Param counts are unchanged for the ternary arms (fp32 shadow weights are retained for
the STE gradient during training); the storage win is arithmetic, realised post-`.freeze()`.
Single seed, one width, one depth.

**A mid-run misread worth recording:** at the 3M-token mark dense-ternary was reported as
tracking *better* than dense-fp (159.2 vs 170.8), which would have been suspicious — ternary
should not beat full precision. It was an artifact of comparing mid-trajectory points across
arms that converge at different rates; the final numbers (143.2 vs 119.4) show ternary
behaving normally. Do not compare arms mid-run.

## Issue #109 re-examined — the trunk, not the 3rd attention layer, was the larger factor (2026-08-02)

Issue #109 attributed a 3-attn recall regression to "an architecture question (layer
count/placement)" and closed without merging. Re-checking the code showed the comparison was
**confounded**: it measured **3-attn on the 102K trunk** against **2-attn on the 51K trunk**,
changing two variables. The writeup noticed the trunks differed ("stronger trunk + lower loss,
worse recall") but read the stronger trunk as making the regression more *surprising* rather
than as a candidate *cause*.

**The control already existed on disk and had never been probed** —
`checkpoints/hybrid_a4_2attn_102k/` is 2-attn on the 102K trunk, i.e. the PASS architecture
with the regression's trunk. All three arms, `eval/synthetic_recall_probe.py`, 200 trials each:

| arm | near-window recall |
|---|---:|
| 2-attn / 51K trunk (PASS baseline) | **37.0%** (74/200) |
| 2-attn / 102K trunk ← *the missing control* | **25.0%** (50/200) |
| 3-attn / 102K trunk | **15.5%** (31/200) |

The baseline reproduces the documented 35.5% at 37.0%, so the harness is sound.

- **Trunk effect** (holding attn=2): 37.0 → 25.0 = **−12.0pp**
- **Layer effect** (holding trunk=102K): 25.0 → 15.5 = **−9.5pp**
- Both significant at n=200 (~2.6 SE and ~2.4 SE)

**Verdict: right decision, incomplete root cause.** A 3rd attention layer does genuinely hurt
(15.5% vs 25.0% on the same trunk), so reverting to 2-attn was correct. But #109 assigned the
entire −21.5pp drop to layer count when the trunk accounted for more of it than layer count
did. Its stated root cause is incomplete, not wrong.

**The larger finding, and it was never documented at all: doubling pretraining made
recall-after-FT WORSE.** 51K trunk → 37.0%, 102K trunk → 25.0%, architecture identical. A
more-converged recurrence-only trunk appears harder to bolt fresh attention onto — consistent
with reduced plasticity in a further-converged model, though the mechanism is not established
here. Five `hybrid_a4_2attn_102k` variants (base, chatmix, chatmix5k, hilr, noternary) were
run and none were written up; `grep "102k" docs/` returned nothing before this entry, which is
why it sat undiscovered.

**Consequence for scaling plans.** This is direct evidence against "pretrain recurrence-heavy,
fine-tune windowed attention in later" as a large-scale recipe: more pretraining can degrade
the attention-insertion step. The trunk-length/recall relationship should be characterised at
small scale before any large pretrain commits to that ordering — a cheap sweep (probe several
trunk checkpoints through the same short A4 FT) would settle it.

**Process note.** Both this and the #122 state-commit bug were found the same way: checking
whether a comparison changed one variable or two. That question is cheaper than any of the GPU
time it recovered.

## GHOST Phase 1 — the detector works; the documented 6K-7.5K "phase transition" does not exist (Issue #125, 2026-08-03)

`ghost/detector.py` (epic #124, method by Rambarun Komaljeet) reads a training-loss CSV,
smooths it, differentiates it, and flags stretches where the gradient sits `k` sigma below its
trailing-window mean — sustained acceleration, i.e. a phase transition. Phase 1's gate was:
correctly identify the transition this file documents at **steps 6000-7500**.

**The gate FAILS, and the detector is not the reason. The ground truth is.**

At the spec defaults (window=50 samples, k=2.0) the detector finds **0 transitions across all
102,000 steps** of `checkpoints/edge_chinchilla83m/training_log.csv`, while correctly finding a
planted transition in synthetic data and correctly reporting 0 on a smooth-decay control.
The 80-point sensitivity sweep (`results/ghost/detected_transitions.json`) is what makes this
conclusive rather than a tuning argument:

| setting | detections on the real log | hits 6K-7.5K | detections on a **pure-noise** control |
|---|---:|:---:|---:|
| window=20, k=1.5, open_run=1 | 92 | yes | **90** |
| window=35, k=2.0, open_run=1 | 35 | yes | **37** |
| window=100, k=1.5, open_run=1 | 71 | yes | **77** |
| window=35, k=1.5, open_run=5 | **1** (at 42550-42800, 2.17 sigma) | no | **0** |

**Every setting that "detects" the 6K-7.5K transition finds just as many transitions in a
smooth synthetic curve that has none.** The hit is noise, not signal. The one setting with a
real signal-to-noise separation finds a single weak candidate at ~42.5K and nothing at 6K-7.5K.

**Independent corroboration from held-out val loss** (`val_log.csv`), which needs no detector at
all — the improvement rate is *decelerating* monotonically straight through the claimed window:

| val step | 2000 | 4000 | 6000 | 8000 | 10000 | 12000 |
|---|---:|---:|---:|---:|---:|---:|
| val loss | 5.204 | 4.893 | 4.738 | 4.634 | 4.563 | 4.504 |
| delta per 1k steps | — | −0.156 | −0.077 | −0.052 | −0.036 | −0.029 |

A phase transition is an *acceleration*. Across 4K→8K the rate roughly halves at every
checkpoint. Nothing accelerates.

**Correction to this document.** The "EdgeHVN 83M — the measured PPL picture" section above
states the run had "a mid-run flat patch ~step 3K-5.5K that then **broke through and
accelerated** ~step 6K-7.5K — a phase transition, the opposite of a ceiling." That reading is
**retracted**. There is a mild flat patch (nearer ~3.3K-4.3K in a 500-step rolling mean), but
the descent that follows it is slower than the descent that preceded it, not faster. The
underlying conclusion that section drew — *undertrained, not capacity-limited* — is unaffected
and still stands on the val-PPL trajectory (110 → 63.45). Only the "phase transition"
characterisation was wrong: it was a narrative read of a noisy train-loss trace whose
sample-to-sample sigma (~0.15 nats) is larger than the entire descent it rides on (~0.1 nats
per 1,000 steps by step 6K).

**What this means for GHOST (#124).** The method's premise is that training consists of
predictable stretches punctuated by phase transitions that must be computed for real. This
project's only long training run contains **no detectable phase transition at all** at a
threshold that doesn't also fire on noise. That is not a refutation of GHOST — it is arguably
the *best possible* case for the extrapolator (a curve with no transitions is entirely
predictable stretch), and it removes the detector from the critical path. But it does mean the
detector cannot be validated against this repo's data, and Phase 3's "disable the extrapolator
near a transition" branch would never fire on a run like this one. Decide at #124 whether to
gate Phase 2 on a different (deliberately transition-rich) run or proceed straight to the
extrapolator. **The `2 sigma above rolling mean` rule itself is sound and reusable** — with the
persistence requirement (`open_run`) added, without which a bare k-sigma test produces
`n * Phi(-k)` false positives by construction (28 on a 1,200-sample noise curve, measured).

## GHOST Phase 2 — weight-delta extrapolation is measurably worse than doing nothing (Issue #126, 2026-08-03)

`ghost/extrapolator.py` (epic #124, method by Rambarun Komaljeet) takes three consecutive
checkpoints, measures each weight tensor's delta consistency, and predicts the weights N steps
later so those N steps need not be computed. Phase 2's gate: predicted weights at **N=10**
within **5%** of the real trained weights' loss at the same step.

**The spec's gate PASSES. The binding gate — beating the null control — FAILS at every N.**

The measurement needed a checkpoint series this repo did not have: its own checkpoints are
spaced 500 steps apart, far coarser than the N=10..500 jumps in question, so
`ghost/harness.py --make-series` continued `edge_chinchilla20k/step10000` for 800 steps
(AdamW, constant LR, batch 4 x seq 512, 1.59 steps/s, 8.4 min) and snapshotted the 11 steps the
analysis actually reads. Anchor at step 10200 (200 steps of warmup so any data-mix transient
has settled), extrapolator inputs at 10180/10190/10200, all arms scored on **one fixed
16-batch held-out probe set**.

| N | real loss | predicted | frozen (null control) | pred err | frozen err |
|---:|---:|---:|---:|---:|---:|
| 10 | 4.14036 | 4.22853 | 4.13901 | +2.13% | **+0.03%** |
| 50 | 4.10952 | 5.36871 | 4.13901 | +30.64% | **+0.72%** |
| 100 | 4.10046 | 7.43693 | 4.13901 | +81.37% | **+0.94%** |
| 500 | 3.98804 | 25.75799 | 4.13901 | +545.88% | **+3.79%** |

At N=10 the prediction is within 5% of real — and **doing nothing is ~70x more accurate**
(0.03% vs 2.13%). This is precisely the failure mode the null control was added to catch: a
+/-5% band is satisfiable by any method that barely moves, because over 10 steps the weights
barely move. By N=500 the extrapolated model is destroyed (loss 25.8 vs 4.0).

**One genuinely useful positive finding: the trajectory is real, but only at short range.**
Median cosine between consecutive weight deltas, measured at four spacings from the same
anchor:

| delta spacing | median cos(D1, D2) | params passing the consistency gate |
|---:|---:|---:|
| 10 steps | **+0.474** | 99.67% |
| 20 steps | +0.251 | 66.13% |
| 50 steps | +0.063 | 31.03% |
| 100 steps | +0.026 | 0.06% |
| 500 steps (repo's own checkpoints) | −0.012 | 0.11% |

Weight motion IS directionally coherent at 10-step spacing and decorrelates to noise by ~100
steps. An earlier reading of this project's native 500-step checkpoints (median cos −0.012)
was simply the wrong measurement scale — worth recording, because it would have supported the
right conclusion for the wrong reason.

**Why coherent deltas still do not buy a jump.** cos ~ 0.47 means the observed delta is roughly
half signal, half noise. Extrapolation multiplies that delta by `f = N/gap`, so injected noise
grows linearly in N while the true drift it is trying to capture is tiny — over 10 real steps
the probe loss moves by 0.001. There is no horizon where the trade wins: short N has no signal
worth capturing, long N has no valid linear trajectory.

**The obvious repairs were tried and the failure is monotone, not a bad configuration.**

| gate | params extrapolated | N=10 error | vs frozen |
|---|---:|---:|---|
| cos>=0.30 + quadratic | 99.67% | +2.13% | loses |
| cos>=0.30, linear only | 99.67% | +2.13% | loses |
| cos>=0.70, linear only | 31.03% | +0.83% | loses |
| cos>=0.90, linear only | 0.06% | +0.03% | ties — it *is* frozen |

Prediction error is monotone increasing in how much gets extrapolated, and the limit of
tightening the gate is the frozen baseline itself. **Shrinkage/damping needs no separate
experiment: damping is equivalent to reducing effective coverage, and this table shows the
optimum coverage is zero.** The extrapolator either injects noise or does nothing.

**Optimizer-state policy (recorded per #126):** this codebase's `_save_edge_checkpoint` never
stores AdamW moments, so every resume — and therefore every GHOST jump — already starts with
cold moment estimates. Nothing is extrapolated for them. This is a real per-jump cost that
would compound in Phase 3, on top of the weight error above.

**Caveats, stated so the result is not oversold.** One anchor, one model (83M ternary EdgeHVN),
one optimizer, constant LR. A different regime — early training with large coherent drift, or
a much smaller LR — could in principle behave differently. But the mechanism that fails here
(noise amplified by `f` against a drift too small to matter) is not specific to this
configuration, and the 5%-gate-passes-while-losing-to-frozen result would repeat wherever the
loss curve is as flat as it is at step 10200.

**Verdict: Phase 2 fails its binding gate.** Building Phase 3's training loop (#127) on this
mechanism would wrap a control loop around a step that is measurably worse than not taking it.
Artifacts: `results/ghost/extrapolation_scores.json`, `results/ghost/scores_{nopoly,cos07,cos09}.json`.

### Oracle upper bound — an adaptive/learned GHOST cannot work either (#126 follow-up)

The natural next proposal is to make GHOST *adaptive*: train normally, watch how wrong the
predictions are, fit the prediction function, repeat until it is good enough. That hypothesis
is testable without building it, by computing the **oracle**: for each tensor, the scalar
`alpha` in `W_pred = W3 + alpha * f * D2` chosen by least squares **using the true future
weights**. No adaptive, fitted, meta-learned or scheduled variant can beat a coefficient
selected with knowledge of the answer, so this is an upper bound on the entire family.

| N | oracle (cheats) | frozen | real | oracle err | median fitted alpha |
|---:|---:|---:|---:|---:|---:|
| 10 | 4.17811 | **4.13901** | 4.14036 | +0.91% | **+0.465** |
| 50 | 4.26663 | **4.13901** | 4.10952 | +3.82% | **+0.120** |
| 100 | 4.28950 | **4.13901** | 4.10046 | +4.61% | **+0.058** |
| 500 | 4.19223 | **4.13901** | 3.98804 | +5.12% | **+0.012** |

**The oracle loses to doing nothing at every horizon.** Two consequences:

1. **The optimal adaptive coefficient collapses toward zero as the jump grows** (0.465 -> 0.012).
   A self-tuning GHOST converges on "predict almost nothing" — the frozen baseline — and it
   does so *faster* the further ahead it tries to jump. The intuition that a fitted function
   improves with horizon is exactly inverted by the measurement.
2. **Even at its best (alpha 0.465 at N=10, which is essentially the measured delta correlation
   of 0.474) it is still worse in loss than freezing.** The oracle minimises *weight-space*
   distance to the true future weights and still loses in *loss space*: the component of the
   update that past deltas can predict is not the component that reduces loss. The predictable
   part is drift the optimizer's own momentum already applies; the part that constitutes
   learning comes from gradients of data the model has not seen yet, and that information is
   not present in past weights at any fitting budget.

This closes the "make it adaptive" branch analytically rather than by exhaustion: the ceiling
is measured, and it is below the do-nothing floor. `results/ghost/oracle_scores.json`.

### Two corrections to the Phase 2 method, both raised by the maintainer — verdict survives both (#126)

The Phase 2 result above was measured with **10-step delta spacing** at an anchor (step 10200)
where the loss curve is nearly flat. Both choices were wrong, and both were challenged:

1. **"It needs to be PER STEP."** Correct, and it mattered. Weight-delta coherence measured at
   1-step spacing is **median cos +0.890**, against +0.463 at the 10-step spacing originally
   used. The block measurement was discarding roughly half the available signal. Full ladder
   (flat anchor): gap 1 **+0.890**, gap 2 +0.864, gap 5 +0.675, gap 10 +0.463, gap 20 +0.219.
2. **"At a flat curve we do not need GHOST."** Also correct, and more serious: where the loss
   barely moves, the frozen null control is a near-perfect predictor for free, so it wins for
   reasons unrelated to the method. That made the original test unfair and the verdict
   unearned. Re-anchored at **step 600**, where the curve drops ~10x faster (−0.7 to −1.2 loss
   per 1k steps vs ~−0.1 at step 10200) and where real training gains **0.48 loss over 500
   steps** while freezing falls **9.55%** behind — i.e. the null control now has real ground
   to lose.

**Per-step deltas, early steep anchor (step 600) — the fair test:**

| N | real | frozen (null) | GHOST | oracle (cheats) | oracle alpha |
|---:|---:|---:|---:|---:|---:|
| 1 | 5.49893 | **5.49368** | 5.50430 | 5.50324 | +0.892 |
| 2 | 5.49948 | **5.49368** | 5.52232 | 5.51626 | +0.830 |
| 5 | 5.49323 | **5.49368** | 5.61955 | 5.56444 | +0.700 |
| 10 | 5.47345 | **5.49368** | 6.03026 | 5.63838 | +0.526 |
| 50 | 5.40985 | **5.49368** | 128.8 | 5.91104 | +0.143 |
| 100 | 5.33866 | **5.49368** | 425.3 | 6.17017 | +0.074 |
| 500 | 5.01457 | **5.49368** | 9023.1 | 7.71704 | +0.020 |

**GHOST and the cheating oracle both lose to doing nothing at every horizon, in the regime
where GHOST is supposed to earn its keep, at the granularity that maximises the signal.** The
N=10 gate now fails the spec's own 5% band too (+10.17%), not just the null control.

**Why 90%-coherent motion still cannot be ridden.** Direction is highly predictable; magnitude
and curvature are not. Extrapolation scales the last delta by `f = N/gap`, so at gap 1 an N=500
jump applies 500x a single step along a direction that is only locally valid — the true path
curves away. The oracle's fitted coefficient shows this directly: it shrinks 0.892 -> 0.020 as
N grows, i.e. the best possible use of the observed direction is to follow it less and less,
converging on the frozen baseline. And even at N=1 with alpha 0.892 the oracle is still
marginally worse than freezing: the reliably predictable component of an update is the
momentum the optimizer already applies, while the component that actually reduces loss comes
from gradients of data not yet seen, which no function of past weights contains.

**Verdict unchanged, now earned rather than assumed.** The negative result stands at the
correct granularity and in the correct regime. Artifacts:
`results/ghost/scores_perstep_flat.json`, `results/ghost/scores_perstep_early.json`.

### Weight averaging — the positive result that came out of the GHOST work (#126 follow-up)

GHOST's premise was that the weight trajectory is coherent. #126 measured that it is (median
cos **+0.890** at 1-step spacing) but that *extrapolating* along it amplifies the delta's noise
by `f = N/gap` and loses to doing nothing at every horizon. **Averaging is the same observation
with the opposite arithmetic: it cancels the noise instead of amplifying it, keeping the shared
drift.** It skips no training step, so it does not violate the rule the extrapolation result
established — it asks only whether the weights already paid for are being used well.

Measured on the EXISTING `edge_chinchilla20k` checkpoints (500-step spacing, no new training),
same fixed 16-batch probe set, three independent anchors:

| anchor | latest | avg k=2 | avg k=3 | avg k=5 | avg k=10 | worth (real steps) |
|---:|---:|---:|---:|---:|---:|---|
| 8,000 | 4.30812 | **4.24677** | **4.26304** | **4.25888** | — | < 500 |
| 10,000 | 4.18270 | **4.14715** | **4.12614** | **4.12857** | 4.20883 (worse) | ~500-1000 |
| 14,000 | 4.01125 | **3.98420** | **3.98165** | **3.99291** | — | ~1000 |

**9 of 9 (anchor x k in 2,3,5) improve on the latest checkpoint**, gains 0.018-0.061 nats, for
the cost of loading a few checkpoints and averaging them. `k=10` (a 4,500-step window) is
*worse* than the latest — the averaging window has to be short relative to the drift, otherwise
it mixes in weights from a materially different model.

Because the gain is roughly constant in loss terms while real training slows down, **the gain
is worth more steps the later it is applied**: <500 steps at anchor 8K, ~1,000 steps at 14K.

**The important caveat — this is an evaluation-time gain, not yet a proven training speedup.**
Every number above evaluates an averaged model. The untested question is whether training
*continued from* the averaged weights stays ahead, or whether the advantage washes out within a
few hundred steps. That is the difference between "free better checkpoint" (certain, measured)
and "free training steps" (plausible, unmeasured), and it is the one experiment that decides
whether this is worth wiring into the training loop.

Artifacts: `results/ghost/weight_averaging.json`, `weight_averaging_{8000,14000}.json`.

#### A/B: the averaging advantage SURVIVES continued training (#128)

The open question above — free better checkpoint, or free training steps? — resolved by a
three-arm A/B from anchor step 10,000. All arms replay a byte-identical, pre-materialised
1,000-batch sequence from a freshly constructed optimizer, so the ONLY difference is the
starting weights. Third arm is a matched-norm control: the anchor plus Gaussian noise scaled
per tensor to the exact displacement averaging applied.

| step | latest | averaged | perturbed (control) | averaged's lead |
|---:|---:|---:|---:|---:|
| 0 | 4.18270 | **4.12614** | 4.32291 | +0.0566 |
| 200 | 4.08988 | **4.06539** | 4.13684 | +0.0245 |
| 500 | 3.95263 | **3.92361** | 3.96371 | +0.0290 |
| 700 | 3.89700 | **3.86775** | 3.90928 | +0.0293 |
| 1000 | 3.85241 | **3.82978** | 3.86580 | +0.0226 |

**Averaged leads at all 11 evaluation points; the perturbed control trails at all 11.** The
control is what makes this interpretable: an identical-magnitude *random* displacement makes
the model worse immediately (4.183 -> 4.323) and never recovers the gap, so the gain comes from
the direction averaging selects, not from perturbation acting as noise injection.

**It is a real head start that partially decays, not a compounding multiplier.** 40% of the
initial advantage is retained at 1,000 steps. In training-step terms: the averaged arm starts
~500 steps ahead and is still ~100-150 steps ahead after 1,000 steps of training (using the
latest arm's local rate of ~0.018 nats/100 steps at that point).

**Practical read:**
- **Certain:** applying it once at the end of any run gives a strictly better final model for
  seconds of arithmetic. There is no reason not to do this.
- **Measured:** it is also a genuine, if modest, training speedup — the head start does not
  wash out within 1,000 steps.
- **Unknown:** whether applying it repeatedly during a run compounds, or whether each
  application just re-buys the same decaying ~0.05 and the run converges to the same place. That
  is the question that decides whether it belongs *inside* the training loop or only at the end.

**Caveats:** one anchor, one seed, one A/B. The absolute gaps (~0.02-0.06 nats) are small,
though the sign is consistent across 11/11 paired evaluations on an identical probe set, and
the control separates cleanly in the opposite direction. `results/ghost/ab_continue.json`.

#### Periodic in-loop averaging does NOT compound — keep it at the end of the run (#128)

Two arms from step 10,000, 2,000 steps each, identical pre-materialised batch sequence.
``baseline`` trains plainly; ``periodic`` replaces its live weights with the mean of the last 3
snapshots every 250 steps and continues from there. The reference that matters is the third
number, free from the baseline arm: its own last-3 snapshots averaged at the end — the
already-proven one-shot application.

| arm | loss @ 2,000 steps | vs baseline |
|---|---:|---:|
| baseline, no averaging | 3.70874 | — |
| **baseline + end-of-run average** | **3.66057** | **−0.04817** |
| periodic averaging in-loop | 3.72787 | **+0.01914 (worse)** |

**Periodic averaging is worse than not averaging at all**, and 0.0673 worse than applying it
once at the end.

**The mechanism, which is the interesting part.** Every injection helped, and the gains did not
diminish: **+0.0243, +0.0405, +0.1401, +0.0732, +0.0615, +0.0646**. The averaged point is
genuinely better each time. But immediately BEFORE each injection the periodic arm was behind
the baseline — by 0.035 (step 1000), 0.056 (1250), 0.061 (1500), 0.054 (1750). Training onward
from an averaged point descends more slowly than training from the raw trajectory, and the
injections spend their gains repairing a deficit they created. The largest single gain (+0.1401)
came where the trajectory had bounced upward, confirming these injections act mainly as
volatility damping, not as progress.

This is the same 40%-decay observed in the #128 A/B, seen from the inside: averaging buys a
better point *now* at the cost of a slower descent afterwards. One application at the end
collects the benefit and pays none of the cost, which is exactly why the end-of-run form wins.

**Method flaw found mid-run and fixed, disclosed rather than buried.** The arms were seeded once
before the data loader rather than once per arm. The model has dropout, so the second arm drew a
different mask sequence and drifted from the first *before any injection*: 0.0036 apart at step
250, 0.0041 at step 500. Both gaps above (0.019 and 0.067) are 5x and 17x that drift, so the
verdict is unaffected — but `torch.manual_seed` is now called per arm in both `repeat_average`
and `ab_continue`. **The #128 A/B reported earlier carries the same flaw**; its margins
(0.011-0.057, averaged ahead 11/11, control behind 11/11) also exceed the drift, so its verdict
stands, but it should be re-run under per-arm seeding to remove the asterisk.

**Settled recommendation: average the last 2-5 checkpoints once, at the end of a run.** Do not
put it in the training loop. `results/ghost/repeat_average.json`.

### 3 parallel networks + weight averaging loses to one network at 3x batch (#130)

Maintainer proposal: train K networks side by side, average their weights periodically, and
hand the average back to each perturbed (`avg+d`, `avg`, `avg-d`) so they stay diverse. This is
a real family of methods — Federated Averaging / Local SGD for the averaging, Population-Based
Training for the perturbation — but its benefit is normally *parallelism across devices*. The
question here is whether it helps on ONE card, where 3 networks each get a third of the
throughput.

The honest comparison holds **total tokens constant**: 3 nets at batch 4 vs 1 net at batch 12,
same 1.8M tokens, byte-identical batch sequence (900 batches of 4 materialised once; the single
arm concatenates each group of 3). Averaging every 10 rounds, ensemble scored on its **averaged**
weights, per-arm seeding, same fixed 16-batch probe set used throughout #126/#128.

| round | `single` (1 net, batch 12) | `avg3` (3 nets, batch 4) | gap |
|---:|---:|---:|---:|
| 50 | **4.08878** | 4.13034 | 0.042 |
| 100 | **4.02920** | 4.08133 | 0.052 |
| 150 | **3.95435** | 4.02212 | 0.068 |
| 200 | **3.91598** | 3.98804 | **0.072** |

**The single network wins at every checkpoint and the gap widens monotonically.** Run stopped at
round 200 of 300 on a pre-declared trigger (stop if the gap widens again at 200), not after
seeing a result that suited a conclusion.

**Interpretation.** Averaging K models trained from a common point on different batches
approximates one model trained on the K-times-larger batch — so `avg3` should have tracked
`single` closely. It does not: it falls steadily behind. The three networks diverge between
averaging points by more than the averaging recovers, and 10 rounds of independent drift is
already too much at this scale.

**One prediction of mine that the data refuted.** I expected the ensemble to lose badly on
wall-clock (3x as many forward/backward passes, plus 20 averaging operations copying 3 x 332 MB
state dicts). It did not: 12.7 min vs 13.0 min at round 200 — **near parity, marginally
faster**. Batch 12 at seq 512 is evidently bandwidth-bound on this card, so three batch-4 passes
cost about what one batch-12 pass costs. The verdict rests entirely on quality; the compute
overhead I predicted would dominate does not exist here.

**`avg3_delta` (the +/-d variant) was deliberately NOT measured.** It is `avg3` plus a
perturbation, so it would have to beat `avg3` *and* then close a 0.072 gap that was still
widening — against the strictly more favourable of the two variants. #128's matched-norm
perturbation control is the relevant prior: perturbing averaged weights cost 0.14 and never
recovered. Recorded as a stated decision, not an omission; the harness supports it
(`--parallel-ensemble`) if it is ever worth 20 minutes.

**Where this DOES apply: multiple GPUs.** With 3 cards the arithmetic inverts — the ensemble
arms run concurrently instead of sequentially, and this becomes standard data-parallel training,
which this repo already has (#65 DDP/FSDP). Nothing here argues against that; it argues against
paying 3x the memory for it on one card.

### Window fit — averaging the direction makes it WORSE, and that closes #124 (#126 follow-up, 2026-08-05)

The one objection #126 never answered. Its oracle proved that no method can win by choosing
*how far* to travel along the last delta — but it never left that direction. The maintainer's
challenge was the right one: fit the *function* of the motion from a window of observed steps
rather than extrapolating a single noisy delta, and do it where the premise holds (settled
training, not the huge-PPL beginning). A window fit averages the per-step deltas, and averaging
is the arithmetic that CANCELS their noise rather than amplifying it — the mechanism behind the
one positive result in this epic (#128). If a denoised direction were usable, the whole family
would reopen.

`ghost/window_fit.py` streams a least-squares polynomial fit of each weight tensor against step
number. The window is never stored: the normal equations need only the scalar moments `sum(u^p)`
and tensor accumulators `sum(u^p * W_u)`, so a K-step window costs `degree+1` tensors instead of
K checkpoints, and one accumulator set serves every degree. Accumulation runs on `W - W_ref`,
never raw weights — per-step motion is ~1e-5 of a weight's magnitude, close enough to fp32
epsilon that summing raw weights would grind away part of the signal being fitted. Verified
against synthetic linear/quadratic trajectories: prediction error 4-7 orders below the motion,
Gram solve in float64 (cond 4.6e2 at K=20/d=1 to 1.0e7 at K=50/d=2).

Measured at the **settled anchor step 10,200** — #126's primary anchor, `--warmup-steps 200`
from `edge_chinchilla20k/step10000`, AdamW constant LR 3e-4, batch 4 x seq 512, same fixed
16-batch probe set. Windows K in {20, 50}, degrees {1, 2}, two prediction modes
(`locked` = real anchor + fitted displacement; `fiteval` = the fit's own value, whose N=0 is a
smoothed anchor), plus `last_delta` (#126 at gap 1) and a per-arm oracle that rescales the
FITTED direction using the true future weights. 13 arms x 8 horizons.

**The test was fair: real training gains 0.236 nats over the 500-step horizon (4.11941 ->
3.88303), so the null control had real ground to lose.**

**Not one arm beats frozen at any horizon.** Best result anywhere is `oracle_w50_d1` at N=1,
4.12019 vs frozen 4.11941 — and that arm is allowed to see the answer.

**The finding, which inverts the hypothesis.** Averaging did not denoise the direction; it
destroyed it. Median cosine between predicted and true displacement at N=1:

| arm | velocity estimate | median cos |
|---|---|---:|
| `last_delta` | 1 step | **+0.890** |
| `w20_d2` | quadratic slope at the anchor | +0.628 |
| `w20_d1` | mean over 20 steps | +0.433 |
| `w50_d2` | quadratic over 50 | +0.340 |
| `w50_d1` | mean over 50 steps | **+0.166** |

Monotone in how much history is mixed in: **the more steps averaged, the worse the direction
points.** The noise that averaging cancels is not separable from the only locally-valid part of
the direction. A window mean estimates where the weights HAVE been going, which is not where
they are ABOUT to go.

**Why the window fit still beat `last_delta` in loss at large N — it moves less, not better.**
At N=500 `w50_d1` travels 3.3x the true distance and `last_delta` 7.5x (loss 15.9 vs 13,661).
That is a smaller wrong step. The repair for over-travel is exactly what the oracle does, and
its fitted coefficient collapses toward zero on the same signature as #126:

| N | oracle alpha (w50_d1) | oracle alpha (w20_d1) |
|---:|---:|---:|
| 1 | +0.293 | +0.561 |
| 10 | +0.164 | +0.349 |
| 100 | +0.014 | +0.038 |
| 500 | +0.006 | +0.010 |

**Secondary result — the averaging gain has a lower bound too.** The `fiteval` arm at N=0 is a
smoothed anchor over 20-50 steps, and it LOSES to the raw anchor (4.12430 vs 4.11941), where
#128's averaging over checkpoints 500 steps apart WON. #128 established the window must be short
relative to the drift; this adds the other side — tens of steps is too short to cancel anything
worth cancelling, and only lags the anchor. One anchor, recorded as an observation rather than a
theory.

**Verdict: #124 is closed, on mechanism rather than exhaustion.** #126 showed the step length
cannot be fixed; this shows the direction cannot be improved either, and that the best available
direction is the most recent delta — which is ridable for about one step. The part of weight
motion recoverable from weight history is drift the optimizer's momentum already applies; the
part that constitutes learning comes from gradients of unseen data. No fitting budget, window
length, polynomial degree or adaptive schedule changes that.

Artifacts: `results/ghost/window_fit.json`, `results/ghost/window_fit.log`. The 2.2 GB of target
checkpoints under `results/ghost/series_windowfit/` were deleted after scoring; the run is
reproducible in ~15 min via `python -m ghost.harness --window-fit`.

## EdgeHVN decode ran at half speed — ternary weights re-derived every token (Issue #131, 2026-08-05)

Opened to scope a MatMul-free integration (Zhu et al. 2024) into EdgeHVN's projection layers.
The **bounding profile that was meant to decide whether to build it found a 2x speedup already
sitting in the existing code**, and retired the original plan on the way.

**The mechanism.** `TernaryLinear.forward` called `ternary_quantize_ste(self.weight)` on every
invocation — `abs -> mean -> div -> round -> clamp -> mul -> sub -> add` across the **entire
weight matrix**. In training that cost amortizes over `batch x seq` (4 x 512 = 2,048 tokens per
forward). In incremental decode a forward pass **is one token**, so the same O(d^2) cost is paid
per token, against a matvec that is also O(d^2) — the same order as the work it feeds. The
weights are frozen at inference, so all of it was redundant.

**Evidence** (`scripts/profile_decode_projections.py`, 64 tokens, profiling `forward_step` and
NOT `forward` — the latter dispatches to `_scan_chunked`, a path serving never takes):

| signal | value |
|---|---|
| projection calls (102 modules x 64 tokens) | 6,528 |
| `aten::mean` calls | **6,656** |
| `aten::div` calls | **6,656** |

One absmean reduction and one divide per projection per token — 1:1, which is the diagnosis.
Op-level device time: `aten::addmm` (the actual GEMM) **66 ms** vs ~**395 ms** for the
quantization chain.

**A flaw in that profile, disclosed rather than buried.** Its `projection_share_pct` (70.1%) and
`PROCEED` verdict are **not trustworthy**: `key_averages()` returns both op-level (`aten::abs`)
and kernel-level (`_ZN2at6native...AbsFunctor...`) entries carrying the *same* device time — they
agree to the last decimal at 54,794.628 us — so summing across all events double-counts. The call
*counts* are unaffected, and the A/B below needs no attribution at all, which is why the verdict
rests on those instead.

**The fix, measured.** Cache the dequantized weight under `eval()` + `no_grad`. Paired
within-process A/B, run in **both arm orders**, because the baseline always running first would
let GPU clock ramp inflate the second arm:

| arm order | pre-#131 | cached | speedup |
|---|---:|---:|---:|
| legacy first | 22.30 | 43.21 | **1.94x** |
| cached first | 48.40 | 21.45 | **2.26x** |

Two earlier runs against a monkeypatched prototype read 2.02x and 1.99x. **Honest claim: ~2x**,
consistent across four measurements and both orders.

**Known-answer cell** (what the #122 retraction requires of every harness): caching a
deterministic function of frozen weights is mathematically a no-op, so the cached arm MUST give
bit-identical logits. `max|dlogits| = 0.000e+00` in all four runs. **No recall regression is
possible by construction** — identical numbers, so no probe re-run is needed to clear it.

**Do NOT quote the absolute tok/s.** The same arm read 21.45/22.30 (legacy) and
40.56/43.21/48.40/53.65/81.05 (cached, across runs) — a ~1.5x spread on identical code, matching
this repo's documented GPU-contention variance ("re-ran identical `main()` twice = 0.7 steps/s").
Only the within-process paired ratio is meaningful on this machine.

**Scope: serving only; no training win was available.** Training stays uncached — the STE needs
the live shadow weight in the autograd graph and the weight moves every step. Nothing here
touches the compute wall or the ~142M local ceiling. What changes: every EdgeHVN checkpoint
*served* (Docker chat app, `EdgeToolCallingSession`, any `forward_step` eval) decodes ~2x faster
with identical output.

**Staleness is the real hazard, and it is guarded.** A weight cache has the exact failure shape
of #117 (register-if-absent dedup silently served untrained weights while aggregate loss still
trended down; no test caught it). So invalidation is aggressive rather than clever: on **any**
mode transition (`train()` is the single entry point for both `.train()` and `.eval()`), on
`_load_from_state_dict`, on `freeze()`, and on any dtype/device mismatch.
`tests/test_issue131_ternary_cache.py` (9/9) asserts a specific weight's update survives a
train->eval round trip **directly**, not via aggregate behaviour. One boundary is documented, not
hidden: mutating a weight in place while *staying* in eval mode is unobservable to any hook and
needs an explicit `invalidate_cache()` — a test pins that boundary so changing it is deliberate.

**Consequence for the original MatMul-free plan.** It proposed replacing the projection GEMMs
with ternary accumulation — the ~66 ms slice; the ~395 ms slice was redundant work that simply
should not run. Also established while scoping: **ternarization is already shipped**
(`apply_ternary_to_model` converts every `nn.Linear`; per #129 every real run trains ternary ON),
and **no training-VRAM win is possible** from ternary QAT because the fp32 shadow weights stay
resident — so the original spec's training-VRAM and memory gates were unachievable by
construction. A fused ternary *kernel* remains open, but decode is launch-bound here (2,815
kernel launches/token, CPU/device self-time ratio 0.91), so a future attempt should target launch
count per the #43 precedent, not multiply count.

**Untested:** batch>1 / the `generate_served` concurrent path, bf16 (this checkpoint decodes in
fp32), and whether the win holds after `freeze()` (which replaces the 8-op chain with a cast+mul,
already cheaper — expect a smaller ratio). Artifacts:
`results/matmul_free/decode_profile.json`, `ternary_cache_ab{,_reversed,_landed,_landed_rev}.json`.

## Padding-free sequence packing — 5.17x cheaper per real token (Issue #132, 2026-08-06)

The fine-tune datasets padded every example to `seq_len+1` with EOS and `-100` labels. Padding
tokens contribute nothing to the gradient, so that compute was pure waste. Measured on the real
corpora with the real GPT-2 tokenizer at `seq_len=512`:

| corpus | n | median | p90 | max | padding waste |
|---|---:|---:|---:|---:|---:|
| `data/grounded_qa_expanded.jsonl` | 5,240 | 86 | 115 | 205 | **83.0%** |
| `data/tool_use_train.jsonl` | 600 | 180 | 340 | 340 | **63.6%** |

The grounded-QA corpus is the one behind the whole recall track — `hybrid_a4_v2` (161.6 min), the
#122 deep-supervised (115.3 min) and fixed-N (196.3 min) FTs all paid the 83%.

**This is not "fewer tokens needed to learn."** Same real tokens, same epochs — packing puts ~5.2
real examples in the row that held one, so the corpus costs 5.18x fewer rows.

**Measured A/B** (`scripts/ab_packing.py`, real 83M hybrid config — 12 layers / 2 windowed-attn /
d=512 / ternary / triton+megakernel scan — on the real corpus, `results/packing/ab_packing.json`):

| arm | s/step | real (non-`-100`) tokens/step | us/real token | peak VRAM |
|---|---:|---:|---:|---:|
| padded | 0.330 | 5 | 61,147 | 2.08 GiB |
| packed | 0.344 | 29 | 11,816 | 2.15 GiB |

**5.17x cheaper per real token**, against a 5.18x row-reduction ceiling — i.e. packing captures
essentially all of the available win. Corpus packing: 5,240 examples -> 1,012 rows, waste
81.4% -> 3.9%. The predicted cost of losing the flash `is_causal` fast path on the 2 attention
layers turned out to be **~4%** (0.330 -> 0.344 s/step), and VRAM is flat.

**Why the boundary reset is exact rather than an epsilon — the finding that made this cheap.**
Document isolation needs the DeltaNet recurrent state cut at each document's first token, which is
expressible as forcing the decay gate `alpha_t = 0` there. `alpha` is a plain tensor passed *as
data* into every scan, so **no kernel changed**. And `hvn/core/deltanet_triton.py`'s scan is
sequential over S with a direct `state = state * a_t` multiply ("fp32 throughout -- no
reparameterization, so stable at any alpha"), so `alpha=0` is a mathematically exact reset there —
measured 4.8e-7 packed-vs-separate max abs diff, against 0.60 for the leakage it removes. The
eager `_scan_chunked`/`_scan_chunked_stable` paths fold alpha into a cumprod/log-space form that
clamps at 1e-6 and *cannot* represent a hard zero; rather than ship a silently-approximate path,
`DeltaNetBlock.forward` **raises** when asked to pack under them. The FT scripts already default
to the Triton scan (2026-08-01 flip), so this costs nothing in practice.

**Three sequence-axis carries, all cut** (`hvn/core/packing.py` + the blocks):
1. **Recurrent state** — `alpha=0` at document starts.
2. **The depthwise causal conv** — its `K-1`(=3) backward taps read across a boundary. Fixed with
   a segment-safe conv (K masked shifted multiplies instead of one `conv1d`), only on the packed
   path. This one is easy to miss and would have leaked 3 tokens per boundary silently.
3. **Attention** — block-diagonal document mask AND-ed into the existing explicit-mask branch.

RoPE needed nothing (it is relative, and cross-document keys are masked anyway). A fourth issue
surfaced while building: the **causal shift** means the last token of document A would be trained
to predict document B's *first* token. The current corpora mask their first token anyway (the
prompt is `-100`), so it is a no-op on them — but `PackedDataset` cuts it explicitly so the class
is not silently corpus-dependent.

**Verification.** `tests/test_packing.py` 27/27, including per-document parity vs
`_scan_sequential`, an explicit cross-document isolation proof on EdgeHVN logits, a
non-vacuousness check (doc_ids must actually *change* the result), the conv-boundary test, the
grad-checkpoint recompute path, and GPU parity for both the triton and megakernel scans. Full
suite 309 passed (`tests/test_issue33.py` syntax error and 6 `ckpt_path`-fixture errors in
`test_issue51/65/67` are pre-existing on `main` — verified by stashing).

**Scope, deliberately narrow.** `--pack` is opt-in on `finetune_edge_conversational.py`,
`continue_ft_hybrid.py`, `continue_ft_looped.py`. Class-level defaults on `DeltaNetBlock`/`EdgeHVN`
are **unchanged** — the same lesson as the #122 scan-default flip, which surfaced two real
regressions when broadened. Each corpus packs independently so a row never mixes corpora, which
keeps the `WeightedRandomSampler` mix fractions meaning what the flags say. `HVNBlock` and
`shared_weights=True` raise `NotImplementedError` rather than leak.

**NOT yet done — the remaining acceptance criterion.** A real packed-vs-padded FT run re-gated on
`eval/synthetic_recall_probe.py` at the 300-trial bar (against the 35.5% near / 8.5% far
baseline). Packing changes gradient composition per step — ~5 examples per row instead of 1 — so
it is **not** a "same result, faster" change on paper, and a speedup that moves recall is not a
win. The isolation is proven exact; what is unproven is whether the denser step is as good a step.

## Packing recall re-gate — PASSES: 5.15x faster, no detectable recall cost (Issue #132, 2026-08-06)

The remaining acceptance criterion from the entry above. Two FT arms, identical except `--pack`,
matched on **real tokens** (same passes over the corpus, so the packed arm takes 5.18x fewer
optimizer steps -- which is exactly the risk being measured, not a flaw in the design), then
`eval/synthetic_recall_probe.py` at the 300-trial bar on both.

| arm | steps | wall-clock | near-window | far-window (W=128, 16 distractors) |
|---|---:|---:|---:|---:|
| padded (control) | 5,000 | **89.1 min** | 7.3% (22/300) | 3.7% (11/300) |
| packed | 965 | **17.3 min** | 6.0% (18/300) | 3.3% (10/300) |

**5.15x end-to-end wall-clock speedup** — confirming the 5.17x/real-token microbenchmark
(`scripts/ab_packing.py`) on a real 90-minute training run rather than a 30-step timing loop.

**Recall difference is not statistically detectable.** Two-proportion tests: near z=+0.65
(p=0.51), far z=+0.22 (p=0.82). Neither is close to significance at n=300 per cell. Packing costs
5.18x fewer optimizer updates and the model does not measurably care.

**PASS.** The concern going in was that packing is not a "same result, faster" change on paper --
~5 examples per row instead of 1 changes gradient composition, and 5x fewer updates at the same LR
schedule is less optimization. Measured: neither shows up above noise.

**Three caveats, stated because the absolute numbers look wrong next to this doc's other entries:**
- **The trunk differs.** `edge_conversational_step51000.pt` (what `hybrid_a4_v2` used) is not in
  this checkout, so both arms warm-started from `step101000`. Absolute numbers are therefore NOT
  comparable to the historical 35.5% near / 8.5% far -- which is precisely why **both** arms were
  measured fresh rather than one against the doc. The same mistake #122 had to correct.
- **`--chat-mix 0`** (pure grounded-QA, both arms) to keep the packing ratio exactly 5.18x and drop
  the OASST cache dependency. a4_v2 used a 30% chat mix.
- **Deep in the overfitting regime**: 5,000 steps x 32 rows over 5,240 examples is ~30 passes;
  final train loss 0.0084 (ppl 1.01). This depresses both arms equally and explains why 7.3% sits
  so far below 35.5%. The probe scores **invented** pseudo-words never in training, so it still
  measures the general recall skill rather than memorized pairs -- but a less-overfit run would be
  a better absolute measurement. The A/B remains controlled either way.

**What this licenses and what it does not.** It licenses using `--pack` on a grounded-QA-style FT
at this scale: the speedup is real and the quality cost is below detection. It does **not**
establish that packing is free at every mix/length distribution -- OASST conversations are longer
(smaller packing ratio, fewer documents per row), so both the win and any risk shrink there, and
that case was not measured. Artifacts: `results/packing/regate_{padded,packed}_{near,far}.txt`,
`scripts/regate_packing.sh`, checkpoints under `checkpoints/regate_{padded,packed}/`.

## OASST packing re-gate — the win shrinks to 1.45x and a small loss cost appears (Issue #132, 2026-08-07)

Closes the "not established at every length distribution" caveat directly above. OASST is the
other end of the distribution from grounded-QA: median 359 tokens with >=10% at the 512 cap,
so it packs **1.45 examples per row** (3,465 -> 2,395 rows, waste 32.9% -> 3.0%) against
grounded-QA's 5.18. Both the available win and the gradient-composition perturbation shrink by
the same factor, which is why this could not be inferred from the grounded result.

**Harness choice that matters.** Both arms ran through `continue_ft_hybrid.py --chat-mix 1.0`,
not `finetune_edge_conversational.py`. That script only enables the Triton scan under `--pack`,
so an A/B there would have confounded packing with a change of scan path — the #122 lesson.
`continue_ft_hybrid.py` has triton+megakernel on unconditionally in both arms, leaving `--pack`
as the single difference. `--chat-mix 1.0` gives the grounded corpus zero sampling weight, so
this is effectively pure OASST on the identical harness as the first re-gate. Arms matched on
**real tokens**: 2000 padded steps vs 1379 packed (= 2000 / 1.45, same passes over the corpus).

| arm | steps | wall-clock | final train loss (last 5 windows) | near-window | far-window (W=128, 16 distr.) |
|---|---:|---:|---:|---:|---:|
| padded (control) | 2,000 | **35.7 min** | 2.614 (ppl 13.65) | 1.3% (4/300) | 4.0% (12/300) |
| packed | 1,379 | **24.7 min** | 2.676 (ppl 14.53) | 3.0% (9/300) | 4.7% (14/300) |

**1.445x end-to-end wall-clock**, against the 1.45x row-reduction ceiling — packing again
captures essentially all of the available win, and per-step cost is unchanged (1.07-1.08 s/step
in both arms, so the ~4% flash-`is_causal` penalty measured on grounded-QA does not grow here).
The scaling law across both re-gates is simply *speedup = packing ratio*; nothing else about the
step got cheaper or dearer.

**Recall: no detectable difference, but the probe is a weak instrument on this run.** Near
z=-1.40 (p=0.16), far z=-0.40 (p=0.69) — both non-significant at n=300, and both nominally
favour the *packed* arm. The honest caveat is that this FT is 100% chat with zero grounded
replay, so both arms sit at or below the pure-trunk floor (4-5% near / 0.5% far): recall was
eroded by the chat FT itself, exactly as Stage 1 (#108) predicted, in both arms equally. A
floored metric can only detect gross damage, so "no difference" is weaker evidence here than
the same words were in the grounded re-gate. **This is why the loss column was added** — on a
chat corpus, next-token loss is the metric that actually tracks the thing being trained.

**The new finding: a small, consistent train-loss cost that the grounded re-gate could not have
seen.** At matched real tokens the packed arm is behind throughout, not just at the end:

| progress (real tokens) | padded | packed | gap |
|---|---:|---:|---:|
| 10% | 3.3134 | 3.3440 | +0.031 |
| 20% | 3.1012 | 3.1259 | +0.025 |
| 30% | 2.8959 | 2.9305 | +0.035 |
| 50% | 2.7288 | 2.7863 | +0.058 |
| 70% | 2.6575 | 2.7061 | +0.049 |
| 100% | 2.614 | 2.676 | **+0.062** |

A stable ~0.03-0.06 nat offset (**+6.4% ppl** at the end), flat-to-slightly-widening — not a
divergence, not noise. The loss is comparable across arms by construction:
`CrossEntropyLoss(ignore_index=-100)` with the default mean reduction is a per-**real**-token
mean in both, and padding is `-100`, so the packed row's extra real tokens change the variance
of the estimate, not its expectation.

**Most likely cause, and it is confounded on purpose: 1.45x fewer optimizer updates.** Matching
on real tokens means the packed arm takes 1379 updates where the control takes 2000, at the same
peak LR and a cosine schedule over its own (shorter) run. That is *the* thing packing trades away,
and it is inseparable from packing in this design — which is the correct thing to measure, since
it is exactly how the flag would be used. What this A/B cannot do is attribute the 0.06 to "fewer
updates" versus "denser rows are worse steps"; separating those needs a third arm matched on
*steps* instead of tokens. The grounded re-gate could not have surfaced this at all: its final
loss was 0.0084 (ppl 1.01, ~30 passes, deep in the overfitting regime), where every arm is
pinned to the floor.

> **⚠️ The decision this section originally reached — "`--pack` stays opt-in, because it costs
> ~6% chat ppl" — is RETRACTED.** That verdict rested on **train** loss, which in this regime
> measures how hard each arm memorised its own training set, not model quality. Measured on
> held-out data the next day, the gap is +0.4% (nothing), and `--pack` is now ON BY DEFAULT.
> See the correction entry below. The measured facts above stand; only the reading of them was
> wrong.

**Original decision (superseded):** keep `--pack` opt-in, on the grounds that 1.45x wall-clock
for ~6% worse chat ppl is a trade rather than a free win, with the #122 scan-default flip as the
precedent for not broadening a verified-narrow default.

**The untested option flagged here as "probably dominates both"** — spend the speedup on passes
instead of wall-clock — **was tested and does NOT dominate.** It overfits. Also in the correction
entry below.

Artifacts: `results/packing/regate_oasst_{padded,packed}_{near,far}.txt`,
`scripts/regate_packing_oasst.sh`, checkpoints under `checkpoints/regate_oasst_{padded,packed}/`.

## CORRECTION — packing is free, `--pack` is now ON BY DEFAULT (Issue #132, 2026-08-07)

**In one line: packing gives the same model in ~2/3 the time. The "6% worse" in the entry above
was the wrong yardstick.**

Two things were measured today that reverse the decision above.

**1. The equal-wall-clock arm (arm C) — spend the speedup on passes, not on time.** Packed at the
control's *step* count instead of its *token* count: 2,000 steps, `--pack`, everything else
identical (`scripts/equal_wallclock_packing_oasst.sh`). Because both runs use cosine over 2,000
with warmup 60, the LR matches at every logged step, so this is also the **step-matched third arm**
the entry above said was needed to separate "fewer updates" from "denser rows are worse steps".

**2. Held-out loss on all three arms** (`eval/heldout_grounded_loss.py`,
`scripts/heldout_loss_three_arms.sh`). Every arm ran `--chat-mix 1.0`, which sets the grounded
corpus's sampler weight to `1/n_g * (1 - 1.0) = 0` — literally never sampled ("0% grounded" in
every log) — and the shared trunk is a pure SlimPajama+Wiki pretrain that was never grounded-FT'd,
with fresh attention layers. So `data/grounded_qa_expanded.jsonl` is **equally unseen by all
three**, and scoring it is a real generalization measurement. Evaluation is padded and in fixed
order for every arm, so packing is absent from the measurement side.

| arm | steps | wall-clock | passes | train loss | **held-out loss** | held-out ppl |
|---|---:|---:|---:|---:|---:|---:|
| A · padded (control) | 2,000 | 35.7 min | 18.5 | 2.6139 | **4.8185** | 123.79 |
| B · packed, token-matched | 1,379 | **24.7 min** | 18.4 | 2.6759 (+6.4%) | **4.8221** | 124.22 (+0.4%) |
| C · packed, equal wall-clock | 2,000 | 35.8 min | 26.7 | **2.4871 (−11.9%)** | **4.8511** | 127.88 (+3.3%) |

**Finding 1 — B's "6% ppl penalty" is not a quality penalty.** On 493,837 held-out tokens B is
+0.4% against the control: nothing. Matching on real tokens gives the packed arm 1,379 updates to
the control's 2,000, so it fits the *training set* less — which is all the train-loss gap was
measuring. **Packing at the same real tokens is quality-neutral and 1.45x faster.** That is the
padded-vs-packed verdict.

**Finding 2 — C's −11.9% train-loss "win" is overfitting, and it inverts on held-out.** Best train
loss of the three, worst held-out (+3.3%). 26.7 passes over a fixed 3,465-example corpus buys
memorization, not generalization. **Spending the speedup on more passes over the same data does
not work** — the direction that would work is more *corpus*, which OASST cannot supply locally.
The pre-launch caveat ("1.45x more passes over a fixed corpus is not 1.45x more data") is exactly
what happened.

**Finding 3 — train loss was the wrong instrument for this whole question, in both directions.**
It made B look worse than it is and C look better than it is. The FT harness logs no validation
loss and every arm trains on all 3,465 OASST conversations, so there is no OASST val split; the
held-out grounded corpus is the available substitute. Worth remembering the next time an arm is
graded on its own training loss in a multi-pass regime.

**Recall probes (300 trials each) are uninformative here and were not used to decide**: near
1.3% / 3.0% / 3.0%, far 4.0% / 4.7% / 2.7% (A/B/C). This FT is 100% chat with zero grounded
replay, so every arm sits at or below the pure-trunk floor — a floored metric detects only gross
damage. Recorded for completeness, not weight.

**A scoring caveat, so the artifacts are not misread.** `heldout_grounded_loss.py` also has an
answer-span-only mode (the corpus as `GroundedQADataset` masks it, ~14K supervised tokens).
Those numbers — 10.749 / 10.914 / 10.687 — sit at or *above* `ln(50257) = 10.82`, i.e. at chance:
none of these checkpoints ever trained the grounded skill. That mode is retained but is
**uninformative for this comparison**; the `--all-tokens` figures in the table are the ones to
read.

**Decision: `--pack` defaults to ON** in `finetune_edge_conversational.py`,
`continue_ft_hybrid.py`, `continue_ft_looped.py` (`argparse.BooleanOptionalAction`; `--no-pack`
restores the padded path). Free 5.15x on grounded-QA, 1.45x on OASST, no measurable quality cost
at either end of the length distribution — which was the open question. Class-level defaults on
`DeltaNetBlock`/`EdgeHVN` remain **unchanged**: the #122 flip's two regressions (compile-fusion
graph breaks, BCD-offload parity) were about `use_triton_scan`/`use_megakernel_scan` as *class*
defaults, and none of these three scripts uses `--block-coordinate`, so that risk does not reach
them. `tests/test_packing.py` 27/27, `test_continue_ft_hybrid` + `test_hybrid_forward` +
`test_looped_early_exit` 22/22 after the flip.

**Limits, stated plainly.** One training seed per arm, so the 0.4% has no error bar — it is "too
small to see", not "proven zero". The held-out corpus measures transfer in the same chat template
rather than OASST fluency itself. Both cut against over-claiming a *quality* win; neither
threatens the actual claim, which is that packing is quality-**neutral** and faster.

Artifacts: `results/packing/heldout_alltokens.json`, `heldout_answerspan.json`,
`regate_oasst_packed2000_{near,far}.txt`, `logs/equal_wallclock_oasst.log`, checkpoints under
`checkpoints/regate_oasst_packed_2000/`.

## The 355M scale-up is STOPPED — measured, not abandoned (Issue #112, 2026-08-07)

**One line: the scale-up costs a month of GPU time for a 2-4x parameter jump, and the
setting it depends on makes the model 4x worse. Stopped on the numbers.**

This entry closes out a day of #112 pre-launch work. Nothing here was a guess; each item
replaced a documented belief with a measurement, and three of the four beliefs were wrong.

### 1. The VRAM envelopes measured a different model (see the correction in training.md)

Every number in `docs/training.md` traces to a bench hardcoding `VOCAB = 8000` against the
real pretrain's 50,257, with no ternary and no Triton scan. `d=1024, 24L` is **355,128,593
params, not 268M**, and it **spills** (6.34-6.75 GiB reserved on a 5.9966 GiB card) rather
than being the "recommended ceiling config, 4.3 GiB". The 4.3 figure was the forward/backward
peak with optimizer state excluded entirely.

### 2. The "1.21B silent stall" was never a stall

Output is gated on `global_step % 50` with `grad_accum=8`, so nothing prints until micro-batch
400; at 1.21B each micro-batch took 82 s because `reserved` exceeded the card by 1.29 GiB and
WDDM backed the overflow with host RAM. First log line was **9.2 hours away**. The recorded
TDR/driver-crash hypothesis for the earlier 268M failure is **refuted**: the Triton scan has no
autotuning (`grep -c triton.autotune` = 0), the `"Not enough SMs"` line prints on every compile
on a 20-SM card, no kernel is within 36x of the 2 s watchdog, and the Windows System log has
**zero Event ID 4101** across its entire retained history. The GPU did end up faulted and that
remains unexplained — but it is not what caused the "hang".

### 3. `--block-coordinate` costs ~4x quality — the load-bearing assumption nobody had tested

BCD is documented purely as a VRAM optimisation (#99), yet it is mandatory at 355M, and only
1 of 14 blocks trains at a time with the outgoing block's optimizer state deleted on every
switch. A/B through the real training script (`scripts/ab_bcd_convergence.sh`, d=512/12L so the
control fits without BCD, 4,200 steps = 3 full rotations, identical seed/data/steps, `--no-fast`
in both arms so `torch.compile` recompiles at block switches cannot confound it):

| step | full model | `--block-coordinate` | gap |
|---|---:|---:|---:|
| 1,000 | 5.53 | 8.79 | 3.27 |
| 2,000 | 5.18 | 6.46 | 1.28 |
| 3,000 | 4.97 | 6.26 | 1.29 |
| 4,000 | 4.85 | 6.20 | 1.35 |
| **4,200 (final)** | **4.83** (ppl 125) | **6.19** (ppl 487) | **1.36** |

The early gap is cycle warmup — one full rotation completes at step 1,400, before which most
of the model has never trained. What remains after that is flat and real: **~3.9x worse
perplexity at matched steps**. Limits: 74M/14 blocks rather than 355M/26, and one seed.

### 4. Adafactor fits without BCD — and had been built but never measured

`scripts/probe_optimizer_vram.py`, real 355M config, built through `build_model_and_config` +
ternary + Triton scan (**not** a standalone `EdgeHVN`, deliberately — repeating the bench's
mistake would produce another wrong envelope), reporting `max_memory_reserved` rather than
`nvidia-smi`:

| optimizer | peak alloc | peak reserved | s/micro | fits (5.9966 GiB card) |
|---|---:|---:|---:|:--:|
| Muon (script default) | 5.67 GiB | 6.75 GiB | 22.82 | ❌ spills |
| **Adafactor** | **3.97 GiB** | **5.08 GiB** | **1.82** | ✅ **0.92 GiB headroom** |
| AdamW | 6.61 GiB | 7.47 GiB | 29.11 | ❌ spills |

The s/micro column is the spill penalty, not optimizer speed — the two that fit nothing run
12-16x slower because of host-RAM backing. **`--optimizer adafactor` removes the need for BCD
at 355M entirely**, at the same step time as the BCD config, without the 4x handicap. It has
been wired since #85; that issue's own closing note records its VRAM benchmark being deferred
"since the GPU is currently busy", and it was closed without the benchmark ever running.

**Adafactor's own quality is still unmeasured** (Muon beat AdamW in #86; Adafactor vs Muon has
never been run) and it saves memory by approximating — a row+column summary per matrix instead
of the full second moment. The arm that would settle it is ~2.3 h against the control above.

### 5. Why the track is stopped anyway — it is time-bound, not memory-bound

At the measured Adafactor throughput, with Chinchilla-optimal tokens (20/param):

| config | real params | tokens needed | days (h=8) | days (h=16 projected) |
|---|---:|---:|---:|---:|
| d=1024, 24L | 355M | 7.1B | 73 | **30** |
| d=768, 16L | 172M | 3.4B | 17 | 7 |
| d=2048, 24L | 1.21B | 24.2B | **31 years** | ~2 years even if it fit |

**1B was never trainable here**, and "1B trains locally today" (this file, 2026-07-12) is
withdrawn — it fit only in the vocab-8000 bench, and even a corrected fit is ~2 years of wall
clock. `docs/index.md` already warned that size decisions must start from
`6·N·D ÷ throughput` and never from whether the model fits; the 1B claim did precisely what
that warning forbids.

**And the existing 83M is already complete** — 102,000 steps x 16,384 = **1.65B tokens against
a 1.66B Chinchilla floor**. It is not undertrained any more. A 5-hour from-scratch run at any
size lands at 0.7-6.6% of its required tokens, i.e. strictly worse than the model already on
disk. There is no short version of a pretrain.

**Decision: stop the scale-up.** The goal is a model trained on *less* compute; a 30-day run
for a 4x parameter jump is the opposite, and 172M/7 days buys a 2x jump that this project's own
capacity analysis says will not close the contextual-reasoning gap regardless. The cheap
remaining wins are fine-tuning (1-3 h, now 1.45-5x faster with #132 packing on by default),
which is where this document has repeatedly concluded the lever actually is. The next
compute-root direction is multi-token prediction — attacking the one-token-per-forward-pass
cost — explicitly deferred, not dropped.

Artifacts: `scripts/ab_bcd_convergence.sh`, `logs/ab_bcd_convergence.log`,
`checkpoints/ab_bcd_{control,blockcoord}/val_log.csv`, `scripts/probe_optimizer_vram.py`,
`results/optimizer_vram_355m.json`.
## Selective token training (Rho-1) — no gain at 83M (2026-08-07)

**One line: training on only the 60% of tokens the model is most wrong about costs
0.29 nats of perplexity and buys nothing measurable downstream.**

Tested as a candidate answer to "train on less compute". Rho-1 (arXiv:2404.07965)
scores every token with a reference model trained on curated data and applies the loss
only to the top fraction by *excess loss* (target minus reference) — the tokens where
the model is still much worse than the reference. Reported: up to 10x faster to
baseline accuracy on math, and +6.8% average across 15 benchmarks for general-domain
continual pretraining.

**Why this repo could test it faithfully rather than approximately.** The method needs
a reference that is the same base trained further on curated data.
`checkpoints/hybrid_chat_v1.pt` **is** `hybrid_a4_v2/hybrid_a4_step8000.pt` fine-tuned
on the curated OASST + grounded-QA corpora (#108) — the paper's exact relationship, not
a stand-in. Both arms continue-pretrain from `hybrid_a4_step8000.pt` on identical,
pre-materialised batches in identical order, 1,500 steps each, differing only in the
token mask (`scripts/ab_selective_tokens.py`, `scripts/run_slm_ab.sh`).

| | baseline (all tokens) | selective (top 60%) |
|---|---:|---:|
| val loss over ALL tokens | **3.9048** | 4.1965 (**+0.2918**) |
| near-window recall, 300 trials | 27.3% (82/300) | 29.7% (89/300) |
| tokens kept | 100% | 60.0% |

**Verdict: no gain.** Perplexity is clearly worse; the downstream probe is
**statistically indistinguishable** (two-proportion z=+0.63, p=0.53). The nominal
+2.4 points is noise at n=300.

**The experiment was built so a false negative was possible to detect, and it still
came out negative.** Two traps were designed around explicitly:
- *Validation is scored on ALL tokens for both arms.* Scoring only the selected tokens
  would rig the metric toward the arm that chose them.
- *Perplexity alone cannot settle this.* The selective arm trains on a deliberately hard
  subset, so a worse all-token loss is expected and not disqualifying — the paper's
  general-domain result was a downstream benchmark gain, **not** a perplexity win. That
  is why the recall probe was run at all. The observed pattern matches the paper's
  *direction* (perplexity worse, downstream not worse) but the downstream **gain** is
  absent.

**Two limits, stated because they bound the claim.** The downstream reading is a single
probe measuring exact-token recall; the paper used 15 benchmarks, and recall may simply
not be where selection helps. And this is 83M against Rho-1's 1.1B floor — the same
order-of-magnitude gap that ruled out multi-token prediction earlier today (worse than
baseline below ~1B per its own scaling ablation). **This is evidence about this scale and
this setup, not a refutation of the method.**

**Context — the compute-reduction search, settled for now.** Four candidates were
examined today for the "train on less compute" goal:

| candidate | outcome |
|---|---|
| padding-free packing (#132) | ✅ **1.45-5x, merged, on by default** |
| block-coordinate descent | ❌ measured 4x worse quality |
| multi-token prediction | ❌ ruled out from the literature (negative below ~1B) |
| selective tokens (this entry) | ❌ measured, no gain at 83M |

Packing is the one that worked, and it is already landed. The remaining honest lever at
this scale is not a training trick — it is spending the fine-tune budget on more and
better data, which this document has concluded repeatedly.

Artifacts: `results/slm/ab_selective_tokens.json`,
`results/slm/probe_{baseline,selective}_near.txt`, `logs/slm_ab.log`,
checkpoints under `checkpoints/slm_{baseline,selective}/`.

## Dynamic phase-aware LR — wins by 0.30 nats, but the win is "the LR was 10x too high" (Issue #133, 2026-08-09)

**One line: the scheduler beat fixed cosine by 0.30 nats held-out, and then the log showed
why — it spent 94% of the run pinned to its own lower bound, so what it discovered is that
the base learning rate was wrong, not that phase-aware scheduling works.**

`ghost/dynamic_lr.py` watches the live loss curve and emits a global LR multiplier applied on
top of the warmup+cosine schedule (`--dynamic-lr`, default OFF). Muon is untouched: every
param group is scaled by the same factor, so its matrix-group / aux-group LR ratio is exact.
Applied as a context manager around `optimizer.step()` and removed before `scheduler.step()`,
because `CosineAnnealingLR.get_lr()` is recursive and would otherwise fold the multiplier into
the cosine permanently (`tests/test_dynamic_lr.py`, 20/20, includes that specific check).

### The A/B

5,000 steps from `checkpoints/hybrid_a4_v2/hybrid_a4_step8000.pt`, Muon, identical
pre-materialised batch sequence, per-arm re-seeding, graded on a fixed 16-batch **held-out**
probe set (never on train loss — the #132 retraction). `scripts/ab_dynamic_lr.py`.

| arm | final held-out | final train | wall-clock | mean multiplier |
|---|---:|---:|---:|---:|
| fixed cosine | 4.3374 | 4.3589 | 26.9 min | 1.000 |
| **dynamic** | **4.0372** | 4.0989 | 26.2 min | **0.126** |

**Gap 0.3002 nats, against a measured identical-run noise floor of sd 0.0046 — ~65x, not
noise.** Both arms started at 4.2945; **only the dynamic arm ever beat its own starting
point.** The cosine arm ended *worse* than the checkpoint it began from, after 5,000 steps.

### What actually fired, which is the whole result

| phase | count |
|---|---:|
| spike (cut, x0.5) | **89** |
| dropping_fast (x1.05) | 36 |
| normal (x1.0) | 25 |
| plateau (x1.5) | **2** |
| transition (x1.25) | **0** |

The multiplier reached its 0.1 floor at **step 294** and never left it for the remaining 4,706
steps. So ~94% of the "dynamic" arm was, mechanically, a constant-0.1x run. The 152 later
adjustments are visible in `results/dynamic_lr/curves.png` as a dense picket of vertical lines
against a completely flat multiplier — the detector firing continuously and changing nothing,
because it was already clamped.

**Both "raise the LR" rules in the spec are dead on this workload.** Plateau fired twice in
5,000 steps; the 2-sigma phase-transition rule fired zero times. Every adjustment that mattered
was a cut.

**And most cuts were not reacting to real events.** At the moment of adjustment the median
`|loss - rolling mean| / sigma` was **0.74**, with only 53/152 outside 1 sigma — i.e. the
majority of triggers were within-noise excursions. That is the question the plot was built to
answer, and the answer is unflattering.

### Verdict

**The 0.30 win is real and reproducible, and it is not evidence for phase-aware scheduling.**
It is evidence that **Muon at `muon_lr=0.02` is roughly 10x too high for continued training
from this checkpoint** — the loss climbs 4.29 -> 4.75 over the first 1,000 steps under the
fixed schedule and never recovers within the run. The scheduler's contribution was noticing
that inside 300 steps and dividing by ten.

**The control arm that would settle this by measurement rather than inference — constant 0.1x,
no detection — was launched and then stopped by the maintainer before completion.** So the
attribution above rests on the multiplier trace (flat at the floor for 94% of the run) and the
phase counts, which are strong but are not the same thing as the arm. Recorded as a known gap,
not glossed: if anyone later wants to claim the detector itself helps, that arm is the bar, and
it is one 27-minute run.

**Recommendation: do NOT make `--dynamic-lr` default.**

### MEASURED — the sane-LR arm beats the scheduler, and that settles it

The third arm, run after the correction below: plain fixed cosine at a 10x lower base LR
(`--lr 2e-5 --muon-lr 0.002`), same harness, same batches, same seed, `--deterministic`.

| arm | base LR | final held-out | vs dynamic |
|---|---|---:|---:|
| cosine | muon 0.02 | 4.3374 | +0.300 |
| dynamic | muon 0.02, auto-cut to ~0.002 | 4.0372 | — |
| **cosine** | **muon 0.002** | **3.9288** | **-0.108** |

**Plain cosine at a sane LR beats the dynamic scheduler by 0.108 nats — 23x the noise floor.**
The scheduler's entire advantage was recovering from a mis-set learning rate; not making the
mistake is better than detecting it, because the dynamic arm still spends its first ~300 steps
at the wrong LR being knocked off the checkpoint before the cuts land. That damage is never
fully recovered inside 5,000 steps.

**Final verdict on #133: the method does not work.** It is a recovery mechanism for a
misconfiguration, and it is beaten by not having the misconfiguration. `--dynamic-lr` stays
non-default. What the exercise actually produced, and what makes it worth keeping: the
determinism fix below, and issue #134 (parameter-free LR), which attacks the real problem —
that a human has to pick this number at all.

### CORRECTION — the baseline was a strawman, and the "10x too high" is narrower than stated

This entry first claimed the LR finding "affects every fine-tune in this repo". **It does
not.** Checked directly after writing that: `continue_ft_hybrid.py:296`,
`finetune_edge_conversational.py:600` and `continue_ft_looped.py:181` all default to
**`--lr 2e-5`**. The repo's fine-tune scripts already encode the right answer.

The `muon_lr=0.02` used above came from `scripts/ab_dynamic_lr.py` copying
`train_edge_conversational.py`'s defaults, which are **pretraining** values — correct for a
freshly initialised model that needs large steps, wrong for a checkpoint already 8,000 steps
into training. New harness, inherited default, nobody had previously run a continue-training
job at pretrain LR.

**So the honest reading is weaker than the table suggests: the cosine arm is a badly-tuned
control, not a fair one.** The 0.30 nats measures how much damage a 10x-too-high LR does and
how quickly a cut rule undoes it — not that dynamic scheduling beats a competently configured
cosine. A cosine baseline at a sane LR would very likely beat both arms here, and that run has
not been done.

What survives the correction: the scheduler's *cut* path demonstrably works and rescued a
mis-set LR inside 300 steps; the raise rules are dead (2 and 0 firings); and the noise-floor /
determinism finding below is independent of any of this.

### A real bug found on the way: the A/B was not reproducible at all

Two arms with identical code paths landed 0.0046 nats apart (sd over 8 runs), **with the sign
of the gap flipping run to run**. Root cause: this torch build has no flash-attention, so the
2 `TransformerBlock`s' `F.scaled_dot_product_attention` falls back to the mem-efficient CUTLASS
kernel, whose backward accumulates `grad_q` with atomics — nondeterministic by PyTorch's own
warning. Isolated three ways: a 15-line standalone repro (`dq` differs, `dk`/`dv` bit-identical);
the in-model signature (in block 10 only `q_proj` and `norm` move, while `k_proj`/`v_proj`/
`out_proj` are bit-identical, and all 10 DeltaNet blocks are clean); and by flag
(`torch.use_deterministic_algorithms(True)` + `CUBLAS_WORKSPACE_CONFIG=:4096:8` makes 40-step
arms bit-identical across 3 repeats). Muon amplifies it — Newton-Schulz normalises the update,
so a ~2e-3 relative gradient perturbation becomes a full-size step in a different direction.

**That floor sat inside the 0.01-0.05 nat band this experiment was built to measure**, so the
A/B could not have detected its own target effect. `--deterministic` is now ON by default in
`scripts/ab_dynamic_lr.py` (~9% wall-clock). Worth carrying to any future A/B in this repo that
touches the hybrid's attention layers.

**One consequence worth its own line:** at 300 steps, two *identical* runs made **7 and 8**
adjustments respectively — the numerical noise flipped a detector threshold and the LR
trajectories diverged from there (final losses 0.0168 apart). A loss-reactive scheduler
amplifies numerical noise into a different training trajectory rather than rejecting it.

Artifacts: `results/dynamic_lr/ab_dynamic_lr.json`, `lr_log.json`, `curves.png`,
`logs/ab_dynamic_lr.log`.

## Parameter-free LR (Prodigy) — wins fine-tuning outright, loses pretraining (Issue #134, 2026-08-09)

**One line: with no learning rate given at all, Prodigy beats hand-tuned Muon by 0.18 nats on a
fine-tune and loses to it by 0.07 on a from-scratch pretrain — so the honest recommendation is a
split by regime, not the single universal setting the issue hoped for.**

Opened directly out of #133, whose real finding was that a human has to pick this number and
picks it wrong when the regime changes. `hvn/optimizers.py` gains `Prodigy`
(Mishchenko & Defazio, arXiv:2306.06101): `lr` is pinned to 1.0 and is not a knob; the
optimizer estimates the step size `d` from observed gradients and keeps re-estimating. Two
variants share one estimator — `--optimizer prodigy` (published Adam-family update) and
`--optimizer muon-prodigy` (Muon's Newton-Schulz **direction**, Prodigy's `d` as the **step
size**). The second coupling is not in the paper and its guarantees do not transfer; that is
stated in the docstring, not left to be discovered.

Both regimes, `scripts/ab_prodigy.py`: identical pre-materialised batches, per-arm re-seeding,
`--deterministic`, graded on held-out probe loss. The Prodigy arms get **no LR, no warmup and
no cosine** — the `d`-ramp is the warmup, and adding a schedule would quietly re-introduce a
tuned knob.

### Fine-tune regime — 5,000 steps from `hybrid_a4_v2/hybrid_a4_step8000.pt` (start 4.29448)

| arm | final held-out | step size | wall | vs tuned |
|---|---:|---|---:|---:|
| Muon, hand-tuned | 3.92877 | 0.002 (tuned) | 25.6 min | — |
| muon-prodigy | 3.80098 | 6.09e-05 (auto) | 30.5 min | **-0.128 (28 sigma)** |
| **prodigy** | **3.75055** | 6.60e-05 (auto) | **23.9 min** | **-0.178 (39 sigma)** |

**Plain Prodigy wins on quality AND is the fastest arm**, having chosen a step size **30x
smaller** than the hand-set value. The harness is cross-validated: its `muon` arm reproduced
#133's independent sane-LR run to 5 decimals at every checkpoint (3.96485 at 1,000, 3.92877
final) from a different script.

### Pretrain regime — 1,500 steps from fresh init, same architecture (start 10.91736)

| arm | final held-out | step size | wall | vs tuned |
|---|---:|---|---:|---:|
| **Muon, hand-tuned** | **5.56450** | 0.02 (tuned) | 8.2 min | — |
| muon-prodigy | 5.81351 | 1.41e-03 (auto) | 9.7 min | +0.249 (54 sigma) |
| prodigy | 5.63261 | 1.69e-03 (auto) | 7.5 min | +0.068 (15 sigma) |

**Hand-tuned Muon wins here**, which is the regime #129 actually validated 0.02 for. Note the
baseline asymmetry that makes this the more trustworthy of the two tables: the pretrain `muon`
arm is genuinely tuned, whereas the fine-tune one used 0.002 — #133's rough 10x correction,
never swept. A properly swept Muon would close part of the fine-tune margin, though not
plausibly all 39 sigma of it.

**A caveat that may explain the muon-prodigy loss entirely:** its `d` was **still climbing at
step 1,500** (9.34e-04 at 1,000 -> 1.41e-03 at 1,500), i.e. it never finished ramping from
`d0=1e-6` inside the test. Plain Prodigy settled by step 500 and lost by far less. A real
pretrain is 100,000+ steps, where a 1,500-step ramp is 1.5% of the run. **This test cannot
distinguish "Prodigy is worse at pretraining" from "1,500 steps is too short to see it" and
should not be read as if it could.**

### The claim the issue was built to test, answered honestly

The premise was that ONE invocation with no learning rate covers both regimes. **It does not.**
It covers fine-tuning better than a human does, and pretraining worse. Recommendation:

- **Fine-tuning: use `--optimizer prodigy`.** Better model, faster, no number to pick. This is
  the regime the repo actually spends its time in (1-3 h runs).
- **Pretraining: keep `--optimizer muon --muon-lr 0.02`.** Validated in #129, wins here.

### The consequence that matters more than the optimizer

**Every fine-tune in this repo ran at a badly mis-set learning rate.** The chat FT (#108), the
tool-use FT (#110), and the grounded-QA FTs behind the 35.5%-near / 8.5%-far recall numbers all
used `--lr 2e-5` AdamW or the pretrain-default Muon, against an estimator that lands 30x lower.
Those runs are 1-3 hours each and cheap to redo.

**In particular the 35.5% near-window recall "plateau" (identical at step 5000 and step 8000,
which the A4 writeup read as a data limit) is now ambiguous.** Those 2 attention layers were
freshly initialised and trained from scratch during that fine-tune, and fresh random layers are
exactly where a too-high LR settles into a mediocre fit and stays. "Plateaued" and "badly
optimised" look identical from outside. Re-running the grounded-QA FT under `--optimizer
prodigy` and re-probing at the 300-trial bar is ~2 h and would settle whether that ceiling was
real or self-inflicted. **Not yet done.**

### Limits

One seed per arm. The pretrain regime is 1,500 steps against a real pretrain's 100,000+. The
fine-tune baseline is corrected-not-swept. `tests/test_prodigy.py` 16/16 — every convergence
test also asserts something about `d`, since "loss goes down" alone would pass with an inert
estimator and a lucky `d0`. Artifacts: `results/prodigy/ab_{finetune,pretrain}.json`,
`logs/ab_prodigy_{finetune,pretrain}.log`.

## RAG works: 0% -> 22.5% on facts the weights cannot hold (Issue #137, 2026-08-10)

**One line: retrieval turns a knowledge question this model answers 0% of the time into one it
answers 22.5% of the time — and the single most important tuning result is that `top-k=1` beats
`top-k=3` by 16 points, because extra retrieved passages actively hurt.**

Opened out of the observation that the 83M's measured weakness is *only* knowledge (closed-book
0/5, ppl ~110 web) while its measured strength is *reading a passage handed to it* (35.5%
near-window exact recall, the capability the 2 windowed-attention layers were added for). RAG
converts the weakness into the strength.

`hvn/retrieval.py` — Okapi BM25, pure Python/numpy, **no new dependency and no second neural
model**. Lexical deliberately: BM25 is the honest floor a dense retriever must later beat, and
introducing an embedding model now would confound "is the retriever good" with "can this model
use retrieved text" — the entanglement that forced #122's retraction.

### The measurement

Three PAIRED arms over the same questions, 200 trials, on `hybrid_a4_v2/hybrid_a4_step8000.pt`.
Facts are about **invented** entities, so closed-book is unanswerable by construction and any
lift is provably retrieval (`eval/rag_probe.py`).

| arm | accuracy | retrieval recall | prompt |
|---|---:|---:|---:|
| closed-book | **0.0%** (0/200) | — | 24 tok |
| RAG, top-3 | 6.0% (12/200) | **100%** | 77 tok |
| **RAG, top-1** | **22.5%** (45/200) | **100%** | 52 tok |
| oracle (perfect passage) | 21.5-22.0% | — | 52 tok |

**RAG at k=1 matches the oracle** (22.5% vs 21.5%, within the n=200 noise of ~2.9pp). Retrieval
is at its ceiling: there is nothing left for a better retriever to win.

### The finding worth carrying forward: more context is worse

The first run reported a 16-point gap between RAG (k=3) and oracle, and the script's own verdict
line said "retrieval is the bottleneck". **That was wrong, and recall@3 = 100% proves it** — the
correct passage was in the context every single time. The gap was the *other two* passages
crowding it out. Dropping to k=1 recovered all 16 points.

The verdict logic is now fixed to check recall before blaming retrieval, because the wrong
version of that line is exactly the kind of thing that gets quoted into a doc unexamined.

**Practical rule for this model: retrieve one passage, not three.** At a 512-token trained window
(#135) context is the scarcest resource in the pipeline, and an 83M model has little capacity to
ignore a distractor. This is likely to relax with scale and with a longer trained window; at this
size it is a 3.75x effect.

### Cost

BM25 index over 2,000 facts builds in **0.01s**; end-to-end query (retrieve + generate) is
**0.20 s**. No training, no GPU-days, no new dependency — the cheapest real capability gain in
this repo's history.

### Limits, stated because they bound the claim

- **The 22% is not comparable to the 35.5% recall figure.** To index 2,000 facts the probe needed
  a wider pseudo-word space than `synthetic_recall_probe`'s 20x20=400 words (which caps at 200
  facts — see the bug below), so it uses 3-syllable words. Longer rare words are harder to
  reproduce exactly. The within-run arms are all on the same words and so are comparable to each
  other; the cross-doc comparison is not.
- **Invented facts are the easy case for BM25** — a unique entity name has very high IDF. Real
  prose, where the query shares common words with thousands of documents, is harder. Recall@k
  will not be 100% there.
- One checkpoint, n=200 (sd ~2.9pp), 320-token context budget.

### A latent infinite loop found and fixed

`synthetic_recall_probe.make_pair` redrew pseudo-words in an unbounded `while True` until it
found an unused pair. The space is 20x20 = **400 words** and each fact consumes two, so asking
for more than 200 facts spins a core **forever**, silently, with no output. The RAG probe
requested 2,000 and hung on fact #201; diagnosed from a live stack (5 identical samples, 5.05s
CPU per 5s wall — spinning, not blocked), which also ruled out the import/network hypotheses.
Now bounded and raises with the actual limit. Existing probe runs are unaffected (they use at
most 17 pairs), but the trap was live for anyone raising `--n-distractors`.

## Prodigy on the grounded-QA FT — recall decays monotonically with steps (Issue #134 follow-up)

Testing whether the 35.5% near-window "plateau" was a badly-optimised fit that a better LR would
break through. **It was not — the opposite.** `continue_ft_hybrid.py --optimizer prodigy`, 2,000
steps from `hybrid_a4_step8000`, then `eval/synthetic_recall_probe.py` at 300 trials:

| FT steps | near-window recall |
|---|---:|
| 0 (baseline) | **36.0%** (108/300) |
| 1,000 | 21.3% (64/300) |
| 2,000 | 9.0% (27/300) |

Far-window: 7.7% -> 1.3%. Train loss fell 2.21 -> 1.40 across the same run.

**Clean monotonic decay.** Optimising *better* walked recall downhill *faster*: the 5,240-example
corpus is exhausted at 8,000 steps, and further optimisation buys memorisation of those specific
pairs — which the probe cannot credit, because it scores **invented** words.

**This retracts the recommendation in the #134 entry above** that every fine-tune should be redone
under Prodigy because they ran at a mis-set LR. A better optimiser does not help when the corpus,
not the optimisation, is the limit. The lever is more and more diverse data, which is the
conclusion this document has now reached four separate times.

The RAG result above is the constructive version of the same observation: if the corpus cannot
teach the knowledge, retrieve it instead of training it.

Artifacts: `results/rag/rag_probe_{baseline,baseline_k1,prodigy}.json`,
`results/prodigy/probe_*.txt`, `logs/rag_probe.log`.

## 62% of the 83M was vocabulary — tied embeddings WIN, the 32K vocab does not (Issue #139, 2026-08-10)

**One line: nobody had ever counted where the parameters went. 62% of this "83M model" is two
vocabulary tables. Tying them is a strict win — 31% fewer parameters and BETTER quality. Shrinking
the vocabulary to 32K is a real trade, not a free win, and on the current English corpus it is
slightly negative.**

Opened after the maintainer refused to accept "training levers are exhausted". That claim was true
of every lever anyone had *examined* — optimizers, kernels, memory, scan, packing, schedulers — and
all of them are downstream of the model's shape. One `named_parameters()` count found this in
seconds.

### The audit (`results/vocab/audit_multi.json`)

Through the real `build_model_and_config` path with ternary on — not a standalone `EdgeHVN`, because
that is precisely the mistake that made every VRAM envelope in [training.md](training.md) wrong:

| vocab | tied | total params | of which real model | MFLOP/token |
|---:|:--:|---:|---:|---:|
| 50,257 | no | 83,079,329 | **38.0%** | 127.0 |
| 50,257 | yes | 57,347,745 | 55.0% | 127.0 |
| 32,000 | yes | 47,981,904 | 65.8% | 108.3 |
| 16,000 | yes | 39,773,904 | 79.4% | 91.9 |

The two levers differ in kind. The input table is a **gather** — it costs memory and no compute, so
tying it is free on FLOPs. The output head is a **matmul**, 512 x 50257 per token, **45% of forward
FLOPs against 38% for all twelve blocks combined** — so shrinking the vocab is a compute lever.

### Tied embeddings — PROVEN, adopt it

1,500 steps from fresh init, identical batches/seed/`--deterministic`, only `tie_embeddings`
differing (`results/vocab/ab_tied.json`, `ab_untied.json`):

| arm | final held-out | params | wall |
|---|---:|---:|---:|
| untied | 5.56450 | 83,079,329 | 8.2 min |
| **tied** | **5.50681** | **57,347,745** | 8.8 min |

**0.058 nats BETTER with 31% fewer parameters** — 12.5 sigma against the #133 noise floor of
sd 0.0046. Not a trade; strictly better and smaller. Both arms share a vocabulary, so per-token loss
IS comparable here.

Likely mechanism: regularisation. 51.5M of the 83M were two independent tables learning nearly the
same mapping from limited data; forcing them to share removes 25.7M parameters that were otherwise
free to memorise.

Honest cost: **~7% slower per step** despite identical FLOPs, because the shared table now receives
gradients from both the input lookup and the output head, serialising an accumulation.

### The 32K mixed tokenizer — a real trade, currently negative on English

Trained ByteLevel BPE on 300 MB, 50/40/10 code/English/chat. Code came from 21 shallow-cloned
permissive repos (`scripts/collect_code_corpus.py`) — `bigcode/the-stack-dedup` lists anonymously
but 401s on download, and this machine's HF token is malformed (#120), so there is no anonymous path.

**Tokens per document, identical source text, at full corpus scale:**

| corpus | GPT-2 | 32K | ratio |
|---|---:|---:|---:|
| code (sample) | 16.12 tok/line | 11.30 | **0.701** |
| OASST chat | 15,724,872 | 11,929,880 | **0.759** |
| SlimPajama | 1,318,230,992 | 1,346,158,575 | 1.021 |
| WikiText | 119,073,409 | 129,653,042 | 1.089 |

The 4,000-line sample predicted 1.022 for SlimPajama; the full 1.3-billion-token corpus gives 1.021.
The method holds at scale.

**The A/B, graded on bits per byte** (1,500 steps each, 2,010,062 held-out bytes of FineWeb-Edu —
downloaded but mix share 0.0, so unseen by both arms):

| arm | **bits/byte** | tokens/byte | per-token NLL | wall |
|---|---:|---:|---:|---:|
| GPT-2 50257, untied | **2.0444** | 0.2164 | 6.5497 | 8.9 min |
| 32K, tied | 2.0829 | 0.2308 | 6.2555 | 8.0 min |

**The 32K arm is 1.9% WORSE per byte and uses 6.7% more tokens on English.** Net compute per byte is
~9% cheaper — not the 20% a sampling-weighted estimate suggested, and not the 24% quoted earlier
from an unweighted mean that included code the corpus does not contain.

**Three limits, because they decide whether this is worth revisiting:**
- **The held-out set is pure English prose, the 32K vocab's WORST domain.** Its measured advantages
  are code (0.701) and chat (0.759), and neither is in this held-out set or in the pretrain corpus.
  This test measures the vocabulary exactly where it cannot win.
- **The arms are matched on STEPS, not compute.** The 32K arm has 42% fewer parameters and finished
  10% faster; at equal compute it gets more steps and part of the 1.9% closes.
- One seed, 1,500 steps.

**Recommendation: adopt tied embeddings; hold the 32K vocab.** It is a one-way door — a model cannot
be fine-tuned onto a different tokenizer — and its case rests on code and chat being in the corpus,
which is #136's job. Revisit it there, not before.

### Why per-token loss could not have answered this

At **step 0**, untrained, zero knowledge: the 32K arm reads **10.41** and the 50257 arm **10.82** —
exactly ln(vocab) in both cases. After 1,500 steps the order **inverts**: 6.26 vs 6.55, now favouring
32K, because each of its tokens carries more text and is genuinely harder to predict.

Two artefacts pulling opposite ways, neither about quality, and the sign flips during training. On
per-token loss this A/B would have reported the 32K vocab as the clear winner.

`eval/bits_per_byte.py` normalises against source bytes instead. The first token of each window is
left unscored — a model must not be charged for predicting a token it had no context for, and that
penalty falls harder on whichever tokenizer produces more windows.

### Infrastructure: four silent hazards closed on the way

None would have raised an error; each would have produced a plausible wrong number.

1. **The data pipeline overrode the caller's tokenizer with GPT-2's unconditionally.** Passing a 32K
   tokenizer would have tokenized the whole corpus with GPT-2 anyway, and the model would have
   trained on ids from a vocabulary it does not have.
2. **The `.bin` cache recorded nothing about which tokenizer produced it.** Switching vocab reused
   GPT-2 ids in silence. Now a `.tokenizer` sidecar, and a mismatch refuses. Unlabelled pre-#139
   caches are adopted as GPT-2 out loud — and that is not a loophole: an unlabelled cache with a
   *custom* tokenizer still refuses.
3. **`pre_training_audit` compared logits width to the 50257 constant**, so it would have rejected a
   32K model before step 1 — the audit blocking the run it exists to protect.
4. **EOS was hardcoded 50256 in 10 files.** The 32K vocab's EOS is 0. A 32K model served with 50256
   could never sample its stop token, so every reply would run to the token limit — reading as a
   quality problem while being a configuration one. `EdgeHVN` now carries `eos_token_id`, persisted
   and restored. Still hardcoded in `hvn/edge/chat.py`, `tool_session.py`, `eval/grounded_qa.py`,
   `finetune_edge_conversational.py` and `packing.py`'s `pad_id`.

**Also: re-tokenizing is 4.28x faster and bit-identical.** `_tokenize_text_to_bin` called
`tokenizer.encode` once per 4 MB block — one Python call into a single-threaded path, pegging one
core of sixteen. Now split across cores via the Rust backend's `encode_batch`: 2.39s to 0.56s on
4 MB, and 1,017,861 tokens either way. Identical is not free — chunks must begin **at** a newline,
not after one, because newline-plus-indent merges are exactly what a code-trained vocabulary learns.
The full 5.4 GB pass went from ~55 minutes to 10.

Artifacts: `results/vocab/*.json`, `logs/ab_tied.log`, `logs/ab_vocab.log`, `logs/ab_bpb.log`,
`data_cache/tokenizer_32k/`.

## The 32K vocab WINS on code -- 6.26% better bits/byte (Issue #139 follow-up, 2026-08-10)

**One line: put code in the training mix and in the held-out set, and the 32K mixed vocabulary
beats GPT-2 by 6.26% on bits per byte -- reversing the 1.9% loss #139 measured on English, and
about 3x larger.**

#139 graded the 32K vocabulary on FineWeb-Edu: pure English prose, no code in either arm's
training mix. That entry flagged the problem in its own limits section -- "this test measures the
vocabulary exactly where it cannot win" -- because the vocabulary's measured advantages are code
(0.701 tokens/line) and chat (0.759), and neither was present. This is the test it never got.

`scripts/ab_vocab_on_code.py`: 165 MB of permissively-licensed code as the training corpus,
held-out is the corpus TAIL (different repositories and languages from the head, since the
collector walks repos in sorted order), 1,500 steps per arm, tokenizer as the only difference.

### The result (`results/vocab/ab_code.json`)

| arm | **bits/byte** | tokens/byte | held-out bytes | windows |
|---|---:|---:|---:|---:|
| GPT-2 50257 | 2.6569 | 0.4071 | 2,110,655 | 1,679 |
| **32K** | **2.4907** | **0.3517** | 2,110,655 | 1,450 |

**6.26% better per byte, 13.60% fewer tokens per byte.** Both arms are scored on the *identical*
2,110,655 held-out bytes, which is the whole reason `eval/bits_per_byte.py` exists: #139 showed
per-token loss is not comparable across vocabularies and would have flipped the sign.

For contrast, the same metric on English (#139): GPT-2 2.0444, 32K 2.0829 -- the 32K vocabulary
1.9% WORSE. Same tokenizer, same metric, opposite sign. The domain is the whole story.

### The caveat that matters: the arms are matched on TOKENS, not bytes

Both arms consume 1,500 x 4 x 512 = 3,072,000 tokens. They do not consume the same amount of
code, because that is precisely what a tokenizer changes:

| arm | corpus as tokens | tokens trained on | code actually seen |
|---|---:|---:|---:|
| GPT-2 | 72,841,334 | 3,072,000 | ~6.96 MB (4.22% of corpus) |
| 32K | 52,598,601 | 3,072,000 | ~9.64 MB (5.84% of corpus) |

**The 32K arm trained on ~38% more code.** Some unknown share of the 6.26% is that extra text
rather than better modelling of the same text.

Stated as a mechanism, not a hole, and here is why the byte-matched control was considered and
NOT run: packing more text into each token is what a tokenizer is *for*. It is the product, not
contamination. A byte-matched run would discard the main benefit and report a number that
understates the vocabulary that would actually ship.

The compute argument runs the same way. From this issue's own audit, 32K tied is **108.3
MFLOP/token against 127.0** for the 50257 head. The 32K arm sees ~38% more code AND costs ~15%
less per token, so matched on compute it wins by more than 6.26%, not less. #139's mirror-image
caveat ("matched on STEPS, not compute ... at equal compute part of the 1.9% closes") pointed the
other way on English; it points this way here.

### Limits

- **Wall times in the JSON are not comparable and must not be quoted.** The gpt2 arm's 1,052 s
  shared the GPU with a duplicate process (below); the v32k arm's 457 s ran alone. That 2.3x is an
  artefact of the launch mistake, not a speedup. Quality is unaffected -- determinism is
  per-process (`torch.use_deterministic_algorithms(True)`, per-arm re-seeding).
- One seed, 1,500 steps, one held-out set.
- Held-out is the corpus tail. Different repos from the head, so not a memorisation check, but it
  is the weakest part of the design and the script says so in its own docstring.
- Both arms see well under one epoch (4-6% of corpus), so neither is in a multi-pass regime.

### What this does NOT settle

**The adoption decision is still open, and #136 still owns it.** This measures code alone. The
English penalty has not been re-measured against a corpus containing both, and the vocabulary is a
one-way door -- a model cannot be fine-tuned onto a different tokenizer. The run that would move
the decision is a mixed-corpus A/B, which is exactly where #139 said to revisit it.

Current reading: 32K wins on code by 6.26%, loses on English by 1.9%. Whether that nets out
depends entirely on the mix, and nobody has measured the mix.

### Two process failures worth recording

**1. The run was launched twice, and the second launch hid the first.** Both processes were
redirected to `logs/ab_code_vocab.log`. Each held its own write offset, so the later process
overwrote the earlier one *in place*: file size stayed pinned at 2,717 bytes, `tail -f` never saw
growth, and a monitor watching for progress stayed silent through a completed run. Silence read as
"still running" for 16 minutes after the verdict had already been written. Two copies of a
12-layer model also pushed the 6 GB card to 5,766 MiB and one arm died of CUDA OOM inside Muon's
Newton-Schulz (`hvn/optimizers.py:73`). Rule added to CLAUDE.md: one run per log file, and check
the GPU is clear before launching.

**2. `logs/ab_code_vocab.log` is not a citable artefact for this run.** Its step lines are two
processes interleaved -- the timings read 6.6, 10.6, 12.4 min for successive checkpoints, and one
line lost its opening bracket where the writes collided. The JSON is clean, written only by the
surviving process. Cite the JSON.

### A reproducibility gap found on the way

**The #139 A/B script was never committed.** `main` carries `results/vocab/ab_bpb_{gpt2,32k}.json`,
`ab_tied.json` and `ab_untied.json`, but no script that produces them -- only
`scripts/audit_vocab_budget.py` and `scripts/train_tokenizer.py`. Searched every `.py` on disk
outside `.venv` and every branch: it does not exist. So the headline "tied embeddings win by 0.058
nats" result, which is now the repo default, cannot be reproduced from the repo. Rewriting that
harness is open work.

Artifacts: `results/vocab/ab_code.json`, `scripts/ab_vocab_on_code.py`.

## The 32K vocab on a MIXED corpus: a split, not a win -- break-even is ~17% code (Issue #139, 2026-08-10)

**One line: on a corpus containing code, English and chat, the 32K vocabulary is 8.6% BETTER on
code, 2.4% WORSE on English and 2.7% better on chat -- so there is no single answer, and the
decision rule is the code share of the corpus: above ~17% code the 32K vocab wins, below it
GPT-2 wins.**

The two prior measurements disagreed in sign. #139 graded the vocabulary on English only and got
1.9% worse; the code follow-up graded it on code only and got 6.26% better. Both were
single-domain, so neither could answer the question that actually decides adoption -- what
happens on a corpus containing both. The tokenizer is a one-way door: a model cannot be
fine-tuned onto a different vocabulary, so the choice is made once, before the pretrain.

`scripts/ab_vocab_mixed.py`: both arms train on the SAME mixed corpus, then each is scored on
THREE separate held-out sets, byte-identical per domain across arms.

### Setup

193 MB mixed corpus, measured shares **code 49.4% / English 39.7% / chat 10.8%** by bytes (the
mix the 32K tokenizer was itself trained on). Sources are `code_corpus/code_all.txt`,
`slimpajama_en.txt` and `oasst1.txt`, interleaved in 1 MB blocks so random window sampling reaches
every domain instead of hitting three contiguous slabs. 1,500 steps per arm, seed 7,
`--deterministic`, tied embeddings, no ternary, tokenizer as the only difference.

Held-out is 2 MB per domain with a **2 MB discarded gap** between train and held-out in each
source. Without the gap the slices touch, and one repository or chat thread straddling the
boundary puts its own text on both sides -- contamination that a larger held-out set does not fix.

### Result (`results/vocab/ab_mixed.json`)

| domain | GPT-2 b/B | 32K b/B | change | 32K tokens/byte |
|---|---:|---:|---:|---:|
| code | 2.1637 | **1.9770** | **-8.63%** | 0.3312 vs 0.5048 |
| English | 2.1983 | 2.2514 | **+2.42%** | 0.2413 vs 0.2329 |
| chat | 2.5755 | **2.5070** | **-2.66%** | 0.2673 vs 0.3355 |
| blended at the trained mix | 2.2220 | **2.1434** | -3.54% | -- |

**The blended -3.54% is the least useful number in the table** and is recorded only for
completeness. It is a property of the mix that happened to be trained on, not of the vocabulary.
Per-domain figures are the durable result because they re-blend for any other mix without
another GPU hour.

### The decision rule

Absolute differences (32K minus GPT-2, bits/byte): code **-0.1867**, English **+0.0531**, chat
**-0.0685**. Solving for where the blend is zero:

| corpus | 32K vs GPT-2 |
|---|---:|
| 100% code (Issue #136's model) | **-8.63% (32K wins)** |
| 30% code / 60% English / 10% chat | -1.4% (32K wins) |
| **~17% code / 73% English / 10% chat** | **break-even** |
| 5% code / 85% English / 10% chat | +1.30% (GPT-2 wins) |
| this repo's current pretrain mix (~65% English / 35% chat) | +0.45% (GPT-2 wins) |

With no chat at all, break-even rises to **~22% code**.

**Recommendation: adopt 32K for the code model (#136); keep GPT-2 for the general/chat line.**
The single universal answer this issue kept looking for does not exist, and the blended number
would have hidden that.

### The asymmetry that cuts against the 32K vocab

The arms are matched on TOKENS, not bytes. On this corpus the 32K vocabulary needs 56,127,531
tokens where GPT-2 needs 70,189,328 for identical text, so at the same 3.07M-token budget **the
32K arm trained on ~25% more text**.

**It lost on English anyway.** The English penalty is therefore measured *against* a 25% data
advantage, which means the true penalty is larger than the 2.42% quoted. The same logic flatters
the code and chat wins by an unknown amount. This is the honest reading in both directions, and
it is why the recommendation splits by domain rather than declaring a winner.

### A clean confirmation of the compute saving

Both arms ran alone on the GPU this time, so wall times ARE comparable (unlike the code A/B,
where a duplicate process contended):

- GPT-2 arm 564.3 s, 32K arm 481.2 s -> **ratio 0.853**.
- Predicted from the audit: 108.3 / 127.0 MFLOP/token = **0.853**.

The measured wall-clock ratio matches the predicted FLOP ratio to three decimals. The smaller
output head delivers exactly the ~15% per-token saving the parameter audit said it would, with no
hidden cost elsewhere.

### Limits

- 1,500 steps, one seed, one mix. This sizes direction and rough magnitude; it is not a pretrain.
- Held-out slices differ from the code-only A/B's, so bits/byte values are NOT comparable across
  the two runs. Only the within-run GPT-2-vs-32K gap carries meaning.
- The break-even shares are linear extrapolations from three measured domains. They assume a
  domain's per-byte difficulty does not change with its share of the mix, which is an assumption,
  not a measurement.

### Two guards whose only job is to fail

`_check_heldout` refuses to run if any domain's held-out set is under 500K chars, and each domain
must score at least 200 windows or the run aborts. #139 shipped a held-out set that was silently
EMPTY -- `read(n)` consumes characters, not bytes, so the slice hit EOF -- and bits/byte would
have reported `nan` without complaint. A silent `nan` is how a broken measurement becomes a
published number.

Artifacts: `results/vocab/ab_mixed.json`, `scripts/ab_vocab_mixed.py`, `logs/ab_vocab_mixed.log`.

## How big a model can this card actually train? 129M body in 9.3 days (Issue #136 sizing, 2026-08-10)

**One line: measured the real ceiling with the real architecture in the real execution mode, and the
answer is that the 32K tied vocabulary buys a 36% larger MODEL BODY at the same wall-clock -- 129M
body in 9.3 days against 95M body in 9.0 days for #136's planned GPT-2 config.**

### Why a new tool when eval/vram_envelope.py already sweeps sizes

That tool cannot answer the question, for three independent reasons, and training.md already warns
its rows "must not be used to size a real run":

1. **It hardcodes `VOCAB = 8000`.** At d=1024 that is ~8M params of embedding against ~103M for an
   untied GPT-2 vocabulary. It makes the vocabulary term -- 42-62% of these models per the #139
   audit -- nearly invisible, so every row overstates how much *body* the card can afford.
2. **It builds `HamiltonianVortexNetwork`**, the core class, which has NO `block_plan` and NO
   `attn_window`. The trained model is `EdgeHVN` with two sliding-attention layers at positions
   3 and 10. The old rows measure a pure DeltaNet trunk -- not the hybrid -- and the attention
   layers are exactly the ones whose memory scales differently with d_model.
3. **No tie_embeddings flag**, so it cannot represent what is now the repo default.

`scripts/size_sweep.py` goes through `build_model_and_config` with `arch="hvn-hybrid"`, ternary on,
Muon, and the `--fast` stack applied.

### The result (`results/size_sweep_fast.json`), --fast ON, head_dim 64, B=4 S=512

| config | arm | total | **body** | vocab share | peak VRAM | tok/s | days | verdict |
|---|---|---:|---:|---:|---:|---:|---:|---|
| 640x16 | gpt2-untied | 130M | 66M | 49.4% | 3,934 MiB (64.1%) | 5,639 | 5.3 | ok |
| 640x16 | **32k-tied** | 86M | 66M | 23.7% | **3,046 MiB (49.6%)** | 6,455 | **3.1** | ok |
| 768x16 | gpt2-untied | 172M | 95M | 44.9% | 5,070 MiB (82.6%) | 4,399 | 9.0 | ok |
| 768x16 | **32k-tied** | 119M | 95M | 20.6% | 3,760 MiB (61.2%) | 4,916 | **5.6** | ok |
| 896x16 | gpt2-untied | 219M | 129M | 41.1% | 5,314 MiB (86.5%) | 3,492 | 14.5 | SUSPECT |
| 896x16 | **32k-tied** | **157M** | **129M** | 18.2% | 4,722 MiB (76.9%) | 3,934 | **9.3** | **ok** |

**Largest body that fits cleanly: 896x16 32k-tied -- 129M of body inside 157M of total.**

The comparison that matters is not total parameters, it is **body at equal wall-clock**:

| | #136 as planned | this recommendation |
|---|---|---|
| config | 768x16, GPT-2 untied | **896x16, 32K tied** |
| body (what learns) | 95M | **129M (+36%)** |
| total params | 172M | 157M |
| peak VRAM | 82.6% | **76.9%** |
| days to Chinchilla | 9.0 | 9.3 |

Same card, essentially the same clock, 36% more model. The budget comes entirely from not spending
45% of the parameters on two lookup tables.

### The measurement error this run made, and corrected

**The first version of this sweep measured eager mode and was wrong about everything absolute.**
`--fast` is `default=True` on `train_edge_conversational` (line 2652, since #129) but is applied in
`main()`, NOT in `build_model_and_config` -- so calling the builder directly silently gets none of
compile, TF32 or Triton scan.

That produced a published-looking claim that #136's config "does not fit, spills at 105% of VRAM,
needs 30.6 days". **Retracted.** Measured properly it fits at 82.6% and needs 9.0 days, and #136's
own ~7-day estimate was approximately right.

| 768x16 gpt2 | peak VRAM | tok/s | days |
|---|---:|---:|---:|
| eager (first, wrong) | 6,446 MiB (105.0%) | 1,300 | 30.6 |
| --fast (correct) | 5,070 MiB (82.6%) | 4,399 | 9.0 |

**Caveat on that contrast, stated because it is not a clean isolation:** the eager sweep used a
fixed `n_heads=8` while the --fast sweep derives `n_heads = d_model / 64`, so at d=768 the two runs
differ in head count (8 vs 12) as well as compile mode. The --fast table is internally consistent
and is the decision-relevant one; the eager-vs-fast deltas are indicative, not measured.

**The lesson, which is the same one #139 recorded in different words:** building through the real
construction path is necessary but NOT sufficient. It must also RUN the way training runs. Two
tools failed here for two different reasons -- one measured the wrong model class, one measured the
wrong execution mode -- and both looked authoritative.

### A hard crash nobody had hit: head_dim must be a power of 2

`hvn/core/deltanet_triton.py:57` does `tl.arange(0, D)` over the head dimension, and Triton requires
that range to be a power of 2. With `n_heads=8`:

| d_model | head_dim | Triton scan |
|---:|---:|---|
| 512 | 64 | ok |
| 640 | 80 | **InductorError** |
| 768 | 96 | **InductorError** |
| 896 | 112 | **InductorError** |
| 1024 | 128 | ok |

**It is a hard crash under --fast, not a silent fallback.** The rule is documented in
`hvn/kernels/hamiltonian_kernel.py:293` and enforced nowhere. The fix is not to avoid those widths
-- it is to pick `n_heads` so `d_model / n_heads` is a power of 2: d=768 wants **12 heads**, d=896
wants **14**. With 8 heads a 7-day run would die partway in with a traceback naming `tl.arange` and
saying nothing about the model config.

**Recommended follow-up: a startup guard.** When `triton_scan` is on, assert `d_model % n_heads == 0`
and that the quotient is a power of 2, and fail immediately with a message naming both numbers.

### The spill detector, and why a plain ok/OOM verdict is not enough

On Windows/WDDM, exceeding VRAM does **not** raise OOM -- the driver spills to host RAM over PCIe
and the step merely gets slower. A run that "succeeded" may be a run that did not fit. The eager
sweep produced rows at 105%, 127% and 138% of VRAM that all completed without error.

So `verdict` is only `ok` when peak reserved is under 85% of total AND step-time variability is
under 0.25. Either signal alone marks the row SUSPECT. Reporting only completion is how the 268M row
in training.md was published as fitting when the real script spills at 355M.

### Limits

- 20 measured steps after 6 warmup, one seed, B=4 S=512, no dataloader in the loop. Real training
  adds data loading and may use longer sequences; these are a floor on memory, not a ceiling.
- Peak memory is reset AFTER warmup, so torch.compile's transient autotuning allocations are
  excluded. Steady-state is what a multi-day run sits at, but a large enough compile-time spike
  could still spill during the first step.
- **The `days` column applies this repo's 20-tokens-per-param convention to TOTAL params.** The two
  arms have identical bodies, so the 32K arm is credited with needing ~30% fewer tokens purely for
  having smaller lookup tables. If the correct budget scales with BODY params instead, the day
  counts converge and the honest 32K advantage shrinks to the throughput margin (1.11-1.15x) plus
  the VRAM headroom. **This is unmeasured and it is the weakest number in the table.**
- Chinchilla-optimality itself is an assumption at this scale; #134 noted a 1,500-step pretrain
  cannot distinguish a real effect from an unfinished ramp.

Artifacts: `results/size_sweep_fast.json` (--fast, decision-relevant), `results/size_sweep.json`
(eager, retained only as the record of the retracted measurement), `scripts/size_sweep.py`,
`logs/size_sweep_fast.log`.

### Addendum (#136, 2026-08-10): megakernel on, and 168M body is reachable -- but only with the 32K vocab

Two follow-up runs after the sweep above, same script, same steps, same seed, so these are paired
against the numbers in the table.

**Design B (`--megakernel-scan`, #96) enabled.** It was never on in the first sweep.

| 896x16 | VRAM | tok/s | days |
|---|---:|---:|---:|
| 32k-tied, megakernel OFF | 4,722 MiB (76.9%) | 3,934 | 9.3 |
| **32k-tied, megakernel ON** | **4,148 MiB (67.6%)** | 4,007 | **9.1** |
| gpt2-untied, megakernel OFF | 5,314 MiB (86.5%) | 3,492 | 14.5 |
| gpt2-untied, megakernel ON | 5,286 MiB (86.1%) | 3,626 | 14.0 |

**The gain is memory, not speed: -574 MiB on the 32K arm for +1.9% throughput.** That is well under
#96's ~1.05x end-to-end figure but consistent with its mechanism -- Design B recomputes in the
backward instead of storing, trading compute for VRAM.

**Unexplained, and recorded as such:** the memory saving is 574 MiB on the 32K arm and 28 MiB on
GPT-2, for the same change at the same size. One run per arm cannot separate a real effect from
allocator variation. Not worth a mechanism story on this evidence.

**Both arms reproduced the previous table within noise, which is the regression check** that moving
the fast stack out of `main()` into `apply_fast_stack` did not change behaviour.

### The headroom bought a bigger body

At 67.6% there was ~1.9 GB spare, so d=1024 was worth measuring -- it had only ever been run eager,
where it read 114.8%.

| 1024x16 | total | body | VRAM | tok/s | days | verdict |
|---|---:|---:|---:|---:|---:|---|
| gpt2-untied | 271M | 168M | 6,224 MiB (101.4%) | 2,324 | 27.0 | SUSPECT (spills) |
| **32k-tied** | 201M | **168M** | **4,652 MiB (75.8%)** | 3,159 | 14.7 | **ok** |

**A 168M body is only reachable with the 32K vocabulary.** The GPT-2 arm spills at the same size,
because 103M of its parameters are two lookup tables. This is the strongest form of the #139 result:
at d=1024 the vocabulary choice is not "cheaper", it is the difference between trainable and not.

### The ladder, and which rung to pick

All --fast + megakernel, all clean fits:

| config | body | VRAM | days | days per 10M body |
|---|---:|---:|---:|---:|
| 896x16 32k-tied | 129M | 67.6% | **9.1** | **0.71** |
| 1024x16 32k-tied | 168M | 75.8% | 14.7 | 0.87 |

**896x16 remains the recommendation.** Going to 1024 buys +30% body for +61% time -- a worse rate
per unit of model. 1024 is genuinely available if 15 days is acceptable; it is not free.

For context on where this started: #136 planned a **95M body in ~9 days**. The same card now gives a
**129M body in 9.1 days**, from the vocabulary and execution-mode changes alone, with no
architecture work.

Artifacts: `results/size_sweep_megakernel.json`, `results/size_sweep_1024.json`,
`logs/size_sweep_mega.log`, `logs/size_sweep_1024.log`.

## HANDOFF: #136 is ready to pretrain (2026-08-10, session close)

Written at the end of a long session so a fresh window can continue without
re-deriving any of it. Everything below is committed; nothing lives only in chat.

### The recommended run

    python train_edge_conversational.py \
      --code-corpus data_cache/code_corpus/stack_code.txt \
      --tokenizer data_cache/tokenizer_32k_fim \
      --fim --fim-rate 0.5 --fim-spm-rate 0.5 \
      --arch hvn-hybrid --attn-layers 3,10 \
      --d-model 896 --num-layers 16 --n-heads 14 \
      --tie-embeddings \
      --checkpoint-dir checkpoints/code_896x16 \
      --val-fraction 0.005

**n_heads=14 is not optional.** head_dim must be a power of 2 or the Triton scan
crashes under --fast; 896/14 = 64. `check_head_dim` now enforces it at startup.

**--checkpoint-dir must be fresh.** Resuming rebuilds the architecture from the
checkpoint; `_check_resume_matches_request` now refuses on mismatch rather than
silently training the wrong model, but a dedicated directory avoids the question.

Expect **9.1 days measured, 10-12 realistic** (measured with synthetic tensors,
no dataloader, on a laptop that throttles).

### What is settled, and where the numbers are

| decision | answer | source |
|---|---|---|
| architecture | hvn-hybrid, d=896, 16L, 14 heads | `results/size_sweep_megakernel.json` |
| size | 157M total / **129M body**, 67.6% VRAM, 9.1 d | same |
| vocabulary | 32K tied + 3 FIM sentinels = **32003** | `results/vocab/ab_mixed.json` |
| corpus | 11.20 GB, 2.86M files, **3.98B tokens** | `results/codecomplete/corpus_stats.json` |
| FIM | rate 0.5, spm 0.5, char-level cuts | `hvn/fim.py`, 39 tests |

Coverage is 1.27x Chinchilla for 157M, so the run stays under one epoch.

### The three open questions, in priority order

1. **Token budget: 3.14B (total params) or 2.58B (body params)?** ~1.5 days of
   difference. The 20-tokens-per-param convention is applied to TOTAL params
   throughout this repo, but the two arms in the size sweep have identical
   bodies, so the 32K arm is credited with needing 30% fewer tokens purely for
   having smaller lookup tables. **Unmeasured, and the weakest number in every
   table.** Recommendation: use 3.14B -- it errs toward more data, and this
   project's most expensive failure was undertraining.

2. **A resume test, before the clock starts.** Designed but not run. Three of the
   five ways resume can break make the loss curve look BETTER, not worse (data
   position restarting re-reads familiar data; optimizer momentum loss; LR
   schedule restart), so it needs an uninterrupted control arm:
   run 60 clean -> run 30, kill, resume to 60 -> compare step count and loss.
   Without arm 1 any plausible number passes.

3. **The eval does not exist.** #136 criterion 4 is line-level accept rate and
   exact-match next-line on held-out repos, against a stated baseline (n-gram or
   nearest-line retrieval). Nothing measures this yet, and a model with no eval
   cannot produce the result the whole track exists for.

### What was found today, condensed

Five bugs, none of which raised an error and all of which produced confident
wrong answers:

- `eval/vram_envelope.py` sized runs from the wrong model class at vocab 8000.
  Deleted; superseded by `scripts/size_sweep.py`.
- `--fast` lives in `main()`, so every measurement harness measured an
  uncompiled model. Published "30.6 days" for a 9.0-day config. Fixed by
  `apply_fast_stack`, the single implementation.
- `main()` has been unable to start a training run since #139 merged
  (`eos_token_id` passed but not accepted). Nothing noticed because every A/B
  harness bypasses `main()`.
- The pre-training audit could not pass on the hybrid (rebuild dropped
  `block_plan`), so it only ever validated the plain architecture.
- Resuming silently replaced the requested architecture with the checkpoint's:
  a request for 157M trained a 28M model, at step 100 of 120, with no warning.

The pattern is one thing: **a stale or partial artifact quietly preferred over
what was asked for.** #139 closed it for the tokenizer cache; today closed it for
execution mode, model class, and resume.

### Known limits carried forward

- The held-out split is the corpus TAIL, and the corpus is written in language
  order, so it is SQL-heavy rather than representative. Fine for watching a loss
  curve, not for a language-stratified quality claim.
- The realised language mix is not the planned mix: filtering removes minified
  JS by BYTES far more than by file count. JavaScript needed a manual top-up
  from 7.9% to 15.9%.
- 2.43% of files in The Stack's "permissive subset" are not permissively
  licensed per file. Measured across 1, 29 and 34 shards (2.55/2.43/2.37%), so
  it is a property of the dataset. They are dropped.

### UPDATE (2026-08-10 19:00): the smoke run CRASHED -- start here

The handoff above says the pipeline was verified end-to-end. **It was not.** The
120-step smoke reached training, passed all four audit checks, compiled, and then
died at 18:35:48:

    extern_kernels.bmm(buf333, reinterpret_tensor(buf327, (56, 64, 32),
                       (2048, 1, 64), 0), out=buf334)
    RuntimeError: CUDA error: CUBLAS_STATUS_INTERNAL_ERROR
                  when calling cublasSgemmStridedBatched(...)
    [W CachingHostAllocator.cpp:26] Exception in pinned allocator free()

`56 = batch 4 x 14 heads`, `64 = head_dim`, so it is the batched matmul at the
shapes this config introduces. Log: `logs/smoke_code.log` (the traceback starts
at line 233; the file also contains a large C-level stack dump, filter with
`grep -vE "blocks\.[0-9]+\.|python314.dll"`).

**Not diagnosed.** Two candidates, neither confirmed:

- **A disguised allocation failure.** CUBLAS_STATUS_INTERNAL_ERROR on a bmm is
  frequently cuBLAS failing to get its workspace. The pinned-allocator warning
  points the same way, at HOST pinned memory rather than VRAM.
- **Something specific to n_heads=14.** This is the first time this config has
  been trained; the size sweep ran the same shapes, but see below.

**The lesson this repeats, and it is the same one as every other bug today:**
`scripts/size_sweep.py` measured d=896/16L/14 heads as fitting cleanly at 67.6%
VRAM, and that measurement is not wrong -- it simply had **no DataLoader, no
pinned host buffers and no gradient accumulation**. The real training path has
all three. "The sweep says it fits" was again treated as evidence about a path
the sweep does not exercise.

**Next window starts here, before anything else:**

1. Reproduce with `--num-workers 0` (removes pinned host memory) and with a
   smaller `--grad-accum`, to separate allocation pressure from a shape problem.
2. If it is allocation, the 896x16 config may need `--grad-checkpoint` or a
   smaller batch, and the 9.1-day estimate changes.
3. If it reproduces at n_heads=14 but not n_heads=8 with the same memory
   settings, it is a shape/kernel problem and the config choice is affected.

Until this is understood, **the 9.1-day plan is not launchable** -- and that is
the honest status, not the "ready to pretrain" heading above.
