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
