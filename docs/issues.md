# Issue History & Related Work

> **Moved from README:** §11 Issue History & Related Work. The compact issue → feature
> ledger is in [features.md](features.md); measured numbers are in the
> [benchmarks_*.md](index.md) files.

## Open Issues

- **#124 — [EPIC] GHOST (Gradient Heuristic Omission via Step Teleportation)** ❌ **Negative
  result** (2026-08-03): method by Rambarun Komaljeet — skip predictable training steps by
  extrapolating weights, compute only phase transitions. **The step-skipping family is closed**;
  what survived is one positive spin-off. Phases: #125 detector (gate failed — the documented
  6K-7.5K transition is not in the data), #126 extrapolator (loses to doing nothing at every
  horizon; a *cheating* oracle allowed to see the true future weights also loses, and its fitted
  coefficient collapses toward zero as the jump grows), #127 trainer loop (**not built**, on the
  #126 evidence). **Three** method corrections from the maintainer were applied and the verdict
  survived all of them: measure deltas **per step** (coherence is median cos **+0.890** at 1-step
  spacing vs +0.463 at 10-step); anchor in a **steep** part of the loss curve, not a flat one;
  and (2026-08-05) fit the motion over a **window** of observed steps instead of extrapolating
  one noisy delta, at a settled anchor rather than the huge-PPL beginning. The window fit
  (`ghost/window_fit.py`) is the one gap the oracle left open — that oracle only searched step
  *length* along a fixed direction, never a different direction — and it closes the epic: a
  denoised direction is measurably **worse**, median cos falling monotonically from +0.890 (last
  delta) to +0.166 (50-step mean) as more history is averaged in. Root cause: the predictable
  part of an update is momentum the optimizer already applies; the part that constitutes learning
  comes from gradients of unseen data, which no function of past weights contains — so the best
  available direction is the most recent delta, ridable for about one step. See
  [state.md](state.md).
- **#128 — Weight averaging** ✅ **Positive** (2026-08-03): the spin-off that worked. Averaging
  the last **2-5** checkpoints at the END of a run is worth ~500-1,000 steps of training for
  seconds of arithmetic (9/9 anchor x k combinations improved; k=10 is *worse* — the window must
  be short relative to the drift). The advantage survives continued training (40% retained at
  1,000 steps) and a matched-norm perturbation control confirms it comes from the direction
  averaging picks, not from displacement. **Periodic in-loop averaging does NOT compound** — it
  is worse than plain training, because each injection spends its gain repairing a deficit it
  created. Open: apply to hybrid/chat checkpoints, recall regression check.
- **#130 — 3 parallel networks + averaging** ❌ **Negative on a single GPU** (2026-08-03): at
  equal tokens, one network at 3x batch beats three at 1x batch with periodic averaging, and the
  gap widens monotonically. Contrary to expectation the ensemble was NOT slower in wall-clock
  (batch 12 at seq 512 is bandwidth-bound on this card), so the verdict rests purely on quality.
  Applies to one card only — with 3 GPUs this is ordinary data parallelism (#65).

- **#40 — Non-weights NN** 🔄 **Hybrid shipped (flag-gated)** (2026-07-04): fully weightless is
  not viable (it hands back the #34 learned-dynamics win), so `--weightless` freezes the DeltaNet
  `q/k/v/out` projections as seed-regenerated random features (`hvn/core/weightless.py`) while
  keeping the learned gates/conv. A/B: −38% params/checkpoint at +12.3% PPL, O(1) preserved,
  exact seed round-trip (see [benchmarks_deltanet.md](benchmarks_deltanet.md)). Defaults unchanged;
  scale-up PPL cost still to be measured.
- **#41 — Real web search tool** ✅ **Done** (2026-07-04): replaced the canned-dict mock
  `web_search` (`hvn/tools.py`) with a live backend — keyless **DuckDuckGo Instant Answer API**
  by default (stdlib `urllib`, no new required dep), auto-switching to **Tavily** when
  `TAVILY_API_KEY` is set (or forced via `HVN_SEARCH_PROVIDER`). Same tool signature/schema
  (registry + `ToolCallingSession` unchanged); network/parse errors degrade to a clear error
  string (no crash), results flow through `ToolExecutor.max_result_len` truncation, and a
  `certifi` CA bundle is used when available. `test_issue41.py` 7/7 (HTTP layer mocked). No
  architecture change — O(1) / no-KV preserved. Feeds #25.
- **#43 — Fuse concurrent-stream serving** ✅ **Done** (2026-07-04): `model.generate_served`
  (`forward_step_served`, backends `fused`/`torch`/`auto`). Found the batched `H_i·v_i` reads are
  block-diagonal (no shared GEMM) and decode is **launch-bound**, so the win is the #33 fused
  Triton kernel wired into the batched path (~**2.5–2.7× tok/s** at 1/8/32/64 streams), **not** a
  GEMM reformulation (measured neutral-to-negative). Also fixes a pre-existing **head-misrouting
  bug**: legacy `generate_batched` used head-outer `repeat_interleave` ordering, so `n_heads>1,
  B>1` streams got the wrong head's weights — served is batch-outer and equals single-stream
  (torch ~1e-6, fused ~1e-2). O(1)/no-KV preserved; `test_issue43.py` 7/7. Benefits #31 under
  multi-user load (see [benchmarks_speed.md](benchmarks_speed.md)).
- **#51 — Chat/base checkpoint pipeline** ✅ **Done** (2026-07-05): `build_chat_checkpoint.py`
  wires BPE tokenizer → WikiText-103 pretrain (DeltaNet + bf16 + #50 Category-A speed levers) →
  synthetic instruction (`hvn/lora.py`) + chat (`hvn/chat.py`) fine-tune → `hvn_chat_checkpoint.pt`
  (+`_tokenizer.json`), gated by a non-degenerate smoke test. On a d=256/3L run the coding prompt
  emits `def add(a, b): return a + b` and held-out fine-tune is **0.051 bits/char** vs the char
  baseline's **4.577** (tokenization-invariant comparison). DeltaNet flag persisted; O(1) preserved;
  `test_issue51.py` 4/4, `demo.py`/`test_issue34.py` regressions pass. Milestone 1 — the first real
  servable checkpoint; **blocks #52**, prereq for #53/#54/#55. *Note for #52:* `web/hvn_worker/worker.py`
  must rebuild the model with `use_deltanet=True` (it currently omits the flag) to serve this checkpoint.
- **#25 — Agentic mode**: ReAct-style loop with planning, tool chaining, self-correction.
  Depends on #24 (done).
- **#26 — Training dashboard**: single-page HTML/CSS/JS (Chart.js via CDN), no server.
  Initial `docs/dashboard.html` built; issue covers refinement.
- **#32 — Extreme weight compression**: see [execution_plan.md](execution_plan.md) (Phase 1 +
  edge-devices branch done; Phases 2–5 on `hvn/core` not layered in by design).
- **#35 — Re-measure scaling on real-scale data** ✅ **Done** (2026-07-04): rebuilt harness
  (compute-matched token budget, full eval PPL, BPE, SSM, saturation tracking, streaming
  WikiText-103 loader). On real WikiText-103 (BPE, 5M-token compute-matched, unsaturated) HVN
  scales at **−0.4237** ≈ Transformer −0.4066 (SSM −0.5311) — the flat **−0.007** was a **34K-char
  artifact**, now refuted; a full-Shakespeare small-data control reproduces the flat exponent via
  early saturation, confirming the mechanism (see [benchmarks_scaling.md](benchmarks_scaling.md) /
  [roadmap.md](roadmap.md) §9.2).
- **#36 — Speed up incremental generation** ✅ **Done** (2026-07-04): removed redundant prefill
  + per-token host↔device syncs; HVN decode 80.7 → 197.7 tok/s (2.45×), DeltaNet
  time-to-first-token 1,612 → 23 ms (70×), stacks with #33 kernels (see [roadmap.md](roadmap.md) §9.3).
- **#39 — Make Gated-DeltaNet the default block?** ✅ **Recommendation reached** (2026-07-05): the
  final corrected-routing pass ([benchmarks_scaling.md](benchmarks_scaling.md)) confirms DeltaNet is
  the quality leader at real scale — steepest exponent (−0.504), beats the default HVN block at every
  width (−12→−33% PPL), beats the Transformer at d≥512, and (via #44) is bf16-stable at all widths.
  **Recommendation: `--use_deltanet` becomes the documented default block.** The actual default-value
  flip in code is left as a small follow-up so this measurement pass keeps defaults unchanged.
- **#45 — Final real-scale benchmark pass** ✅ **Done** (2026-07-05): the end-state 4-model
  (HVN / HVN-DeltaNet / HVN-DeltaNet-Weightless / Transformer) WikiText-103 sweep under corrected
  #46 routing (`results/scaling_final_pass.{json,png}`, [benchmarks_scaling.md](benchmarks_scaling.md)).
  Added the `HVN-DeltaNet-Weightless` entry to `eval/scaling_laws.py`. SSM dropped (reference
  competitor, unaffected by #46, §8.4 stands). Feeds the #39 recommendation.
- **#47 — Re-measure under corrected head routing** ✅ **Done** (2026-07-05, folded into #45): the
  A/B shows the #46 fix was **also a quality fix** — correct head routing lifts the default HVN
  block 6–12% PPL by restoring multi-head specialization the bug had diluted; DeltaNet is unaffected
  (doesn't use the buggy modules). §8.4/§8.6/§8.7 default-HVN columns superseded by §8.12 (see
  [benchmarks_scaling.md](benchmarks_scaling.md)).
- **#44 — Fix large-width bf16 instability in the Gated-DeltaNet block** ✅ **Done** (2026-07-04):
  fp32 islands for `_l2norm` / read RMSNorm / decay-write gates, flooring the intra-chunk cumulative
  decay `P` (bounds the `k/P` overflow), and a training-loop loss-spike guard. Full bf16 sweep now
  NaN-free at every width; DeltaNet PPL 522.5→410.6 (d128), 348→251.9 (d256), 236→159.4 (d512),
  205.9→128.0 (d768) — beating both HVN and Transformer at d≥512, steepest exponent (−0.507).
  `test_issue34.py`/`test_issue33.py`/`demo.py` all pass (see [benchmarks_deltanet.md](benchmarks_deltanet.md)).

## #34 — Close the PPL Gap via Readout/Update Dynamics — ✅ Done (2026-07-04)

Confirmed the diagnosis that **dynamics, not state capacity, were the limiter**. HVN is a
linear-attention fast-weight net (`v_t = Γ·(H·ω_t)` read; `H ← (1-η)H + η(P − HPH)` delta-rule
update). Every prior capacity-oriented strategy (#21 rank-k, #22 multi-thread) moved PPL <3%;
swapping in Gated-DeltaNet dynamics cut PPL 8.8 → 3.4 (61% better) at equal/lower FLOPs with
O(1) preserved. Wired `--use_deltanet` / `--deltanet_conv_kernel` through model/train/
quality_benchmark/generate (checkpoint round-trips the flag). Mirrored to `edge-devices`
(`EdgeHVN(use_deltanet=True)`). `test_issue34.py` 5/5: forward↔incremental parity (6e-7),
O(1) scaling (2.57 ms/tok flat N=256→4096), 1500-token stability with no clamp.

## The Quality Gap Summary

The per-model PPL / scaling-exponent comparison table, the #35 char-Shakespeare exponent
caveat, and the "root cause = dynamics not capacity" note are the canonical quality record —
see **[benchmarks_quality.md § The Quality Gap Summary](benchmarks_quality.md)** (moved there
to keep the quality numbers in one place; #34 above validated the dynamics-not-capacity
diagnosis).

## Closed Issues (Quality Journey, condensed)

#21 six strategies (rank-k, causal re-injection, cross-head mixing, nonlinear update,
per-token η, SwiGLU); #22 multi-thread memory (PPL 8.80→8.41 alone, 8.13 combined); #23
vision; #24 tool calling; #27 multimodal fusion; #28 conversation memory; #29 LoRA/QLoRA;
#31 OpenAI server; #33 kernels; plus #1/#12/#13/#14/#15/#16 (see [features.md](features.md)).
