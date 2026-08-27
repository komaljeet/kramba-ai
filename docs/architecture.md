# Architecture Components & Project Structure

> **Moved from README:** §4 Architecture Components & Project Structure · the hybrid
> token-recall design block (originally in §5). See [index.md](index.md) for the overview.

## Layering: `hvn/core/` is a shared block library, NOT "the other architecture" (2026-08-11)

> **Read this before proposing to split, duplicate or move code between `core/` and `edge/`.**
> The repo does not contain two parallel stacks. It contains **one block library with two
> top-level wirings**, and mistaking the first for the second led to a proposal to move ~4,600
> lines (including the Triton scan) in order to isolate two files.

| layer | what it is |
|---|---|
| `hvn/core/` (blocks) | **shared library** — `DeltaNetBlock`, `TransformerBlock`, `HVNBlock`, `HVNDecoder`, packing, block-coordinate, MoE, activations |
| `hvn/core/model.py` | `HamiltonianVortexNetwork` — the original top-level model |
| `hvn/edge/model.py` | `EdgeHVN` — **the production model; every documented training run uses this** |

Both `model.py` files import the *same* blocks (`core/model.py:18-21`, `edge/model.py:32-35`),
which is why the two "architectures" feel entangled: they are two wirings of one stack, and the
thing actually trained (`DeltaNetBlock`) lives in `core/`.

### Reachability, computed rather than assumed

Walking `hvn/core/`'s import graph from each top-level model:

| set | modules |
|---|---|
| **Reachable from EdgeHVN — must stay live** (13) | `activations` `batched_memory` `batched_vortex` `block_coordinate` `decoder` `deltanet` `deltanet_triton` `hamiltonian_memory` `hvn_block` `moe` `packing` `transformer_block` `vortex_injection` |
| **Legacy-only — freezable** (2) | `model` (`HamiltonianVortexNetwork`) · `weightless` (`WeightlessDeltaNetBlock`, imported only by it) |
| **Orphaned** (1) | `head_parallel` — referenced by no model; only `tests/test_issue66.py` |

Note that EdgeHVN reaches `vortex_injection`, `hamiltonian_memory` and the batched variants,
because `HVNBlock` pulls them in and `block_plan` can dispatch a layer to `"hvn"`. **The vortex /
Hamiltonian-memory machinery is live infrastructure, not legacy.**

### Consequences for maintenance

- **Freezing "HVN" means freezing two files** — `core/model.py` and `core/weightless.py` — not the
  `core` package. No new features, no parity expectations, keep them importable.
- **Do not move blocks into `edge/`.** `core/model.py` imports the same blocks, so a move forces
  either an inverted dependency (legacy importing production) or a duplicate of `deltanet.py` +
  `deltanet_triton.py` — the code with a documented history of being *silently wrong* (#83/#96/#122).
  Two copies guarantee a fix landing in one and missing in the other, which is the failure mode this
  repo has already hit twice: the audit that "could not have passed on a hybrid … because it only
  ever ran on the plain arch", and the VRAM benches hardcoding `VOCAB = 8000` while training used
  50,257.
- **There is no EdgeHVN-vs-HVN consistency test.** The belief that the legacy model is retained as a
  non-ternary reference is *not implemented* — a grep for parity/allclose assertions across the two
  finds nothing. Code that builds a `HamiltonianVortexNetwork` (e.g. `tests/test_edge_devices.py:368`)
  uses it as a convenient small model for exercising *other* machinery. If a ternary reference check
  is wanted, the honest form is a quantizer parity test (`apply_ternary_to_model` at `group_size=0`
  against the same-shape fp32 model) — no second architecture required.

## 4. Architecture Components & Project Structure

```
kramba-ai/
├── hvn/
│   ├── core/
│   │   ├── vortex_injection.py    # VortexInjectionLayer: Biot-Savart velocity field (+ RoPE)
│   │   ├── hamiltonian_memory.py  # HamiltonianMemoryBlock: MPS entanglement update
│   │   ├── batched_vortex.py      # MultiHeadVortexInjection (all heads in one batched call)
│   │   ├── batched_memory.py      # MultiHeadHamiltonianMemory (residual H, input-dep η, rank-k, multi-thread)
│   │   ├── deltanet.py            # DeltaNetBlock: Gated-DeltaNet dynamics (Issue #34)
│   │   ├── hvn_block.py           # HVNBlock: fluid + quantum wrapper, RMSNorm, parallel scan/forward
│   │   ├── decoder.py             # HVNDecoder: shallow logit projection
│   │   └── model.py               # HamiltonianVortexNetwork: full model + incremental/graph/fused generation
│   ├── kernels/hamiltonian_kernel.py  # Fused Triton kernel (Issue #33 Fix 1)
│   ├── tokenizer.py               # BPETokenizer (HuggingFace tokenizers wrapper)
│   ├── quantize.py                # INT8 + NF4 Hamiltonian quantization (Issue #20)
│   ├── compress.py / sparse.py    # GPTQ, Hadamard, rate-distortion, 2:4 sparsity (Issue #32 Phase 1)
│   ├── lora.py                    # LoRA / QLoRA / instruction tuning (Issue #29)
│   ├── vision.py / tools.py / chat.py / server.py  # Multimodal, tool calling, chat, OpenAI API
│   ├── hf_model.py                # HVNConfig, HVNForCausalLM (HuggingFace wrapper)
│   └── edge/                      # edge-devices branch: full Issue #32 compression suite
├── docs/                          # ISSUE_30_PLAN.md, dashboard.html (compendium → reorganized here)
├── ghost/                         # GHOST training-acceleration research (epic #124): detector, extrapolator, harness
├── eval/                          # run_eval, scaling_laws, long_context, datasets, curriculum, coding, instruction, multimodal
├── tests/                         # test_*.py standalone test scripts (run from root: python tests/test_issueNN.py)
├── demos/                         # demo*.py showcase scripts (run from root: python demos/demo.py)
├── results/                       # benchmark JSON/PNG outputs
├── train.py / generate.py / benchmark.py / quality_benchmark.py   # CLI entry points (kept at root)
├── hvn_checkpoint*.pt / hvn_checkpoint_tokenizer.json             # trained checkpoints (default --checkpoint paths)
└── requirements.txt
```

| Component | File | Role |
|-----------|------|------|
| `VortexInjectionLayer` | `vortex_injection.py` | Token → vortex particle → velocity field via Biot-Savart |
| `HamiltonianMemoryBlock` | `hamiltonian_memory.py` | Fixed-size MPS Hamiltonian with entanglement renormalization |
| `HVNBlock` | `hvn_block.py` | Causal wrapper: sequential vortex injection + H update |
| `DeltaNetBlock` | `deltanet.py` | Gated-DeltaNet readout/update dynamics (Issue #34) |
| `HVNDecoder` | `decoder.py` | Shallow projection: hidden state → vocabulary logits |
| `HamiltonianVortexNetwork` | `model.py` | Full model: embedding + N×block + decoder |

## Hybrid token-recall architecture — Phase A (code landed 2026-07-10)

**Problem.** Pure EdgeHVN (DeltaNet recurrence) **cannot recall exact tokens**: the whole
stream compresses into a fixed `d_head×d_head` state (rank ≤ 64/head) that overwrites
similar keys and reads back a blurry linear blend (no softmax sharpening).
`eval/grounded_qa.py` confirms it — pure HVN scores **0/5 retrieval lift**. **Agentic
coding needs exact recent recall** (precise variable name / code line / error message),
so pure HVN can't be an agentic coder — and the user's P0 bar is "I won't use HVN if I
can't do agentic coding with it."

**Design (settled).** A **hybrid**: 2 sliding-attention `TransformerBlock` layers among 10
`DeltaNetBlock` layers (~layer 3 + ~layer 10 — lower bind + upper retrieve), with **W=4096**
→ KV cache locked at **~16 MB flat** (batch=1 bf16: `2 KB × 2 attn × 4096`). Older context
stays a lossy gist via the DeltaNet state (no cache growth); exact older info is re-fetched
via tools. VRAM stays **flat over an arbitrarily long agentic session** — the defining
constraint ("not a RAM glutton"). W=4K (not 32K) is the v1 baseline because the model trains
at seq_len=512 → W=4K is 8× training length (RoPE generalizes cleanly) and a tight window
forces tool-use discipline. v2 W=32K (128 MB) + long-ctx FT is deferred. This is HVN's
production form (like Mamba→Jamba): 10/12 DeltaNet keeps the identity; the 2 attention
layers add the one thing recurrence structurally can't do.

**Phase A code (landed on `feat/hybrid-attention-recall`, no GPU needed):**

- **A1 — per-layer block plan** (`hvn/edge/model.py`): `EdgeHVN(block_plan=[...],
  attn_window=...)` dispatches each layer by `plan[i]` (`"deltanet"` / `"transformer"`). No
  `block_plan` → legacy single-type broadcast (back-compat). The forward loop is unchanged.
  CLI: `train_edge_conversational.py` gains `--arch hvn-hybrid --attn-layers "3,10"
  --attn-window 4096`; checkpoint metadata widens `block_type` (string) → `block_plan`
  (list) + `attn_window`.
- **A2 — sliding-window mask** (`hvn/core/transformer_block.py`):
  `TransformerBlock(window=W)` builds an additive causal mask `mask[i,j] = -inf if (i-j) >=
  W`, used only when `S > W` (inference/gate). At training seq_len=512 (< W=4096) the
  `is_causal=True` flash path is unchanged → **no training regression**. No new params.
- **A3 — filtered trunk-init** (`train_edge_conversational.py`):
  `load_hybrid_trunk_into_model` warm-starts the hybrid's DeltaNet trunk from a pure-HVN
  checkpoint — copies matching trunk keys, skips the attention positions (fresh init), logs
  matched/missing/unexpected. Plumbed via `--hybrid-trunk-init <ckpt>`.
  `load_checkpoint_for_resume` is now `block_plan`-aware so a hybrid checkpoint resumes.
- **Tests** (`tests/test_hybrid_forward.py`, 9/9 pass on CPU): block-plan dispatch,
  sliding-mask correctness, filtered-load key accounting, hybrid-checkpoint resume roundtrip.

**Sequencing.** Phase A *code* is done. Phase A *training* (A4 short grounded continue-FT
from the pure-HVN trunk → A5 gate = `grounded_qa.py` retrieval lift > 0) follows. The recipe
is **scale-invariant** — if 83M is capacity-limited, scale up via #84 GaLore *with this same
hybrid recipe*. Full plan:
`C:\Users\komal\.claude\plans\bubbly-waddling-sundae.md`.

### A5 gate finding — the mechanism PASSES; what the hybrid can and cannot buy (2026-07-12)

**Superseded note:** an earlier pass through this finding used `grounded_qa.py`'s 5-item
keyword gate and concluded (a) the first A4 FT *degraded* the trunk and (b) beyond-window
recall was flatly ~0 for everything. Both conclusions were **wrong — 5-item-gate noise, not
architecture facts.** Replaced with `eval/synthetic_recall_probe.py` (200 trials/point,
invented-entity QA in the model's trained template — see [state.md](state.md) for the full
numbers and the retraction). The corrected picture:

- **Windowed attention delivers real, large exact recall within the window** — ~9× the pure
  recurrence's floor (4% → 35.5% at 200-trial N). This is the hybrid's core claim, proven with
  a real statistic, not "not broken." It **plateaus** with more training on a fixed corpus —
  the lever to push it further is corpus size/diversity, not step count.
- **Beyond the window, the recurrence carries some genuine exact-token signal — small, but
  real and growing with training** (0.5% trunk floor → 5.5% → **8.5%** at 8,000 FT steps,
  monotonic, ~17× the floor). This *contradicts* the earlier "architecturally capped at ~0"
  claim: the O(1) state is not purely a semantic gist with zero exact-token content — it's
  weak exact recall that improves with training, currently far below near-window quality.
  Whether it keeps climbing with more data/scale or saturates near here is **open** — the
  honest position is "real but not yet reliable," not "impossible" and not "solved." If a
  workload needs **reliable** long-range exact recall today, a bigger window or tool-refetch
  is still the safer answer than trusting the recurrence's gist.
- **First A4 FT (step5000, 40-example corpus) did NOT degrade the trunk** — it was already
  ~3–4× better at near-window recall (16% vs the trunk's 4–5%); the earlier "made it worse"
  reading was 5-item noise. Root cause of it being *insufficient* (not regressive) was data
  volume: `scripts/expand_grounded_qa.py` scaled 40 → 5,240 triples (WikiText-103 copy-cloze),
  and the corpus-expanded re-FT (`hybrid_a4_v2/step8000.pt`) is the checkpoint the numbers
  above are measured on.

**Decision:** the 83M R&D probe **passes** — go-signal for the next staged milestone (README's
"The Dream": ~20B-class native ternary-hybrid), not a reason to invoke the capacity/scale-up
fallback (#84 GaLore was the fallback for a *failed* gate; the gate passed).
## Tensor-Train projections (Issue #161, 2026-08-27) — implemented, opt-in, default off

`hvn/edge/tt_linear.py::TTLinear` stores a projection's weight as a Tensor-Train (Novikov et
al. 2015) instead of a dense matrix: the TT cores **are** the parameters. Opt-in via
`DeltaNetBlock(tt_targets=..., tt_rank=...)` and the same arguments on `EdgeHVN`; unset leaves
both byte-identical to the pre-#161 block.

**Targets: `"qkv"`, `"out"`, `"ffn"`** — ~8d² of the block's ~10d². Measured 3.96x fewer block
parameters at d=896, rank 32, for **1.05x** step time (see [state.md](state.md) for the table
and for the corrected speed claim).

**`alpha_proj` and `beta_proj` are excluded by design and refused by name.** Both are
zero-initialised. TT-SVD of a zero matrix gives zero cores, and because the cores enter the
output *multiplicatively*, those cores also have zero **gradient** — the decay gate would be
permanently dead and the loss curve would look perfectly healthy. This is the same class of
silent-wrong-answer failure as #117's stale cache and #138's frozen context.

**Two contraction orders**, selected by `tt_contract`:

| order | what it does | when |
|---|---|---|
| `"fused"` (default) | folds the cores together first (batch-independent), then one cuBLAS GEMM | training — 1.05x dense |
| `"direct"` | streams the batch through the cores, never forms the `[d,d]` matrix | memory-constrained inference — 2.34x dense, and worse with rank |

A known-answer test asserts the two are the same function in forward and in the gradients
into the cores; without it, a divergence would surface only as an A/B arm quietly computing
something else.

**Relationship to ternary.** `apply_ternary_to_model` targets `nn.Linear` and `TTLinear`
deliberately does not subclass it, so ternary **skips** TT layers rather than stacking two
compression schemes on one weight. `ternary_cores=True` is an explicit opt-in for measuring
the composition.

> ### ⛔ MEASURED AND NOT RECOMMENDED (2026-08-27)
>
> **Do not enable this.** It is kept because it is tested, opt-in and default-off, not
> because it is a good idea. Measured at both scales it is **dominated by plain dense**:
>
> | comparison | result |
> |---|---|
> | equal width, d=384 | TT **0.198 nats worse** (~22% worse perplexity) for 1.81x fewer params |
> | matched params, d=384/512 | inconclusive on quality, but TT uses **11.4% MORE** VRAM |
> | matched params at the 1B target | dense 478M does **1,911 tok/s in 2.64 GB**; TT 499M does **1,033 tok/s in 4.29 GB** |
> | genuine ~1B TT + offload | 576 tok/s, **spills** at 109% of the card, vs dense 1.21B at 1,014 tok/s and 73% |
>
> **The mechanism, and why no larger run is expected to reverse it:** TT trades parameters
> for **width**. Width costs d² compute per token and larger activations, while the PCIe
> transfer cost TT was meant to avoid scales with **parameter count, not width**. So a TT
> model pays width's costs and collects none of the transfer saving. That argument is not
> size-dependent.
>
> Full numbers and the refuted "TT removes the need for offload" hypothesis are in
> [state.md](state.md).

## Mixture-of-Experts FFN (Issue #123, 2026-08-02) — implemented, opt-in, default off

`hvn/core/moe.py::MoEFFN` routes a DeltaNet block's **feed-forward sub-layer** through `E`
experts with top-`k` token routing plus a Switch-style load-balancing auxiliary loss. Wired
into `DeltaNetBlock(moe_experts=..., moe_top_k=...)` and `EdgeHVN(...)`; `moe_experts=1` is the
default and leaves the dense path byte-identical.

**Scope: the FFN only (~40% of block FLOPs).** From the code, the block is ~10d² per layer —
`q/k/v` 3d², `alpha/gate/out_proj` 3d², FFN 4d². The recurrence is deliberately left dense.

**Why the recurrence is NOT routed.** The DeltaNet state accumulates k/v from *every* token:

    S_t = diag(α_t) S_{t-1} + β_t (v_t − S_{t-1} k_t) k_tᵀ

If token A writes through expert-1's `v_proj` and token B through expert-3's, the state mixes
representations from different learned subspaces, and the next token reads all of it back
through a single query. **Transformers do not have this failure mode** — attention is stateless
per layer, so per-token expert routing is safe there. It is not obviously safe inside a
recurrent state. Treated as an open research question, not an extension.

**Measured (d=512, E=8, top-2):** 3.79× stored params at 1.40× active compute → 2.71× better
params-per-FLOP. **But the quality A/B failed its bar** — see docs/state.md: MoE beat dense by
11.8% where the bar (what the same compute buys by simply scaling dense, via the measured
−0.504 exponent) was 19%. The code is landed and tested; the lever is not currently recommended.

**Known-answer property.** `num_experts=1, top_k=1` is numerically identical to the dense FFN in
**both forward and backward** — the E=1 path skips the router entirely, because a softmax over a
single logit would match in forward while adding a router gradient path the dense FFN lacks.
`tests/test_moe_deltanet.py` asserts the router receives no gradient at E=1. This is the
configuration-with-a-known-answer that this project now requires in any new measurement harness
(see the #122 retraction in docs/state.md for what its absence cost).
