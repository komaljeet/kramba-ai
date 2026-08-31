# Frontier Training Plan / Spec (Issue #68)

Part of the frontier-readiness epic (#64). The "aircraft, not the wind
tunnel" document: turns the RTX 4050 measurements into a concrete, budgeted
plan for a real training run, so a cluster commitment is de-risked **before**
money is spent.

Evidence discipline (§1.1 item 4): every number below is either **[measured]**
(traceable to committed JSON in `results/`) or an explicitly **[assumption]**.
No claim rests on the 34K-char artifact.

---

## 1. Config selection — from the measured scaling exponents

Source: `results/scaling_final_pass.json` [measured] (WikiText-103, BPE-8000,
5M-token compute-matched budget, corrected #46 routing, RTX 4050):

| Model | PPL ∝ params^b | d=768 PPL | Notes |
|---|---|---|---|
| **HVN-DeltaNet** | **−0.504** | **129.4** | quality leader at every width; bf16-stable post-#44 |
| HVN-DeltaNet-Weightless | −0.498 | 157.0 | −38% params, small PPL cost |
| Transformer | −0.406 | 150.8 | reference |
| HVN (default block) | −0.397 | 194.3 | flattest — not the frontier block |

**Decision: the frontier block is Gated-DeltaNet (`--use_deltanet`, #34/#39)**
— steepest exponent (its advantage *grows* with scale, §1.1 item 6), beats
the Transformer at d≥512, and is additionally the **tensor-parallel-friendly**
block: its update needs zero collectives in the recurrence, while the default
block's `HPH` term forces cross-shard reductions under within-head sharding
(see `docs/TENSOR_PARALLEL_DESIGN.md` §3–4, Issue #66).

Proposed configs ([assumption]: depth/width ratios follow standard LM
practice; the wind-tunnel sweep varied width at 2 layers, so depth scaling is
*not* yet measured — flagged as a ramp checkpoint in §6):

Param arithmetic [measured]: the DeltaNet block is ≈10·d² params/layer
(29.25M at d=512/L=8/vocab-8000 in `results/vram_envelope_rtx4050.json`,
i.e. 2.6M/layer = 10.0·d²), + ~2·V·d embeddings/decoder:

| Target | d_model | layers | heads | params (≈) | Role |
|---|---|---|---|---|---|
| Pilot | 1024 | 16 | 16 | ~0.2B | first multi-GPU run; exponent on-trend check |
| Research-frontier | 2048 | 24 | 32 | ~1B | first "real" model; serving pilot |
| Flagship | 4096 | 60 | 64 | ~10B | requires the full #64 stack + cluster |

## 2. Token budget & compute estimate

Chinchilla-style compute-matched budget ([assumption]: the ~20 tokens/param
rule transfers to linear-attention fast-weight models — check at the pilot):

| Config | Tokens (20/param) | Train FLOPs (6·N·D) | H100-hours @40% MFU* | Cost @ $2.5/h |
|---|---|---|---|---|
| 0.2B pilot | 4B | 4.8e18 | ~3 | ~$8 |
| 1B | 20B | 1.2e20 | ~85 | ~$210 |
| 10B | 200B | 1.2e22 | ~8,400 | ~$21K |

*[assumption] 40% MFU on H100 bf16 (≈4e14 FLOP/s effective). The DeltaNet
chunked scan is matmul-dominated so transformer-like MFU is plausible, but
**measure it at the pilot** — it is the single biggest cost unknown.

Projected quality: the measured fit `PPL ≈ 652703 × params^−0.504` [measured]
extrapolates to ~19 PPL at 1B and ~6 at 10B — **[assumption] treat as
ordering, not absolute**: the fit was taken at a fixed 5M-token budget;
absolute PPL at a Chinchilla budget will differ (lower). The use of the exponent is comparative (DeltaNet > Transformer > default HVN at scale).

## 3. Data pipeline at scale

WikiText-103 (~0.1B tokens) is 2–3 orders of magnitude short of the 6–200B
needed. Plan (extends the #53 loader discipline — permissive-license filter,
exact + MinHash near-dedup, budget caps, doc-level splits):

* **Mix** [assumption, standard practice]: ~60% curated web (FineWeb-Edu-class),
  ~20% code (the #53 curriculum stages scaled up), ~10% reference/encyclopedic,
  ~10% instruction/chat (the #51 fine-tune stage, scaled).
* **Licensing**: keep the #53 permissive-only gate for the code slice;
  document the web slice's terms-of-use posture explicitly before the run.
* **Sharding**: rank-disjoint streaming via the #65 `DistributedSampler` /
  shard-aware loaders; per-stage budget caps stay compute-matched.
* **Held-out**: fixed, deduped-against-train eval shards per domain;
  bits-per-char (tokenization-invariant, the #51/#55 metric) alongside PPL.

## 4. Deployment target — the edge/ternary lane fork (decide up front)

A ~2 GB, 6-GB-servable 10B artifact only comes from BitNet-b1.58-style
**quantization-aware training** (`hvn/edge/ternary.py`, #32): 10B × 1.58 bits
≈ 1.98 GB [arithmetic]. Ternary is **not reachable post-hoc** from an fp
checkpoint, and during training it keeps full-precision shadow weights +
optimizer state (no training-memory relief — see #69/#70).

| Lane | Training cost | 6 GB-servable? | Quality |
|---|---|---|---|
| bf16 flagship only | 1 run | ❌ (20 GB weights) | best |
| flagship → post-hoc GPTQ/NF4 (#20/#32) | 1 run + calib | ⚠️ ~5 GB @4-bit | small loss |
| ternary from scratch (QAT) | **separate full run** | ✅ ~2 GB | QAT loss — validate |
| flagship → ternary QAT fine-tune | 1 run + substantial QAT phase | ✅ ~2 GB | between |

**Recommended lane: bf16 flagship first; edge artifact (if wanted) as a
ternary QAT fine-tune of the flagship** — it reuses the flagship compute and
the #32 edge module, and its cost is a fraction of a from-scratch QAT run
[assumption: QAT-fine-tune quality near from-scratch, per BitNet-adjacent
literature — validate at the 0.3B pilot before promising it]. The ~2 GB
figure is a laptop-scale projection; validate at 10B.

## 5. Risk register

| Risk | Evidence | Mitigation / gate |
|---|---|---|
| bf16 instability at frontier width | #39 NaN at d≥512 pre-fix; #44 fixed to d=768 [measured] | #44 fp32 islands + P-floor; **re-verify at d=2048/4096** before scale-up; loss-spike guard armed |
| DeltaNet divergence while *winning* | #39 d=768 diverged at PPL 244 [measured] | trajectory monitoring + auto-rollback to last stage checkpoint (#54 resume) |
| Distributed ≠ single-stream | #46 head-misrouting precedent | #65/#66/#67 parity gates re-run on the actual cluster (gloo→NCCL) before the pilot |
| Depth scaling unmeasured | wind-tunnel swept width at 2L only | pilot includes a 2-point depth check (16L vs 24L at fixed params) |
| MFU / cost model wrong | never measured on H100 | pilot measures MFU; budget re-baselined before the 1.4B run |
| O(1) inference regression | the architecture's value proposition | serving smoke (#52 path) on every promoted checkpoint; no KV-cache-shaped shortcuts accepted |
| Data licensing | #53 gate covers code only | web-slice posture documented + signed off before tokens flow |

## 6. Phased ramp & go/no-go criteria

1. **Gate 0 — laptop (done)**: #65/#66/#67 parity artifacts green [measured,
   this repo]; #44 bf16 sweep NaN-free to d=768 [measured].
2. **Gate 1 — single node, K GPUs (0.3B pilot)**: NCCL parity re-run green;
   MFU measured; loss on-trend vs the −0.504 fit at a compute-matched
   checkpoint; bf16 clean at d=1024. *No-go*: any parity drift or NaN.
3. **Gate 2 — multi-node (1.4B)**: #67 launch spec exercised (M×K); sharded
   checkpoint → ModelVersion serving smoke passes; cost within 1.5× of the §2
   estimate. *No-go*: MFU < 25% or unexplained loss-curve divergence.
4. **Gate 3 — flagship (10B)**: only after Gates 1–2 and a written budget
   sign-off; edge/ternary QAT fine-tune decision made per §4.

Each gate's artifacts (JSON + logs) get committed under `results/` — the same
evidence discipline as this repo's wind-tunnel record.

## 7. Pointers

* Scaling evidence: `results/scaling_final_pass.json` (§8.12 of the README).
* On-box envelopes (what the laptop can validate): #69 (RTX 4050),
  #70 (RTX 4060) — `results/vram_envelope_rtx4050.json`.
* Distributed stack: #65 (`hvn/distributed.py`), #66
  (`docs/TENSOR_PARALLEL_DESIGN.md`), #67 (`docs/DISTRIBUTED_LAUNCH.md`).
