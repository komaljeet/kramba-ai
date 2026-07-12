# Architecture Components & Project Structure

> **Moved from README:** §4 Architecture Components & Project Structure · the hybrid
> token-recall design block (originally in §5). See [index.md](index.md) for the overview.

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

### A5 gate finding — what the hybrid can and cannot buy (2026-07-12)

The `--far-probe` A/B (shrink the attention window, push the passage beyond it — see
[state.md](state.md) for the numbers) pins down the design's real envelope:

- **Beyond the window, exact-token recall is ~0 — and windowed attention cannot fix it.**
  The DeltaNet recurrence scored `far-open 0/5` (far-lift 0) on both the trunk and the
  hybrid: the O(1) state does not preserve exact tokens. Because the 2 attention layers are
  *sliding-window*, they are masked out beyond `W` and add nothing there. So for **exact**
  recall past `W`, hybrid ≈ a plain sliding-window transformer ≈ 0.
- **The hybrid's achievable win is exact recall *within* the window.** The trunk's recurrence
  already reads adjacent context weakly (`near-open 2/5`); the attention layers' job is to
  push that toward 5/5 with clean, exact copies. That — plus O(1) cost on 10/12 layers and a
  *lossy semantic gist* of older context (which the keyword gate can't measure) — is the
  honest value proposition over a windowed transformer. If the workload needs **exact**
  long-range recall, the answer is a bigger window or tool-refetch, not the recurrence.
- **First A4 FT (step5000) FAILED** — it *degraded* the trunk (open-book lift +1 → −1) by
  overfitting a 40-example corpus. Root cause was data volume, not architecture or capacity.
  Fix: `scripts/expand_grounded_qa.py` (40 → 5,240 triples via WikiText-103 copy-cloze);
  re-FT from the clean trunk is running (see [state.md](state.md)). Reconcile with the
  re-gate result when it lands.