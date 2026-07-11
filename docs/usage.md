# Notable Runtime Features

> **Moved from README:** §5 "Notable Runtime Features" — the per-feature usage walkthroughs
> (DeltaNet, kernels, quantization, server, chat/curriculum pipeline, coding eval). The
> closed-issue ledger is in [features.md](features.md); CLI flag reference in
> [training.md](training.md).

## Gated-DeltaNet Dynamics — the real PPL lever (Issue #34)

`--use_deltanet` swaps each block for a self-contained Gated-DeltaNet block
(`hvn/core/deltanet.py`): per-channel data-dependent decay gate `α_t=σ(W_α x)` (replaces the
Frobenius clamp), per-head output RMSNorm, gated DeltaNet update
`S_t = diag(α_t)S_{t-1} + β_t(v_t−S_{t-1}k_t)k_tᵀ` with separate k/v projections and query
read `o_t=S_t q_t` (drops the ill-conditioned HPH), plus an output gate `SiLU(W_g x)` +
short causal depthwise conv. Result: eval PPL **8.81 → 3.42 (61% better)**, O(1) inference
preserved. Real-scale results: [benchmarks_deltanet.md](benchmarks_deltanet.md).

```bash
python train.py --epochs 20 --use_deltanet --generate --prompt "The "
```

## Custom Kernels & Parallel Processing (Issue #33)

| Technique | Flag | Speedup | Applies to |
|-----------|------|---------|------------|
| CUDA Graph capture | `--cuda_graph` | 2.82× (78 → 220 tok/s) | Generation |
| Fused Triton kernel | `--fused_kernel` | 2.14–7.2× (78 → 167–340 tok/s) | Generation |
| Parallel scan (associative H update) | `--use_parallel_scan` | 1.59× (7.63 → 4.79 ms/fwd) | Training |
| Multi-chunk barrier sync | (internal via `parallel_forward()`) | 1.14× | Training |

## Hamiltonian Quantization (Issue #20)

INT8 (~4×, <1% error) and NF4 4-bit (~8×, <10% error) via
`hvn.quantize.QuantizedHVNInference` and `estimate_memory_savings`.

## OpenAI-Compatible API Server (Issue #31)

`python -m hvn.server --checkpoint hvn_checkpoint.pt --port 8000` exposes
`/v1/chat/completions`, `/v1/completions`, `/v1/models`, `/v1/health` (SSE streaming + tool
calling), built directly on `forward_step` — no KV cache, O(1) preserved.

## Real chat/base checkpoint pipeline (Issue #51)

One command wires the existing pieces into the first *servable* checkpoint: BPE tokenizer →
WikiText-103 pretrain (DeltaNet + bf16 + the #50 speed levers) → instruction/chat fine-tune →
`hvn_chat_checkpoint.pt` + `hvn_chat_checkpoint_tokenizer.json`, gated by a non-degenerate
smoke test.

```bash
# Recommended run (d=256, 3 layers, streams WikiText-103 from ./wikitext-103-v1):
python build_chat_checkpoint.py --out hvn_chat_checkpoint.pt
# Fast loop-validation (tiny budget/model — CI):
python build_chat_checkpoint.py --smoke --out hvn_chat_smoke.pt
```

The smoke gate requires a non-degenerate, on-topic completion — e.g.
`Create a function that adds two numbers` → `def add(a, b):\n    return a + b`. Held-out
**bits-per-char** (tokenization-invariant) is reported against the bundled char baseline.
First of a 5-issue milestone tree (#51 → #52 app wiring → #53/#54/#55 coding curriculum).

## Coding-curriculum dataset loaders (Issue #53)

`eval/curriculum.py` adds per-language code corpora for the curriculum order **HTML → CSS →
JS → PHP → Python → DevOps**, re-exported from `eval/datasets.py`:

```python
from eval.datasets import load_curriculum_stage, CURRICULUM_STAGES
train, valid = load_curriculum_stage("python")            # cached, license-filtered, deduped
train, valid = load_curriculum_stage("html", local_dir="./code_corpus")  # use pre-fetched files
```

Each stage is **permissive-license-filtered** (copyleft/share-alike rejected),
**near-deduped** (exact + MinHash-Jaccard), budget-capped (compute-matched), split 90/10 by
document. Resolution mirrors the WikiText-103 loader — local dir → best-effort curated
permissive download → **offline synthetic fallback**, so `python -m eval.curriculum`
smoke-loads all six stages with no network. Feeds the #54 orchestrator and #55 coding eval.

## Auto-train orchestrator — staged curriculum scheduler (Issue #54)

`auto_train.py` is a single command that trains the #53 stages **in order**, continuing
from the #51 base checkpoint, evaluating + checkpointing each stage, and **resuming**
cleanly after an interruption (kill / cron / `/loop`):

```bash
# Full curriculum, continuing from the #51 checkpoint:
python auto_train.py --from hvn_chat_checkpoint.pt --stages html,css,js,php,python,devops
# Re-run the same command to RESUME (state in auto_train_state.json):
python auto_train.py --from hvn_chat_checkpoint.pt
# Fast offline loop-validation (tiny budget/epochs — CI / 6 GB box):
python auto_train.py --from tiny_base.pt --smoke
```

It reuses `train.train_model` (the #50 Category-A exact speed levers) and the #53 loaders;
each stage writes `checkpoint_stage_{name}.pt` (+ tokenizer, so every stage is independently
servable) and appends a metrics row to `auto_train_metrics.jsonl`. Progress is committed to
`auto_train_state.json` **atomically after every stage**. `evaluate_stage()` is the seam the
#55 coding-eval harness plugs into.

## Coding eval harness — per-language held-out completion scoring (Issue #55)

`eval/coding.py` gives each curriculum stage a **measurable** score — a per-language,
HumanEval-style held-out code-completion eval:

```bash
python -m eval.coding --checkpoint hvn_chat_checkpoint.pt --lang python
python -m eval.coding --checkpoint hvn_chat_checkpoint.pt --lang all --json results/coding_baseline.json
```

Picks the strongest signal each language's tooling allows, always reporting the
tokenization-invariant fallback:

- **Exec-based pass@k** where a runtime exists — Python always (`sys.executable`), JS when
  `node` is installed, PHP when `php` is installed. Bundled HumanEval-style problems
  (stub + hidden asserts), executed in a throwaway temp dir via `subprocess` + timeout,
  graded pass/fail (model output untrusted → sandbox + timeout).
- **Structural / compile validity** where no runtime is available (HTML, CSS, DevOps, and
  off-box fallback for JS/PHP): the model completes a held-out prefix and the concatenation
  is checked for structural validity — balanced tags/braces, `compile()` for Python,
  `yaml.safe_load` for manifests.
- **Held-out bits-per-char / PPL** is *always* computed (tokenization-invariant).

The orchestrator path defaults to **structural-only** (`allow_exec=False`) so unattended
runs stay fast/side-effect-free; exec pass@k is an explicit `python -m eval.coding` pass.
**#51 base-model baselines** (`results/coding_baseline_issue51.json`): `python` pass@1 0%
(5.08 bits/char), `html`/`css` 0% structural (7.20 / 6.57 bits/char), `js`/`php` 25% (5.77 /
5.73).