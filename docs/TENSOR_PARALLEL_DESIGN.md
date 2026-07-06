# Tensor / Model Parallelism for the HVN & DeltaNet Blocks (Issue #66)

Part of the frontier-readiness epic (#64). Companion to the data-parallel path
(#65): DDP replicates the model, capping size at one card; this design shards
the block's **per-head recurrent state and compute** across ranks, and
specifies the harder within-head axis for when a single head outgrows a card.

Status: **head-parallel implemented for the DeltaNet block**
(`hvn/core/head_parallel.py`, parity-proven in `tests/test_issue66.py`);
default-HVN head-parallel and within-head sharding are **specified here**
(same recipe, see §3–§4) and left to the cluster-bring-up phase — this laptop
can only run the 2-process gloo parity artifacts, which the DeltaNet
implementation already provides for the #39-recommended frontier block.

---

## 1. Shard axes — what lives where

| Tensor (DeltaNet block) | Shape | Shard axis | Placement |
|---|---|---|---|
| `q/k/v_proj.weight` | `[d_model, d_model]` | output rows, by head (`H/W` × `d_head` rows each) | column-parallel |
| `alpha_proj`, `gate_proj` (w+b) | `[d_model, d_model]` | output rows, by head | column-parallel |
| `beta_proj` (w+b) | `[n_heads, d_model]` | output rows, one per head | column-parallel |
| `read_norm_weight` | `[n_heads, d_head]` | head dim | with its head |
| recurrent state `S` | `[B·H, d_head, d_head]` | head dim → `[B·H/W, ·, ·]` | **the point**: per-rank state −1/W |
| `norm`, `conv`, `ffn`, `out_proj` | — | replicated | full grad locally, no sync |

Default HVN block (`batched_vortex.py` / `batched_memory.py`): identical
table — since #46 the per-head parameters are **batch-outer tiled** tensors
indexed by head (`vorticity_proj`, per-head `gamma`, per-head Hamiltonians
`H ∈ [B·H, d_head, d_head]`, per-head `eta`/`memory_gate`), so the head shard
is a contiguous slice of the same tiled tensors the batched forward already
uses. The gather point is the same: per-head outputs are concatenated
head-major into `[B, S, d_model]` before the block's shared output projection.

## 2. Collective placement (Megatron-style f/g pair)

```
x_norm = norm(x)                 replicated
xc     = causal_conv(x_norm)     replicated
xh     = f(xc)                   f: identity fwd, all-reduce(SUM) bwd
q,k,v,α,β,g = local head rows    sharded compute
o_loc  = per-head recurrence     sharded state (H/W heads per rank)
o      = g(o_loc)                g: all-gather(head dim) fwd, local-slice bwd
out    = x + drop(out_proj(o)) + ffn(x_norm)     replicated
```

Exactly **one all-gather per block per forward** (`[B, S, d_model]` payload)
and **one all-reduce per block per backward** — no per-token collectives, so
the O(1)-in-N property gains no N-dependent communication.

Why g's backward is a *slice*, not reduce-scatter: the loss is computed
identically on every rank (tensor parallelism replicates the loss), so the
gradient arriving at the gathered tensor is already the full one; a
reduce-scatter would multiply it by W. Why f's backward is all-reduce: each
rank's head branch only produces its shard of `dL/dxc`; the sum reassembles
the full gradient so `conv`/`norm`/upstream layers see exact gradients.

Gradient placement after `backward()` (proven in `tests/test_issue66.py`):
replicated modules hold full identical grads on every rank; head-sharded
projections hold exact grads in the local head rows and zeros elsewhere. A
training loop must either keep optimizer state sharded the same way (each
rank steps its rows — the natural choice) or all-reduce the head grads.

## 3. The update rules under sharding

**DeltaNet** `S_t = diag(α_t) S_{t-1} + β_t (v_t − S_{t-1} k_t) k_tᵀ`,
read `o_t = S_t q_t`: everything is per-head — no cross-head term exists, so
head-parallel needs **zero collectives inside the recurrence**. This is the
#46 guarantee (heads independent, batch-outer tiling) surviving sharding.

**Default HVN** `H ← (1−η)H + η(P_t − H P_t H)` with `P_t = v_t v_tᵀ`:
also strictly per-head (each head owns its `H`), so head-parallel is the same
zero-collective story. The `HPH` term only matters for *within-head* sharding
(§4).

## 4. Within-head width sharding (the harder axis, design only)

Needed when one head's `d_head × d_head` state outgrows a card (frontier
widths). Shard the **value axis** (rows of `S`):

* Rank r holds `S_r = S[v_rows_r, :]` (`d_head/W` rows), plus the matching
  rows of `v_proj` and `read_norm_weight`. `k`/`q` (and `α` columns, `β`) are
  computed redundantly on every rank (cheap: one row-slice GEMM).
* Update: `S_r ← diag(α)S_r + β(v_r − S_r k)k_tᵀ` — needs only full `k`
  (replicated) and the local rows of `v`: **no collective**.
* Read: `o_r = S_r q` gives the local rows of `o` → **one all-gather** on the
  value axis before the read-norm/gate, same g placement as head-parallel.

Default HVN within-head is costlier: `H P H = (Hv)(Hᵀv)ᵀ` needs both `Hv`
(row-shard × full `v` — local) **and** `Hᵀv` (column reduction across the row
shards — one all-reduce of a `d_head` vector **per chunk**, not per token,
when fused into the chunked forward). This is the concrete reason the
DeltaNet block (already the #39 quality recommendation) is also the
**tensor-parallel-friendly** choice at frontier width — its update has no
`HᵀH`-style term.

## 5. O(1) inference under sharding

Per-token cost per rank: `O(d_model · d_head · H/W)` for the projections +
`O(d_head² · H/W)` for the recurrence — constant in N, 1/W of the
single-device cost. Per-rank state memory: `H/W × d_head²` — constant in N,
1/W of total. The single all-gather per block per token is `O(d_model)`
payload, independent of N. **No KV cache, no N-dependent memory or
communication is introduced** (§1.1 checklist item 1 holds).

## 6. bf16 note (#44)

Sharding does not touch the numerics: the fp32 islands (`_l2norm`, read
RMSNorm, decay/write gates, the fp32 chunked scan with the `P` floor) run
identically on the local head slice — the parity test passes with the fp32
scan path exercised. Wider models on more ranks change *which* heads a rank
computes, not how any head computes.

## 7. Launch / composition

* Compose with #65: data-parallel groups × tensor-parallel groups
  (`dist.new_group` per TP island; DDP over the DP dimension). The #67
  launcher owns process-group construction.
* Checkpoints: the wrapper reads the wrapped block's parameters in place —
  the saved `state_dict` is exactly the single-GPU layout (the #67 serving
  contract), no consolidation step needed for head-parallel.
