# Distributed Launch & Checkpoint I/O Spec (Issue #67)

Part of the frontier-readiness epic (#64). How to launch the #65
data-parallel / #66 tensor-parallel training paths on 1 node × K GPUs and
M nodes × K GPUs, and how a distributed run produces a checkpoint the
existing serving stack (`generate.py`, `web/hvn_worker`, the `ModelVersion`
registry) loads **unchanged**.

---

## 1. Launch commands

### 1 node × K GPUs (torchrun)

```bash
torchrun --standalone --nproc_per_node=K train.py --distributed --full ...
torchrun --standalone --nproc_per_node=K train.py --distributed --fsdp ...   # params outgrow one card
torchrun --standalone --nproc_per_node=K auto_train.py --distributed --from hvn_chat_checkpoint.pt
```

`--standalone` runs the rendezvous on localhost — no MASTER_ADDR needed.

### M nodes × K GPUs (torchrun, c10d rendezvous)

Run **the same command on every node**, with a shared rendezvous endpoint
(any stable host:port reachable from all nodes, typically node 0):

```bash
torchrun --nnodes=M --nproc_per_node=K \
    --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:29500 --rdzv_id=$JOB_ID \
    train.py --distributed --fsdp ...
```

### SLURM

Two supported shapes:

```bash
# (a) torchrun-per-node (recommended): 1 task per node, torchrun fans out K procs
#     #SBATCH --nodes=M --ntasks-per-node=1 --gpus-per-node=K
export HEAD_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
srun torchrun --nnodes=$SLURM_NNODES --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_backend=c10d --rdzv_endpoint=$HEAD_NODE:29500 --rdzv_id=$SLURM_JOB_ID \
    train.py --distributed --fsdp ...

# (b) srun-per-process: 1 task per GPU; hvn.launch translates SLURM_* -> RANK/...
#     #SBATCH --nodes=M --ntasks-per-node=K --gpus-per-node=K
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
export MASTER_PORT=29500
srun python train.py --distributed ...
```

Shape (b) works because `hvn.launch.resolve_launch_env()` maps
`SLURM_PROCID → RANK`, `SLURM_NTASKS → WORLD_SIZE`, `SLURM_LOCALID →
LOCAL_RANK`; it **fails loudly** if the sbatch script forgot
`MASTER_ADDR`/`MASTER_PORT` rather than guessing a head node.

## 2. Backend & interface selection

| Environment | Backend | Notes |
|---|---|---|
| CUDA available | **NCCL** | multi-GPU/multi-node fabric; set `NCCL_SOCKET_IFNAME=<fabric-if>` on multi-homed hosts |
| CPU (parity tests, CI) | **gloo** | same commands, same code path |

Hard-won laptop/CI caveat: gloo derives each rank's address from the machine
hostname; on hosts where that resolves to an unroutable adapter (e.g. a
WSL/Hyper-V vEthernet on Windows), the full-mesh connect **hangs forever**.
Fix: pin the interface — `GLOO_SOCKET_IFNAME="Loopback Pseudo-Interface 1"`
for single-machine tests (what `tests/test_issue65/66/67.py` do), or the real
fabric interface on a cluster. The parity tests additionally rendezvous over
a `dist.FileStore` so no TCP store is involved at all.

Timeout: `hvn.launch.init_from_launcher(timeout_s=...)` defaults to 30 min —
long enough to cover rank-0 checkpoint consolidation stalls at scale.

## 3. Checkpoint contract (the ModelVersion serving path)

A distributed run — DDP-replicated **or FSDP-sharded** — must land on disk as
the **same single-file layout** the serving stack already loads:

```python
{"model_state_dict": ...,           # single-GPU key layout, unsharded
 "vocab_size": V, "d_model": D, "num_layers": L, "n_heads": H,
 "n_threads": T, "chunk_size": C,
 "use_deltanet": bool, "deltanet_conv_kernel": int,
 "weightless": bool, "weightless_seed": int}
```

`hvn.launch.save_consolidated_checkpoint(model, path)` guarantees this:

* **FSDP**: rank-0-only `FULL_STATE_DICT` gather (CPU-offloaded) — the
  collective every rank enters; DDP/compiled models are unwrapped in place.
* **Rank-0-only, atomic write** (tmp + `os.replace`, the #54 discipline): a
  killed job never leaves a truncated checkpoint for the resume logic or the
  worker to trip over.
* **Barrier on exit**: when any rank returns, the file exists for all.

Consequences:

* `ModelVersion(checkpoint_path=...)` registration needs **no serving-side
  changes** — verified by `tests/test_issue67.py`, which loads and serves a
  2-process-DDP-produced checkpoint through the real `generate.load_model`
  path and checks param parity against an identical single-process run.
* Test-coverage note: the **FSDP** consolidation branch is the
  standard `FULL_STATE_DICT` recipe but cannot produce a runnable artifact on
  the laptop — torch's FSDP refuses CPU-only execution and Windows has no
  NCCL — so `tests/test_issue67.py` proves the layout/serving contract via
  the DDP run and skips the FSDP spawn with an explicit message. Re-run
  check 4 on the first CUDA multi-GPU box during cluster bring-up.
* `auto_train.py --distributed` keeps its resume contract: state/metrics are
  rank-0-only and atomic (#65), and stage checkpoints consolidate through the
  same layout.

## 4. What is deliberately NOT here

* No sharded-on-disk checkpoint format (one file per rank): at the current
  research scale consolidation is cheap, and a single file keeps the serving
  contract untouched. When weights outgrow rank-0 RAM, switch the gather to
  `StateDictType.SHARDED_STATE_DICT` + `torch.distributed.checkpoint`, and
  add a consolidation step *before* ModelVersion registration — the seam is
  `hvn.launch._full_state_dict`.
* No elastic/fault-tolerant rendezvous (torchrun `--max-restarts`) — resume
  after failure is handled at the #54 stage granularity instead.
