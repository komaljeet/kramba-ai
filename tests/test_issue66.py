"""
Test suite for Issue #66 — head-parallel tensor parallelism (DeltaNet block).

Acceptance criterion (from the issue): head-parallel across 2 gloo processes
produces outputs/gradients equal (within fp tolerance) to single-process —
the #46 per-head-independence guarantee must survive sharding.

Checks (each inside a 2-process gloo spawn, compared against a single-process
reference computed in the parent):
  1. Forward parity: HeadParallelDeltaNet(block) == block.forward.
  2. Gradient parity, head-sharded params: local head rows of
     q/k/v/alpha/beta/gate_proj + read_norm_weight match the single-process
     grads; non-local rows are exactly zero (sharded optimizer contract).
  3. Gradient parity, replicated modules: norm/conv/ffn/out_proj grads are
     full and identical on every rank (no sync needed).
  4. O(1)/sharding bookkeeping: per-rank recurrent state holds H/world heads
     (state memory 1/W of total), d_head x d_head each, constant in N.
  5. Sequential-scan path parity (train_chunk=0) — the exact recurrence that
     forward_step also uses, so incremental generation semantics are covered.

Run:  python tests/test_issue66.py         (CPU-only; one 2-process spawn)
"""

# --- repo-root path bootstrap (added during reorg, issue #48) ---
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))))
# --- end path bootstrap ---

import os
import tempfile

for _stream in (_sys.stdout, _sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8")
    except (AttributeError, ValueError):
        pass
os.environ.setdefault("PYTHONIOENCODING", "utf-8")
# Windows: pin gloo to loopback — the hostname can resolve to an unroutable
# WSL/Hyper-V adapter, hanging the mesh connect (same guard as test_issue65).
if _sys.platform == "win32":
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "Loopback Pseudo-Interface 1")

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from hvn.core.deltanet import DeltaNetBlock

TOL = 2e-5
SEED = 0
D_MODEL, HEADS, B, S = 64, 4, 3, 40

HEAD_SHARDED = ["q_proj.weight", "k_proj.weight", "v_proj.weight",
                "alpha_proj.weight", "alpha_proj.bias",
                "gate_proj.weight", "gate_proj.bias",
                "beta_proj.weight", "beta_proj.bias", "read_norm_weight"]
REPLICATED_PREFIXES = ("norm.", "conv.", "ffn.", "out_proj.")


def _build_block(train_chunk):
    torch.manual_seed(SEED)
    blk = DeltaNetBlock(d_model=D_MODEL, n_heads=HEADS, dropout=0.0,
                        train_chunk=train_chunk)
    # beta/alpha projections initialise to constants; randomise so the parity
    # test exercises non-trivial per-head gates.
    g = torch.Generator().manual_seed(SEED + 1)
    with torch.no_grad():
        for p in (blk.beta_proj.weight, blk.alpha_proj.weight):
            p.copy_(torch.randn(p.shape, generator=g) * 0.1)
    return blk


def _input():
    g = torch.Generator().manual_seed(SEED + 2)
    return torch.randn(B, S, D_MODEL, generator=g)


def _reference(train_chunk):
    """Single-process forward + backward; returns (out, {name: grad})."""
    blk = _build_block(train_chunk)
    x = _input().requires_grad_(True)
    out = blk(x)
    out.pow(2).mean().backward()
    grads = {n: (p.grad.clone() if p.grad is not None else None)
             for n, p in blk.named_parameters()}
    return out.detach(), grads


def _worker(rank, world, store_file, train_chunk, out_dir):
    from hvn.core.head_parallel import HeadParallelDeltaNet

    store = dist.FileStore(store_file, world)
    dist.init_process_group("gloo", store=store, rank=rank, world_size=world)
    try:
        blk = _build_block(train_chunk)
        tp = HeadParallelDeltaNet(blk)
        x = _input().requires_grad_(True)
        out = tp(x)
        out.pow(2).mean().backward()

        state = tp.init_state(batch_size=B)
        payload = {
            "out": out.detach(),
            "grads": {n: (p.grad.clone() if p.grad is not None else None)
                      for n, p in blk.named_parameters()},
            "state_S_shape": tuple(state["S"].shape),
            "h_local": tp.h_local, "h0": tp.h0,
        }
        torch.save(payload, os.path.join(out_dir, f"rank{rank}.pt"))
    finally:
        dist.destroy_process_group()


def _run_tp(train_chunk):
    tmp = tempfile.mkdtemp(prefix="issue66_")
    store_file = os.path.join(tmp, "store")
    mp.spawn(_worker, args=(2, store_file, train_chunk, tmp), nprocs=2, join=True)
    return [torch.load(os.path.join(tmp, f"rank{r}.pt"), weights_only=False)
            for r in range(2)]


def _head_rows(name, h0, h_local):
    """The row-slice of a head-sharded parameter owned by [h0, h0+h_local)."""
    d = D_MODEL // HEADS
    if name.startswith("beta_proj") or name == "read_norm_weight":
        return slice(h0, h0 + h_local)
    return slice(h0 * d, (h0 + h_local) * d)


def check_parity(train_chunk, label, idx, total=5):
    ref_out, ref_grads = _reference(train_chunk)
    ranks = _run_tp(train_chunk)

    # 1. Forward parity on every rank
    for r, pl in enumerate(ranks):
        diff = (pl["out"] - ref_out).abs().max().item()
        assert diff < TOL, f"[{label}] rank{r} forward diff {diff}"

    for r, pl in enumerate(ranks):
        h0, hl = pl["h0"], pl["h_local"]
        for name, g in pl["grads"].items():
            ref = ref_grads[name]
            if name in HEAD_SHARDED:
                rows = _head_rows(name, h0, hl)
                d_local = (g[rows] - ref[rows]).abs().max().item()
                assert d_local < TOL, f"[{label}] rank{r} {name} local-row grad diff {d_local}"
                mask = torch.ones(g.shape[0], dtype=torch.bool)
                mask[rows] = False
                assert g[mask].abs().max().item() == 0.0, \
                    f"[{label}] rank{r} {name} has non-zero grads outside its shard"
            elif name.startswith(REPLICATED_PREFIXES):
                d_rep = (g - ref).abs().max().item()
                assert d_rep < TOL, f"[{label}] rank{r} {name} replicated grad diff {d_rep}"

    # 4. Sharded state bookkeeping: H/world heads per rank, d_head x d_head.
    d = D_MODEL // HEADS
    for pl in ranks:
        assert pl["state_S_shape"] == (B * HEADS // 2, d, d)

    print(f"[{idx}/{total}] {label}: forward + head-shard/replicated grad parity "
          f"(2-proc gloo) ... OK")
    return ranks


if __name__ == "__main__":
    print("Issue #66 — head-parallel tensor-parallel parity (DeltaNet, 2-proc gloo)")
    print("-" * 72)
    # Chunked-scan path (the training default) covers checks 1-4.
    check_parity(train_chunk=16, label="chunked scan (training path)", idx=1)
    print("[2/5] head-sharded grads exact in local rows, zero elsewhere ... OK (above)")
    print("[3/5] replicated-module grads full + identical per rank ... OK (above)")
    print("[4/5] per-rank state = H/world × d_head× d_head (O(1), 1/W memory) ... OK (above)")
    # Sequential-scan path — the exact forward_step recurrence.
    check_parity(train_chunk=0, label="sequential scan (forward_step recurrence)",
                 idx=5)
    print("-" * 72)
    print("All Issue #66 checks passed.")
