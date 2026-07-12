"""Tests for the GaLore optimizer (Issue #84): gradient low-rank projection so
Adam's optimizer state is O(rank) instead of O(m*n) for large weight matrices.
No GPU needed — runs on CPU with tiny dims.

Covers:
  1. Projection shape correctness for both tall (m>=n, left-project) and wide
     (m<n, right-project) matrices.
  2. Compact optimizer state: exp_avg/exp_avg_sq are rank-shaped, not
     full-shaped -- the actual memory-savings mechanism, verified directly.
  3. Projection refresh on schedule (update_proj_gap) + moment reset on
     refresh (stale moments in an abandoned subspace would misdirect the
     next update).
  4. Small matrices (min(m,n) <= rank) fall back to full AdamW -- GaLore
     would save nothing there, so build_optimizer must not route them into
     the compact path.
  5. Loss decreases over a real short training run through the actual
     build_optimizer("galore") + EdgeHVN path (correctness smoke).

Run:  python -m pytest tests/test_galore_optimizer.py -q
  or:  python tests/test_galore_optimizer.py
"""
from __future__ import annotations

import os
import sys

import torch
import torch.nn.functional as F

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hvn.optimizers import GaLore, _is_galore_param, build_optimizer
from hvn.edge.model import EdgeHVN


def test_projection_shape_tall_matrix():
    """m >= n -> project LEFT: proj is [m, r], compact state is [r, n]."""
    torch.manual_seed(0)
    g = torch.randn(64, 16)  # tall
    proj, side = GaLore._compute_projection(g, rank=4)
    assert side == "left"
    assert proj.shape == (64, 4)
    R = proj.T @ g
    assert R.shape == (4, 16)


def test_projection_shape_wide_matrix():
    """m < n -> project RIGHT: proj is [n, r], compact state is [m, r]."""
    torch.manual_seed(0)
    g = torch.randn(16, 64)  # wide
    proj, side = GaLore._compute_projection(g, rank=4)
    assert side == "right"
    assert proj.shape == (64, 4)
    R = g @ proj
    assert R.shape == (16, 4)


def test_compact_state_is_smaller_than_full():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(128, 32))
    p.grad = torch.randn(128, 32)
    opt = GaLore([{"params": [p], "use_galore": True, "lr": 1e-3,
                  "weight_decay": 0.0, "rank": 8, "update_proj_gap": 200,
                  "scale": 0.25}])
    opt.step()
    state = opt.state[p]
    full_numel = p.numel()  # 128*32 = 4096
    compact_numel = state["exp_avg"].numel()  # should be 8*32=256 (left-project, m>=n)
    assert compact_numel < full_numel, (
        f"compact state ({compact_numel}) is not smaller than full ({full_numel})"
    )
    assert state["exp_avg"].shape == state["exp_avg_sq"].shape
    assert state["proj"].shape[1] == 8  # rank


def test_projection_refresh_resets_moments():
    torch.manual_seed(0)
    p = torch.nn.Parameter(torch.randn(64, 16))
    opt = GaLore([{"params": [p], "use_galore": True, "lr": 1e-3,
                  "weight_decay": 0.0, "rank": 4, "update_proj_gap": 3,
                  "scale": 0.25}])

    proj_ids = []
    for step in range(7):
        p.grad = torch.randn(64, 16)
        opt.step()
        state = opt.state[p]
        proj_ids.append(id(state["proj"]))
        if step in (0, 3, 6):  # refresh happens at step 1 (first), 4, 7 (1-indexed)
            # moments should be freshly re-initialized post-refresh: after
            # exactly ONE update since refresh, moment_step should be 1
            assert state["moment_step"] == 1, (
                f"step {step}: expected moment_step==1 right after a refresh, "
                f"got {state['moment_step']}"
            )

    # projection object should have changed across refreshes (not the same
    # tensor reused the whole time)
    assert len(set(proj_ids)) > 1, "projection never actually refreshed"


def test_small_matrix_falls_back_to_full_adamw():
    """min(m,n) <= rank -> _is_galore_param must reject it (no compression
    possible), so build_optimizer routes it to the AdamW fallback group."""
    small = torch.nn.Parameter(torch.randn(8, 8))
    assert not _is_galore_param("blocks.0.some_weight", small, rank=128)

    big = torch.nn.Parameter(torch.randn(256, 256))
    assert _is_galore_param("blocks.0.some_weight", big, rank=128)

    # exclusions still apply regardless of size
    emb = torch.nn.Parameter(torch.randn(1000, 256))
    assert not _is_galore_param("token_embedding.weight", emb, rank=128)
    norm = torch.nn.Parameter(torch.randn(256, 256))
    assert not _is_galore_param("blocks.0.norm.weight", norm, rank=128)


def test_loss_decreases_with_galore_optimizer():
    torch.manual_seed(0)
    model = EdgeHVN(vocab_size=64, d_model=64, num_layers=3, n_heads=4,
                    use_deltanet=True, chunk_size=8)
    opt, desc = build_optimizer(model, "galore", lr=5e-3, weight_decay=0.0,
                                galore_rank=8, galore_update_gap=5)
    assert "GaLore" in desc

    torch.manual_seed(1)
    ids = torch.randint(0, 64, (4, 16))

    losses = []
    for _ in range(60):
        logits = model(ids)
        loss = F.cross_entropy(logits.reshape(-1, 64), ids.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    early_mean = sum(losses[:10]) / 10
    late_mean = sum(losses[-10:]) / 10
    assert late_mean < early_mean, (
        f"loss did not decrease with GaLore: early={early_mean:.4f} late={late_mean:.4f}"
    )


def test_build_optimizer_galore_partitions_correctly():
    model = EdgeHVN(vocab_size=64, d_model=128, num_layers=2, n_heads=4,
                    use_deltanet=True, chunk_size=8)
    opt, desc = build_optimizer(model, "galore", lr=1e-3, weight_decay=0.01,
                                galore_rank=16)
    all_opt_params = {id(p) for g in opt.param_groups for p in g["params"]}
    all_model_params = {id(p) for p in model.parameters() if p.requires_grad}
    assert all_opt_params == all_model_params, "optimizer is missing/has extra params"

    galore_group = next(g for g in opt.param_groups if g.get("use_galore"))
    assert len(galore_group["params"]) > 0, "no params routed to the GaLore group"


# ---------------------------------------------------------------------------

def _run_all():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:  # noqa
            failed += 1
            import traceback
            print(f"  FAIL  {t.__name__}: {e}")
            traceback.print_exc()
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_run_all())
