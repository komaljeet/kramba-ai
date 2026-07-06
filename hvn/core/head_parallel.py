"""
Head-parallel tensor parallelism for the DeltaNet block (Issue #66)
===================================================================

Part of the frontier-readiness epic (#64). Data parallelism (#65) replicates
the model on every GPU, which caps model size at one card. This module shards
the block's **per-head recurrent state and compute** across ranks — the
architecture's natural tensor-parallel axis, because heads have been fully
independent since the #46 batch-ordering fix.

Scope: implemented for :class:`~hvn.core.deltanet.DeltaNetBlock`, the
#39-recommended frontier block. The default HVN block's head-parallel recipe
(identical structure over ``MultiHeadVortexInjection`` /
``MultiHeadHamiltonianMemory``) and the within-head width-sharding axis are
specified in ``docs/TENSOR_PARALLEL_DESIGN.md``.

How the sharding works (Megatron-style f/g operator pair)
----------------------------------------------------------
Per rank, with ``H`` heads over ``W`` ranks (``H % W == 0``)::

    x_norm = norm(x)              replicated   (full grad locally)
    xc     = causal_conv(x_norm)  replicated   (full grad locally)
    xh     = f(xc)                identity fwd / all-reduce bwd
    q,k,v,α,β,g = LOCAL head rows of the projections (sharded compute)
    o_loc  = delta-rule scan over the LOCAL H/W heads (sharded d×d state)
    o      = g(o_loc)             all-gather fwd / local-slice bwd
    out    = x + dropout(out_proj(o)) + ffn(x_norm)     replicated

* **f** makes the conv/norm/input gradients exact: each rank's head branch
  only produces its shard of ``dL/dxc``, and the all-reduce in f's backward
  sums the shards into the full gradient.
* **g** makes the head-shard gradients exact: the loss is computed
  (identically) on every rank, so the gathered tensor's gradient is already
  the full one — g's backward takes the local slice instead of
  reduce-scattering, which would multiply it by the world size.

Gradient placement after ``backward()``:

* Replicated modules (``norm``, ``conv``, ``ffn``, ``out_proj``) hold the
  full, identical gradient on every rank — no sync needed.
* Head-sharded projections (``q/k/v/alpha/beta/gate_proj``,
  ``read_norm_weight``) hold the exact gradient in the LOCAL head rows and
  zeros elsewhere; a training loop must either keep optimizer state sharded
  the same way (each rank steps its rows) or all-reduce these gradients.

O(1) inference is preserved: the recurrent state is still a fixed
``d_head × d_head`` matrix per head with no N-dependence — sharding just
places ``H/W`` of those matrices on each rank (per-rank state memory drops
by ``1/W``; per-token cost per rank drops to ``O(d·d_head·H/W)``).

The wrapper reads (never copies) the wrapped block's parameters, so a
checkpoint saved from the unwrapped block is exactly the single-GPU layout.
"""

from __future__ import annotations

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from .deltanet import DeltaNetBlock


class _AllReduceBackward(torch.autograd.Function):
    """Megatron's *f* operator: identity forward, all-reduce(SUM) backward."""

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, op=dist.ReduceOp.SUM, group=ctx.group)
        return grad, None


class _GatherHeads(torch.autograd.Function):
    """Megatron's *g* operator over the head axis.

    Forward: all-gather the per-rank ``[B, S, H_local, d]`` head outputs and
    concatenate on the head dim. Backward: return the LOCAL slice of the
    incoming gradient — the loss is replicated on every rank, so the incoming
    gradient is already the full one (a reduce-scatter would count it W times).
    """

    @staticmethod
    def forward(ctx, o_local, group):
        world = dist.get_world_size(group)
        rank = dist.get_rank(group)
        ctx.rank, ctx.h_local = rank, o_local.shape[2]
        gathered = [torch.empty_like(o_local) for _ in range(world)]
        dist.all_gather(gathered, o_local.contiguous(), group=group)
        # all_gather doesn't track gradients; rank's own slot must be the
        # autograd-connected tensor so replicated modules downstream see it.
        gathered[rank] = o_local
        return torch.cat(gathered, dim=2)

    @staticmethod
    def backward(ctx, grad):
        h0 = ctx.rank * ctx.h_local
        return grad[:, :, h0:h0 + ctx.h_local].contiguous(), None


class HeadParallelDeltaNet(nn.Module):
    """Tensor-parallel (head-sharded) forward for a replicated DeltaNetBlock.

    Every rank holds the full block parameters (built from the same seed or
    checkpoint); this wrapper computes only the rank's ``H / world`` heads and
    all-gathers the head outputs before the output projection. Outputs and
    gradients are equal to ``block.forward`` within fp tolerance
    (``tests/test_issue66.py``).
    """

    def __init__(self, block: DeltaNetBlock, group=None):
        super().__init__()
        if not (dist.is_available() and dist.is_initialized()):
            raise RuntimeError("HeadParallelDeltaNet requires an initialised "
                               "process group (torchrun / init_distributed)")
        self.block = block
        self.group = group
        self.world = dist.get_world_size(group)
        self.rank = dist.get_rank(group)
        if block.n_heads % self.world:
            raise ValueError(f"n_heads ({block.n_heads}) must be divisible by "
                             f"world_size ({self.world})")
        self.h_local = block.n_heads // self.world
        self.h0 = self.rank * self.h_local

    # ------------------------------------------------------------------
    def _project_local(self, xh: torch.Tensor):
        """The block's ``_project`` restricted to the LOCAL head rows.

        Row-slicing the projection weights is exact: head ``h`` reads rows
        ``[h·d_head, (h+1)·d_head)`` of ``q/k/v/alpha/gate`` and row ``h`` of
        ``beta`` — the same values the full projection would produce.
        """
        from .deltanet import _l2norm

        blk = self.block
        B, S, _ = xh.shape
        HL, d = self.h_local, blk.d_head
        r0, r1 = self.h0 * d, (self.h0 + self.h_local) * d

        q = _l2norm(F.linear(xh, blk.q_proj.weight[r0:r1]).view(B, S, HL, d))
        k = _l2norm(F.linear(xh, blk.k_proj.weight[r0:r1]).view(B, S, HL, d))
        v = F.linear(xh, blk.v_proj.weight[r0:r1]).view(B, S, HL, d)

        beta = torch.sigmoid(
            F.linear(xh, blk.beta_proj.weight[self.h0:self.h0 + HL],
                     blk.beta_proj.bias[self.h0:self.h0 + HL]).float()
        ).view(B, S, HL, 1)

        alpha = None
        if blk.use_decay_gate:
            alpha = torch.sigmoid(
                F.linear(xh, blk.alpha_proj.weight[r0:r1],
                         blk.alpha_proj.bias[r0:r1]).float()
            ).view(B, S, HL, d)

        gate = None
        if blk.use_output_gate:
            gate = F.silu(F.linear(xh, blk.gate_proj.weight[r0:r1],
                                   blk.gate_proj.bias[r0:r1]))  # [B, S, HL*d]
        return q, k, v, alpha, beta, gate

    def _read_normalize_local(self, o: torch.Tensor) -> torch.Tensor:
        """Per-head read RMSNorm with the LOCAL slice of the norm weight."""
        blk = self.block
        w = blk.read_norm_weight[self.h0:self.h0 + self.h_local]
        of = o.float()
        rms = of.pow(2).mean(dim=-1, keepdim=True).add(blk.read_norm_eps).rsqrt()
        return (of * rms * w.float()).to(o.dtype)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor, position_ids=None) -> torch.Tensor:
        blk = self.block
        B, S, D = x.shape
        HL, d = self.h_local, blk.d_head
        N = B * HL

        # Replicated region: full-width, full gradients on every rank.
        x_norm = blk.norm(x)
        xc = blk._causal_conv(x_norm)
        # f: identity fwd; all-reduce bwd sums the per-rank head-branch grads.
        xh = _AllReduceBackward.apply(xc, self.group)

        # Sharded region: this rank's H/W heads only.
        q, k, v, alpha, beta, gate = self._project_local(xh)

        def to_N(t):  # [B, S, HL, *] → [N, S, *]
            return t.permute(0, 2, 1, 3).reshape(N, S, -1)

        qN, kN, vN, betaN = to_N(q), to_N(k), to_N(v), to_N(beta)
        alphaN = to_N(alpha) if alpha is not None else None

        if blk.train_chunk and S > 1:
            oN = blk._scan_chunked(qN, kN, vN, alphaN, betaN, blk.train_chunk)
        else:
            oN = blk._scan_sequential(qN, kN, vN, alphaN, betaN)

        o = oN.view(B, HL, S, d).permute(0, 2, 1, 3)      # [B, S, HL, d]
        o = self._read_normalize_local(o)
        if gate is not None:
            o = o * gate.view(B, S, HL, d)                # local gate slice

        # g: all-gather the head outputs; bwd takes the local slice.
        o_full = _GatherHeads.apply(o, self.group)        # [B, S, H, d]
        o_full = o_full.reshape(B, S, D)

        # Replicated region again (identical inputs on every rank).
        deflected = blk.out_proj(o_full)
        return x + blk.dropout(deflected) + blk.ffn(x_norm)

    # ------------------------------------------------------------------
    def init_state(self, batch_size: int, device=None, dtype=None):
        """O(1) recurrent state for the LOCAL head shard (1/world of total)."""
        if device is None:
            device = next(self.block.parameters()).device
        if dtype is None:
            dtype = next(self.block.parameters()).dtype
        d = self.block.d_head
        state = {"S": torch.zeros(batch_size * self.h_local, d, d,
                                  device=device, dtype=dtype)}
        if self.block.conv_kernel:
            state["conv"] = torch.zeros(batch_size, self.block.d_model,
                                        self.block.conv_kernel - 1,
                                        device=device, dtype=dtype)
        return state
