"""Mixture-of-Experts FFN for the DeltaNet block (Issue #123).

Why this and not another memory lever
-------------------------------------
Training compute is ``~6 * params * tokens``, and Chinchilla scales tokens with
params, so cost grows as **N squared**. Decomposing this repo's measured
6B-vs-142M gap on a 6 GB laptop (3,300x total): 42x from more params, 42x from
more tokens, and only **~2x** from CPU-offload/PCIe inefficiency. The offload
overhead is the SMALLEST term -- making PCIe free would take 33 years down to
16 years, not to something feasible.

MoE is the only remaining lever that attacks **FLOPs per token** directly: it
buys capacity at sub-linear compute. With E experts and top-k routing, the FFN
holds ``E x`` the parameters while computing only ``k`` of them per token.

Scope: the FFN only
-------------------
This routes ONLY the block's feed-forward sub-layer (~40% of DeltaNet block
FLOPs: the FFN is 4d^2 of the block's ~10d^2). The recurrence -- q/k/v/alpha/
gate/out_proj -- stays dense, deliberately.

**Why not route the recurrence.** The DeltaNet state accumulates k/v from
*every* token::

    S_t = diag(alpha_t) S_{t-1} + beta_t (v_t - S_{t-1} k_t) k_t^T

If token A writes through expert-1's ``v_proj`` and token B through expert-3's,
the state mixes representations from different learned subspaces, and the next
token reads all of it back through a single query. **Transformers do not have
this failure mode** -- attention is stateless per layer, so per-token expert
routing is safe there. It is not obviously safe inside a recurrent state, and
this repo has twice found that "obviously fine" assumptions about the
recurrence were wrong (the #122 state-commit bug; the #109 trunk confound).
Measure before extending.

Known-answer property
---------------------
``num_experts=1, top_k=1`` is constructed to be **numerically identical** to the
dense FFN it replaces: the router's output is irrelevant when there is only one
expert to choose, and the gate value is forced to exactly 1.0 rather than
``softmax`` of a single logit (which would also be 1.0 mathematically but
introduces a spurious gradient path through the router). This gives every test
and sweep a configuration whose correct answer is known by construction -- the
check that an "is it wired up" test alone cannot provide. See the retraction in
``docs/state.md``: a bug that survived six experiments would have been caught
immediately by a cell like this.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MoEFFN(nn.Module):
    """Top-k routed mixture of FFN experts, drop-in for a DeltaNet block's ``ffn``.

    Parameters
    ----------
    d_model : int
        Model width. Input and output are both ``[..., d_model]``.
    num_experts : int
        Number of expert FFNs held. Storage scales with this; compute does not.
    top_k : int
        Experts evaluated per token. Compute scales with this.
    expansion : int
        Hidden width multiplier per expert. Matches the dense FFN it replaces
        (2 for the default GELU FFN, 3 for SwiGLU).
    dropout : float
        Dropout on the expert output, matching the dense FFN.
    use_swiglu : bool
        Use SwiGLU experts (``w2(silu(w1 x) * w3 x)``) instead of the default
        ``Linear -> GELU -> Linear``. Must match the block's dense FFN choice
        for the E=1 known-answer equivalence to hold.
    aux_loss_weight : float
        Weight of the load-balancing auxiliary loss (Shazeer et al. / Switch
        Transformer). Without it, routing collapses onto a few experts and the
        rest are dead weight -- the classic MoE failure. Exposed as
        ``self.aux_loss`` after each forward for the training loop to add.
    """

    def __init__(
        self,
        d_model: int,
        num_experts: int = 8,
        top_k: int = 2,
        expansion: int = 2,
        dropout: float = 0.1,
        use_swiglu: bool = False,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        if top_k > num_experts:
            raise ValueError(f"top_k={top_k} cannot exceed num_experts={num_experts}")
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_swiglu = use_swiglu
        self.aux_loss_weight = aux_loss_weight
        hidden = d_model * expansion

        # Router: one logit per expert. Zero-init so the initial distribution is
        # exactly uniform -- no expert is favoured before any learning, which
        # also makes the E=1 case trivially exact.
        self.router = nn.Linear(d_model, num_experts, bias=False)
        nn.init.zeros_(self.router.weight)

        if use_swiglu:
            self.w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden))
            self.w3 = nn.Parameter(torch.empty(num_experts, d_model, hidden))
            self.w2 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
            for w in (self.w1, self.w3, self.w2):
                for e in range(num_experts):
                    nn.init.xavier_uniform_(w[e])
        else:
            # Matches nn.Sequential(Linear(d,2d), GELU, Dropout, Linear(2d,d), Dropout)
            self.w1 = nn.Parameter(torch.empty(num_experts, d_model, hidden))
            self.b1 = nn.Parameter(torch.zeros(num_experts, hidden))
            self.w2 = nn.Parameter(torch.empty(num_experts, hidden, d_model))
            self.b2 = nn.Parameter(torch.zeros(num_experts, d_model))
            for e in range(num_experts):
                nn.init.xavier_uniform_(self.w1[e])
                nn.init.xavier_uniform_(self.w2[e])

        self.dropout = nn.Dropout(dropout)
        self.register_buffer("_zero", torch.zeros(()), persistent=False)
        self.aux_loss = None

    # ------------------------------------------------------------------
    def _expert_forward(self, x_flat: torch.Tensor, expert_idx: int) -> torch.Tensor:
        """Apply a single expert to ``[N, d_model]``."""
        if self.use_swiglu:
            h = F.silu(x_flat @ self.w1[expert_idx]) * (x_flat @ self.w3[expert_idx])
            return h @ self.w2[expert_idx]
        h = F.gelu(x_flat @ self.w1[expert_idx] + self.b1[expert_idx])
        return h @ self.w2[expert_idx] + self.b2[expert_idx]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``[B, S, D] -> [B, S, D]``, routing each token to its top-k experts."""
        orig_shape = x.shape
        x_flat = x.reshape(-1, self.d_model)  # [N, D]
        n_tokens = x_flat.shape[0]

        # --- E=1 fast path: provably identical to the dense FFN ---------------
        # With one expert there is nothing to route. Taking softmax over a
        # single logit would also yield 1.0, but it would create a gradient
        # path through the router that the dense FFN does not have, so the
        # known-answer equivalence would hold in forward but NOT in backward.
        # Skipping the router entirely keeps both exact.
        if self.num_experts == 1:
            out = self._expert_forward(x_flat, 0)
            self.aux_loss = self._zero
            return self.dropout(out).reshape(orig_shape)

        logits = self.router(x_flat)                         # [N, E]
        probs = F.softmax(logits, dim=-1)
        topk_probs, topk_idx = torch.topk(probs, self.top_k, dim=-1)  # [N, k]
        # Renormalise over the selected experts so each token's gates sum to 1
        # (Switch/Mixtral convention) -- otherwise output magnitude depends on
        # how confident the router happened to be.
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)

        out = torch.zeros_like(x_flat)
        for e in range(self.num_experts):
            # Tokens for which expert e is among the top-k, and its gate value.
            sel = (topk_idx == e)                            # [N, k]
            if not sel.any():
                continue
            token_ids, slot = sel.nonzero(as_tuple=True)
            gate = topk_probs[token_ids, slot].unsqueeze(-1)  # [n_e, 1]
            out.index_add_(0, token_ids, gate * self._expert_forward(x_flat[token_ids], e))

        # --- Load-balancing auxiliary loss ------------------------------------
        # Switch-Transformer form: E * sum_e (fraction of tokens routed to e) *
        # (mean router prob for e). Minimised when both are uniform. Without
        # this, routing collapses onto a handful of experts and the remaining
        # capacity is dead weight -- the standard MoE failure mode.
        with torch.no_grad():
            frac = torch.zeros(self.num_experts, device=x.device, dtype=probs.dtype)
            frac.index_add_(
                0, topk_idx.reshape(-1),
                torch.ones(topk_idx.numel(), device=x.device, dtype=probs.dtype),
            )
            frac = frac / max(n_tokens * self.top_k, 1)
        mean_prob = probs.mean(dim=0)                        # [E], differentiable
        self.aux_loss = self.aux_loss_weight * self.num_experts * (frac * mean_prob).sum()

        return self.dropout(out).reshape(orig_shape)

    # ------------------------------------------------------------------
    def active_param_count(self) -> int:
        """Parameters actually used per token (what training compute scales with).

        Total stored params scale with ``num_experts``; this scales with
        ``top_k``. The ratio is the whole point of the exercise.
        """
        per_expert = sum(
            p.numel() // self.num_experts
            for n, p in self.named_parameters()
            if n.startswith(("w1", "w2", "w3", "b1", "b2"))
        )
        return per_expert * self.top_k + self.router.weight.numel()


def collect_moe_aux_loss(model: nn.Module) -> torch.Tensor:
    """Sum ``aux_loss`` across every :class:`MoEFFN` in *model*.

    The training loop must add this to the task loss; MoE routing does not
    balance itself. Returns a scalar zero tensor if the model has no MoE
    layers, so call sites need no branching.
    """
    total = None
    for mod in model.modules():
        if isinstance(mod, MoEFFN) and mod.aux_loss is not None:
            total = mod.aux_loss if total is None else total + mod.aux_loss
    if total is None:
        return torch.zeros((), device=next(model.parameters()).device)
    return total
