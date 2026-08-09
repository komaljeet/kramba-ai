"""Faster-convergence optimizers for the EdgeHVN training loop (Issue #86).

This module adds training-time-only optimizer options that aim to reach a
target validation loss in *fewer* steps than AdamW — less total compute for the
same model quality.  Nothing here touches the DeltaNet/HVN forward pass, so the
HVN identity constraint holds: a model trained with these is the *same* HVN,
trained more cheaply.  Acceptance is convergence efficiency (steps-to-target)
with final val-loss parity vs. the fp32 AdamW baseline (epic #78 / issue #86).

Optimizers
----------
``Muon``
    Momentum-Orthogonalised-by-Newton-schulz (Keller Jordan, 2024).  Applies a
    Newton–Schulz orthogonalisation to the momentum of 2-D weight matrices so
    every update is (approximately) semi-orthogonal — spreading the step evenly
    across singular directions.  Vocabulary-sized matrices (embedding / decoder
    head), norms, biases, convs and any non-2-D tensor fall back to a standard
    AdamW update inside the *same* optimizer, so a single LR schedule drives
    both groups.

``Lion``
    EvoLved Sign Momentum (Chen et al., 2023).  Sign-of-interpolated-momentum
    update; ~half the optimizer memory of Adam (one state tensor) and often
    competitive convergence at a much smaller LR.

``GaLore``
    Gradient Low-Rank Projection (Zhao et al., 2024, arXiv:2403.03507). For
    each 2-D weight matrix, projects the gradient into a rank-``r`` subspace
    (via a periodically-refreshed SVD-derived projection matrix) *before*
    Adam tracks its moments — so Adam's ``exp_avg``/``exp_avg_sq`` state is
    ``O(r)`` instead of ``O(m*n)``. Unlike LoRA, the *weights* stay full-rank
    the whole time (only the optimizer's view of the gradient is
    compressed) -- no capacity loss, this is still full-parameter training.
    Vocabulary-sized matrices, norms and biases fall back to standard AdamW,
    same pattern as Muon above. Composes with #99 (block-coordinate-descent,
    shrinks *which* params need any optimizer state at all) and #100
    (activation checkpointing) — three independent, stackable memory axes.

Use :func:`build_optimizer` to construct any of ``adamw`` / ``muon`` / ``lion``
/ ``galore`` with the model's parameters correctly partitioned.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# Newton–Schulz orthogonalisation
# ---------------------------------------------------------------------------
def zeropower_via_newtonschulz5(G: Tensor, steps: int = 5, eps: float = 1e-7) -> Tensor:
    """Return an (approximately) semi-orthogonal matrix sharing G's column space.

    Quintic Newton–Schulz iteration (the coefficients ``(3.4445, -4.7750,
    2.0315)`` are Keller Jordan's tuned values that push the singular values
    toward 1 fastest).  Runs in bfloat16 for speed/stability; the caller casts
    the result back.  ``G`` must be a 2-D matrix.
    """
    assert G.ndim == 2, "Newton–Schulz expects a 2-D matrix"
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    # Normalise so the top singular value is <= 1 (iteration convergence region).
    X = X / (X.norm() + eps)
    # Operate on the shorter dimension for efficiency, transpose back after.
    transposed = G.size(0) > G.size(1)
    if transposed:
        X = X.mT
    for _ in range(steps):
        A = X @ X.mT
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transposed:
        X = X.mT
    return X


# ---------------------------------------------------------------------------
# Muon
# ---------------------------------------------------------------------------
class Muon(torch.optim.Optimizer):
    """Muon for 2-D weight matrices, with an AdamW fallback for everything else.

    Each parameter group carries a boolean ``use_muon``.  Muon groups get the
    momentum + Newton–Schulz orthogonalised update; non-Muon groups (embedding,
    decoder head, norms, biases, convs) get a standard decoupled-weight-decay
    AdamW update.  Keeping both in one optimizer means the shared cosine LR
    schedule scales every group proportionally.

    Parameters
    ----------
    momentum : float
        Heavy-ball momentum for the Muon groups.
    nesterov : bool
        Use Nesterov-style lookahead on the Muon momentum.
    ns_steps : int
        Newton–Schulz iteration count (5 is plenty).
    adamw_betas, adamw_eps : AdamW hyper-params for the fallback groups.
    """

    def __init__(self, param_groups, *, momentum: float = 0.95, nesterov: bool = True,
                 ns_steps: int = 5, adamw_betas=(0.9, 0.95), adamw_eps: float = 1e-8,
                 weight_decay: float = 0.0):
        defaults = dict(momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
                        adamw_betas=adamw_betas, adamw_eps=adamw_eps,
                        weight_decay=weight_decay)
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            wd = group["weight_decay"]
            lr = group["lr"]
            if group.get("use_muon", False):
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(g)
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(g)
                    update = g.add(buf, alpha=momentum) if nesterov else buf
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    # Shape-aware scale: keeps the RMS of the update ~constant
                    # across matrices of differing aspect ratios.
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(update.to(p.dtype), alpha=-lr * scale)
            else:
                beta1, beta2 = group["adamw_betas"]
                eps = group["adamw_eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bias1 = 1 - beta1 ** state["step"]
                    bias2 = 1 - beta2 ** state["step"]
                    denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(eps)
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(exp_avg, denom, value=-lr / bias1)
        return loss


# ---------------------------------------------------------------------------
# GaLore (Issue #84)
# ---------------------------------------------------------------------------
class GaLore(torch.optim.Optimizer):
    """GaLore for 2-D weight matrices, with an AdamW fallback for everything else.

    Each GaLore-managed parameter's gradient is projected into a rank-``r``
    subspace via a projection matrix ``proj`` (top-``r`` left singular
    vectors of the gradient if the matrix is tall, top-``r`` right singular
    vectors if wide — whichever gives the smaller compact shape). Adam's
    moments (``exp_avg``/``exp_avg_sq``) track that *compact* representation,
    not the full matrix — this is the entire memory win: ``O(r * min(m,n))``
    state instead of ``O(m * n)``. The normalized update is projected back to
    full rank and applied with a fixed scale factor (paper's ``alpha``,
    default 0.25) before the usual decoupled weight decay + LR.

    ``proj`` is refreshed every ``update_proj_gap`` steps via
    ``torch.svd_lowrank`` (a randomized approximate SVD — far cheaper than a
    full/economy SVD for the matrix sizes this project trains, and all that's
    needed since only the top-r directions are used). Refreshing changes the
    subspace the compact moments live in, so they are reset to zero on
    refresh — same rationale as BAdam (#99) resetting state on a block
    switch: stale moments in a now-irrelevant subspace would misdirect the
    next update.
    """

    def __init__(self, param_groups, *, betas=(0.9, 0.999), eps: float = 1e-8,
                 weight_decay: float = 0.0):
        defaults = dict(betas=betas, eps=eps, weight_decay=weight_decay,
                        rank=128, update_proj_gap=200, scale=0.25)
        super().__init__(param_groups, defaults)

    @staticmethod
    def _compute_projection(g: Tensor, rank: int) -> tuple[Tensor, str]:
        """Top-r singular-vector projection matrix + which side to project on.

        Returns (proj, side): side="left" -> proj is [m, r], project via
        proj.T @ g -> [r, n]. side="right" -> proj is [n, r], project via
        g @ proj -> [m, r]. Always picks the side giving the smaller compact
        shape (tall matrix -> project left, wide matrix -> project right).
        """
        m, n = g.shape
        r = min(rank, m, n)
        U, _S, V = torch.svd_lowrank(g.float(), q=r)
        if m >= n:
            return U[:, :r].to(g.dtype), "left"
        return V[:, :r].to(g.dtype), "right"

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            wd = group["weight_decay"]
            lr = group["lr"]
            if group.get("use_galore", False):
                rank = group["rank"]
                gap = group["update_proj_gap"]
                scale = group["scale"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["proj"] = None
                    state["step"] += 1

                    if state["proj"] is None or (state["step"] - 1) % gap == 0:
                        proj, side = self._compute_projection(g, rank)
                        state["proj"] = proj
                        state["side"] = side
                        r_shape = (proj.shape[1], g.shape[1]) if side == "left" \
                            else (g.shape[0], proj.shape[1])
                        state["exp_avg"] = torch.zeros(r_shape, device=g.device, dtype=g.dtype)
                        state["exp_avg_sq"] = torch.zeros(r_shape, device=g.device, dtype=g.dtype)
                        state["moment_step"] = 0

                    proj, side = state["proj"], state["side"]
                    R = (proj.T @ g) if side == "left" else (g @ proj)

                    state["moment_step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(R, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(R, R, value=1 - beta2)
                    bias1 = 1 - beta1 ** state["moment_step"]
                    bias2 = 1 - beta2 ** state["moment_step"]
                    denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(eps)
                    update_compact = exp_avg / denom  # still needs /bias1, folded into alpha below

                    full_update = (proj @ update_compact) if side == "left" \
                        else (update_compact @ proj.T)

                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.add_(full_update.to(p.dtype), alpha=-lr * scale / bias1)
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    state = self.state[p]
                    if "step" not in state:
                        state["step"] = 0
                        state["exp_avg"] = torch.zeros_like(p)
                        state["exp_avg_sq"] = torch.zeros_like(p)
                    state["step"] += 1
                    exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]
                    exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)
                    bias1 = 1 - beta1 ** state["step"]
                    bias2 = 1 - beta2 ** state["step"]
                    denom = (exp_avg_sq.sqrt() / (bias2 ** 0.5)).add_(eps)
                    if wd != 0:
                        p.mul_(1 - lr * wd)
                    p.addcdiv_(exp_avg, denom, value=-lr / bias1)
        return loss


# ---------------------------------------------------------------------------
# Lion
# ---------------------------------------------------------------------------
class Lion(torch.optim.Optimizer):
    """EvoLved Sign Momentum (Chen et al., 2023).

    One state tensor (momentum).  Update is the sign of an interpolation between
    the gradient and momentum; decoupled weight decay.  Rule of thumb: Lion's LR
    should be ~3-10x smaller than AdamW's and weight decay ~3-10x larger.
    """

    def __init__(self, params, lr: float = 1e-4, betas=(0.9, 0.99),
                 weight_decay: float = 0.0):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            lr = group["lr"]
            wd = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]
                if "exp_avg" not in state:
                    state["exp_avg"] = torch.zeros_like(p)
                exp_avg = state["exp_avg"]
                if wd != 0:
                    p.mul_(1 - lr * wd)
                update = exp_avg.mul(beta1).add_(g, alpha=1 - beta1).sign_()
                p.add_(update, alpha=-lr)
                exp_avg.mul_(beta2).add_(g, alpha=1 - beta2)
        return loss


# ---------------------------------------------------------------------------
# Partitioning + factory
# ---------------------------------------------------------------------------
def _is_muon_param(name: str, param: Tensor) -> bool:
    """A 2-D hidden weight matrix that benefits from orthogonalised updates.

    Excludes vocabulary-sized matrices (embedding / decoder head), any norm
    weights (some are 2-D, e.g. per-head ``read_norm_weight``), biases, convs
    and every non-2-D tensor — those take the AdamW fallback.
    """
    if param.ndim != 2:
        return False
    lname = name.lower()
    if "embedding" in lname or "decoder" in lname or "lm_head" in lname:
        return False
    if "norm" in lname:
        return False
    return True


def _is_galore_param(name: str, param: Tensor, rank: int) -> bool:
    """A 2-D hidden weight matrix large enough that rank-``rank`` projection
    actually compresses its optimizer state (min(m,n) > rank -- otherwise
    GaLore's compact representation would be no smaller than the full one).

    Same exclusions as Muon: vocabulary-sized matrices (embedding / decoder
    head) and norm weights fall back to standard AdamW -- GaLore's projected
    updates are a worse fit for those (embeddings in particular are already
    naturally sparse-gradient; the paper itself scopes GaLore to attention/
    feedforward weight matrices).
    """
    if param.ndim != 2:
        return False
    lname = name.lower()
    if "embedding" in lname or "decoder" in lname or "lm_head" in lname:
        return False
    if "norm" in lname:
        return False
    return min(param.shape) > rank


# ---------------------------------------------------------------------------
# Prodigy — parameter-free learning rate (Issue #134)
# ---------------------------------------------------------------------------
class Prodigy(torch.optim.Optimizer):
    """Prodigy (Mishchenko & Defazio, arXiv:2306.06101) — no LR to choose.

    Written by: Rambarun Komaljeet

    ``lr`` is passed as **1.0** and left there. The optimizer estimates the step
    size itself from observed gradients and keeps re-estimating as training
    proceeds, so the same invocation is correct for a from-scratch pretrain and
    for continuing from a trained checkpoint — the two regimes #133 showed need
    learning rates an order of magnitude apart.

    The estimate ``d`` approximates the distance from the initial point to the
    solution, which is the quantity a well-chosen learning rate is really
    standing in for. Per step:

        d_num <- beta3 * d_num + (d/d0) * dlr * <grad, p0 - p>
        s     <- beta3 * s     + (d/d0) * dlr * grad
        d_hat  = d_coef * d_num / sum|s|
        d     <- min(max(d_max, d_hat), d * growth_rate)

    ``d`` is monotone non-decreasing by construction (it tracks ``d_max``), so it
    ramps up from a deliberately tiny ``d0`` rather than starting somewhere
    arbitrary and having to come back down. **That ramp is the warmup** — a
    separate warmup schedule is not needed and would fight it.

    Two variants share this estimator, selected by ``use_muon`` on a param group:

    ``use_muon=False``
        The published Adam-family update. This is the reference implementation
        and the control: if it misbehaves, the estimator is at fault rather than
        the coupling below.

    ``use_muon=True``
        Muon's momentum + Newton-Schulz orthogonalised **direction**, with ``d``
        setting the **step size** in place of a hand-tuned ``muon_lr``. The two
        mechanisms are separable — Prodigy decides how far, Muon decides which
        way — and this repo measured Muon's direction as worth ~0.40 nats over
        Adam's (#86, #129), so replacing Muon wholesale to gain auto-tuning
        would trade away a known win.

        **This coupling is not in the paper and its convergence guarantees do
        not transfer.** Stated plainly because it is the standing risk in #134.
        The argument that it is even plausible: both updates are
        magnitude-normalised before ``dlr`` is applied — Adam's by ``sqrt(v)``,
        Muon's by Newton-Schulz orthogonalisation — so both consume a step size
        on roughly the same scale. The estimator is driven by the raw gradient
        in both variants, since ``d`` is a property of the loss landscape (how
        far away the solution is), not of how the step is formed.

    ``d`` is shared across ALL param groups, exactly as in the reference
    implementation: it is one global distance estimate, not a per-group one.
    """

    def __init__(self, param_groups, *, lr: float = 1.0, betas=(0.9, 0.999),
                 beta3: float | None = None, eps: float = 1e-8,
                 weight_decay: float = 0.0, d0: float = 1e-6, d_coef: float = 1.0,
                 growth_rate: float = float("inf"), safeguard_warmup: bool = False,
                 use_bias_correction: bool = True,
                 momentum: float = 0.95, nesterov: bool = True, ns_steps: int = 5):
        if lr <= 0.0:
            raise ValueError(
                f"Prodigy lr must be > 0 — it is a multiplier on the ESTIMATED "
                f"step, not a learning rate, and 1.0 is the intended value; "
                f"got {lr}")
        defaults = dict(
            lr=lr, betas=betas, beta3=beta3, eps=eps, weight_decay=weight_decay,
            d=d0, d0=d0, d_max=d0, d_hat=d0, d_numerator=0.0, d_denom=0.0,
            d_coef=d_coef, growth_rate=growth_rate,
            safeguard_warmup=safeguard_warmup,
            use_bias_correction=use_bias_correction, k=0,
            momentum=momentum, nesterov=nesterov, ns_steps=ns_steps,
            use_muon=False,
        )
        super().__init__(param_groups, defaults)

    @property
    def d(self) -> float:
        """The current distance estimate — the number a human would have tuned."""
        return self.param_groups[0]["d"]

    @property
    def effective_lr(self) -> float:
        """``d * lr``: what a hand-set step size would have had to be."""
        g = self.param_groups[0]
        return g["d"] * g["lr"]

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        g0 = self.param_groups[0]
        beta1, beta2 = g0["betas"]
        beta3 = g0["beta3"] if g0["beta3"] is not None else math.sqrt(beta2)
        k = g0["k"]
        d, d_max, d0 = g0["d"], g0["d_max"], g0["d0"]
        # One shared step size across groups, so no group can be silently scaled
        # relative to another the way a hand-set per-group LR can be.
        lr = max(pg["lr"] for pg in self.param_groups)

        if g0["use_bias_correction"]:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1.0
        dlr = d * lr * bias_correction

        d_numerator = g0["d_numerator"] * beta3
        d_denom = 0.0

        # --- pass 1: accumulate the estimator -----------------------------
        for group in self.param_groups:
            safeguard = group["safeguard_warmup"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad
                state = self.state[p]
                if "s" not in state:
                    state["s"] = torch.zeros_like(p)
                    state["p0"] = p.detach().clone()
                    state["exp_avg"] = torch.zeros_like(p)
                    state["exp_avg_sq"] = torch.zeros_like(p)
                    if group.get("use_muon", False):
                        state["momentum_buffer"] = torch.zeros_like(p)
                s, p0 = state["s"], state["p0"]

                # d/d0 rather than d alone keeps the accumulators off denormals
                # in the early steps, where d is still ~1e-6.
                d_numerator += (d / d0) * dlr * float(
                    torch.dot(grad.flatten().float(), (p0 - p).flatten().float()))

                state["exp_avg"].mul_(beta1).add_(grad, alpha=d * (1 - beta1))
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    grad, grad, value=d * d * (1 - beta2))

                s.mul_(beta3).add_(grad, alpha=((d / d0) * d) if safeguard
                                   else ((d / d0) * dlr))
                d_denom += float(s.abs().sum())

        if d_denom == 0.0:
            # Every gradient was zero. Dividing here would produce a nan that
            # silently poisons d for the rest of the run, and d is global.
            for group in self.param_groups:
                group["k"] = k + 1
            return loss

        d_hat = g0["d_coef"] * d_numerator / d_denom
        if d == d0:
            d = max(d, d_hat)
        d_max = max(d_max, d_hat)
        d = min(d_max, d * g0["growth_rate"])

        # --- pass 2: apply the update -------------------------------------
        for group in self.param_groups:
            group["d"] = d
            group["d_max"] = d_max
            group["d_hat"] = d_hat
            group["d_numerator"] = d_numerator
            group["d_denom"] = d_denom
            group["k"] = k + 1

            wd = group["weight_decay"]
            eps = group["eps"]
            if group.get("use_muon", False):
                momentum = group["momentum"]
                nesterov = group["nesterov"]
                ns_steps = group["ns_steps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    buf = state["momentum_buffer"]
                    buf.mul_(momentum).add_(p.grad)
                    update = p.grad.add(buf, alpha=momentum) if nesterov else buf
                    update = zeropower_via_newtonschulz5(update, steps=ns_steps)
                    # The same shape-aware scale Muon uses, so the two differ
                    # ONLY in where the step size comes from.
                    scale = max(1.0, p.size(0) / p.size(1)) ** 0.5
                    if wd != 0:
                        p.mul_(1 - dlr * wd)
                    p.add_(update.to(p.dtype), alpha=-dlr * scale)
            else:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    state = self.state[p]
                    denom = state["exp_avg_sq"].sqrt().add_(d * eps)
                    if wd != 0:
                        p.add_(p, alpha=-wd * dlr)
                    p.addcdiv_(state["exp_avg"], denom, value=-dlr)
        return loss


def build_optimizer(model, name: str, *, lr: float, weight_decay: float,
                    betas=(0.9, 0.95), eps: float = 1e-8, fused: bool = False,
                    muon_lr: float | None = None, muon_momentum: float = 0.95,
                    adamw_aux_lr: float | None = None,
                    galore_rank: int = 128, galore_update_gap: int = 200,
                    galore_scale: float = 0.25):
    """Build an optimizer by name: ``adamw`` | ``muon`` | ``lion`` | ``galore``.

    Returns ``(optimizer, description)``.  For ``muon`` the model's parameters
    are split into a Muon group (hidden 2-D matrices, LR ``muon_lr``) and an
    AdamW fallback group (embedding/head/norms/biases, LR ``adamw_aux_lr``); the
    shared cosine schedule scales both groups proportionally from their base LR.
    ``galore`` follows the same split pattern (GaLore group + AdamW fallback
    group), both driven by the single shared ``lr``.
    """
    name = name.lower()
    decay, no_decay = [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        # No weight decay on biases / norms / 1-D params (standard practice).
        if p.ndim <= 1 or "norm" in n.lower() or n.lower().endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)

    if name == "adamw":
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        opt = torch.optim.AdamW(groups, lr=lr, betas=betas, eps=eps, fused=fused)
        return opt, f"AdamW (lr={lr:g}, wd={weight_decay:g}, fused={fused})"

    if name == "lion":
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        opt = Lion(groups, lr=lr, weight_decay=weight_decay)
        return opt, f"Lion (lr={lr:g}, wd={weight_decay:g})"

    if name == "muon":
        muon_lr = muon_lr if muon_lr is not None else 0.02
        adamw_aux_lr = adamw_aux_lr if adamw_aux_lr is not None else lr
        muon_params, aux_decay, aux_no_decay = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _is_muon_param(n, p):
                muon_params.append(p)
            elif p.ndim <= 1 or "norm" in n.lower() or n.lower().endswith(".bias"):
                aux_no_decay.append(p)
            else:
                aux_decay.append(p)
        groups = [
            {"params": muon_params, "use_muon": True, "lr": muon_lr,
             "weight_decay": weight_decay},
            {"params": aux_decay, "use_muon": False, "lr": adamw_aux_lr,
             "weight_decay": weight_decay},
            {"params": aux_no_decay, "use_muon": False, "lr": adamw_aux_lr,
             "weight_decay": 0.0},
        ]
        opt = Muon(groups, momentum=muon_momentum, adamw_betas=betas, adamw_eps=eps)
        n_muon = sum(p.numel() for p in muon_params)
        n_aux = sum(p.numel() for p in aux_decay + aux_no_decay)
        desc = (f"Muon (muon_lr={muon_lr:g} on {len(muon_params)} matrices / "
                f"{n_muon:,} params; AdamW aux lr={adamw_aux_lr:g} / {n_aux:,} params)")
        return opt, desc

    if name in ("prodigy", "muon-prodigy", "muon_prodigy"):
        # Issue #134: no learning rate is chosen here. `lr` is deliberately
        # pinned to 1.0 and ignored as a tuning knob — Prodigy's own estimate
        # `d` supplies the step size. Passing a tuned lr would defeat the point,
        # so it is not threaded through.
        use_muon_dir = name != "prodigy"
        if use_muon_dir:
            muon_params, aux_decay, aux_no_decay = [], [], []
            for n, p in model.named_parameters():
                if not p.requires_grad:
                    continue
                if _is_muon_param(n, p):
                    muon_params.append(p)
                elif p.ndim <= 1 or "norm" in n.lower() or n.lower().endswith(".bias"):
                    aux_no_decay.append(p)
                else:
                    aux_decay.append(p)
            groups = [
                {"params": muon_params, "use_muon": True,
                 "weight_decay": weight_decay},
                {"params": aux_decay, "use_muon": False,
                 "weight_decay": weight_decay},
                {"params": aux_no_decay, "use_muon": False, "weight_decay": 0.0},
            ]
            n_muon = sum(p.numel() for p in muon_params)
            n_aux = sum(p.numel() for p in aux_decay + aux_no_decay)
            desc = (f"Muon-Prodigy (NO tuned LR; Muon direction on "
                    f"{len(muon_params)} matrices / {n_muon:,} params, Adam "
                    f"direction on {n_aux:,}; step size estimated)")
        else:
            groups = [
                {"params": decay, "use_muon": False, "weight_decay": weight_decay},
                {"params": no_decay, "use_muon": False, "weight_decay": 0.0},
            ]
            n_all = sum(p.numel() for p in decay + no_decay)
            desc = (f"Prodigy (NO tuned LR; Adam direction on {n_all:,} params; "
                    f"step size estimated)")
        opt = Prodigy(groups, lr=1.0, weight_decay=weight_decay, eps=eps,
                      momentum=muon_momentum)
        return opt, desc

    if name == "galore":
        galore_params, aux_decay, aux_no_decay = [], [], []
        for n, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if _is_galore_param(n, p, galore_rank):
                galore_params.append(p)
            elif p.ndim <= 1 or "norm" in n.lower() or n.lower().endswith(".bias"):
                aux_no_decay.append(p)
            else:
                aux_decay.append(p)
        groups = [
            {"params": galore_params, "use_galore": True, "lr": lr,
             "weight_decay": weight_decay, "rank": galore_rank,
             "update_proj_gap": galore_update_gap, "scale": galore_scale},
            {"params": aux_decay, "use_galore": False, "lr": lr,
             "weight_decay": weight_decay},
            {"params": aux_no_decay, "use_galore": False, "lr": lr,
             "weight_decay": 0.0},
        ]
        opt = GaLore(groups, betas=betas, eps=eps, weight_decay=weight_decay)
        n_galore = sum(p.numel() for p in galore_params)
        n_aux = sum(p.numel() for p in aux_decay + aux_no_decay)
        desc = (f"GaLore (rank={galore_rank}, update_gap={galore_update_gap}, "
                f"scale={galore_scale:g} on {len(galore_params)} matrices / "
                f"{n_galore:,} params; AdamW aux / {n_aux:,} params)")
        return opt, desc

    if name == "adafactor":
        # Issue #85: factored second-moment optimizer (Shazeer & Stern, 2018).
        # Standard AdamW keeps a full m x n exp_avg_sq per 2-D weight; Adafactor
        # factors it into a row vector + column vector (O(m+n) instead of
        # O(m*n)), and this project's platform (Windows, no WSL) can't reliably
        # install bitsandbytes for 8-bit Adam (no prebuilt Windows wheel path),
        # so this uses torch's own native `torch.optim.Adafactor` -- no extra
        # dependency, no build step. Training-time only, same HVN forward pass;
        # acceptance is val-loss parity vs the fp32 AdamW baseline (see
        # scripts/bench_harness.py, #79).
        groups = [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ]
        opt = torch.optim.Adafactor(groups, lr=lr)
        return opt, f"Adafactor (lr={lr:g}, factored second moment, native torch)"

    raise ValueError(f"Unknown optimizer '{name}' (expected adamw|muon|lion|galore|adafactor)")
