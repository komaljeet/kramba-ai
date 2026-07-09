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

Use :func:`build_optimizer` to construct any of ``adamw`` / ``muon`` / ``lion``
with the model's parameters correctly partitioned.
"""

from __future__ import annotations

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


def build_optimizer(model, name: str, *, lr: float, weight_decay: float,
                    betas=(0.9, 0.95), eps: float = 1e-8, fused: bool = False,
                    muon_lr: float | None = None, muon_momentum: float = 0.95,
                    adamw_aux_lr: float | None = None):
    """Build an optimizer by name: ``adamw`` | ``muon`` | ``lion``.

    Returns ``(optimizer, description)``.  For ``muon`` the model's parameters
    are split into a Muon group (hidden 2-D matrices, LR ``muon_lr``) and an
    AdamW fallback group (embedding/head/norms/biases, LR ``adamw_aux_lr``); the
    shared cosine schedule scales both groups proportionally from their base LR.
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

    raise ValueError(f"Unknown optimizer '{name}' (expected adamw|muon|lion)")
