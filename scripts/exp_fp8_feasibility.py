"""FP8 pretrain feasibility, Phase 1 (exp/fp8-pretrain-feasibility).

EXPERIMENTAL, not wired into any trainer. Question: can this architecture
(DeltaNet + sliding attention hybrid, ternary BitNet weights, fp32 decay gates)
train stably with its GEMMs in FP8, versus the same model in bf16?

FP8 recipe ("tensorwise dynamic", the torchao/TE default): per-tensor amax
scaling recomputed every call. Forward x, w -> e4m3; backward grad -> e5m2;
all three GEMMs (y, dX, dW) run through torch._scaled_mm with fp32 accumulate,
bf16 out. Bias and its grad stay in high precision. Master weights, optimizer
state, norms, the scan, attention (SDPA) and the loss are untouched: this
tests FP8 *compute*, not FP8 *storage*.

Arms (--arm):
  bf16        reference (production autocast path)
  fp8_all     every Linear in fp8, incl. alpha/beta gates and the LM head
  fp8_mixed   fp8 except alpha_proj / beta_proj / decoder (stay bf16)
  fp8_noscale CONTROL that must fail: fp8 casts with NO scaling. Exists only to
              prove this harness can see an fp8 failure (grads underflow).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Must be imported before CUDA initialises: it pulls in pyarrow/pandas, which
# crash natively (exit 127, no traceback) if loaded after CUDA on this machine.
from train_edge_conversational import apply_fast_stack  # noqa: E402
from hvn.edge.model import EdgeHVN  # noqa: E402
from hvn.edge.ternary import TernaryLinear, apply_ternary_to_model  # noqa: E402
from hvn.optimizers import build_optimizer  # noqa: E402

E4M3, E5M2 = torch.float8_e4m3fn, torch.float8_e5m2
E4M3_MAX, E5M2_MAX = 448.0, 57344.0
FP8_CALLS = {"fwd": 0, "bwd": 0}


def _q(t: torch.Tensor, dtype, maxv: float, scaled: bool):
    """Quantize to fp8. Returns (q, dequant_scale fp32 scalar tensor)."""
    if not scaled:
        one = torch.ones((), device=t.device, dtype=torch.float32)
        return t.float().clamp(-maxv, maxv).to(dtype), one
    amax = t.detach().abs().amax().float().clamp(min=1e-12)
    s = maxv / amax
    return (t.float() * s).clamp(-maxv, maxv).to(dtype), (1.0 / s)


def _colmajor(t: torch.Tensor) -> torch.Tensor:
    return t.t().contiguous().t()


class _FP8MM(torch.autograd.Function):
    """y = x @ w.T with x[N,K], w[O,K] (bf16 in, bf16 out), all GEMMs fp8."""

    @staticmethod
    def forward(ctx, x, w, scaled):
        xq, xs = _q(x, E4M3, E4M3_MAX, scaled)
        wq, ws = _q(w, E4M3, E4M3_MAX, scaled)
        try:
            y = torch._scaled_mm(xq, wq.t(), scale_a=xs, scale_b=ws,
                                 out_dtype=torch.bfloat16)
        except RuntimeError:
            print("scaled_mm fail", tuple(xq.shape), xq.stride(), xq.data_ptr() % 256,
                  tuple(wq.shape), wq.stride(), wq.data_ptr() % 256, xs, ws, flush=True)
            raise
        ctx.save_for_backward(xq, xs, wq, ws)
        ctx.scaled = scaled
        FP8_CALLS["fwd"] += 1
        return y

    @staticmethod
    def backward(ctx, g):
        xq, xs, wq, ws = ctx.saved_tensors
        gq, gs = _q(g.contiguous(), E5M2, E5M2_MAX, ctx.scaled)
        dx = torch._scaled_mm(gq, _colmajor(wq), scale_a=gs, scale_b=ws,
                              out_dtype=torch.bfloat16)
        dw = torch._scaled_mm(gq.t().contiguous(), _colmajor(xq), scale_a=gs,
                              scale_b=xs, out_dtype=torch.bfloat16)
        FP8_CALLS["bwd"] += 1
        return dx, dw, None


def fp8_linear(x: torch.Tensor, w: torch.Tensor, b, scaled: bool = True):
    lead = x.shape[:-1]
    K = x.shape[-1]
    O = w.shape[0]
    # contiguous: at 896 wide the fused conv hands over a column-major xc
    # (stride (1, N)) and _scaled_mm rejects a column-major A.
    x2 = x.reshape(-1, K).to(torch.bfloat16).contiguous()
    wb = w.to(torch.bfloat16)
    pad = (-O) % 16  # _scaled_mm needs every dim % 16 == 0
    if pad:
        wb = F.pad(wb, (0, 0, 0, pad))
    with torch.autocast("cuda", enabled=False):
        y = _FP8MM.apply(x2, wb, scaled)
    if pad:
        y = y[:, :O]
    if b is not None:
        y = y + b.to(torch.bfloat16)
    return y.reshape(*lead, O)


def install_fp8(model: nn.Module, exempt: tuple[str, ...], scaled: bool) -> list[str]:
    done = []
    for name, mod in model.named_modules():
        if not isinstance(mod, (nn.Linear, TernaryLinear)):
            continue
        if any(e in name for e in exempt):
            continue
        if isinstance(mod, TernaryLinear):
            def fwd(x, _m=mod):
                return fp8_linear(x, _m._dequantized(torch.bfloat16), _m.bias, scaled)
        else:
            def fwd(x, _m=mod):
                return fp8_linear(x, _m.weight, _m.bias, scaled)
        mod.forward = fwd
        done.append(name)
    return done


def build(args, seed: int):
    torch.manual_seed(seed)
    attn = set(int(a) for a in args.attn_layers.split(",") if a != "")
    plan = ["transformer" if i in attn else "deltanet" for i in range(args.layers)]
    model = EdgeHVN(vocab_size=50257, d_model=args.d_model, num_layers=args.layers,
                    n_heads=args.n_heads, use_deltanet=False, deltanet_conv_kernel=4,
                    block_plan=plan, dropout=0.0, eos_token_id=50256)
    lin = [n for n, m in model.named_modules() if isinstance(m, nn.Linear)]
    tgt = [n for n in lin if not any(p in n for p in ("beta_proj", "alpha_proj", "decoder"))]
    apply_ternary_to_model(model, target_names=tgt)
    model = model.cuda()
    apply_fast_stack(model, triton_scan=True, megakernel_scan=True, tf32=True,
                     compile_model=False, verbose=False, fused_ternary=True,
                     fused_rmsnorm=True)
    return model, plan


class Data:
    def __init__(self, path, seq, batch, seed):
        self.mm = np.memmap(path, dtype=np.uint16, mode="r")
        self.seq, self.batch = seq, batch
        self.rng = np.random.default_rng(seed)
        self.train_hi = min(len(self.mm) - seq - 2, 2_000_000_000)
        vrng = np.random.default_rng(12345)
        lo = len(self.mm) - 200_000_000
        self.val_starts = vrng.integers(lo, len(self.mm) - seq - 2, size=args_val_batches * batch)

    def _rows(self, starts):
        a = np.stack([np.asarray(self.mm[s:s + self.seq + 1], dtype=np.int64) for s in starts])
        t = torch.from_numpy(a).cuda(non_blocking=True)
        return t[:, :-1], t[:, 1:]

    def train_batch(self):
        return self._rows(self.rng.integers(0, self.train_hi, size=self.batch))

    def val_batches(self):
        for i in range(0, len(self.val_starts), self.batch):
            yield self._rows(self.val_starts[i:i + self.batch])


args_val_batches = 16


def loss_fn(model, x, y):
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
    return F.cross_entropy(logits.float().view(-1, logits.size(-1)), y.reshape(-1))


@torch.no_grad()
def evaluate(model, data, probes):
    model.eval()
    for p in probes.values():
        p["on"] = True
        p["sum"] = 0.0
        p["n"] = 0
    tot, n = 0.0, 0
    for x, y in data.val_batches():
        tot += loss_fn(model, x, y).item()
        n += 1
    for p in probes.values():
        p["on"] = False
    model.train()
    return tot / n, {k: (p["sum"] / max(p["n"], 1)) for k, p in probes.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True,
                    choices=["bf16", "fp8_all", "fp8_mixed", "fp8_noscale"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--n-heads", type=int, default=4)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--attn-layers", default="2")
    ap.add_argument("--seq", type=int, default=1024)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval-every", type=int, default=250)
    ap.add_argument("--data", default=r"C:\Users\komal\Documents\Codes\kramba-ai\data_cache\tokenized\fineweb_edu.bin")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, plan = build(args, args.seed)
    if args.arm == "fp8_all":
        fp8_mods = install_fp8(model, exempt=(), scaled=True)
    elif args.arm == "fp8_mixed":
        fp8_mods = install_fp8(model, exempt=("alpha_proj", "beta_proj", "decoder"), scaled=True)
    elif args.arm == "fp8_noscale":
        fp8_mods = install_fp8(model, exempt=("alpha_proj", "beta_proj", "decoder"), scaled=False)
    else:
        fp8_mods = []
    print(f"arm={args.arm} plan={plan} fp8 modules={len(fp8_mods)}", flush=True)
    print("  fp8:", fp8_mods, flush=True)

    # Gate probes: data-dependent mean alpha / beta on the val set.
    probes = {}
    for name, mod in model.named_modules():
        if name.endswith("alpha_proj") or name.endswith("beta_proj"):
            p = {"on": False, "sum": 0.0, "n": 0}
            probes[name] = p

            def hook(_m, _i, out, _p=p):
                if _p["on"]:
                    _p["sum"] += torch.sigmoid(out.float()).mean().item()
                    _p["n"] += 1
            mod.register_forward_hook(hook)
    alpha_bias = {n: p for n, p in model.named_parameters() if n.endswith("alpha_proj.bias")}
    alpha_w = {n: p for n, p in model.named_parameters() if n.endswith("alpha_proj.weight")}

    opt, desc = build_optimizer(model, "muon", lr=args.lr, weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]
    print("  opt:", desc, flush=True)

    def lr_scale(step):
        if step < args.warmup:
            return (step + 1) / args.warmup
        t = (step - args.warmup) / max(1, args.steps - args.warmup)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * t))

    data = Data(args.data, args.seq, args.batch, args.data_seed)
    step_f = open(os.path.join(args.out, "steps.csv"), "w", newline="")
    sw = csv.writer(step_f)
    sw.writerow(["step", "loss", "grad_norm", "lr_scale", "dt"])
    evals = []
    v0, g0 = evaluate(model, data, probes)
    evals.append({"step": 0, "val": v0, "gates": g0,
                  "alpha_bias_mean": {n: p.mean().item() for n, p in alpha_bias.items()}})
    print(f"step 0 val {v0:.4f}", flush=True)
    init_bias = {n: p.detach().clone() for n, p in alpha_bias.items()}
    nonfinite = 0
    t_start = time.time()
    torch.cuda.reset_peak_memory_stats()
    for step in range(args.steps):
        t0 = time.time()
        for g, b0 in zip(opt.param_groups, base):
            g["lr"] = b0 * lr_scale(step)
        x, y = data.train_batch()
        loss = loss_fn(model, x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        lv, gv = loss.item(), gn.item()
        if not (math.isfinite(lv) and math.isfinite(gv)):
            nonfinite += 1
        opt.step()
        if step == 0 and fp8_mods:
            # Must-fail guard: the fp8 path really ran, forward and backward.
            assert FP8_CALLS["fwd"] >= len(fp8_mods) and FP8_CALLS["bwd"] >= len(fp8_mods), FP8_CALLS
        if step == 0 and not fp8_mods:
            assert FP8_CALLS["fwd"] == 0
        sw.writerow([step + 1, f"{lv:.5f}", f"{gv:.5f}", f"{lr_scale(step):.4f}", f"{time.time() - t0:.4f}"])
        if (step + 1) % args.eval_every == 0 or step + 1 == args.steps:
            step_f.flush()
            v, gates = evaluate(model, data, probes)
            ab = {n: p.mean().item() for n, p in alpha_bias.items()}
            disp = {n: (p.detach() - init_bias[n]).abs().mean().item() for n, p in alpha_bias.items()}
            wn = {n: p.norm().item() for n, p in alpha_w.items()}
            evals.append({"step": step + 1, "val": v, "gates": gates, "alpha_bias_mean": ab,
                          "alpha_bias_mean_abs_disp": disp, "alpha_w_norm": wn,
                          "nonfinite_so_far": nonfinite})
            print(f"step {step + 1} loss {lv:.4f} val {v:.4f} gn {gv:.3f} "
                  f"abias {[round(a, 4) for a in ab.values()]} disp {[round(d, 4) for d in disp.values()]} "
                  f"alpha {[round(gates[k], 4) for k in gates if 'alpha' in k]} "
                  f"nonfinite {nonfinite} {time.time() - t_start:.0f}s", flush=True)
            with open(os.path.join(args.out, "summary.json"), "w") as f:
                json.dump({"args": vars(args), "fp8_modules": fp8_mods, "evals": evals,
                           "peak_mem_mib": torch.cuda.max_memory_allocated() / 2**20,
                           "wall_s": time.time() - t_start}, f, indent=1)
    step_f.close()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
