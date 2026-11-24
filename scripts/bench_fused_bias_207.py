"""#207: per-call GPU time of the bias add and its consumers at the 1B shapes,
eager vs hvn/edge/fused_bias.py (forward and backward), CUDA events, median of 50.

Shapes: batch 1 x seq 4096 rows; gate_proj and ffn[3] out 2048, ffn[0] out 4096.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import torch
import torch.nn.functional as F
import hvn.edge.fused_bias as FB
from hvn.core.fused_gates import fused_residual_add3


def t_ms(fn, n=50):
    for _ in range(5):
        fn()
    ts = []
    for _ in range(n):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        ts.append(a.elapsed_time(b))
    return sorted(ts)[len(ts) // 2]


def main():
    dev = "cuda"
    M = 4096
    res = {}
    g = torch.Generator(device=dev).manual_seed(0)
    for name, O, act in (("gate silu", 2048, "silu"), ("ffn0 gelu", 4096, "gelu"),
                         ("ffn3 add", 2048, "none")):
        y = torch.randn(M, O, device=dev, generator=g).bfloat16()
        b = torch.randn(O, device=dev, generator=g).bfloat16()
        dy = torch.randn(M, O, device=dev, generator=g).bfloat16()
        eager_act = {"silu": F.silu, "gelu": F.gelu, "none": lambda t: t}[act]
        r = {"add_eager": t_ms(lambda: y + b), "add_fused": t_ms(lambda: FB.bias_add(y, b))}
        if act != "none":
            t = y + b
            r["act_eager"] = t_ms(lambda: eager_act(t))
            r["add+act_eager"] = r["add_eager"] + r["act_eager"]
            fn = FB.bias_silu if act == "silu" else FB.bias_gelu
            r["fused_fwd"] = t_ms(lambda: fn(y, b))
            # backward: eager act backward vs fused backward kernel
            if act == "silu":
                r["bwd_eager"] = t_ms(lambda: torch.ops.aten.silu_backward(dy, t))
            else:
                r["bwd_eager"] = t_ms(lambda: torch.ops.aten.gelu_backward(dy, t))
            yy = y.clone().requires_grad_()
            out = fn(yy, b)
            r["bwd_fused"] = t_ms(lambda: torch.autograd.grad(out, yy, dy, retain_graph=True))
        res[name] = r
    x = torch.randn(1, M, 2048, device=dev, generator=g)
    d = torch.randn(1, M, 2048, device=dev, generator=g).bfloat16()
    y = torch.randn(M, 2048, device=dev, generator=g).bfloat16()
    b = torch.randn(2048, device=dev, generator=g).bfloat16()
    f = (y + b).view(1, M, 2048)
    res["residual"] = {"add_eager": t_ms(lambda: y + b),
                       "add3_eager": t_ms(lambda: fused_residual_add3(x, d, f)),
                       "add3_bias_fused": t_ms(lambda: FB.add3_bias(x, d, y, b))}
    for k, v in res.items():
        print(k, {a: round(c, 4) for a, c in v.items()})
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results", "round2_207", "bench_fused_bias.json")
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
