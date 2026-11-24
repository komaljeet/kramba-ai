"""#203: what would folding the Linear bias into the fp8 GEMM epilogue save?

Author: Rambarun Komaljeet

Production: y = _scaled_mm(x8, w8.T) (bf16 out), then y + bias.to(bf16) as a separate
broadcast add kernel (fp8_fused.fp8_module_forward). _scaled_mm takes a bias argument.
Times both at the three biased fp8 Linears of a DeltaNet layer (gate_proj, ffn[0],
ffn[3]; M = 4096 tokens) and reports how many output elements differ (the epilogue
rounds once, the add rounds twice, so bit-equality is NOT expected).
"""
import json, sys
import torch
dev = torch.device("cuda")
E4M3 = torch.float8_e4m3fn
one = torch.ones((), device=dev)


def t(fn, n=50):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    e0.record()
    for _ in range(n):
        fn()
    e1.record(); torch.cuda.synchronize()
    return e0.elapsed_time(e1) / n


res = {}
for name, (M, O, K) in {"gate_proj": (4096, 2048, 2048), "ffn0": (4096, 4096, 2048),
                        "ffn3": (4096, 2048, 4096)}.items():
    x = (torch.randn(M, K, device=dev) * 2).to(E4M3)
    w = (torch.randn(O, K, device=dev) * 2).to(E4M3)
    b = torch.randn(O, device=dev).bfloat16()
    sep = lambda: torch._scaled_mm(x, w.t(), scale_a=one, scale_b=one, out_dtype=torch.bfloat16) + b
    epi = lambda: torch._scaled_mm(x, w.t(), scale_a=one, scale_b=one, bias=b, out_dtype=torch.bfloat16)
    ys, ye = sep(), epi()
    res[name] = {"separate_ms": t(sep), "epilogue_ms": t(epi),
                 "frac_elems_differ": float((ys != ye).float().mean()),
                 "max_abs_diff": float((ys.float() - ye.float()).abs().max())}
    res[name]["saving_ms"] = res[name]["separate_ms"] - res[name]["epilogue_ms"]
    print(name, res[name], flush=True)
res["total_saving_ms_per_forward"] = sum(r["saving_ms"] for r in res.values() if isinstance(r, dict))
print(res["total_saving_ms_per_forward"])
json.dump(res, open(sys.argv[1], "w"), indent=1)
