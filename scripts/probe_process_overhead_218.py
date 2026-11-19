"""#218 gate 1, step 2: what is the host-RAM floor of a training process that keeps
no model state in host RAM? Working set after each stage of start-up.

Author: Rambarun Komaljeet

Stages: python start -> import torch -> CUDA context -> import the trainer module
(pandas / pyarrow / transformers come in here) -> one production 1B DeltaNet layer
built ON THE GPU (no CPU copy kept) -> one fwd+bwd of one micro-batch under the
production --fast --fp8-linear stack (Triton / cuBLAS / fp8 kernels load) -> one
Muon step -> the same for the attention layer -> gc.

#217's bench reached 3.79 GB working set with 1.61 GB of host state, i.e. 2.18 GB
of "everything else"; #216 took 3.07 GB from the trainer. This says which part
is the irreducible CUDA/torch floor and which part is imports / kernels / data.
"""
import os, sys, json, time, gc

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO); sys.path.insert(0, os.path.join(REPO, "scripts"))
os.chdir(REPO)
from bench_streamed_group_217 import host_mem

rows = []


def stage(name):
    gc.collect()
    m = host_mem()
    rows.append({"stage": name, **m})
    print(f"[o218] {name:34s} ws {m['ws']:.3f} GB  private {m['private']:.3f} GB  "
          f"sys avail {m['sys_avail']:.2f}", flush=True)


def main():
    out = sys.argv[1]
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")
    stage("python start")
    import torch
    import torch.nn as nn
    stage("import torch")
    # trainer import BEFORE any CUDA call: pyarrow loaded after CUDA init crashes
    # this venv (heap corruption, see bench_streamed_group_217.py; the first run
    # of this probe died there with exit 127).
    import train_edge_conversational as T
    stage("import trainer module")
    torch.zeros(1, device="cuda"); torch.cuda.synchronize()
    stage("CUDA context")
    mods = {k: (k in sys.modules) for k in ("pandas", "pyarrow", "transformers", "datasets", "numpy")}
    print(f"[o218] loaded by trainer import: {mods}", flush=True)
    import hvn.core.deltanet_triton as DT
    import hvn.edge.fused_bias as FB
    from hvn.edge.ternary import apply_ternary_to_model
    from hvn.optimizers import build_optimizer
    from hvn.core.deltanet import DeltaNetBlock
    from hvn.core.transformer_block import TransformerBlock
    DT.SCAN_KERNEL[0] = "t"; FB.set_mode("fused")
    dev = torch.device("cuda")
    pos = torch.arange(4096, device=dev).unsqueeze(0)

    def one(kind):
        with torch.device(dev):
            blk = (TransformerBlock(d_model=2048, n_heads=64, dropout=0.0, window=4096) if kind == "attn"
                   else DeltaNetBlock(d_model=2048, n_heads=64, dropout=0.0, conv_kernel=4,
                                      use_triton_scan=False, moe_experts=1, moe_top_k=2,
                                      use_megakernel_scan=False, ffn_activation="gelu"))
        g = nn.Module(); g.blocks = nn.ModuleList([blk])
        targets = [n for n, m in g.named_modules() if isinstance(m, nn.Linear)
                   and not any(p in n for p in ("beta_proj", "alpha_proj", "decoder"))]
        apply_ternary_to_model(g, target_names=targets)
        T.apply_fast_stack(g, triton_scan=True, megakernel_scan=True, tf32=True, compile_model=False,
                           verbose=False, fused_ternary=True, ternary_step_cache=True,
                           fused_rmsnorm=True, fp8_linear=True)
        g.to(dev).train()
        assert all(p.is_cuda for p in g.parameters())
        stage(f"{kind} layer built on GPU")
        x = (torch.randn(1, 4096, 2048, device=dev) * 0.02).requires_grad_()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = blk(x, position_ids=pos)
        y.float().pow(2).mean().backward()
        torch.cuda.synchronize()
        stage(f"{kind} fwd+bwd (kernels loaded)")
        opt, _ = build_optimizer(blk, "muon", lr=3e-4, weight_decay=0.01, muon_lr=0.02)
        opt.step(); torch.cuda.synchronize()
        stage(f"{kind} Muon step")
        del g, blk, opt, x, y
        torch.cuda.empty_cache()

    one("deltanet")
    one("attn")
    stage("after freeing layers")
    json.dump({"rows": rows, "modules": mods}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
