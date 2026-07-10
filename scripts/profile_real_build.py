"""
Decisive combination test + profiler diff.

The real train_edge_conversational.py builds the model via build_model_and_config
(EdgeHVN WITHOUT use_triton_scan at construction -> ternary applied to 84 Linears
-> .to(device)), then sets triton-scan post-hoc on each block, then torch.compile.
NONE of my isolates used that exact build: isolate_ternary passed use_triton_scan
at construction; isolate_trainloop built a bare EdgeHVN with no ternary.

This script uses the REAL build path (build_model_and_config with ternary + the
exact post-hoc triton-scan + compile sequence) and runs the SAME minimal loop my
isolates use, both timed AND under torch.profiler so I can diff the kernel table
against the real script's profile_run.log.

  - If this reproduces ~0.4 steps/s -> ternary-in-the-real-build is the cause.
  - If this gives ~0.81 steps/s      -> the cause is in the real train() loop /
    main() flow, NOT the build, and I diff profiles to find the extra GPU work.

    python scripts/profile_real_build.py
"""
from __future__ import annotations

import os, sys, time
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
import torch

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
import train_edge_conversational as T
from hvn.edge.model import EdgeHVN  # noqa: F401  (ensures import path)

D_MODEL, NUM_LAYERS, N_HEADS = 512, 12, 8
BATCH, SEQ, GRAD_ACCUM, VOCAB = 4, 512, 8, T.GPT2_VOCAB_SIZE
DEV = torch.device("cuda")
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

SETTINGS = {
    "d_model": D_MODEL, "num_layers": NUM_LAYERS, "n_heads": N_HEADS,
    "batch_size": BATCH, "grad_accum": GRAD_ACCUM, "seq_len": SEQ,
}


def real_build():
    """Exact real path: build_model_and_config (ternary) -> post-hoc triton-scan -> compile."""
    model, _cfg = T.build_model_and_config(
        settings=SETTINGS, tokenizer=None, device=DEV,
        checkpoint_dir=os.path.join(_ROOT, "checkpoints", "profile_real_build"),
        arch="hvn", apply_ternary=True,
    )
    # Post-hoc triton-scan enable (mirrors main() lines ~1772-1780).
    for blk in model.blocks:
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
    print(f"\n  [probe] post-hoc triton-scan set on {sum(hasattr(b,'use_triton_scan') for b in model.blocks)} blocks")
    return torch.compile(model)


def measure(n_warmup=8, n_time=20, do_profile=True):
    print("\n" + "=" * 72)
    print("REAL-BUILD ISOLATE — real build_model_and_config (ternary) + post-hoc triton + compile")
    print("=" * 72)
    model = real_build()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01, fused=True)
    inps = [torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV) for _ in range(GRAD_ACCUM)]
    tgts = [torch.randint(0, VOCAB, (BATCH, SEQ), device=DEV) for _ in range(GRAD_ACCUM)]

    def step():
        opt.zero_grad(set_to_none=True)
        for i in range(GRAD_ACCUM):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inps[i])
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, VOCAB), tgts[i].reshape(-1))
            (loss / GRAD_ACCUM).backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    print(f"  warming {n_warmup} steps (compile)...")
    for _ in range(n_warmup):
        step()
    torch.cuda.synchronize()

    # Time
    t0 = time.perf_counter()
    for _ in range(n_time):
        step()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n_time * 1e3
    sps = 1000.0 / ms
    print(f"  -> {ms:.0f} ms/step  =  {sps:.2f} steps/s")

    # Profile a few steps for the kernel diff vs the real script
    if do_profile:
        import torch.profiler as _tp
        print("\n  [probe] profiling 8 steps under torch.profiler for kernel diff...")
        _prof = _tp.profile(activities=[_tp.ProfilerActivity.CPU, _tp.ProfilerActivity.CUDA],
                            schedule=_tp.schedule(wait=2, warmup=1, active=8, repeat=1))
        _prof.__enter__()
        try:
            for _ in range(11):  # 2 wait + 1 warmup + 8 active
                step()
                _prof.step()
        finally:
            print("\n" + "=" * 78)
            print("[probe] REAL-BUILD kernel breakdown (self GPU time, top 25)")
            print("=" * 78)
            try:
                print(T._fmt_profile_table(_prof, row_limit=25))
            except Exception as _e:
                print(f"[probe] render failed: {_e!r}")
            _prof.__exit__(None, None, None)

    del model, opt
    torch.cuda.empty_cache()
    return sps, ms


if __name__ == "__main__":
    assert torch.cuda.is_available()
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    sps, ms = measure()
    print("\n" + "=" * 72)
    print("VERDICT — real-build + minimal loop")
    print("=" * 72)
    print(f"  this run    : {sps:.2f} steps/s  ({ms:.0f} ms/step)")
    print(f"  real script : 0.40 steps/s  (2480 ms/step)")
    print(f"  my isolates : 0.81 steps/s  (1223 ms/step)")
    print(f"  -> reproduces 0.4? {'YES (ternary-in-real-build is the cause)' if sps < 0.55 else 'NO (cause is in the real loop/main flow)'}")
    print("=" * 72)