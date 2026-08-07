"""Does 355M fit WITHOUT --block-coordinate under a cheaper optimizer? (Issue #112)

The BCD quality A/B measured a large cost: at matched steps, block-coordinate lands
at val loss 6.19 against the full model's 4.83 (~4x worse perplexity). BCD is only in
the plan because d=1024/24L at the real vocab needs 6.34 GiB reserved with Muon on a
5.9966 GiB card. If a cheaper optimizer closes that ~0.4 GiB gap, BCD -- and its
quality cost -- can be dropped entirely.

`--optimizer adafactor` has been wired since Issue #85 (native `torch.optim.Adafactor`,
factoring each 2-D weight's second moment into a row + column vector, O(m+n) instead of
O(m*n); bitsandbytes 8-bit was rejected as unreliable on Windows). #85's own closing
note records that its VRAM benchmark was deferred because the GPU was busy, and the
issue was then closed without it ever running. So this number has never been measured.

**Built through the REAL path, deliberately.** `build_model_and_config` +
`apply_ternary_to_model` + `use_triton_scan`, exactly as `train_edge_conversational.py`
does it -- not a standalone `EdgeHVN(...)`. Every VRAM envelope in docs/training.md was
wrong precisely because the benches hardcoded `VOCAB = 8000` and skipped ternary and the
Triton scan, so a fresh bench that repeats that mistake would be worthless.

**Reports `max_memory_reserved`, not `nvidia-smi`.** WDDM silently backs overflow with
host RAM instead of OOM-ing, so a spilling config looks healthy in `nvidia-smi` while
running 12x slower. `reserved` crossing the card's capacity is the actual fit criterion.

One optimizer per subprocess (allocator state does not fully reset in-process).

Usage:
    python scripts/probe_optimizer_vram.py --optimizer adafactor
    python scripts/probe_optimizer_vram.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# The real 355M pretrain shape (docs/training.md's "ceiling config", real vocab).
CFG = dict(d_model=1024, num_layers=24, n_heads=8, batch_size=4, seq_len=512,
           grad_accum=8)   # build_model_and_config reads this into its config dict
N_MICRO = 6          # enough to materialise optimizer state (it allocates on step 1)


def probe(optimizer_name: str, grad_checkpoint: bool, block_coordinate: bool) -> dict:
    import torch
    from transformers import GPT2Tokenizer

    from train_edge_conversational import build_model_and_config
    from hvn.optimizers import build_optimizer

    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()
    tok = GPT2Tokenizer.from_pretrained("gpt2")

    settings = dict(CFG)
    model, _cfg = build_model_and_config(
        settings, tok, device, checkpoint_dir=".",
        arch="hvn", apply_ternary=True,
        grad_checkpoint=grad_checkpoint, optimizer=optimizer_name, precision="bf16",
    )
    # Match the training script's scan wiring (train_edge_conversational.py:2481).
    for blk in (model.blocks or []):
        if hasattr(blk, "use_triton_scan"):
            blk.use_triton_scan = True
            blk.use_megakernel_scan = True

    n_params = sum(p.numel() for p in model.parameters())
    opt, desc = build_optimizer(model, optimizer_name, lr=3e-4, weight_decay=0.01)

    bcd = None
    if block_coordinate:
        from hvn.core.block_coordinate import BlockCoordinateScheduler
        bcd = BlockCoordinateScheduler(model, opt, steps_per_block=100)

    B, S = CFG["batch_size"], CFG["seq_len"]
    x = torch.randint(0, 50257, (B, S), device=device)
    y = torch.randint(0, 50257, (B, S), device=device)
    crit = torch.nn.CrossEntropyLoss()

    times = []
    for i in range(N_MICRO):
        t0 = time.time()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(x)
            loss = crit(logits.reshape(-1, 50257).float(), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()                      # optimizer state materialises here
        if bcd is not None:
            bcd.step()
        torch.cuda.synchronize()
        times.append(time.time() - t0)

    total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    peak_alloc = torch.cuda.max_memory_allocated() / 1024**3
    peak_res = torch.cuda.max_memory_reserved() / 1024**3
    return {
        "optimizer": optimizer_name, "desc": desc,
        "grad_checkpoint": grad_checkpoint, "block_coordinate": block_coordinate,
        "params": n_params,
        "peak_alloc_gib": round(peak_alloc, 3),
        "peak_reserved_gib": round(peak_res, 3),
        "card_total_gib": round(total, 4),
        "fits": bool(peak_res < total),
        # Steady-state (drop the first, which pays one-time allocation/warmup).
        "s_per_micro": round(sum(times[1:]) / max(1, len(times) - 1), 3),
        "s_per_micro_first": round(times[0], 3),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", default="adafactor")
    ap.add_argument("--grad-checkpoint", action="store_true", default=True)
    ap.add_argument("--block-coordinate", action="store_true")
    ap.add_argument("--all", action="store_true",
                    help="run muon/adafactor/adamw, each in its own subprocess")
    ap.add_argument("--json-out", default="results/optimizer_vram_355m.json")
    ap.add_argument("--_child", action="store_true", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.all and not args._child:
        results = []
        for name in ("muon", "adafactor", "adamw"):
            print(f"\n{'='*64}\n  PROBE: {name}\n{'='*64}", flush=True)
            cmd = [sys.executable, "-u", __file__, "--optimizer", name, "--_child",
                   "--json-out", ""]
            if args.block_coordinate:
                cmd.append("--block-coordinate")
            r = subprocess.run(cmd, capture_output=True, text=True)
            print(r.stdout[-3000:] or r.stderr[-3000:], flush=True)
            for line in r.stdout.splitlines():
                if line.startswith("RESULT_JSON:"):
                    results.append(json.loads(line[len("RESULT_JSON:"):]))
        if args.json_out:
            os.makedirs(os.path.dirname(args.json_out) or ".", exist_ok=True)
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(results, f, indent=2)
        print(f"\n{'='*64}\n  SUMMARY (card = "
              f"{results[0]['card_total_gib'] if results else '?'} GiB)\n{'='*64}")
        print(f"  {'optimizer':12s} {'params':>13s} {'alloc':>9s} {'reserved':>9s} "
              f"{'s/micro':>8s}  fits")
        for r in results:
            print(f"  {r['optimizer']:12s} {r['params']:13,d} "
                  f"{r['peak_alloc_gib']:8.2f}G {r['peak_reserved_gib']:8.2f}G "
                  f"{r['s_per_micro']:8.2f}  {'YES' if r['fits'] else 'NO -- SPILLS'}")
        return

    res = probe(args.optimizer, args.grad_checkpoint, args.block_coordinate)
    print("RESULT_JSON:" + json.dumps(res))
    print(f"  {res['optimizer']}: {res['params']:,} params | "
          f"alloc {res['peak_alloc_gib']:.2f} GiB | reserved {res['peak_reserved_gib']:.2f} GiB "
          f"| card {res['card_total_gib']:.2f} GiB | "
          f"{'FITS' if res['fits'] else 'SPILLS'} | {res['s_per_micro']:.2f} s/micro")


if __name__ == "__main__":
    main()
