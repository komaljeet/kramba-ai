"""Can a CUDA graph capture one EdgeHVN decode step? (Issue #138)

Written by: Rambarun Komaljeet

A probe, not an implementation. Decode is launch-bound — 533 launches/token with
the GPU idle ~87% of the time — and a graph collapses the whole per-token
sequence into a single replay. Everything else in #138 is a rounding error next
to that, and the in-place state work only exists to make capture possible.

**This repo has already seen CUDA graphs fail on this card**: the speed-ceiling
investigation records `reduce-overhead` / CUDA-graph modes crashing on
`nn.Embedding`. That was training, with dynamic shapes; decode has static ones,
which is the friendlier case. But it is a recorded prior failure, so this probe
exists to find out cheaply rather than after a day of plumbing.

Answers three things in order, stopping at the first that fails:

1. does capture even complete without raising?
2. does replay produce numerically identical logits to eager?
3. how many launches per token does a replay actually cost?

Usage
-----
    python scripts/probe_cuda_graph_decode.py --checkpoint checkpoints/hybrid_chat_v1.pt
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from train_edge_conversational import load_checkpoint_for_resume  # noqa: E402


def flatten_state(states) -> list:
    """Every tensor in the per-block state list, in a stable order."""
    out = []
    for st in states:
        if isinstance(st, dict):
            for k in sorted(st):
                v = st[k]
                if torch.is_tensor(v):
                    out.append(v)
        elif torch.is_tensor(st):
            out.append(st)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default="checkpoints/hybrid_chat_v1.pt")
    ap.add_argument("--warmup", type=int, default=16,
                    help="tokens before capture — must exceed the attention "
                         "window's fill so the KV ring is in its WRAPPED, "
                         "fixed-shape steady state (#138 approach 0)")
    ap.add_argument("--out", default="results/decode_launches/cuda_graph_probe.json")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA required (docs/ai_rules.md).")
    device = torch.device("cuda")
    result = {"checkpoint": args.checkpoint.replace("\\", "/")}

    model, _step, _ck = load_checkpoint_for_resume(args.checkpoint, device)
    model.eval()
    V = getattr(model, "vocab_size", 50257)

    # RoPE tables, so cos/sin come from an index_select on a DEVICE tensor
    # rather than a per-token CPU->GPU construction (#138).
    touched = model.enable_graph_decode()
    result["graph_prep"] = touched
    print(f"[graph] 0. prepared {touched['attention_blocks']} attention blocks, "
          f"RoPE table {touched['max_positions']} positions")

    # --- static buffers: the same tensors on every replay -------------------
    static_tok = torch.zeros((1, 1), dtype=torch.long, device=device)
    # Position lives on the DEVICE and is incremented in place, so the captured
    # graph indexes a moving value instead of a folded constant.
    static_pos = torch.zeros((), dtype=torch.long, device=device)
    states = model.init_state(batch_size=1, device=device)

    pos = 0
    with torch.no_grad():
        for _ in range(args.warmup):
            static_tok.fill_(int(torch.randint(0, V, (1,))))
            static_pos.fill_(pos)
            _lg, states = model.forward_step(static_tok, states,
                                             position_idx=static_pos)
            pos += 1
    torch.cuda.synchronize()

    ptrs_before = [t.data_ptr() for t in flatten_state(states)]
    result["state_tensors"] = len(ptrs_before)

    # --- 1. does capture complete? ----------------------------------------
    # A side stream is required by the CUDA graph API; the capture itself must
    # not allocate, which is what the in-place state work was for.
    graph = torch.cuda.CUDAGraph()
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            with torch.no_grad():
                for _ in range(3):          # settle allocator/autotune choices
                    model.forward_step(static_tok, states,
                                       position_idx=static_pos)
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        with torch.no_grad(), torch.cuda.graph(graph):
            static_logits, _captured_states = model.forward_step(
                static_tok, states, position_idx=static_pos)
        result["capture"] = "OK"
        print("[graph] 1. capture completed")
    except Exception as e:
        result["capture"] = f"{type(e).__name__}: {e}"
        result["traceback"] = traceback.format_exc()[-1800:]
        print(f"[graph] 1. capture FAILED: {type(e).__name__}: {e}")
        print("[graph]    -> graphs are not viable on this stack as written; the "
              "in-place state change should be reverted unless something else "
              "justifies it (#138 commit message says so explicitly).")
        _write(args.out, result)
        return 1

    # --- 2. is a replay numerically identical to eager? -------------------
    ptrs_after = [t.data_ptr() for t in flatten_state(states)]
    result["state_addresses_stable_through_capture"] = ptrs_before == ptrs_after

    tok_id = int(torch.randint(0, V, (1,)))
    eager_states = model.init_state(batch_size=1, device=device)
    with torch.no_grad():
        # Replay path
        static_tok.fill_(tok_id)
        graph.replay()
        torch.cuda.synchronize()
        replay_logits = static_logits.clone()

    result["replay"] = "OK"
    result["replay_logits_finite"] = bool(torch.isfinite(replay_logits).all())

    # A graph that is fast and WRONG is worthless, and wrongness here is silent:
    # the captured position, token or state could be stale and the logits would
    # still look plausible. So compare replay against eager for the SAME input
    # and the SAME starting state -- snapshot, replay, restore, run eager.
    snap = [t.clone() for t in flatten_state(states)]
    live = flatten_state(states)

    static_tok.fill_(tok_id)
    static_pos.fill_(pos)
    with torch.no_grad():
        graph.replay()
        torch.cuda.synchronize()
        g_logits = static_logits.clone()
        g_state = [t.clone() for t in live]

        for dst, src in zip(live, snap):     # restore the pre-replay state
            dst.copy_(src)
        e_logits, _ = model.forward_step(static_tok, states,
                                         position_idx=static_pos)
        torch.cuda.synchronize()
        e_state = flatten_state(states)

    dlogits = float((g_logits - e_logits).abs().max())
    dstate = max((float((a - b).abs().max()) for a, b in zip(g_state, e_state)),
                 default=0.0)
    result["max_abs_dlogits_graph_vs_eager"] = dlogits
    result["max_abs_dstate_graph_vs_eager"] = dstate
    print(f"[graph] 2. replay vs eager: max|dlogits| = {dlogits:.3e}, "
          f"max|dstate| = {dstate:.3e}")
    if dlogits > 1e-5 or dstate > 1e-5:
        print("[graph]    -> DIVERGENT. The speed number below is meaningless.")

    # --- 3. launches per replay ------------------------------------------
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.no_grad():
            for _ in range(32):
                graph.replay()
        torch.cuda.synchronize()
    kernels = 0
    for evt in prof.key_averages():
        dt = getattr(evt, "device_type", None)
        is_dev = (str(dt).endswith("CUDA") if dt is not None
                  else bool(getattr(evt, "is_gpu", False)))
        if is_dev:
            kernels += evt.count
    result["replay_launches_per_token"] = kernels / 32.0
    print(f"[graph] 3. device kernels/token on replay: {kernels/32.0:.1f} "
          f"(eager 533.0) -- NOT the metric a graph improves: a replay still "
          f"EXECUTES every kernel, it just submits them as one unit.")

    # --- 4. the metric that actually matters: wall-clock per token ---------
    # Decode was CPU-bound (491 ms CPU vs 66 ms device over 32 tokens), so the
    # graph's win is in launch OVERHEAD, which device kernel counts cannot see.
    # Timed in BOTH orders, because #131 measured the same arm at 21.45 and
    # 48.40 tok/s on identical code -- clock ramp on this card makes
    # first-vs-second ordering worth more than the effect being measured.
    import time

    def time_eager(n):
        with torch.no_grad():
            for _ in range(8):
                model.forward_step(static_tok, states, position_idx=static_pos)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(n):
                model.forward_step(static_tok, states, position_idx=static_pos)
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    def time_graph(n):
        for _ in range(8):
            graph.replay()
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n):
            graph.replay()
        torch.cuda.synchronize()
        return (time.perf_counter() - t0) / n

    N = 128
    e1, g1 = time_eager(N), time_graph(N)      # eager first
    g2, e2 = time_graph(N), time_eager(N)      # graph first
    eager_s = (e1 + e2) / 2
    graph_s = (g1 + g2) / 2
    result["eager_ms_per_token"] = eager_s * 1e3
    result["graph_ms_per_token"] = graph_s * 1e3
    result["speedup"] = eager_s / max(graph_s, 1e-12)
    result["speedup_eager_first"] = e1 / max(g1, 1e-12)
    result["speedup_graph_first"] = e2 / max(g2, 1e-12)
    print(f"[graph] 4. eager {eager_s*1e3:.2f} ms/token, "
          f"graph {graph_s*1e3:.2f} ms/token -> {eager_s/max(graph_s,1e-12):.2f}x")
    print(f"[graph]    both orders: {e1/max(g1,1e-12):.2f}x (eager first), "
          f"{e2/max(g2,1e-12):.2f}x (graph first)")

    _write(args.out, result)
    return 0


def _write(path, result):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)
    print(f"[graph] wrote {path}")


if __name__ == "__main__":
    raise SystemExit(main())
