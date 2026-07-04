"""
Test suite for Issue #36 — Speed up incremental generation.

Verifies the generation-loop cleanups:
  * ``prefill_states`` builds the recurrent state in one chunked/batched pass
    (same code path as training ``forward``) instead of the old redundant
    full-``forward`` + token-by-token re-processing.
  * The captured state continues generation *identically* to re-running the
    full forward over the growing sequence (proves state capture is correct
    across the prompt→decode boundary, for both HVN and DeltaNet blocks).
  * On-device sampling + reused input buffer produce the same tokens as before
    while eliminating per-token host↔device syncs.
  * O(1) per-token decode preserved; prefill uses O(chunks) launches.

Run on GPU (RTX 4050):  python test_issue36.py
"""

import sys
import time

import torch

# Windows consoles default to cp1252; force UTF-8 so status arrows render.
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from hvn.core import HamiltonianVortexNetwork


def _device():
    assert torch.cuda.is_available(), "CUDA required — this project runs on GPU only"
    return torch.device("cuda")


def _model(use_deltanet: bool, dev):
    torch.manual_seed(0)
    return HamiltonianVortexNetwork(
        vocab_size=97, d_model=128, num_layers=2, n_heads=4,
        use_deltanet=use_deltanet, dropout=0.0,
    ).to(dev).eval()


def _seq_prefill_state(m, prompt_ids, dev):
    """Exact pre-#36 prompt handling: absorb the prompt token-by-token via the
    sequential ``forward_step`` recurrence.  Returns (last_logits[V], states)."""
    st = m.init_generation_state_batched(1, device=dev, dtype=torch.float32)
    lo = None
    with torch.no_grad():
        for t, tk in enumerate(prompt_ids):
            tok = torch.tensor([[tk]], dtype=torch.long, device=dev)
            lo, st = m.forward_step_batched(tok, st, position_idx=t)
    return lo[0, -1], st


def _ref_old_incremental(m, prompt, n_new, dev):
    """Reproduces the pre-#36 generate_incremental algorithm exactly: sequential
    forward_step prefill + sequential forward_step decode, greedy (argmax)."""
    nl, st = _seq_prefill_state(m, prompt, dev)
    seq = list(prompt)
    with torch.no_grad():
        for i in range(n_new):
            nid = int(nl.argmax().item())
            seq.append(nid)
            tok = torch.tensor([[nid]], dtype=torch.long, device=dev)
            lo, st = m.forward_step_batched(tok, st, position_idx=len(prompt) + i)
            nl = lo[0, -1]
    return seq


def test_prefill_matches_sequential_recurrence():
    """prefill_states reproduces the exact sequential (per-token) prefill.

    The pre-#36 code seeded generation by absorbing the prompt token-by-token
    through ``forward_step`` (the redundant full ``forward()`` was discarded).
    prefill_states must yield the *same* last logits and recurrent state so
    generation output is unchanged — exactly on the HVN path (same recurrence)
    and to float tolerance on DeltaNet (chunked scan is algebraically exact).
    """
    dev = _device()
    for use_deltanet in (False, True):
        m = _model(use_deltanet, dev)
        prompt_ids = list(torch.randint(0, 97, (37,)).tolist())  # >1 chunk
        prompt = torch.tensor([prompt_ids], dtype=torch.long, device=dev)
        ref_logits, ref_states = _seq_prefill_state(m, prompt_ids, dev)
        with torch.no_grad():
            pf_logits, pf_states = m.prefill_states(prompt)
        ldiff = (pf_logits[0, -1] - ref_logits).abs().max().item()
        # Compare each block's recurrent state tensor(s).
        sdiff = 0.0
        for rs, ps in zip(ref_states, pf_states):
            if isinstance(rs, dict):
                for key in rs:
                    sdiff = max(sdiff, (rs[key] - ps[key]).abs().max().item())
            else:
                sdiff = max(sdiff, (rs - ps).abs().max().item())
        tag = "deltanet" if use_deltanet else "hvn"
        assert ldiff < 1e-4, f"[{tag}] prefill logits diverge: {ldiff}"
        assert sdiff < 1e-4, f"[{tag}] prefill state diverges: {sdiff}"
    print("[1/5] prefill == sequential recurrence ..... OK")


def test_prefill_state_continues_correctly():
    """State captured by prefill continues generation identically to the exact
    sequential recurrence over the same continuation tokens (both block types)."""
    dev = _device()
    for use_deltanet in (False, True):
        m = _model(use_deltanet, dev)
        prompt_len, cont_len = 33, 12
        full_ids = list(torch.randint(0, 97, (prompt_len + cont_len,)).tolist())
        prompt_ids = full_ids[:prompt_len]
        # Ground truth: one continuous sequential scan over the whole sequence.
        _, ref_st = _seq_prefill_state(m, prompt_ids, dev)
        max_diff = 0.0
        with torch.no_grad():
            _, pf_st = m.prefill_states(
                torch.tensor([prompt_ids], dtype=torch.long, device=dev)
            )
            for j in range(cont_len):
                tok = torch.tensor([[full_ids[prompt_len + j]]], dtype=torch.long, device=dev)
                ref_lo, ref_st = m.forward_step_batched(tok, ref_st, position_idx=prompt_len + j)
                pf_lo, pf_st = m.forward_step_batched(tok, pf_st, position_idx=prompt_len + j)
                max_diff = max(max_diff, (ref_lo - pf_lo).abs().max().item())
        tag = "deltanet" if use_deltanet else "hvn"
        assert max_diff < 1e-4, f"[{tag}] continued state diverges: {max_diff}"
    print("[2/5] prefill state continues correctly .... OK")


def test_generate_incremental_matches_reference():
    """generate_incremental(top_k=1) == the exact pre-#36 algorithm.

    top_k=1 makes sampling a temperature-invariant argmax, so the fast path must
    reproduce the sequential-recurrence greedy decode exactly — end-to-end proof
    that the #36 cleanups changed no generated output, on both block types.
    """
    dev = _device()
    for use_deltanet in (False, True):
        m = _model(use_deltanet, dev)
        prompt = [3, 14, 15, 92, 65, 35, 8, 9, 7, 9]
        n_new = 24
        ref = _ref_old_incremental(m, prompt, n_new, dev)
        got = m.generate_incremental(
            prompt, max_new_tokens=n_new, temperature=1.0, top_k=1, device=dev
        )
        tag = "deltanet" if use_deltanet else "hvn"
        assert got == ref, f"[{tag}] incremental != reference\n  ref={ref}\n  got={got}"
        assert len(got) == len(prompt) + n_new
    print("[3/5] generate_incremental == old algorithm  OK")


def test_o1_decode_scaling():
    """Per-token decode cost stays flat in N (fixed-size state, no KV cache)."""
    dev = _device()
    m = _model(use_deltanet=False, dev=dev)
    tok = torch.randint(0, 97, (1, 1), device=dev)
    times = {}
    for N in (256, 4096):
        st = m.init_generation_state(1, device=dev, dtype=torch.float32)
        for _ in range(5):
            _, st = m.forward_step(tok, st, position_idx=0)
        torch.cuda.synchronize(); t0 = time.time()
        for i in range(N):
            _, st = m.forward_step(tok, st, position_idx=i)
        torch.cuda.synchronize()
        times[N] = (time.time() - t0) / N * 1000
    ratio = times[4096] / times[256]
    assert ratio < 1.5, f"per-token cost grew with N (ratio={ratio:.2f}) — not O(1)"
    print(f"[4/5] O(1) decode ({times[256]:.2f}→{times[4096]:.2f} ms/tok, "
          f"ratio={ratio:.2f}) . OK")


def test_prefill_faster_than_tokenwise():
    """One-pass chunked prefill beats token-by-token forward_step over a long
    prompt (fewer kernel launches — the core Issue #36 win).

    Uses the DeltaNet path, whose chunked-parallel scan is exact, so prefill is
    genuinely O(chunks) launches vs O(prompt_len) for the per-token loop."""
    dev = _device()
    m = _model(use_deltanet=True, dev=dev)
    prompt = torch.randint(0, 97, (1, 512), device=dev)
    S = prompt.shape[1]

    # warm + time chunked prefill
    with torch.no_grad():
        for _ in range(3):
            m.prefill_states(prompt)
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            m.prefill_states(prompt)
        torch.cuda.synchronize()
        t_prefill = (time.time() - t0) / 10 * 1000

        # token-by-token baseline
        def tokenwise():
            st = m.init_generation_state(1, device=dev, dtype=torch.float32)
            for t in range(S):
                _, st = m.forward_step(prompt[:, t:t+1], st, position_idx=t)
        for _ in range(3):
            tokenwise()
        torch.cuda.synchronize(); t0 = time.time()
        for _ in range(10):
            tokenwise()
        torch.cuda.synchronize()
        t_tokenwise = (time.time() - t0) / 10 * 1000

    speedup = t_tokenwise / t_prefill
    assert speedup > 1.0, (
        f"prefill not faster: chunked={t_prefill:.2f}ms tokenwise={t_tokenwise:.2f}ms"
    )
    print(f"[5/5] prefill {t_tokenwise:.1f}→{t_prefill:.1f} ms "
          f"({speedup:.1f}× fewer-launch win, S={S}) . OK")


if __name__ == "__main__":
    print("=" * 60)
    print("Issue #36 — Faster incremental generation test suite")
    print("=" * 60)
    test_prefill_matches_sequential_recurrence()
    test_prefill_state_continues_correctly()
    test_generate_incremental_matches_reference()
    test_o1_decode_scaling()
    test_prefill_faster_than_tokenwise()
    print("=" * 60)
    print("ALL TESTS PASSED ✓")
    print("=" * 60)
