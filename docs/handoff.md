# Live handoff — read this first (written 2026-08-16 11:05)

Author: Rambarun Komaljeet

State of play for a session picking this up cold, or for the same session after a
context reset. Everything below is measured unless it says otherwise.

---

## 1. What is running RIGHT NOW

**Issue #160 Phase 1 — the 48M architecture head-to-head.** A detached PowerShell
runner drives both arms sequentially and scores them; it survives this session
ending. Script: `scratchpad/phase1_48m.ps1` (in the session temp dir), log:
`logs/phase1.log`.

| arm | checkpoint dir | budget | ends |
|---|---|---|---|
| HVN (`--arch hvn-hybrid`, attn 3,9) | `checkpoints/p1_hvn_48m` | 22 h wall clock | **17 Aug 03:51** |
| transformer (`--arch transformer`) | `checkpoints/p1_tf_48m` | 22 h wall clock | **18 Aug 01:55** |
| both scored to `results/scaling/p1_*.json` | — | ~50 min | **18 Aug ~02:45** |

Both arms: `--seed 1234`, `--lr-schedule constant`, d_model 512, 12 layers,
16 heads, seq 4096, grad-accum 8, `--max-steps 60000` (never reached; the runner
stops each arm on the clock). Matched on **wall clock**, not step count, because
the two architectures have different step times and hours are the real budget.

**Do not run anything else on the GPU while these are training.** The arms are
timed, so contention silently steals steps from whichever arm is running and
biases the comparison.

### When it finishes

Compare `results/scaling/p1_hvn_48m.json` vs `p1_tf_48m.json` on final held-out
NLL, exactly as Phase 0 was compared. Then run the capability gates on the final
checkpoint of each — this is the part Phase 0 could not do:

```
python -m eval.next_token_gate                 # substantive slice + both floors
python -m eval.fim_completion_gate --checkpoint <ckpt> --n-cases 300 ...
```

**Pre-registered thresholds (fixed before the run, do not move them):**
noise floor is ±0.011 held-out NLL. Inside that = tie. HVN worse by >0.05 = a
real problem needing explanation before any scale-up.

---

## 2. Queued next, in order

1. **100K-token memory measurement (~1 h, laptop only).** The page's headline —
   6.29 MB vs 2.46 GB at 100K tokens — is **calculated** (`docs/theory.md:129`),
   not measured. What IS measured (`docs/benchmarks_speed.md:44`) is flat
   ms/token from N=256→4096 and a 20% memory edge at N=4096. The 390× figure at
   100K has never been observed. Build both architectures at matched size, feed
   100K tokens incrementally, report peak VRAM at 1K/10K/50K/100K so the curve is
   visible rather than two endpoints. Weights do not matter — memory follows the
   shapes, so random init is fine. Runs on the desktop 4060 in principle; the
   maintainer wants it on the laptop because the desktop has no battery.
2. **Publish `site/index.html`** once the 48M result and the 100K measurement are
   in. Do not publish before: the Phase 0 margin (+0.016) is above the noise floor
   but not comfortably, so a reversal at 48M would mean correcting a public claim.
3. **#158 chunked + recomputed cross-entropy** — ~1 GB, the largest unclaimed VRAM
   on the card. The loss uses more memory than the whole model.

---

## 3. Decisions already made — do not relitigate

- **Tokens per parameter ≈ 30 for this architecture, not Chinchilla's 20** (#159).
  20→30 buys 1.9–3.3%; 30→40 buys ~1.1% for 33% more time. **The size trend is
  UNRESOLVED** — the elbow estimates are not ordered by size, one curve is
  truncated, and a parametric fit to all 120 points is degenerate (it returns 0.3
  tokens/param at 1B, contradicting every raw curve). Use 30 as a working number;
  do not extrapolate it across scale.
- **The 157M is stopped on purpose** (`checkpoints/code_157m_v2`, step 100). At
  25,000 steps it would reach 0.32× Chinchilla — the #112 trap. At 30 tokens/param
  a 157M needs 9.7 days; a **108M needs 4.7** and is the better use of a week.
- **Phase 0 result stands:** at 12M body, matched hours, HVN 1.9654 vs transformer
  1.9814 (+0.016, above the ±0.011 threshold) — while seeing **40% less data**.
- **HVN is NOT cheaper to train.** The transformer is **1.41× faster per step** at
  384 wide / 4096 long. The compute win is at inference and only inference. Do not
  let this claim drift back.
- **The existing `code_48m_v2` cannot be used as an A/B arm** — it ran unseeded.

---

## 4. Gotchas that have already cost time

- **Check the clock every reply** (`Get-Date`). Extrapolating it has been wrong by
  2 hours and by 7 hours in this session alone.
- **Background watchers get killed.** Five in a row died mid-session. The detached
  PowerShell runners survive; harness watchers do not. Verify the artifact itself
  (checkpoint file, CSV rows, result JSON), never a watcher's exit status. A
  "DONE" that arrives implausibly fast means CRASHED, not finished.
- **Report results at the ETA you gave, unprompted.** Phase 0 finished at 02:04
  and was reported at 05:47 only because the maintainer asked.
- **Read the docs before designing anything.** Two experiments were specced this
  week that were already measured and recorded (#155's A/B, most of #156).
- **Ask whether the test could pass for the wrong reason, before running it.** The
  first scaling study had to be restarted 2 h in because mid-run checkpoints under
  a cosine schedule are not "models trained to X tokens".
- `tests/test_issue33.py` does not parse on `main` — exclude it from full runs.
  6 collection errors in `test_issue51/65/67` are also pre-existing.

---

## 5. Where things live

| | |
|---|---|
| held-out evaluator | `eval/heldout_loss.py` (fixed 400K-token slice, last 20 MB) |
| capability gates | `eval/next_token_gate.py`, `eval/fim_completion_gate.py` |
| scaling curves | `results/scaling/*.json` |
| the page | `site/index.html` — self-contained, mechanism deliberately removed |
| open issues | #158 chunked loss, #159 ratio, #160 head-to-head, #135, #152 |

**The page must not describe how the architecture works.** The mechanism sections
were removed on request; keep results, remove method. Check for leaks before
publishing (`grep -i` for the architecture name and its component vocabulary).
